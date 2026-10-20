import numpy as np, pandas as pd, pytest
import components.p0_pipeline_config as p0
import components.p2_sample_processing as p2
from components.p3_epoching import Epoching, combo_dir
from components.p4_features import Features
from components.p7_loader import Loader, labels

SECS = 30
RATES = {"stroke_aaaaaabk": 256.0, "stroke_aaaaaaxy": 400.0, "normal_aaaaacdh": 1000.0, "normal_aaaaacqq": 250.0}  # TUH rates vary per recording


@pytest.fixture
def tuh_cfg(cfg):
    cfg["p1_data_formatting"]["dataset"] = p0.Dataset.TUH
    return cfg


@pytest.fixture
def electrodes(tuh_cfg):
    return tuh_cfg["p1_data_formatting"]["tuh"]["electrodes"]


@pytest.fixture
def tuh_df(tuh_cfg, electrodes):  # fake p1 TUH output: signals not loaded, one EDF path per subject
    t = tuh_cfg["p1_data_formatting"]["tuh"]
    df = pd.DataFrame([{"subject_id": sid, "label": "stroke" if sid.startswith("stroke") else "healthy", "subtype": sid.split("_")[0], "split": None,
                        "patient": sid.split("_")[1], "sfreq": fs, "signal": None, "path": f"/edf/{sid}.edf"} for sid, fs in RATES.items()])
    df.attrs = {"dataset": p0.Dataset.TUH, "sfreq": None, "channels": list(electrodes), "electrodes": list(electrodes), "references": list(t["references"])}
    return df


def synthetic(path, n_ch, common=0.0):  # (19, T) float32 µV at the path's rate: noise + a 10 Hz rhythm scaled per channel (+ an optional common-mode reference signal)
    sid = path.rsplit("/", 1)[1].removesuffix(".edf")
    fs = RATES[sid]
    t = np.arange(int(SECS * fs)) / fs
    rng = np.random.default_rng(list(RATES).index(sid))
    x = 10 * rng.standard_normal((n_ch, t.size)) + np.linspace(5, 40, n_ch)[:, None] * np.sin(2 * np.pi * 10 * t)
    return (x + common * np.sin(2 * np.pi * 7 * t)).astype(np.float32), fs


@pytest.fixture
def fake_read(monkeypatch, electrodes):  # stands in for components.tuh.read_tuh; checks p2 passes the configured electrodes / references
    calls = []
    def read(path, el, refs, common=0.0):
        assert list(el) == list(electrodes) and list(refs) == ["REF", "LE"]
        calls.append(path)
        return synthetic(path, len(el), common)
    monkeypatch.setattr(p2, "read_tuh", read)
    return calls


class InlinePool:  # p2's Pool run in this process, so the patched read_tuh applies
    def __init__(self, n): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def imap(self, f, it): return map(f, it)


def test_sampler_lazy(tuh_df, electrodes, fake_read):
    out = list(p2.tuh_samples(tuh_df))
    assert [s["subject_id"] for s in out] == list(RATES) and [s["sfreq"] for s in out] == list(RATES.values())
    assert all(s["signal"] is None and s["channels"] == electrodes and s["paths"] == [f"/edf/{s['subject_id']}.edf"] for s in out) and not fake_read


@pytest.mark.parametrize("sid", list(RATES))
def test_process_loads_resamples_car(sid, tuh_df, tuh_cfg, electrodes, fake_read, monkeypatch):
    opts = tuh_cfg["p2_sample_processing"]
    s = next(x for x in p2.tuh_samples(tuh_df) if x["subject_id"] == sid)
    out = p2.process(s, opts)
    assert fake_read == [s["paths"][0]] and out["signal"].shape == (19, SECS * 250) and out["signal"].dtype == np.float32
    assert out["sfreq"] == 250.0 and out["channels"] == electrodes and out["eye_regressed"]
    f = np.fft.rfftfreq(out["signal"].shape[1], 1 / 250)
    assert f[np.abs(np.fft.rfft(out["signal"][-1])).argmax()] == pytest.approx(10, abs=0.05)  # rhythm still at 10 Hz after e.g. 256 -> 250
    monkeypatch.setattr(p2, "read_tuh", lambda p, e, r: synthetic(p, len(e), common=500.0))  # same EEG on another reference (REF vs LE)
    assert np.allclose(p2.process(s, opts)["signal"], out["signal"], atol=1e-3 * np.abs(out["signal"]).max())  # CAR over the 19 removes it


def test_wrong_channel_count(tuh_df, tuh_cfg, monkeypatch):
    monkeypatch.setattr(p2, "read_tuh", lambda p, e, r: synthetic(p, 18))
    with pytest.raises(ValueError, match="18 channels"):
        p2.process(next(p2.tuh_samples(tuh_df)), tuh_cfg["p2_sample_processing"])


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")  # ANOVA on 1 training subject per class: 0/0 variance, scored 0
def test_end_to_end(tuh_df, tuh_cfg, electrodes, fake_read, monkeypatch):  # p2 -> p3 -> p4 -> p7 on 19 channels with TUH subject ids
    monkeypatch.setattr(p2, "Pool", InlinePool)
    p4, q = tuh_cfg["p4_features"], tuh_cfg["p4_features"]["quantiles"]
    tuh_cfg["p2_sample_processing"].update(debug=False)
    tuh_cfg["p3_epoching"]["combinations"] = [(4, 50)]
    p4["processes"] = 2
    tuh_cfg["p7_loader"].update(folds=2, debug=True)
    tuh_cfg["p5_selection"]["method"] = p0.Selector.ANOVA  # mutual information needs more than 1 training subject per class
    processed = p2.SampleProcessing(tuh_cfg).output(data=tuh_df)["processed"]
    assert processed.attrs["sfreq"] == 250.0 and (processed.sfreq == 250.0).all() and all(c == electrodes for c in processed.channels)
    assert sorted(fake_read) == sorted(tuh_df.path) and processed.eye_regressed.all()
    res = Features(tuh_cfg).output(**Epoching(tuh_cfg).output(processed=processed))
    shape, welch, bsi, ent = (res[k][(4, 50)] for k in ("features", "welch_features", "asymmetry_features", "entropy_features"))
    assert shape.shape == (4, 19 * 3 * len(q)) and welch.shape == (4, 19 * 13 * len(q)) and ent.shape == (4, 19 * 3 * len(q))
    assert {c.split("_", 1)[1] for c in bsi.columns} == {f"{l}_{r}_bsi" for l, r in p4["asymmetry"]["pairs"]} | {"mean_bsi"}  # all 8 pairs; FZ/CZ/PZ unpaired
    assert any(c.endswith("_CZ_kurtosis") for c in shape.columns) and (combo_dir(p4["out_dirs"][p0.Dataset.TUH], 4, 50) / "normal_aaaaacdh.png").exists()
    sel = Loader(tuh_cfg).output()["selection"][(4, 50)]
    assert sel["subjects"] == sorted(RATES) and list(sel["labels"]) == [0, 0, 1, 1]  # normal_* -> 0, stroke_* -> 1


def test_labels_tuh_ids():
    assert list(labels(["normal_aaaaacdh", "stroke_aaaaaabk"], ["normal"])) == [0, 1]
