import numpy as np, pandas as pd
import components.p0_pipeline_config as p0
import ekg_pipeline as ep
from ekg_components.hrv import FEATURES, hrv, rr_intervals

FS = 250.0


def ecg(rr, fs=FS, invert=False):  # spike train: one sharp QRS-like pulse per beat at the given RR intervals (s), plus noise
    t = np.arange(int((sum(rr) + 2) * fs)) / fs
    x = 0.05 * np.random.default_rng(0).standard_normal(t.size)
    for b in np.cumsum(rr):
        x += np.exp(-((t - b) / 0.01) ** 2)
    return -x if invert else x


def test_rr_intervals_recover_heart_rate_either_polarity():
    for invert in (False, True):
        rr = rr_intervals(ecg([0.8] * 100, invert=invert), FS, (5.0, 25.0))
        assert len(rr) >= 95 and np.allclose(rr, 0.8, atol=0.01)  # 75 bpm


def test_hrv_values_and_irregular_rhythm():
    steady = hrv(np.full(200, 1.0))
    assert set(steady) == set(FEATURES) and abs(steady["hr"] - 60) < 1e-9 and steady["sdnn"] == steady["rmssd"] == steady["pnn50"] == 0
    irregular = hrv(np.random.default_rng(1).uniform(0.5, 1.2, 200))  # AF-like: beat gaps all over the place
    assert irregular["cv"] > 0.15 and irregular["pnn50"] > 0.5 and irregular["rmssd"] > 100
    assert np.isnan(hrv(np.full(10, 1.0))["hr"])  # too few beats


def test_ekg_features_keys_cache_and_missing_lead(cfg, monkeypatch, tmp_path):
    rows = pd.DataFrame({"patient": ["aaaaaabk", "aaaaacdh"], "session": [1, 1], "label": ["stroke", "healthy"],
                         "edf_paths": [["a.edf", "b.edf"], ["c.edf"]], "seconds": [[None, 60.0], [None]]})
    monkeypatch.setattr(ep, "labelled_recordings", lambda *a: rows)
    reads = []
    def read(path, secs):
        reads.append((path.rsplit("/", 1)[1], secs))
        return (None, FS) if path.endswith("c.edf") else (ecg([0.75] * 60), FS)  # c.edf: no EKG lead
    monkeypatch.setattr(ep, "read_ekg", read)
    root = tmp_path / "in"
    root.mkdir(exist_ok=True)
    for f in ("a.edf", "b.edf", "c.edf"):
        (root / f).touch()
    cfg["p1_data_formatting"].update(dataset=p0.Dataset.TUH, roots={p0.Dataset.TUH: str(root)})
    cfg["ekg_model"]["out_dirs"] = {p0.Dataset.TUH: str(tmp_path / "ekg")}
    f = ep.ekg_features(cfg)
    assert list(f.index) == ["stroke_aaaaaabk-s001", "normal_aaaaacdh-s001"] and reads == [("a.edf", None), ("b.edf", 60.0), ("c.edf", None)]
    assert abs(f.hr.iloc[0] - 80) < 1 and f.iloc[1].isna().all()  # RR from both files pooled; no lead -> NaN (training medians later)
    assert len(list((tmp_path / "ekg").glob("*.csv"))) == 1 and ep.ekg_features(cfg).index.tolist() == f.index.tolist() and len(reads) == 3  # second call: cache
