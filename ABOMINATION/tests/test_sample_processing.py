from pathlib import Path
import numpy as np, pandas as pd, pytest
import components.p0_pipeline_config as p0
import components.p2_sample_processing as p2
from components.p1_data_formatting import DataFormat
from components.p2_sample_processing import SampleProcessing, process

FIG_CH = ["FP1", "FP2", "C3", "C4", "O1", "O2", "HEOL", "VEOR", "trigger"]
rng = np.random.default_rng(0)


@pytest.fixture
def opts(cfg):
    return cfg["p2_sample_processing"]


@pytest.fixture
def archive_df(cfg):
    rows = [{"subject_id": f"ischemic_{s}", "label": "stroke", "subtype": "ischemic", "split": "train", "subject": s, "file": f,
             "trial": 1, "signal": rng.standard_normal((16, 1024)).astype(np.float32), "path": f"ischemic_subject_{s}_{f}.npy"}
            for s in (1, 2) for f in (7, 3, 5)]  # deliberately out of file order
    df = pd.DataFrame(rows)
    df.attrs = {"sfreq": 256.0, "channels": cfg["p1_data_formatting"][p0.Dataset.ARCHIVE]["channel_names"], "dataset": p0.Dataset.ARCHIVE}
    return df


@pytest.fixture
def figshare_df():
    df = pd.DataFrame([{"subject_id": f"sub-{i:02d}", "label": "stroke", "subtype": "ischemic",
                        "signal": rng.standard_normal((len(FIG_CH), 4000)).astype(np.float32), "path": f"sub-{i:02d}.edf"} for i in range(5)])
    df.attrs = {"sfreq": 500.0, "channels": FIG_CH, "dataset": p0.Dataset.FIGSHARE}
    return df


def sample(signal, channels, sfreq=256.0):
    return {"subject_id": "x", "label": "normal", "subtype": "normal", "split": None, "signal": signal, "sfreq": sfreq,
            "channels": channels, "paths": ["x"]}


def test_samplers(archive_df, figshare_df):
    out = list(p2.archive_samples(archive_df))
    assert [s["subject_id"] for s in out] == ["ischemic_1", "ischemic_2"]
    for s in out:
        trials = archive_df[archive_df.subject_id == s["subject_id"]].sort_values("file")
        assert np.array_equal(s["signal"], np.concatenate(list(trials.signal), axis=1)) and s["paths"] == list(trials.path)
    out = list(p2.figshare_samples(figshare_df))
    assert all(s["split"] is None and np.array_equal(s["signal"], r) for s, r in zip(out, figshare_df.signal)) and len(out) == 5


def test_process(archive_df, figshare_df, opts):
    out = process(next(p2.figshare_samples(figshare_df)), opts)
    assert out["channels"] == FIG_CH[:6] and out["signal"].shape == (6, 2000) and out["signal"].dtype == np.float32
    assert out["sfreq"] == opts["sfreq"] and out["eye_regressed"]
    s = next(p2.archive_samples(archive_df))
    assert process(s, opts)["eye_regressed"] and process(s, opts)["signal"].shape == (16, 3000)
    out = process({**s, "channels": [f"ch{i}" for i in range(16)]}, opts)  # no FP1/FP2, so only CAR shifts the mean
    assert not out["eye_regressed"] and np.abs(out["signal"].mean(axis=0)).max() < 1e-4 * np.abs(out["signal"]).max()


@pytest.mark.parametrize("freq, lo, hi", [(10.0, 0.5, 2.0), (0.2, 0, 0.1), (60.0, 0, 0.01)])
def test_bandpass(freq, lo, hi, opts):
    t = np.arange(20 * 256) / 256
    x = np.zeros((8, t.size))
    x[0] = np.sin(2 * np.pi * freq * t)
    y = process(sample(x, [f"ch{i}" for i in range(8)]), opts)["signal"][0][750:-750]  # trim 3 s filter edges
    assert lo < np.mean(y ** 2) / 0.5 < hi


def test_output_one_file_per_subject(figshare_df, cfg, opts):
    out_dir = Path(opts["out_dirs"][p0.Dataset.FIGSHARE])
    out_dir.mkdir(parents=True)
    (out_dir / "stale.npz").write_bytes(b"stale")
    (out_dir / "keep.txt").write_text("x")
    opts.update(processes=2)
    out = SampleProcessing(cfg).output(data=figshare_df, extra=1)
    files = sorted(out_dir.glob("*.npz"))
    assert [f.name for f in files] == [f"sub-{i:02d}.npz" for i in range(5)] and (out_dir / "keep.txt").exists()
    assert out["extra"] == 1 and list(out["processed"].file) == [str(f) for f in files]
    with np.load(files[0], allow_pickle=False) as z:
        assert z.files == ["signal"] and z["signal"].dtype == np.float32


@pytest.mark.parametrize("real_cfg", [p0.Dataset.ARCHIVE, p0.Dataset.FIGSHARE], indirect=True)
def test_real_data(real_cfg):
    df = DataFormat(real_cfg).output()["data"]
    opts = real_cfg["p2_sample_processing"]
    out = SampleProcessing(real_cfg).output(data=df)["processed"]
    n = df.subject_id.nunique() if df.attrs["dataset"] == p0.Dataset.ARCHIVE else len(df)
    assert len(out) == n == len(list(Path(opts["out_dirs"][df.attrs["dataset"]]).glob("*.npz")))
