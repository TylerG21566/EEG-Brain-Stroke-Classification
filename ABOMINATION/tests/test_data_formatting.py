from pathlib import Path
import numpy as np, pytest
import components.p0_pipeline_config as p0
from components.base import load_config
from components.p1_data_formatting import DataFormat

N = 16385
FAKES = {("train", "stroke", "ischemic", 3, 7): 5, ("train", "stroke", "hemorrhagic", 10, 450): 1,
         ("eval", "normal", "normal", 12, 0): 2, ("eval", "stroke", "ischemic", 4, 9): 3}


@pytest.fixture
def fake(cfg):
    root = Path(cfg["p1_data_formatting"]["roots"][p0.Dataset.ARCHIVE])
    arrays = {}
    for (split, label, sub, subj, idx), trial in FAKES.items():
        d = root / "archive" / split / label
        d.mkdir(parents=True, exist_ok=True)
        a = np.random.rand(N).astype(np.float32)
        a[-1] = trial
        np.save(d / f"{sub}_subject_{subj}_{idx}.npy", a)
        arrays[(split, label, sub, subj, idx)] = a
    cfg["p1_data_formatting"]["dataset"] = p0.Dataset.ARCHIVE
    return DataFormat(cfg).output(extra=1), arrays, cfg


def key(r):
    return (r.split, r.label, r.subtype, int(r.subject), int(r.file))


def test_config_loads():
    c = load_config()
    assert {"p1_data_formatting", "p2_sample_processing"} <= c.keys()
    c["p1_data_formatting"]["roots"][p0.Dataset.ARCHIVE] = "x"  # deep copy: nested edits don't leak into p0.CONFIG
    assert p0.CONFIG["p1_data_formatting"]["roots"][p0.Dataset.ARCHIVE] != "x"


def test_kwargs_attrs_and_subject_id(fake):
    out, _, cfg = fake
    df = out["data"]
    assert out["extra"] == 1 and df.attrs["dataset"] == p0.Dataset.ARCHIVE
    assert df.attrs["channels"] == cfg["p1_data_formatting"][p0.Dataset.ARCHIVE]["channel_names"]
    assert (df.subject_id == df.subtype + "_" + df.subject.astype(str)).all()


def test_one_row_per_file_parsed_from_path(fake):
    out, arrays, _ = fake
    assert sorted(key(r) for r in out["data"].itertuples()) == sorted(arrays)


def test_trial_and_signal(fake):
    out, arrays, _ = fake
    for r in out["data"].itertuples():
        a = arrays[key(r)]
        assert r.trial == a[-1] and r.signal.ndim == 2 and r.signal.size == N - 1
        assert np.allclose(np.sort(r.signal, axis=None), np.sort(a[:-1]))


@pytest.mark.parametrize("real_cfg", [p0.Dataset.ARCHIVE, p0.Dataset.FIGSHARE], indirect=True)
def test_real_data(real_cfg):
    df = DataFormat(real_cfg).output()["data"]
    assert len(df) and set(df.label) <= {"stroke", "normal"}
    assert all(s.ndim == 2 and len(s) == len(df.attrs["channels"]) for s in df.signal)
    assert all(Path(p).exists() for p in df.path)
