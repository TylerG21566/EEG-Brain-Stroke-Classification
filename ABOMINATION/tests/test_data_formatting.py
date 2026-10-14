import sys
from pathlib import Path
import numpy as np, pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import components.p1_data_formatting as p1
from components.p1_data_formatting import DataFormat

p1.DEBUG_VIEW = p1.NO_VIEW
N = 16385
FAKES = {("train", "stroke", "ischemic", 3, 7): 5, ("train", "stroke", "hemorrhagic", 10, 450): 1,
         ("eval", "normal", "normal", 12, 0): 2, ("eval", "stroke", "ischemic", 4, 9): 3}


@pytest.fixture
def fake(tmp_path):
    arrays = {}
    for (split, label, sub, subj, idx), trial in FAKES.items():
        d = tmp_path / "archive" / split / label
        d.mkdir(parents=True, exist_ok=True)
        a = np.random.rand(N).astype(np.float32)
        a[-1] = trial
        np.save(d / f"{sub}_subject_{subj}_{idx}.npy", a)
        arrays[(split, label, sub, subj, idx)] = a
    return DataFormat(p1.ARCHIVE, root=tmp_path).output(extra=1), arrays


def key(r):
    return (r.split, r.label, r.subtype, int(r.subject), int(r.file))


def test_kwargs_and_subject_id(fake):
    out, _ = fake
    df = out["data"]
    assert out["extra"] == 1
    assert (df.subject_id == df.subtype + "_" + df.subject.astype(str)).all()


def test_one_row_per_file_parsed_from_path(fake):
    out, arrays = fake
    assert sorted(key(r) for r in out["data"].itertuples()) == sorted(arrays)


def test_trial_and_signal(fake):
    out, arrays = fake
    for r in out["data"].itertuples():
        a = arrays[key(r)]
        assert r.trial == a[-1] and r.signal.ndim == 2 and r.signal.size == N - 1
        assert np.allclose(np.sort(r.signal, axis=None), np.sort(a[:-1]))


@pytest.mark.parametrize("dataset", [p1.ARCHIVE, p1.FIGSHARE])
def test_real_data(dataset):
    if not p1.ROOTS[dataset].exists():
        pytest.skip("real data missing")
    df = DataFormat(dataset).output()["data"]
    assert len(df) and set(df.label) <= {"stroke", "normal"}
    assert all(s.ndim == 2 and len(s) == len(df.attrs["channels"]) for s in df.signal)
    assert all(Path(p).exists() for p in df.path)
