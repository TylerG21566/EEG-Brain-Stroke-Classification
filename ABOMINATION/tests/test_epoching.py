import types
from pathlib import Path
import numpy as np, pandas as pd, pytest
import components.p0_pipeline_config as p0
from components.p3_epoching import Epoching, combo_dir, epoch_name, epochs, subject_epochs

SFREQ, CH = 250.0, ["FP1", "C3", "O1", "O2"]
SPIKE = (1, 1234)  # (channel, sample): lands in epoch 1 for 4 s / 0 %
rng = np.random.default_rng(0)


@pytest.fixture
def opts(cfg):
    return cfg["p3_epoching"]


@pytest.fixture
def processed(tmp_path):  # fake p2 output: one 200 s subject and one 3 s subject (shorter than any window)
    rows = []
    for sid, secs in (("ischemic_5", 200), ("ischemic_6", 3)):
        x = rng.standard_normal((len(CH), int(secs * SFREQ))).astype(np.float32) * 5 + 3
        x[3] = 7.0  # flat channel: std 0 -> zeros
        if secs == 200:
            x[SPIKE] = 1e4
        f = tmp_path / f"{sid}.npz"
        np.savez(f, signal=x)
        rows.append({"subject_id": sid, "label": "stroke", "subtype": "ischemic", "split": "train", "sfreq": SFREQ,
                     "channels": CH, "eye_regressed": True, "paths": [f"{sid}.npy"], "file": str(f)})
    df = pd.DataFrame(rows)
    df.attrs = {"dataset": p0.Dataset.ARCHIVE, "sfreq": SFREQ, "out_dir": str(tmp_path)}
    return df


def zscore(x, clip):
    x = x.astype(np.float64)
    sd = x.std(axis=1, keepdims=True)
    return np.clip(np.where(sd > 0, (x - x.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1), 0), -clip, clip)


@pytest.mark.parametrize("duration, overlap, n", [(4, 0, 50), (4, 50, 99), (10, 0, 20), (10, 50, 39)])
def test_counts_and_starts(processed, opts, duration, overlap, n):
    out = list(epochs(processed, duration, overlap, opts))
    win = round(duration * SFREQ)
    step = round(win * (1 - overlap / 100))
    assert len(out) == n and {e["subject_id"] for e in out} == {"ischemic_5"}  # 3 s subject yields nothing
    assert [e["epoch"] for e in out] == list(range(n)) and [e["start"] for e in out] == pytest.approx([i * step / SFREQ for i in range(n)])
    e = out[-1]
    assert e["signal"].shape == (len(CH), win) and e["signal"].dtype == np.float32
    assert (e["label"], e["subtype"], e["split"], e["sfreq"], e["channels"]) == ("stroke", "ischemic", "train", SFREQ, CH)
    assert (e["duration"], e["overlap"]) == (duration, overlap)


def test_content_zscored_and_clipped(processed, opts):
    src = np.load(processed.file[0])["signal"]
    out = list(epochs(processed, 4, 50, opts))
    for e in out:
        i = round(e["start"] * SFREQ)
        assert np.allclose(e["signal"], zscore(src[:, i:i + 1000], opts["clip"]), atol=1e-4)
        assert not np.isnan(e["signal"]).any() and not e["signal"][3].any()  # flat channel -> zeros
        assert np.allclose(e["signal"][[0, 2]].mean(axis=1), 0, atol=1e-4) and np.allclose(e["signal"][[0, 2]].std(axis=1), 1, atol=1e-3)
    spiked = [e for e in out if e["start"] * SFREQ <= SPIKE[1] < e["start"] * SFREQ + 1000]
    assert len(spiked) == 2 and all(e["signal"][SPIKE[0]].max() == pytest.approx(opts["clip"]) for e in spiked)


def test_names():
    e = {"subject_id": "ischemic_5", "epoch": 3, "start": 12.0}
    assert epoch_name(e) == "ischemic_5_e0003_t0012.00s"
    assert epoch_name({**e, "epoch": 12, "start": 198.5}) == "ischemic_5_e0012_t0198.50s"
    assert combo_dir("x", 4.0, 50) == Path("x") / "4s" / "50pct" and combo_dir("x", 2.5, 0) == Path("x") / "2.5s" / "0pct"


def test_output_lazy_and_debug_files(processed, cfg, opts):
    out_dir = Path(opts["out_dirs"][p0.Dataset.ARCHIVE])
    out = Epoching(cfg).output(processed=processed, duration=10, overlap=50, extra=1)  # debug off: nothing written
    assert isinstance(out["subjects"], types.GeneratorType) and out["extra"] == 1 and out["processed"] is processed
    tasks = list(out["subjects"])  # one small task per subject, expanded by p4's workers
    assert [t["row"]["subject_id"] for t in tasks] == ["ischemic_5", "ischemic_6"]
    assert sum(1 for t in tasks for _ in subject_epochs(**t)) == 39 and not out_dir.exists()

    opts["debug"] = True
    combo = combo_dir(out_dir, 10, 50)
    combo.mkdir(parents=True)
    (combo / "stale.npy").write_bytes(b"stale")
    (combo / "keep.txt").write_text("x")
    gen = Epoching(cfg).output(processed=processed, duration=10, overlap=50)["subjects"]
    assert (combo / "stale.npy").exists()  # lazy: nothing happens until iterated
    eps = [e for t in gen for e in subject_epochs(**t)]
    files = sorted(combo.glob("*.npy"))
    assert [f.name for f in files] == sorted(f"{epoch_name(e)}.npy" for e in eps) and (combo / "keep.txt").exists()
    assert files[0].name == "ischemic_5_e0000_t0000.00s.npy" and np.array_equal(np.load(files[0]), eps[0]["signal"])
