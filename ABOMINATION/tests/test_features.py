from pathlib import Path
import numpy as np, pandas as pd, pytest
from scipy.stats import kurtosis, skew
import components.p0_pipeline_config as p0
from components.p3_epoching import Epoching, combo_dir, epochs
from components.p4_features import Features, epoch_features

CH = ["FP1", "C3", "O1"]


@pytest.fixture
def processed(tmp_path):  # fake p2 output: two 40 s subjects at 250 Hz
    rng, rows = np.random.default_rng(0), []
    for sid in ("normal_1", "tia_2"):
        f = tmp_path / f"{sid}.npz"
        np.savez(f, signal=rng.standard_normal((len(CH), 10000)).astype(np.float32))
        rows.append({"subject_id": sid, "label": "x", "subtype": "x", "split": "train", "sfreq": 250.0, "channels": CH,
                     "eye_regressed": False, "paths": [], "file": str(f)})
    df = pd.DataFrame(rows)
    df.attrs = {"dataset": p0.Dataset.ARCHIVE}
    return df


def test_epoch_features():
    x = np.random.default_rng(1).standard_normal((3, 1000))
    x[2] = 0  # flat channel
    f = epoch_features(x, bins=64, clip=20.0)
    assert f.shape == (3, 3) and not np.isnan(f).any() and (f[2] == 0).all()
    assert np.allclose(f[:2, 0], skew(x[:2], axis=1)) and np.allclose(f[:2, 1], kurtosis(x[:2], axis=1)) and (f[:2, 2] > 0).all()


def test_output_quantile_summaries(processed, cfg):
    cfg["p4_features"]["processes"] = 2
    kw = Epoching(cfg).output(processed=processed, duration=4, overlap=50)
    out = Features(cfg).output(**kw)
    feats, q = out["features"], cfg["p4_features"]["quantiles"]
    assert feats.shape == (2, len(CH) * 3 * len(q)) and list(feats.index) == ["normal_1", "tia_2"] and "q90_C3_kurtosis" in feats
    d = combo_dir(cfg["p4_features"]["out_dirs"][p0.Dataset.ARCHIVE], 4, 50)
    assert sorted(f.name for f in d.glob("*.csv")) == ["normal_1.csv", "tia_2.csv"]
    csv = pd.read_csv(d / "normal_1.csv", index_col="feature")
    assert csv.shape == (len(CH) * 3, len(q)) and list(csv.columns) == [f"q{p}" for p in q]
    ref = np.array([epoch_features(e["signal"], 64, 20.0)[1, 1] for e in epochs(processed.iloc[:1], 4, 50, cfg["p3_epoching"])])
    assert csv.loc["C3_kurtosis", "q50"] == pytest.approx(np.percentile(ref, 50)) == pytest.approx(feats.loc["normal_1", "q50_C3_kurtosis"])
