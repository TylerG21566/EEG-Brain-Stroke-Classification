import numpy as np, pandas as pd, pytest
from scipy.stats import entropy, kurtosis, skew
import components.p0_pipeline_config as p0
from components.p3_epoching import Epoching, combo_dir, epochs
from components.features.shape import epoch_features
from components.p4_features import Features

CH = ["FP1", "C3", "O1"]
NW = len(CH) * 10  # Welch rows: 4 rel + 6 ratios per channel (160 for 16 channels)


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
    x = np.random.default_rng(1).standard_normal((2, 3, 1000))
    x[:, 2] = 0  # flat channel
    f = epoch_features(x, bins=64, clip=20.0)
    ref = [[entropy(np.histogram(c, 64, (-20, 20))[0], base=2) for c in e[:2]] for e in x]
    assert f.shape == (2, 3, 3) and not np.isnan(f).any() and (f[:, 2] == 0).all()
    assert np.allclose(f[:, :2, 0], skew(x[:, :2], axis=-1)) and np.allclose(f[:, :2, 1], kurtosis(x[:, :2], axis=-1))
    assert np.allclose(f[:, :2, 2], ref)


def test_output_quantile_summaries(processed, cfg):
    cfg["p4_features"]["processes"] = 2
    cfg["p3_epoching"]["combinations"] = [(4, 50), (10, 0)]
    d = combo_dir(cfg["p4_features"]["out_dirs"][p0.Dataset.ARCHIVE], 4, 50)
    d.mkdir(parents=True)
    (d / "stale.csv").write_text("x"), (d / "stale.png").write_text("x")
    res = Features(cfg).output(**Epoching(cfg).output(processed=processed))
    out, welch, q = res["features"], res["welch_features"], cfg["p4_features"]["quantiles"]
    assert set(out) == set(welch) == {(4, 50), (10, 0)} and all(f.shape == (2, len(CH) * 3 * len(q)) for f in out.values())
    assert all(f.shape == (2, NW * len(q)) for f in welch.values())
    for c in welch:
        cd = combo_dir(cfg["p4_features"]["out_dirs"][p0.Dataset.ARCHIVE], *c)
        assert len(pd.read_csv(cd / "normal_1_welch.csv")) == NW and (cd / "normal_1.png").exists()
    assert sorted(f.name for f in d.glob("*.csv")) == ["normal_1.csv", "normal_1_welch.csv", "tia_2.csv", "tia_2_welch.csv"]  # stale deleted
    assert sorted(f.name for f in d.glob("*.png")) == ["normal_1.png", "tia_2.png"]
    csv = pd.read_csv(d / "normal_1.csv", index_col="feature")
    assert csv.shape == (len(CH) * 3, len(q)) and list(csv.columns) == [f"q{p}" for p in q]
    assert np.allclose(out[(4, 50)].loc["normal_1", [f"{c}_{f}" for f in csv.index for c in csv.columns]], csv.values.ravel())
    x = np.stack([e["signal"] for e in epochs(processed.iloc[:1], 4, 50, cfg["p3_epoching"])])
    assert csv.loc["C3_kurtosis", "q90"] == pytest.approx(np.percentile(kurtosis(x[:, 1], axis=-1), 90))
