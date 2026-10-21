import numpy as np, pandas as pd, pytest
import components.p0_pipeline_config as p0
import components.p7_loader as p7
from components.base import combo_dir
from components.p7_loader import Loader, fold_metrics, pooled_auc

Q = ["q10", "q90"]
NAMES = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity"]


def write_subjects(cfg, combos, n=4, seed=1):  # n normal + n ischemic shape CSVs per combo; F4_kurtosis tracks the label
    rng = np.random.default_rng(seed)
    for d, o in combos:
        f = combo_dir(cfg["p4_features"]["out_dirs"][p0.Dataset.ARCHIVE], d, o)
        f.mkdir(parents=True)
        for lab, sub in enumerate(("normal", "ischemic")):
            for i in range(n):
                t = pd.DataFrame(rng.standard_normal((2, 2)), index=["F4_kurtosis", "F4_skew"], columns=Q)
                t.loc["F4_kurtosis"] += 3 * lab
                t.to_csv(f / f"{sub}_{i}.csv", index_label="feature")


@pytest.fixture
def lcfg(cfg, tmp_path):  # loader config: archive, shape CSVs only, quiet, outputs under tmp_path
    p4 = cfg["p4_features"]
    for k in ("welch", "asymmetry", "entropy", "wavelet"):
        p4[k]["enabled"] = False
    cfg["p1_data_formatting"]["dataset"] = p0.Dataset.ARCHIVE
    cfg["p5_selection"].update(k=2, debug=False)
    cfg["p7_loader"].update(folds=2, repeats=2, metrics=NAMES, out_dirs={p0.Dataset.ARCHIVE: str(tmp_path / "p7")}, debug=False)
    cfg["p3_epoching"]["combinations"] = [(4, 0), (10, 50)]
    write_subjects(cfg, cfg["p3_epoching"]["combinations"])
    return cfg


class FakeModel:  # p6 stand-in: score = sigmoid of the first chosen column; records what it was given
    seen = []

    def __init__(self, config): pass

    def output(self, X, y, X_test, y_test, **kwargs):
        FakeModel.seen.append({"train": list(X.index), "test": list(X_test.index), "y": np.asarray(y), "y_test": np.asarray(y_test), "cols": list(X_test.columns)})
        s = 1 / (1 + np.exp(-X_test.iloc[:, 0].to_numpy()))
        return {**kwargs, "X": X, "y": y, "y_test": y_test, "y_score": s, "y_pred": (s >= 0.5).astype(int)}


@pytest.fixture
def fake(monkeypatch):
    FakeModel.seen = []
    monkeypatch.setattr(p7, "MINI", [p7.Selection, FakeModel])
    return FakeModel.seen


def test_folds_get_test_subjects_without_overlap(lcfg, fake):
    res = Loader(lcfg).output()
    r = res["selection"][(4, 0)]
    seen = fake[:4]  # (4, 0): 2 repeats × 2 folds
    assert len(r["folds"]) == 4
    for f, s in zip(r["folds"], seen):
        assert set(s["train"]).isdisjoint(s["test"]) and s["train"] == f["train"] and s["test"] == f["test"]
        assert s["cols"] == f["chosen"]  # p5 reduced X_test to its choice
        assert s["y_test"].tolist() == p7.labels(s["test"], ["normal"]).tolist()
    for rep in (0, 1):  # every subject tested exactly once per repeat
        tested = [sid for f in r["folds"] if f["repeat"] == rep for sid in f["test"]]
        assert sorted(tested) == sorted(r["subjects"])


def test_fold_metrics_by_hand():
    y, p = np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
    s = np.array([0.1, 0.6, 0.4, 0.9])  # 3 of 4 pos/neg pairs ranked right -> AUC 0.75
    m = fold_metrics(y, s, p, NAMES)
    assert m == pytest.approx({"auc": 0.75, "accuracy": 0.75, "balanced_accuracy": 0.75, "sensitivity": 1.0, "specificity": 0.5})
    one = fold_metrics(np.array([1, 1]), np.array([0.2, 0.8]), np.array([0, 1]), NAMES)  # one class in the test fold
    assert np.isnan(one["auc"]) and np.isnan(one["specificity"]) and np.isnan(one["balanced_accuracy"]) and one["sensitivity"] == 0.5 and one["accuracy"] == 0.5
    assert list(fold_metrics(y, s, p, ["auc", "accuracy"])) == ["auc", "accuracy"]  # only the configured metrics


def test_pooled_auc_concatenates_folds_per_repeat():
    pred = pd.DataFrame({"repeat": [0, 0, 0, 0, 1, 1, 1, 1], "fold": [0, 0, 1, 1] * 2,
                         "y_true": [0, 1, 0, 1] * 2, "y_score": [0.1, 0.9, 0.2, 0.3, 0.5, 0.4, 0.6, 0.7]})
    # repeat 0: neg 0.1, 0.2 vs pos 0.9, 0.3 -> 4/4 pairs (each fold alone: 1, 1); repeat 1: neg 0.5, 0.6 vs pos 0.4, 0.7 -> 2/4
    assert pooled_auc(pred) == pytest.approx((1.0 + 0.5) / 2)


def test_outputs_written_and_summarised(lcfg, fake, tmp_path, capsys):
    res = Loader(lcfg).output(passthrough=1)
    assert res["passthrough"] == 1
    for combo in [(4, 0), (10, 50)]:
        d = combo_dir(tmp_path / "p7", *combo)
        m, pr = pd.read_csv(d / "metrics.csv"), pd.read_csv(d / "predictions.csv")
        assert list(m.columns) == ["repeat", "fold", "n_train", "n_test", *NAMES] and len(m) == 4 and (m["n_train"] + m["n_test"] == 8).all()
        assert list(pr.columns) == ["subject_id", "repeat", "fold", "y_true", "y_score", "y_pred"] and len(pr) == 16
        assert (pr.groupby("repeat")["subject_id"].nunique() == 8).all()
        r = res["selection"][combo]
        assert r["summary"]["auc_mean"] == pytest.approx(m["auc"].mean()) and r["summary"]["pooled_auc"] == pytest.approx(pooled_auc(pr))
        assert len(r["metrics"]) == 4 and len(r["predictions"]) == 16 and {"frequency", "subjects", "labels", "folds"} <= set(r)
    s = pd.read_csv(tmp_path / "p7" / "summary.csv")
    assert len(s) == 2 and s["pooled_auc"].is_monotonic_decreasing and {"duration", "overlap", "pooled_auc", "auc_mean", "auc_std"} <= set(s.columns)
    assert res["summary"]["pooled_auc"].tolist() == s["pooled_auc"].tolist()
    out = capsys.readouterr().out
    assert "most often chosen" in out and "pooled auc" in out and "combinations by pooled AUC" in out
    Loader(lcfg).output()  # rerun overwrites, does not append
    assert len(pd.read_csv(combo_dir(tmp_path / "p7", 4, 0) / "metrics.csv")) == 4


def test_skip_guard(lcfg, fake, capsys):
    lcfg["p7_loader"]["folds"] = 5  # only 4 subjects per class
    res = Loader(lcfg).output()
    assert res["selection"] == {} and res["summary"].empty and "skipped" in capsys.readouterr().out


def test_real_model_end_to_end(lcfg):  # p5 -> real p6 (XGBoost) on synthetic CSVs with a clear signal
    pytest.importorskip("xgboost")
    lcfg["p6_model"]["debug"] = False
    lcfg["p3_epoching"]["combinations"] = [(2, 0)]
    write_subjects(lcfg, [(2, 0)], n=20)
    r = Loader(lcfg).output()["selection"][(2, 0)]
    assert r["predictions"]["y_score"].between(0, 1).all() and r["summary"]["pooled_auc"] > 0.5
