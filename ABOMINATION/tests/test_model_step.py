import numpy as np, pandas as pd, pytest
import components.p0_pipeline_config as p0
import components.p6_model as p6
from components.features import FEATURE_CLASSES as EXTRACTORS
from components.p5_selection import Selection
from components.p6_model import Model

NAMES = ["F4_kurtosis", "F4_skew", "C3_C4_bsi", "mean_bsi"]


class Fake:  # mean-based scorer: P(1) rises with the signal column's distance above the class-0 training mean
    def __init__(self, name, opts):
        self.name, self.opts = name, opts

    def fit(self, X, y):
        self.col = X.columns[0]
        self.m0, self.m1 = X[self.col][y == 0].mean(), X[self.col][y == 1].mean()
        return self

    def score(self, X):
        return 1 / (1 + np.exp(-(X[self.col].to_numpy() - (self.m0 + self.m1) / 2)))


@pytest.fixture
def fake(monkeypatch):  # p6 builds a Fake instead of a real model; records what it was asked for
    calls = []
    monkeypatch.setattr(p6, "build", lambda name, opts: calls.append((name, opts)) or Fake(name, opts))
    return calls


def split(cfg, n=20, n_test=8, seed=0):  # p7-style fold layer: q90_F4_kurtosis tracks the label
    rng = np.random.default_rng(seed)
    def table(m, prefix):
        y = np.repeat([0, 1], m // 2)
        X = pd.DataFrame(rng.standard_normal((m, len(NAMES))), columns=[f"q90_{c}" for c in NAMES], index=[f"{prefix}{i}" for i in range(m)])
        X["q90_F4_kurtosis"] += 4 * y
        return X, y
    (X, y), (X_test, y_test) = table(n, "train_"), table(n_test, "test_")
    fam = {c: "asymmetry" if c.endswith("_bsi") else "shape" for c in X.columns}
    ext = {k: EXTRACTORS[k](cfg["p4_features"]) for k in ("shape", "asymmetry")}
    return {"X": X, "y": y, "X_test": X_test, "y_test": y_test, "families": fam, "extractors": ext, "fold": "t"}


def run(cfg, layer):  # p7's MINI: p5 -> p6
    cfg["p5_selection"].update(k=2, debug=False)
    for step in (Selection, Model):
        layer = step(cfg).output(**layer)
    return layer


def test_p5_p6_fit_and_score(cfg, fake):
    out = run(cfg, split(cfg))
    assert out["chosen"][0] == "q90_F4_kurtosis" and list(out["X_test"].columns) == out["chosen"]
    assert out["y_score"].shape == (8,) and set(out["y_pred"].tolist()) <= {0, 1}
    assert (out["y_pred"] == out["y_test"]).mean() >= 0.75  # easy signal
    assert {"chosen", "scores", "fill", "model", "y_test"} <= set(out) and out["fold"] == "t"


def test_threshold(cfg, fake):
    layer = split(cfg)
    for t, expect in ((0.0, 1), (1.01, 0)):
        cfg["p6_model"]["threshold"] = t
        assert (run(cfg, dict(layer))["y_pred"] == expect).all()
    cfg["p6_model"]["threshold"] = 0.5
    out = run(cfg, dict(layer))
    assert (out["y_pred"] == (out["y_score"] >= 0.5)).all()


def test_one_class_fold_raises(cfg, fake):
    layer = split(cfg)
    layer.update(y=np.zeros_like(layer["y"]), fold="repeat 1 | fold 3")
    with pytest.raises(ValueError, match="fold 3"):
        Model(cfg).output(**layer)


def test_model_picks_configured_subdict(cfg, fake):  # opts[model] is the sub-dict handed to build
    cfg["p6_model"].update(model="other", other={"depth": 7})
    out = run(cfg, split(cfg))
    assert fake == [("other", {"depth": 7})] and out["model"].opts == {"depth": 7}


def test_test_labels_not_used(cfg, fake):  # shuffling y_test changes nothing but y_test itself
    layer = split(cfg)
    a = run(cfg, dict(layer))
    b = run(cfg, {**layer, "y_test": np.random.default_rng(3).permutation(layer["y_test"])})
    assert np.array_equal(a["y_score"], b["y_score"]) and np.array_equal(a["y_pred"], b["y_pred"])


def test_no_test_set(cfg, fake):
    layer = split(cfg)
    del layer["X_test"], layer["y_test"]
    out = run(cfg, layer)
    assert out["y_score"] is None and out["y_pred"] is None and out["X_test"] is None


def test_real_registry_xgboost(cfg):
    pytest.importorskip("xgboost")
    out = run(cfg, split(cfg))
    assert out["y_score"].shape == (8,) and ((out["y_score"] >= 0) & (out["y_score"] <= 1)).all()
    assert (out["y_pred"] == out["y_test"]).mean() >= 0.75
