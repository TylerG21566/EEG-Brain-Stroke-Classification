import numpy as np, pandas as pd, pytest
from sklearn.metrics import roc_auc_score
from components.models.logreg import LogRegModel

OPTS = {"C": 1.0, "l1_ratio": 0.0, "class_weight": "balanced", "seed": 0, "max_iter": 1000}


def data(n_neg=30, n_pos=10, seed=0):  # q90_F4_kurtosis separates the classes, the rest is noise
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], [n_neg, n_pos])
    X = pd.DataFrame(rng.standard_normal((len(y), 4)), columns=["q90_F4_kurtosis", "q10_F4_skew", "q50_C3_C4_bsi", "q90_mean_bsi"])
    X["q90_F4_kurtosis"] += 4 * y
    return X, y


@pytest.mark.parametrize("l1, solver", [(0.0, "lbfgs"), (0.5, "saga"), (1.0, "saga")])
def test_fit_score_separable(l1, solver):
    X, y = data()
    m = LogRegModel({**OPTS, "l1_ratio": l1}).fit(X, y)
    s = m.score(X)
    assert m.pipe[-1].solver == solver and s.shape == (len(y),) and ((s >= 0) & (s <= 1)).all() and roc_auc_score(y, s) > 0.95


def test_nan_tolerated():
    X, y = data()
    X.iloc[::3, 1] = np.nan
    X.iloc[0, :] = np.nan  # a whole row missing
    s = LogRegModel(OPTS).fit(X, y).score(X)
    assert np.isfinite(s).all() and roc_auc_score(y, s) > 0.9


def test_deterministic_with_seed():
    X, y = data()
    a, b = (LogRegModel({**OPTS, "l1_ratio": 0.5}).fit(X, y).score(X) for _ in range(2))
    np.testing.assert_array_equal(a, b)
