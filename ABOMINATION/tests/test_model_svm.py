import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from components.models.svm import SVMModel

OPTS = {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "seed": 0}


def data(n_neg=30, n_pos=10, seed=0):  # q90_F4_kurtosis separates the classes, the rest is noise
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], [n_neg, n_pos])
    X = pd.DataFrame(rng.standard_normal((len(y), 4)), columns=["q90_F4_kurtosis", "q10_F4_skew", "q50_C3_C4_bsi", "q90_mean_bsi"])
    X["q90_F4_kurtosis"] += 4 * y
    return X, y


def test_fit_score_separable():
    X, y = data()
    m = SVMModel(OPTS).fit(X, y)
    s = m.score(X)
    assert s.shape == (len(y),) and ((s >= 0) & (s <= 1)).all() and roc_auc_score(y, s) > 0.95
    p = m.pipe[-1].estimator.get_params()
    assert p["kernel"] == "rbf" and p["class_weight"] == "balanced" and p["random_state"] == 0


def test_nan_tolerated():
    X, y = data()
    X.iloc[::3, 1] = np.nan
    X.iloc[0, :] = np.nan  # a whole row missing
    s = SVMModel(OPTS).fit(X, y).score(X)
    assert np.isfinite(s).all() and roc_auc_score(y, s) > 0.9


def test_deterministic_with_seed():
    X, y = data()
    a, b = (SVMModel(OPTS).fit(X, y).score(X) for _ in range(2))
    np.testing.assert_array_equal(a, b)
