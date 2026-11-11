import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from components.models.forest import RandomForestModel

OPTS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced", "seed": 0, "n_jobs": 1}


def data(n_neg=30, n_pos=10, seed=0):  # q90_F4_kurtosis separates the classes, the rest is noise
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], [n_neg, n_pos])
    X = pd.DataFrame(rng.standard_normal((len(y), 4)), columns=["q90_F4_kurtosis", "q10_F4_skew", "q50_C3_C4_bsi", "q90_mean_bsi"])
    X["q90_F4_kurtosis"] += 4 * y
    return X, y


def test_fit_score_separable():
    X, y = data()
    m = RandomForestModel(OPTS).fit(X, y)
    s = m.score(X)
    assert s.shape == (len(y),) and ((s >= 0) & (s <= 1)).all() and roc_auc_score(y, s) > 0.95
    assert len(m.pipe.steps) == 2 and m.pipe[-1].get_params()["random_state"] == 0  # imputer + forest, no scaler


def test_nan_tolerated():
    X, y = data()
    X.iloc[::3, 1] = np.nan
    X.iloc[0, :] = np.nan  # a whole row missing
    s = RandomForestModel(OPTS).fit(X, y).score(X)
    assert np.isfinite(s).all() and roc_auc_score(y, s) > 0.9


def test_deterministic_with_seed():
    X, y = data()
    a, b = (RandomForestModel(OPTS).fit(X, y).score(X) for _ in range(2))
    np.testing.assert_array_equal(a, b)
