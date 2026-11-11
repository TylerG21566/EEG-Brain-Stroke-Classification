import numpy as np, pandas as pd, pytest
from sklearn.metrics import roc_auc_score
from components.models.lda import LDAModel

OPTS = {"solver": "lsqr", "shrinkage": "auto", "balanced": False}


def data(n_neg=30, n_pos=10, seed=0):  # q90_F4_kurtosis separates the classes, the rest is noise
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], [n_neg, n_pos])
    X = pd.DataFrame(rng.standard_normal((len(y), 4)), columns=["q90_F4_kurtosis", "q10_F4_skew", "q50_C3_C4_bsi", "q90_mean_bsi"])
    X["q90_F4_kurtosis"] += 4 * y
    return X, y


@pytest.mark.parametrize("solver", ["lsqr", "eigen", "svd"])
def test_fit_score_separable(solver):
    X, y = data()
    s = LDAModel({**OPTS, "solver": solver}).fit(X, y).score(X)
    assert s.shape == (len(y),) and ((s >= 0) & (s <= 1)).all() and roc_auc_score(y, s) > 0.95


def test_nan_tolerated():
    X, y = data()
    X.iloc[::3, 1] = np.nan
    X.iloc[0, :] = np.nan  # a whole row missing
    s = LDAModel(OPTS).fit(X, y).score(X)
    assert np.isfinite(s).all() and roc_auc_score(y, s) > 0.9


@pytest.mark.parametrize("balanced, want", [(True, [0.5, 0.5]), (False, [0.75, 0.25])])
def test_priors(balanced, want):
    X, y = data(30, 10)
    m = LDAModel({**OPTS, "balanced": balanced}).fit(X, y)
    np.testing.assert_allclose(m.pipe[-1].priors_, want)
    assert m.pipe[-1].shrinkage == "auto"
