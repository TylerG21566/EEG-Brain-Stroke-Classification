import numpy as np, pandas as pd, pytest
from sklearn.metrics import roc_auc_score
from components.models import MODELS, build
from components.models.xgb import XGBoostModel

OPTS = {"n_estimators": 60, "max_depth": 2, "learning_rate": 0.04, "reg_lambda": 20.0, "reg_alpha": 1.0, "gamma": 0.1,
        "subsample": 1.0, "colsample_bytree": 0.7, "min_child_weight": 1, "scale_pos_weight": "auto", "seed": 0, "n_jobs": 1}  # Tamim Table 4.3


def data(n_neg=30, n_pos=10, seed=0):  # q90_F4_kurtosis separates the classes, the rest is noise
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], [n_neg, n_pos])
    X = pd.DataFrame(rng.standard_normal((len(y), 4)), columns=["q90_F4_kurtosis", "q10_F4_skew", "q50_C3_C4_bsi", "q90_mean_bsi"])
    X["q90_F4_kurtosis"] += 4 * y
    return X, y


def test_build_by_name_and_unknown():
    assert isinstance(build("xgboost", OPTS), XGBoostModel) and MODELS["xgboost"] is XGBoostModel
    with pytest.raises(ValueError, match="available: \\['xgboost'\\]"):
        build("rnn", {})


def test_fit_score_separable():
    X, y = data()
    s = build("xgboost", OPTS).fit(X, y).score(X)
    assert s.shape == (len(y),) and ((s >= 0) & (s <= 1)).all() and roc_auc_score(y, s) > 0.95


@pytest.mark.parametrize("spw, want", [("auto", 3.0), (2.5, 2.5)])
def test_scale_pos_weight(spw, want):
    X, y = data(30, 10)  # neg / pos = 3
    m = XGBoostModel({**OPTS, "scale_pos_weight": spw}).fit(X, y)
    p = m.clf.get_params()
    assert p["scale_pos_weight"] == pytest.approx(want) and p["random_state"] == 0 and p["max_depth"] == 2 and "seed" not in p


def test_columns_reordered_and_mismatch():
    X, y = data()
    m = build("xgboost", OPTS).fit(X, y)
    np.testing.assert_allclose(m.score(X[X.columns[::-1]]), m.score(X))  # reordered columns -> same scores
    with pytest.raises(ValueError, match="missing \\['q90_mean_bsi'\\]"):
        m.score(X.drop(columns="q90_mean_bsi"))
    with pytest.raises(ValueError, match="unexpected \\['extra'\\]"):
        m.score(X.assign(extra=0.0))


def test_nan_tolerated():
    X, y = data()
    X.iloc[::3, 1] = np.nan
    X.iloc[0, :] = np.nan  # a whole row missing
    s = build("xgboost", OPTS).fit(X, y).score(X)
    assert np.isfinite(s).all() and roc_auc_score(y, s) > 0.9


def test_deterministic_with_seed():
    X, y = data()
    a, b = (build("xgboost", OPTS).fit(X, y).score(X) for _ in range(2))
    np.testing.assert_array_equal(a, b)
