import numpy as np, pandas as pd, pytest
from sklearn.metrics import roc_auc_score
import components.p0_pipeline_config as p0
import components.p7_loader as p7
import demographic_pipeline as dp
from combiner_components.model import CombinerModel
from combiner_pipeline import Ensemble
from components.p7_loader import Loader
from demographic_components.model import DemographicModel
from test_loader import lcfg, write_subjects  # noqa: F401 (fixture)

rng = np.random.default_rng(0)


def demo_frame(ids, y, informative=True):  # age tracks the label when informative; some gaps
    age = 40 + 25 * np.asarray(y) * informative + 5 * rng.standard_normal(len(ids))
    age[::7] = np.nan
    return pd.DataFrame({"age": age, "sex": rng.integers(0, 2, len(ids)).astype(float)}, index=list(ids))


def test_demographic_model_uses_age_and_fills_gaps(cfg):
    y = np.repeat([0, 1], 40)
    d = demo_frame(range(80), y)
    s = DemographicModel(cfg["demographic_model"]).fit(d, y).score(d)
    assert s.shape == (80,) and not np.isnan(s).any() and roc_auc_score(y, s) > 0.9


def test_combiner_trusts_the_informative_model(cfg):
    y = np.repeat([0, 1], 50)
    good, noise = np.clip(0.2 + 0.6 * y + 0.1 * rng.standard_normal(100), 0.01, 0.99), rng.uniform(0.01, 0.99, 100)
    c = CombinerModel(cfg["combiner"]).fit(np.column_stack([good, noise]), y)
    assert c.weights[0] > 5 * abs(c.weights[1]) and roc_auc_score(y, c.score(np.column_stack([good, noise]))) > 0.95


class FakeEEG:  # EEG mini-pipeline stand-in: noise scores; records every (train, test) split it is given
    seen = []

    def __init__(self, config): pass

    def output(self, X, y, X_test, y_test, **kwargs):
        FakeEEG.seen.append((set(X.index), set(X_test.index)))
        return {**kwargs, "X": X, "y": y, "X_test": X_test, "y_test": y_test, "y_score": rng.uniform(0.2, 0.8, len(X_test)), "chosen": ["c"], "scores": pd.Series({"c": 1.0})}


def test_ensemble_never_shows_test_cases_and_leans_on_demographics(cfg):
    FakeEEG.seen = []
    cfg["combiner"]["inner_folds"] = 3
    ids = [f"{s}_{i}" for s in ("normal", "stroke") for i in range(30)]
    y = np.repeat([0, 1], 30)
    X = pd.DataFrame({"c": rng.standard_normal(60)}, index=ids)
    test = list(range(0, 60, 5))
    train = [i for i in range(60) if i not in test]
    out = Ensemble(cfg).output(X=X.iloc[train], y=y[train], X_test=X.iloc[test], y_test=y[test], side={"demographic": demo_frame(ids, y)}, steps=[FakeEEG], fold="f")
    test_ids = set(X.index[test])
    assert len(FakeEEG.seen) == 4 and all(not (a & test_ids) for a, _ in FakeEEG.seen[:3])  # 3 inner folds use training cases only
    assert FakeEEG.seen[3] == (set(X.index[train]), test_ids)  # final refit: all training cases -> test cases
    assert out["y_score"].shape == out["y_score_eeg"].shape == out["y_score_demographic"].shape == (len(test),)
    assert out["weights"]["demographic"] > abs(out["weights"]["eeg"]) and roc_auc_score(y[test], out["y_score"]) > 0.9


def test_demographics_keys_match_p1_ids(cfg, monkeypatch):
    rows = pd.DataFrame({"patient": ["aaaaaabk", "aaaaacdh"], "session": [2, 1], "label": ["stroke", "healthy"], "age": [70.0, np.nan], "gender": ["M", None]})
    monkeypatch.setattr(dp, "labelled_recordings", lambda *a: rows)
    cfg["p1_data_formatting"]["dataset"] = p0.Dataset.TUH
    cfg["p1_data_formatting"]["tuh"]["recordings"] = p0.Recordings.CAPPED
    d = dp.demographics(cfg)
    assert list(d.index) == ["stroke_aaaaaabk-s002", "normal_aaaaacdh-s001"] and d.sex.iloc[0] == 1.0 and d.isna().iloc[1].all()
    cfg["p1_data_formatting"]["dataset"] = p0.Dataset.ARCHIVE
    with pytest.raises(ValueError, match="TUH"):
        dp.demographics(cfg)


def test_p7_with_combiner_writes_every_score(lcfg, monkeypatch, tmp_path):
    lcfg["combiner"].update(enabled=True, inner_folds=2)
    lcfg["p3_epoching"]["combinations"] = [(4, 0)]
    lcfg["p4_features"]["out_dirs"] = {p0.Dataset.ARCHIVE: str(tmp_path / "p4_big")}
    write_subjects(lcfg, [(4, 0)], n=10)  # inner folds need >= 2 cases per class for p5's mutual information
    ids = [f"{s}_{i}" for s in ("normal", "ischemic") for i in range(10)]
    y = [0] * 10 + [1] * 10
    monkeypatch.setattr(p7, "side_inputs", lambda cfg: {"demographic": demo_frame(ids, y), "ekg": pd.DataFrame({"hr": 70 + 10 * np.array(y) + rng.standard_normal(20)}, index=ids)})
    lcfg["ekg_model"]["features"] = ["hr"]
    Loader(lcfg).output()
    pred = pd.read_csv(lcfg["p7_loader"]["out_dirs"][p0.Dataset.ARCHIVE] + "/4s/0pct/predictions.csv")
    assert {"y_score", "y_score_eeg", "y_score_demographic", "y_score_ekg"} <= set(pred.columns) and len(pred) == 2 * 20  # 2 repeats × 20 cases
