import numpy as np, pandas as pd
import components.p0_pipeline_config as p0
import components.p6_model as p6
import components.p7_loader as p7
from components.models import build
from components.p7_loader import Loader
from components.tuning import sample, tune
from test_ensemble import demo_frame
from test_loader import lcfg, write_subjects  # noqa: F401 (fixture)


def data(n=40, seed=0):  # n normal + n stroke cases; column a tracks the label
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n)
    X = pd.DataFrame({"a": y + rng.standard_normal(2 * n), "b": rng.standard_normal(2 * n), "c": rng.standard_normal(2 * n)},
                     index=[f"{s}_{i}" for s in ("normal", "stroke") for i in range(n)])
    X.iloc[::9, 1] = np.nan
    return X, y


def small(cfg, trials=6):  # quick tuning options
    t = cfg["p6_model"]["tuning"]
    t.update(enabled=True, trials=trials, inner_folds=3)
    t["space"] = {**t["space"], "n_estimators": (20, 80, "log_int")}
    return t


def test_sample_stays_in_range(cfg):
    space, rng = cfg["p6_model"]["tuning"]["space"], np.random.default_rng(0)
    for _ in range(200):
        s = sample(space, rng)
        assert s["booster"] in ("gbtree", "gblinear") and isinstance(s["n_estimators"], int) and 50 <= s["n_estimators"] <= 1000
        assert all(v[0] <= s[k] <= v[1] for k, v in space.items() if not isinstance(v, list))


def test_tune_keeps_fixed_settings_as_trial_zero_and_is_repeatable(cfg):
    X, y = data()
    t, fixed = small(cfg), cfg["p6_model"]["xgboost"]
    best, inner, trials = tune("xgboost", fixed, X, y, t)
    assert len(trials) == t["trials"] and trials.iloc[0]["n_estimators"] == fixed["n_estimators"] and trials.iloc[0]["booster"] == "gbtree"
    assert inner == trials["inner_auc"].max() >= trials["inner_auc"].iloc[0] and inner > 0.7
    assert best["max_depth"] == fixed["max_depth"] and tune("xgboost", fixed, X, y, t)[1] == inner  # untuned keys kept; same seed, same answer


def test_gblinear_fits_with_tree_settings_present(cfg):
    X, y = data()
    s = build("xgboost", {**cfg["p6_model"]["xgboost"], "booster": "gblinear", "reg_alpha": 0.0, "reg_lambda": 1.0, "learning_rate": 0.3, "n_estimators": 100}).fit(X.fillna(0), y).score(X.fillna(0))
    assert s.shape == (len(y),) and np.all((0 < s) & (s < 1)) and s[y == 1].mean() > s[y == 0].mean()


def test_p7_tunes_on_training_cases_only_and_logs_settings(lcfg, monkeypatch, tmp_path):
    lcfg["p3_epoching"]["combinations"] = [(4, 0)]
    lcfg["p4_features"]["out_dirs"] = {p0.Dataset.ARCHIVE: str(tmp_path / "p4_big")}
    write_subjects(lcfg, [(4, 0)], n=12)
    small(lcfg, trials=3)
    seen = []
    def spy(name, fixed, X, y, t):
        seen.append(set(X.index))
        return tune(name, fixed, X, y, t)
    monkeypatch.setattr(p6, "tune", spy)
    res = Loader(lcfg).output()["selection"][(4, 0)]
    assert len(seen) == len(res["folds"]) == 4 and all(s == set(f["train"]) and s.isdisjoint(f["test"]) for s, f in zip(seen, res["folds"]))
    log = pd.read_csv(lcfg["p7_loader"]["out_dirs"][p0.Dataset.ARCHIVE] + "/4s/0pct/tuning.csv")
    assert len(log) == 4 and {"booster", "learning_rate", "inner_auc", "default_inner_auc", "test_auc"} <= set(log.columns)


def test_combiner_tunes_only_its_final_refit(lcfg, monkeypatch, tmp_path):
    lcfg["combiner"].update(enabled=True, inner_folds=2, models=["demographic"])
    lcfg["p3_epoching"]["combinations"] = [(4, 0)]
    lcfg["p4_features"]["out_dirs"] = {p0.Dataset.ARCHIVE: str(tmp_path / "p4_big")}
    write_subjects(lcfg, [(4, 0)], n=12)
    small(lcfg, trials=2)
    ids = [f"{s}_{i}" for s in ("normal", "ischemic") for i in range(12)]
    monkeypatch.setattr(p7, "side_inputs", lambda cfg: {"demographic": demo_frame(ids, [0] * 12 + [1] * 12)})
    calls = []
    monkeypatch.setattr(p6, "tune", lambda *a: calls.append(1) or tune(*a))
    Loader(lcfg).output()
    assert len(calls) == 4  # 2 repeats × 2 outer folds; none inside the combiner's inner folds
