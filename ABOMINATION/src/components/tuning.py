# Nested hyperparameter tuning for p6: random search scored by patient-grouped inner CV on one fold's training cases only.
# p6 calls tune() before fitting when p6_model.tuning.enabled; the outer test cases never reach this module.
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from components.base import Opts, groups
from components.models import build

def sample(space: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]: # one setting per key: list -> choice, (low, high, scale) -> number
    out = {}
    for k, v in space.items():
        if isinstance(v, list):
            out[k] = v[rng.integers(len(v))]
            continue
        lo, hi, scale = v
        x = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if scale in ("log", "log_int") else float(rng.uniform(lo, hi))
        out[k] = int(round(x)) if scale == "log_int" else x
    return out

def inner_auc(name: str, opts: Opts, X: pd.DataFrame, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> float: # mean AUC of one setting over the inner folds
    return float(np.mean([roc_auc_score(y[b], build(name, opts).fit(X.iloc[a], y[a]).score(X.iloc[b])) for a, b in splits]))

def tune(name: str, fixed: Opts, X: pd.DataFrame, y: np.ndarray, t: Opts) -> tuple[Opts, float, pd.DataFrame]: # -> (best settings, their inner AUC, every trial)
    y, g = np.asarray(y), groups(X.index)
    kind = StratifiedGroupKFold if len(set(g)) < len(g) else StratifiedKFold # several sessions per patient -> keep them in one inner fold
    cv = kind(t["inner_folds"], shuffle=True, random_state=t["seed"])
    splits = list(cv.split(X, y, g) if kind is StratifiedGroupKFold else cv.split(X, y))
    rng = np.random.default_rng(t["seed"])
    trials = [{k: fixed[k] for k in t["space"] if k in fixed}] + [sample(t["space"], rng) for _ in range(t["trials"] - 1)] # trial 0 = the fixed settings
    rows = [{**p, "inner_auc": inner_auc(name, {**fixed, **p}, X, y, splits)} for p in trials]
    table = pd.DataFrame(rows)
    best = table["inner_auc"].idxmax()
    return {**fixed, **trials[best]}, float(table.loc[best, "inner_auc"]), table
