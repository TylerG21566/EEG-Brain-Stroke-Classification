# Nested hyperparameter tuning, on one outer fold's training cases only (the outer test cases never reach this module).
# p6: tune() = random search scored by patient-grouped inner CV (p6_model.tuning). p7_cnn: staged() = lr sweep, then random search (p7_cnn.tuning).
from collections.abc import Callable
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

def staged(score: Callable[[Opts], dict[str, Any]], defaults: Opts, t: Opts, space: dict[str, Any], rng: np.random.Generator,
           log: Callable[[dict[str, Any]], None] | None = None) -> tuple[Opts, pd.DataFrame]: # CNN search (data/GOAT/DOCS/Tuning/CNN) -> (best settings, every trial)
    # Stage 1: sweep lr over t["lr_grid"] with everything else at its default (trial 0 = the defaults). Stage 2: best lr fixed, t["trials"] random draws of space.
    # score(settings) -> {"val_auc": ..., ...}; settings = defaults' keys. Ties and NaN scores never beat an earlier trial, so the defaults win unless beaten.
    rows: list[dict[str, Any]] = []
    def trial(stage: int, s: Opts) -> None:
        rows.append({"trial": len(rows), "stage": stage, **s, **score(s)})
        if log:
            log(rows[-1])
    def best() -> dict[str, Any]:
        vals = [r["val_auc"] for r in rows]
        return rows[int(np.argmax([-np.inf if np.isnan(v) else v for v in vals]))] # argmax = first of equal maxima
    for lr in [defaults["lr"]] + [float(v) for v in t["lr_grid"] if not np.isclose(v, defaults["lr"])]:
        trial(1, {**defaults, "lr": float(lr)})
    lr = best()["lr"]
    for _ in range(t["trials"] if space else 0):
        trial(2, {**defaults, **sample(space, rng), "lr": lr})
    return {k: best()[k] for k in defaults}, pd.DataFrame(rows)
