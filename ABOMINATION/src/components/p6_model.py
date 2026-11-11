# p6: fits the configured classifier on one fold's training subjects (p5's chosen features) and scores the test subjects.
# Runs inside p7 after p5 (MINI), once per fold; fitting sees only X, y, so test labels (y_test) are passed through untouched for p7's metrics.
from typing import Any
import numpy as np
import pandas as pd
from components.base import Layer, PipelineStep
from components.models import build # name, opts -> unfitted model with fit(X, y) / score(X) -> P(class 1)
from components.tuning import tune # nested random search on the training cases (p6_model.tuning)

class Model(PipelineStep): # p6: one model per fold; y_score = P(stroke) for each test subject, y_pred = y_score >= threshold
    section = "p6_model"

    def output(self, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame | None = None, y_test: np.ndarray | None = None, fold: str = "", **kwargs: Any) -> Layer: # X / X_test: p5's chosen columns, test gaps already filled with training medians
        opts = self.opts
        if len(set(np.asarray(y).tolist())) < 2:
            raise ValueError(f"[p6] {fold}: training subjects have one class only ({sorted(set(np.asarray(y).tolist()))}); need both to fit")
        params, tuned = opts[opts["model"]], None
        if opts.get("tuning", {}).get("enabled", False): # search on X, y only; test cases stay unseen
            params, inner, trials = tune(opts["model"], params, X, y, opts["tuning"])
            tuned = {**{k: params.get(k) for k in opts["tuning"]["space"]}, "inner_auc": inner, "default_inner_auc": float(trials["inner_auc"].iloc[0])}
            if opts["tuning"]["debug"]:
                print(f"[p6] {fold} | tuned inner AUC {inner:.3f} (fixed {tuned['default_inner_auc']:.3f}) | " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in tuned.items() if "auc" not in k), flush=True)
        model = build(opts["model"], params).fit(X, y)
        y_score = y_pred = None
        if X_test is not None:
            y_score = np.asarray(model.score(X_test), dtype=float)
            y_pred = (y_score >= opts["threshold"]).astype(int)
        if opts["debug"]:
            print(f"\n[p6] {fold} | {opts['model']} | train {len(y)} subjects × {X.shape[1]} features, test {0 if X_test is None else len(X_test)}"
                  + ("" if y_score is None else "\n" + "\n".join(f"    {sid}  {p:.3f}" for sid, p in zip(X_test.index, y_score))), flush=True)
        return {**kwargs, "X": X, "y": y, "X_test": X_test, "y_test": y_test, "fold": fold, "model": model, "y_score": y_score, "y_pred": y_pred, "tuned": tuned}
