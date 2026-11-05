# Stacking for one p7 fold: EEG model (p5 -> p6) + demographic model -> combiner. Runs instead of p7's MINI when combiner.enabled.
# The combiner is fitted on out-of-fold scores from an inner CV over the training subjects, so it never sees a score from a model trained on that subject.
import copy
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from components.base import Layer, PipelineStep, groups
from combiner_components.model import CombinerModel
from demographic_components.model import DemographicModel

class Ensemble(PipelineStep):
    section = "combiner"

    def output(self, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray, demo: pd.DataFrame, steps: list[type[PipelineStep]], fold: str = "", **kwargs: Any) -> Layer: # demo: subject_id -> demographic columns; steps: the EEG mini-pipeline
        opts, quiet = self.opts, copy.deepcopy(self.config)
        quiet["p5_selection"]["debug"] = quiet["p6_model"]["debug"] = False # inner folds stay silent
        def eeg(a: pd.Index, b: pd.Index, cfg: dict, name: str) -> Layer: # EEG mini-pipeline fitted on cases a, scoring cases b
            layer = {**kwargs, "X": X_all.loc[a], "y": y_all[a].to_numpy(), "X_test": X_all.loc[b], "y_test": y_all[b].to_numpy(), "fold": name}
            for step in steps:
                layer = step(cfg).output(**layer)
            return layer
        def dem(a: pd.Index, b: pd.Index) -> np.ndarray: # demographic model fitted on cases a, scoring cases b
            return DemographicModel(self.config["demographic_model"]).fit(demo.reindex(a), y_all[a].to_numpy()).score(demo.reindex(b))
        X_all, y_all = pd.concat([X, X_test]), pd.Series(np.concatenate([y, y_test]), index=[*X.index, *X_test.index])
        # 1. inner CV on the training cases -> out-of-fold scores from both base models
        oof, g = np.zeros((len(X), 2)), groups(X.index)
        kind = StratifiedGroupKFold if len(set(g)) < len(g) else StratifiedKFold
        cv = kind(opts["inner_folds"], shuffle=True, random_state=self.config["p7_loader"]["seed"])
        for i, (a, b) in enumerate(cv.split(X, y, g) if kind is StratifiedGroupKFold else cv.split(X, y)):
            oof[b] = np.column_stack([eeg(X.index[a], X.index[b], quiet, f"{fold} | inner {i + 1}")["y_score"], dem(X.index[a], X.index[b])])
        # 2. both base models refit on every training case score the test cases; the combiner joins them
        out = eeg(X.index, X_test.index, self.config, fold)
        s = np.column_stack([out["y_score"], dem(X.index, X_test.index)])
        comb = CombinerModel(opts).fit(oof, y)
        y_score = comb.score(s)
        if opts["debug"]:
            print(f"[combiner] {fold} | weights eeg {comb.weights[0]:+.2f}, demographic {comb.weights[1]:+.2f}", flush=True)
        return {**out, "y_test": y_test, "y_score": y_score, "y_pred": (y_score >= opts["threshold"]).astype(int),
                "y_score_eeg": s[:, 0], "y_score_demo": s[:, 1], "weights": comb.weights}
