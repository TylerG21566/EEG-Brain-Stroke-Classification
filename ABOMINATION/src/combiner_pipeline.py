# Stacking for one p7 fold: EEG model (p5 -> p6) + the extra models in combiner.models (demographic, EKG) -> combiner. Runs instead of p7's MINI when combiner.enabled.
# The combiner is fitted on out-of-fold scores from an inner CV over the training subjects, so it never sees a score from a model trained on that subject.
import copy
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from components.base import Layer, PipelineStep, groups
from combiner_components.model import CombinerModel
from components.base import Config
from demographic_components.model import DemographicModel
from demographic_pipeline import demographics
from ekg_components.model import EKGModel
from ekg_pipeline import ekg_features

SIDE = {"demographic": (demographics, DemographicModel, "demographic_model"), "ekg": (ekg_features, EKGModel, "ekg_model")} # name -> (inputs loader, model, config section)

def side_inputs(config: Config) -> dict[str, pd.DataFrame]: # each enabled extra model's table (subject_id -> features), loaded once per p7 run
    return {n: SIDE[n][0](config) for n in config["combiner"]["models"]}

def inner_splits(config: Config, ids: Any, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]: # the combiner's inner CV over training cases ids (positions); patient-grouped when patients repeat
    g = groups(ids)
    kind = StratifiedGroupKFold if len(set(g)) < len(g) else StratifiedKFold
    cv = kind(config["combiner"]["inner_folds"], shuffle=True, random_state=config["p7_loader"]["seed"])
    return list(cv.split(ids, y, g) if kind is StratifiedGroupKFold else cv.split(ids, y))

def extra_scores(config: Config, side: dict[str, pd.DataFrame], a: Any, y_a: np.ndarray, b: Any) -> list[np.ndarray]: # every extra model fitted on cases a, scoring cases b (cases without inputs -> training medians)
    return [SIDE[n][1](config[SIDE[n][2]]).fit(d.reindex(a), y_a).score(d.reindex(b)) for n, d in side.items()]

class Ensemble(PipelineStep):
    section = "combiner"

    def output(self, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray, side: dict[str, pd.DataFrame], steps: list[type[PipelineStep]], fold: str = "", **kwargs: Any) -> Layer: # side: extra model name -> its inputs (side_inputs); steps: the EEG mini-pipeline
        opts, quiet = self.opts, copy.deepcopy(self.config)
        quiet["p5_selection"]["debug"] = quiet["p6_model"]["debug"] = False # inner folds stay silent
        quiet["p6_model"].setdefault("tuning", {})["enabled"] = False # tune only on the final refit (no third level of nesting)
        def eeg(a: pd.Index, b: pd.Index, cfg: dict, name: str) -> Layer: # EEG mini-pipeline fitted on cases a, scoring cases b
            layer = {**kwargs, "X": X_all.loc[a], "y": y_all[a].to_numpy(), "X_test": X_all.loc[b], "y_test": y_all[b].to_numpy(), "fold": name}
            for step in steps:
                layer = step(cfg).output(**layer)
            return layer
        extra = lambda a, b: extra_scores(self.config, side, a, y_all[a].to_numpy(), b)
        names = ["eeg", *side]
        X_all, y_all = pd.concat([X, X_test]), pd.Series(np.concatenate([y, y_test]), index=[*X.index, *X_test.index])
        # 1. inner CV on the training cases -> out-of-fold scores from both base models
        oof = np.zeros((len(X), len(names)))
        for i, (a, b) in enumerate(inner_splits(self.config, X.index, y)):
            oof[b] = np.column_stack([eeg(X.index[a], X.index[b], quiet, f"{fold} | inner {i + 1}")["y_score"], *extra(X.index[a], X.index[b])])
        # 2. both base models refit on every training case score the test cases; the combiner joins them
        out = eeg(X.index, X_test.index, self.config, fold)
        s = np.column_stack([out["y_score"], *extra(X.index, X_test.index)])
        comb = CombinerModel(opts).fit(oof, y)
        y_score = comb.score(s)
        if opts["debug"]:
            print(f"[combiner] {fold} | weights " + ", ".join(f"{n} {w:+.2f}" for n, w in zip(names, comb.weights)), flush=True)
        return {**out, "y_test": y_test, "y_score": y_score, "y_pred": (y_score >= opts["threshold"]).astype(int),
                **{f"y_score_{n}": s[:, i] for i, n in enumerate(names)}, "weights": dict(zip(names, comb.weights))}
