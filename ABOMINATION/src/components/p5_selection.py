import re
from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from components.base import Layer, PipelineStep
from components.features import FeatureClass
from components.p0_pipeline_config import Selector

def feature_name(col: str) -> str: # wide column -> per-epoch feature name, e.g. q90_F4_kurtosis -> F4_kurtosis
    return re.sub(r"^q\d+_", "", col)

def score(X: np.ndarray, y: np.ndarray, method: Selector, seed: int) -> np.ndarray: # one score per column, higher = more informative
    if method == Selector.MUTUAL_INFO:
        return mutual_info_classif(X, y, random_state=seed)
    return np.nan_to_num(f_classif(X, y)[0]) # ANOVA F; constant columns give NaN -> 0

def ranked(s: pd.Series, columns: int = 3) -> str: # scores (sorted) -> aligned rank / feature / score table, filled column by column
    cells = [f"{i:>3}  {name:<{max(map(len, s.index))}}  {v:6.3f}" for i, (name, v) in enumerate(s.items(), 1)]
    rows = -(-len(cells) // columns)
    return "\n".join("    " + "     ".join(cells[r::rows]) for r in range(rows))

class Selection(PipelineStep): # p5: top-k features on one fold's training subjects. Runs inside p7; test subjects (X_test) only get the same columns and training fill, nothing is computed from them
    section = "p5_selection"

    def output(self, X: pd.DataFrame, y: np.ndarray, families: dict[str, str], extractors: dict[str, FeatureClass], fold: str, X_test: pd.DataFrame | None = None, **kwargs: Any) -> Layer: # X: training subjects × features, families: column -> p4 feature class key, extractors: key -> class instance, X_test: test subjects × the same columns
        opts = self.opts
        # 1. only features switched on in the config (whole class and single features)
        on = [c for c in X.columns if families[c] in extractors and extractors[families[c]].keep(feature_name(c))]
        X = X[on]
        # 2. fill gaps (zero-epoch subjects) with this fold's training medians
        X = X.fillna(X.median()).fillna(0.0)
        # 3. score every feature and keep the top k
        s = pd.Series(score(X.to_numpy(), y, opts["method"], opts["seed"]), index=X.columns).sort_values(ascending=False)
        chosen = list(s.index[:min(opts["k"], len(s))])
        if opts["debug"]:
            print(f"\n[p5] {fold} | {opts['method']} | top {len(chosen)} of {len(on)} features\n{ranked(s[chosen])}", flush=True)
        fill = X.median()[chosen] # training medians (X is gap-free by now), for the test subjects
        out = {**kwargs, "X": X[chosen], "y": y, "chosen": chosen, "scores": s[chosen], "fill": fill, "fold": fold}
        if X_test is not None: # 4. test subjects: same columns, gaps filled with the training medians
            out["X_test"] = X_test[chosen].fillna(fill)
        return out
