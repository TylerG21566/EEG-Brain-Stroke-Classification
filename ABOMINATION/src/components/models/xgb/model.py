import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from components.base import Opts

DEPRECATED = r".*gblinear.*deprecated" # xgboost >= 3.1 warns that gblinear will be removed; still works, so silence it

def quiet(fn): # run fn without the gblinear deprecation warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=DEPRECATED)
        return fn()

TREE_ONLY = {"max_depth", "min_child_weight", "gamma", "subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode", "max_leaves", "max_bin", "grow_policy"} # dropped for gblinear

class XGBoostModel: # gradient-boosted trees (or gblinear) with the configured hyperparameters (Tamim Table 4.3 by default); NaNs handled natively
    def __init__(self, opts: Opts) -> None: # opts = config["p6_model"]["xgboost"]; every key but seed / scale_pos_weight goes straight to XGBClassifier
        self.opts, self.clf, self.columns = dict(opts), None, []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "XGBoostModel":
        params = {k: v for k, v in self.opts.items() if k not in ("seed", "scale_pos_weight")}
        linear = params.get("booster", "gbtree") == "gblinear"
        params = {k: v for k, v in params.items() if k not in TREE_ONLY} if linear else params
        extra = {"updater": "coord_descent"} if linear else {"tree_method": "hist"} # coord_descent: deterministic, unlike the default parallel shotgun
        spw = self.opts.get("scale_pos_weight", 1.0)
        if spw == "auto": # neg / pos of this fold's training labels; 1 if a class is missing
            neg, pos = int(np.sum(np.asarray(y) == 0)), int(np.sum(np.asarray(y) == 1))
            spw = neg / pos if neg and pos else 1.0
        self.clf = XGBClassifier(**params, random_state=self.opts.get("seed", 0), scale_pos_weight=float(spw),
                                 objective="binary:logistic", eval_metric="logloss", **extra)
        self.columns = list(X.columns) # score() reorders its X to this
        quiet(lambda: self.clf.fit(X.to_numpy(dtype=float), np.asarray(y)))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray: # P(class 1), shape (n,)
        if self.clf is None:
            raise RuntimeError("XGBoostModel.score called before fit")
        if set(X.columns) != set(self.columns) or X.shape[1] != len(self.columns):
            missing, extra = sorted(set(self.columns) - set(X.columns)), sorted(set(X.columns) - set(self.columns))
            raise ValueError(f"score X columns differ from training: missing {missing}, unexpected {extra}")
        return quiet(lambda: self.clf.predict_proba(X[self.columns].to_numpy(dtype=float)))[:, 1]
