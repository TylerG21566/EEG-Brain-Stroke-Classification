import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from components.base import Opts

class XGBoostModel: # gradient-boosted trees with fixed hyperparameters (Tamim Table 4.3); NaNs handled natively
    def __init__(self, opts: Opts) -> None: # opts = config["p6_model"]["xgboost"]; every key but seed / scale_pos_weight goes straight to XGBClassifier
        self.opts, self.clf, self.columns = dict(opts), None, []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "XGBoostModel":
        params = {k: v for k, v in self.opts.items() if k not in ("seed", "scale_pos_weight")}
        spw = self.opts.get("scale_pos_weight", 1.0)
        if spw == "auto": # neg / pos of this fold's training labels; 1 if a class is missing
            neg, pos = int(np.sum(np.asarray(y) == 0)), int(np.sum(np.asarray(y) == 1))
            spw = neg / pos if neg and pos else 1.0
        self.clf = XGBClassifier(**params, random_state=self.opts.get("seed", 0), scale_pos_weight=float(spw),
                                 objective="binary:logistic", eval_metric="logloss", tree_method="hist")
        self.columns = list(X.columns) # score() reorders its X to this
        self.clf.fit(X.to_numpy(dtype=float), np.asarray(y))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray: # P(class 1), shape (n,)
        if self.clf is None:
            raise RuntimeError("XGBoostModel.score called before fit")
        if set(X.columns) != set(self.columns) or X.shape[1] != len(self.columns):
            missing, extra = sorted(set(self.columns) - set(X.columns)), sorted(set(X.columns) - set(self.columns))
            raise ValueError(f"score X columns differ from training: missing {missing}, unexpected {extra}")
        return self.clf.predict_proba(X[self.columns].to_numpy(dtype=float))[:, 1]
