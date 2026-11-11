import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from components.base import Opts

class SklearnModel: # shared by the sklearn families: training medians fill NaNs, optional standardising, then the classifier from make()
    scale = True # False for scale-free models (trees)

    def __init__(self, opts: Opts) -> None: # opts = config["p6_model"][<name>]
        self.opts, self.pipe, self.columns = dict(opts), None, []

    def make(self, y: np.ndarray): # unfitted sklearn classifier with predict_proba; y = this fold's training labels
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SklearnModel":
        y = np.asarray(y)
        self.pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), *([StandardScaler()] if self.scale else []), self.make(y))
        self.columns = list(X.columns) # score() reorders its X to this
        self.pipe.fit(X.to_numpy(dtype=float), y)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray: # P(class 1), shape (n,)
        if self.pipe is None:
            raise RuntimeError(f"{type(self).__name__}.score called before fit")
        if set(X.columns) != set(self.columns) or X.shape[1] != len(self.columns):
            missing, extra = sorted(set(self.columns) - set(X.columns)), sorted(set(X.columns) - set(self.columns))
            raise ValueError(f"score X columns differ from training: missing {missing}, unexpected {extra}")
        return self.pipe.predict_proba(X[self.columns].to_numpy(dtype=float))[:, 1]
