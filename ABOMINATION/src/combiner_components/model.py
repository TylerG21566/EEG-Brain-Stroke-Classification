# Combiner: logistic regression on the base models' P(stroke), as log-odds; its weights say how much each model is trusted
import numpy as np
from sklearn.linear_model import LogisticRegression
from components.base import Opts

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

class CombinerModel:
    def __init__(self, opts: Opts) -> None: # opts = config["combiner"]
        self.model = LogisticRegression(C=opts["C"])

    def fit(self, S: np.ndarray, y: np.ndarray) -> "CombinerModel": # S: (n, models) out-of-fold scores
        self.model.fit(logit(S), y)
        return self

    def score(self, S: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(logit(S))[:, 1]

    @property
    def weights(self) -> np.ndarray: # one per base model
        return self.model.coef_[0]
