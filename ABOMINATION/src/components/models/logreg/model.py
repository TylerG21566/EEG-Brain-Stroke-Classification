import numpy as np
from sklearn.linear_model import LogisticRegression
from components.models.base import SklearnModel

class LogRegModel(SklearnModel): # standardised logistic regression; l1_ratio picks the penalty (sklearn >= 1.8 deprecates `penalty`)
    scale = True

    def make(self, y: np.ndarray) -> LogisticRegression:
        o, l1 = self.opts, float(self.opts.get("l1_ratio", 0.0)) # 0 = L2, 1 = L1, between = elastic net
        return LogisticRegression(C=float(o.get("C", 1.0)), l1_ratio=l1, solver="saga" if l1 > 0 else "lbfgs", # lbfgs is L2-only
                                  class_weight=o.get("class_weight"), random_state=o.get("seed", 0), max_iter=int(o.get("max_iter", 1000)))
