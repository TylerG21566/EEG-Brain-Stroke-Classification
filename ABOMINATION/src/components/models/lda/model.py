import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from components.models.base import SklearnModel

class LDAModel(SklearnModel): # linear discriminant analysis on standardised features; shrinkage keeps the covariance stable with few cases per feature
    scale = True

    def make(self, y: np.ndarray) -> LinearDiscriminantAnalysis: # opts = config["p6_model"]["lda"]
        solver = self.opts.get("solver", "lsqr")
        shrink = {} if solver == "svd" else {"shrinkage": self.opts.get("shrinkage", "auto")} # "auto" = Ledoit-Wolf; svd doesn't support shrinkage
        priors = [0.5, 0.5] if self.opts.get("balanced", False) else None # equal priors ~ class_weight="balanced"; None = training class frequencies
        return LinearDiscriminantAnalysis(solver=solver, priors=priors, **shrink)
