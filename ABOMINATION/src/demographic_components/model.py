# Demographic model: logistic regression on patient details (age, sex); gaps filled with the training medians, then standardised
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from components.base import Opts

class DemographicModel: # same fit / score interface as components.models
    def __init__(self, opts: Opts) -> None: # opts = config["demographic_model"]
        self.features = list(opts["features"])
        self.pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LogisticRegression(C=opts["C"]))

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DemographicModel":
        self.pipe.fit(X[self.features], y)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray: # P(stroke)
        return self.pipe.predict_proba(X[self.features])[:, 1]
