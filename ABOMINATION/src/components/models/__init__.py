from typing import Protocol
import numpy as np
import pandas as pd
from components.base import Opts

class Model(Protocol): # what p6 expects from every model family; one folder per family under components/models/
    def __init__(self, opts: Opts) -> None: ... # opts = config["p6_model"][<name>]
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Model": ... # training subjects × features (NaNs allowed), labels 0/1
    def score(self, X: pd.DataFrame) -> np.ndarray: ... # P(class 1), shape (n,); X must have the training columns

from components.models.xgb import XGBoostModel

MODELS: dict[str, type[Model]] = {"xgboost": XGBoostModel} # model name in the config -> class

def build(name: str, opts: Opts) -> Model: # fresh, unfitted model by name
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](opts)
