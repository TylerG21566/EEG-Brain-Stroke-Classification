import numpy as np
from sklearn.ensemble import RandomForestClassifier
from components.models.base import SklearnModel

class RandomForestModel(SklearnModel): # bagged decision trees; opts = config["p6_model"]["forest"]
    scale = False # trees are scale-free

    def make(self, y: np.ndarray) -> RandomForestClassifier:
        o = self.opts
        return RandomForestClassifier(n_estimators=o.get("n_estimators", 300), max_depth=o.get("max_depth"), min_samples_leaf=o.get("min_samples_leaf", 1),
                                      max_features=o.get("max_features", "sqrt"), class_weight=o.get("class_weight"), random_state=o.get("seed", 0), n_jobs=o.get("n_jobs"))
