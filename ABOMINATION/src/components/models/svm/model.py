import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from components.models.base import SklearnModel

class SVMModel(SklearnModel): # support vector machine (RBF kernel by default) on standardised features
    scale = True # SVMs are distance-based, so features must share a scale

    def make(self, y: np.ndarray) -> CalibratedClassifierCV: # opts = config["p6_model"]["svm"]
        o = self.opts
        svc = SVC(C=o.get("C", 1.0), kernel=o.get("kernel", "rbf"), gamma=o.get("gamma", "scale"), class_weight=o.get("class_weight"), random_state=o.get("seed", 0))
        return CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False) # Platt scaling on internal 5-fold CV (sklearn >= 1.9 replacement for SVC(probability=True)); can slightly reorder vs decision_function, fine since AUC is what matters
