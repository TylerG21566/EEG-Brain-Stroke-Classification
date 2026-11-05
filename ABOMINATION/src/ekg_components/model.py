# EKG model: logistic regression on the HRV features; same imputer -> scaler -> logistic regression as the demographic model
from demographic_components.model import DemographicModel

class EKGModel(DemographicModel): # opts = config["ekg_model"]
    pass
