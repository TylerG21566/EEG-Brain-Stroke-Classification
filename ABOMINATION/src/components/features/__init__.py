from typing import Protocol
import numpy as np

class FeatureClass(Protocol): # what p4 and p5 expect from every per-epoch feature class (WaveletFeatures only plots, so it isn't one)
    suffix: str # CSV name: <subject><suffix>.csv
    def names(self, channels: list[str]) -> list[str]: ... # feature names, in compute() column order
    def keep(self, name: str) -> bool: ... # is this feature switched on in the config?
    def compute(self, x: np.ndarray, sfreq: float, clip: float) -> np.ndarray: ... # (n_epochs, ch, win) -> (n_epochs, len(names))

from components.features.shape import ShapeFeatures
from components.features.welch import WelchFeatures
from components.features.wavelet import WaveletFeatures
from components.features.asymmetry import AsymmetryFeatures
from components.features.entropy import EntropyFeatures
