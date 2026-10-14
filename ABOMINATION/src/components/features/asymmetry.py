import numpy as np
from components.features.welch import WelchFeatures

class AsymmetryFeatures: # Brain Symmetry Index per left/right pair: mean over 1-30 Hz of |(R - L) / (R + L)| of the Welch PSD
    suffix = "_bsi" # csv: <sid>_bsi.csv

    def __init__(self, opts): # opts = config["p4_features"]; reads opts["asymmetry"] and the Welch settings
        self.pairs, self.welch = [tuple(p) for p in opts["asymmetry"]["pairs"]], WelchFeatures(opts)
        self.idx = []

    def names(self, channels): # one per pair present in channels (case-insensitive) + their mean; also fixes the indices compute() uses
        up = [c.upper() for c in channels]
        found = [(l, r) for l, r in self.pairs if l.upper() in up and r.upper() in up]
        self.idx = [(up.index(l.upper()), up.index(r.upper())) for l, r in found]
        return [f"{l}_{r}_bsi" for l, r in found] + ["mean_bsi"] if found else []

    def compute(self, x, sfreq, clip): # (n, ch, win) -> (n, pairs + 1); call names() first; clip is unused
        k = len(self.idx) + 1 if self.idx else 0
        if len(x) == 0 or not k:
            return np.empty((len(x), k))
        _, psd = self.welch.psd(x, sfreq)
        l, r = np.array(self.idx).T
        with np.errstate(invalid="ignore", divide="ignore"): # flat pair: 0/0, zeroed below
            bsi = np.nan_to_num(np.abs((psd[:, r] - psd[:, l]) / (psd[:, r] + psd[:, l]))).mean(-1) # (n, pairs); 0 = symmetric, 1 = one-sided
        return np.concatenate([bsi, bsi.mean(-1, keepdims=True)], axis=-1)
