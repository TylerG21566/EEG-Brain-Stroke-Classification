import numpy as np
from scipy.stats import kurtosis, skew

FEATURES = ["skew", "kurtosis", "entropy"] # per channel, so 3 × 16 = 48 per epoch

def entropy(x, bins, clip): # Shannon entropy (bits) of each channel's value histogram over [-clip, clip]; x: (..., win)
    idx = np.clip(((x + clip) * (bins / (2 * clip))).astype(np.int64), 0, bins - 1) # bin index per sample
    rows = np.arange(idx[..., 0].size).reshape(idx.shape[:-1])[..., None] # one histogram per (epoch, channel)
    h = np.bincount((rows * bins + idx).ravel(), minlength=rows.size * bins).reshape(*idx.shape[:-1], bins)
    p = h / x.shape[-1]
    return -np.sum(p * np.log2(p, where=p > 0, out=np.zeros_like(p)), axis=-1)

def epoch_features(x, bins, clip): # (n_epochs, channels, win) -> (n_epochs, channels, 3); flat channels give 0 rather than NaN
    return np.nan_to_num(np.stack([skew(x, axis=-1), kurtosis(x, axis=-1), entropy(x, bins, clip)], axis=-1))

class ShapeFeatures: # skew, kurtosis, histogram entropy per channel -> 3 × channels per epoch
    suffix = "" # csv: <sid>.csv

    def __init__(self, opts): # opts = config["p4_features"]; reads opts["shape"]
        s, on = opts["shape"], opts["shape"].get("features", {})
        self.bins, self.on = s["entropy_bins"], [k for k in FEATURES if on.get(k, True)] # toggled-off features are dropped

    def names(self, channels): # e.g. F4_kurtosis
        return [f"{ch}_{k}" for ch in channels for k in self.on]

    def keep(self, name): # is this feature (e.g. F4_kurtosis) switched on in the config?
        return name.rsplit("_", 1)[-1] in self.on

    def compute(self, x, sfreq, clip): # (n_epochs, channels, win) -> (n_epochs, enabled × channels); sfreq unused
        if not len(x):
            return np.empty((0, x.shape[1] * len(self.on)))
        return epoch_features(x, self.bins, clip)[..., [FEATURES.index(k) for k in self.on]].reshape(len(x), -1)
