from itertools import combinations
import numpy as np
from scipy.signal import welch

class WelchFeatures: # Welch PSD per channel -> relative band power + pairwise band ratios
    suffix = "_welch" # csv: <sid>_welch.csv

    def __init__(self, opts): # opts = config["p4_features"]; reads opts["welch"]
        w = opts["welch"]
        self.bands, self.window, self.max_s, self.min_n = dict(w["bands"]), w["window"], w["max_nperseg_s"], w["min_nperseg"]
        self.pairs = list(combinations(range(len(self.bands)), 2)) # a < b in band order

    def names(self, channels): # per channel: rel for each band, then each pairwise ratio (4 bands -> 10)
        b = list(self.bands)
        return [n for ch in channels for n in [f"{ch}_{k}_rel" for k in b] + [f"{ch}_{b[i]}_{b[j]}_ratio" for i, j in self.pairs]]

    def compute(self, x, sfreq, clip): # (n, ch, win) -> (n, ch * (bands + pairs)); clip is unused
        n, ch, win = x.shape
        k = len(self.bands) + len(self.pairs)
        if n == 0:
            return np.empty((0, ch * k))
        nperseg = min(win, max(self.min_n, min(round(self.max_s * sfreq), win // 2))) # ~2 s segments, never longer than the epoch
        f, psd = welch(x, fs=sfreq, window=self.window, nperseg=nperseg, noverlap=nperseg // 2, axis=-1) # one call: (n, ch, freqs)
        bp = np.stack([psd[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in self.bands.values()], axis=-1) # (n, ch, bands)
        i, j = np.array(self.pairs).T
        with np.errstate(invalid="ignore", divide="ignore"): # flat channels: 0/0, zeroed below
            rel, ratio = bp / bp.sum(-1, keepdims=True), bp[..., i] / (bp[..., j] + 1e-10)
        return np.nan_to_num(np.concatenate([rel, ratio], axis=-1)).reshape(n, ch * k)
