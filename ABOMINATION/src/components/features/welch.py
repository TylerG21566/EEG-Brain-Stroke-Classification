from itertools import combinations
import numpy as np
from scipy.signal import welch

EXTRA = ["dtabr", "spec_mean", "slope"] # per channel, after the band features

class WelchFeatures: # Welch PSD per channel -> relative band power + pairwise band ratios + DTABR, spectral mean, 1/f slope
    suffix = "_welch" # csv: <sid>_welch.csv

    def __init__(self, opts): # opts = config["p4_features"]; reads opts["welch"]
        w = opts["welch"]
        self.bands, self.window, self.max_s, self.min_n = dict(w["bands"]), w["window"], w["max_nperseg_s"], w["min_nperseg"]
        self.slow, self.fast = [list(self.bands).index(b) for b in w["slow"]], [list(self.bands).index(b) for b in w["fast"]]
        self.pairs = list(combinations(range(len(self.bands)), 2)) # a < b in band order
        self.fmin, self.fmax = min(lo for lo, _ in self.bands.values()), max(hi for _, hi in self.bands.values()) # 1-30 Hz

    def names(self, channels): # per channel: rel for each band, each pairwise ratio, then EXTRA (4 bands -> 13)
        b = list(self.bands)
        return [n for ch in channels for n in [f"{ch}_{k}_rel" for k in b] + [f"{ch}_{b[i]}_{b[j]}_ratio" for i, j in self.pairs] + [f"{ch}_{e}" for e in EXTRA]]

    def psd(self, x, sfreq): # (n, ch, win) -> freqs, (n, ch, freqs) limited to [fmin, fmax); one vectorised call
        win = x.shape[-1]
        nperseg = min(win, max(self.min_n, min(round(self.max_s * sfreq), win // 2))) # ~2 s segments, never longer than the epoch
        f, p = welch(x, fs=sfreq, window=self.window, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
        keep = (f >= self.fmin) & (f < self.fmax)
        return f[keep], p[..., keep]

    def compute(self, x, sfreq, clip): # (n, ch, win) -> (n, ch * (bands + pairs + 3)); clip is unused
        n, ch, _ = x.shape
        k = len(self.bands) + len(self.pairs) + len(EXTRA)
        if n == 0:
            return np.empty((0, ch * k))
        f, psd = self.psd(x, sfreq)
        bp = np.stack([psd[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in self.bands.values()], axis=-1) # (n, ch, bands)
        i, j = np.array(self.pairs).T
        lf, lp = np.log10(f), np.log10(psd + 1e-20) # 1/f slope: least-squares line through log-log PSD
        with np.errstate(invalid="ignore", divide="ignore"): # flat channels: 0/0, zeroed below
            rel, ratio = bp / bp.sum(-1, keepdims=True), bp[..., i] / (bp[..., j] + 1e-10)
            dtabr = bp[..., self.slow].sum(-1) / (bp[..., self.fast].sum(-1) + 1e-10) # (delta + theta) / (alpha + beta)
            spec_mean = (psd * f).sum(-1) / psd.sum(-1) # power-weighted mean frequency (Hz)
        slope = ((lf - lf.mean()) * (lp - lp.mean(-1, keepdims=True))).sum(-1) / ((lf - lf.mean()) ** 2).sum()
        extra = np.stack([dtabr, spec_mean, slope], axis=-1)
        return np.nan_to_num(np.concatenate([rel, ratio, extra], axis=-1)).reshape(n, ch * k)
