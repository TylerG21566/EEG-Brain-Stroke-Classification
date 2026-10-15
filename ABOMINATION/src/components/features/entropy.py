import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from components.base import Opts

FEATURES = ["fuzzy", "sample", "approx"] # column names: <ch>_fuzzyen, <ch>_sampen, <ch>_apen
NAMES = {"fuzzy": "fuzzyen", "sample": "sampen", "approx": "apen"}

def templates(x: np.ndarray, m: int, count: int, baseline: bool) -> np.ndarray: # (B, N) -> (B, count, m) embedding vectors; baseline: subtract each vector's mean (fuzzy entropy)
    t = sliding_window_view(x, m, axis=-1)[:, :count]
    return t - t.mean(-1, keepdims=True) if baseline else t

def chebyshev(t: np.ndarray) -> np.ndarray: # (B, M, m) -> (B, M, M): max over the m coordinates of |t_i - t_j|
    d = np.abs(t[:, :, None, 0] - t[:, None, :, 0])
    for k in range(1, t.shape[-1]):
        np.maximum(d, np.abs(t[:, :, None, k] - t[:, None, :, k]), out=d)
    return d

def fuzzy(x: np.ndarray, m: int, r: np.ndarray, n: float) -> np.ndarray: # FuzzyEn = ln phi_m - ln phi_m+1; phi = mean over i != j of exp(-ln2 (d_ij / r)^n) on baseline-removed vectors
    count = x.shape[-1] - m # same number of vectors for m and m + 1
    phi = []
    for k in (m, m + 1):
        s = np.exp(-np.log(2) * (chebyshev(templates(x, k, count, True)) / r[:, None, None]) ** n).sum((1, 2))
        phi.append((s - count) / (count * (count - 1))) # drop the diagonal (self-similarity = 1)
    return np.log(phi[0]) - np.log(phi[1])

def sample(x: np.ndarray, m: int, r: np.ndarray) -> np.ndarray: # SampEn = -ln(A / B); B, A = matching pairs (d <= r, i != j) at lengths m, m + 1
    count = x.shape[-1] - m
    b, a = [((chebyshev(templates(x, k, count, False)) <= r[:, None, None]).sum((1, 2)) - count) for k in (m, m + 1)]
    return np.log(np.maximum(b, 1)) - np.log(np.maximum(a, 1)) # no matches (A = 0) capped at A = 1 instead of +inf

def approx(x: np.ndarray, m: int, r: np.ndarray) -> np.ndarray: # ApEn = phi_m - phi_m+1; phi = mean over i of ln(fraction of vectors within r, self included)
    phi = []
    for k in (m, m + 1):
        count = x.shape[-1] - k + 1
        phi.append(np.log((chebyshev(templates(x, k, count, False)) <= r[:, None, None]).mean(-1)).mean(-1))
    return phi[0] - phi[1]

class EntropyFeatures: # regularity entropies per channel: fuzzy (Tong 2025), sample (Peterson 2024, Tong 2024), approximate (Tong 2025)
    suffix = "_entropy" # csv: <sid>_entropy.csv

    def __init__(self, opts: Opts) -> None: # opts = config["p4_features"]; reads opts["entropy"]
        e, on = opts["entropy"], opts["entropy"].get("features", {}) # toggles: fuzzy, sample, approx (missing = on)
        self.on = [k for k in FEATURES if on.get(k, True)]
        self.m, self.r, self.n, self.decim, self.chunk = e["m"], e["r"], e["n"], e["decim"], e["chunk"]

    def names(self, channels: list[str]) -> list[str]: # e.g. F4_fuzzyen
        return [f"{ch}_{NAMES[k]}" for ch in channels for k in self.on]

    def keep(self, name: str) -> bool: # is this feature (e.g. F4_fuzzyen) switched on in the config?
        return name.rsplit("_", 1)[-1] in {NAMES[k] for k in self.on}

    def compute(self, x: np.ndarray, sfreq: float, clip: float) -> np.ndarray: # (n, ch, win) -> (n, ch * enabled); O(win²) per series, so decimated and run in chunks
        n, ch, _ = x.shape
        if n == 0 or not self.on:
            return np.empty((n, ch * len(self.on)))
        s = np.ascontiguousarray(x[..., ::self.decim], dtype=np.float32).reshape(n * ch, -1) # p2 bandpassed to 30 Hz, so 250 / 4 Hz loses nothing
        sd = s.std(-1)
        r = self.r * np.where(sd == 0, 1.0, sd).astype(np.float32) # tolerance = r × SD of each series; flat series give 0 below
        out = np.empty((n * ch, len(self.on)))
        for i in range(0, n * ch, self.chunk): # bounds memory: chunk × M² distances at a time
            c, rc = s[i:i + self.chunk], r[i:i + self.chunk]
            fns = {"fuzzy": lambda: fuzzy(c, self.m, rc, self.n), "sample": lambda: sample(c, self.m, rc), "approx": lambda: approx(c, self.m, rc)}
            out[i:i + self.chunk] = np.stack([fns[k]() for k in self.on], axis=-1)
        return np.nan_to_num(out).reshape(n, ch * len(self.on))
