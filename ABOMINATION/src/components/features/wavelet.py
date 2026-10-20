import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg") # headless: runs inside spawned multiprocessing workers
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_array_morlet
from components.base import Opts

def mean_power(x: np.ndarray, sfreq: float, freqs: np.ndarray, n_cycles: float, decim: int, chunk: int) -> np.ndarray: # (n_epochs, ch, win) -> (ch, freqs, times) Morlet power averaged over epochs, `chunk` epochs at a time
    cycles = np.minimum(n_cycles, 0.95 * 2 * np.pi * np.asarray(freqs) * x.shape[-1] / (10 * sfreq)) # MNE's wavelet spans 10σ = 10·n/(2πf) s; cap cycles so low freqs fit short epochs
    total = 0.0
    for i in range(0, len(x), chunk):
        total = total + tfr_array_morlet(x[i:i + chunk].astype(np.float64), sfreq, freqs, n_cycles=cycles, decim=decim, output="power", verbose=False).sum(axis=0)
    return total / len(x)

class WaveletFeatures: # Morlet time-frequency maps -> one PNG per subject per (duration, overlap)
    def __init__(self, opts: Opts) -> None: # opts = config["p4_features"]; reads opts["wavelet"]
        self.opts = opts["wavelet"]

    def plot(self, x: np.ndarray, sfreq: float, channels: list[str], title: str, path: str | Path) -> None: # x (n_epochs, ch, win) z-scored float32; writes path (.png); no-op if n_epochs == 0
        if len(x) == 0:
            return
        o = self.opts
        freqs = np.asarray(o["freqs"], float)
        db = 10 * np.log10(mean_power(x, sfreq, freqs, o["n_cycles"], o["decim"], o["chunk"]) + np.finfo(float).tiny) # (ch, freqs, times)
        t = np.arange(db.shape[-1]) * o["decim"] / sfreq # time within epoch (s)
        vmin, vmax = db.min(), db.max() # one colour scale across channels
        cols = math.ceil(math.sqrt(len(channels))) # near-square grid: 4 × 4 for 16 channels, 4 rows × 5 for TUH's 19
        rows = math.ceil(len(channels) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows), sharex=True, sharey=True, squeeze=False, constrained_layout=True)
        for ax, ch, p in zip(axes.flat, channels, db):
            mesh = ax.pcolormesh(t, freqs, p, shading="auto", vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(ch, fontsize=9)
        for i in range(len(channels), rows * cols): # unused panels; the one above each gets the x axis labels
            axes.flat[i].set_visible(False)
            axes.flat[i - cols].tick_params(labelbottom=True)
            axes.flat[i - cols].set_xlabel("time in epoch (s)")
        for ax in axes[-1]:
            ax.set_xlabel("time in epoch (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Hz")
        fig.colorbar(mesh, ax=axes, label="power (dB)", shrink=0.8)
        fig.suptitle(f"{title} (mean Morlet power over {len(x)} epochs)")
        fig.savefig(path, dpi=o["dpi"])
        plt.close(fig)
