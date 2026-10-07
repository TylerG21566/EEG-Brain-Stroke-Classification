"""Synthetic PSD demo: one 4 s epoch of one channel -> Welch PSD.

Matches Tamim's settings: fs = 250 Hz, Hamming window, nperseg = 500 (2 s), 50% overlap.
Run: python3 psd_demo.py  -> writes psd_demo.png
"""
import numpy as np
import matplotlib.pyplot as plt

FS = 250
EPOCH_S = 4
NPERSEG = 500
STEP = NPERSEG // 2  # 50% overlap
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}
BAND_COLOURS = ["#2b83ba", "#e6a817", "#1a9e77", "#d95f02"]

rng = np.random.default_rng(0)
t = np.arange(EPOCH_S * FS) / FS


def make_epoch(amps):
    """Sum of sines (delta 2 Hz, theta 6 Hz, alpha 10 Hz, beta 20 Hz) plus noise."""
    freqs = [2, 6, 10, 20]
    sig = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for a, f in zip(amps, freqs))
    return sig + 0.5 * rng.standard_normal(t.size)


def welch(x):
    """Welch PSD: split into overlapping Hamming-tapered sub-windows, FFT each, average."""
    win = np.hamming(NPERSEG)
    scale = 1.0 / (FS * (win ** 2).sum())
    starts = range(0, x.size - NPERSEG + 1, STEP)
    spectra = []
    for s in starts:
        seg = x[s:s + NPERSEG]
        seg = (seg - seg.mean()) * win
        p = np.abs(np.fft.rfft(seg)) ** 2 * scale
        p[1:-1] *= 2  # one-sided
        spectra.append(p)
    return np.fft.rfftfreq(NPERSEG, 1 / FS), np.mean(spectra, axis=0), list(starts)


normal = make_epoch([0.8, 0.6, 2.0, 0.5])   # strong alpha
stroke = make_epoch([2.2, 1.4, 0.6, 0.25])  # slowing: delta/theta up, alpha/beta down

f, p_norm, starts = welch(normal)
_, p_stroke, _ = welch(stroke)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1.4]})

# Top: the epoch in time, with the Welch sub-windows marked
ax1.plot(t, normal, color="#1a9e77", lw=0.8, label="Normal-like")
ax1.plot(t, stroke - 8, color="#d95f02", lw=0.8, label="Stroke-like (offset)")
for i, s in enumerate(starts):
    y = 8.5 - 1.3 * i
    ax1.fill_between([s / FS, (s + NPERSEG) / FS], y - 0.5, y + 0.5, color="grey", alpha=0.35)
    ax1.text((s + NPERSEG / 2) / FS, y, f"Welch sub-window {i + 1} (2 s)", ha="center",
             va="center", fontsize=8)
ax1.set_title(f"Step 1: one {EPOCH_S} s epoch, one channel ({EPOCH_S * FS} samples)")
ax1.set_xlabel("Time (s)")
ax1.set_yticks([])
ax1.legend(loc="lower right", fontsize=8)

# Bottom: the resulting PSD
for (name, (lo, hi)), c in zip(BANDS.items(), BAND_COLOURS):
    ax2.axvspan(lo, hi, color=c, alpha=0.12)
    ax2.text((lo + hi) / 2, ax2.get_ylim()[1], name, ha="center", va="bottom", color=c,
             fontweight="bold", transform=ax2.get_xaxis_transform())
ax2.semilogy(f, p_norm, color="#1a9e77", lw=2, label="Normal-like")
ax2.semilogy(f, p_stroke, color="#d95f02", lw=2, label="Stroke-like")
ax2.set_xlim(0, 32)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Power (log scale)")
ax2.set_title("Step 2: Welch PSD of that epoch (one curve per channel per epoch)", pad=18)
ax2.legend(fontsize=9)

fig.tight_layout()
fig.savefig("psd_demo.png", dpi=130)

# Band powers for the table in visualizations.md
for name, p in [("Normal-like", p_norm), ("Stroke-like", p_stroke)]:
    bp = {b: p[(f >= lo) & (f < hi)].sum() for b, (lo, hi) in BANDS.items()}
    print(name, {k: round(v, 3) for k, v in bp.items()},
          "delta/alpha =", round(bp["Delta"] / bp["Alpha"], 2))
