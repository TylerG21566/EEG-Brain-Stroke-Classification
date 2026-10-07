"""Welch's method, step by step, on a synthetic EEG-like signal.

Writes two figures:
  welch_steps.png  - split -> taper -> FFT each -> average
  welch_why.png    - why: single FFT is noisy, longer signal doesn't help,
                     tapering stops leakage, segment-length trade-off
Run: python3 welch_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt

FS = 250
rng = np.random.default_rng(1)


def make_signal(seconds):
    """Delta 2.5 Hz + alpha 10 Hz + weaker 11.5 Hz rhythm, buried in noise."""
    t = np.arange(int(seconds * FS)) / FS
    x = (1.2 * np.sin(2 * np.pi * 2.5 * t)
         + 1.0 * np.sin(2 * np.pi * 10 * t)
         + 0.6 * np.sin(2 * np.pi * 11.5 * t)
         + 2.0 * rng.standard_normal(t.size))
    return t, x


def spectrum(seg, window):
    """Power spectrum of one segment (one-sided, density-scaled)."""
    w = window(seg.size)
    p = np.abs(np.fft.rfft((seg - seg.mean()) * w)) ** 2 / (FS * (w ** 2).sum())
    p[1:-1] *= 2
    return np.fft.rfftfreq(seg.size, 1 / FS), p


def welch(x, nperseg, window=np.hamming):
    """Split into 50%-overlapping segments, taper, FFT each, average."""
    starts = list(range(0, x.size - nperseg + 1, nperseg // 2))
    spectra = [spectrum(x[s:s + nperseg], window)[1] for s in starts]
    return np.fft.rfftfreq(nperseg, 1 / FS), np.mean(spectra, axis=0), spectra, starts


def mark_peaks(ax):
    for f0 in (2.5, 10, 11.5):
        ax.axvline(f0, color="k", ls=":", lw=0.8, alpha=0.5)


t, x = make_signal(8)
NPERSEG = 500  # 2 s, as in Tamim's pipeline
f, p_welch, spectra, starts = welch(x, NPERSEG)
colours = plt.cm.viridis(np.linspace(0, 0.9, len(starts)))

# ---------------------------------------------------------------- Figure 1: the steps
fig, ax = plt.subplots(4, 1, figsize=(11, 13))

ax[0].plot(t, x, color="0.3", lw=0.6)
for i, (s, c) in enumerate(zip(starts, colours)):
    y = 9 + 1.4 * i
    ax[0].fill_between([s / FS, (s + NPERSEG) / FS], y - 0.55, y + 0.55, color=c, alpha=0.6)
    ax[0].text((s + NPERSEG / 2) / FS, y, f"seg {i + 1}", ha="center", va="center",
               fontsize=8, color="white", fontweight="bold")
ax[0].set_title(f"Step 1 - SPLIT: an 8 s signal cut into {len(starts)} segments of 2 s, "
                "each overlapping the next by 50%")
ax[0].set_xlabel("Time (s)")
ax[0].set_yticks([])

seg = x[:NPERSEG]
ts = t[:NPERSEG]
w = np.hamming(NPERSEG)
ax[1].plot(ts, seg, color="0.75", lw=0.8, label="segment 1 (raw)")
ax[1].plot(ts, seg * w, color=colours[0], lw=1, label="segment 1 x Hamming (tapered)")
ax[1].plot(ts, w * np.abs(seg).max(), color="#d95f02", lw=2, label="Hamming window shape")
ax[1].set_title("Step 2 - TAPER: multiply each segment by a window that fades the edges to zero")
ax[1].set_xlabel("Time (s)")
ax[1].set_yticks([])
ax[1].legend(fontsize=8, loc="lower right")

for i, (p, c) in enumerate(zip(spectra, colours)):
    ax[2].semilogy(f, p, color=c, lw=1, alpha=0.8, label=f"seg {i + 1}")
ax[2].set_title("Step 3 - FFT EACH: one power spectrum per segment. Each one alone is noisy "
                "and they disagree")
ax[2].legend(fontsize=7, ncol=len(starts), loc="upper right")

for p in spectra:
    ax[3].semilogy(f, p, color="0.8", lw=0.8)
ax[3].semilogy(f, p_welch, color="#d95f02", lw=2.5, label="average = Welch PSD")
ax[3].set_title("Step 4 - AVERAGE: noise partly cancels, real rhythms (dotted lines) are in every "
                "segment so they survive")
ax[3].legend(fontsize=9, loc="upper right")

for a in ax[2:]:
    mark_peaks(a)
    a.set_xlim(0, 30)
    a.set_ylim(1e-4, 2)
    a.set_xlabel("Frequency (Hz)")
    a.set_ylabel("Power (log)")

fig.tight_layout()
fig.savefig("welch_steps.png", dpi=120)

# ---------------------------------------------------------------- Figure 2: the why
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# A: one FFT of the whole epoch vs Welch
f_per, p_per = spectrum(x, np.hamming)
ax[0, 0].semilogy(f_per, p_per, color="0.6", lw=0.8, label="single FFT of whole 8 s (periodogram)")
ax[0, 0].semilogy(f, p_welch, color="#d95f02", lw=2.5, label="Welch (7 x 2 s segments, averaged)")
ax[0, 0].set_title("A. One big FFT is jagged. Welch is smooth")

# B: a longer signal does not make the periodogram less noisy
t32, x32 = make_signal(32)
f32, p32 = spectrum(x32, np.hamming)
ax[0, 1].semilogy(f_per, p_per, color="0.6", lw=0.8, label="periodogram, 8 s")
ax[0, 1].semilogy(f32, p32 / 30, color="#2b83ba", lw=0.6, label="periodogram, 32 s (shifted down x30)")
ax[0, 1].set_title("B. 4x longer signal: more bins, each still just as noisy")

for a in (ax[0, 0], ax[0, 1]):
    mark_peaks(a)
    a.set_xlim(0, 30)
    a.set_xlabel("Frequency (Hz)")
    a.set_ylabel("Power (log)")
    a.legend(fontsize=8, loc="upper right")

# C: spectral leakage, rectangular vs Hamming on a pure sine between bins
tt = np.arange(NPERSEG) / FS
sine = np.sin(2 * np.pi * 10.25 * tt)  # 10.25 Hz falls between the 0.5 Hz bins
NPAD = 8192  # zero-padding only to draw the leakage shape smoothly
for win, name, c in [(np.ones, "no taper (rectangular)", "0.4"), (np.hamming, "Hamming taper", "#d95f02")]:
    pc = np.abs(np.fft.rfft(sine * win(NPERSEG), n=NPAD)) ** 2
    fc = np.fft.rfftfreq(NPAD, 1 / FS)
    ax[1, 0].plot(fc, 10 * np.log10(pc / pc.max() + 1e-12), color=c, lw=1.5, label=name)
ax[1, 0].set_xlim(0, 30)
ax[1, 0].set_ylim(-90, 5)
ax[1, 0].axvline(10.25, color="k", ls=":", lw=0.8)
ax[1, 0].set_title("C. Leakage: a pure 10.25 Hz sine. Without a taper,\n"
                   "power smears across every frequency")
ax[1, 0].set_xlabel("Frequency (Hz)")
ax[1, 0].set_ylabel("Power relative to peak (dB)")
ax[1, 0].legend(fontsize=9)

# D: segment-length trade-off
for nps, c, lw in [(125, "#2b83ba", 2.2), (500, "#d95f02", 2.2), (2000, "0.55", 1)]:
    fd, pd, sp, _ = welch(x, nps)
    ax[1, 1].semilogy(fd, pd, color=c, lw=lw,
                      label=f"segment {nps / FS:g} s: {FS / nps:g} Hz bins, {len(sp)} averaged")
mark_peaks(ax[1, 1])
ax[1, 1].set_xlim(0, 30)
ax[1, 1].set_title("D. Trade-off: short segments are smooth but blur 10 & 11.5 Hz together;\n"
                   "one long segment resolves them but is noisy")
ax[1, 1].set_xlabel("Frequency (Hz)")
ax[1, 1].set_ylabel("Power (log)")
ax[1, 1].legend(fontsize=8, loc="upper right")

fig.tight_layout()
fig.savefig("welch_why.png", dpi=110)
print("wrote welch_steps.png, welch_why.png")
