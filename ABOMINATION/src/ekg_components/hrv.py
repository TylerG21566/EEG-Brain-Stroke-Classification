# EKG -> heart-rate variability: R-peaks from the band-passed lead, beat-to-beat (RR) intervals, then per-case HRV features
import mne
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

FEATURES = ["hr", "sdnn", "rmssd", "pnn50", "cv", "lf_hf"] # bpm, ms, ms, share of beat changes > 50 ms, SD / mean RR, low / high frequency power

def read_ekg(path: str, seconds: float | None) -> tuple[np.ndarray | None, float]: # first EKG / ECG channel, first `seconds` of it (None = whole); None if the file has no lead
    raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
    fs = raw.info["sfreq"]
    ch = next((c for c in raw.ch_names if "EKG" in c.upper() or "ECG" in c.upper()), None)
    if ch is None:
        return None, fs
    return raw.get_data(picks=[ch], stop=None if seconds is None else min(raw.n_times, int(round(seconds * fs))))[0], fs

def rr_intervals(x: np.ndarray | None, fs: float, band: tuple[float, float]) -> np.ndarray: # seconds between R-peaks, kept to 30-200 bpm
    if x is None or len(x) < 10 * fs:
        return np.empty(0)
    y = np.abs(sosfiltfilt(butter(3, band, "bandpass", fs=fs, output="sos"), x)) # QRS energy band; abs, so lead polarity doesn't matter
    peaks, _ = find_peaks(y, height=0.4 * np.percentile(y, 98), distance=int(0.3 * fs)) # >= 0.3 s apart
    rr = np.diff(peaks) / fs
    return rr[(rr >= 0.3) & (rr <= 2.0)]

def hrv(rr: np.ndarray) -> dict[str, float]: # NaN when fewer than 30 beats
    if len(rr) < 30:
        return dict.fromkeys(FEATURES, np.nan)
    d, t = np.diff(rr), np.cumsum(rr)
    grid = np.arange(t[0], t[-1], 0.25) # RR series resampled at 4 Hz for the spectrum
    f, p = welch(np.interp(grid, t, rr) - rr.mean(), fs=4.0, nperseg=min(256, len(grid)))
    lf, hf = p[(f >= 0.04) & (f < 0.15)].sum(), p[(f >= 0.15) & (f < 0.4)].sum()
    return {"hr": 60 / rr.mean(), "sdnn": 1000 * rr.std(), "rmssd": 1000 * np.sqrt(np.mean(d ** 2)), "pnn50": float(np.mean(np.abs(d) > 0.05)),
            "cv": rr.std() / rr.mean(), "lf_hf": lf / hf if hf > 0 else np.nan}
