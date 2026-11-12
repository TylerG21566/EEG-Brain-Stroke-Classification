import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from components.p3_epoching import windows

def subject_epochs(file: Path, sfreq_in: float, decimate: int, duration: float, overlap: float, clip: float, per_subject: int) -> np.ndarray: # one subject's p2 .npz -> (≤ per_subject, ch, win)
    with np.load(file) as z:
        sig = z["signal"][:, ::decimate] # plain decimation, no anti-alias filter: p2 already low-passed at 30 Hz, below the new Nyquist (62.5 Hz at 250 / 2)
    _, x = windows(sig, sfreq_in / decimate, duration, overlap, clip) # p3's epochs: z-scored per channel, clipped at ±clip SD
    if len(x) > per_subject: # evenly spaced across the recording
        x = x[np.linspace(0, len(x) - 1, per_subject).round().astype(int)]
    return np.ascontiguousarray(x)

def load_epochs(p2_dir: str | Path, subject_ids: list[str], sfreq_in: float, decimate: int, duration: float, overlap: float, clip: float, per_subject: int,
                workers: int | None = None) -> tuple[np.ndarray, np.ndarray]: # every subject's epochs, loaded once -> X float32 (N, ch, win), subj_index (N,) into subject_ids
    files = [Path(p2_dir) / f"{s}.npz" for s in subject_ids]
    with ThreadPoolExecutor(workers or min(8, os.cpu_count() or 1)) as pool: # numpy releases the GIL for file reads and most array maths
        parts = list(pool.map(lambda f: subject_epochs(f, sfreq_in, decimate, duration, overlap, clip, per_subject), files))
    idx = np.concatenate([np.full(len(x), i) for i, x in enumerate(parts)]).astype(int)
    return np.concatenate(parts).astype(np.float32, copy=False), idx
