from collections.abc import Iterator
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from components.base import Combo, Layer, Opts, PipelineStep, combo_dir, resolve

Row = dict[str, Any] # one row of p2's "processed" table
Epoch = dict[str, Any] # one epoch: signal (channels, win) + metadata
Task = dict[str, Any] # one subject's work for p4: row, combinations, opts, out

def epoch_name(e: Epoch) -> str: # e.g. ischemic_5_e0003_t0012.00s
    return f"{e['subject_id']}_e{e['epoch']:04d}_t{e['start']:07.2f}s"

def load(row: Row) -> np.ndarray: # one subject's p2 signal (channels, time)
    with np.load(row["file"]) as z:
        return z["signal"]

def windows(signal: np.ndarray, sfreq: float, duration: float, overlap: float, clip: float) -> tuple[np.ndarray, np.ndarray]: # all of one subject's epochs at once -> start times (s), (n_epochs, channels, win)
    # 1. window + step in samples (overlap % of the window is shared with the next one)
    win = round(duration * sfreq)
    step = max(1, round(win * (1 - overlap / 100)))
    if signal.shape[1] < win:
        return np.empty(0), np.empty((0, signal.shape[0], win), np.float32)
    x = sliding_window_view(signal, win, axis=1)[:, ::step].transpose(1, 0, 2).astype(np.float64) # view -> one copy
    # 2. z-score each channel along time; flat channels (std 0) become zeros, not NaN
    sd = x.std(axis=-1, keepdims=True)
    x = (x - x.mean(axis=-1, keepdims=True)) / np.where(sd == 0, 1.0, sd)
    # 3. clip outliers at ±clip SD
    return np.arange(len(x)) * step / sfreq, np.clip(x, -clip, clip).astype(np.float32)

def as_epochs(row: Row, duration: float, overlap: float, starts: np.ndarray, x: np.ndarray) -> Iterator[Epoch]: # epoch dicts for one subject (debugging / tests)
    for i, (s, sig) in enumerate(zip(starts, x)):
        yield {"subject_id": row["subject_id"], "label": row["label"], "subtype": row["subtype"], "split": row["split"], "epoch": i,
               "start": s, "duration": duration, "overlap": overlap, "sfreq": row["sfreq"], "channels": row["channels"], "signal": sig}

def save_epochs(out_dir: str | Path, row: Row, duration: float, overlap: float, starts: np.ndarray, x: np.ndarray) -> None: # debug: one .npy per epoch
    d = combo_dir(out_dir, duration, overlap)
    for e in as_epochs(row, duration, overlap, starts, x):
        np.save(d / f"{epoch_name(e)}.npy", e["signal"])

def clear(out_dir: str | Path, combinations: list[Combo]) -> None: # debug: fresh combo folders
    for d, o in combinations:
        c = combo_dir(out_dir, d, o)
        c.mkdir(parents=True, exist_ok=True)
        for f in c.glob("*.npy"):
            f.unlink()

def subjects(processed: pd.DataFrame, combinations: list[Combo], opts: Opts, out_dir: Path | None = None) -> Iterator[Task]: # lazy: one small picklable task per subject covering every combination
    if out_dir is not None: # cleared on the first pull
        clear(out_dir, combinations)
    for row in processed.to_dict("records"):
        yield {"row": row, "combinations": list(combinations), "opts": opts, "out": out_dir}

def epochs(processed: pd.DataFrame, duration: float, overlap: float, opts: Opts, out_dir: Path | None = None) -> Iterator[Epoch]: # every epoch for one combination, in this process (debugging / tests)
    if out_dir is not None:
        clear(out_dir, [(duration, overlap)])
    for row in processed.to_dict("records"):
        starts, x = windows(load(row), row["sfreq"], duration, overlap, opts["clip"])
        if out_dir is not None:
            save_epochs(out_dir, row, duration, overlap, starts, x)
        yield from as_epochs(row, duration, overlap, starts, x)

class Epoching(PipelineStep):
    section = "p3_epoching"

    def output(self, **kwargs: Any) -> Layer: # returns the task generator unconsumed; p4's workers do the epoching
        processed, opts = kwargs["processed"], self.opts
        out_dir = resolve(opts["out_dirs"][processed.attrs["dataset"]]) if opts["debug"] else None
        return {**kwargs, "subjects": subjects(processed, opts["combinations"], opts, out_dir)}
