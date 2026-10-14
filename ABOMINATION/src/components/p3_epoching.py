from pathlib import Path
import numpy as np
from components.base import PipelineStep, resolve

def epoch_name(e): # e.g. ischemic_5_e0003_t0012.00s
    return f"{e['subject_id']}_e{e['epoch']:04d}_t{e['start']:07.2f}s"

def combo_dir(out_dir, duration, overlap): # e.g. out/p3/Archive/4s/50pct
    return Path(out_dir) / f"{duration:g}s" / f"{overlap:g}pct"

def subject_epochs(row, duration, overlap, opts, out=None): # one subject's epochs, lazily; runs inside a p4 worker
    with np.load(row["file"]) as z:
        signal = z["signal"]
    sfreq = row["sfreq"]
    # 1. window + step in samples (overlap % of the window is shared with the next one)
    win = round(duration * sfreq)
    step = max(1, round(win * (1 - overlap / 100)))
    for i, s in enumerate(range(0, signal.shape[1] - win + 1, step)):
        x = signal[:, s:s + win].astype(np.float64)
        # 2. z-score each channel along time; flat channels (std 0) become zeros, not NaN
        sd = x.std(axis=1, keepdims=True)
        x = (x - x.mean(axis=1, keepdims=True)) / np.where(sd == 0, 1.0, sd)
        # 3. clip outliers at ±clip SD
        x = np.clip(x, -opts["clip"], opts["clip"]).astype(np.float32)
        e = {"subject_id": row["subject_id"], "label": row["label"], "subtype": row["subtype"], "split": row["split"], "epoch": i,
             "start": s / sfreq, "duration": duration, "overlap": overlap, "sfreq": sfreq, "channels": row["channels"], "signal": x}
        if out is not None:
            np.save(out / f"{epoch_name(e)}.npy", x)
        yield e

def subjects(processed, duration, overlap, opts, out_dir=None): # lazy: one small picklable task per subject; p4 workers expand them
    out = None
    if out_dir is not None: # debug: fresh combo folder, cleared on the first pull
        out = combo_dir(out_dir, duration, overlap)
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.npy"):
            f.unlink()
    for row in processed.to_dict("records"):
        yield {"row": row, "duration": duration, "overlap": overlap, "opts": opts, "out": out}

def epochs(processed, duration, overlap, opts, out_dir=None): # every epoch in order, in this process (debugging / tests)
    for task in subjects(processed, duration, overlap, opts, out_dir):
        yield from subject_epochs(**task)

class Epoching(PipelineStep):
    section = "p3_epoching"

    def output(self, **kwargs): # returns the task generator unconsumed; p4 drives it
        processed, opts = kwargs["processed"], self.opts
        dataset = processed.attrs["dataset"]
        out_dir = resolve(opts["out_dirs"][dataset]) if opts["debug"] else None
        return {**kwargs, "subjects": subjects(processed, kwargs["duration"], kwargs["overlap"], opts, out_dir)}
