from functools import partial
from multiprocessing import Pool
import mne
import numpy as np
import pandas as pd
from components.base import PipelineStep, resolve
from components.p0_pipeline_config import Dataset

# per-sample fields kept in the returned "processed" table (signals are saved to the .npz files)
META = ["subject_id", "label", "subtype", "split", "sfreq", "channels", "eye_regressed", "paths"]

def archive_samples(df): # Tamim: stitch each subject's 4 s trials into one recording
    df = df.sort_values(["split", "label", "subtype", "subject", "file"])
    # rows are sorted by file number, so each group holds one subject's trials in order;
    # concatenating along time (axis=1) joins 50 × (16, 1024) trials into one (16, 51200) ≈ 200 s signal
    for sid, g in df.groupby("subject_id", sort=False):
        r = g.iloc[0]
        yield {"subject_id": sid, "label": r.label, "subtype": r.subtype, "split": r.split, "signal": np.concatenate(list(g.signal), axis=1),
               "sfreq": df.attrs["sfreq"], "channels": list(df.attrs["channels"]), "paths": list(g.path)}

def figshare_samples(df): # FIgshare: already one continuous EDF per subject, no stitching
    for r in df.itertuples():
        yield {"subject_id": r.subject_id, "label": r.label, "subtype": r.subtype, "split": None, "signal": r.signal,
               "sfreq": df.attrs["sfreq"], "channels": list(df.attrs["channels"]), "paths": [r.path]}

SAMPLERS = {Dataset.ARCHIVE: archive_samples, Dataset.FIGSHARE: figshare_samples}

def process(sample, opts): # opts = config["p2_sample_processing"]
    sfreq = opts["sfreq"]
    # 1. drop EOG + trigger channels
    non_eeg = {c.upper() for c in opts["non_eeg"]}
    keep = [i for i, c in enumerate(sample["channels"]) if c.upper() not in non_eeg]
    chans = [sample["channels"][i] for i in keep]
    x = np.asarray(sample["signal"], dtype=np.float64)[keep]
    # 2. resample to sfreq (250 Hz)
    if sample["sfreq"] != sfreq:
        x = mne.filter.resample(x, up=sfreq, down=sample["sfreq"], verbose=False)
    # 3. CAR: subtract the mean across channels at each time point
    x = x - x.mean(axis=0)
    # 4. bandpass l_freq-h_freq (1-30 Hz)
    x = mne.filter.filter_data(x, sfreq, opts["l_freq"], opts["h_freq"], method="fir", verbose=False)
    # 5. eye artefacts: regress out the mean of the eye channels (FP1, FP2) when all exist
    upper = [c.upper() for c in chans]
    eyes = [c.upper() for c in opts["eye_channels"]]
    eye = bool(eyes) and all(c in upper for c in eyes)
    if eye:
        ref = x[[upper.index(c) for c in eyes]].mean(axis=0)
        for ch in x:
            beta = 0.0 if ch.std() == 0 or ref.std() == 0 else np.corrcoef(ch, ref)[0, 1]
            ch -= opts["eye_factor"] * beta * ref
    return {**sample, "signal": x.astype(np.float32), "sfreq": sfreq, "channels": chans, "eye_regressed": eye}

class SampleProcessing(PipelineStep):
    section = "p2_sample_processing"

    def output(self, **kwargs):
        df, opts = kwargs["data"], self.opts
        dataset = df.attrs.get("dataset") or self.config["p1_data_formatting"]["dataset"]
        out = resolve(opts["out_dirs"][dataset])
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.npz"): # delete old outputs before starting
            f.unlink()
        rows = []
        with Pool(opts["processes"]) as pool: # workers pull samples lazily from the generator, in order
            for s in pool.imap(partial(process, opts=opts), SAMPLERS[dataset](df)):
                f = out / f"{s['subject_id']}.npz" # one file per subject, e.g. ischemic_5.npz / sub-01.npz
                np.savez(f, signal=s["signal"])
                rows.append({**{k: s[k] for k in META}, "file": str(f)})
        processed = pd.DataFrame(rows, columns=META + ["file"])
        processed.attrs = {"sfreq": opts["sfreq"], "dataset": dataset, "out_dir": str(out)}
        return {**kwargs, "processed": processed}
