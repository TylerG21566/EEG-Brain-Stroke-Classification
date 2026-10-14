import os, time
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from components.base import PipelineStep, resolve
from components.p3_epoching import combo_dir, load, save_epochs, windows

FEATURES = ["skew", "kurtosis", "entropy"] # per channel, so 3 × 16 = 48 per epoch

def entropy(x, bins, clip): # Shannon entropy (bits) of each channel's value histogram over [-clip, clip]; x: (..., win)
    idx = np.clip(((x + clip) * (bins / (2 * clip))).astype(np.int64), 0, bins - 1) # bin index per sample
    rows = np.arange(idx[..., 0].size).reshape(idx.shape[:-1])[..., None] # one histogram per (epoch, channel)
    h = np.bincount((rows * bins + idx).ravel(), minlength=rows.size * bins).reshape(*idx.shape[:-1], bins)
    p = h / x.shape[-1]
    return -np.sum(p * np.log2(p, where=p > 0, out=np.zeros_like(p)), axis=-1)

def epoch_features(x, bins, clip): # (n_epochs, channels, win) -> (n_epochs, channels, 3); flat channels give 0 rather than NaN
    return np.nan_to_num(np.stack([skew(x, axis=-1), kurtosis(x, axis=-1), entropy(x, bins, clip)], axis=-1))

def summarise(task, opts): # one subject (runs in a worker): load once, then for each combination epochs -> features -> quantiles
    row, p3 = task["row"], task["opts"]
    print(f"[worker {os.getpid()}] start {row['subject_id']} ({len(task['combinations'])} combinations)", flush=True)
    t0 = time.perf_counter()
    signal = load(row)
    names = [f"{ch}_{k}" for ch in row["channels"] for k in FEATURES]
    out = {}
    for duration, overlap in task["combinations"]:
        t = time.perf_counter()
        starts, x = windows(signal, row["sfreq"], duration, overlap, p3["clip"])
        if task["out"] is not None: # p3 debug: write the epochs
            save_epochs(task["out"], row, duration, overlap, starts, x)
        f = epoch_features(x, opts["entropy_bins"], p3["clip"]).reshape(len(x), len(names))
        q = np.percentile(f, opts["quantiles"], axis=0).T if len(f) else np.full((len(names), len(opts["quantiles"])), np.nan)
        print(f"[worker {os.getpid()}]   {row['subject_id']} @ {duration:g} s / {overlap:g}%: {len(x)} epochs -> {len(names)} features in {time.perf_counter() - t:.2f} s", flush=True)
        out[(duration, overlap)] = len(f), pd.DataFrame(q, index=names, columns=[f"q{p}" for p in opts["quantiles"]]), time.perf_counter() - t
    print(f"[worker {os.getpid()}] done {row['subject_id']} in {time.perf_counter() - t0:.2f} s", flush=True)
    return row["subject_id"], out

class Features(PipelineStep):
    section = "p4_features"

    def output(self, **kwargs):
        opts, processed = self.opts, kwargs["processed"]
        combinations = self.config["p3_epoching"]["combinations"]
        base = resolve(opts["out_dirs"][processed.attrs["dataset"]])
        for d, o in combinations: # delete old outputs before starting
            c = combo_dir(base, d, o)
            c.mkdir(parents=True, exist_ok=True)
            for f in c.glob("*.csv"):
                f.unlink()
        rows, secs = {c: {} for c in combinations}, dict.fromkeys(combinations, 0.0)
        with Pool(opts["processes"]) as pool: # one task per subject, in order
            for sid, results in pool.imap(partial(summarise, opts=opts), kwargs["subjects"]):
                for (d, o), (n, summary, dt) in results.items():
                    secs[(d, o)] += dt
                    summary.to_csv(combo_dir(base, d, o) / f"{sid}.csv", index_label="feature") # 48 features × quantiles
                    if opts["debug"] and not rows[(d, o)]:
                        print(f"\n== p4 first subject {sid} @ {d:g} s / {o:g}%: {n} epochs\n{summary}")
                    rows[(d, o)][sid] = summary.stack() # wide row: (feature, quantile) -> value
        for (d, o), dt in secs.items(): # compute time summed over subjects (they run in parallel, so this exceeds wall time)
            print(f"[p4] {d:g} s / {o:g}%: {dt:.2f} s across {len(rows[(d, o)])} subjects")
        features = {}
        for (d, o), r in rows.items(): # one table per combination: a row per subject, 48 × len(quantiles) columns (240)
            df = pd.DataFrame(r).T
            df.columns = [f"{q}_{name}" for name, q in df.columns] # e.g. q90_F4_kurtosis
            df.attrs = {"out_dir": str(combo_dir(base, d, o)), "duration": d, "overlap": o}
            features[(d, o)] = df
        return {**kwargs, "features": features}
