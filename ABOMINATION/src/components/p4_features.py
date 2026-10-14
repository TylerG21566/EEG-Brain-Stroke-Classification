from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from components.base import PipelineStep, resolve
from components.p3_epoching import combo_dir, subject_epochs

FEATURES = ["skew", "kurtosis", "entropy"] # per channel, so 3 × 16 = 48 per epoch

def entropy(x, bins, clip): # Shannon entropy (bits) of each channel's value histogram over [-clip, clip]
    h = np.stack([np.histogram(ch, bins=bins, range=(-clip, clip))[0] for ch in x]).astype(np.float64)
    p = h / h.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log2(p, where=p > 0, out=np.zeros_like(p)), axis=1)

def epoch_features(x, bins, clip): # (channels, win) -> (channels, 3); flat channels give 0 rather than NaN
    return np.nan_to_num(np.column_stack([skew(x, axis=1), kurtosis(x, axis=1), entropy(x, bins, clip)]))

def summarise(task, opts): # one subject (runs in a worker): epochs -> 48 features each -> quantiles across epochs
    clip = task["opts"]["clip"]
    f = np.array([epoch_features(e["signal"], opts["entropy_bins"], clip) for e in subject_epochs(**task)]) # (n_epochs, ch, 3)
    names = [f"{ch}_{k}" for ch in task["row"]["channels"] for k in FEATURES]
    f = f.reshape(len(f), len(names))
    q = np.percentile(f, opts["quantiles"], axis=0).T if len(f) else np.full((len(names), len(opts["quantiles"])), np.nan)
    return task["row"]["subject_id"], len(f), pd.DataFrame(q, index=names, columns=[f"q{p}" for p in opts["quantiles"]])

class Features(PipelineStep):
    section = "p4_features"

    def output(self, **kwargs):
        opts, processed = self.opts, kwargs["processed"]
        out = combo_dir(resolve(opts["out_dirs"][processed.attrs["dataset"]]), kwargs["duration"], kwargs["overlap"])
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.csv"): # delete old outputs for this combination before starting
            f.unlink()
        rows = {}
        with Pool(opts["processes"]) as pool: # workers pull one subject at a time from p3's generator, in order
            for sid, n, summary in pool.imap(partial(summarise, opts=opts), kwargs["subjects"]):
                summary.to_csv(out / f"{sid}.csv", index_label="feature") # 48 features × quantiles
                if opts["debug"] and not rows:
                    print(f"\n== p4 first subject {sid}: {n} epochs\n{summary}")
                rows[sid] = summary.stack() # wide row: (feature, quantile) -> value
        features = pd.DataFrame(rows).T # one row per subject, 48 × len(quantiles) columns (240)
        features.columns = [f"{q}_{name}" for name, q in features.columns] # e.g. q90_F4_kurtosis
        features.attrs = {"out_dir": str(out), "duration": kwargs["duration"], "overlap": kwargs["overlap"]}
        return {**kwargs, "features": features}
