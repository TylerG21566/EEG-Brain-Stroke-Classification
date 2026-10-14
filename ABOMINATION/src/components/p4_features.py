import os, time
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
from components.base import PipelineStep, resolve
from components.features import AsymmetryFeatures, ShapeFeatures, WaveletFeatures, WelchFeatures
from components.p3_epoching import combo_dir, load, save_epochs, windows

EXTRACTORS = {"shape": ShapeFeatures, "welch": WelchFeatures, "asymmetry": AsymmetryFeatures} # config key -> class
RETURNS = {"shape": "features", "welch": "welch_features", "asymmetry": "asymmetry_features"} # config key -> output key

def table(f, names, quantiles): # (n_epochs, n_features) -> percentiles across epochs (NaN when there are no epochs)
    q = np.percentile(f, quantiles, axis=0).T if len(f) else np.full((len(names), len(quantiles)), np.nan)
    return pd.DataFrame(q, index=names, columns=[f"q{p}" for p in quantiles])

def summarise(task, opts, base): # one subject (runs in a worker): load once, then for each combination epochs -> features -> quantiles (+ wavelet PNG)
    row, p3, sid = task["row"], task["opts"], task["row"]["subject_id"]
    print(f"[worker {os.getpid()}] start {sid} ({len(task['combinations'])} combinations)", flush=True)
    t0 = time.perf_counter()
    signal = load(row)
    extractors = [cls(opts) for key, cls in EXTRACTORS.items() if opts[key]["enabled"]] # toggled in config
    wavelet = WaveletFeatures(opts) if opts["wavelet"]["enabled"] else None
    names = [e.names(row["channels"]) for e in extractors] # also fixes AsymmetryFeatures' pair indices
    extractors, names = [e for e, n in zip(extractors, names) if n], [n for n in names if n] # nothing left on (or no BSI pairs) -> no CSV
    out = {}
    for d, o in task["combinations"]:
        t = time.perf_counter()
        starts, x = windows(signal, row["sfreq"], d, o, p3["clip"])
        if task["out"] is not None: # p3 debug: write the epochs
            save_epochs(task["out"], row, d, o, starts, x)
        tables = {e.suffix: table(e.compute(x, row["sfreq"], p3["clip"]), n, opts["quantiles"]) for e, n in zip(extractors, names)}
        if wavelet:
            wavelet.plot(x, row["sfreq"], row["channels"], f"{sid} @ {d:g} s / {o:g}%", combo_dir(base, d, o) / f"{sid}.png")
        counts = " + ".join(f"{len(n)} {e.suffix.lstrip('_') or 'shape'}" for e, n in zip(extractors, names))
        print(f"[worker {os.getpid()}]   {sid} @ {d:g} s / {o:g}%: {len(x)} epochs -> {counts} features in {time.perf_counter() - t:.2f} s", flush=True)
        out[(d, o)] = len(x), tables, time.perf_counter() - t
    print(f"[worker {os.getpid()}] done {sid} in {time.perf_counter() - t0:.2f} s", flush=True)
    return sid, out

class Features(PipelineStep):
    section = "p4_features"

    def output(self, **kwargs):
        opts, processed = self.opts, kwargs["processed"]
        combinations = self.config["p3_epoching"]["combinations"]
        base = resolve(opts["out_dirs"][processed.attrs["dataset"]])
        for d, o in combinations: # delete old outputs before starting
            c = combo_dir(base, d, o)
            c.mkdir(parents=True, exist_ok=True)
            for f in [*c.glob("*.csv"), *c.glob("*.png")]:
                f.unlink()
        rows, secs = {c: {} for c in combinations}, dict.fromkeys(combinations, 0.0) # rows[combo][sid] = {suffix: wide row}
        with Pool(opts["processes"]) as pool: # one task per subject, in order
            for sid, results in pool.imap(partial(summarise, opts=opts, base=base), kwargs["subjects"]):
                for (d, o), (n, tables, dt) in results.items():
                    secs[(d, o)] += dt
                    for suffix, summary in tables.items(): # e.g. <sid>.csv (48 shape), <sid>_welch.csv (208 band), <sid>_bsi.csv (9 asymmetry) × quantiles
                        summary.to_csv(combo_dir(base, d, o) / f"{sid}{suffix}.csv", index_label="feature")
                        if opts["debug"] and not rows[(d, o)]:
                            print(f"\n== p4 first subject {sid}{suffix} @ {d:g} s / {o:g}%: {n} epochs\n{summary}")
                    rows[(d, o)][sid] = {suffix: summary.stack() for suffix, summary in tables.items()} # wide row: (feature, quantile) -> value
        for (d, o), dt in secs.items(): # compute time summed over subjects (they run in parallel, so this exceeds wall time)
            print(f"[p4] {d:g} s / {o:g}%: {dt:.2f} s across {len(rows[(d, o)])} subjects")
        def wide(suffix): # one table per combination: a row per subject, features × len(quantiles) columns
            tables = {}
            for (d, o), r in rows.items():
                df = pd.DataFrame({sid: t[suffix] for sid, t in r.items() if suffix in t}).T
                df.columns = [f"{q}_{name}" for name, q in df.columns] # e.g. q90_F4_kurtosis, q90_F4_alpha_rel
                df.attrs = {"out_dir": str(combo_dir(base, d, o)), "duration": d, "overlap": o}
                tables[(d, o)] = df
            return tables
        return {**kwargs, **{RETURNS[k]: wide(cls.suffix) for k, cls in EXTRACTORS.items() if opts[k]["enabled"]}} # disabled classes: key absent
