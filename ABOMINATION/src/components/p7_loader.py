from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from components.base import Layer, PipelineStep, resolve
from components.p3_epoching import combo_dir
from components.p4_features import EXTRACTORS
from components.p5_selection import Selection

MINI: list[type[PipelineStep]] = [Selection] # per-fold mini-pipeline, fed the training subjects only; p6 (model) goes here next

def split_name(stem: str, suffixes: list[str]) -> tuple[str, str]: # "tia_2_bsi" -> ("tia_2", "_bsi"); no known suffix -> shape CSV ("")
    return next(((stem[:-len(s)], s) for s in suffixes if s and stem.endswith(s)), (stem, ""))

def load_combo(folder: Path, keys: list[str]) -> tuple[pd.DataFrame, dict[str, str], list[str]]: # one (duration, overlap) folder -> (subjects × features table, column -> feature class key); only the enabled classes are read
    suffixes = sorted({cls.suffix for cls in EXTRACTORS.values()}, key=len, reverse=True)
    wanted = {EXTRACTORS[k].suffix: k for k in keys}
    rows, families = {}, {}
    for f in sorted(folder.glob("*.csv")):
        sid, suffix = split_name(f.stem, suffixes)
        if suffix not in wanted:
            continue
        t = pd.read_csv(f, index_col="feature").stack() # (feature, quantile) -> value
        cols = [f"{q}_{name}" for name, q in t.index] # e.g. q90_F4_kurtosis, as in p4
        rows.setdefault(sid, {}).update(zip(cols, t.to_numpy()))
        families.update(dict.fromkeys(cols, wanted[suffix]))
    X = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    missing = sorted(set(wanted.values()) - set(families.values()))
    return X.dropna(axis=0, how="all"), families, missing

def labels(sids: list[str] | pd.Index, negative: list[str]) -> np.ndarray: # subtype = subject_id before the last "_" (e.g. ischemic_5 -> ischemic); negative subtypes -> 0, others -> 1
    return np.array([0 if s.rsplit("_", 1)[0] in negative else 1 for s in sids])

class Loader(PipelineStep): # p7: reads p4's CSVs from disk (works without stages 1-4), then per (duration, overlap) runs MINI in subject-level CV
    section = "p7_loader"

    def output(self, **kwargs: Any) -> Layer: # earlier stages' outputs are not needed; they are passed through untouched
        opts, p4 = self.opts, self.config["p4_features"]
        keys = [k for k in EXTRACTORS if p4[k]["enabled"]]
        extractors = {k: EXTRACTORS[k](p4) for k in keys} # their keep() decides which features p5 may use
        base = resolve(p4["out_dirs"][self.config["p1_data_formatting"]["dataset"]])
        results = {}
        for d, o in self.config["p3_epoching"]["combinations"]:
            folder = combo_dir(base, d, o)
            X, families, missing = load_combo(folder, keys)
            if missing:
                print(f"[p7] {d:g} s / {o:g}%: no CSVs for {missing} in {folder}; rerun p4 with them enabled", flush=True)
            y = labels(X.index, opts["negative"])
            if opts["debug"]:
                print(f"[p7] {d:g} s / {o:g}%: {X.shape[0]} subjects × {X.shape[1]} features, labels {dict(Counter(y.tolist()))}", flush=True)
            if X.empty or len(set(y)) < 2 or min(Counter(y.tolist()).values()) < opts["folds"]:
                print(f"[p7] {d:g} s / {o:g}%: skipped (need both classes with at least {opts['folds']} subjects each)", flush=True)
                continue
            folds = []
            for r in range(opts["repeats"]):
                cv = StratifiedKFold(opts["folds"], shuffle=True, random_state=opts["seed"] + r)
                for i, (train, test) in enumerate(cv.split(X, y)):
                    layer = {"X": X.iloc[train], "y": y[train], "families": families, "extractors": extractors,
                             "fold": f"{d:g} s / {o:g}% | repeat {r + 1}/{opts['repeats']} | fold {i + 1}/{opts['folds']}"}
                    for step in MINI: # p5 -> (p6), training subjects only
                        layer = step(self.config).output(**layer)
                    folds.append({"repeat": r, "fold": i, "train": list(X.index[train]), "test": list(X.index[test]), **{k: layer[k] for k in ("chosen", "scores")}})
            freq = pd.Series(Counter(c for f in folds for c in f["chosen"])).sort_values(ascending=False) / len(folds) # share of folds that chose each feature
            print(f"\n[p7] {d:g} s / {o:g}% | most often chosen over {len(folds)} folds\n" + "\n".join(f"    {c:<{max(map(len, freq.index[:10]))}}  {v:>4.0%}" for c, v in freq.head(10).items()), flush=True)
            results[(d, o)] = {"folds": folds, "frequency": freq, "subjects": list(X.index), "labels": y}
        return {**kwargs, "selection": results}
