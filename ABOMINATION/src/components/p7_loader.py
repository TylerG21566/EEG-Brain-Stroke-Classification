from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from components.base import Combo, Layer, PipelineStep, combo_dir, resolve
from components.features import FEATURE_CLASSES as EXTRACTORS # no p3/p4 imports: p7 only shares the config and the CSV layout on disk
from components.p5_selection import Selection
from components.p6_model import Model

MINI: list[type[PipelineStep]] = [Selection, Model] # per-fold mini-pipeline: fitted on the training subjects; p5 only applies its choice + training medians to X_test, p6 scores it
METRICS = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity"] # fallback when the config has no p7_loader["metrics"]

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

def auc(y_true: np.ndarray, y_score: np.ndarray) -> float: # NaN when only one class is present
    return float(roc_auc_score(y_true, y_score)) if len(set(np.asarray(y_true).tolist())) == 2 else np.nan

def fold_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict[str, float]: # stroke = 1; a recall with no subjects of its class -> NaN
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    two = len(set(y_true.tolist())) == 2
    every = {
        "auc": lambda: auc(y_true, y_score),
        "accuracy": lambda: accuracy_score(y_true, y_pred),
        "balanced_accuracy": lambda: balanced_accuracy_score(y_true, y_pred) if two else np.nan,
        "sensitivity": lambda: recall_score(y_true, y_pred, pos_label=1, zero_division=np.nan) if (y_true == 1).any() else np.nan,
        "specificity": lambda: recall_score(y_true, y_pred, pos_label=0, zero_division=np.nan) if (y_true == 0).any() else np.nan,
    }
    return {m: float(every[m]()) for m in names}

def pooled_auc(pred: pd.DataFrame) -> float: # out-of-fold AUC per repeat (every subject scored once), averaged over repeats
    return float(np.nanmean([auc(g["y_true"], g["y_score"]) for _, g in pred.groupby("repeat")]))

def summarise(metrics: pd.DataFrame, pred: pd.DataFrame, names: list[str]) -> dict[str, float]: # mean / SD over every fold and repeat + pooled AUC
    return {**{f"{m}_{s}": float(getattr(metrics[m], s)()) for m in names for s in ("mean", "std")}, "pooled_auc": pooled_auc(pred), "folds": len(metrics)}

def pm(s: dict[str, float], m: str) -> str: # "0.712 ± 0.081"
    return f"{s[f'{m}_mean']:.3f} ± {s[f'{m}_std']:.3f}"

class Loader(PipelineStep): # p7: reads p4's CSVs from disk (works without stages 1-4), then per (duration, overlap) runs MINI in subject-level CV and scores the test folds
    section = "p7_loader"

    def output(self, **kwargs: Any) -> Layer: # starts a fresh layer (see pipeline.main), so kwargs is empty; everything comes from disk + config
        opts, p4 = self.opts, self.config["p4_features"]
        dataset = self.config["p1_data_formatting"]["dataset"]
        names = opts.get("metrics", METRICS)
        keys = [k for k in EXTRACTORS if p4[k]["enabled"]]
        extractors = {k: EXTRACTORS[k](p4) for k in keys} # their keep() decides which features p5 may use
        base = resolve(p4["out_dirs"][dataset])
        out = resolve(opts.get("out_dirs", {}).get(dataset, Path(self.config["paths"]["out"]) / "p7" / str(dataset))) # metrics / predictions CSVs
        results: dict[Combo, dict[str, Any]] = {}
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
            folds, rows, preds = [], [], []
            for r in range(opts["repeats"]):
                cv = StratifiedKFold(opts["folds"], shuffle=True, random_state=opts["seed"] + r)
                for i, (train, test) in enumerate(cv.split(X, y)):
                    layer = {"X": X.iloc[train], "y": y[train], "X_test": X.iloc[test], "y_test": y[test], "families": families, "extractors": extractors,
                             "fold": f"{d:g} s / {o:g}% | repeat {r + 1}/{opts['repeats']} | fold {i + 1}/{opts['folds']}"}
                    for step in MINI: # p5 -> p6: fitted on the training subjects; test labels only come back out for scoring
                        layer = step(self.config).output(**layer)
                    y_test, y_score, y_pred = (np.asarray(layer[k]) for k in ("y_test", "y_score", "y_pred"))
                    m = fold_metrics(y_test, y_score, y_pred, names)
                    folds.append({"repeat": r, "fold": i, "train": list(X.index[train]), "test": list(X.index[test]), **{k: layer[k] for k in ("chosen", "scores")},
                                  "y_test": y_test, "y_score": y_score, "y_pred": y_pred, "metrics": m})
                    rows.append({"repeat": r, "fold": i, "n_train": len(train), "n_test": len(test), **m})
                    preds.append(pd.DataFrame({"subject_id": X.index[test], "repeat": r, "fold": i, "y_true": y_test, "y_score": y_score, "y_pred": y_pred}))
            metrics, pred = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
            summary = summarise(metrics, pred, names)
            dest = combo_dir(out, d, o)
            dest.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(dest / "metrics.csv", index=False)
            pred.to_csv(dest / "predictions.csv", index=False)
            freq = pd.Series(Counter(c for f in folds for c in f["chosen"])).sort_values(ascending=False) / len(folds) # share of folds that chose each feature
            print(f"\n[p7] {d:g} s / {o:g}% | most often chosen over {len(folds)} folds\n" + "\n".join(f"    {c:<{max(map(len, freq.index[:10]))}}  {v:>4.0%}" for c, v in freq.head(10).items()), flush=True)
            w = max(map(len, names + ["pooled auc"]))
            print(f"[p7] {d:g} s / {o:g}% | test metrics, mean ± SD over {len(folds)} folds\n" + "\n".join(f"    {m:<{w}}  {pm(summary, m)}" for m in names)
                  + f"\n    {'pooled auc':<{w}}  {summary['pooled_auc']:.3f} (out-of-fold, mean over {opts['repeats']} repeats) -> {dest}", flush=True)
            results[(d, o)] = {"folds": folds, "frequency": freq, "subjects": list(X.index), "labels": y, "metrics": metrics, "predictions": pred, "summary": summary}
        table = pd.DataFrame([{"duration": d, "overlap": o, **r["summary"]} for (d, o), r in results.items()])
        if results:
            table = table.sort_values("pooled_auc", ascending=False, ignore_index=True)
            out.mkdir(parents=True, exist_ok=True)
            table.to_csv(out / "summary.csv", index=False)
            shown = pd.DataFrame({"combo": [f"{d:g} s / {o:g}%" for d, o in zip(table["duration"], table["overlap"])], **{m: [pm(s, m) for s in table.to_dict("records")] for m in names},
                                  "pooled_auc": table["pooled_auc"].map("{:.3f}".format)})
            print(f"\n[p7] {dataset} | combinations by pooled AUC (mean ± SD over folds) -> {out / 'summary.csv'}\n" + shown.to_string(index=False), flush=True)
        return {**kwargs, "selection": results, "summary": table}
