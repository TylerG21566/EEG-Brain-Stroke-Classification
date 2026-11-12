import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from components.base import Layer, PipelineStep, combo_dir, groups, resolve
from components.models.cnn.data import load_epochs
from components.models.cnn.train import fit_score, subject_scores
from components.p7_loader import METRICS, auc, fold_metrics, labels, pm, summarise

NET_FACTORY: Callable[..., torch.nn.Module] | None = None # tests: stand-in for build_net(name, n_chans, n_times, sfreq, opts)

def device(choice: str) -> str: # "auto" -> MPS when available, else CPU; anything else is used as given
    return ("mps" if torch.backends.mps.is_available() else "cpu") if choice == "auto" else choice

class CNNLoader(PipelineStep): # p7 for CNNs: reads p2's .npz from disk, trains each net on raw epochs in the same subject-level folds as p7 and scores the test subjects
    section = "p7_cnn"

    def output(self, **kwargs: Any) -> Layer: # starts a fresh layer (see pipeline.main); everything comes from disk + config
        opts, p7o, p2 = self.opts, self.config["p7_loader"], self.config["p2_sample_processing"]
        dataset = self.config["p1_data_formatting"]["dataset"]
        names, d, o = p7o.get("metrics", METRICS), opts["duration"], opts["overlap"]
        src = resolve(opts.get("p2_dir") or p2["out_dirs"][dataset])
        out = resolve(opts.get("out_dirs", {}).get(dataset, Path(self.config["paths"]["out"]) / "p7_cnn" / str(dataset)))
        build = NET_FACTORY
        if build is None:
            from components.models.cnn.nets import build_net as build # lazy: torch nets only when this stage runs
        dev = device(opts["device"])
        # 1. every subject's epochs, held in memory once and shared by every fold and net
        sids = sorted(f.stem for f in src.glob("*.npz")) # same ids (and order) as p4's CSVs / p7's table
        t = time.perf_counter()
        X, si = load_epochs(src, sids, p2["sfreq"], opts["decimate"], d, o, opts["clip"], opts["per_subject"])
        keep = np.bincount(si, minlength=len(sids)) > 0
        if not keep.all(): # recordings shorter than one window
            print(f"[p7_cnn] dropped {int((~keep).sum())} subjects with no {d:g} s epoch", flush=True)
            si, sids = np.cumsum(keep)[si] - 1, [s for s, k in zip(sids, keep) if k]
        y, g = labels(sids, p7o["negative"]), groups(sids)
        y_ep = y[si].astype(np.float32)
        sfreq = p2["sfreq"] / opts["decimate"]
        print(f"[p7_cnn] {src}: {len(sids)} subjects, {len(X)} epochs {X.shape[1:]} @ {sfreq:g} Hz ({X.nbytes / 2**30:.2f} GB) in {time.perf_counter() - t:.1f} s, "
              f"labels {dict(Counter(y.tolist()))}, device {dev}", flush=True)
        if len(set(y)) < 2 or min(Counter(dict(zip(g, y.tolist())).values()).values()) < p7o["folds"]: # counts patients, as in p7
            print(f"[p7_cnn] skipped (need both classes with at least {p7o['folds']} subjects each)", flush=True)
            return {**kwargs, "cnn": {}}
        # 2. p7's folds (same kind, seed and split call), each with an inner patient-grouped validation split for early stopping
        rows, preds = {n: [] for n in opts["nets"]}, {n: [] for n in opts["nets"]}
        for r in range(p7o["repeats"]):
            kind = StratifiedGroupKFold if len(set(g)) < len(g) else StratifiedKFold # several sessions per patient -> keep them in one fold
            cv = kind(p7o["folds"], shuffle=True, random_state=p7o["seed"] + r)
            for i, (train, test) in enumerate(cv.split(sids, y, g) if kind is StratifiedGroupKFold else cv.split(sids, y)):
                fit, val = next(StratifiedGroupKFold(opts["val_folds"], shuffle=True, random_state=opts["seed"] + r).split(train, y[train], g[train]))
                fit, val = train[fit], train[val]
                assert not (set(g[fit]) & set(g[val]) or set(g[fit]) & set(g[test]) or set(g[val]) & set(g[test])), "patient in more than one of train/val/test"
                ep = [np.flatnonzero(np.isin(si, s)) for s in (fit, val, test)] # epoch indices of each subject set
                if opts["debug"]:
                    print(f"[p7_cnn] repeat {r + 1} fold {i + 1}: subjects train/val/test {len(fit)}/{len(val)}/{len(test)}, epochs {'/'.join(str(len(e)) for e in ep)}", flush=True)
                for n in opts["nets"]:
                    t = time.perf_counter()
                    make = lambda: build(n, X.shape[1], X.shape[2], sfreq, opts.get(n, {}))
                    prob, info = fit_score(make, X, y_ep, si, *ep, opts, dev, opts["seed"] + r)
                    score = subject_scores(prob, si[ep[2]]).reindex(test).to_numpy() # mean epoch P(stroke) per test subject, in test order
                    pred = (score >= opts["threshold"]).astype(int)
                    m = fold_metrics(y[test], score, pred, names)
                    rows[n].append({"repeat": r, "fold": i, "n_train": len(train), "n_test": len(test), **m})
                    preds[n].append(pd.DataFrame({"subject_id": np.asarray(sids)[test], "repeat": r, "fold": i, "y_true": y[test], "y_score": score, "y_pred": pred}))
                    print(f"[p7_cnn] {n} | repeat {r + 1}/{p7o['repeats']} fold {i + 1}/{p7o['folds']} | val auc {info['val_auc']:.3f} (pass {info['best_pass']}/{info['passes']}) "
                          f"| test auc {auc(y[test], score):.3f} | {time.perf_counter() - t:.0f} s", flush=True)
        # 3. per net: p7's CSV layout + summary
        results, w = {}, max(map(len, names + ["pooled auc"]))
        for n in opts["nets"]:
            metrics, pred = pd.DataFrame(rows[n]), pd.concat(preds[n], ignore_index=True)
            summary = summarise(metrics, pred, names)
            dest = combo_dir(out / n, d, o)
            dest.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(dest / "metrics.csv", index=False)
            pred.to_csv(dest / "predictions.csv", index=False)
            pd.DataFrame([{"duration": d, "overlap": o, **summary}]).to_csv(out / n / "summary.csv", index=False)
            print(f"\n[p7_cnn] {n} | {d:g} s / {o:g}% | test metrics, mean ± SD over {len(metrics)} folds\n" + "\n".join(f"    {m:<{w}}  {pm(summary, m)}" for m in names)
                  + f"\n    {'pooled auc':<{w}}  {summary['pooled_auc']:.3f} (out-of-fold, mean over {p7o['repeats']} repeats) -> {dest}", flush=True)
            results[n] = {"metrics": metrics, "predictions": pred, "summary": summary}
        return {**kwargs, "cnn": results}
