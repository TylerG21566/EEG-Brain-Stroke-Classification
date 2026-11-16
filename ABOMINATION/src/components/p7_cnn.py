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
from components.p7_loader import METRICS, auc, fold_metrics, labels, pm, pooled_auc, summarise
from components.tuning import staged
from combiner_components.model import CombinerModel
from combiner_pipeline import extra_scores, inner_splits, side_inputs

NET_FACTORY: Callable[..., torch.nn.Module] | None = None # tests: stand-in for build_net(name, n_chans, n_times, sfreq, opts)

def device(choice: str) -> str: # "auto" -> MPS when available, else CPU; anything else is used as given
    return ("mps" if torch.backends.mps.is_available() else "cpu") if choice == "auto" else choice

def append_csv(path: Path, row: dict[str, Any]) -> None: # one row at a time, so a stopped run keeps what it has done
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)

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
        cc = self.config.get("combiner", {})
        ensemble = cc.get("enabled", False)
        side = side_inputs(self.config) if ensemble else {} # extra models' inputs, loaded once
        models, ids = ["eeg", *side], np.asarray(sids)
        def plan(tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]: # subjects tr -> patient-grouped early-stopping split (fit, val) + epoch indices of fit / val / te
            f, v = next(StratifiedGroupKFold(opts["val_folds"], shuffle=True, random_state=opts["seed"] + r).split(tr, y[tr], g[tr]))
            fit, val = tr[f], tr[v]
            assert not (set(g[fit]) & set(g[val]) or set(g[fit]) & set(g[te]) or set(g[val]) & set(g[te])), "patient in more than one of train/val/test"
            return fit, val, [np.flatnonzero(np.isin(si, s)) for s in (fit, val, te)]
        def run(n: str, ep: list[np.ndarray], te: np.ndarray, net_o: dict[str, Any], train_o: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]: # net n (kwargs net_o) trained with train_o on ep's fit epochs (early-stopped on val) -> mean epoch P(stroke) per te subject, in te order
            prob, info = fit_score(lambda: build(n, X.shape[1], X.shape[2], sfreq, net_o), X, y_ep, si, *ep, train_o, dev, opts["seed"] + r)
            return subject_scores(prob, si[ep[2]]).reindex(te).to_numpy(), info
        tc = opts.get("tuning", {})
        tuned = [n for n in opts["nets"] if tc.get("enabled", False) and n in tc.get("space", {})]
        root = {n: out / (f"{n}_tuned" if n in tuned else n) for n in opts["nets"]} # tuned runs never overwrite the untuned baseline
        dirs = {n: combo_dir(root[n], d, o) for n in opts["nets"]}
        rows, preds, done = {n: [] for n in opts["nets"]}, {n: [] for n in opts["nets"]}, {n: set() for n in opts["nets"]}
        for n in opts["nets"]:
            dirs[n].mkdir(parents=True, exist_ok=True)
            mf, pf = dirs[n] / "metrics.csv", dirs[n] / "predictions.csv"
            if opts.get("resume", False) and mf.exists() and pf.exists(): # p7_cnn.resume: finished folds are reloaded, not rerun (each fold's seeds are fixed, so results match a straight run)
                m0 = pd.read_csv(mf)
                done[n] = {(int(a), int(b)) for a, b in zip(m0["repeat"], m0["fold"])}
                rows[n], preds[n] = m0.to_dict("records"), [pd.read_csv(pf)]
                print(f"[p7_cnn] {n} | resuming: {len(done[n])} finished folds reloaded from {dirs[n]}", flush=True)
            for f in ("tuning.csv", "tuning_trials.csv"): # logs keep finished folds only (none when not resuming); new rows are appended as trials finish
                p = dirs[n] / f
                if p.exists():
                    log0 = pd.read_csv(p)
                    keep = log0[[(int(a), int(b)) in done[n] for a, b in zip(log0["repeat"], log0["fold"])]]
                    keep.to_csv(p, index=False) if len(keep) else p.unlink()
        def tune(n: str, ep: list[np.ndarray], i: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: # staged search on this fold's fit/val subjects only -> (net kwargs, training opts, log row)
            space = tc["space"][n]
            defaults = {"lr": opts["lr"], **{k: opts[n][k] for k in space}} # trial 0 = the fixed settings
            def score(s: dict[str, Any]) -> dict[str, Any]: # shortened training; no test epochs passed in
                t0 = time.perf_counter()
                _, info = fit_score(lambda: build(n, X.shape[1], X.shape[2], sfreq, {**opts.get(n, {}), **{k: s[k] for k in space}}), X, y_ep, si, ep[0], ep[1], ep[2][:0],
                                    {**opts, "lr": s["lr"], "max_epochs": tc["max_epochs"], "patience": tc["patience"]}, dev, opts["seed"] + r)
                return {"val_auc": info["val_auc"], "best_pass": info["best_pass"], "passes": info["passes"], "seconds": round(time.perf_counter() - t0, 1)}
            def log(row: dict[str, Any]) -> None:
                append_csv(dirs[n] / "tuning_trials.csv", {"repeat": r, "fold": i, **row})
                print(f"[p7_cnn tune] {n} | repeat {r + 1} fold {i + 1} | trial {row['trial']} (stage {row['stage']}) | "
                      + ", ".join(f"{k} {row[k]:.3g}" for k in defaults) + f" | val auc {row['val_auc']:.3f} (pass {row['best_pass']}/{row['passes']}) | {row['seconds']:.0f} s", flush=True)
            best, table = staged(score, defaults, tc, space, np.random.default_rng([tc["seed"], r, i]), log)
            row = {**best, "val_auc": float(table["val_auc"].max()), "default_val_auc": float(table["val_auc"].iloc[0]), "tuning_s": round(float(table["seconds"].sum()), 1)}
            return {**opts.get(n, {}), **{k: best[k] for k in space}}, {**opts, "lr": best["lr"]}, row
        # 2. p7's folds (same kind, seed and split call), each with an inner patient-grouped validation split for early stopping
        for r in range(p7o["repeats"]):
            kind = StratifiedGroupKFold if len(set(g)) < len(g) else StratifiedKFold # several sessions per patient -> keep them in one fold
            cv = kind(p7o["folds"], shuffle=True, random_state=p7o["seed"] + r)
            for i, (train, test) in enumerate(cv.split(sids, y, g) if kind is StratifiedGroupKFold else cv.split(sids, y)):
                todo = [n for n in opts["nets"] if (r, i) not in done[n]] # resume: nets that still need this fold
                if not todo:
                    continue
                fit, val, ep = plan(train, test)
                splits = inner_splits(self.config, ids[train], y[train]) if ensemble else [] # Ensemble's inner CV (positions within train)
                inner = [plan(train[a], train[b])[2] for a, b in splits] # each inner split's own early-stopping split
                oof = np.zeros((len(train), len(models))) # out-of-fold scores: column 0 = the net (filled per net), then each extra model
                for a, b in splits:
                    oof[b, 1:] = np.reshape(extra_scores(self.config, side, ids[train[a]], y[train[a]], ids[train[b]]), (len(side), len(b))).T
                s_side = np.reshape(extra_scores(self.config, side, ids[train], y[train], ids[test]), (len(side), len(test))).T # extra models refit on all training subjects
                if opts["debug"]:
                    print(f"[p7_cnn] repeat {r + 1} fold {i + 1}: subjects train/val/test {len(fit)}/{len(val)}/{len(test)}, epochs {'/'.join(str(len(e)) for e in ep)}", flush=True)
                for n in todo:
                    t = time.perf_counter()
                    net_o, train_o, tune_row = tune(n, ep, i) if n in tuned else (opts.get(n, {}), opts, None) # tuned on EEG alone; the combiner below reuses the choice
                    score, info = run(n, ep, test, net_o, train_o) # mean epoch P(stroke) per test subject, in test order
                    eeg = score
                    parts = {}
                    if ensemble: # the net + extra models -> combiner fitted on the inner CV's out-of-fold scores
                        for (a, b), e in zip(splits, inner):
                            oof[b, 0] = run(n, e, train[b], net_o, train_o)[0]
                        comb, s = CombinerModel(cc).fit(oof, y[train]), np.column_stack([score, s_side])
                        score, parts = comb.score(s), {f"y_score_{k}": s[:, j] for j, k in enumerate(models)}
                        if cc["debug"]:
                            print(f"[combiner] {n} | repeat {r + 1} fold {i + 1} | weights " + ", ".join(f"{k} {w:+.2f}" for k, w in zip(models, comb.weights)), flush=True)
                    pred = (score >= (cc["threshold"] if ensemble else opts["threshold"])).astype(int)
                    m = fold_metrics(y[test], score, pred, names)
                    rows[n].append({"repeat": r, "fold": i, "n_train": len(train), "n_test": len(test), **m})
                    preds[n].append(pd.DataFrame({"subject_id": np.asarray(sids)[test], "repeat": r, "fold": i, "y_true": y[test], "y_score": score, "y_pred": pred, **parts}))
                    dirs[n].mkdir(parents=True, exist_ok=True) # results so far, rewritten after every fold
                    pd.DataFrame(rows[n]).to_csv(dirs[n] / "metrics.csv", index=False)
                    pd.concat(preds[n], ignore_index=True).to_csv(dirs[n] / "predictions.csv", index=False)
                    if tune_row is not None: # this fold's chosen settings, their val AUC vs the defaults', and the test AUC they got
                        append_csv(dirs[n] / "tuning.csv", {"repeat": r, "fold": i, **tune_row, "test_auc_eeg": auc(y[test], eeg), "test_auc": m.get("auc", auc(y[test], score))})
                        print(f"[p7_cnn tune] {n} | repeat {r + 1} fold {i + 1} | chosen " + ", ".join(f"{k} {tune_row[k]:.3g}" for k in ("lr", *tc["space"][n]))
                              + f" | val auc {tune_row['val_auc']:.3f} (defaults {tune_row['default_val_auc']:.3f}) | tuning {tune_row['tuning_s'] / 60:.1f} min", flush=True)
                    print(f"[p7_cnn] {n} | repeat {r + 1}/{p7o['repeats']} fold {i + 1}/{p7o['folds']} | val auc {info['val_auc']:.3f} (pass {info['best_pass']}/{info['passes']}) "
                          f"| test auc {auc(y[test], score):.3f}" + (f" (eeg {auc(y[test], parts['y_score_eeg']):.3f})" if ensemble else "") + f" | {time.perf_counter() - t:.0f} s", flush=True)
        # 3. per net: p7's CSV layout + summary
        results, w = {}, max(map(len, names + ["pooled auc"]))
        for n in opts["nets"]:
            metrics, pred = pd.DataFrame(rows[n]), pd.concat(preds[n], ignore_index=True)
            summary = summarise(metrics, pred, names)
            dest = dirs[n]
            dest.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(dest / "metrics.csv", index=False)
            pred.to_csv(dest / "predictions.csv", index=False)
            pd.DataFrame([{"duration": d, "overlap": o, **summary}]).to_csv(root[n] / "summary.csv", index=False)
            print(f"\n[p7_cnn] {n} | {d:g} s / {o:g}% | test metrics, mean ± SD over {len(metrics)} folds\n" + "\n".join(f"    {m:<{w}}  {pm(summary, m)}" for m in names)
                  + f"\n    {'pooled auc':<{w}}  {summary['pooled_auc']:.3f} (out-of-fold, mean over {p7o['repeats']} repeats) -> {dest}"
                  + "".join(f"\n    {k + ' only':<{w}}  {pooled_auc(pred.assign(y_score=pred[f'y_score_{k}'])):.3f} (pooled auc)" for k in models if ensemble)
                  + (f"\n    tuned: val auc {(tl := pd.read_csv(dest / 'tuning.csv'))['val_auc'].mean():.3f} vs defaults {tl['default_val_auc'].mean():.3f} (inner, optimistic); "
                     f"settings per fold -> {dest / 'tuning.csv'}" if n in tuned else ""), flush=True)
            results[n] = {"metrics": metrics, "predictions": pred, "summary": summary}
        return {**kwargs, "cnn": results}
