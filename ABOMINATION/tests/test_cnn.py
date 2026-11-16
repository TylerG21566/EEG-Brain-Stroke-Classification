import numpy as np, pandas as pd, pytest, torch
from sklearn.model_selection import StratifiedGroupKFold
import components.p0_pipeline_config as p0
import components.p7_cnn as p7c
from combiner_pipeline import inner_splits
from components.base import combo_dir, groups
from components.models.cnn.data import load_epochs
from components.models.cnn.train import subject_scores
from components.p3_epoching import windows
from components.p7_loader import labels

NAMES = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity"]
SFREQ, SECONDS = 250.0, 20
SIDS = [f"normal_aaaa000{i}-s001" for i in range(6)] + ["normal_aaaa0000-s002"] + [f"stroke_bbbb000{i}-s001" for i in range(6)] # one patient with two sessions


def write_npz(folder, seed=0):  # 2-channel signals: 5 Hz (normal) vs 12 Hz (stroke) sine + noise
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(SFREQ * SECONDS)) / SFREQ
    for s in SIDS:
        f = 12.0 if s.startswith("stroke") else 5.0
        sig = np.stack([np.sin(2 * np.pi * f * t + p) for p in (0, 1)]) + 0.5 * rng.standard_normal((2, len(t)))
        np.savez(folder / f"{s}.npz", signal=sig.astype(np.float32))


class Tiny(torch.nn.Module):  # stand-in for build_net's nets: conv -> |.| -> mean over time -> logit
    def __init__(self, n_chans, n_times):
        super().__init__()
        self.conv, self.fc = torch.nn.Conv1d(n_chans, 4, 15), torch.nn.Linear(4, 1)
    def forward(self, x): return self.fc(self.conv(x).abs().mean(-1)).squeeze(-1)
    def constrain(self): pass


@pytest.fixture
def ccfg(cfg, tmp_path, monkeypatch):  # TUH, 2 folds × 1 repeat, tiny CPU nets, outputs under tmp_path
    write_npz(tmp_path / "npz")
    cfg["p1_data_formatting"]["dataset"] = p0.Dataset.TUH
    cfg["p7_loader"].update(folds=2, repeats=1, seed=0, metrics=NAMES)
    cfg["p7_cnn"].update(nets=["a", "b"], p2_dir=str(tmp_path / "npz"), duration=1, overlap=50, decimate=2, per_subject=10, val_folds=2, batch_size=16,
                         max_epochs=2, patience=5, device="cpu", busy=1.0, out_dirs={p0.Dataset.TUH: str(tmp_path / "cnn")})
    monkeypatch.setattr(p7c, "NET_FACTORY", lambda name, n_chans, n_times, sfreq, opts: Tiny(n_chans, n_times))
    return cfg


def test_load_epochs_decimates_and_spaces(tmp_path):
    write_npz(tmp_path)
    X, si = load_epochs(tmp_path, SIDS, SFREQ, 2, 1, 50, 20.0, 10)
    assert X.dtype == np.float32 and X.shape == (len(SIDS) * 10, 2, 125) and np.bincount(si).tolist() == [10] * len(SIDS)  # 125 Hz × 1 s
    sig = np.load(tmp_path / f"{SIDS[3]}.npz")["signal"]
    _, every = windows(sig[:, ::2], 125.0, 1, 50, 20.0)
    keep = np.linspace(0, len(every) - 1, 10).round().astype(int)
    assert keep[0] == 0 and keep[-1] == len(every) - 1 and np.allclose(X[si == 3], every[keep])  # evenly spaced, first to last
    X_all, si_all = load_epochs(tmp_path, SIDS[:2], SFREQ, 2, 1, 50, 20.0, 10_000)
    assert len(X_all) == 2 * len(every) and (si_all < 2).all()  # fewer than per_subject -> all kept


def test_subject_scores_mean_per_subject():
    s = subject_scores(np.array([0.2, 0.4, 0.9, 1.0]), np.array([1, 1, 0, 0]))
    assert s.index.tolist() == [0, 1] and s.tolist() == pytest.approx([0.95, 0.3])


def test_cnn_loader_writes_p7_layout_in_p7_folds(ccfg, tmp_path, monkeypatch):
    seen = []
    fit = p7c.fit_score
    def spy(make, X, y, si, tr, va, te, *a):  # record each net's train/val/test subjects
        seen.append([set(np.unique(si[e])) for e in (tr, va, te)])
        return fit(make, X, y, si, tr, va, te, *a)
    monkeypatch.setattr(p7c, "fit_score", spy)
    res = p7c.CNNLoader(ccfg).output()
    sids = sorted(SIDS)
    g, y = groups(sids), labels(sids, ["normal"])
    folds = list(StratifiedGroupKFold(2, shuffle=True, random_state=0).split(sids, y, g))  # p7's split (patients repeat -> grouped)
    assert len(seen) == 2 * 2  # 2 folds × 2 nets
    for (tr, va, te), (train, test) in zip(seen[::2], folds):
        pt, pv, pe = ({g[i] for i in s} for s in (tr, va, te))
        assert not (pt & pv or pt & pe or pv & pe) and te == set(test) and tr | va == set(train)
    for n in ("a", "b"):
        dest = combo_dir(tmp_path / "cnn" / n, 1, 50)
        m, p = pd.read_csv(dest / "metrics.csv"), pd.read_csv(dest / "predictions.csv")
        assert list(m.columns) == ["repeat", "fold", "n_train", "n_test", *NAMES] and len(m) == 2
        assert list(p.columns) == ["subject_id", "repeat", "fold", "y_true", "y_score", "y_pred"] and sorted(p["subject_id"]) == sids  # every subject tested once
        assert p["y_true"].tolist() == labels(p["subject_id"], ["normal"]).tolist() and p["y_score"].between(0, 1).all()
        assert (tmp_path / "cnn" / n / "summary.csv").exists() and "pooled_auc" in res["cnn"][n]["summary"]


def test_cnn_loader_is_deterministic(ccfg):
    a = p7c.CNNLoader(ccfg).output()["cnn"]["a"]["predictions"]
    b = p7c.CNNLoader(ccfg).output()["cnn"]["a"]["predictions"]
    pd.testing.assert_frame_equal(a, b)


def test_cnn_loader_with_combiner_writes_every_score(ccfg, tmp_path, monkeypatch):
    seen = []
    fit = p7c.fit_score
    def spy(make, X, y, si, tr, va, te, *a):  # record every train/val/test subject set (outer and inner)
        seen.append([set(np.unique(si[e])) for e in (tr, va, te)])
        return fit(make, X, y, si, tr, va, te, *a)
    monkeypatch.setattr(p7c, "fit_score", spy)
    sids = sorted(SIDS)
    y = labels(sids, ["normal"])
    age = pd.DataFrame({"age": 40 + 25 * y + np.random.default_rng(0).standard_normal(len(sids)), "sex": 1.0}, index=sids)  # fake demographics: age tracks the label
    monkeypatch.setattr(p7c, "side_inputs", lambda cfg: {"demographic": age})
    ccfg["combiner"].update(enabled=True, inner_folds=2, models=["demographic"], debug=True)
    ccfg["p7_cnn"]["nets"] = ["a"]
    res = p7c.CNNLoader(ccfg).output()
    g = groups(sids)
    assert len(seen) == 2 * (1 + 2)  # 2 outer folds × (outer fit + 2 inner fits)
    for tr, va, te in seen:
        pt, pv, pe = ({g[i] for i in s} for s in (tr, va, te))
        assert not (pt & pv or pt & pe or pv & pe)
    for k, (train, test) in enumerate(StratifiedGroupKFold(2, shuffle=True, random_state=0).split(sids, y, g)):  # outer fit, then Ensemble's inner splits of its training subjects
        outer, inner = seen[3 * k], seen[3 * k + 1:3 * k + 3]
        assert outer[2] == set(test) and outer[0] | outer[1] == set(train)
        assert [(tr | va, te) for tr, va, te in inner] == [(set(train[a]), set(train[b])) for a, b in inner_splits(ccfg, np.asarray(sids)[train], y[train])]
    p = pd.read_csv(combo_dir(tmp_path / "cnn" / "a", 1, 50) / "predictions.csv")
    assert {"y_score", "y_score_eeg", "y_score_demographic"} <= set(p.columns) and sorted(p["subject_id"]) == sids
    assert p[["y_score", "y_score_eeg", "y_score_demographic"]].stack().between(0, 1).all() and "pooled_auc" in res["cnn"]["a"]["summary"]


def test_staged_search_sweeps_lr_then_samples_at_best_lr():
    from components.tuning import staged
    t = {"lr_grid": [1e-4, 1e-3, 1e-2], "trials": 3}
    space = {"width": (2, 8, "log_int")}
    seen = []
    def score(s):  # best lr is 1e-2; wider is better
        seen.append(dict(s))
        return {"val_auc": 0.5 + (s["lr"] == 1e-2) * 0.2 + s["width"] / 100}
    best, table = staged(score, {"lr": 1e-3, "width": 4}, t, space, np.random.default_rng(0))
    assert seen[0] == {"lr": 1e-3, "width": 4}  # trial 0 = the defaults
    assert [s["lr"] for s in seen[:3]] == [1e-3, 1e-4, 1e-2] and all(s["width"] == 4 for s in seen[:3])  # stage 1: lr only, default lr not repeated
    assert len(seen) == 6 and all(s["lr"] == 1e-2 and 2 <= s["width"] <= 8 for s in seen[3:])  # stage 2: best lr fixed, width sampled
    assert table["stage"].tolist() == [1, 1, 1, 2, 2, 2] and best == seen[int(table["val_auc"].idxmax())]
    flat = staged(lambda s: {"val_auc": 0.7}, {"lr": 1e-3, "width": 4}, t, space, np.random.default_rng(0))[0]
    assert flat == {"lr": 1e-3, "width": 4}  # ties never replace the defaults
    nan = staged(lambda s: {"val_auc": np.nan if s["lr"] != 1e-3 else 0.6}, {"lr": 1e-3, "width": 4}, t, {}, np.random.default_rng(0))
    assert nan[0]["lr"] == 1e-3 and len(nan[1]) == 3  # NaN never wins; empty space -> stage 1 only


def test_cnn_tuning_is_nested_logged_and_reused_by_combiner(ccfg, tmp_path, monkeypatch):
    seen = []
    fit = p7c.fit_score
    def spy(make, X, y, si, tr, va, te, o, *a):  # every fit: subjects seen, lr and epoch budget used
        seen.append({"sets": [set(np.unique(si[e])) for e in (tr, va, te)], "lr": o["lr"], "max_epochs": o["max_epochs"], "tuning": len(te) == 0})
        return fit(make, X, y, si, tr, va, te, o, *a)
    monkeypatch.setattr(p7c, "fit_score", spy)
    sids = sorted(SIDS)
    y = labels(sids, ["normal"])
    age = pd.DataFrame({"age": 40 + 25 * y + np.random.default_rng(0).standard_normal(len(sids)), "sex": 1.0}, index=sids)
    monkeypatch.setattr(p7c, "side_inputs", lambda cfg: {"demographic": age})
    ccfg["combiner"].update(enabled=True, inner_folds=2, models=["demographic"])
    ccfg["p7_cnn"].update(a={"width": 4}, max_epochs=2)
    ccfg["p7_cnn"]["tuning"] = {"enabled": True, "lr_grid": [1e-3, 1e-2], "trials": 2, "max_epochs": 1, "patience": 1, "seed": 0, "space": {"a": {"width": (2, 8, "log_int")}}}
    res = p7c.CNNLoader(ccfg).output()
    g = groups(sids)
    folds = list(StratifiedGroupKFold(2, shuffle=True, random_state=0).split(sids, y, g))
    tun = pd.read_csv(combo_dir(tmp_path / "cnn" / "a_tuned", 1, 50) / "tuning.csv")
    trials = pd.read_csv(combo_dir(tmp_path / "cnn" / "a_tuned", 1, 50) / "tuning_trials.csv")
    assert len(tun) == 2 and len(trials) == 2 * (2 + 2)  # per fold: 2 lr values + 2 random draws
    assert {"lr", "width", "val_auc", "default_val_auc", "test_auc_eeg", "test_auc"} <= set(tun.columns)
    per_fold = len(seen) // 2
    assert per_fold == (4 + 1 + 2) + (1 + 2)  # a: 4 tuning fits, outer refit, 2 combiner inner refits; then b (untuned): outer + 2 inner
    for k, (train, test) in enumerate(folds):
        fits = seen[k * per_fold:k * per_fold + 7]
        assert not any(f["tuning"] for f in seen[k * per_fold + 7:(k + 1) * per_fold])  # b never tuned
        for f in fits[:4]:  # tuning: shortened, no test epochs, never a test subject
            assert f["tuning"] and f["max_epochs"] == 1 and not (f["sets"][0] | f["sets"][1]) & set(test)
        for f in fits[4:]:  # refits: full budget, the chosen lr
            assert not f["tuning"] and f["max_epochs"] == 2 and f["lr"] == pytest.approx(tun["lr"].iloc[k])
        assert fits[4]["sets"][2] == set(test)
    assert not (tmp_path / "cnn" / "a").exists() and (tmp_path / "cnn" / "a_tuned" / "summary.csv").exists()  # untuned folder untouched
    assert (tmp_path / "cnn" / "b").exists()  # nets without a space run untuned, as before
    assert sorted(pd.read_csv(combo_dir(tmp_path / "cnn" / "a_tuned", 1, 50) / "predictions.csv")["subject_id"]) == sids and "pooled_auc" in res["cnn"]["a"]["summary"]


def test_throttle_pauses_in_proportion_to_work(monkeypatch):
    import time as _time
    from components.models.cnn import train
    slept = []
    monkeypatch.setattr(train.time, "sleep", slept.append)
    t0 = _time.perf_counter() - 0.2  # a batch that took 0.2 s
    train.throttle("cpu", 1.0, t0)
    assert slept == []  # full speed: no pause
    train.throttle("cpu", 0.5, t0)
    train.throttle("cpu", 0.25, t0)
    assert slept[0] == pytest.approx(0.2, abs=0.05) and slept[1] == pytest.approx(0.6, abs=0.1)  # busy 50% -> pause = work; 25% -> 3x work


def test_cnn_resume_reruns_only_missing_folds_and_matches_a_straight_run(ccfg, tmp_path, monkeypatch):
    ccfg["p7_cnn"].update(nets=["a"], a={"width": 4})
    ccfg["p7_cnn"]["tuning"] = {"enabled": True, "lr_grid": [1e-3, 1e-2], "trials": 1, "max_epochs": 1, "patience": 1, "seed": 0, "space": {"a": {"width": (2, 8, "log_int")}}}
    dest = combo_dir(tmp_path / "cnn" / "a_tuned", 1, 50)
    full = p7c.CNNLoader(ccfg).output()["cnn"]["a"]["predictions"]
    logs = {f: pd.read_csv(dest / f) for f in ("tuning.csv", "tuning_trials.csv")}
    for f in ("metrics.csv", "predictions.csv", "tuning.csv"):  # simulate a crash during fold 2: fold 1 finished, fold 2 half-tuned
        t = pd.read_csv(dest / f)
        t[t["fold"] == 0].to_csv(dest / f, index=False)
    trials = logs["tuning_trials.csv"]
    trials[(trials["fold"] == 0) | (trials["trial"] == 0)].to_csv(dest / "tuning_trials.csv", index=False)
    seen = []
    fit = p7c.fit_score
    def spy(make, X, y, si, tr, va, te, *a):
        seen.append(1)
        return fit(make, X, y, si, tr, va, te, *a)
    monkeypatch.setattr(p7c, "fit_score", spy)
    ccfg["p7_cnn"]["resume"] = True
    res = p7c.CNNLoader(ccfg).output()["cnn"]["a"]["predictions"]
    assert len(seen) == 3 + 1  # only fold 2 reruns: 3 tuning fits + refit
    pd.testing.assert_frame_equal(res, full, check_dtype=False)  # same as an uninterrupted run
    for f, t in logs.items():  # logs: fold 1 kept once, fold 2's partial trials replaced, not duplicated
        pd.testing.assert_frame_equal(pd.read_csv(dest / f).drop(columns=["seconds", "tuning_s"], errors="ignore"),
                                      t.drop(columns=["seconds", "tuning_s"], errors="ignore"), check_dtype=False)
