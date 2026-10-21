import numpy as np, pandas as pd, pytest
import components.p0_pipeline_config as p0
from components.base import combo_dir
from components.features import FEATURE_CLASSES as EXTRACTORS
from components.p5_selection import Selection
from components.p7_loader import Loader, labels, split_name

Q = ["q10", "q90"]


def fold(cfg, n=20, seed=0):  # training table: F4_kurtosis tracks the label, everything else is noise
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    names = ["F4_kurtosis", "F4_skew", "F4_entropy", "C3_C4_bsi", "mean_bsi"]
    X = pd.DataFrame(rng.standard_normal((n, len(names) * len(Q))), columns=[f"{q}_{m}" for m in names for q in Q])
    X["q90_F4_kurtosis"] += 3 * y
    fam = {c: "asymmetry" if c.endswith("_bsi") else "shape" for c in X.columns}
    ext = {k: EXTRACTORS[k](cfg["p4_features"]) for k in ("shape", "asymmetry")}
    return X, y, fam, ext


@pytest.mark.parametrize("method", list(p0.Selector))
def test_selection_ranks_signal_first(cfg, method):
    cfg["p5_selection"].update(method=method, k=3, debug=False)
    X, y, fam, ext = fold(cfg)
    X.iloc[0, 1] = np.nan  # a gap is filled, not fatal
    out = Selection(cfg).output(X=X, y=y, families=fam, extractors=ext, fold="t")
    assert out["chosen"][0] == "q90_F4_kurtosis" and len(out["chosen"]) == 3 and list(out["X"].columns) == out["chosen"]


def test_selection_respects_config_toggles(cfg):
    cfg["p5_selection"].update(k=100, debug=False)
    cfg["p4_features"]["shape"]["features"]["kurtosis"] = False
    cfg["p4_features"]["asymmetry"]["features"]["mean"] = False
    X, y, fam, ext = fold(cfg)
    del ext["asymmetry"]  # class switched off -> none of its columns
    chosen = Selection(cfg).output(X=X, y=y, families=fam, extractors=ext, fold="t")["chosen"]
    assert sorted(chosen) == sorted(c for c in X.columns if c.endswith(("_skew", "_entropy")))


def test_selection_passes_test_subjects_through(cfg):  # same columns, gaps filled with the training medians
    cfg["p5_selection"].update(k=3, debug=False)
    X, y, fam, ext = fold(cfg)
    X_test = fold(cfg, n=6, seed=1)[0]
    X_test.loc[X_test.index[0], "q90_F4_kurtosis"] = np.nan
    out = Selection(cfg).output(X=X, y=y, families=fam, extractors=ext, fold="t", X_test=X_test)
    assert list(out["X_test"].columns) == out["chosen"] and len(out["X_test"]) == 6 and not out["X_test"].isna().any().any()
    assert out["X_test"].iloc[0]["q90_F4_kurtosis"] == out["fill"]["q90_F4_kurtosis"] == X["q90_F4_kurtosis"].median()
    assert "X_test" not in Selection(cfg).output(X=X, y=y, families=fam, extractors=ext, fold="t")  # no test data -> no key


@pytest.mark.parametrize("method", list(p0.Selector))
def test_selection_ignores_test_values(cfg, method):  # chosen features, scores and fill come from the training subjects only
    cfg["p5_selection"].update(method=method, k=3, debug=False)
    X, y, fam, ext = fold(cfg)
    X.iloc[2, 3] = np.nan
    a, b = fold(cfg, n=6, seed=1)[0], fold(cfg, n=6, seed=2)[0] * 1e6
    b.iloc[:, ::2] = np.nan
    run = lambda T: Selection(cfg).output(X=X.copy(), y=y, families=fam, extractors=ext, fold="t", X_test=T)
    ra, rb = run(a), run(b)
    assert ra["chosen"] == rb["chosen"] and ra["scores"].equals(rb["scores"]) and ra["fill"].equals(rb["fill"]) and ra["X"].equals(rb["X"])


def test_loader_reads_enabled_csvs_and_runs_cv(cfg, capsys):
    p4 = cfg["p4_features"]
    for k in ("welch", "entropy"):
        p4[k]["enabled"] = False
    cfg["p3_epoching"]["combinations"] = [(4, 0)]
    cfg["p5_selection"].update(k=2, debug=False)
    cfg["p7_loader"].update(folds=2, repeats=2)
    d = combo_dir(p4["out_dirs"][p0.Dataset.ARCHIVE], 4, 0)
    d.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for sid in [f"normal_{i}" for i in range(4)] + [f"ischemic_{i}" for i in range(4)]:
        pd.DataFrame(rng.standard_normal((2, 2)), index=["F4_kurtosis", "F4_skew"], columns=Q).to_csv(d / f"{sid}.csv", index_label="feature")
        pd.DataFrame(rng.standard_normal((1, 2)), index=["mean_bsi"], columns=Q).to_csv(d / f"{sid}_bsi.csv", index_label="feature")
        pd.DataFrame(rng.standard_normal((1, 2)), index=["F4_alpha_rel"], columns=Q).to_csv(d / f"{sid}_welch.csv", index_label="feature")  # disabled class: ignored
    res = Loader(cfg).output(passthrough=1)
    r = res["selection"][(4, 0)]
    assert res["passthrough"] == 1 and len(r["folds"]) == 4 and list(r["labels"]).count(0) == 4
    assert all(set(f["train"]).isdisjoint(f["test"]) and len(f["chosen"]) == 2 for f in r["folds"])
    assert not any("alpha_rel" in c for f in r["folds"] for c in f["chosen"])
    assert "most often chosen" in capsys.readouterr().out


def test_helpers():
    assert split_name("tia_2_bsi", ["_entropy", "_welch", "_bsi", ""]) == ("tia_2", "_bsi")
    assert split_name("tia_2", ["_bsi", ""]) == ("tia_2", "")
    assert labels(["normal_1", "tia_2", "ischemic_10"], ["normal"]).tolist() == [0, 1, 1]
