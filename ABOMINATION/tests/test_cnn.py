import numpy as np, pandas as pd, pytest, torch
from sklearn.model_selection import StratifiedGroupKFold
import components.p0_pipeline_config as p0
import components.p7_cnn as p7c
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
                         max_epochs=2, patience=5, device="cpu", out_dirs={p0.Dataset.TUH: str(tmp_path / "cnn")})
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
