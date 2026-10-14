import numpy as np
from components.features.asymmetry import AsymmetryFeatures

OPTS = {"welch": {"bands": {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)},
                  "window": "hamming", "max_nperseg_s": 2.0, "min_nperseg": 64, "slow": ["delta", "theta"], "fast": ["alpha", "beta"]},
        "asymmetry": {"pairs": [("FP1", "FP2"), ("C3", "C4"), ("O1", "O2")]}}
CH = ["fp1", "C3", "FP2", "C4", "O1"]  # O2 missing -> O1/O2 skipped; case-insensitive


def test_names_and_bsi():
    a, rng = AsymmetryFeatures(OPTS), np.random.default_rng(0)
    assert a.names(CH) == ["FP1_FP2_bsi", "C3_C4_bsi", "mean_bsi"]
    t = np.arange(1000) / 250.0
    x = rng.standard_normal((3, len(CH), 1000))
    x[:, 2] = x[:, 0]  # FP1 == FP2 -> symmetric
    x[:, 1], x[:, 3] = np.sin(2 * np.pi * 5 * t), np.sin(2 * np.pi * 20 * t)  # C3 vs C4 at different frequencies -> one-sided
    f = a.compute(x.astype(np.float32), 250.0, 20.0)
    assert f.shape == (3, 3) and np.allclose(f[:, 0], 0) and (f[:, 1] > 0.8).all() and np.allclose(f[:, 2], f[:, :2].mean(-1))


def test_no_pairs_and_empty():
    a = AsymmetryFeatures(OPTS)
    assert a.names(["T3", "T4"]) == [] and a.compute(np.zeros((2, 2, 1000), np.float32), 250.0, 20.0).shape == (2, 0)
    a.names(CH)
    assert a.compute(np.empty((0, len(CH), 1000), np.float32), 250.0, 20.0).shape == (0, 3)
