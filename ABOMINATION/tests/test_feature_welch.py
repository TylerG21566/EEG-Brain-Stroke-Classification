import numpy as np
from components.features.welch import WelchFeatures

OPTS = {"welch": {"bands": {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)},
                  "window": "hamming", "max_nperseg_s": 2.0, "min_nperseg": 64}}
CH = ["FP1", "C3", "O1"]


def test_names():
    n = WelchFeatures(OPTS).names(CH)
    assert len(n) == len(CH) * 10 and n[:4] == ["FP1_delta_rel", "FP1_theta_rel", "FP1_alpha_rel", "FP1_beta_rel"]
    assert n[4:10] == ["FP1_delta_theta_ratio", "FP1_delta_alpha_ratio", "FP1_delta_beta_ratio",
                       "FP1_theta_alpha_ratio", "FP1_theta_beta_ratio", "FP1_alpha_beta_ratio"] and n[10] == "C3_delta_rel"


def test_alpha_sine():
    w = WelchFeatures(OPTS)
    t = np.arange(1000) / 250.0
    x = np.broadcast_to(np.sin(2 * np.pi * 10 * t), (2, len(CH), 1000)).astype(np.float32)
    f = dict(zip(w.names(CH), w.compute(x, 250.0, 20.0)[0]))
    rel = [f[f"O1_{b}_rel"] for b in OPTS["welch"]["bands"]]
    assert np.argmax(rel) == 2 and rel[2] > 0.95 and f["O1_delta_alpha_ratio"] < 0.01 and 1 / f["O1_delta_alpha_ratio"] > 100


def test_shape_and_empty():
    w = WelchFeatures(OPTS)
    x = np.random.default_rng(0).standard_normal((5, len(CH), 1000)).astype(np.float32)
    x[:, 2] = 0  # flat channel -> zeros, not NaN
    f = w.compute(x, 250.0, 20.0)
    assert f.shape == (5, 30) and not np.isnan(f).any() and (f[:, 20:] == 0).all()
    assert w.compute(np.empty((0, len(CH), 1000), np.float32), 250.0, 20.0).shape == (0, 30)
