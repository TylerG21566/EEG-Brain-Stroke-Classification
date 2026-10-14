import numpy as np
from components.features.welch import WelchFeatures

OPTS = {"welch": {"bands": {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)},
                  "window": "hamming", "max_nperseg_s": 2.0, "min_nperseg": 64,
                  "slow": ["delta", "theta"], "fast": ["alpha", "beta"]}}
CH = ["FP1", "C3", "O1"]


def test_names():
    n = WelchFeatures(OPTS).names(CH)
    assert len(n) == len(CH) * 13 and n[:4] == ["FP1_delta_rel", "FP1_theta_rel", "FP1_alpha_rel", "FP1_beta_rel"]
    assert n[4:13] == ["FP1_delta_theta_ratio", "FP1_delta_alpha_ratio", "FP1_delta_beta_ratio",
                       "FP1_theta_alpha_ratio", "FP1_theta_beta_ratio", "FP1_alpha_beta_ratio",
                        "FP1_dtabr", "FP1_spec_mean", "FP1_slope"] and n[13] == "C3_delta_rel"


def test_alpha_sine():
    w = WelchFeatures(OPTS)
    t = np.arange(1000) / 250.0
    x = np.broadcast_to(np.sin(2 * np.pi * 10 * t), (2, len(CH), 1000)).astype(np.float32)
    f = dict(zip(w.names(CH), w.compute(x, 250.0, 20.0)[0]))
    rel = [f[f"O1_{b}_rel"] for b in OPTS["welch"]["bands"]]
    assert np.argmax(rel) == 2 and rel[2] > 0.95 and f["O1_delta_alpha_ratio"] < 0.01 and 1 / f["O1_delta_alpha_ratio"] > 100
    assert f["O1_dtabr"] < 0.01 and abs(f["O1_spec_mean"] - 10) < 1


def test_shape_and_empty():
    w = WelchFeatures(OPTS)
    x = np.random.default_rng(0).standard_normal((5, len(CH), 1000)).astype(np.float32)
    x[:, 2] = 0  # flat channel -> zeros, not NaN
    f = w.compute(x, 250.0, 20.0)
    assert f.shape == (5, 39) and not np.isnan(f).any() and (f[:, 26:] == 0).all()
    assert w.compute(np.empty((0, len(CH), 1000), np.float32), 250.0, 20.0).shape == (0, 39)


def test_slope_pink_noise():  # 1/f power -> log-log slope ≈ -1; white noise -> ≈ 0
    w, rng = WelchFeatures(OPTS), np.random.default_rng(2)
    white = rng.standard_normal((4, 1, 2500))
    spec = np.fft.rfft(white, axis=-1)
    spec[..., 1:] /= np.sqrt(np.arange(1, spec.shape[-1]))
    pink = np.fft.irfft(spec, n=2500, axis=-1)
    slope = lambda x: w.compute(x.astype(np.float32), 250.0, 20.0)[:, -1].mean()
    assert abs(slope(pink) + 1) < 0.2 and abs(slope(white)) < 0.2
