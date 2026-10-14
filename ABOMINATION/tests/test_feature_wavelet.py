import numpy as np
from components.features.wavelet import WaveletFeatures, mean_power

OPTS = {"wavelet": {"enabled": True, "freqs": list(range(1, 31)), "n_cycles": 6, "decim": 5, "chunk": 2, "dpi": 50}}
CH = [f"C{i}" for i in range(16)]


def test_plot_writes_png(tmp_path):
    x = np.random.default_rng(0).standard_normal((5, 16, 1000)).astype(np.float32)
    WaveletFeatures(OPTS).plot(x, 250.0, CH, "normal_1 @ 4 s / 50%", tmp_path / "w.png")
    assert (tmp_path / "w.png").stat().st_size > 0


def test_plot_no_epochs(tmp_path):
    WaveletFeatures(OPTS).plot(np.empty((0, 16, 1000), np.float32), 250.0, CH, "x", tmp_path / "w.png")
    assert not (tmp_path / "w.png").exists()


def test_mean_power_peaks_at_sine():  # 10 Hz sine peaks at 10 Hz; chunking matches one pass
    t = np.arange(1000) / 250
    x = np.tile(np.sin(2 * np.pi * 10 * t), (5, 2, 1)) + 0.1 * np.random.default_rng(1).standard_normal((5, 2, 1000))
    freqs = np.arange(1, 31.0)
    p = mean_power(x, 250.0, freqs, 6, 5, 2)
    assert p.shape == (2, 30, 200) and (freqs[p.mean(-1).argmax(-1)] == 10).all()
    assert np.allclose(p, mean_power(x, 250.0, freqs, 6, 5, 16))
