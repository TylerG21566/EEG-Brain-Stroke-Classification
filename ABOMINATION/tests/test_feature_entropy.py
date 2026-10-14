import numpy as np
from components.features.entropy import EntropyFeatures, approx, fuzzy, sample

OPTS = {"entropy": {"features": {"fuzzy": True, "sample": True, "approx": True}, "m": 2, "r": 0.2, "n": 2, "decim": 1, "chunk": 3}}


def naive(x, m, r, n):  # textbook loops: (fuzzy, sample, approx) for one series
    N = len(x)
    def vecs(k, count, base): return [x[i:i + k] - (x[i:i + k].mean() if base else 0) for i in range(count)]
    def cheb(a, b): return np.abs(a - b).max()
    phi = [np.mean([np.exp(-np.log(2) * (cheb(v[i], v[j]) / r) ** n) for i in range(len(v)) for j in range(len(v)) if i != j])
           for v in (vecs(m, N - m, True), vecs(m + 1, N - m, True))]
    cnt = [sum(cheb(v[i], v[j]) <= r for i in range(len(v)) for j in range(len(v)) if i != j) for v in (vecs(m, N - m, False), vecs(m + 1, N - m, False))]
    ap = [np.mean([np.log(np.mean([cheb(a, b) <= r for b in v])) for a in v]) for v in (vecs(m, N - m + 1, False), vecs(m + 1, N - m, False))]
    return np.log(phi[0]) - np.log(phi[1]), -np.log(cnt[1] / cnt[0]), ap[0] - ap[1]


def test_matches_naive():
    x = np.random.default_rng(0).standard_normal((2, 150)).astype(np.float32)
    r = 0.2 * x.std(-1)
    got = np.stack([fuzzy(x, 2, r, 2), sample(x, 2, r), approx(x, 2, r)], axis=-1)
    assert np.allclose(got, [naive(s.astype(np.float64), 2, rs, 2) for s, rs in zip(x, r)], rtol=1e-4)


def test_noise_beats_sine_and_shapes():
    e, rng = EntropyFeatures(OPTS), np.random.default_rng(1)
    t = np.arange(300) / 62.5
    x = np.stack([np.sin(2 * np.pi * 5 * t)[None].repeat(2, 0), rng.standard_normal((2, 300)), np.zeros((2, 300))], axis=1).astype(np.float32)
    f = e.compute(x, 62.5, 20.0).reshape(2, 3, 3)  # (epochs, channels [sine, noise, flat], [fuzzy, sample, approx])
    assert e.names(["A", "B", "C"])[:3] == ["A_fuzzyen", "A_sampen", "A_apen"] and not np.isnan(f).any()
    assert (f[:, 1] > f[:, 0]).all() and (f[:, 2] == 0).all()  # irregular > regular; flat -> 0
    assert e.compute(np.empty((0, 3, 300), np.float32), 62.5, 20.0).shape == (0, 9)


def test_toggle():
    e = EntropyFeatures({"entropy": {**OPTS["entropy"], "features": {"fuzzy": True, "sample": False, "approx": False}}})
    assert e.names(["A"]) == ["A_fuzzyen"] and e.compute(np.random.default_rng(2).standard_normal((1, 2, 100)).astype(np.float32), 62.5, 20.0).shape == (1, 2)
