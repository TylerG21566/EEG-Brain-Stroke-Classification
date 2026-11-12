import pytest, torch
from components.models.cnn import NETS, build_net

SHAPES = [(19, 375, 125.0), (19, 750, 250.0)] # 3 s epochs at 125 / 250 Hz
BALLPARK = {"eegnet": (1_000, 3_000), "shallow": (10_000, 100_000)}


def n_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


@pytest.mark.parametrize("name", NETS)
@pytest.mark.parametrize("c,t,sf", SHAPES)
def test_shape_and_params(name, c, t, sf):
    net = build_net(name, c, t, sf, {})
    out = net(torch.randn(4, c, t))
    n = n_params(net)
    print(f"{name} {c}x{t}@{sf:g}Hz: {n} params")
    assert out.shape == (4,) and out.dtype == torch.float32 and torch.isfinite(out).all()
    assert BALLPARK[name][0] <= n <= BALLPARK[name][1]


@pytest.mark.parametrize("name", NETS)
def test_gradients_flow(name):
    net = build_net(name, 19, 375, 125.0, {})
    loss = torch.nn.functional.binary_cross_entropy_with_logits(net(torch.randn(8, 19, 375)), torch.tensor([0., 1.] * 4))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in net.parameters())


def test_eegnet_max_norm():
    net = build_net("eegnet", 19, 375, 125.0, {})
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(10) # blow the norms up
    net.constrain()
    norms = lambda w: w.flatten(1).norm(dim=1)
    assert (norms(net.spatial.weight) <= 1.0 + 1e-5).all() and (norms(net.dense.weight) <= 0.25 + 1e-5).all()


def test_shallow_constrain_noop():
    net = build_net("shallow", 19, 375, 125.0, {})
    before = [p.clone() for p in net.parameters()]
    net.constrain()
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters()))


def test_kwargs_and_unknown_name():
    assert build_net("eegnet", 19, 375, 125.0, {"F1": 4, "D": 2, "F2": 8})(torch.randn(2, 19, 375)).shape == (2,)
    with pytest.raises(ValueError, match="available"):
        build_net("deep", 19, 375, 125.0, {})


@pytest.mark.parametrize("name", NETS)
def test_deterministic(name):
    x = torch.randn(3, 19, 375)
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        outs.append(build_net(name, 19, 375, 125.0, {}).eval()(x))
    assert torch.equal(*outs)
