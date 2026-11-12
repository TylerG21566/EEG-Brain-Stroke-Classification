import torch
from torch import nn

# Raw-epoch CNNs: (batch, n_chans, n_times) float32 -> one logit per epoch, shape (batch,).
# Kernel / pool lengths are given in seconds and turned into samples with sfreq, so the nets adapt to 125 or 250 Hz.
# Max-norm (EEGNet) is a post-step projection: the trainer calls net.constrain() after every optimiser.step(); no-op elsewhere.

def samples(s: float, sfreq: float) -> int: # seconds -> samples, rounding half up (Python's round() is banker's), at least 1
    return max(1, int(s * sfreq + 0.5))

def same_pad(k: int) -> nn.ZeroPad2d: # Keras 'same' along time for stride 1: extra sample on the right for even k (avoids torch's even-kernel warning)
    return nn.ZeroPad2d(((k - 1) // 2, k - 1 - (k - 1) // 2, 0, 0))

@torch.no_grad()
def max_norm_(w: torch.Tensor, c: float) -> None: # clip the L2 norm of each output unit's weights to c, in place (Keras MaxNorm)
    w.copy_(torch.renorm(w, p=2, dim=0, maxnorm=c))

class EEGNet(nn.Module): # EEGNet-F1,D (Lawhern et al. 2018, Table 2); defaults = EEGNet-8,2 with the cross-subject dropout
    def __init__(self, n_chans: int, n_times: int, sfreq: float, F1: int = 8, D: int = 2, F2: int = 16, kernel_s: float = 0.5,
                 sep_kernel_s: float = 0.125, pool1: int = 4, pool2: int = 8, dropout: float = 0.25,
                 spatial_max_norm: float = 1.0, dense_max_norm: float = 0.25) -> None:
        super().__init__()
        k1, k2 = samples(kernel_s, sfreq), samples(sep_kernel_s, sfreq) # 64 / 16 at 128 Hz; sep kernel spans 0.5 s at the pooled rate sfreq / 4
        self.spatial_max_norm, self.dense_max_norm = spatial_max_norm, dense_max_norm
        self.features = nn.Sequential(
            same_pad(k1), nn.Conv2d(1, F1, (1, k1), bias=False), nn.BatchNorm2d(F1), # temporal (band-pass) filters
            nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(), # D spatial filters per band
            nn.AvgPool2d((1, pool1)), nn.Dropout(dropout),
            same_pad(k2), nn.Conv2d(F1 * D, F1 * D, (1, k2), groups=F1 * D, bias=False), # separable = depthwise temporal ...
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(), # ... + pointwise mix
            nn.AvgPool2d((1, pool2)), nn.Dropout(dropout), nn.Flatten())
        self.spatial = self.features[3]
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 1, n_chans, n_times)).shape[1] # F2 * (n_times // pool1 // pool2)
        self.dense = nn.Linear(n_flat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(self.features(x.unsqueeze(1))).squeeze(1)

    def constrain(self) -> None: # call after each optimiser step
        max_norm_(self.spatial.weight, self.spatial_max_norm)
        max_norm_(self.dense.weight, self.dense_max_norm)

class Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x

class SafeLog(nn.Module): # log, clamped so empty power can't give -inf
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))

class ShallowConvNet(nn.Module): # Schirrmeister et al. 2017 (Fig. 2): learnable FBCSP, temporal -> spatial -> square -> mean pool -> log
    def __init__(self, n_chans: int, n_times: int, sfreq: float, n_filters: int = 40, kernel_s: float = 0.1,
                 pool_s: float = 0.3, pool_stride_s: float = 0.06, dropout: float = 0.5) -> None:
        super().__init__()
        k, p, st = samples(kernel_s, sfreq), samples(pool_s, sfreq), samples(pool_stride_s, sfreq) # 25 / 75 / 15 at 250 Hz
        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, k)), # temporal filters (linear, with bias)
            nn.Conv2d(n_filters, n_filters, (n_chans, 1), bias=False), nn.BatchNorm2d(n_filters), # spatial filters across all channels
            Square(), nn.AvgPool2d((1, p), stride=(1, st)), SafeLog(), nn.Dropout(dropout), nn.Flatten())
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 1, n_chans, n_times)).shape[1]
        self.dense = nn.Linear(n_flat, 1) # = the paper's conv over all pooled steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(self.features(x.unsqueeze(1))).squeeze(1)

    def constrain(self) -> None: # no weight constraints in the paper
        pass

NETS: dict[str, type[nn.Module]] = {"eegnet": EEGNet, "shallow": ShallowConvNet} # net name in the config -> class

def build_net(name: str, n_chans: int, n_times: int, sfreq: float, opts: dict) -> nn.Module: # fresh, untrained net; opts = constructor kwargs
    if name not in NETS:
        raise ValueError(f"unknown net {name!r}; available: {sorted(NETS)}")
    return NETS[name](n_chans, n_times, sfreq, **opts)
