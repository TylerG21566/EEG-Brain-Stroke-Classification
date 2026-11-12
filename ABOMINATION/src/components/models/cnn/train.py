from collections.abc import Callable
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch import nn
from components.base import Opts
from components.p7_loader import auc

def subject_scores(prob: np.ndarray, subj_index: np.ndarray) -> pd.Series: # mean epoch P(stroke) per subject, indexed by subject index
    return pd.Series(prob).groupby(np.asarray(subj_index)).mean()

@torch.no_grad()
def predict(net: nn.Module, X: np.ndarray, idx: np.ndarray, device: str, batch: int) -> np.ndarray: # epoch sigmoids, eval mode, batched
    net.eval()
    return np.concatenate([torch.sigmoid(net(torch.from_numpy(X[idx[i:i + batch]]).to(device))).float().cpu().numpy() for i in range(0, len(idx), batch)]) if len(idx) else np.empty(0, np.float32)

def fit_score(make_net: Callable[[], nn.Module], X: np.ndarray, y_epoch: np.ndarray, subj_index: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
              opts: Opts, device: str, seed: int) -> tuple[np.ndarray, dict[str, Any]]: # trains on train_idx epochs, early-stops on val subjects' AUC -> test epoch P(stroke), info
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed) # batch shuffling
    net = make_net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=opts["lr"])
    pos = float(y_epoch[train_idx].sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_idx) - pos) / max(pos, 1.0), device=device)) # neg/pos of the training epochs
    yt = torch.from_numpy(np.asarray(y_epoch, np.float32))
    y_val = subject_scores(y_epoch[val_idx], subj_index[val_idx]) # subject labels (every epoch shares its subject's)
    bs, eval_bs = opts["batch_size"], 4 * opts["batch_size"]
    best, best_state, best_pass, bad, passes = -np.inf, None, 0, 0, 0
    for passes in range(1, opts["max_epochs"] + 1):
        net.train()
        order = train_idx[rng.permutation(len(train_idx))]
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            opt.zero_grad()
            loss_fn(net(torch.from_numpy(X[b]).to(device)), yt[b].to(device)).backward()
            opt.step()
            net.constrain() # max-norm etc.
        v = auc(y_val.to_numpy(), subject_scores(predict(net, X, val_idx, device, eval_bs), subj_index[val_idx]).to_numpy())
        if best_state is None or v > best: # NaN (one class in val) never beats a real AUC
            best, best_pass, bad = v, passes, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= opts["patience"]:
                break
    net.load_state_dict(best_state) # restore the best pass
    return predict(net, X, test_idx, device, eval_bs), {"val_auc": float(best), "best_pass": best_pass, "passes": passes}
