import numpy as np
from components.p3_epoching import epoch_name
from components.p4_features import Features, describe


def fake_epochs(n):
    for i in range(n):
        yield {"subject_id": f"sub-{i % 2:02d}", "label": "stroke", "subtype": "ischemic", "split": None, "epoch": i,
               "start": 2.0 * i, "duration": 4, "overlap": 50, "sfreq": 250.0, "channels": ["C3", "C4", "O1"],
               "signal": np.arange(3 * 1000, dtype=np.float32).reshape(3, 1000)}


def test_describe():
    e = next(fake_epochs(1))
    lines = describe(e, head=5).splitlines()
    assert epoch_name(e) in lines[0] and "sub-00" in lines[0]
    assert any(l.strip() == "shape (3, 1000)" for l in lines[1:]) and "2999" not in "\n".join(lines)  # only the head is printed


def test_output(cfg, capsys):
    cfg["p4_features"].update(processes=2, head=3)
    out = Features(cfg).output(epochs=fake_epochs(7), extra=1)
    text = capsys.readouterr().out
    pos = [text.find(epoch_name(e)) for e in fake_epochs(7)]
    assert out["n_epochs"] == 7 and out["extra"] == 1
    assert -1 not in pos and pos == sorted(pos) and text.count("shape (3, 1000)") == 7
