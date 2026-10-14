import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from components.base import load_config, resolve
from components.p0_pipeline_config import View


@pytest.fixture
def cfg(tmp_path):  # fresh config per test: no popups, all paths under tmp_path
    c = load_config()
    c["p1_data_formatting"]["debug_view"] = View.NONE
    c["p1_data_formatting"]["roots"] = {k: str(tmp_path / "in") for k in c["p1_data_formatting"]["roots"]}
    c["p1_data_formatting"]["debug_dir"] = str(tmp_path / "p1_debug")
    c["p2_sample_processing"]["out_dirs"] = {k: str(tmp_path / "out") for k in c["p2_sample_processing"]["out_dirs"]}
    c["p4_features"]["out_dirs"] = {k: str(tmp_path / "p4") for k in c["p4_features"]["out_dirs"]}
    c["p3_epoching"]["out_dirs"] = {k: str(tmp_path / "p3") for k in c["p3_epoching"]["out_dirs"]}
    return c


@pytest.fixture
def real_cfg(cfg, request):  # real data roots, outputs still under tmp_path; skips if data is missing
    root = resolve(load_config()["p1_data_formatting"]["roots"][request.param])
    if not root.exists():
        pytest.skip("real data missing")
    cfg["p1_data_formatting"].update(dataset=request.param, roots={request.param: str(root)})
    return cfg
