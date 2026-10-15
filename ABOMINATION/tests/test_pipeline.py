import pytest
from pipeline import stage_range


def test_stage_range():
    assert stage_range([]) == (0, 7) and stage_range(["-s", "7", "-e", "7"]) == (7, 7)
    for bad in (["-s", "4"], ["-e", "5"], ["-s", "7", "-e", "4"]):
        with pytest.raises(SystemExit):
            stage_range(bad)
