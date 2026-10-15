import pytest
from pipeline import stage_range


def test_stage_range():
    assert stage_range([]) == (0, 7) and stage_range(["-s", "7", "-e", "7"]) == (7, 7)
    for bad in (["-s", "4"], ["-e", "5"], ["-s", "7", "-e", "4"]):
        with pytest.raises(SystemExit):
            stage_range(bad)


def test_entry_stages_start_with_empty_layer(monkeypatch):  # 0 -> 7: stage 7 must not see stage 4's outputs
    import pipeline
    seen = {}
    class Step:
        def __init__(self, n): self.n = n
        def output(self, **kwargs):
            seen[self.n] = set(kwargs)
            return {**kwargs, f"from_{self.n}": 1}
    monkeypatch.setattr(pipeline, "Registered_pipelines", {0: [Step(0)], 4: [Step(4)], 7: [Step(7)]})
    pipeline.main(["-s", "0", "-e", "7"])
    assert seen == {0: set(), 4: {"from_0"}, 7: set()}
