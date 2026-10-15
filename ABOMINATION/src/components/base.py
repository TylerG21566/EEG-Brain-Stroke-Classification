import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from components.p0_pipeline_config import CONFIG

ROOT = Path(__file__).resolve().parents[2] # ABOMINATION/; relative paths in the config resolve against this

Config = dict[str, Any] # the whole CONFIG
Opts = dict[str, Any] # one section of it (a step's self.opts)
Layer = dict[str, Any] # outputs passed from one step to the next
Combo = tuple[float, float] # (duration s, overlap %)

def load_config() -> Config: # deep copy so callers (tests, steps) can mutate freely
    return copy.deepcopy(CONFIG)

def resolve(p: str | Path) -> Path: # absolute paths (e.g. tmp dirs in tests) pass through unchanged
    return ROOT / p

class PipelineStep(ABC):
    section: str | None = None # this step's key in the config

    def __init__(self, config: Config | None = None, name: str | None = None) -> None:
        self.config = load_config() if config is None else config
        self.opts = self.config.get(self.section, {}) if self.section else {}
        self.name = name

    @abstractmethod
    def output(self, **kwargs: Any) -> Layer:
        pass
