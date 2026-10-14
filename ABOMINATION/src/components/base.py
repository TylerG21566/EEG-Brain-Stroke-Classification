import copy
from abc import ABC, abstractmethod
from pathlib import Path
from components.p0_pipeline_config import CONFIG

ROOT = Path(__file__).resolve().parents[2] # ABOMINATION/; relative paths in the config resolve against this

def load_config(): # deep copy so callers (tests, steps) can mutate freely
    return copy.deepcopy(CONFIG)

def resolve(p): # absolute paths (e.g. tmp dirs in tests) pass through unchanged
    return ROOT / p

class PipelineStep(ABC):
    section = None # this step's key in the config

    def __init__(self, config=None, name=None):
        self.config = load_config() if config is None else config
        self.opts = self.config.get(self.section, {}) if self.section else {}
        self.name = name

    @abstractmethod
    def output(self, **kwargs):
        pass
