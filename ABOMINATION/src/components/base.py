from abc import ABC, abstractmethod

class PipelineStep(ABC):
    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def output(self, **kwargs):
        pass
