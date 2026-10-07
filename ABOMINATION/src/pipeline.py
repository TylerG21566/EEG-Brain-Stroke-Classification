from abc import ABC, abstractmethod

class PipelineStep(ABC):
    def __init__(self, name):
        pass
    
    @abstractmethod
    def output(self, pipeline_input):
        pass

class SanityStep(PipelineStep):
    def output(self, pipeline_input):
        print("SANITY CHECKKKKK")

Registered_pipelines: list[PipelineStep] = [
    SanityStep(),
]

def main():
    prev_output = None
    for element in Registered_pipelines:
        prev_output = element.output(prev_output)