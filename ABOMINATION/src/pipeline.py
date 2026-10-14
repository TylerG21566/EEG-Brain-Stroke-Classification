import numpy as np
import pandas as pd
from components.base import PipelineStep
from components.p1_data_formatting import DataFormat

class SanityStep(PipelineStep):
    def output(self, **kwargs):
        print("SANITY CHECKKKKK")
        return kwargs

class Show(PipelineStep):
    def output(self, **kwargs):
        with pd.option_context("display.max_columns", None, "display.width", None), np.printoptions(threshold=6, edgeitems=2):
            for k, v in kwargs.items():
                if isinstance(v, pd.DataFrame):
                    print(f"\n== {k} {v.shape}\n{v.dtypes}\n{v.head()}")
                else:
                    print(f"\n== {k}: {getattr(v, 'shape', repr(v)[:200])}")
        return kwargs

Registered_pipelines: list[PipelineStep] = [
    SanityStep(),
    DataFormat(),
    Show(),
]

def main():
    previous_layer = {}
    for element in Registered_pipelines:
        previous_layer = element.output(**previous_layer)

if __name__ == "__main__":
    main()
