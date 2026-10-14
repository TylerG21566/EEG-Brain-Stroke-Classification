import numpy as np
import pandas as pd
from components.base import PipelineStep, load_config
from components.p1_data_formatting import DataFormat
from components.p2_sample_processing import SampleProcessing
from components.p3_epoching import Epoching
from components.p4_features import Features

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

config = load_config()

# p1/p2 run once; p3/p4 run once per (duration, overlap) in config["p3_epoching"]["combinations"]
Once: list[PipelineStep] = [
    SanityStep(config),
    DataFormat(config),
    SampleProcessing(config),
    Show(config),
]

Per_combination: list[PipelineStep] = [
    Epoching(config),
    Features(config),
]

def main():
    layer = {}
    for step in Once:
        layer = step.output(**layer)
    for duration, overlap in config["p3_epoching"]["combinations"]:
        print(f"\n#### {duration:g} s / {overlap:g}% overlap")
        out = {**layer, "duration": duration, "overlap": overlap}
        for step in Per_combination:
            out = step.output(**out)

if __name__ == "__main__":
    main()
