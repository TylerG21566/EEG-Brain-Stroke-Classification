import time
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

Registered_pipelines: list[PipelineStep] = [
    SanityStep(config),
    DataFormat(config),
    SampleProcessing(config),
    Show(config),
    Epoching(config), # lazy: hands p4 one task per subject
    Features(config), # each task runs every (duration, overlap) in config["p3_epoching"]["combinations"]
]

def main():
    layer = {}
    start = time.perf_counter()
    for step in Registered_pipelines:
        t = time.perf_counter()
        layer = step.output(**layer)
        print(f"[{type(step).__name__}] done in {time.perf_counter() - t:.2f} s", flush=True) # p3 is lazy, so ~0 s; its work is timed inside p4
    print(f"[pipeline] total {time.perf_counter() - start:.2f} s")

if __name__ == "__main__":
    main()
