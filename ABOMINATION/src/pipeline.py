import argparse, time
import numpy as np
import pandas as pd
from components.base import PipelineStep, load_config
from components.p1_data_formatting import DataFormat
from components.p2_sample_processing import SampleProcessing
from components.p3_epoching import Epoching
from components.p4_features import Features
from components.p7_loader import Loader

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

# stage number -> its steps; the number matches the pN_ file (0 = sanity check). --start / --end pick a range
Registered_pipelines: dict[int, list[PipelineStep]] = {
    0: [SanityStep(config)],
    1: [DataFormat(config)],
    2: [SampleProcessing(config), Show(config)],
    3: [Epoching(config)], # lazy: hands p4 one task per subject
    4: [Features(config)], # each task runs every (duration, overlap) in config["p3_epoching"]["combinations"]
    7: [Loader(config)], # reads p4's CSVs from disk; runs p5 Select (-> p6 Model) inside each CV fold, so 5 and 6 are not stages of their own
}
STARTABLE = {0, 7} # stages a run may start from: 0, or 7 (loads p4's CSVs, skipping preprocessing). Others need earlier outputs in memory

def stage_range(argv=None): # parse and check --start / --end -> (start, end)
    last = max(Registered_pipelines)
    ap = argparse.ArgumentParser(description="Run pipeline stages start..end (inclusive).")
    ap.add_argument("-s", "--start", type=int, default=0, help=f"first stage, one of {sorted(STARTABLE)} (default 0)")
    ap.add_argument("-e", "--end", type=int, default=last, help=f"last stage, 0-{last} (default {last})")
    a = ap.parse_args(argv)
    if a.end not in Registered_pipelines:
        ap.error(f"--end {a.end} is not a stage; stages are {sorted(Registered_pipelines)}")
    if a.start not in STARTABLE:
        ap.error(f"--start {a.start} is not allowed; can start from {sorted(STARTABLE)}")
    if a.start > a.end:
        ap.error(f"--start {a.start} is after --end {a.end}")
    return a.start, a.end

def main(argv=None):
    start, end = stage_range(argv)
    layer = {}
    t0 = time.perf_counter()
    print(f"[pipeline] stages {start}-{end}", flush=True)
    for n in sorted(k for k in Registered_pipelines if start <= k <= end):
        for step in Registered_pipelines[n]:
            t = time.perf_counter()
            layer = step.output(**layer)
            print(f"[{n} {type(step).__name__}] done in {time.perf_counter() - t:.2f} s", flush=True) # p3 is lazy, so ~0 s; its work is timed inside p4
    print(f"[pipeline] total {time.perf_counter() - t0:.2f} s")

if __name__ == "__main__":
    main()
