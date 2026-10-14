from functools import partial
from multiprocessing import Pool
import pandas as pd
from components.base import PipelineStep
from components.p3_epoching import epoch_name

def describe(e, head): # placeholder feature: a readable summary of one epoch (runs in a worker)
    x = e["signal"]
    df = pd.DataFrame(x[:, :head], index=e["channels"], columns=range(min(head, x.shape[1])))
    return (f"{epoch_name(e)}  (sample {e['subject_id']}, start {e['start']:.2f} s, duration {e['duration']:g} s, overlap {e['overlap']:g}%)\n"
            f"shape {x.shape}\n{df.to_string(float_format='{:.3f}'.format)}\n")

class Features(PipelineStep):
    section = "p4_features"

    def output(self, **kwargs):
        opts, n = self.opts, 0
        with Pool(opts["processes"]) as pool: # pulls epochs lazily from p3's generator; prints stay in order
            for text in pool.imap(partial(describe, head=opts["head"]), kwargs["epochs"]):
                print(text)
                n += 1
        return {**kwargs, "n_epochs": n}
