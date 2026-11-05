# Third architecture's input: HRV features per case from the EKG lead, keyed by the subject_id p1 builds (TUH only).
# Each case reads the same files / seconds as its EEG; RR intervals are found per file, so joins add no fake beats. Cached per recordings mode.
import numpy as np
import pandas as pd
from components.base import Config, resolve
from components.p0_pipeline_config import Dataset, Recordings
from components.tuh import case_ids, labelled_recordings
from ekg_components.hrv import hrv, read_ekg, rr_intervals

def ekg_features(config: Config) -> pd.DataFrame: # subject_id -> FEATURES; rerun after deleting the cache CSV
    p1, e = config["p1_data_formatting"], config["ekg_model"]
    if p1["dataset"] != Dataset.TUH:
        raise ValueError(f"EKG features are only available for TUH, not {p1['dataset']}; drop 'ekg' from combiner.models")
    t = p1["tuh"]
    cache = resolve(e["out_dirs"][Dataset.TUH]) / (f"{t['recordings']}_{t['max_minutes']:g}min.csv" if t["recordings"] == Recordings.CAPPED else f"{t['recordings']}.csv")
    if cache.exists():
        return pd.read_csv(cache, index_col=0)
    df, root = labelled_recordings(resolve(t["db_path"]), t["recordings"], t["max_minutes"]), resolve(p1["roots"][Dataset.TUH])
    rows = {}
    for sid, paths, secs in zip(case_ids(df, t["labels"], t["recordings"]), df.edf_paths, df.seconds):
        files = [(root / p, s) for p, s in zip(paths, secs) if (root / p).exists()]
        rows[sid] = hrv(np.concatenate([rr_intervals(*read_ekg(str(p), s), tuple(e["band"])) for p, s in files] or [np.empty(0)]))
    feats = pd.DataFrame.from_dict(rows, orient="index")
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(cache)
    print(f"[ekg] {len(feats)} cases, {feats.isna().any(axis=1).sum()} without usable beats -> {cache}", flush=True)
    return feats
