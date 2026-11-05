# Second architecture's input: patient details per case, keyed by the subject_id p1 builds, so they line up with p4's CSVs (TUH only)
import pandas as pd
from components.base import Config, resolve
from components.p0_pipeline_config import Dataset
from components.tuh import case_ids, labelled_recordings

def demographics(config: Config) -> pd.DataFrame: # subject_id -> age, sex (1 = male, 0 = female, NaN = unknown), from the case's first EDF header
    p1 = config["p1_data_formatting"]
    if p1["dataset"] != Dataset.TUH:
        raise ValueError(f"demographics are only available for TUH, not {p1['dataset']}; set combiner.enabled = False")
    t = p1["tuh"]
    df = labelled_recordings(resolve(t["db_path"]), t["recordings"], t["max_minutes"])
    return pd.DataFrame({"age": df.age.to_numpy(), "sex": df.gender.map({"M": 1.0, "F": 0.0}).to_numpy()}, index=case_ids(df, t["labels"], t["recordings"]))
