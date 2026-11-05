# TUH EEG (TUEG v2.0.2) helpers: pick each labelled subject's recording from the headers db, read one EDF's 19 electrodes.
# Plain top-level functions so p2's worker processes can import and call read_tuh.

import sqlite3
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from components.p0_pipeline_config import Recordings

# one row per subject: longest recording in any labelled session (ties: earliest session, then token, then path)
LONGEST = """
SELECT patient, session, token, edf_path, label, sample_frequency, duration_s, montage, age, gender FROM (
    SELECT r.patient, r.session, r.token, r.edf_path, l.label, h.sample_frequency, r.duration_s, r.montage, h.age, h.gender,
           ROW_NUMBER() OVER (PARTITION BY r.patient ORDER BY r.duration_s DESC, r.session, r.token, r.edf_path) AS n
    FROM recordings r JOIN labels l USING (patient, session) JOIN edf_headers h USING (edf_path))
WHERE n = 1 ORDER BY label, patient
"""

# every recording in a labelled session (CAPPED / ALL)
LABELLED = """
SELECT r.patient, r.session, r.token, r.edf_path, l.label, h.sample_frequency, r.duration_s, r.montage, h.age, h.gender
FROM recordings r JOIN labels l USING (patient, session) JOIN edf_headers h USING (edf_path)
"""

def labelled_recordings(db_path: str | Path, mode: Recordings = Recordings.LONGEST, max_minutes: float = 20.0) -> pd.DataFrame: # one row per subject; edf_paths + seconds (None = whole file) to read and join
    with sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True) as con: # read-only; fails if the db is missing
        df = pd.read_sql_query(LONGEST if mode == Recordings.LONGEST else LABELLED, con)
    df["edf_path"] = df.edf_path.str.removeprefix("../") # "../edf/..." -> "edf/...", relative to the corpus version dir (p1's TUH root)
    if mode == Recordings.LONGEST:
        return df.assign(edf_paths=df.edf_path.map(lambda p: [p]), seconds=[[None]] * len(df))
    if mode == Recordings.ALL:
        return join(df.sort_values(["patient", "session", "token", "edf_path"]).assign(seconds=None), ["patient"])
    return join(capped(df, max_minutes * 60), ["patient", "session"])

def case_ids(df: pd.DataFrame, labels: dict[str, str], mode: Recordings) -> pd.Series: # labelled_recordings rows -> subject_id, e.g. stroke_aaaaaabk (CAPPED: stroke_aaaaaabk-s001; "-" keeps p7's subtype rule)
    return df.label.map(labels) + "_" + df.patient + (("-s" + df.session.map("{:03d}".format)) if mode == Recordings.CAPPED else "")

def capped(df: pd.DataFrame, max_s: float) -> pd.DataFrame: # per labelled session: its files longest first until max_s in total; the last file is cut short
    df = df.sort_values(["patient", "session", "duration_s", "token", "edf_path"], ascending=[True, True, False, True, True])
    before = df.groupby(["patient", "session"]).duration_s.cumsum() - df.duration_s # time already taken in this session before this file
    keep = before < max_s
    return df[keep].assign(seconds=(max_s - before[keep]).clip(upper=df.duration_s[keep]))

def join(df: pd.DataFrame, by: list[str]) -> pd.DataFrame: # one row per file (in join order, seconds None = whole) -> one row per `by` group; metadata (rate, montage) from the first file
    g = df.groupby(by, sort=False)
    out = g.first().drop(columns="seconds").assign(edf_paths=g.edf_path.agg(list), seconds=g.seconds.agg(list),
                                                 duration_s=g.apply(lambda f: f.seconds.fillna(f.duration_s).sum(), include_groups=False))
    return out.reset_index().sort_values(["label", *by], ignore_index=True)

def has_ekg(path: str | Path) -> bool: # header only: any channel named EKG / ECG
    return any("EKG" in c.upper() or "ECG" in c.upper() for c in mne.io.read_raw_edf(path, preload=False, verbose="error").ch_names)

def read_tuh(path: str | Path, electrodes: list[str], references: list[str]) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(path, preload=False, verbose="error") # header only; data read below for the picked channels
    names = {c.upper(): c for c in raw.ch_names}
    picks = {e: next((names[k] for r in references if (k := f"EEG {e}-{r}".upper()) in names), None) for e in electrodes}
    if missing := [e for e, c in picks.items() if c is None]:
        raise ValueError(f"{path}: no EEG <E>-{{{'|'.join(references)}}} channel for electrodes {missing}")
    return raw.get_data(picks=list(picks.values()), units="uV").astype(np.float32), raw.info["sfreq"] # (electrodes, time) in config order
