# TUH EEG (TUEG v2.0.2) helpers: pick each labelled subject's recording from the headers db, read one EDF's 19 electrodes.
# Plain top-level functions so p2's worker processes can import and call read_tuh.

import sqlite3
from pathlib import Path
import mne
import numpy as np
import pandas as pd

# one row per subject: longest recording in any labelled session (ties: earliest session, then token, then path)
LONGEST = """
SELECT patient, session, token, edf_path, label, sample_frequency, duration_s, montage, age, gender FROM (
    SELECT r.patient, r.session, r.token, r.edf_path, l.label, h.sample_frequency, r.duration_s, r.montage, h.age, h.gender,
           ROW_NUMBER() OVER (PARTITION BY r.patient ORDER BY r.duration_s DESC, r.session, r.token, r.edf_path) AS n
    FROM recordings r JOIN labels l USING (patient, session) JOIN edf_headers h USING (edf_path))
WHERE n = 1 ORDER BY label, patient
"""

def labelled_recordings(db_path: str | Path) -> pd.DataFrame:
    with sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True) as con: # read-only; fails if the db is missing
        df = pd.read_sql_query(LONGEST, con)
    df["edf_path"] = df.edf_path.str.removeprefix("../") # "../edf/..." -> "edf/...", relative to the corpus version dir (p1's TUH root)
    return df

def read_tuh(path: str | Path, electrodes: list[str], references: list[str]) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(path, preload=False, verbose="error") # header only; data read below for the picked channels
    names = {c.upper(): c for c in raw.ch_names}
    picks = {e: next((names[k] for r in references if (k := f"EEG {e}-{r}".upper()) in names), None) for e in electrodes}
    if missing := [e for e, c in picks.items() if c is None]:
        raise ValueError(f"{path}: no EEG <E>-{{{'|'.join(references)}}} channel for electrodes {missing}")
    return raw.get_data(picks=list(picks.values()), units="uV").astype(np.float32), raw.info["sfreq"] # (electrodes, time) in config order
