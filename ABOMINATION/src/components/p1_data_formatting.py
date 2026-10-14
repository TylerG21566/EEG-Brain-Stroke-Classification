import re, webbrowser
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from components.base import PipelineStep

NO_VIEW, TABLES_VIEW, NPY_VIEW = range(3) # no popup, full tables, first recording only
DEBUG_VIEW = TABLES_VIEW

ARCHIVE, FIGSHARE = range(2) # Tamim's synthetic Kaggle archive, Liu & Lv real stroke EDFs
DATASET = FIGSHARE

DATA = Path(__file__).resolve().parents[2] / "data"
ROOTS = {ARCHIVE: DATA / "Tamim-data", FIGSHARE: DATA / "FIgshare"}
OUT = DATA.parent / "out"
VIEW_SAMPLES = 1024
CHANNELS, SAMPLES, SFREQ = 16, 1024, 256.0
NAME = re.compile(r"(\w+)_subject_(\d+)_(\d+)\.npy")

class DataFormat(PipelineStep):
    def __init__(self, dataset=None, root=None, name=None):
        super().__init__(name)
        self.dataset = DATASET if dataset is None else dataset
        self.root = Path(root or ROOTS[self.dataset])

    def output(self, **kwargs):
        data = (archive if self.dataset == ARCHIVE else figshare)(self.root)
        if DEBUG_VIEW == NPY_VIEW:
            signal = data.signal[0][:, :VIEW_SAMPLES]
            popup(**{Path(data.path[0]).name: pd.DataFrame(signal.T, columns=data.attrs["channels"], index=np.arange(signal.shape[1]) / data.attrs["sfreq"])})
        elif DEBUG_VIEW == TABLES_VIEW:
            popup(data=data)
        return {**kwargs, "data": data}

def archive(root):
    rows = []
    for p in sorted((root / "archive").glob("*/*/*.npy")):
        subtype, subject, file = NAME.fullmatch(p.name).groups()
        flat = np.load(p)
        rows.append({"subject_id": f"{subtype}_{subject}", "label": p.parts[-2], "subtype": subtype, "split": p.parts[-3], "subject": int(subject),
                     "file": int(file), "trial": int(flat[-1]), "signal": flat[:-1].reshape(CHANNELS, SAMPLES), "path": str(p)})
    df = pd.DataFrame(rows).sort_values(["split", "label", "subtype", "subject", "file"], ignore_index=True)
    df.attrs = {"sfreq": SFREQ, "channels": [f"ch{i + 1}" for i in range(CHANNELS)]}
    return df

def figshare(root):
    df = pd.read_csv(root / "metadata" / "participants.tsv", sep="\t").rename(columns={"Participant_ID": "subject_id"})
    df.insert(1, "label", "stroke")
    df.insert(2, "subtype", "ischemic")
    rows = []
    for s in df.subject_id:
        p = root / "edffile" / s / "eeg" / f"{s}_task-motor-imagery_eeg.edf"
        raw = mne.io.read_raw_edf(p, preload=True, verbose="error")
        rows.append({"signal": raw.get_data(units="uV").astype(np.float32), "path": str(p)})
    df = df.join(pd.DataFrame(rows))
    df.attrs = {"sfreq": raw.info["sfreq"], "channels": [c or "trigger" for c in raw.ch_names]}
    return df

def popup(**tables):
    css = "<style>body{font:13px sans-serif}table{border-collapse:collapse}th{position:sticky;top:0;background:#eee}td,th{border:1px solid #ccc;padding:2px 6px}</style>"
    tables = {k: df.map(lambda x: x.shape if isinstance(x, np.ndarray) else x) for k, df in tables.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    for k, df in tables.items():
        df.to_csv(OUT / f"p1_{Path(k).stem}.csv")
    body = "".join(f"<h2>{k} {df.shape}</h2>" + df.to_html() for k, df in tables.items())
    f = OUT / "p1_tables.html"
    f.write_text(css + body)
    webbrowser.open(f.as_uri())
