import re, webbrowser
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from components.base import PipelineStep, resolve
from components.p0_pipeline_config import Dataset, View

class DataFormat(PipelineStep):
    section = "p1_data_formatting"

    def output(self, **kwargs):
        dataset = self.opts["dataset"]
        root = resolve(self.opts["roots"][dataset])
        data = archive(root, self.opts) if dataset == Dataset.ARCHIVE else figshare(root)
        data.attrs["dataset"] = dataset
        view, out = self.opts["debug_view"], resolve(self.opts["debug_dir"])
        if view == View.NPY:
            signal = data.signal[0][:, :self.opts["view_samples"]]
            popup(out, **{Path(data.path[0]).name: pd.DataFrame(signal.T, columns=data.attrs["channels"], index=np.arange(signal.shape[1]) / data.attrs["sfreq"])})
        elif view == View.TABLES:
            popup(out, data=data)
        return {**kwargs, "data": data}

def archive(root, opts):
    a = opts["archive"]
    name = re.compile(a["name_pattern"])
    rows = []
    for p in sorted((root / "archive").glob("*/*/*.npy")):
        subtype, subject, file = name.fullmatch(p.name).groups()
        flat = np.load(p)
        rows.append({"subject_id": f"{subtype}_{subject}", "label": p.parts[-2], "subtype": subtype, "split": p.parts[-3], "subject": int(subject),
                     "file": int(file), "trial": int(flat[-1]), "signal": flat[:-1].reshape(a["channels"], a["samples"]), "path": str(p)})
    df = pd.DataFrame(rows).sort_values(["split", "label", "subtype", "subject", "file"], ignore_index=True)
    df.attrs = {"sfreq": a["sfreq"], "channels": a["channel_names"]} # assumed 10-20 order
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

def popup(out, **tables):
    css = "<style>body{font:13px sans-serif}table{border-collapse:collapse}th{position:sticky;top:0;background:#eee}td,th{border:1px solid #ccc;padding:2px 6px}</style>"
    tables = {k: df.map(lambda x: x.shape if isinstance(x, np.ndarray) else x) for k, df in tables.items()}
    out.mkdir(parents=True, exist_ok=True)
    for k, df in tables.items():
        df.to_csv(out / f"p1_{Path(k).stem}.csv")
    body = "".join(f"<h2>{k} {df.shape}</h2>" + df.to_html() for k, df in tables.items())
    f = out / "p1_tables.html"
    f.write_text(css + body)
    webbrowser.open(f.as_uri())
