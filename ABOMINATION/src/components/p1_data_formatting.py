import re, webbrowser
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from typing import Any
from components.base import Layer, Opts, PipelineStep, resolve
from components.p0_pipeline_config import Dataset, View
from components.tuh import labelled_recordings, read_tuh

class DataFormat(PipelineStep):
    section = "p1_data_formatting"

    def output(self, **kwargs: Any) -> Layer:
        dataset = self.opts["dataset"]
        root = resolve(self.opts["roots"][dataset])
        data = {Dataset.ARCHIVE: archive, Dataset.FIGSHARE: lambda r, _: figshare(r), Dataset.TUH: tuh}[dataset](root, self.opts)
        data.attrs["dataset"] = dataset
        view, out = self.opts["debug_view"], resolve(self.opts["debug_dir"])
        if view == View.NPY:
            signal, sfreq = (data.signal[0], data.attrs["sfreq"]) if data.signal[0] is not None else read_tuh(data.path[0], data.attrs["electrodes"], data.attrs["references"]) # TUH: read on demand
            signal = signal[:, :self.opts["view_samples"]]
            popup(out, **{Path(data.path[0]).name: pd.DataFrame(signal.T, columns=data.attrs["channels"], index=np.arange(signal.shape[1]) / sfreq)})
        elif view == View.TABLES:
            popup(out, data=data)
        return {**kwargs, "data": data}

def archive(root: Path, opts: Opts) -> pd.DataFrame:
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

def figshare(root: Path) -> pd.DataFrame:
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

def tuh(root: Path, opts: Opts) -> pd.DataFrame:
    t = opts["tuh"]
    df = labelled_recordings(resolve(t["db_path"]))
    subtype = df.label.map(t["labels"])
    df = pd.DataFrame({"subject_id": subtype + "_" + df.patient, "label": df.label, "subtype": subtype, "split": None, "patient": df.patient, "session": df.session,
                       "token": df.token, "sfreq": df.sample_frequency, "montage": df.montage, "age": df.age, "gender": df.gender, "signal": None,
                       "path": [str(root / p) for p in df.edf_path]}) # signals stay on disk: p2 workers read them with read_tuh (all at once would be ~8 GB)
    on_disk = df.path.map(lambda p: Path(p).exists())
    if not on_disk.all():
        if not t["skip_missing"]:
            raise FileNotFoundError(f"{(~on_disk).sum()} TUH EDFs missing under {root}, e.g. {df.path[~on_disk].iloc[0]}; run scripts/download_tuh.py")
        print(f"p1 TUH: skipped {(~on_disk).sum()} of {len(df)} subjects (EDF not on disk), {on_disk.sum()} remain {df[on_disk].label.value_counts().to_dict()}")
    if not on_disk.any():
        raise FileNotFoundError(f"no TUH EDFs found under {root}; run scripts/download_tuh.py first")
    df = df[on_disk].reset_index(drop=True)
    df.attrs = {"sfreq": None, "channels": list(t["electrodes"]), "electrodes": list(t["electrodes"]), "references": list(t["references"])} # sfreq varies per row (250/256/400/1000 Hz)
    return df

def popup(out: Path, **tables: pd.DataFrame) -> None:
    css = "<style>body{font:13px sans-serif}table{border-collapse:collapse}th{position:sticky;top:0;background:#eee}td,th{border:1px solid #ccc;padding:2px 6px}</style>"
    tables = {k: df.map(lambda x: x.shape if isinstance(x, np.ndarray) else x) for k, df in tables.items()}
    out.mkdir(parents=True, exist_ok=True)
    for k, df in tables.items():
        df.to_csv(out / f"p1_{Path(k).stem}.csv")
    body = "".join(f"<h2>{k} {df.shape}</h2>" + df.to_html() for k, df in tables.items())
    f = out / "p1_tables.html"
    f.write_text(css + body)
    webbrowser.open(f.as_uri())
