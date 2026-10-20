import sqlite3
from pathlib import Path
import mne, numpy as np, pandas as pd, pytest
import components.p0_pipeline_config as p0
import components.p1_data_formatting as p1
from components.base import load_config
from components.p1_data_formatting import DataFormat
from components.tuh import labelled_recordings, read_tuh

E = load_config()["p1_data_formatting"]["tuh"]["electrodes"]
REFS = ["REF", "LE"]
# (patient, session, token, duration_s, sfreq, label of the session or None); the longest labelled recording per patient wins
RECS = [("aaaaaabk", 1, 0, 600.0, 250.0, "stroke"), ("aaaaaabk", 1, 1, 1200.0, 250.0, "stroke"), ("aaaaaabk", 2, 0, 9999.0, 250.0, None),
        ("aaaaacdh", 1, 0, 300.0, 256.0, "healthy"), ("aaaaacdh", 2, 0, 900.0, 1000.0, "healthy"),
        ("aaaaazzz", 1, 0, 500.0, 400.0, "stroke"), ("aaaaazzz", 1, 1, 500.0, 400.0, "stroke"), ("aaaaanone", 1, 0, 800.0, 250.0, None)]
WINNERS = {"stroke_aaaaaabk": (1, 1), "normal_aaaaacdh": (2, 0), "stroke_aaaaazzz": (1, 0)} # subject_id -> (session, token); tie -> lower token


def edf(patient, session, token):
    return f"../edf/000/{patient}/s{session:03d}_2003/02_tcp_le/{patient}_s{session:03d}_t{token:03d}.edf"


@pytest.fixture
def db(tmp_path):
    f = tmp_path / "tueg_headers.db"
    with sqlite3.connect(f) as con:
        con.execute("CREATE TABLE labels (patient TEXT, session INTEGER, v1_subject INTEGER, label TEXT, source TEXT)")
        con.execute("CREATE TABLE recordings (edf_path TEXT, patient TEXT, v1_subject INT, session INT, session_year INT, token INT, montage TEXT, duration_s REAL, label TEXT)")
        con.execute("CREATE TABLE edf_headers (edf_path TEXT, patient TEXT, session INT, token INT, montage TEXT, gender TEXT, age REAL, sample_frequency REAL, duration_s REAL, nsig INT)")
        for pat, ses, tok, dur, fs, lab in RECS:
            con.execute("INSERT INTO recordings VALUES (?,?,0,?,2003,?,'02_tcp_le',?,?)", (edf(pat, ses, tok), pat, ses, tok, dur, lab))
            con.execute("INSERT INTO edf_headers VALUES (?,?,?,?,'02_tcp_le','F',50.0,?,?,33)", (edf(pat, ses, tok), pat, ses, tok, fs, dur))
        con.executemany("INSERT INTO labels VALUES (?,?,0,?,'x')", sorted({(p, s, l) for p, s, _, _, _, l in RECS if l}))
    return f


def fake_raw(names, n=500, sfreq=250.0):  # channel k holds the constant k + 1 µV, so picks can be traced back to names
    data = (np.arange(len(names))[:, None] + 1.0) * np.ones(n) * 1e-6
    return mne.io.RawArray(data, mne.create_info(names, sfreq, "eeg"), verbose="error")


@pytest.fixture
def edfs(monkeypatch):  # path -> channel names; mne.io.read_raw_edf serves fake_raw for them
    files = {}
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda p, **kw: fake_raw(files[str(p)]))
    return files


def test_labelled_recordings_longest_per_subject(db):
    df = labelled_recordings(db)
    assert df.patient.is_unique and set(df.label) == {"stroke", "healthy"}
    assert {f"{'normal' if r.label == 'healthy' else r.label}_{r.patient}": (r.session, r.token) for r in df.itertuples()} == WINNERS
    assert all(p.startswith("edf/") for p in df.edf_path) and {"sample_frequency", "duration_s", "montage", "age", "gender"} <= set(df.columns)


def test_read_tuh_order_across_refs(edfs):
    edfs["le"] = ["EEG EKG1-REF", *[f"EEG {e}-LE" for e in reversed(E)], "PHOTIC PH"]  # extras + reversed order
    edfs["ref"] = [f"EEG {e.capitalize()}-ref" for e in E] + ["EEG A1-REF"]  # case-insensitive match
    for f, name in (("le", "EEG {}-LE"), ("ref", "EEG {}-ref")):
        x, sfreq = read_tuh(f, E, REFS)
        assert x.dtype == np.float32 and x.shape == (len(E), 500) and sfreq == 250.0
        assert np.allclose(x[:, 0], [[n.upper() for n in edfs[f]].index(name.format(e).upper()) + 1 for e in E])  # µV, config order


def test_read_tuh_first_reference_wins(edfs):
    edfs["both"] = [f"EEG {e}-LE" for e in E] + [f"EEG {e}-REF" for e in E]
    assert np.allclose(read_tuh("both", E, REFS)[0][:, 0], np.arange(len(E)) + len(E) + 1)  # REF block is second
    assert np.allclose(read_tuh("both", E, ["LE", "REF"])[0][:, 0], np.arange(len(E)) + 1)


def test_read_tuh_missing_electrode(edfs):
    edfs["short"] = [f"EEG {e}-REF" for e in E if e not in ("CZ", "T6")]
    with pytest.raises(ValueError, match=r"short.*\['CZ', 'T6'\]"):
        read_tuh("short", E, REFS)


@pytest.fixture
def tuh_cfg(cfg, db):
    cfg["p1_data_formatting"].update(dataset=p0.Dataset.TUH)
    cfg["p1_data_formatting"]["tuh"]["db_path"] = str(db)
    return cfg


def put(cfg, *subjects):  # create the winning EDFs of these subjects under the TUH root
    root = Path(cfg["p1_data_formatting"]["roots"][p0.Dataset.TUH])
    for s in subjects:
        f = root / edf(s.split("_")[1], *WINNERS[s]).removeprefix("../")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()


def test_p1_tuh_table(tuh_cfg):
    put(tuh_cfg, *WINNERS)
    out = DataFormat(tuh_cfg).output(extra=1)
    df = out["data"]
    assert out["extra"] == 1 and sorted(df.subject_id) == sorted(WINNERS)
    assert (df.subject_id == df.subtype + "_" + df.patient).all() and set(df.subtype) == {"stroke", "normal"}
    assert df.signal.isna().all() and df.split.isna().all() and all(Path(p).is_file() for p in df.path)
    assert dict(zip(df.subject_id, df.sfreq)) == {"stroke_aaaaaabk": 250.0, "normal_aaaaacdh": 1000.0, "stroke_aaaaazzz": 400.0}
    assert df.attrs["dataset"] == p0.Dataset.TUH and df.attrs["channels"] == E and df.attrs["sfreq"] is None


def test_p1_tuh_skips_missing(tuh_cfg, capsys):
    put(tuh_cfg, "normal_aaaaacdh")
    df = DataFormat(tuh_cfg).output()["data"]
    assert list(df.subject_id) == ["normal_aaaaacdh"] and "skipped 2 of 3" in capsys.readouterr().out
    tuh_cfg["p1_data_formatting"]["tuh"]["skip_missing"] = False
    with pytest.raises(FileNotFoundError, match="download_tuh"):
        DataFormat(tuh_cfg).output()


def test_p1_tuh_none_on_disk(tuh_cfg):
    with pytest.raises(FileNotFoundError, match="download_tuh"):
        DataFormat(tuh_cfg).output()


@pytest.mark.parametrize("view", [p0.View.NPY, p0.View.TABLES])
def test_p1_tuh_views(tuh_cfg, edfs, monkeypatch, view):
    put(tuh_cfg, *WINNERS)
    root = Path(tuh_cfg["p1_data_formatting"]["roots"][p0.Dataset.TUH])
    for s, (ses, tok) in WINNERS.items():
        edfs[str(root / edf(s.split("_")[1], ses, tok).removeprefix("../"))] = [f"EEG {e}-LE" for e in E]
    monkeypatch.setattr(p1.webbrowser, "open", lambda *a: None)
    tuh_cfg["p1_data_formatting"]["debug_view"] = view
    DataFormat(tuh_cfg).output()
    out = Path(tuh_cfg["p1_data_formatting"]["debug_dir"])
    tables = {f.stem: pd.read_csv(f, index_col=0) for f in out.glob("p1_*.csv")}
    assert (out / "p1_tables.html").exists() and len(tables) == 1
    t = next(iter(tables.values()))
    assert (list(t.columns), len(t)) == (E, 500) if view == p0.View.NPY else len(t) == len(WINNERS)


@pytest.mark.parametrize("real_cfg", [p0.Dataset.TUH], indirect=True)
def test_real_tuh(real_cfg):
    df = DataFormat(real_cfg).output()["data"]
    assert len(df) and set(df.subtype) <= {"stroke", "normal"} and df.subject_id.is_unique
    x, sfreq = read_tuh(df.path[0], df.attrs["electrodes"], df.attrs["references"])
    assert x.shape[0] == len(df.attrs["channels"]) and sfreq == df.sfreq[0]
