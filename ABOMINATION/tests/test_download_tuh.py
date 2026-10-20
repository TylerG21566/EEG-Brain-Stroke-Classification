import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_tuh as dt

ROWS = [ # edf_path, patient, duration_s, label
    ("../edf/000/aaa/s001_2003/02_tcp_le/aaa_s001_t000.edf", "aaa", 600.0, "stroke"),
    ("../edf/000/aaa/s002_2004/02_tcp_le/aaa_s002_t000.edf", "aaa", 1200.0, "stroke"), # longest for aaa
    ("../edf/000/bbb/s001_2003/02_tcp_le/bbb_s001_t001.edf", "bbb", 900.0, "stroke"), # tie: t000 wins
    ("../edf/000/bbb/s001_2003/02_tcp_le/bbb_s001_t000.edf", "bbb", 900.0, "stroke"),
    ("../edf/000/ccc/s001_2003/02_tcp_le/ccc_s001_t000.edf", "ccc", 300.0, "healthy"),
    ("../edf/000/ddd/s001_2003/02_tcp_le/ddd_s001_t000.edf", "ddd", 5000.0, None), # unlabelled: never picked
    ("../edf/000/ccc/s009_2010/02_tcp_le/ccc_s009_t000.edf", "ccc", 9999.0, None), # longer but unlabelled
]


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "h.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE recordings(edf_path TEXT, patient TEXT, v1_subject INT, session INT, session_year INT, token INT, montage TEXT, duration_s REAL, label TEXT)")
    con.executemany("INSERT INTO recordings(edf_path, patient, duration_s, label) VALUES (?,?,?,?)", ROWS)
    con.commit()
    con.close()
    return p


def paths(rows):
    return [r["edf_path"].rsplit("/", 1)[1] for r in rows]


def test_longest_per_subject(db):
    assert paths(dt.select(db)) == ["aaa_s002_t000.edf", "bbb_s001_t000.edf", "ccc_s001_t000.edf"]


def test_all_labelled(db):
    assert len(dt.select(db, "all")) == 5


def test_label_filter(db):
    assert paths(dt.select(db, "all", "healthy")) == ["ccc_s001_t000.edf"]
    assert {r["label"] for r in dt.select(db, label="stroke")} == {"stroke"}


def test_limit_per_label(db):
    rows = dt.select(db, "all", limit=1)
    assert {(r["patient"], r["label"]) for r in rows} == {("aaa", "stroke"), ("ccc", "healthy")}
    assert len(rows) == 3 # both of aaa's recordings + ccc's


def test_missing_db(tmp_path):
    with pytest.raises(FileNotFoundError):
        dt.select(tmp_path / "nope.db")


def test_rel_path():
    assert dt.rel_path("../edf/000/a/x.edf") == "edf/000/a/x.edf"


def test_files_from_and_cmd(db, tmp_path):
    rows = dt.select(db)
    f = dt.write_files_from(rows, tmp_path / "files.txt")
    assert f.read_text().splitlines() == [dt.rel_path(r["edf_path"]) for r in rows]
    cmd = dt.rsync_cmd(f, tmp_path / "root", key="/k", dry_run=True)
    assert cmd[0] == "rsync" and f"--files-from={f}" in cmd and "--dry-run" in cmd
    assert cmd[cmd.index("-e") + 1].startswith("ssh -i /k")
    assert cmd[-2:] == [f"{dt.HOST}:{dt.BASE}", f"{tmp_path / 'root'}/"]


def test_run_rsync_mocked(db, tmp_path, monkeypatch):
    seen = {}
    def fake(cmd):
        seen["cmd"] = cmd
        seen["list"] = Path(next(c for c in cmd if c.startswith("--files-from=")).split("=", 1)[1]).read_text() # read while the temp file still exists
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(dt.subprocess, "run", fake)
    dt.run_rsync(dt.select(db, label="healthy"), tmp_path / "root", host="h", base="b/", key="k")
    assert seen["list"] == "edf/000/ccc/s001_2003/02_tcp_le/ccc_s001_t000.edf\n"
    assert seen["cmd"][-2:] == ["h:b/", f"{tmp_path / 'root'}/"] and "--dry-run" not in seen["cmd"]


def test_ssh_failure_exits(db, tmp_path, monkeypatch):
    monkeypatch.setattr(dt.subprocess, "run", lambda cmd: SimpleNamespace(returncode=255))
    with pytest.raises(SystemExit, match="SSH"):
        dt.run_rsync(dt.select(db), tmp_path / "root")


def test_summary_counts_existing(db, tmp_path):
    rows = dt.select(db)
    p = tmp_path / dt.rel_path(rows[0]["edf_path"])
    p.parent.mkdir(parents=True)
    p.touch()
    s = dt.summary(rows, tmp_path)
    assert "stroke: 2 subjects" in s and "healthy: 1 subjects" in s and "1/3" in s


def test_cli_overrides(db, tmp_path):
    a = dt.parse(["--db", str(db), "--root", str(tmp_path), "--label", "stroke", "--limit-subjects", "2"])
    assert (a.db, a.root, a.label, a.limit_subjects, a.which) == (str(db), str(tmp_path), "stroke", 2, "longest")
