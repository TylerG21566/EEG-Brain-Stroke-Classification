"""Download labelled TUH EEG (TUEG v2.0.2) recordings over rsync/SSH.

Selection comes from the headers db (table `recordings`); files land at <root>/edf/... (edf_path minus '../').

  .venv/bin/python scripts/download_tuh.py --dry-run                       # what would be fetched (+ rsync --stats size)
  .venv/bin/python scripts/download_tuh.py --label stroke --limit-subjects 2 # 2 stroke subjects, longest recording each
  .venv/bin/python scripts/download_tuh.py --which all                     # every labelled recording (~14 GB)
"""
import argparse
import shlex
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")) # so components.* imports when run as a script
from components.base import load_config, resolve

DEFAULT_ROOT = "data/GOAT/tuh_eeg/v2.0.2" # fallbacks if the config has no TUH keys yet
DEFAULT_DB = "data/GOAT/DOCS/DataFeatures/tueg_headers.db"
HOST = "nedc-tuh-eeg@www.isip.piconepress.com"
BASE = "data/tuh_eeg/tuh_eeg/v2.0.2/"
KEY = "~/.ssh/id_ed25519"

Row = dict[str, Any] # one recordings row: edf_path, patient, label, duration_s

def config_defaults() -> tuple[str, str]: # (root, db) from CONFIG when present, else the literals above
    p1 = load_config().get("p1_data_formatting", {})
    return p1.get("roots", {}).get("tuh", DEFAULT_ROOT), p1.get("tuh", {}).get("db_path", DEFAULT_DB)

def rel_path(edf_path: str) -> str: # '../edf/000/...' -> 'edf/000/...' (relative to the version dir)
    return edf_path.removeprefix("../").lstrip("/")

def select(db: Path, which: str = "longest", label: str | None = None, limit: int | None = None) -> list[Row]:
    if not db.is_file():
        raise FileNotFoundError(f"TUH headers db not found: {db}")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True) # read-only
    con.row_factory = sqlite3.Row
    q = "SELECT edf_path, patient, label, duration_s FROM recordings WHERE label IS NOT NULL" + (" AND label = ?" if label else "")
    rows = [dict(r) for r in con.execute(q + " ORDER BY patient, duration_s DESC, session, token, edf_path", (label,) if label else ())]
    con.close()
    if which == "longest": # first row per patient = max duration_s; ties: earliest session, token, path (as components/tuh.py)
        seen: set[str] = set()
        rows = [r for r in rows if not (r["patient"] in seen or seen.add(r["patient"]))]
    if limit is not None: # first `limit` subjects (sorted by patient) per label
        keep: dict[str, list[str]] = {}
        for r in rows:
            ps = keep.setdefault(r["label"], [])
            if r["patient"] not in ps and len(ps) < limit:
                ps.append(r["patient"])
        rows = [r for r in rows if r["patient"] in keep[r["label"]]]
    return rows

def summary(rows: list[Row], root: Path) -> str:
    lines = [f"{len(rows)} files, {sum(r['duration_s'] for r in rows) / 3600:.1f} h total"]
    for lab in sorted({r["label"] for r in rows}):
        rs = [r for r in rows if r["label"] == lab]
        lines.append(f"  {lab}: {len({r['patient'] for r in rs})} subjects, {len(rs)} files, {sum(r['duration_s'] for r in rs) / 3600:.1f} h")
    have = sum((root / rel_path(r["edf_path"])).is_file() for r in rows)
    lines.append(f"  already in {root}: {have}/{len(rows)}")
    return "\n".join(lines)

def rsync_cmd(files_from: str | Path, root: Path, host: str = HOST, base: str = BASE, key: str = KEY, dry_run: bool = False) -> list[str]:
    ssh = f"ssh -i {Path(key).expanduser()} -o BatchMode=yes" # BatchMode: fail instead of prompting
    cmd = ["rsync", "-auvL", "--partial", f"--files-from={files_from}", "-e", ssh] # --files-from implies --relative: keeps edf/... under root
    cmd += ["--dry-run", "--stats"] if dry_run else ["--progress"]
    return cmd + [f"{host}:{base.rstrip('/')}/", f"{root}/"]

def write_files_from(rows: list[Row], path: Path) -> Path:
    path.write_text("".join(rel_path(r["edf_path"]) + "\n" for r in rows))
    return path

def run_rsync(rows: list[Row], root: Path, dry_run: bool = False, **remote: str) -> int:
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = rsync_cmd(write_files_from(rows, Path(tmp) / "files.txt"), root, dry_run=dry_run, **remote)
        print("$", shlex.join(cmd), flush=True)
        try:
            rc = subprocess.run(cmd).returncode
        except FileNotFoundError:
            sys.exit("rsync not found on PATH")
    if rc == 255 or rc == 12: # ssh / protocol stream failure
        sys.exit(f"rsync failed ({rc}): SSH connection problem; check the key ({remote.get('key', KEY)}) and host")
    if rc:
        sys.exit(f"rsync failed with exit code {rc}")
    return rc

def parse(argv: list[str] | None = None) -> argparse.Namespace:
    root, db = config_defaults()
    ap = argparse.ArgumentParser(description="Download labelled TUH EEG recordings via rsync.")
    ap.add_argument("--which", choices=["longest", "all"], default="longest", help="longest labelled recording per subject, or every labelled one")
    ap.add_argument("--label", choices=["stroke", "healthy"], help="only this class")
    ap.add_argument("--limit-subjects", type=int, metavar="N", help="at most N subjects per label")
    ap.add_argument("--dry-run", action="store_true", help="print the selection and rsync --dry-run --stats; download nothing")
    ap.add_argument("--db", default=db, help=f"headers db (default {db})")
    ap.add_argument("--root", default=root, help=f"local corpus version dir (default {root})")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--base", default=BASE, help="remote corpus version dir")
    ap.add_argument("--key", default=KEY, help="SSH private key")
    return ap.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    a = parse(argv)
    root = resolve(a.root)
    try:
        rows = select(resolve(a.db), a.which, a.label, a.limit_subjects)
    except FileNotFoundError as e:
        sys.exit(str(e))
    if not rows:
        sys.exit("no recordings match")
    print(f"{'Dry run' if a.dry_run else 'Downloading'} ({a.which}) {a.host}:{a.base} -> {root}\n{summary(rows, root)}", flush=True)
    run_rsync(rows, root, dry_run=a.dry_run, host=a.host, base=a.base, key=a.key)

if __name__ == "__main__":
    main()
