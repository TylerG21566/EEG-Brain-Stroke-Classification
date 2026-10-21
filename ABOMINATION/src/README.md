# Pipeline

Run from `ABOMINATION/` with the venv set up (see the top-level README).

## Run

```bash
.venv/bin/python src/pipeline.py                 # all stages
.venv/bin/python src/pipeline.py --start 0 --end 2 # stages 0-2 only (also -s / -e)
.venv/bin/python src/pipeline.py -s 7 -e 7         # modelling only, from p4's CSVs on disk
```

Stages are numbered in `Registered_pipelines` in `pipeline.py` (0 sanity, 1–4 = p1–p4, 7 = p7; p5 Select and p6 Model run inside p7, once per CV fold). `--end` must be a stage and `--start <= --end`; `--start` must be in `STARTABLE` (0, or 7 which reads p4's CSVs from disk). Steps run in order; each step's `output(**kwargs)` returns a dict that is passed to the next step. Startable stages always begin with an empty dict, so p7 (and the p5/p6 inside it) never sees p1–p4's outputs: the only links are the config and p4's CSVs on disk. p2 drops p1's raw signals (`data`) once they are on disk.

Notes:

- p3 is lazy: its `subjects` output is a generator of small per-subject tasks; nothing is computed until p4's workers run them.
- Each p4 worker loads one subject's p2 signal once and runs every `(duration, overlap)` in `config["p3_epoching"]["combinations"]`: all epochs are cut as one array (`windows`), then features are computed in single vectorised calls.
- p3 only writes epoch files when its `"debug"` is `True`, to `out/p3/<Dataset>/<duration>s/<overlap>pct/<subject>_e<idx>_t<start>s.npy`.
- p4 computes skew, kurtosis and Shannon entropy per channel for each epoch (3 × channels; 48 for the archive, 57 for TUH), then the 10/25/50/75/90th percentiles across the subject's epochs (240 values). Writes `out/p4/<Dataset>/<duration>s/<overlap>pct/<subject>.csv` (rows = features, columns = quantiles) and returns `features`: `{(duration, overlap): table with one row per subject}`; Welch band features go to `<subject>_welch.csv` / `welch_features` and left/right BSI to `<subject>_bsi.csv` / `asymmetry_features` likewise, plus a wavelet TF `<subject>.png`. Extractors live in `components/features/`. Each class (`shape`, `welch`, `asymmetry`, `entropy`, `wavelet`) has an `enabled` flag in `config["p4_features"]`, and the first three a `features` dict that toggles single features; a disabled class writes no CSV/PNG and its output key is absent.
- Worker pools (p2, p4) use every CPU core (`os.cpu_count()`).

## Config

Options live in `components/p0_pipeline_config.py`: a plain Python dict (`CONFIG`) with comments. Edit it to change options. Dataset and view choices are `StrEnum` classes there (`Dataset.ARCHIVE`/`Dataset.FIGSHARE`, `View.NONE`/`View.TABLES`/`View.NPY`) for steps to import. `base.load_config()` returns a deep copy, loaded once in `pipeline.py` and passed to every step.

- One top-level key per stage (`p1_data_formatting` … `p7_loader`) plus shared `paths`.
- Relative paths resolve against `ABOMINATION/` (`base.resolve`).
- Each step reads its section as `self.opts` (full dict in `self.config`); the section is set by `section = "..."` on the class.

## Test

```bash
.venv/bin/python -m pytest -q
```

## Adding a step

1. Create `components/pN_<name>.py` with a class subclassing `PipelineStep` (from `components.base`), set `section = "pN_<name>"`, read options from `self.opts`, and return a dict from `output`.
2. Add a `"pN_<name>"` section to `CONFIG` in `components/p0_pipeline_config.py`.
3. Import it in `pipeline.py` and add `N: [YourStep(config)]` to `Registered_pipelines` (add `N` to `STARTABLE` if it can run without earlier stages).
4. Add tests in `tests/test_<name>.py`.

## Stages

| Stage | What it does |
|---|---|
| p1 Load | Raw trials / EDFs into a table: signal, label, subtype, split (TUH: one row per subject from the headers db, signal read later in p2) |
| p2 Preprocess | Stitch, resample 250 Hz, CAR, bandpass 1–30 Hz, FP1/FP2 eye regression; one `.npz` per subject |
| p3 Epoch | Sliding window (duration × overlap); z-score per epoch, clip ±20 SD |
| p4 Features | Per-epoch shape features (skew, kurtosis, entropy × channels: 48 archive, 57 TUH) + Welch band features (4 relative powers + 6 ratios + DTABR, spectral mean, 1/f slope × channels: 208 archive, 247 TUH) + BSI per left/right pair (8 + mean = 9) + fuzzy / sample / approximate entropy (× channels: 48 archive, 57 TUH) → q10/25/50/75/90 across epochs, one quantile CSV per subject each; Morlet wavelet TF PNG per subject |
| p5 Select | Top k (30) by mutual information or ANOVA, on each fold's training subjects, only over features enabled in `p4_features`; prints them |
| p6 Model | Fits the classifier named by `p6_model.model` (`ModelName`; XGBoost now, fixed hyperparameters in `p6_model.xgboost`, `scale_pos_weight` = neg/pos per fold) on the fold's training subjects; returns `y_score` = P(stroke) and `y_pred` (`y_score >= threshold`) for the test subjects. One folder per model family in `components/models/<name>/`, picked by `build(name, opts)` |
| p7 Loader | Loads p4's CSVs (enabled classes only), labels subjects (`negative` subtypes = 0), runs p5 → p6 per stratified subject-level fold (`folds` × `repeats`) and scores each test fold (`p7_loader.metrics`: AUC, accuracy, balanced accuracy, sensitivity = stroke recall, specificity = healthy recall; AUC is NaN for a one-class fold). Per (duration, overlap) prints the most-chosen features, mean ± SD of each metric and the pooled out-of-fold AUC (all test scores of a repeat together, averaged over repeats), then a table of all combinations sorted by pooled AUC. Writes `out/p7/<Dataset>/<d>s/<o>pct/metrics.csv` (repeat, fold, n_train, n_test, metrics) and `predictions.csv` (subject_id, repeat, fold, y_true, y_score, y_pred), plus `out/p7/<Dataset>/summary.csv` (one row per combination: duration, overlap, folds, `<metric>_mean` / `_std`, pooled_auc); overwritten on rerun. Returns `selection` (per combination: folds, frequency, subjects, labels, metrics, predictions, summary) and `summary` (that table). *Planned:* permutation test |

- p5–p6 run inside each CV fold (`MINI = [Selection, Model]` in `p7_loader.py`), so p7 calls them per fold rather than them running once in the chain. Each fold's layer holds the training subjects (`X`, `y`) and the test subjects (`X_test`, `y_test`): p5 scores and picks features on `X`/`y` only, then cuts `X_test` to the same columns and fills its gaps with the training medians; p6 fits on `X`/`y` and only predicts `X_test`; `y_test` is passed through untouched for p7's metrics. Neither sees p4's in-memory output.
- p3–p7 repeat per duration × overlap combination; p1–p2 run once and are reused.

## Adding a model

1. Create `components/models/<name>/__init__.py` with a class taking `opts` (its config sub-dict) and providing `fit(X, y) -> self` and `score(X) -> P(stroke)` of shape (n,) (the `Model` protocol in `components/models/__init__.py`).
2. Register it in `MODELS` in `components/models/__init__.py` (`"<name>": YourModel`).
3. Add a `ModelName` member in `components/p0_pipeline_config.py`, a `"<name>": {...}` hyperparameter sub-dict under `p6_model`, and set `p6_model.model` to it to use it.
4. Add tests in `tests/`. p6 and p7 need no changes: metrics and outputs are the same for every model.

## TUH data

`scripts/download_tuh.py` fetches labelled TUH EEG (TUEG v2.0.2) recordings over rsync/SSH (needs NEDC access and `~/.ssh/id_ed25519`). The selection comes from the headers db (`recordings` table, `label` = stroke/healthy); files land at `<root>/<edf_path without "../">`. `--db` / `--root` default to `p1_data_formatting.tuh.db_path` / `roots.tuh` in the config.

```bash
.venv/bin/python scripts/download_tuh.py --dry-run                  # count, hours, already-present, rsync --stats size (~10.8 GB for longest)
.venv/bin/python scripts/download_tuh.py --limit-subjects 2         # small sample: 2 subjects per label
.venv/bin/python scripts/download_tuh.py                            # longest labelled recording per subject (337 files)
.venv/bin/python scripts/download_tuh.py --which all --label stroke # every labelled stroke recording
```

- `--which longest` (default) matches the p1 loader: max `duration_s` per subject, ties by session, token, path.
- Re-running skips files already downloaded (rsync); `--host`, `--base`, `--key` override the remote.
