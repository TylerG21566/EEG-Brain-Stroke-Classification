# Pipeline

Run from `ABOMINATION/` with the venv set up (see the top-level README).

## Run

```bash
.venv/bin/python src/pipeline.py
```

Steps run in the order listed in `Registered_pipelines` in `pipeline.py`; each step's `output(**kwargs)` returns a dict that is passed to the next step. p2 drops p1's raw signals (`data`) once they are on disk.

Notes:

- p3 is lazy: its `subjects` output is a generator of small per-subject tasks; nothing is computed until p4's workers run them.
- Each p4 worker loads one subject's p2 signal once and runs every `(duration, overlap)` in `config["p3_epoching"]["combinations"]`: all epochs are cut as one array (`windows`), then features are computed in single vectorised calls.
- p3 only writes epoch files when its `"debug"` is `True`, to `out/p3/<Dataset>/<duration>s/<overlap>pct/<subject>_e<idx>_t<start>s.npy`.
- p4 computes skew, kurtosis and Shannon entropy per channel for each epoch (3 × channels; 48 for the archive), then the 10/25/50/75/90th percentiles across the subject's epochs (240 values). Writes `out/p4/<Dataset>/<duration>s/<overlap>pct/<subject>.csv` (rows = features, columns = quantiles) and returns `features`: `{(duration, overlap): table with one row per subject}`.
- Worker pools (p2, p4) use every CPU core (`os.cpu_count()`).

## Config

Options live in `components/p0_pipeline_config.py`: a plain Python dict (`CONFIG`) with comments. Edit it to change options. Dataset and view choices are `StrEnum` classes there (`Dataset.ARCHIVE`/`Dataset.FIGSHARE`, `View.NONE`/`View.TABLES`/`View.NPY`) for steps to import. `base.load_config()` returns a deep copy, loaded once in `pipeline.py` and passed to every step.

- One top-level key per stage (`p1_data_formatting` … `p4_features`) plus shared `paths`.
- Relative paths resolve against `ABOMINATION/` (`base.resolve`).
- Each step reads its section as `self.opts` (full dict in `self.config`); the section is set by `section = "..."` on the class.

## Test

```bash
.venv/bin/python -m pytest -q
```

## Adding a step

1. Create `components/pN_<name>.py` with a class subclassing `PipelineStep` (from `components.base`), set `section = "pN_<name>"`, read options from `self.opts`, and return a dict from `output`.
2. Add a `"pN_<name>"` section to `CONFIG` in `components/p0_pipeline_config.py`.
3. Import it in `pipeline.py` and add `YourStep(config)` to `Registered_pipelines`.
4. Add tests in `tests/test_<name>.py`.

## Stages

| Stage | What it does |
|---|---|
| p1 Load | Raw trials / EDFs into a table: signal, label, subtype, split |
| p2 Preprocess | Stitch, resample 250 Hz, CAR, bandpass 1–30 Hz, FP1/FP2 eye regression; one `.npz` per subject |
| p3 Epoch | Sliding window (duration × overlap); z-score per epoch, clip ±20 SD |
| p4 Features | Per-epoch features (kurtosis, skew, entropy × 16 channels = 48), then q10/25/50/75/90 across epochs: 240 values, one row per subject |
| p5 Select | Top 30 features by mutual information |
| p6 Model | XGBoost, fixed hyperparameters, `scale_pos_weight` = neg/pos |
| p7 Evaluate | Subject-level stratified 5-fold CV × 50 repeats (AUC) + 32/8 holdout, threshold 0.5 |

- p5–p6 are fitted inside each CV fold, so p7 calls them per fold rather than them running once in the chain.
- p3–p7 repeat per duration × overlap combination; p1–p2 run once and are reused.
