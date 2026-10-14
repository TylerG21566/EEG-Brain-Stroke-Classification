# Pipeline

Run from `ABOMINATION/` with the venv set up (see the top-level README).

## Run

```bash
.venv/bin/python src/pipeline.py
```

Steps run in the order listed in `Registered_pipelines` in `pipeline.py`. Each step's `output(**kwargs)` returns a dict that is passed to the next step.

## Config

Options live in `components/p0_pipeline_config.py`: a plain Python dict (`CONFIG`) with comments. Edit it to change options. Dataset and view choices are `StrEnum` classes there (`Dataset.ARCHIVE`/`Dataset.FIGSHARE`, `View.NONE`/`View.TABLES`/`View.NPY`) for steps to import. `base.load_config()` returns a deep copy, loaded once in `pipeline.py` and passed to every step.

- One top-level key per stage (`p1_data_formatting`, `p2_sample_processing`) plus shared `paths`.
- Relative paths resolve against `ABOMINATION/` (`base.resolve`).
- Each step reads its section as `self.opts` (full dict in `self.config`); the section is set by `section = "..."` on the class.

## p2: sample processing

`SampleProcessing` (`components/p2_sample_processing.py`) takes p1's `data` DataFrame and adds `processed`, a metadata DataFrame. Dataset comes from `data.attrs["dataset"]` (set by p1), else `p1_data_formatting.dataset`. Options are under `p2_sample_processing` in the config.

- Preprocessing per recording: resample to `sfreq` (250 Hz) → common average reference → bandpass `l_freq`–`h_freq` (1–30 Hz) → `eye_channels` (FP1/FP2) eye-artefact regression (when those channels exist).
- Archive trials are stitched per subject; FIgshare recordings are not.
- Writes to `out_dirs[dataset]` (`out/p2/Tamim/` or `out/p2/FIgshare/`) as one `.npz` per subject named `<subject_id>.npz` (e.g. `ischemic_5.npz`, `sub-01.npz`) holding just `signal`; labels/split are in the returned `processed` table. Existing `.npz` files there are deleted at start.
- Runs over a `multiprocessing.Pool` (`processes` workers, 4), so `pipeline.py` must keep `main()` under `if __name__ == "__main__"`.

## Test

```bash
.venv/bin/python -m pytest -q
```

## Adding a step

1. Create `components/pN_<name>.py` with a class subclassing `PipelineStep` (from `components.base`), set `section = "pN_<name>"`, read options from `self.opts`, and return a dict from `output`.
2. Add a `"pN_<name>"` section to `CONFIG` in `components/p0_pipeline_config.py`.
3. Import it in `pipeline.py` and add `YourStep(config)` to `Registered_pipelines` (before `Show`).
4. Add tests in `tests/test_<name>.py`.
