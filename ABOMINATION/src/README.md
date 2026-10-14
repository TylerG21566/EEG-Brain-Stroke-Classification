# Pipeline

Run from `ABOMINATION/` with the venv set up (see the top-level README).

## Run

```bash
.venv/bin/python src/pipeline.py
```

Steps run in the order listed in `Registered_pipelines` in `pipeline.py`. Each step's `output(**kwargs)` returns a dict that is passed to the next step.

## Test

```bash
.venv/bin/python -m pytest -q
```

## Adding a step

1. Create `components/pN_<name>.py` with a class subclassing `PipelineStep` (from `components.base`) and returning a dict from `output`.
2. Import it in `pipeline.py` and add it to `Registered_pipelines`.
3. Add tests in `tests/test_<name>.py`.
