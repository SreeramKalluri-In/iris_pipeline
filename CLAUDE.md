# Iris Pipeline Project

Educational, end-to-end ML workflow for the classic iris classification task.
It walks through the full lifecycle in three independent scripts:
**train → serve → monitor**. Each stage is a standalone `python` file, not a
package; they link only through artifacts written to the repo root.

## Run commands

The venv is uv-managed (`.venv/pyvenv.cfg` shows `uv = 0.11.14`). Prefer `uv run`
so the interpreter/lockfile stay in sync:

```bash
uv run python pipeline.py      # Stage 1: train + evaluate, writes artifacts
uv run streamlit run app.py    # Stage 2: live web app (Streamlit, default :8501)
uv run python app_gradio.py    # Stage 2 alt: live web app (Gradio, listens :7860)
uv run python monitor.py       # Stage 3: Evidently drift + performance reports
```

Docker / Hugging Face Space (Stage 2 hosted):
```bash
docker build -t iris-space .
docker run --rm -p 7860:7860 iris-space   # app listens on 7860, not 8501
```

`app_gradio.py` is a Gradio port of `app.py` for HF Spaces on the free CPU
tier (zero GPU). Launch mode is detected automatically:
- Local: `uv run python app_gradio.py` serves `127.0.0.1:7860` and opens a
  browser; `--port 7861` for a different port; `--self-test` runs two
  sample predictions headlessly (no server) for a quick logic check.
- Space: when the `SPACE_ID` env var is present it binds `0.0.0.0:7860`
  instead.
Both app files load `iris_model.joblib` +
`training_reference.csv` at import time — keep them in sync if you change the
prediction or drift-panel logic. Gradio is pinned to `6.31.0` (pinned like
evidently — the API changes between major versions).

There is **no test suite** (no `pytest`/`unittest` setup, no `tests/` dir).
Verification = running the scripts and reading printed metrics / opening the HTML
reports.

## Workflow order & artifact flow

The scripts must run in order; later stages read what earlier ones write:

1. `pipeline.py` (train) → writes:
   - `iris_model.joblib` — `StandardScaler + LogisticRegression` pipeline
   - `training_reference.csv` — 100-row "known-good" baseline, cols = 4 features + `target`
   - `test_predictions.csv` — held-out test answers, cols = `target`, `predicted`
2. `app.py` (serve) → loads `iris_model.joblib`
3. `monitor.py` (monitor) → reads `training_reference.csv` + `test_predictions.csv`,
   writes `drift_report.html` and `performance_report.html`

Re-run `pipeline.py` after changing the model, then re-run `monitor.py` to
regenerate the reports. The two `*_report.html` files (~3 MB each) are generated
outputs — safe to delete/regenerate.

## Key details that matter

- **Feature order is load-bearing.** The model and `app.py` both rely on the exact
  4 feature columns in this order:
  `sepal length (cm)`, `sepal width (cm)`, `petal length (cm)`, `petal width (cm)`.
  `app.py` hardcodes this order in its `FEATURES` list; the CSVs rely on it too.
  Reorder a column and predictions silently go wrong.
- **Evidently is pinned to `0.7.23`** (in `pyproject.toml`), and `monitor.py`
  is written against the **0.7.x API**, which broke from 0.6:
  - `Report`, `Dataset`, `DataDefinition` are imported from the top-level
    `evidently` package (0.6 used `evidently.report.Report`).
  - Presets live in `evidently.presets`; the old `DataQualityPreset` was
    renamed **`DataSummaryPreset`** (drift checks are a separate
    `DataDriftPreset`).
  - `Report.run(current_data, reference_data)` — argument order is the
    **reverse of 0.6**, and it returns a `Snapshot`; you call
    `snapshot.save_html(...)` (not `report.save_html(...)`).
  - Column mapping uses `DataDefinition` (with `MulticlassClassification`)
    instead of 0.6's `ColumnMapping` object.
  - `evidently` only pins `pandas>=1.3.5`; the lock resolves to pandas
    **2.1.4**. Do not bump pandas to 3.x — evidently's legacy modules
    import `infer_dtype_from_object`, which 3.x removed.
- **model = a scikit-learn pipeline, not a bare estimator.** It's
  `make_pipeline(StandardScaler(), LogisticRegression(...))`, so
  `model.predict()` / `.predict_proba()` scale internally. Keep it as a
  pipeline when retraining; the app depends on that.

## Dependency management

Two parallel specs coexist (don't get confused):
- `pyproject.toml` + `uv.lock` — the canonical local/dev spec, used by `uv run`.
- `requirements.txt` — minimal set for the **HF Space** (Streamlit only).
- `requirements-monitoring.txt` — minimal set for the **monitoring** step.

`main.py` is just a leftover scaffold stub (`print("Hello...")`), not part of
the real pipeline.

## Repo hygiene

Untracked junk to be aware of (not covered by `.gitignore`): the generated
`*_report.html` files, and three empty stray files `0.20.0`, `1.0.0`, `1.28.0`
(accidentally-named version strings, 0 bytes — safe to delete). `.gitignore`
only excludes `.venv` and Python build artifacts, so generated outputs show up
as untracked in `git status`.
