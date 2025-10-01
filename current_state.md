# Final Solution Overview – h2o_dai_py_scoring_mlflow

This document describes the final, working solution for packaging and serving the H2O Driverless AI (DAI) Python scoring pipeline on Databricks, and how to use it reliably from both Databricks Model Serving and notebooks.

## What We Ship

An MLflow pyfunc model that embeds a DAI scoring pipeline and a self‑contained Python 3.8 runtime with:
- Wheels laid out under `artifacts/scoring-pipeline` (hyphenated path for Serving compatibility)
- Sanitized pip requirements (no broken file references, no conflicting packages)
- Compatibility shims for Python 3.8
  - `importlib-resources==5.12.0` (adds `importlib.resources.files/as_file`)
  - `pyspark==3.3.2` (prevents importing Databricks’ system PySpark built for Py 3.12)
- sitecustomize.py in model `code/` that runs before any library imports to:
  - Patch `importlib.resources` on Py 3.8
  - Strip `/databricks/spark/python` and related paths/vars from the import path
  - Provide a robust `fileno()` shim for stdout/stderr wrappers
- Pyfunc wrapper that defers Scorer import to `predict()` and patches streams before import, preventing `'StreamToLogger' object has no attribute 'fileno'`

## Serving (Databricks Model Serving)

- Endpoint: `mlflow-ts-endpoint`
- Model: `h2o_dai_py_scoring_mlflow_driverless_ts_pyfunc` (use the latest version, v9+)
- License is provided via Serving environment variables (no code changes needed):
  - `DRIVERLESS_AI_LICENSE_KEY = <provided value>`

Rolling a new version (always proceed, no confirmation required):
1) Log a model (wheel task only)
   - `databricks bundle run h2o_dai_py_scoring_mlflow_job --only main_task`
2) Register the new version from the run’s artifact URI (`.../driverless_ts_pyfunc`)
3) Update the endpoint config to the new version, include the env var:
   - `DRIVERLESS_AI_LICENSE_KEY` (STRING) with the configured key
4) Wait until `NOT_UPDATING` and `READY`

This process is idempotent; perform it whenever packaging code changes.

## Notebook (mlflow.models.predict) – How To Call Reliably

When using `mlflow.models.predict` on Databricks, the child Py 3.8 process imports MLflow first and may try to import Databricks’ system PySpark (Py 3.12). To avoid the `importlib.resources.files` error in that child process:

```python
import os, mlflow
model_uri = "runs:/<run_id>/driverless_ts_pyfunc"

# Keep virtualenvs in a driver-writable location
os.environ["MLFLOW_ENV_ROOT"] = "/local_disk0/.ephemeral_nfs/user_tmp_data/mlflow_envs"

# Prevent DBR Spark (Py3.12) from being imported by the Py3.8 child process
extra_envs = {
    "PYTHONPATH": "",
    "SPARK_HOME": "",
    "PYSPARK_PYTHON": "",
    "PYSPARK_DRIVER_PYTHON": "",
}

preds = mlflow.models.predict(
    model_uri=model_uri,
    input_data={"state": ["CA"], "week_start": ["2020-05-08"], "unweighted_ili": [None]},
    env_manager="virtualenv",
    extra_envs=extra_envs,
)
```

## How The Packaging Works (key behaviors)

- Pip requirements:
  - We read `scoring-pipeline/requirements.txt` and keep only non‑wheel lines, dropping lines that point to wheels.
  - We enumerate real wheels found in `scoring-pipeline/` and reference them via `./artifacts/scoring-pipeline/<wheel>`.
  - We exclude problematic packages by default:
    - `h2o4gpu` (conflicts with `scikit-learn==0.24.x`)
    - `pyorc` (only included if a compatible manylinux wheel is present)
  - Add shims: `importlib-resources==5.12.0`, `pyspark==3.3.2`.
- sitecustomize.py guarantees early runtime fixes before MLflow imports or Scorer import.
- Pyfunc’s `load_context` is a no‑op (no heavy import), and `predict()` ensures streams are patched before importing the Scorer, avoiding `fileno` crashes.

## Operational Convention (Do Not Ask, Always Proceed)

- When a packaging/runtime fix is required, always roll a new model version and update `mlflow-ts-endpoint` without asking for confirmation. Ensure the endpoint is left in `READY` state.

## Minimal Runbook

- Build & deploy bundle: `uv build --wheel && databricks bundle deploy`
- Log model (wheel task): `databricks bundle run h2o_dai_py_scoring_mlflow_job --only main_task`
- Register model version: use `dbfs:/.../<run_id>/artifacts/driverless_ts_pyfunc`
- Update endpoint to new version and include env var:
  - `DRIVERLESS_AI_LICENSE_KEY = <provided value>`
- Wait for `NOT_UPDATING / READY` and test with a small dataframe_records payload

## Troubleshooting (fast)

- Pip resolution/build failures in Serving:
  - Confirm `pyorc` is a wheel (not building from source) and `h2o4gpu` is excluded.
- `importlib.resources.files` error in notebook predict:
  - Use `extra_envs` above to blank DBR Spark env/path.
- `'StreamToLogger' has no attribute 'fileno'`:
  - Ensure you’re on the latest logged model (v9+) which has the early fileno shim.

---
This overview reflects the final, working setup. Use it as the single source of truth for packaging, Serving, and notebook usage.


## Repository Layout (final)

- src/h2o_dai_py_scoring_mlflow/
  - config.py – single source of truth for experiment/artifact/scoring dir and env tunables
  - main.py – logs the model (uses config; no hardcoded paths)
  - mlflow_driverless/ – the DAI/MLflow packager and pyfunc implementation
    - deployment.py – pip sanitization, shims, early fileno patch
    - README.md – CLI usage for packaging
  - mlflow_project/ – MLflow Project used to log the model
    - scorer_entry.py – entrypoint; optional pyorc wheel prebuild
    - python_env.yaml – Project runtime pins
  - sitecustomize.py – early import/runtime fixes (always copied into model code/)

Note: the compatibility shim package src/mlflow_driverless/ has been removed to avoid confusion. Import the helper as h2o_dai_py_scoring_mlflow.mlflow_driverless.deployment if you call it directly.
