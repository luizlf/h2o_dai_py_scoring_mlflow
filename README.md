# h2o_dai_py_scoring_mlflow

Package and serve an H2O Driverless AI (DAI) Python scoring pipeline as an MLflow `pyfunc` model that always runs under Python 3.8, even on Databricks clusters that default to newer Pythons. This README consolidates the current implementation and usage across packaging, Databricks Bundle deployment, Model Serving, and notebook inference.

Key outcomes:

- Logs a Driverless AI scoring pipeline as an MLflow `pyfunc` with a pinned Python 3.8 runtime and bundled wheels.
- Works with Databricks Model Serving and `mlflow.models.predict` reliably by adding compatibility shims and runtime patches.
- Provides a repeatable Databricks Bundle workflow to build, deploy, and run the logging job.

**What’s Included**

- MLflow helper module and CLI for packaging and logging the scoring pipeline: `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/deployment.py`.
- MLflow Project template that guarantees a Python 3.8 environment: `src/h2o_dai_py_scoring_mlflow/mlflow_project/`.
- Databricks Bundle job that runs the end‑to‑end logging flow: `resources/h2o_dai_py_scoring_mlflow.job.yml`.
- Early runtime fixes via `src/h2o_dai_py_scoring_mlflow/sitecustomize.py` to avoid import/runtime conflicts on Databricks.

## How It Works

The logger builds an MLflow `pyfunc` model that embeds your exported DAI scoring pipeline, along with a self‑contained Python 3.8 environment:

- Pip requirements are synthesized from the exported `requirements.txt` in the scoring bundle while ignoring wheel references there and instead enumerating the actual wheels present on disk. Default excludes avoid fragile packages for CPU scoring (e.g., `h2o4gpu`, `pyorc` unless a wheel is present).
- Wheels are embedded under `artifacts/scoring-pipeline` (hyphenated path) so pip paths resolve consistently inside Serving containers.
- Compatibility shims are added for Databricks compatibility under Python 3.8: `importlib-resources==5.12.0` and `pyspark==3.3.2` (configurable via env vars).
- A `sitecustomize.py` is automatically added at model load time so Python imports it before any library imports to:
  - Backport `importlib.resources.files/as_file` when needed on Python 3.8.
  - Strip Databricks runtime Spark paths from `sys.path` and neutralize Spark env vars so the model uses its own pip `pyspark` (not DBR’s).
  - Add a robust `fileno()` shim for the wrapped stdout/stderr that some native libs expect.
- The pyfunc wrapper defers importing the Driverless Scorer until `predict()` and patches streams before import, preventing `'StreamToLogger' object has no attribute 'fileno'`.
- During logging, the helper temporarily sets `MLFLOW_VALIDATE_SERVING_INPUT=false` to avoid strict validation issues when attaching input examples; the original value is restored afterwards. See `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/deployment.py:798`.
- Input/output examples and signature:
  - Input schema is derived from the scoring pipeline itself (typed columns parsed from `example.py`), falling back to `training_data_column_stats.json`, and finally to any provided input example.
  - If you pass example dataframes/paths, they are logged; output schema is inferred when an output example is provided, otherwise the target column is inferred from the experiment summary (fallback: a numeric `prediction`).
  - This explicit signature enables MLflow input validation without relying on function type hints.

Under the hood the logger will either:

- Log directly when running on Python 3.8; or
- Spawn an MLflow Project run that pins Python 3.8 and then logs from there when running on any other Python version.

Relevant code:

- Model logging and env building: `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/deployment.py:280`.
- Python env and conda env builders: `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/deployment.py:380`, `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/deployment.py:403`.
- Project template with entry points: `src/h2o_dai_py_scoring_mlflow/mlflow_project/MLproject:1`.
- Project’s Python 3.8 runtime pins: `src/h2o_dai_py_scoring_mlflow/mlflow_project/python_env.yaml:1`.
- Early runtime fixes: `src/h2o_dai_py_scoring_mlflow/sitecustomize.py:1`.

## Repository Layout

- `src/h2o_dai_py_scoring_mlflow/`
  - `config.py` – centralizes paths and env flags (experiment path, artifact path, pipeline dir). See `src/h2o_dai_py_scoring_mlflow/config.py:1`.
  - `main.py` – wheel entry point that logs the model using the helper. See `src/h2o_dai_py_scoring_mlflow/main.py:1`.
  - `sitecustomize.py` – runtime shims loaded automatically inside model env. See `src/h2o_dai_py_scoring_mlflow/sitecustomize.py:1`.
  - `mlflow_driverless/`
    - `deployment.py` – all packaging logic, CLI, and pyfunc model wrapper. See `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/deployment.py:1`.
    - `README.md` – extra CLI usage notes. See `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/README.md:1`.
  - `mlflow_project/` – MLflow Project that enforces Python 3.8 for packaging and batch scoring.
    - `MLproject` – entry points `score` and `log_model`. See `src/h2o_dai_py_scoring_mlflow/mlflow_project/MLproject:1`.
    - `python_env.yaml` – Python 3.8.12 + dependencies used by the project. See `src/h2o_dai_py_scoring_mlflow/mlflow_project/python_env.yaml:1`.
    - `scorer_entry.py` – project entry script to score or log models. See `src/h2o_dai_py_scoring_mlflow/mlflow_project/scorer_entry.py:1`.
- Databricks Bundle job: `resources/h2o_dai_py_scoring_mlflow.job.yml:1`.
- Project packaging metadata: `pyproject.toml:1` (wheel entry point `main`).

## Requirements

- Logging/building (developer environment): Python 3.11+ to build the wheel for the bundle (`pyproject.toml` targets 3.11+), Databricks CLI, and access to a Databricks workspace.
- Serving/inference (model runtime): Linux x86_64 with Python 3.8 environment managed by MLflow Projects/pyfunc. The helper pins `python==3.8.12` for the project, and the model environment is recreated via pip/conda specs embedded with the model.
- Driverless license: provide via `DRIVERLESS_AI_LICENSE_KEY` (or `DRIVERLESS_AI_LICENSE_FILE`) when serving or scoring.

## Building and Deploying with Databricks Bundles

This repo is configured as a Databricks Bundle. The main job does two tasks: run a notebook (optional) and then execute the wheel entry point that logs the model.

1. Build the wheel and deploy the bundle:

- `uv build --wheel`
- `databricks bundle deploy`

2. Run the logging job (wheel task only):

- `databricks bundle run h2o_dai_py_scoring_mlflow_job --only main_task`

The wheel entry point calls `h2o_dai_py_scoring_mlflow.main:main` (see `pyproject.toml:37`), which resolves the scoring pipeline path and logs the model using the `mlflow_driverless` helper.

Where to place the exported scoring pipeline:

- Preferred: set `SCORING_PIPELINE_DIR` to the absolute directory or `.zip` bundle.
- Fallbacks: `src/h2o_dai_py_scoring_mlflow/config.py` tries `./scoring-pipeline` next to the repo, or `/Workspace/Users/luiz.santos@h2o.ai/mlflow_proj/scoring-pipeline` if present.

Logged model artifact default path: `h2o_dai_scoring_pyfunc` (configurable via `H2O_DAI_MLFLOW_ARTIFACT_PATH`). See `src/h2o_dai_py_scoring_mlflow/config.py:14`.

## Alternative: CLI Logging (no bundle)

From any Python where `mlflow` is available and you have the exported scoring bundle:

- `python -m h2o_dai_py_scoring_mlflow.mlflow_driverless.deployment /path/to/scoring-pipeline --artifact-path "$H2O_DAI_MLFLOW_ARTIFACT_PATH"`

Notes:

- If not on Python 3.8, the helper will automatically spawn an MLflow Project run that pins Python 3.8 to perform the logging.
- You can also pass a `.zip` scoring export; it will be extracted to a temp folder automatically.

See the module doc for details: `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/README.md:1`.

## Batch Scoring via MLflow Project

You can run ad‑hoc batch scoring with the project’s `score` entry point. This guarantees Python 3.8 and installs bundle wheels inside the project environment:

- From Python (inside Databricks or locally on Linux):

```python
import mlflow
mlflow.projects.run(
    uri="src/h2o_dai_py_scoring_mlflow/mlflow_project",
    entry_point="score",
    parameters={
        "scoring_dir": "scoring-pipeline",  # or absolute path / zip
        "input_path": "/dbfs/path/to/input.csv",
        "output_path": "/dbfs/path/to/output.csv",
        "apply_data_recipes": "false",
        "batch_size": "0",
    },
    env_manager="virtualenv",
)
```

Entry point details: `src/h2o_dai_py_scoring_mlflow/mlflow_project/MLproject:1`, script `src/h2o_dai_py_scoring_mlflow/mlflow_project/scorer_entry.py:1`.

## Model Serving on Databricks

- Configure your Model Serving endpoint to reference the model version you register.
- Environment variables required on the endpoint:
  - `DRIVERLESS_AI_LICENSE_KEY = <provided value>`

Rolling a new version (idempotent):

1. Run the wheel task in the bundle to log a new model version.
2. Register a new version from the run’s artifact URI (e.g., `runs:/<run_id>/<artifact_path>`). Use the artifact path configured via `H2O_DAI_MLFLOW_ARTIFACT_PATH`.
3. Update your serving endpoint to the new version and set `DRIVERLESS_AI_LICENSE_KEY`.
4. Wait until the endpoint is `READY`.

Operational convention: this process is safe and idempotent. When packaging code changes, always roll a new version and update your serving endpoint without waiting for additional confirmation; leave the endpoint in `READY` state.

## Notebook Inference with mlflow.models.predict

On Databricks, the child Python process used by `mlflow.models.predict` can import DBR’s system PySpark built for a different Python (3.12) before the model’s Python 3.8 env initializes. Use the following pattern to avoid import issues and store virtualenvs in a driver‑writable path:

```python
import os, mlflow

run_id = "<run_id>"  # replace with your run ID
artifact_path = os.environ.get("H2O_DAI_MLFLOW_ARTIFACT_PATH", "h2o_dai_scoring_pyfunc")
model_uri = f"runs:/{run_id}/{artifact_path}"

os.environ["MLFLOW_ENV_ROOT"] = "/local_disk0/.ephemeral_nfs/user_tmp_data/mlflow_envs"

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

## Configuration (Environment Variables)

Primary paths and defaults (see `src/h2o_dai_py_scoring_mlflow/config.py:1`):

- `H2O_DAI_MLFLOW_EXPERIMENT` – MLflow experiment path (default `/Shared/h2o_dai_py_scoring_mlflow`).
- `H2O_DAI_MLFLOW_ARTIFACT_PATH` – artifact path for the logged model (default `h2o_dai_scoring_pyfunc`).
- `SCORING_PIPELINE_DIR` – absolute path to the exported `scoring-pipeline` directory or `.zip`.

Packaging toggles (consumed by `mlflow_driverless`):

- `H2O_DAI_MLFLOW_EXCLUDE_PACKAGES` – comma‑separated package names to exclude from `pip_requirements` (defaults include `h2o4gpu`, `pyorc`). See `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/deployment.py:360`.
- `H2O_DAI_MLFLOW_IMPORTLIB_RESOURCES_VERSION` – version for `importlib-resources` backport (default `5.12.0`).
- `H2O_DAI_MLFLOW_DISABLE_IMPORTLIB_RESOURCES` – set to `1` to skip adding the backport.
- `H2O_DAI_MLFLOW_PYSPARK_VERSION` – version for `pyspark` shim (default `3.3.2`).
- `H2O_DAI_MLFLOW_DISABLE_PYSPARK` – set to `1` to skip adding `pyspark`.
- `H2O_DAI_MLFLOW_FORCE_CONDA` – defaults to enabled; the logger synthesizes a conda `conda_env` so non‑pip deps (e.g., `libmagic`) are installed in Serving. Set to `0` to disable if you must use a pip‑only env. See `src/h2o_dai_py_scoring_mlflow/config.py:64`.
- `H2O_DAI_MLFLOW_DISABLE_LIBMAGIC` – set to `1` to skip adding `libmagic` to the conda env (not recommended; required by `python-magic`).
- `H2O_DAI_MLFLOW_DISABLE_PROJECT` / `H2O_DAI_MLFLOW_FORCE_PROJECT` – control whether to force or skip launching the MLflow Project when not on Python 3.8. See `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/deployment.py:505`.
- `H2O_DAI_MLFLOW_PROJECT_MODE` – internal flag set when already inside the project‑managed environment.
- `H2O_DAI_MLFLOW_BUILD_PYORC` – set `0` to skip building a `pyorc` wheel in project mode; default builds a wheel to avoid source builds in Serving. See `src/h2o_dai_py_scoring_mlflow/mlflow_project/scorer_entry.py:418`.
- `H2O_DAI_MLFLOW_PYORC_VERSION` – preferred `pyorc` version to build (default `0.9.0`).

Runtime:

- `DRIVERLESS_AI_LICENSE_KEY` / `DRIVERLESS_AI_LICENSE_FILE` – required by the scoring pipeline at runtime.
- `MLFLOW_ENV_ROOT` – override where MLflow creates per‑model virtualenvs (use a writable path on DBR).

## Minimal Runbook

- Build + deploy bundle: `uv build --wheel && databricks bundle deploy`
- Log a model (wheel task only): `databricks bundle run h2o_dai_py_scoring_mlflow_job --only main_task`
- Register a model version from the run URI (e.g., `runs:/<run_id>/<artifact_path>`). Use the artifact path configured via `H2O_DAI_MLFLOW_ARTIFACT_PATH`.
- Update your Databricks Model Serving endpoint to point to the new version and set `DRIVERLESS_AI_LICENSE_KEY`.
- Wait for `READY`, then test with a small `dataframe_records` payload.

## Troubleshooting

- Pip resolution or native build failures during Serving:
  - Ensure `pyorc` is provided as a wheel (not built from source) and `h2o4gpu` is excluded unless needed.
- `importlib.resources.files` errors when calling from notebooks:
  - Use the `mlflow.models.predict` pattern above to blank the DBR Spark env and set `MLFLOW_ENV_ROOT`.
- `StreamToLogger` object has no attribute `fileno`:
  - Use a recently logged model; packaging includes early stream shims in both project and model environments.

## Notes and Limitations

- Linux x86_64 only for actual scoring; macOS/arm64 can log the model but not execute the scorer binaries.
- The `tests/` folder contains legacy examples not wired to the current entry point and can be ignored.

## License and Attribution

This repository provides packaging glue to run H2O Driverless AI scoring pipelines under MLflow/Databricks. A valid Driverless AI license is required to run the scorer.
