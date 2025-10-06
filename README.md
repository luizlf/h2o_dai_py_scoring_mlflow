# h2o_dai_py_scoring_mlflow

Package and run an H2O Driverless AI (DAI) Python scoring pipeline as an MLflow `pyfunc` model that always runs under Python 3.8, even on Databricks clusters that default to newer Pythons. This README consolidates packaging and notebook inference (`mlflow.models.predict`).

Key outcomes:

- Logs a Driverless AI scoring pipeline as an MLflow `pyfunc` with a pinned Python 3.8 runtime and bundled wheels.
- Works with `mlflow.models.predict` reliably by adding compatibility shims and runtime patches.
- Provides a repeatable Databricks Bundle workflow to build, deploy, and run the logging job.

**What’s Included**

- MLflow helper module for packaging and logging the scoring pipeline: `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/deployment.py`.
- MLflow Project template that guarantees a Python 3.8 environment: `src/h2o_dai_py_scoring_mlflow/mlflow_project/`.
- Databricks Bundle job (single wheel task) to log the model: `resources/h2o_dai_py_scoring_mlflow.job.yml`.
- Early runtime fixes via `src/h2o_dai_py_scoring_mlflow/sitecustomize.py` to avoid import/runtime conflicts on Databricks.
- Workspace notebook/script to package + predict: `driverless_ai_packager.py`.

## Quick Start (on Databricks)

- Zip the contents of this repository (or add it to Databricks Repos) and import it into your workspace so you have a folder containing `src/`.
- Upload the exported `scoring-pipeline.zip` (or the `scoring-pipeline/` folder) to the same workspace location.
- Open and edit `driverless_ai_packager.py` in your workspace:
  - Set `MODULE_PATH` to the workspace path of this repo’s `src/` folder (for example, `./src/`).
  - Set `SCORING_DIR_PATH` to the workspace path of your `scoring-pipeline` (zip or folder).
  - Optionally set `REGISTERED_MODEL_NAME`.
- Run the notebook/script top to bottom. It logs the model and shows how to call `mlflow.models.predict`.

## Databricks‑Only Usage (No Local Python)

If your users can only run Python inside Databricks, bring these artifacts:

- A ZIP of this repository (or the unzipped folder imported into your workspace)
- `driverless_ai_packager.py` – a Databricks notebook/script that:
  - adds the zip to `sys.path`
  - logs the scoring pipeline as an MLflow `pyfunc`
  - optionally registers a model
  - optionally registers a model version

Steps (user):
- Zip this repository (or clone/import into Databricks Repos) and import it into your workspace so you have a folder that contains `src/`.
- Upload or place your Driverless scoring bundle (folder or `.zip`) in your workspace (for example, in the same folder as `src/`).
- Import `driverless_ai_packager.py` into your workspace and run it:
  - Edit the first cell variables (module path to ZIP or to the workspace folder with `src/`, scoring dir, experiment path, artifact path, license key).
  - Optionally set a Registered Model name.

Notes:
- The notebook forces conda by default so `libmagic` is included; no cluster‑level setup is needed.
- If you provide `registered_model_name`, the notebook registers a new version using the run’s model URI.
- References: `driverless_ai_packager.py:1`
 

## How It Works

The logger builds an MLflow `pyfunc` model that embeds your exported DAI scoring pipeline, along with a self‑contained Python 3.8 environment:

- Pip requirements are synthesized from the exported `requirements.txt` in the scoring bundle while ignoring wheel references there and instead enumerating the actual wheels present on disk. Default excludes avoid fragile packages for CPU scoring (e.g., `h2o4gpu`, `pyorc` unless a wheel is present).
- If present, selected `pip` entries from `scoring-pipeline/environment.yml` are merged into the model’s requirements to cover small runtime helpers that exports sometimes miss (default allowlist: `psutil`, `tabulate`, `cgroupspy`). Configure via `H2O_DAI_MLFLOW_ENV_YAML_PIP_ALLOWLIST` or disable with `H2O_DAI_MLFLOW_ENV_YAML_PIP_DISABLE=1`.
- Wheels are embedded under `artifacts/scoring-pipeline` (hyphenated path) so pip paths resolve consistently inside the model environment.
- Compatibility shims are added for Databricks compatibility under Python 3.8: `importlib-resources==5.12.0` and `pyspark==3.3.2` (configurable via env vars).
- A `sitecustomize.py` is automatically added at model load time so Python imports it before any library imports to:
  - Backport `importlib.resources.files/as_file` when needed on Python 3.8.
  - Strip Databricks runtime Spark paths from `sys.path` and neutralize Spark env vars so the model uses its own pip `pyspark` (not DBR’s).
  - Add a robust `fileno()` shim for the wrapped stdout/stderr that some native libs expect.
- The pyfunc wrapper defers importing the Driverless Scorer until `predict()` and patches streams before import, preventing `'StreamToLogger' object has no attribute 'fileno'`.
- During logging, the helper temporarily sets `MLFLOW_VALIDATE_SERVING_INPUT=false` to avoid strict validation issues when attaching input examples; the original value is restored afterwards.
- The generated conda env includes `libmagic` and hard‑pins `libopenblas=0.3.30` (override with `H2O_DAI_MLFLOW_OPENBLAS_VERSION`) plus `libgfortran`/`libgcc-ng` to stabilize BLAS loading at runtime.
- We force installing `psutil` from PyPI (pip) in the model environment to avoid the limited conda `psutil` variant that can miss `RLIM_INFINITY`. You can override the version with `H2O_DAI_MLFLOW_PSUTIL_VERSION`.
- Input/output examples and signature:
  - On MLflow >= 2.20.0, the PythonModel uses the `TypeFromExample` type hint and your input example for validation and signature inference. In this mode, we do not pass `signature` explicitly (per docs) and rely on hints + example.
  - On older MLflow, we attach an explicit `ModelSignature` derived from the scoring pipeline (typed columns parsed from `example.py`, fallback to `training_data_column_stats.json`). If you pass examples, output schema is inferred or falls back to the target column.

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
  - `mlflow_project/` – MLflow Project that enforces Python 3.8 for packaging and batch scoring.
    - `MLproject` – entry points `score` and `log_model`. See `src/h2o_dai_py_scoring_mlflow/mlflow_project/MLproject:1`.
    - `python_env.yaml` – Python 3.8.12 + dependencies used by the project. See `src/h2o_dai_py_scoring_mlflow/mlflow_project/python_env.yaml:1`.
    - `scorer_entry.py` – project entry script to score or log models. See `src/h2o_dai_py_scoring_mlflow/mlflow_project/scorer_entry.py:1`.
- Databricks Bundle job: `resources/h2o_dai_py_scoring_mlflow.job.yml:1`.
- Project packaging metadata: `pyproject.toml:1` (wheel entry point `main`).

## Requirements

- Logging/building (developer environment): Python 3.11+ to build the wheel for the bundle (`pyproject.toml` targets 3.11+), Databricks CLI, and access to a Databricks workspace.
- Inference (model runtime): Linux x86_64 with Python 3.8 environment managed by MLflow Projects/pyfunc. The helper pins `python==3.8.12` for the project, and the model environment is recreated via pip/conda specs embedded with the model.
- Driverless license: provide via `DRIVERLESS_AI_LICENSE_KEY` (or `DRIVERLESS_AI_LICENSE_FILE`) when scoring.

## Building and Deploying with Databricks Bundles

This repo is configured as a Databricks Bundle. The main job executes a single wheel task that logs the model.

1. Build the wheel and deploy the bundle:

- `uv build --wheel`
- `databricks bundle deploy`

2. Run the logging job:

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

See implementation details in: `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/deployment.py`.

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

## Notebook Inference with mlflow.models.predict

On Databricks, the child Python process used by `mlflow.models.predict` can import DBR’s system PySpark built for a different Python (3.12) before the model’s Python 3.8 env initializes. This template injects safe env variables into the model’s conda environment so you don’t need to pass `extra_envs`.

Use the conda env manager when predicting:

```python
import os, mlflow

run_id = "<run_id>"  # replace with your run ID
artifact_path = os.environ.get("H2O_DAI_MLFLOW_ARTIFACT_PATH", "h2o_dai_scoring_pyfunc")
model_uri = f"runs:/{run_id}/{artifact_path}"

preds = mlflow.models.predict(
    model_uri=model_uri,
    input_data={"state": ["CA"], "week_start": ["2020-05-08"], "unweighted_ili": [None]},
    env_manager="conda",
)
```

## Notebook Inference with mlflow.models.predict

On Databricks, the child Python process used by `mlflow.models.predict` can import DBR’s system PySpark built for a different Python (3.12) before the model’s Python 3.8 env initializes. This template injects safe env variables into the model’s conda environment so you don’t need to pass `extra_envs`.

Use the conda env manager when predicting:

```python
import os, mlflow

run_id = "<run_id>"  # replace with your run ID
artifact_path = os.environ.get("H2O_DAI_MLFLOW_ARTIFACT_PATH", "h2o_dai_scoring_pyfunc")
model_uri = f"runs:/{run_id}/{artifact_path}"

preds = mlflow.models.predict(
    model_uri=model_uri,
    input_data={"state": ["CA"], "week_start": ["2020-05-08"], "unweighted_ili": [None]},
    env_manager="conda",
)
```

## Configuration (Environment Variables)

Primary paths and defaults (see `src/h2o_dai_py_scoring_mlflow/config.py:1`):

- `H2O_DAI_MLFLOW_EXPERIMENT` – MLflow experiment path (default `/Shared/h2o_dai_py_scoring_mlflow`).
- `H2O_DAI_MLFLOW_ARTIFACT_PATH` – artifact path for the logged model (default `h2o_dai_scoring_pyfunc`).
- `SCORING_PIPELINE_DIR` – absolute path to the exported `scoring-pipeline` directory or `.zip`.
- `H2O_DAI_MLFLOW_WORKSPACE_SCORING_DIR` – optional workspace path to the scoring‑pipeline when you don’t want to set `SCORING_PIPELINE_DIR`.

Packaging toggles (consumed by `mlflow_driverless`):

- `H2O_DAI_MLFLOW_EXCLUDE_PACKAGES` – comma‑separated package names to exclude from `pip_requirements` (defaults include `h2o4gpu`, `pyorc`). See `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/deployment.py:360`.
- `H2O_DAI_MLFLOW_IMPORTLIB_RESOURCES_VERSION` – version for `importlib-resources` backport (default `5.12.0`).
- `H2O_DAI_MLFLOW_DISABLE_IMPORTLIB_RESOURCES` – set to `1` to skip adding the backport.
- `H2O_DAI_MLFLOW_PYSPARK_VERSION` – version for `pyspark` shim (default `3.3.2`).
- `H2O_DAI_MLFLOW_DISABLE_PYSPARK` – set to `1` to skip adding `pyspark`.
- `H2O_DAI_MLFLOW_FORCE_CONDA` – defaults to enabled; the logger synthesizes a conda `conda_env` so non‑pip deps (e.g., `libmagic`) are installed.
- `H2O_DAI_MLFLOW_DISABLE_LIBMAGIC` – set to `1` to skip adding `libmagic` to the conda env (not recommended; required by `python-magic`).
- `H2O_DAI_MLFLOW_DISABLE_OPENBLAS` – set to `1` to skip adding `libopenblas` to the conda env.
- `H2O_DAI_MLFLOW_OPENBLAS_VERSION` – version pin for `libopenblas` (default `0.3.30`).
- `H2O_DAI_MLFLOW_PSUTIL_VERSION` – optional version pin for `psutil` (PyPI) installed via pip to override conda’s psutil.
- `H2O_DAI_MLFLOW_DISABLE_PROJECT` / `H2O_DAI_MLFLOW_FORCE_PROJECT` – control whether to force or skip launching the MLflow Project when not on Python 3.8. See `src/h2o_dai_py_scoring_mlflow/mlflow_driverless/deployment.py:505`.
- `H2O_DAI_MLFLOW_PROJECT_MODE` – internal flag set when already inside the project‑managed environment.
- `H2O_DAI_MLFLOW_BUILD_PYORC` – set `0` to skip building a `pyorc` wheel in project mode; default builds a wheel to avoid source builds in production runtimes. See `src/h2o_dai_py_scoring_mlflow/mlflow_project/scorer_entry.py:418`.
- `H2O_DAI_MLFLOW_PYORC_VERSION` – preferred `pyorc` version to build (default `0.9.0`).
 

Runtime:

- `DRIVERLESS_AI_LICENSE_KEY` / `DRIVERLESS_AI_LICENSE_FILE` – required by the scoring pipeline at runtime.
- `MLFLOW_ENV_ROOT` – optional override for where MLflow creates per‑model environments (use a writable path on DBR). Not required when using conda + injected variables, but can be helpful to control disk usage.

## Minimal Runbook

- Build + deploy bundle: `uv build --wheel && databricks bundle deploy`
- Log a model (wheel task only): `databricks bundle run h2o_dai_py_scoring_mlflow_job --only main_task`
- Register a model version from the run URI (e.g., `runs:/<run_id>/<artifact_path>`). Use the artifact path configured via `H2O_DAI_MLFLOW_ARTIFACT_PATH`.
- Predict from a notebook with `mlflow.models.predict` as shown above.

## Troubleshooting

- Pip resolution or native build failures when creating the model environment or importing the scorer:
  - Ensure `pyorc` is provided as a wheel (not built from source) and `h2o4gpu` is excluded unless needed.
- `importlib.resources.files` errors when calling from notebooks:
  - Use the `mlflow.models.predict` pattern above to blank the DBR Spark env and set `MLFLOW_ENV_ROOT`.
- `StreamToLogger` object has no attribute `fileno`:
  - Use a recently logged model; packaging includes early stream shims in both project and model environments.
- Native libs not found:
  - The logged model’s conda env includes `libopenblas` and `libmagic` by default. The model’s runtime injects the environment’s `lib` dir into `LD_LIBRARY_PATH` and prints diagnostics at load time.
  - Set `H2O_DAI_MLFLOW_DIAG=1` (default) to emit diagnostics when loading/predicting (e.g., `find_library('openblas')`).
- AttributeError: module 'psutil' has no attribute 'RLIM_INFINITY':
  - Newer `psutil` versions don’t expose this constant; the model’s `sitecustomize.py` shims it from Python’s `resource` module. Re‑log to pick up the shim if you see this.
  - As an additional safeguard, if any `psutil` RLIMIT constants are missing (`RLIMIT_*`, `RLIM_INFINITY`), the model defines them as `None` at import time to maintain compatibility with older Driverless code paths.

## Notes and Limitations

- Linux x86_64 only for actual scoring; macOS/arm64 can log the model but not execute the scorer binaries.
- The `tests/` folder contains legacy examples not wired to the current entry point and can be ignored.

## License and Attribution

This repository provides packaging glue to run H2O Driverless AI scoring pipelines under MLflow/Databricks. A valid Driverless AI license is required to run the scorer.
