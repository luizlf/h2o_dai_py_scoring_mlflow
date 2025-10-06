# h2o_dai_py_scoring_mlflow

Package and run an H2O Driverless AI (DAI) Python scoring pipeline as an MLflow pyfunc model under Python 3.8 on Databricks. The model is logged with a self-contained environment so you can score with mlflow.models.predict.

Tested on
- Databricks ML Runtime 16.4
- H2O Driverless AI 1.10.7.3

## Quick Start (on Databricks)

- Zip this repository (or import it into Databricks Repos) and place it in your workspace so you have a folder containing `src/`.
- Put your exported `scoring-pipeline.zip` (or the `scoring-pipeline/` folder) in the same workspace location.
- Open `driverless_ai_packager.py` in your workspace and set:
  - `MODULE_PATH` → path to this repo’s `src/` folder (for example, `./src/`).
  - `SCORING_DIR_PATH` → path to your `scoring-pipeline` (zip or folder).
  - `REGISTERED_MODEL_NAME` (optional).
- Run the cells top to bottom. It logs the model and shows how to call `mlflow.models.predict`.

## Predict (conda env)

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
print(preds)
```

## Configuration

Minimum
- `DRIVERLESS_AI_LICENSE_KEY` or `DRIVERLESS_AI_LICENSE_FILE` – required by the scoring pipeline at runtime.
- `SCORING_PIPELINE_DIR` – absolute path to `scoring-pipeline` (zip or folder). Alternative: `H2O_DAI_MLFLOW_WORKSPACE_SCORING_DIR`.

Optional
- `H2O_DAI_MLFLOW_EXPERIMENT` – MLflow experiment path (default `/Shared/h2o_dai_py_scoring_mlflow`).
- `H2O_DAI_MLFLOW_ARTIFACT_PATH` – artifact path for the logged model (default `h2o_dai_scoring_pyfunc`).

## What this project does for you

- Packages the Driverless AI scoring pipeline as an MLflow pyfunc with Python 3.8.
- Synthesizes pip requirements from the export and embeds wheels under `artifacts/scoring-pipeline`.
- Adds small compatibility shims for Python 3.8 and Databricks (importlib-resources backport, sitecustomize, stream fileno shim).
- Uses a conda env by default so required OS libs (libmagic, openblas) are available at runtime.

## Troubleshooting (quick)

- License missing: set `DRIVERLESS_AI_LICENSE_KEY` or `DRIVERLESS_AI_LICENSE_FILE`.
- Use `env_manager="conda"` for `mlflow.models.predict`.
- If predicting in a shared cluster, set `MLFLOW_ENV_ROOT` to a writable location.

## Repo contents

- `src/h2o_dai_py_scoring_mlflow/` – library code, MLflow Project, and runtime shims.
- `driverless_ai_packager.py` – workspace notebook/script to package + predict.
- `resources/h2o_dai_py_scoring_mlflow.job.yml` – Databricks Bundle job (wheel task) to log models.
