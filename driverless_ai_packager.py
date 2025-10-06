# Databricks notebook source
"""
Driverless AI scoring pipeline → MLflow pyfunc logger (Databricks-only)

This notebook lets you:
- Import the helper package from a single ZIP file or from an unzipped `src/` folder (no local Python needed)
- Log the exported scoring pipeline as an MLflow pyfunc
- Optionally register a model version

Instructions (one-time per workspace):
1) Bring the helper package into your workspace:
   - import the ZIP into your workspace; Databricks will unzip to a folder that contains `src/`. In that case, use the path to that folder or its `src/` subfolder.
   - or clone the repo into Databricks Repos and use its `src/` folder path.
2) Place your Driverless scoring bundle (directory or .zip) in your workspace.
3) Fill the variables and run the notebook from top to bottom
"""

# COMMAND ----------

# EDIT THESE VALUES (first cell)
# These defaults assume you've imported/unzipped this repo into your workspace
# so that a `src/` folder is present next to this notebook and your scoring
# pipeline folder is at `./scoring-pipeline`.
import os

MODULE_PATH = "./src/"  # workspace path to this repo's src/
SCORING_DIR_PATH = (
    "./scoring-pipeline"  # workspace path to scoring-pipeline (dir or .zip)
)
EXPERIMENT_PATH = "/Shared/h2o_dai_py_scoring_mlflow"
ARTIFACT_PATH = "h2o_dai_scoring_pyfunc"
DRIVERLESS_AI_LICENSE_KEY = os.getenv("DRIVERLESS_AI_LICENSE_KEY") or ""
REGISTERED_MODEL_NAME = ""  # e.g., "catalog.schema.model_name" (leave empty to skip)

# COMMAND ----------

import os
import sys


def _dbfs_to_local(dbfs_path: str) -> str:
    if dbfs_path.startswith("dbfs:/"):
        return "/dbfs/" + dbfs_path[len("dbfs:/") :].lstrip("/")
    return dbfs_path


module_path = _dbfs_to_local(MODULE_PATH.strip())


def _resolve_module_import_path(path: str) -> str:
    # If a zip is provided, import directly from zip
    if path.endswith(".zip") and os.path.exists(path):
        return path
    # If a folder is provided, prefer its src/ child when present
    if os.path.isdir(path):
        src_candidate = os.path.join(path, "src")
        if os.path.isdir(src_candidate):
            return src_candidate
        return path
    raise FileNotFoundError(f"Module path does not exist: {path}")


module_import_path = _resolve_module_import_path(module_path)
if module_import_path not in sys.path:
    sys.path.insert(0, module_import_path)

print("Using module import path:", module_import_path)

# COMMAND ----------

import mlflow

from h2o_dai_py_scoring_mlflow.config import (
    get_artifact_path,
)
from h2o_dai_py_scoring_mlflow.mlflow_driverless.deployment import (
    log_driverless_scoring_pipeline,
)

# Configure env
os.environ["H2O_DAI_MLFLOW_FORCE_CONDA"] = "1"  # ensure libmagic is present
# Ensure MLflow uses a writable env root for creating envs during predict/logging
if not os.environ.get("MLFLOW_ENV_ROOT"):
    candidates = [
        "/databricks/driver/mlflow_envs",
        "/local_disk0/.ephemeral_nfs/user_tmp_data/mlflow_envs",
        "/local_disk0/tmp/mlflow_envs",
        f"/tmp/mlflow_envs_{os.getuid()}",
    ]
    for p in candidates:
        try:
            os.makedirs(p, exist_ok=True)
            # quick writability check
            test_path = os.path.join(p, ".__test__")
            with open(test_path, "w") as _:
                pass
            os.remove(test_path)
            os.environ["MLFLOW_ENV_ROOT"] = p
            break
        except Exception:
            continue

# Require a Driverless AI license key or file
_license_key = (
    DRIVERLESS_AI_LICENSE_KEY
    or os.getenv("DRIVERLESS_AI_LICENSE_KEY")
    or os.getenv("DRIVERLESS_AI_LICENSE")
    or ""
).strip()
_license_file = (os.getenv("DRIVERLESS_AI_LICENSE_FILE") or "").strip()
if not _license_key and not _license_file:
    raise RuntimeError(
        "Driverless AI license not found. Set DRIVERLESS_AI_LICENSE_KEY (or DRIVERLESS_AI_LICENSE) "
        "or provide DRIVERLESS_AI_LICENSE_FILE in the environment before running."
    )
if _license_key:
    os.environ["DRIVERLESS_AI_LICENSE_KEY"] = _license_key
if _license_file:
    os.environ["DRIVERLESS_AI_LICENSE_FILE"] = _license_file

artifact_path = (ARTIFACT_PATH or get_artifact_path()).strip()
experiment_path = EXPERIMENT_PATH.strip()
scoring_dir_path = SCORING_DIR_PATH.strip()

local_scoring_path = _dbfs_to_local(scoring_dir_path)
mlflow.set_experiment(experiment_path)

with mlflow.start_run() as active_run:
    info = log_driverless_scoring_pipeline(
        scoring_pipeline_dir=local_scoring_path,
        artifact_path=artifact_path,
        apply_data_recipes=False,
    )
    print("Run ID:", active_run.info.run_id)
    print("Model URI:", info.model_uri)
    # Retain variables for subsequent cells (prediction)

# COMMAND ----------

# (Optional) Register the model in the Workspace Model Registry
REGISTERED_MODEL_VERSION = None
if REGISTERED_MODEL_NAME.strip():
    print(f"Registering model as: {REGISTERED_MODEL_NAME}")
    try:
        result = mlflow.register_model(
            model_uri=info.model_uri, name=REGISTERED_MODEL_NAME.strip()
        )
        print("Registered version:", result.version)
        REGISTERED_MODEL_VERSION = str(result.version)
    except Exception as e:
        print("Model registration failed:", repr(e))
else:
    print("Skipping registration: set REGISTERED_MODEL_NAME to register.")

# COMMAND ----------


# Predict with mlflow.models.predict (virtualenv + extra_envs)
import os
import pandas as pd


# Ensure MLflow uses a writable env root for virtualenv environments
def _ensure_mlflow_env_root() -> str:
    candidates = [
        "/databricks/driver/mlflow_envs",
        "/local_disk0/.ephemeral_nfs/user_tmp_data/mlflow_envs",
        "/local_disk0/tmp/mlflow_envs",
        f"/tmp/mlflow_envs_{os.getuid()}",
    ]
    for p in candidates:
        try:
            os.makedirs(p, exist_ok=True)
            test_path = os.path.join(p, ".__test__")
            with open(test_path, "w") as _:
                pass
            os.remove(test_path)
            # Set both MLFLOW_ENV_ROOT (used by modern MLflow) and MLFLOW_HOME
            # (used by some older code paths) so envs land under a writable root.
            os.environ["MLFLOW_ENV_ROOT"] = p
            os.environ["MLFLOW_HOME"] = os.path.join(p, "mlflow")
            os.makedirs(os.environ["MLFLOW_HOME"], exist_ok=True)
            # Also influence tempdir resolution used by some MLflow/virtualenv paths
            os.environ["TMPDIR"] = p
            os.environ["TMP"] = p
            os.environ["TEMP"] = p
            # Some MLflow versions allow overriding the venv root explicitly
            os.environ["MLFLOW_VIRTUALENV_ROOT"] = os.path.join(p, "virtualenv_envs")
            # Ensure Python uses this temp dir even if tempfile cached earlier
            import tempfile as _temp

            _temp.tempdir = p
            return p
        except Exception:
            continue
    # Fall back to /tmp; may fail if not writable but we tried
    p = f"/tmp/mlflow_envs_{os.getuid()}"
    os.environ["MLFLOW_ENV_ROOT"] = p
    os.environ["MLFLOW_HOME"] = os.path.join(p, "mlflow")
    os.environ["TMPDIR"] = p
    os.environ["TMP"] = p
    os.environ["TEMP"] = p
    os.environ["MLFLOW_VIRTUALENV_ROOT"] = os.path.join(p, "virtualenv_envs")
    os.makedirs(os.environ["MLFLOW_HOME"], exist_ok=True)
    import tempfile as _temp

    _temp.tempdir = p
    return p


env_root = _ensure_mlflow_env_root()
import tempfile as _temp

print("Using MLFLOW_ENV_ROOT:", env_root, "tempfile.gettempdir():", _temp.gettempdir())

extra_envs = {
    "PYTHONPATH": "",
    "SPARK_HOME": "",
    "PYSPARK_PYTHON": "",
    "PYSPARK_DRIVER_PYTHON": "",
}

# Example input; adjust to your model's columns
input_df = pd.DataFrame(
    {
        "state": ["California", "California"],
        "week_start": ["2018-03-04", "2018-03-11"],
        "unweighted_ili": [None, None],
    }
)

# Save predictions to JSON file
mlflow.models.predict(
    model_uri=info.model_uri,
    input_data=input_df,
    output_path="preds.json",
    env_manager="virtualenv",
    extra_envs=extra_envs,
)
