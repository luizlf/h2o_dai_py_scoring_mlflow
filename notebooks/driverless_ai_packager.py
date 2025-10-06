# Databricks notebook source
"""
Driverless AI scoring pipeline → MLflow pyfunc logger (Databricks-only)

This notebook lets you:
- Import the helper package from a single ZIP file or from an unzipped `src/` folder (no local Python needed)
- Log the exported scoring pipeline as an MLflow pyfunc
- Optionally register a model version
- Optionally update a Model Serving endpoint

Instructions (one-time per workspace):
1) Upload the helper package either:
   - as a ZIP to DBFS (e.g., dbfs:/FileStore/h2o_dai_py_scoring_mlflow.zip), OR
   - import the ZIP into your workspace; Databricks will unzip to a folder that contains `src/`. In that case, use the path to that folder or its `src/` subfolder.
2) Upload your Driverless scoring bundle to DBFS as either a directory or a .zip
   - Example: dbfs:/FileStore/scoring-pipeline.zip
3) Fill the widgets and run the notebook from top to bottom
"""

# COMMAND ----------

# EDIT THESE VALUES (first cell)
# These defaults assume you've imported/unzipped this repo into your workspace
# so that a `src/` folder is present next to this notebook and your scoring
# pipeline folder is at `./scoring-pipeline`.
import os
MODULE_PATH = "./src/"
SCORING_DIR_DBFS = "./scoring-pipeline"
EXPERIMENT_PATH = "/Shared/h2o_dai_py_scoring_mlflow"
ARTIFACT_PATH = "h2o_dai_scoring_pyfunc"
DRIVERLESS_AI_LICENSE_KEY = os.getenv("DRIVERLESS_AI_LICENSE_KEY") or ""
REGISTERED_MODEL_NAME = ""  # e.g., "h2o_dai_scoring_pyfunc_model" (leave empty to skip)

# COMMAND ----------

import os, sys
from typing import Any

def _dbfs_to_local(dbfs_path: str) -> str:
    if dbfs_path.startswith("dbfs:/"):
        return "/dbfs/" + dbfs_path[len("dbfs:/"):].lstrip("/")
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
_license_key = (DRIVERLESS_AI_LICENSE_KEY or os.getenv("DRIVERLESS_AI_LICENSE_KEY") or os.getenv("DRIVERLESS_AI_LICENSE") or "").strip()
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
scoring_dir_dbfs = SCORING_DIR_DBFS.strip()

local_scoring_path = _dbfs_to_local(scoring_dir_dbfs)
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
        result = mlflow.register_model(model_uri=info.model_uri, name=REGISTERED_MODEL_NAME.strip())
        print("Registered version:", result.version)
        REGISTERED_MODEL_VERSION = str(result.version)
    except Exception as e:
        print("Model registration failed:", repr(e))
else:
    print("Skipping registration: set REGISTERED_MODEL_NAME to register.")

# COMMAND ----------

# Wait for Deployment Job to finish updating the Serving endpoint, then score it
# Relies on WorkspaceClient so credentials are resolved using the active Databricks configuration
# (Unity Catalog credential guidance: https://learn.microsoft.com/azure/databricks/query-federation/http)
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist

target_model_name = REGISTERED_MODEL_NAME.strip()

if target_model_name:
    workspace = WorkspaceClient()
    serving_endpoint_name = target_model_name.replace('.', '-') + "-sr"

    target_version = REGISTERED_MODEL_VERSION
    if target_version is None:
        try:
            from mlflow.tracking import MlflowClient

            # Prefer UC registry, fallback to workspace registry
            try:
                client = MlflowClient(registry_uri="databricks-uc")
                client.get_registered_model(target_model_name)
            except Exception:
                client = MlflowClient()

            run_id = getattr(info, "run_id", None)
            for mv in client.search_model_versions(f"name='{target_model_name}'"):
                if run_id and getattr(mv, "run_id", None) == run_id:
                    target_version = str(mv.version)
                    break
            if target_version is None:
                versions = list(client.search_model_versions(f"name='{target_model_name}'"))
                if versions:
                    target_version = str(max(int(v.version) for v in versions))
        except Exception as exc:
            print("[wait] Unable to resolve registered version:", repr(exc))

    if target_version is None:
        print("[wait] Skipping endpoint wait/score because the registered version could not be determined.")
    else:
        timeout_s = int(os.environ.get("DEPLOY_WAIT_TIMEOUT_S", "900"))
        poll_s = int(os.environ.get("DEPLOY_WAIT_POLL_S", "10"))
        deadline = time.time() + timeout_s
        last_state = None
        serves_version = False
        ready_state = ""
        endpoint = None

        def _enum_name(v: object) -> str:
            try:
                # Databricks SDK enums expose .name
                return str(getattr(v, "name")).upper()
            except Exception:
                s = str(v) if v is not None else ""
                if "." in s:
                    s = s.rsplit(".", 1)[-1]
                return s.upper()

        while True:
            try:
                endpoint = workspace.serving_endpoints.get(serving_endpoint_name)
            except ResourceDoesNotExist:
                print(f"[wait] Serving endpoint '{serving_endpoint_name}' not found; skipping.")
                break

            served_entities = list(endpoint.config.served_entities or [])
            serves_version = any(
                getattr(entity, "entity_name", None) == target_model_name
                and str(getattr(entity, "entity_version", None)) == str(target_version)
                for entity in served_entities
            )

            ready_state = _enum_name(getattr(endpoint.state, "ready", ""))
            update_state = _enum_name(getattr(endpoint.state, "config_update", ""))
            state_tuple = (ready_state, update_state)
            if state_tuple != last_state:
                print(f"[wait] state={state_tuple}, serves_version={serves_version}")
                last_state = state_tuple

            # Break when endpoint serves the target version and is stable/ready
            if serves_version and ready_state == "READY" and update_state in {"NOT_UPDATING", ""}:
                print("[wait] Endpoint is updated and ready.")
                break

            if time.time() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for endpoint {serving_endpoint_name} to deploy {target_model_name} v{target_version}"
                )

            time.sleep(poll_s)

        if endpoint is not None and serves_version and ready_state == "READY":
            sample_records = [{"state": "CA", "week_start": "2020-05-08", "unweighted_ili": None}]
            try:
                response = workspace.serving_endpoints.query(
                    serving_endpoint_name,
                    dataframe_records=sample_records,
                )
                print("[score] predictions:", response.predictions)
            except Exception as exc:
                print("[score] Endpoint invocation failed:", repr(exc))
else:
    print("[wait] Skipping endpoint wait/score (REGISTERED_MODEL_NAME not provided).")

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
input_df = pd.DataFrame({
    "state": ["California", "California"],
    "week_start": ["2018-03-04", "2018-03-11"],
    "unweighted_ili": [None, None],
})

preds = mlflow.models.predict(
    model_uri=info.model_uri,
    input_data=input_df,
    env_manager="virtualenv",
    extra_envs=extra_envs,
)

display(preds if hasattr(preds, 'head') else preds)
