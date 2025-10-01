"""Centralized configuration for mlflow_ts template.

All options are driven by environment variables with safe defaults. This allows
the same codebase to be used as a template without hardcoded paths.

Environment variables:
  - MLFLOW_TS_EXPERIMENT: MLflow experiment path (default: /Shared/mlflow_ts)
  - MLFLOW_TS_ARTIFACT_PATH: Artifact path for the logged model (default: driverless_ts_pyfunc)
  - SCORING_PIPELINE_DIR: Absolute path to the exported scoring-pipeline directory
                           (fallback: ./scoring-pipeline next to this package)

Driverless/packaging toggles (read in deployment/scorer_entry):
  - MLFLOW_DRIVERLESS_EXCLUDE_PACKAGES: comma-separated names to drop from pip
  - MLFLOW_DRIVERLESS_PYSPARK_VERSION: version string (default: 3.3.2)
  - MLFLOW_DRIVERLESS_DISABLE_PYSPARK: if "1", do not add pyspark shim
  - MLFLOW_DRIVERLESS_IMPORTLIB_RESOURCES_VERSION: version string (default: 5.12.0)
  - MLFLOW_DRIVERLESS_DISABLE_IMPORTLIB_RESOURCES: if "1", do not add backport
  - MLFLOW_DRIVERLESS_BUILD_PYORC: if "0", skip building pyorc wheel during logging (default enabled)
  - MLFLOW_DRIVERLESS_PYORC_VERSION: preferred pyorc version to build (default: 0.9.0)
"""

from __future__ import annotations

import os
from pathlib import Path


def get_experiment_path() -> str:
    return os.environ.get("MLFLOW_TS_EXPERIMENT", "/Shared/mlflow_ts").strip()


def get_artifact_path() -> str:
    return os.environ.get("MLFLOW_TS_ARTIFACT_PATH", "driverless_ts_pyfunc").strip()


def get_scoring_dir() -> str:
    env = os.environ.get("SCORING_PIPELINE_DIR")
    if env:
        return env
    # fallback: repo-relative ./scoring-pipeline
    here = Path(__file__).resolve().parent.parent.parent
    local = (here / "scoring-pipeline").resolve()
    if local.exists():
        return str(local)
    # Databricks workspace fallback (project-specific); ok as optional convenience
    ws_default = Path("/Workspace/Users/luiz.santos@h2o.ai/mlflow_proj/scoring-pipeline")
    if ws_default.exists():
        return str(ws_default)
    # Last resort: return the local path even if missing; downstream will raise
    return str(local)
