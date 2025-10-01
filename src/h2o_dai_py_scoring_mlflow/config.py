"""Centralized configuration for h2o_dai_py_scoring_mlflow template.

All options are driven by environment variables with safe defaults. This allows
the same codebase to be used as a template without hardcoded paths.

Environment variables:
  - H2O_DAI_MLFLOW_EXPERIMENT: MLflow experiment path (default: /Shared/h2o_dai_py_scoring_mlflow)
  - H2O_DAI_MLFLOW_ARTIFACT_PATH: Artifact path for the logged model (default: h2o_dai_scoring_pyfunc)
  - SCORING_PIPELINE_DIR: Absolute path to the exported scoring-pipeline directory
                           (fallback: ./scoring-pipeline next to this package)

Driverless/packaging toggles (read in deployment/scorer_entry):
  - H2O_DAI_MLFLOW_EXCLUDE_PACKAGES: comma-separated names to drop from pip (default: h2o4gpu,pyorc)
  - H2O_DAI_MLFLOW_PYSPARK_VERSION: version string (default: 3.3.2)
  - H2O_DAI_MLFLOW_DISABLE_PYSPARK: if "1", do not add pyspark shim
  - H2O_DAI_MLFLOW_IMPORTLIB_RESOURCES_VERSION: version string (default: 5.12.0)
  - H2O_DAI_MLFLOW_DISABLE_IMPORTLIB_RESOURCES: if "1", do not add backport
  - H2O_DAI_MLFLOW_BUILD_PYORC: if "0", skip building pyorc wheel during logging (default enabled)
  - H2O_DAI_MLFLOW_PYORC_VERSION: preferred pyorc version to build (default: 0.9.0)
  - H2O_DAI_MLFLOW_DISABLE_LIBMAGIC: if "1", do not add libmagic to conda env
  - H2O_DAI_MLFLOW_FORCE_CONDA: if "1", synthesize a conda env even when pip env would be sufficient
  - H2O_DAI_MLFLOW_DISABLE_PROJECT: if "1", never launch MLflow Project (even on non-3.8)
  - H2O_DAI_MLFLOW_FORCE_PROJECT: if "1", always launch MLflow Project
  - H2O_DAI_MLFLOW_PROJECT_MODE: internal flag set to "1" when running inside the MLflow Project
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


# ========= Defaults (single source of truth) =========

EXPERIMENT_DEFAULT = "/Shared/h2o_dai_py_scoring_mlflow"
ARTIFACT_PATH_DEFAULT = "h2o_dai_scoring_pyfunc"

# Model runtime Python
REQUIRED_PYTHON_VERSION: Tuple[int, int] = (3, 8)
DEFAULT_PYTHON_VERSION = "3.8.12"

# Compatibility shims
IMPORTLIB_RESOURCES_VERSION_DEFAULT = "5.12.0"
PYSPARK_VERSION_DEFAULT = "3.3.2"

# Packaging behavior
EXCLUDED_PACKAGES_DEFAULT: Sequence[str] = ("h2o4gpu", "pyorc")
BUILD_PYORC_DEFAULT = True
PYORC_VERSION_DEFAULT = "0.9.0"
DISABLE_LIBMAGIC_DEFAULT = False
FORCE_CONDA_DEFAULT = True

# Project launch behavior
DISABLE_PROJECT_DEFAULT = False
FORCE_PROJECT_DEFAULT = False
PROJECT_MODE_ENV = "H2O_DAI_MLFLOW_PROJECT_MODE"

# Scoring pipeline env var aliases (for convenience discovery)
SCORING_ENV_VAR_ALIASES: Sequence[str] = (
    "SCORING_PIPELINE_DIR",
    "DRIVERLESS_SCORING_PIPELINE_DIR",
    "DRIVERLESS_SCORING_DIR",
    "DRIVERLESS_AI_SCORING_PIPELINE_DIR",
)


def get_experiment_path() -> str:
    return os.environ.get("H2O_DAI_MLFLOW_EXPERIMENT", EXPERIMENT_DEFAULT).strip()


def get_artifact_path() -> str:
    return os.environ.get("H2O_DAI_MLFLOW_ARTIFACT_PATH", ARTIFACT_PATH_DEFAULT).strip()


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


# ========= Derived getters for packaging/runtime =========

def _dedup_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def get_excluded_packages() -> List[str]:
    raw = os.environ.get("H2O_DAI_MLFLOW_EXCLUDE_PACKAGES", "").strip()
    extras = [p.strip() for p in raw.split(",") if p.strip()]
    return _dedup_preserve_order(list(EXCLUDED_PACKAGES_DEFAULT) + extras)


def is_importlib_resources_disabled() -> bool:
    return os.environ.get("H2O_DAI_MLFLOW_DISABLE_IMPORTLIB_RESOURCES", "0").strip() == "1"


def get_importlib_resources_version() -> str:
    return os.environ.get(
        "H2O_DAI_MLFLOW_IMPORTLIB_RESOURCES_VERSION",
        IMPORTLIB_RESOURCES_VERSION_DEFAULT,
    ).strip()


def is_pyspark_disabled() -> bool:
    return os.environ.get("H2O_DAI_MLFLOW_DISABLE_PYSPARK", "0").strip() == "1"


def get_pyspark_version() -> str:
    return os.environ.get("H2O_DAI_MLFLOW_PYSPARK_VERSION", PYSPARK_VERSION_DEFAULT).strip()


def is_libmagic_disabled() -> bool:
    return os.environ.get("H2O_DAI_MLFLOW_DISABLE_LIBMAGIC", "0").strip() == "1"


def is_conda_forced() -> bool:
    # Default to forcing conda to ensure non-pip deps like libmagic are present
    return os.environ.get("H2O_DAI_MLFLOW_FORCE_CONDA", "1" if FORCE_CONDA_DEFAULT else "0").strip() == "1"


def is_project_disabled() -> bool:
    return os.environ.get("H2O_DAI_MLFLOW_DISABLE_PROJECT", "0").strip() == "1"


def is_project_forced() -> bool:
    return os.environ.get("H2O_DAI_MLFLOW_FORCE_PROJECT", "0").strip() == "1"


def is_project_mode() -> bool:
    return os.environ.get(PROJECT_MODE_ENV) == "1"


def get_default_python_version() -> str:
    return DEFAULT_PYTHON_VERSION


def get_required_python_version() -> Tuple[int, int]:
    return REQUIRED_PYTHON_VERSION


def get_scoring_env_var_aliases() -> Sequence[str]:
    return SCORING_ENV_VAR_ALIASES


def get_build_pyorc_enabled() -> bool:
    # Enabled unless explicitly disabled via env
    return os.environ.get("H2O_DAI_MLFLOW_BUILD_PYORC", "1").strip() != "0"


def get_pyorc_version() -> str:
    return os.environ.get("H2O_DAI_MLFLOW_PYORC_VERSION", PYORC_VERSION_DEFAULT).strip()
