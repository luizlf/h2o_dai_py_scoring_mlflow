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
  - H2O_DAI_MLFLOW_DISABLE_OPENBLAS: if "1", do not add libopenblas to conda env
  - H2O_DAI_MLFLOW_OPENBLAS_VERSION: pin for libopenblas package version (default: 0.3.30)
  - H2O_DAI_MLFLOW_FORCE_CONDA: if "1", synthesize a conda env even when pip env would be sufficient
  - H2O_DAI_MLFLOW_DISABLE_PROJECT: if "1", never launch MLflow Project (even on non-3.8)
  - H2O_DAI_MLFLOW_FORCE_PROJECT: if "1", always launch MLflow Project
  - H2O_DAI_MLFLOW_PROJECT_MODE: internal flag set to "1" when running inside the MLflow Project
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict


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
DISABLE_OPENBLAS_DEFAULT = False
OPENBLAS_VERSION_DEFAULT = "0.3.30"
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

# Env vars to neutralize DBR Spark leakage in child predict processes
SAFE_PREDICT_ENV_DEFAULTS: Dict[str, str] = {
    "PYTHONPATH": "",
    "SPARK_HOME": "",
    "PYSPARK_PYTHON": "",
    "PYSPARK_DRIVER_PYTHON": "",
}

# Limit native linear algebra/thread pools to avoid oversubscription and OOM/crashes
THREAD_ENV_DEFAULTS: Dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    # Reduce glibc arena usage to avoid fragmentation and high RSS in long-lived servers
    "MALLOC_ARENA_MAX": "1",
}

# environment.yml pip merge behavior
ENV_YAML_PIP_ALLOWLIST_DEFAULT: Sequence[str] = ("psutil", "tabulate", "cgroupspy")


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


def is_openblas_disabled() -> bool:
    return os.environ.get("H2O_DAI_MLFLOW_DISABLE_OPENBLAS", "0").strip() == "1"

def get_openblas_version() -> str:
    return os.environ.get(
        "H2O_DAI_MLFLOW_OPENBLAS_VERSION", OPENBLAS_VERSION_DEFAULT
    ).strip()


def is_conda_forced() -> bool:
    # Default to forcing conda to ensure non-pip deps like libmagic are present
    return os.environ.get("H2O_DAI_MLFLOW_FORCE_CONDA", "1" if FORCE_CONDA_DEFAULT else "0").strip() == "1"


def is_project_disabled() -> bool:
    return os.environ.get("H2O_DAI_MLFLOW_DISABLE_PROJECT", "0").strip() == "1"


def is_project_forced() -> bool:
    return os.environ.get("H2O_DAI_MLFLOW_FORCE_PROJECT", "0").strip() == "1"


def is_project_mode() -> bool:
    return os.environ.get(PROJECT_MODE_ENV) == "1"


def get_scoring_env_var_aliases() -> Sequence[str]:
    return SCORING_ENV_VAR_ALIASES


def get_build_pyorc_enabled() -> bool:
    # Enabled unless explicitly disabled via env
    return os.environ.get("H2O_DAI_MLFLOW_BUILD_PYORC", "1").strip() != "0"


def get_pyorc_version() -> str:
    return os.environ.get("H2O_DAI_MLFLOW_PYORC_VERSION", PYORC_VERSION_DEFAULT).strip()


def is_safe_predict_envs_disabled() -> bool:
    return os.environ.get("H2O_DAI_MLFLOW_DISABLE_SAFE_PREDICT_ENVS", "0").strip() == "1"


def get_conda_env_variables() -> Dict[str, str]:
    """Environment variables to include in conda.yaml.

    This ensures child processes launched by MLflow under conda have Spark-related
    env vars blanked without users passing extra_envs.
    """
    out: Dict[str, str] = {}
    if not is_safe_predict_envs_disabled():
        out.update(SAFE_PREDICT_ENV_DEFAULTS)
    # Apply thread limiting unless disabled by user
    if os.environ.get("H2O_DAI_MLFLOW_DISABLE_THREAD_LIMITS", "0").strip() != "1":
        # Only set if not already set by user
        for k, v in THREAD_ENV_DEFAULTS.items():
            out.setdefault(k, v)
    # Default BLAS preference: force conda OpenBLAS path and do not prefer numpy vendored lib
    out.setdefault("H2O_DAI_MLFLOW_PREFER_NUMPY_OPENBLAS", "0")
    # Provide a commonly available conda path; runtime checks existence and falls back safely
    out.setdefault(
        "H2O_DAI_MLFLOW_FORCE_OPENBLAS_PATH",
        "/opt/conda/envs/mlflow-env/lib/libopenblasp-r0.3.30.so",
    )
    return out


def is_env_yaml_pip_merge_enabled() -> bool:
    """Whether to merge selected pip packages from scoring-pipeline/environment.yml.

    Defaults to enabled; disable with H2O_DAI_MLFLOW_ENV_YAML_PIP_DISABLE=1.
    """
    return os.environ.get("H2O_DAI_MLFLOW_ENV_YAML_PIP_DISABLE", "0").strip() != "1"


def get_env_yaml_pip_allowlist() -> List[str]:
    """Return package names allowed to be merged from environment.yml's pip section.

    Users can extend/override via H2O_DAI_MLFLOW_ENV_YAML_PIP_ALLOWLIST
    (comma-separated names). Defaults: psutil, tabulate, cgroupspy.
    """
    extra = os.environ.get("H2O_DAI_MLFLOW_ENV_YAML_PIP_ALLOWLIST", "").strip()
    base = list(ENV_YAML_PIP_ALLOWLIST_DEFAULT)
    if extra:
        base.extend([p.strip() for p in extra.split(",") if p.strip()])
    # dedup preserve order
    seen = set()
    out: List[str] = []
    for p in base:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out
