import argparse
import base64
import importlib
import inspect
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import mlflow
import mlflow.artifacts
import pandas as pd
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
import yaml
from mlflow.types import ColSpec, Schema
from packaging.version import Version
import typing as _t

from h2o_dai_py_scoring_mlflow.config import (
    DEFAULT_PYTHON_VERSION,
    REQUIRED_PYTHON_VERSION,
    PROJECT_MODE_ENV,
    get_excluded_packages,
    get_importlib_resources_version,
    get_pyspark_version,
    is_importlib_resources_disabled,
    is_libmagic_disabled,
    is_openblas_disabled,
    get_openblas_version,
    is_project_disabled,
    is_project_forced,
    is_project_mode,
    is_pyspark_disabled,
    is_conda_forced,
    get_conda_env_variables,
    is_env_yaml_pip_merge_enabled,
    get_env_yaml_pip_allowlist,
)


_logger = logging.getLogger(__name__)

# Track whether we've already preloaded an OpenBLAS library in this process
_OPENBLAS_PRELOADED: _t.Optional[str] = None

# Decide at import time whether to use MLflow 2.20+ type-hint based validation.
# We only enable this when the active MLflow supports it, to avoid runtime
# errors/warnings in older environments (e.g., Serving images).
_USE_TYPEHINTS: bool = False
try:
    _USE_TYPEHINTS = Version(mlflow.__version__) >= Version("2.20.0") and os.environ.get(
        "H2O_DAI_MLFLOW_DISABLE_TYPEHINTS", "0"
    ).strip() != "1"
    if _USE_TYPEHINTS:
        from mlflow.types.type_hints import TypeFromExample  # type: ignore
except Exception:
    _USE_TYPEHINTS = False


def _patch_streams_fileno() -> None:
    """Ensure stdout/stderr expose fileno() for native libs that expect it."""
    try:
        import sys as _sys

        class _WithFileno:
            def __init__(self, base, fd):
                self._b = base
                self._fd = fd
            def fileno(self):
                return self._fd
            def __getattr__(self, name):
                return getattr(self._b, name)

        if not hasattr(_sys.stdout, "fileno") or not callable(getattr(_sys.stdout, "fileno", None)):
            _sys.stdout = _WithFileno(_sys.stdout, 1)  # type: ignore
        if not hasattr(_sys.stderr, "fileno") or not callable(getattr(_sys.stderr, "fileno", None)):
            _sys.stderr = _WithFileno(_sys.stderr, 2)  # type: ignore
    except Exception:
        pass


def _log_diag(msg: str) -> None:
    try:
        sys.stderr.write(f"[DriverlessDiag] {msg}\n")
    except Exception:
        try:
            print(f"[DriverlessDiag] {msg}")
        except Exception:
            pass


def _ensure_native_libs_and_diagnose() -> None:
    """Best-effort: make OpenBLAS/libmagic discoverable and emit diagnostics.

    This mirrors the behavior we ship in sitecustomize.py but runs inline so it
    applies even if sitecustomize wasn't auto-imported (as can happen in Serving).
    """
    try:
        if os.name == "posix":
            prefixes = []
            conda_prefix = os.environ.get("CONDA_PREFIX")
            if conda_prefix:
                prefixes.append(conda_prefix)
            prefixes.append(sys.prefix)

            # 1) Build a search list for shared libs
            prefer_numpy = os.environ.get("H2O_DAI_MLFLOW_PREFER_NUMPY_OPENBLAS", "0").strip() != "0"
            search_dirs: List[str] = []
            for p in prefixes:
                if not p:
                    continue
                cand = os.path.join(p, "lib")
                if os.path.isdir(cand):
                    search_dirs.append(cand)
            # Include numpy’s private .libs folder (manylinux wheels)
            try:
                import numpy as _np
                np_libs = os.path.join(os.path.dirname(_np.__file__), "..", "numpy.libs")
                np_libs = os.path.abspath(np_libs)
                if os.path.isdir(np_libs):
                    # If preferring numpy, restrict search to numpy.libs only
                    if prefer_numpy:
                        search_dirs = [np_libs]
                    else:
                        search_dirs.insert(0, np_libs)
            except Exception:
                pass

            # 2) Ensure LD_LIBRARY_PATH contains core env lib and numpy.libs
            if search_dirs:
                cur = os.environ.get("LD_LIBRARY_PATH", "")
                parts = [x for x in cur.split(":") if x]
                changed = False
                for d in search_dirs:
                    if d not in parts:
                        parts.append(d)
                        changed = True
                if changed:
                    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

            # 3) Preload OpenBLAS explicitly if present on disk (once)
            try:
                import ctypes, ctypes.util
            except Exception:
                ctypes = None  # type: ignore

            def _try_load(path: str) -> bool:
                global _OPENBLAS_PRELOADED
                if _OPENBLAS_PRELOADED:
                    # Already loaded once, skip additional loads
                    return True
                try:
                    # Prepend to LD_PRELOAD so the dynamic linker loads it first
                    try:
                        cur = os.environ.get("LD_PRELOAD", "")
                        parts = [x for x in cur.split(":") if x]
                        if path not in parts:
                            parts.insert(0, path)
                            os.environ["LD_PRELOAD"] = ":".join(parts)
                            _log_diag(f"LD_PRELOAD updated with {path}")
                    except Exception:
                        pass
                    ctypes.CDLL(path, mode=getattr(ctypes, 'RTLD_GLOBAL', 0))
                    _log_diag(f"preloaded openblas: {path}")
                    _OPENBLAS_PRELOADED = path
                    # Patch ctypes.util.find_library to return our path for openblas queries
                    try:
                        orig_find = ctypes.util.find_library  # type: ignore[attr-defined]
                        def _patched(name: str):
                            n = (name or "").lower()
                            if "openblas" in n and os.path.exists(path):
                                return path
                            return orig_find(name)
                        ctypes.util.find_library = _patched  # type: ignore[assignment]
                    except Exception:
                        pass
                    return True
                except Exception as exc:
                    _log_diag(f"failed to preload {path}: {exc}")
                    return False

            forced_path = os.environ.get("H2O_DAI_MLFLOW_FORCE_OPENBLAS_PATH", "").strip()
            if not forced_path:
                # Provide a reasonable default for Databricks Serving images
                default_conda_path = "/opt/conda/envs/mlflow-env/lib/libopenblasp-r0.3.30.so"
                if os.path.isfile(default_conda_path):
                    forced_path = default_conda_path
            if ctypes and forced_path and os.path.isfile(forced_path):
                _try_load(forced_path)
            elif ctypes and search_dirs and os.environ.get("H2O_DAI_MLFLOW_PRELOAD_OPENBLAS", "1").strip() != "0":
                loaded = False
                patterns = ("libopenblas", "libopenblas64_")
                for d in search_dirs:
                    try:
                        for fname in os.listdir(d):
                            if any(fname.startswith(p) for p in patterns) and (fname.endswith('.so') or '.so.' in fname or fname.endswith('.dylib')):
                                if _try_load(os.path.join(d, fname)):
                                    loaded = True
                                    break
                        if loaded:
                            break
                    except Exception:
                        continue
    except Exception:
        pass

    if os.environ.get("H2O_DAI_MLFLOW_DIAG", "1").strip() == "0":
        return
    # Emit quick diagnostics without loading additional BLAS/LAPACK libraries
    try:
        import ctypes.util  # noqa: F401
        import numpy as _np  # type: ignore

        cfg = getattr(_np, "__config__", None)
        info = {}
        if cfg is not None:
            for key in ("openblas_info", "blas_ilp64_info", "blas_opt_info", "lapack_opt_info"):
                try:
                    info[key] = bool(cfg.get_info(key))
                except Exception:
                    pass
        _log_diag(f"numpy={_np.__version__} build_info={info}")
    except Exception:
        pass
    _log_diag(f"sys.prefix={sys.prefix} CONDA_PREFIX={os.environ.get('CONDA_PREFIX','')}")
    _log_diag(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH','')}")
    # Report find_library results (do not dlopen to avoid mixing backends)
    try:
        for name in ("openblas", "blas", "lapack", "magic"):
            try:
                path = ctypes.util.find_library(name)
                _log_diag(f"find_library('{name}') -> {path}")
            except Exception as exc:
                _log_diag(f"find_library failed for {name}: {exc}")
    except Exception:
        pass


def _ensure_psutil_compat() -> None:
    """Ensure psutil exposes RLIMIT constants used by h2oaicore.

    Some conda builds of psutil omit RLIMIT_* and RLIM_INFINITY. We first try to
    populate RLIM_INFINITY from the stdlib 'resource' module, then define any
    missing RLIMIT_* names to None so attribute access does not fail.
    """
    try:
        import psutil  # type: ignore
        # Prefer resource for RLIM_INFINITY value when present
        if not hasattr(psutil, "RLIM_INFINITY"):
            try:
                import resource as _resource  # type: ignore
                setattr(psutil, "RLIM_INFINITY", getattr(_resource, "RLIM_INFINITY", None))
            except Exception:
                try:
                    setattr(psutil, "RLIM_INFINITY", None)
                except Exception:
                    pass
        names = (
            "RLIMIT_AS",
            "RLIMIT_CORE",
            "RLIMIT_CPU",
            "RLIMIT_DATA",
            "RLIMIT_FSIZE",
            "RLIMIT_LOCKS",
            "RLIMIT_MEMLOCK",
            "RLIMIT_MSGQUEUE",
            "RLIMIT_NICE",
            "RLIMIT_NOFILE",
            "RLIMIT_NPROC",
            "RLIMIT_RSS",
            "RLIMIT_RTPRIO",
            "RLIMIT_RTTIME",
            "RLIMIT_SIGPENDING",
            "RLIMIT_STACK",
        )
        for name in names:
            if not hasattr(psutil, name):
                try:
                    setattr(psutil, name, None)
                except Exception:
                    pass
    except Exception:
        pass


def _deduplicate_preserve_order(items: Iterable[str]) -> List[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for item in items:
        if item not in seen:
            seen[item] = None
    return list(seen.keys())


def _ensure_pandas_frame(predictions: Any) -> pd.DataFrame:
    if hasattr(predictions, "to_pandas"):
        predictions = predictions.to_pandas()
    if isinstance(predictions, pd.DataFrame):
        return predictions
    if isinstance(predictions, pd.Series):
        return predictions.to_frame(name="prediction")
    return pd.DataFrame(predictions)


def _load_training_data_stats(scoring_path: Path) -> Dict[str, Any]:
    stats_path = scoring_path / "training_data_column_stats.json"
    if not stats_path.exists():
        return {}
    try:
        raw = stats_path.read_text()
        stats_obj = json.loads(raw)
        if isinstance(stats_obj, str):
            stats_obj = json.loads(stats_obj)
        if isinstance(stats_obj, dict):
            return stats_obj
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning("Failed to parse training_data_column_stats.json: %s", exc)
    return {}


def _coerce_sample_value(value: Any, dtype: Optional[str]) -> Any:
    if value is None:
        if dtype:
            dtype_lower = dtype.lower()
            if any(token in dtype_lower for token in ("real", "float", "int", "num")):
                return 0.0
            return ""
        return ""
    if dtype:
        dtype_lower = dtype.lower()
        if any(token in dtype_lower for token in ("real", "float", "int", "num")):
            try:
                return float(value)
            except (TypeError, ValueError):
                return value
        if "time" in dtype_lower or "date" in dtype_lower:
            return str(value)
    return value


def _build_input_example(
    columns: Sequence[str], stats: Mapping[str, Any]
) -> pd.DataFrame:
    row: Dict[str, Any] = {}
    for name in columns:
        col_info = stats.get(name) if isinstance(stats, dict) else None
        dtype = None
        if isinstance(col_info, dict):
            dtype = col_info.get("data_type")
            data_sample = col_info.get("data") or []
            value = data_sample[0] if data_sample else None
        else:
            value = None
        row[name] = _coerce_sample_value(value, dtype)
    return pd.DataFrame([row], columns=list(columns))


def _extract_column_names_from_example(example_path: Path) -> List[str]:
    if not example_path.exists():
        return []
    try:
        text = example_path.read_text()
    except Exception:  # pragma: no cover - defensive
        return []
    names = re.findall(r"name=['\"]([^'\"]+)['\"]", text)
    if not names:
        return []
    # preserve order while dropping duplicates
    ordered: OrderedDict[str, None] = OrderedDict()
    for name in names:
        if name not in ordered:
            ordered[name] = None
    return list(ordered.keys())


def _mlflow_type_from_dai(dtype: _t.Optional[str]) -> str:
    if not dtype:
        return "string"
    t = dtype.strip().lower()
    if t in {"string", "str", "unicode", "enum", "category"}:
        return "string"
    if t in {"int", "integer"}:
        return "long"
    if t in {"long", "int64"}:
        return "long"
    if t in {"real", "float", "double", "number", "numeric"}:
        return "double"
    if t in {"bool", "boolean"}:
        return "boolean"
    if t in {"date", "datetime", "time", "timestamp"}:
        return "datetime"
    if t in {"binary", "bytes"}:
        return "binary"
    return "string"


def _extract_schema_from_example(example_path: Path) -> List[Tuple[str, str]]:
    if not example_path.exists():
        return []
    try:
        text = example_path.read_text()
    except Exception:
        return []

    pairs: List[Tuple[str, str]] = []

    # Pattern for dict(name='...', type='...') style
    import re as _re
    pattern1 = _re.compile(
        r"dict\s*\(.*?name\s*=\s*['\"]([^'\"]+)['\"].*?type\s*=\s*['\"]([^'\"]+)['\"].*?\)",
        _re.IGNORECASE | _re.DOTALL,
    )
    for m in pattern1.finditer(text):
        name = m.group(1)
        dtype = m.group(2)
        pairs.append((name, _mlflow_type_from_dai(dtype)))

    # Pattern for {'name': '...', 'type': '...'} style
    pattern2 = _re.compile(
        r"\{[^}]*['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"][^}]*['\"]type['\"]\s*:\s*['\"]([^'\"]+)['\"][^}]*\}",
        _re.IGNORECASE | _re.DOTALL,
    )
    for m in pattern2.finditer(text):
        name = m.group(1)
        dtype = m.group(2)
        pairs.append((name, _mlflow_type_from_dai(dtype)))

    # Deduplicate preserving order
    seen: OrderedDict[str, str] = OrderedDict()
    for name, dtype in pairs:
        if name not in seen:
            seen[name] = dtype
    return [(n, d) for n, d in seen.items()]


def _build_input_schema(
    scoring_path: Path,
    *,
    stats: Mapping[str, Any],
    resolved_input_example: _t.Optional[pd.DataFrame],
) -> _t.Optional[Schema]:
    # Prefer explicit schema from example.py if present
    example_pairs = _extract_schema_from_example(scoring_path / "example.py")
    if example_pairs:
        return Schema([ColSpec(type=d, name=n) for n, d in example_pairs])

    # Next, derive from training_data_column_stats.json data_type fields
    if isinstance(stats, dict) and stats:
        cols: List[ColSpec] = []
        for name, meta in stats.items():
            if not isinstance(meta, dict):
                continue
            dtype = _mlflow_type_from_dai(str(meta.get("data_type", "")))
            cols.append(ColSpec(type=dtype, name=str(name)))
        if cols:
            return Schema(cols)

    # Finally, if the user supplied an input example, infer schema from it
    if resolved_input_example is not None:
        try:
            inferred = infer_signature(resolved_input_example)
            return inferred.inputs
        except Exception:
            return None
    return None


def _load_experiment_summary(scoring_path: Path) -> Dict[str, Any]:
    summary_zip = next(scoring_path.glob("h2oai_experiment_summary_*.zip"), None)
    if summary_zip is None:
        return {}
    try:
        with zipfile.ZipFile(summary_zip) as archive:
            if "summary.json" in archive.namelist():
                with archive.open("summary.json") as handle:
                    return json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning("Failed to parse experiment summary: %s", exc)
    return {}


def _load_dataframe_from_path(path: Union[str, os.PathLike]) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".json", ".jsn"}:
        return pd.read_json(file_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    if suffix == ".jay":
        try:
            import datatable as dt  # type: ignore

            return dt.Frame(str(file_path)).to_pandas()
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Reading .jay files requires the 'datatable' package to be installed"
            ) from exc
    raise ValueError(f"Unsupported example data format for {file_path!s}")


class _DriverlessAIScoringModelBase(mlflow.pyfunc.PythonModel):
    """Common logic for the Driverless AI Python scoring pyfunc model.

    The predict method is provided by a thin typed/untyped subclass depending on
    MLflow version support for type hints.
    """

    def __init__(
        self,
        scorer_module_name: str,
        *,
        apply_data_recipes: bool = False,
        scorer_kwargs: Optional[Mapping[str, Any]] = None,
        predict_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.scorer_module_name = scorer_module_name
        self.apply_data_recipes = apply_data_recipes
        self.scorer_kwargs = dict(scorer_kwargs or {})
        self.predict_kwargs = dict(predict_kwargs or {})
        self.scorer = None
        self._import_error: Optional[ModuleNotFoundError] = None
        self.input_columns: Sequence[str] = []

    def load_context(self, context: mlflow.pyfunc.model.PythonModelContext) -> None:
        return

    @staticmethod
    def _normalize_column_labels(raw: Any) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            return list(raw.keys())
        if isinstance(raw, (list, tuple)):
            if not raw:
                return []
            first = raw[0]
            if isinstance(first, (list, tuple)) and first:
                return [str(item[0]) for item in raw]
            return [str(item) for item in raw]
        return [str(raw)]

    def _predict_impl(self, model_input: Any) -> pd.DataFrame:
        self._ensure_scorer_initialized()
        frame = self._to_dataframe(model_input)
        predict_kwargs = dict(self.predict_kwargs)
        predict_kwargs.setdefault("apply_data_recipes", self.apply_data_recipes)
        try:
            predictions = self.scorer.score_batch(frame, **predict_kwargs)
            return _ensure_pandas_frame(predictions)
        except Exception:
            import traceback as _tb
            _log_diag("scoring failed:\n" + _tb.format_exc())
            raise

    def _to_dataframe(
        self, model_input: Union[pd.DataFrame, Mapping[str, Any], Sequence[Any]]
    ) -> pd.DataFrame:
        if isinstance(model_input, pd.DataFrame):
            df = model_input.copy()
        elif isinstance(model_input, dict):
            df = pd.DataFrame([model_input])
        elif isinstance(model_input, list):
            if not model_input:
                return pd.DataFrame(columns=list(self.input_columns))
            first = model_input[0]
            if isinstance(first, dict):
                df = pd.DataFrame(model_input)
            else:
                df = pd.DataFrame(model_input, columns=list(self.input_columns))
        else:
            raise TypeError(
                "Unsupported input type for predict: {}".format(
                    type(model_input).__name__
                )
            )

        if self.input_columns:
            missing = [name for name in self.input_columns if name not in df.columns]
            if missing:
                raise ValueError(
                    "Missing required columns: {}".format(", ".join(missing))
                )
            return df[self.input_columns]
        return df

    def _initialize_scorer(self) -> None:
        try:
            _patch_streams_fileno()
            _ensure_native_libs_and_diagnose()
            # Ensure psutil exposes expected RLIMIT constants even if conda psutil is limited
            _ensure_psutil_compat()
            # Best-effort import of sitecustomize to run any additional runtime shims
            try:
                import sitecustomize  # noqa: F401
            except Exception:
                pass
            scorer_module = importlib.import_module(self.scorer_module_name)
            scorer_cls = getattr(scorer_module, "Scorer")
            self.scorer = scorer_cls(**self.scorer_kwargs)
            raw_columns = self.scorer.get_column_names()
            self.input_columns = self._normalize_column_labels(raw_columns)
        except Exception:
            import traceback as _tb
            _log_diag("scorer initialization failed:\n" + _tb.format_exc())
            raise

    def _ensure_scorer_initialized(self) -> None:
        if self.scorer is not None:
            return
        if self._import_error is not None:
            raise ModuleNotFoundError(
                f"Unable to import Driverless scorer module '{self.scorer_module_name}'. "
                f"Missing dependency: '{self._import_error.name}'. Ensure the model is run "
                "inside the MLflow-managed Python 3.8 environment."
            ) from self._import_error
        self._initialize_scorer()


# Define the public model class with or without type hints depending on MLflow.
if _USE_TYPEHINTS:
    class DriverlessAIScoringModel(_DriverlessAIScoringModelBase):
        def predict(
            self,
            context: mlflow.pyfunc.model.PythonModelContext,
            model_input: "TypeFromExample",  # type: ignore[name-defined]
        ) -> pd.DataFrame:
            return self._predict_impl(model_input)
else:
    class DriverlessAIScoringModel(_DriverlessAIScoringModelBase):
        def predict(
            self,
            context: mlflow.pyfunc.model.PythonModelContext,
            model_input,
        ) -> pd.DataFrame:
            return self._predict_impl(model_input)


def build_pip_requirements(
    scoring_pipeline_dir: str,
    *,
    include_mlflow: bool = True,
    extra_requirements: Optional[Sequence[str]] = None,
    wheel_prefix: str = "./artifacts/scoring-pipeline",
) -> List[str]:
    """Build the pip requirements for the logged MLflow model.

    On-the-fly sanitization rules:
      - Ignore any wheel filenames listed in the exported requirements.txt and
        instead enumerate actual wheels present next to the scorer on disk.
      - Exclude packages known to cause dependency conflicts for CPU-only
        scoring (e.g., h2o4gpu pins scikit-learn==0.21.x). You can extend the
        exclusion list via H2O_DAI_MLFLOW_EXCLUDE_PACKAGES (comma-separated
        names).
    """

    def _pkg_name(spec: str) -> str:
        # For wheel paths: take basename, then name before first dash
        base = Path(spec).name
        if base.endswith(".whl"):
            return base.split("-")[0].lower()
        # For package specs (name==ver, name[extras]==ver, etc.) extract name
        s = spec.split(";")[0].strip()
        for sep in ["==", ">=", "<=", "~=", ">", "<", "!="]:
            if sep in s:
                return s.split(sep)[0].split("[")[0].strip().lower()
        return s.split("[")[0].strip().lower()

    exclude_pkgs = set(x.lower() for x in get_excluded_packages())

    scoring_path = Path(scoring_pipeline_dir)
    requirements_file = scoring_path / "requirements.txt"
    if not requirements_file.exists():
        raise FileNotFoundError(
            "requirements.txt not found inside scoring pipeline directory"
        )

    package_specs: List[str] = []

    # Keep only non-wheel lines from requirements.txt and drop excluded pkgs
    for line in requirements_file.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.lower().endswith(".whl"):
            # Skip: we will add real wheels discovered on disk below
            continue
        if _pkg_name(entry) in exclude_pkgs:
            continue
        package_specs.append(entry)

    # Discover actual wheels present in the export and reference them relative
    # to how MLflow lays out model artifacts during build (/model/artifacts/...)
    discovered_wheels: List[str] = []
    found_wheels = list(sorted(scoring_path.glob("*.whl")))
    # If a prebuilt pyorc wheel is present, allow it (remove from excludes)
    if any(Path(w).name.lower().startswith("pyorc-") for w in found_wheels):
        exclude_pkgs.discard("pyorc")

    for wheel_path in found_wheels:
        if _pkg_name(wheel_path.name) in exclude_pkgs:
            continue
        discovered_wheels.append(f"{wheel_prefix.rstrip('/')}/{wheel_path.name}")

    # Ensure compatibility shims are always present in the model env.
    # Compose compatibility shims from env-configurable pins
    compat_pkgs: List[str] = []
    if not is_importlib_resources_disabled():
        compat_pkgs.append(f"importlib-resources=={get_importlib_resources_version()}")
    if not is_pyspark_disabled():
        compat_pkgs.append(f"pyspark=={get_pyspark_version()}")

    # Optionally merge selected pip packages from environment.yml to cover
    # runtime helpers (e.g., psutil, tabulate, cgroupspy) that may be omitted
    # from requirements.txt exports.
    env_yaml_specs: List[str] = []
    if is_env_yaml_pip_merge_enabled():
        for fname in ("environment.yml", "environment.yaml"):
            env_file = scoring_path / fname
            if env_file.exists():
                try:
                    data = yaml.safe_load(env_file.read_text()) or {}
                    deps = data.get("dependencies") or []
                    allow = set(x.lower() for x in get_env_yaml_pip_allowlist())
                    # dependencies can include dicts like {pip: [list]}
                    for item in deps:
                        if isinstance(item, dict) and "pip" in item:
                            for spec in item["pip"] or []:
                                name = _pkg_name(str(spec))
                                if name in allow and name not in exclude_pkgs:
                                    env_yaml_specs.append(str(spec))
                except Exception:
                    pass

    # Prefer PyPI psutil over any conda-provided psutil. Remove any psutil specs
    # we collected so far and add a forced pip psutil at the end to ensure the
    # wheel installed by pip wins in the environment.
    def _not_psutil(spec: str) -> bool:
        try:
            return _pkg_name(spec) != "psutil"
        except Exception:
            return True

    package_specs = [s for s in package_specs if _not_psutil(s)]
    env_yaml_specs = [s for s in env_yaml_specs if _not_psutil(s)]

    psutil_ver = os.environ.get("H2O_DAI_MLFLOW_PSUTIL_VERSION", "").strip()
    psutil_spec = f"psutil=={psutil_ver}" if psutil_ver else "psutil"

    combined = package_specs + env_yaml_specs + discovered_wheels + compat_pkgs + [psutil_spec]
    if include_mlflow:
        combined.append(f"mlflow=={mlflow.__version__}")
    if extra_requirements:
        # Drop any excluded packages that may have been explicitly added
        combined.extend([r for r in extra_requirements if _pkg_name(r) not in exclude_pkgs])

    return _deduplicate_preserve_order(combined)


# Python version pinned for the project/model runtime
# Centralized in config.py


def build_python_env_config(
    python_version: str = DEFAULT_PYTHON_VERSION,
    *,
    build_dependencies: Optional[Sequence[str]] = None,
    pip_packages: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    deps = (
        list(build_dependencies)
        if build_dependencies
        else [
            "pip==21.1",
            "setuptools==49.6.0",
            "wheel==0.35.1",
        ]
    )
    packages = list(pip_packages) if pip_packages else []
    return {
        "python": python_version,
        "build_dependencies": deps,
        "dependencies": packages,
    }


def build_conda_env_config(python_env: Mapping[str, Any]) -> Dict[str, Any]:
    python_version = python_env.get("python", DEFAULT_PYTHON_VERSION)
    build_deps = list(python_env.get("build_dependencies", []))
    pip_packages = list(python_env.get("dependencies", []))

    dependencies: List[Any] = [f"python={python_version}"]
    dependencies.extend(build_deps)
    # Ensure libmagic is present for python-magic (h2oaicore depends on it)
    if not is_libmagic_disabled():
        if not any(str(d).startswith("libmagic") for d in dependencies if isinstance(d, str)):
            dependencies.append("libmagic")
    # Ensure OpenBLAS shared library is present for numpy/h2oaicore openblas detection
    if not is_openblas_disabled():
        # Remove any existing unpinned libopenblas entries and add a pinned one
        deps_norm: List[Any] = []
        for d in dependencies:
            if isinstance(d, str) and d.startswith("libopenblas"):
                continue
            deps_norm.append(d)
        dependencies = deps_norm
        version = get_openblas_version()
        dependencies.append(f"libopenblas={version}" if version else "libopenblas")
        # Also include toolchain libs that OpenBLAS often requires
        if not any(str(d).startswith("libgfortran") for d in dependencies if isinstance(d, str)):
            dependencies.append("libgfortran")
        if not any(str(d).startswith("libgcc-ng") for d in dependencies if isinstance(d, str)):
            dependencies.append("libgcc-ng")
        if not any(str(d).startswith("libgomp") for d in dependencies if isinstance(d, str)):
            dependencies.append("libgomp")
        if not any(str(d).startswith("libstdcxx-ng") for d in dependencies if isinstance(d, str)):
            dependencies.append("libstdcxx-ng")
    if pip_packages:
        dependencies.append({"pip": pip_packages})

    config: Dict[str, Any] = {
        "name": "driverless_ai_scoring",
        "channels": ["conda-forge", "defaults"],
        "dependencies": dependencies,
    }
    variables = get_conda_env_variables()
    if variables:
        # Add env vars for conda activation to neutralize DBR Spark leakage
        config["variables"] = {str(k): str(v) for k, v in variables.items()}
    return config


MODEL_INFO_ARTIFACT = "_driverless/model_info.json"
PROJECT_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "mlflow_project"


def _encode_b64_json(payload: Any) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")


def _resolve_scoring_path(
    scoring_pipeline_dir: str,
) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    scoring_path = Path(scoring_pipeline_dir).expanduser().resolve()
    temp_dir: Optional[tempfile.TemporaryDirectory] = None
    if scoring_path.is_file() and scoring_path.suffix == ".zip":
        temp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(scoring_path) as archive:
            archive.extractall(temp_dir.name)
        extracted_root = Path(temp_dir.name)
        subdirs = [p for p in extracted_root.iterdir() if p.is_dir()]
        scoring_path = subdirs[0] if len(subdirs) == 1 else extracted_root
    if not scoring_path.exists():
        raise FileNotFoundError(
            f"Scoring pipeline directory does not exist: {scoring_path}"
        )
    return scoring_path.resolve(), temp_dir


def _prepare_project_workspace(
    scoring_path: Path,
) -> Tuple[Path, Path, Optional[tempfile.TemporaryDirectory]]:
    if (scoring_path / "MLproject").exists():
        project_scoring_dir = scoring_path / "scoring-pipeline"
        if project_scoring_dir.exists():
            return scoring_path, project_scoring_dir, None
        # Handle case where the provided directory already points at a scoring pipeline
        # but also contains a stray MLproject file from a previous staging step.
        temp_dir = tempfile.TemporaryDirectory()
        shutil.copytree(PROJECT_TEMPLATE_DIR, temp_dir.name, dirs_exist_ok=True)
        staged_scoring_dir = Path(temp_dir.name) / "scoring-pipeline"
        if staged_scoring_dir.exists():
            shutil.rmtree(staged_scoring_dir)
        shutil.copytree(scoring_path, staged_scoring_dir)
        _stage_helper_package(Path(temp_dir.name))
        return Path(temp_dir.name), staged_scoring_dir, temp_dir

    parent = scoring_path.parent
    if (parent / "MLproject").exists():
        return parent, scoring_path, None

    if not PROJECT_TEMPLATE_DIR.exists():
        raise FileNotFoundError(
            f"MLflow project template not found at {PROJECT_TEMPLATE_DIR}"
        )

    temp_dir = tempfile.TemporaryDirectory()
    shutil.copytree(PROJECT_TEMPLATE_DIR, temp_dir.name, dirs_exist_ok=True)
    staged_scoring_dir = Path(temp_dir.name) / "scoring-pipeline"
    if staged_scoring_dir.exists():
        shutil.rmtree(staged_scoring_dir)
    shutil.copytree(scoring_path, staged_scoring_dir)
    _stage_helper_package(Path(temp_dir.name))
    return Path(temp_dir.name), staged_scoring_dir, temp_dir


def _stage_helper_package(destination: Path) -> None:
    """Copy the helper package into the staged project so imports succeed."""

    package_root = Path(__file__).resolve().parent.parent
    target_dir = destination / package_root.name
    if target_dir.exists():
        return
    shutil.copytree(
        package_root,
        target_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def _should_launch_project() -> bool:
    if is_project_mode():
        return False
    if is_project_disabled():
        return False
    if is_project_forced():
        return True
    return sys.version_info[:2] != REQUIRED_PYTHON_VERSION

def _discover_scorer_module_name(scoring_pipeline_dir: str) -> str:
    scoring_path = Path(scoring_pipeline_dir)
    wheel_candidates = sorted(scoring_path.glob("scoring_h2oai_experiment_*.whl"))
    if not wheel_candidates:
        raise FileNotFoundError(
            "Unable to find scoring_h2oai_experiment_*.whl in scoring pipeline"
        )
    return wheel_candidates[-1].name.split("-")[0]




def _launch_project_logging(
    scoring_path: Path,
    *,
    artifact_path: str,
    run_id: Optional[str],
    apply_data_recipes: bool,
    scorer_kwargs: Optional[Mapping[str, Any]],
    predict_kwargs: Optional[Mapping[str, Any]],
    pip_requirements: Optional[Sequence[str]],
    python_env: Optional[Union[str, os.PathLike, Mapping[str, Any]]],
    extra_artifacts: Optional[Mapping[str, str]],
    registered_model_name: Optional[str],
    input_example_path: Optional[Union[str, os.PathLike]] = None,
    output_example_path: Optional[Union[str, os.PathLike]] = None,
) -> mlflow.models.model.ModelInfo:
    project_dir, project_scoring_dir, project_temp_dir = _prepare_project_workspace(
        scoring_path
    )
    temp_files: List[str] = []

    parameters = {
        "artifact_path": artifact_path,
        "registered_model_name": registered_model_name or "",
        "run_id": run_id or "",
        "apply_data_recipes": "true" if apply_data_recipes else "false",
        "scorer_kwargs_json": _encode_b64_json(scorer_kwargs or {}),
        "predict_kwargs_json": _encode_b64_json(predict_kwargs or {}),
        "extra_artifacts_json": _encode_b64_json(extra_artifacts or {}),
        "pip_requirements_json": _encode_b64_json(
            list(pip_requirements) if pip_requirements is not None else []
        ),
        "python_env_path": "__NONE__",
    }

    try:
        parameters["scoring_dir"] = str(project_scoring_dir.relative_to(project_dir))
    except ValueError:
        parameters["scoring_dir"] = str(project_scoring_dir)

    if python_env is not None:
        if isinstance(python_env, (str, os.PathLike)):
            parameters["python_env_path"] = str(Path(python_env).resolve())
        elif isinstance(python_env, Mapping):
            tmp_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
            yaml.safe_dump(dict(python_env), tmp_file)
            tmp_file.flush()
            tmp_file.close()
            temp_files.append(tmp_file.name)
            parameters["python_env_path"] = tmp_file.name
        else:
            raise TypeError("python_env must be a mapping or path when provided")

    parameters["input_path"] = (
        str(Path(input_example_path).resolve())
        if input_example_path is not None
        else "__NONE__"
    )
    parameters["output_path"] = (
        str(Path(output_example_path).resolve())
        if output_example_path is not None
        else "__NONE__"
    )

    try:
        project_run = mlflow.projects.run(
            uri=str(project_dir),
            entry_point="log_model",
            parameters=parameters,
            env_manager="virtualenv",
            synchronous=True,
        )

        info_path = mlflow.artifacts.download_artifacts(
            run_id=project_run.run_id, artifact_path=MODEL_INFO_ARTIFACT
        )
        info = json.loads(Path(info_path).read_text())
        model_uri = info["model_uri"]
        try:
            return mlflow.pyfunc.get_model_info(model_uri)  # type: ignore[attr-defined]
        except AttributeError:
            return mlflow.models.get_model_info(model_uri)
    finally:
        for path in temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        if project_temp_dir is not None:
            project_temp_dir.cleanup()


def _log_driverless_scoring_pipeline_impl(
    scoring_path: Path,
    *,
    artifact_path: str,
    run_id: Optional[str],
    apply_data_recipes: bool,
    scorer_kwargs: Optional[Mapping[str, Any]],
    predict_kwargs: Optional[Mapping[str, Any]],
    pip_requirements: Optional[Sequence[str]],
    python_env: Optional[Union[str, os.PathLike, Mapping[str, Any]]],
    extra_artifacts: Optional[Mapping[str, str]],
    registered_model_name: Optional[str],
    input_example_df: Optional[pd.DataFrame] = None,
    output_example_df: Optional[pd.DataFrame] = None,
    input_example_path: Optional[Union[str, os.PathLike]] = None,
    output_example_path: Optional[Union[str, os.PathLike]] = None,
) -> mlflow.models.model.ModelInfo:
    module_name = _discover_scorer_module_name(str(scoring_path))
    python_model = DriverlessAIScoringModel(
        module_name,
        apply_data_recipes=apply_data_recipes,
        scorer_kwargs=scorer_kwargs,
        predict_kwargs=predict_kwargs,
    )

    # Use a hyphenated artifact subdir to match MLflow's path normalization and
    # ensure pip wheel paths resolve during container builds.
    artifact_dir_name = "scoring-pipeline"
    wheel_prefix = f"./artifacts/{artifact_dir_name}"
    if pip_requirements is not None:
        pip_reqs = list(pip_requirements)
    else:
        pip_reqs = build_pip_requirements(scoring_path, wheel_prefix=wheel_prefix)

    artifacts = {artifact_dir_name: str(scoring_path)}
    if extra_artifacts:
        artifacts.update(extra_artifacts)

    log_model_signature = inspect.signature(mlflow.pyfunc.log_model)

    python_env_cfg: Optional[Dict[str, Any]] = None
    if python_env is not None:
        if isinstance(python_env, (str, os.PathLike)):
            python_env_cfg = yaml.safe_load(Path(python_env).read_text())
        elif isinstance(python_env, Mapping):
            python_env_cfg = dict(python_env)
        else:
            raise TypeError("python_env must be a mapping or file path when provided")
    elif is_conda_forced():
        # Synthesize a python_env from pip requirements so we can emit a conda env
        python_env_cfg = build_python_env_config(
            python_version=DEFAULT_PYTHON_VERSION,
            build_dependencies=None,
            pip_packages=pip_reqs,
        )

    conda_env_cfg: Optional[Dict[str, Any]] = None
    if python_env_cfg is not None:
        conda_env_cfg = build_conda_env_config(python_env_cfg)

    resolved_input_example: Optional[pd.DataFrame] = None
    if input_example_df is not None:
        resolved_input_example = input_example_df.copy()
    elif input_example_path is not None:
        resolved_input_example = _load_dataframe_from_path(input_example_path)

    if resolved_input_example is not None:
        resolved_input_example = resolved_input_example.reset_index(drop=True)
        if len(resolved_input_example) > 5:
            resolved_input_example = resolved_input_example.head(5)

    resolved_output_example: Optional[pd.DataFrame] = None
    if output_example_df is not None:
        resolved_output_example = output_example_df.copy()
    elif output_example_path is not None:
        resolved_output_example = _load_dataframe_from_path(output_example_path)

    if resolved_output_example is not None:
        resolved_output_example = resolved_output_example.reset_index(drop=True)
        if len(resolved_output_example) > 5:
            resolved_output_example = resolved_output_example.head(5)

    stats = _load_training_data_stats(scoring_path)
    normalized_columns = list(stats.keys()) if stats else []
    if not normalized_columns:
        normalized_columns = _extract_column_names_from_example(
            scoring_path / "example.py"
        )

    input_example_final: Optional[pd.DataFrame] = resolved_input_example
    if input_example_final is None and normalized_columns:
        input_example_final = _build_input_example(normalized_columns, stats)

    # Build input schema from example.py / stats / input_example
    input_schema: Optional[Schema] = _build_input_schema(
        scoring_path, stats=stats, resolved_input_example=input_example_final
    )

    # Build output schema
    signature: Optional[ModelSignature] = None
    try:
        if input_schema is not None and resolved_output_example is not None:
            # Let MLflow infer outputs from example but keep our input schema
            inferred = infer_signature(
                _ensure_pandas_frame(input_example_final) if input_example_final is not None else None,
                _ensure_pandas_frame(resolved_output_example),
            )
            signature = ModelSignature(inputs=input_schema, outputs=inferred.outputs)
        elif input_schema is not None:
            # Derive output name from experiment summary or fallback
            summary = _load_experiment_summary(scoring_path)
            output_name = None
            if isinstance(summary, dict):
                output_name = summary.get("target") or summary.get("target_column")
            if not output_name:
                numeric_candidates = [
                    name
                    for name, meta in (stats or {}).items()
                    if isinstance(meta, dict)
                    and meta.get("stats", {}).get("is_numeric")
                    and name not in (normalized_columns or [])
                ]
                output_name = numeric_candidates[0] if numeric_candidates else "prediction"
            output_schema = Schema([ColSpec(type="double", name=str(output_name))])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning("Unable to construct signature from schema: %s", exc)
        signature = None

    input_example_for_logging: Optional[pd.DataFrame] = None
    if input_example_final is not None:
        limit = min(len(input_example_final), 5)
        input_example_for_logging = input_example_final.head(limit)

    def _log() -> mlflow.models.model.ModelInfo:
        code_root = Path(__file__).resolve().parent.parent
        log_kwargs = dict(
            python_model=python_model,
            artifacts=artifacts,
            registered_model_name=registered_model_name,
        )

        if "name" in log_model_signature.parameters:
            log_kwargs["name"] = artifact_path
        elif "artifact_path" in log_model_signature.parameters:
            log_kwargs["artifact_path"] = artifact_path
        else:
            raise RuntimeError(
                "Unsupported mlflow.pyfunc.log_model signature: missing name/artifact_path parameter"
            )

        code_paths = [str(code_root)]
        # Place sitecustomize.py at code/ root so Python auto-imports it before
        # mlflow loads dependencies. Copy the file explicitly in addition to the
        # package directory so it lands at the top-level of the 'code' folder.
        sitecustomize_path = code_root / "sitecustomize.py"
        if sitecustomize_path.exists():
            code_paths.append(str(sitecustomize_path))

        if "code_paths" in log_model_signature.parameters:
            log_kwargs["code_paths"] = code_paths
        elif "code_path" in log_model_signature.parameters:
            log_kwargs["code_path"] = code_paths
        else:
            raise RuntimeError(
                "Unsupported mlflow.pyfunc.log_model signature: missing code path parameter"
            )

        if python_env_cfg is not None and conda_env_cfg is not None and "conda_env" in log_model_signature.parameters:
            # Prefer conda_env so non-pip deps (libmagic) are installed in Serving
            log_kwargs["conda_env"] = conda_env_cfg
            if "python_version" in log_model_signature.parameters:
                requested_python = python_env_cfg.get("python", DEFAULT_PYTHON_VERSION)
                log_kwargs.setdefault("python_version", requested_python)
        elif python_env_cfg is not None and "python_env" in log_model_signature.parameters:
            log_kwargs["python_env"] = python_env_cfg
            if "python_version" in log_model_signature.parameters:
                requested_python = python_env_cfg.get("python", DEFAULT_PYTHON_VERSION)
                log_kwargs.setdefault("python_version", requested_python)
        else:
            log_kwargs["pip_requirements"] = pip_reqs
            if "python_version" in log_model_signature.parameters:
                requested_python = DEFAULT_PYTHON_VERSION
                log_kwargs.setdefault("python_version", requested_python)
        # With MLflow >= 2.20.0 and TypeFromExample type hints enabled, let MLflow
        # infer signature from hints and input_example. Do not pass signature to
        # avoid mismatch warnings as per docs.
        if not _USE_TYPEHINTS and signature is not None:
            log_kwargs["signature"] = signature
        if input_example_for_logging is not None:
            log_kwargs["input_example"] = input_example_for_logging

        mlflow_validate_env = os.environ.get("MLFLOW_VALIDATE_SERVING_INPUT")
        os.environ["MLFLOW_VALIDATE_SERVING_INPUT"] = "false"
        try:
            return mlflow.pyfunc.log_model(**log_kwargs)
        finally:
            if mlflow_validate_env is None:
                os.environ.pop("MLFLOW_VALIDATE_SERVING_INPUT", None)
            else:
                os.environ["MLFLOW_VALIDATE_SERVING_INPUT"] = mlflow_validate_env

    active_run = mlflow.active_run()
    if run_id and (active_run is None or active_run.info.run_id != run_id):
        with mlflow.start_run(run_id=run_id):
            return _log()
    return _log()


def log_driverless_scoring_pipeline_in_project(
    scoring_pipeline_dir: str,
    *,
    artifact_path: str = "driverless_ai_model",
    run_id: Optional[str] = None,
    apply_data_recipes: bool = False,
    scorer_kwargs: Optional[Mapping[str, Any]] = None,
    predict_kwargs: Optional[Mapping[str, Any]] = None,
    pip_requirements: Optional[Sequence[str]] = None,
    python_env: Optional[Union[str, os.PathLike, Mapping[str, Any]]] = None,
    extra_artifacts: Optional[Mapping[str, str]] = None,
    registered_model_name: Optional[str] = None,
    input_example_df: Optional[pd.DataFrame] = None,
    output_example_df: Optional[pd.DataFrame] = None,
    input_example_path: Optional[Union[str, os.PathLike]] = None,
    output_example_path: Optional[Union[str, os.PathLike]] = None,
) -> mlflow.models.model.ModelInfo:
    scoring_path, temp_dir = _resolve_scoring_path(scoring_pipeline_dir)
    try:
        return _log_driverless_scoring_pipeline_impl(
            scoring_path,
            artifact_path=artifact_path,
            run_id=run_id,
            apply_data_recipes=apply_data_recipes,
            scorer_kwargs=scorer_kwargs,
            predict_kwargs=predict_kwargs,
            pip_requirements=pip_requirements,
            python_env=python_env,
            extra_artifacts=extra_artifacts,
            registered_model_name=registered_model_name,
            input_example_df=input_example_df,
            output_example_df=output_example_df,
            input_example_path=input_example_path,
            output_example_path=output_example_path,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def log_driverless_scoring_pipeline(
    scoring_pipeline_dir: str,
    *,
    artifact_path: str = "driverless_ai_model",
    run_id: Optional[str] = None,
    apply_data_recipes: bool = False,
    scorer_kwargs: Optional[Mapping[str, Any]] = None,
    predict_kwargs: Optional[Mapping[str, Any]] = None,
    pip_requirements: Optional[Sequence[str]] = None,
    python_env: Optional[Union[str, os.PathLike, Mapping[str, Any]]] = None,
    extra_artifacts: Optional[Mapping[str, str]] = None,
    registered_model_name: Optional[str] = None,
    input_example_df: Optional[pd.DataFrame] = None,
    output_example_df: Optional[pd.DataFrame] = None,
    input_example_path: Optional[Union[str, os.PathLike]] = None,
    output_example_path: Optional[Union[str, os.PathLike]] = None,
) -> mlflow.models.model.ModelInfo:
    scoring_path, temp_dir = _resolve_scoring_path(scoring_pipeline_dir)
    temp_example_files: List[str] = []
    try:
        if os.environ.get(PROJECT_MODE_ENV) == "1" or not _should_launch_project():
            return _log_driverless_scoring_pipeline_impl(
                scoring_path,
                artifact_path=artifact_path,
                run_id=run_id,
                apply_data_recipes=apply_data_recipes,
                scorer_kwargs=scorer_kwargs,
                predict_kwargs=predict_kwargs,
                pip_requirements=pip_requirements,
                python_env=python_env,
                extra_artifacts=extra_artifacts,
                registered_model_name=registered_model_name,
                input_example_df=input_example_df,
                output_example_df=output_example_df,
                input_example_path=input_example_path,
                output_example_path=output_example_path,
            )

        project_input_path = input_example_path
        project_output_path = output_example_path

        if project_input_path is None and input_example_df is not None:
            tmp_input = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
            input_example_df.to_csv(tmp_input.name, index=False)
            tmp_input.flush()
            tmp_input.close()
            temp_example_files.append(tmp_input.name)
            project_input_path = tmp_input.name

        if project_output_path is None and output_example_df is not None:
            tmp_output = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
            output_example_df.to_csv(tmp_output.name, index=False)
            tmp_output.flush()
            tmp_output.close()
            temp_example_files.append(tmp_output.name)
            project_output_path = tmp_output.name

        return _launch_project_logging(
            scoring_path,
            artifact_path=artifact_path,
            run_id=run_id,
            apply_data_recipes=apply_data_recipes,
            scorer_kwargs=scorer_kwargs,
            predict_kwargs=predict_kwargs,
            pip_requirements=pip_requirements,
            python_env=python_env,
            extra_artifacts=extra_artifacts,
            registered_model_name=registered_model_name,
            input_example_path=project_input_path,
            output_example_path=project_output_path,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        for temp_file in temp_example_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass



def _parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log a Driverless AI scoring pipeline as an MLflow model"
    )
    parser.add_argument(
        "scoring_pipeline_dir", help="Path to the exported scoring-pipeline directory"
    )
    parser.add_argument(
        "--artifact-path", default="driverless_ai_model", help="MLflow artifact path"
    )
    parser.add_argument(
        "--run-id", default=None, help="Existing MLflow run ID to log into"
    )
    parser.add_argument(
        "--registered-model-name", default=None, help="Optional registered model name"
    )
    parser.add_argument(
        "--apply-data-recipes",
        action="store_true",
        help="Enable apply_data_recipes during scoring",
    )
    return parser.parse_args(args)


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(cli_args)
    model_info = log_driverless_scoring_pipeline(
        args.scoring_pipeline_dir,
        artifact_path=args.artifact_path,
        run_id=args.run_id,
        apply_data_recipes=args.apply_data_recipes,
        registered_model_name=args.registered_model_name,
    )
    print("Logged MLflow model:", model_info.model_uri)


if __name__ == "__main__":
    main()
