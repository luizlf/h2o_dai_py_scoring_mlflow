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


_logger = logging.getLogger(__name__)


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


class DriverlessAIScoringModel(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper around a Driverless AI Python scoring pipeline."""

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
        # Defer heavy imports (and any license checks) to first predict() call so
        # model servers can load without requiring external configuration.
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

    def predict(
        self,
        context: mlflow.pyfunc.model.PythonModelContext,
        model_input: Union[pd.DataFrame, Mapping[str, Any], Sequence[Mapping[str, Any]]],
    ) -> pd.DataFrame:
        self._ensure_scorer_initialized()

        frame = self._to_dataframe(model_input)
        predict_kwargs = dict(self.predict_kwargs)
        predict_kwargs.setdefault("apply_data_recipes", self.apply_data_recipes)

        predictions = self.scorer.score_batch(frame, **predict_kwargs)
        return _ensure_pandas_frame(predictions)

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
        # Make sure logging streams behave for native extensions
        _patch_streams_fileno()
        scorer_module = importlib.import_module(self.scorer_module_name)
        scorer_cls = getattr(scorer_module, "Scorer")
        self.scorer = scorer_cls(**self.scorer_kwargs)
        raw_columns = self.scorer.get_column_names()
        self.input_columns = self._normalize_column_labels(raw_columns)

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
        exclusion list via MLFLOW_DRIVERLESS_EXCLUDE_PACKAGES (comma-separated
        names).
    """

    def _parse_excludes() -> List[str]:
        # Drop packages that are unnecessary for CPU inference or known to
        # cause heavy/fragile builds or dependency conflicts.
        default = ["h2o4gpu", "pyorc"]
        env = os.environ.get("MLFLOW_DRIVERLESS_EXCLUDE_PACKAGES", "").strip()
        if not env:
            return default
        extra = [p.strip() for p in env.split(",") if p.strip()]
        # preserve order and drop dups
        return [x for x in default + extra if x]

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

    exclude_pkgs = set(x.lower() for x in _parse_excludes())

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
    if os.environ.get("MLFLOW_DRIVERLESS_DISABLE_IMPORTLIB_RESOURCES", "0").strip() != "1":
        ir_ver = os.environ.get("MLFLOW_DRIVERLESS_IMPORTLIB_RESOURCES_VERSION", "5.12.0").strip()
        compat_pkgs.append(f"importlib-resources=={ir_ver}")
    if os.environ.get("MLFLOW_DRIVERLESS_DISABLE_PYSPARK", "0").strip() != "1":
        pyspark_ver = os.environ.get("MLFLOW_DRIVERLESS_PYSPARK_VERSION", "3.3.2").strip()
        compat_pkgs.append(f"pyspark=={pyspark_ver}")

    combined = package_specs + discovered_wheels + compat_pkgs
    if include_mlflow:
        combined.append(f"mlflow=={mlflow.__version__}")
    if extra_requirements:
        # Drop any excluded packages that may have been explicitly added
        combined.extend([r for r in extra_requirements if _pkg_name(r) not in exclude_pkgs])

    return _deduplicate_preserve_order(combined)


DEFAULT_PYTHON_VERSION = "3.8.12"


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
    if os.environ.get("MLFLOW_DRIVERLESS_DISABLE_LIBMAGIC", "0").strip() != "1":
        if not any(str(d).startswith("libmagic") for d in dependencies if isinstance(d, str)):
            dependencies.append("libmagic")
    if pip_packages:
        dependencies.append({"pip": pip_packages})

    return {
        "name": "driverless_ai_scoring",
        "channels": ["conda-forge", "defaults"],
        "dependencies": dependencies,
    }


PROJECT_MODE_ENV = "MLFLOW_DRIVERLESS_PROJECT_MODE"
MODEL_INFO_ARTIFACT = "_driverless/model_info.json"
REQUIRED_PYTHON_VERSION = (3, 8)
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
    if os.environ.get(PROJECT_MODE_ENV) == "1":
        return False
    if os.environ.get("MLFLOW_DRIVERLESS_DISABLE_PROJECT") == "1":
        return False
    if os.environ.get("MLFLOW_DRIVERLESS_FORCE_PROJECT") == "1":
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
    elif os.environ.get("MLFLOW_DRIVERLESS_FORCE_CONDA", "0").strip() == "1":
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

    signature: Optional[ModelSignature] = None
    if input_example_final is not None:
        try:
            if resolved_output_example is not None:
                signature = infer_signature(input_example_final, resolved_output_example)
            else:
                inferred = infer_signature(input_example_final)
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
                        and name not in normalized_columns
                    ]
                    output_name = (
                        numeric_candidates[0] if numeric_candidates else "prediction"
                    )
                output_schema = Schema([ColSpec(type="double", name=str(output_name))])
                signature = ModelSignature(inputs=inferred.inputs, outputs=output_schema)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning("Unable to infer signature from input example: %s", exc)
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
        if signature is not None:
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
