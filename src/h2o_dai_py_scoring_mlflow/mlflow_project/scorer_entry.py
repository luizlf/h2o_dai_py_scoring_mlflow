#!/usr/bin/env python3
"""Utility entrypoint for scoring or logging the Driverless AI pipeline via MLflow Projects."""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type

try:
    import importlib.resources as stdlib_resources
except ImportError:  # pragma: no cover - Py<3.7 not supported
    stdlib_resources = None  # type: ignore[assignment]

try:  # Py 3.8 needs the backport for files()/as_file()
    import importlib_resources as backport_resources  # type: ignore
except ImportError:
    backport_resources = None

if stdlib_resources is not None and backport_resources is not None:
    if not hasattr(stdlib_resources, "files"):
        stdlib_resources.files = backport_resources.files  # type: ignore[attr-defined]
    if not hasattr(stdlib_resources, "as_file"):
        stdlib_resources.as_file = backport_resources.as_file  # type: ignore[attr-defined]

import mlflow
import pandas as pd

from h2o_dai_py_scoring_mlflow.mlflow_driverless.deployment import (
    log_driverless_scoring_pipeline_in_project,
)


SCORING_WHEEL_MARKER_ENV = "MLFLOW_DRIVERLESS_WHEEL_SOURCE"
_SCORING_ENV_VARS = (
    "SCORING_PIPELINE_DIR",
    "DRIVERLESS_SCORING_PIPELINE_DIR",
    "DRIVERLESS_SCORING_DIR",
    "DRIVERLESS_AI_SCORING_PIPELINE_DIR",
)


def _candidate_roots() -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    roots: List[Path] = []
    seen: set[str] = set()

    def add(root: Path) -> None:
        resolved = root if root.is_absolute() else root.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)

    cwd = Path.cwd().resolve()
    add(cwd)
    for parent in cwd.parents:
        add(parent)

    add(script_dir)
    for parent in script_dir.parents:
        add(parent)

    home = Path.home().resolve()
    add(home)
    add(home / "mlflow_proj")

    return roots


def _generate_scoring_candidates(raw_value: str) -> List[Path]:
    normalized = (raw_value or "").strip()
    candidates: List[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        expanded = path.expanduser()
        key = str(expanded)
        if key not in seen:
            seen.add(key)
            candidates.append(expanded)

    if normalized and normalized not in {"__AUTO__", "__NONE__"}:
        direct = Path(normalized)
        add(direct)
        if direct.suffix != ".zip":
            add(direct.with_suffix(".zip"))
    else:
        add(Path("scoring-pipeline"))
        add(Path("scoring-pipeline.zip"))

    for env_name in _SCORING_ENV_VARS:
        env_value = os.environ.get(env_name)
        if not env_value:
            continue
        env_path = Path(env_value)
        add(env_path)
        if env_path.suffix != ".zip":
            add(env_path.with_suffix(".zip"))

    for root in _candidate_roots():
        add(root / "scoring-pipeline")
        add(root / "scoring-pipeline.zip")
        add(root / "mlflow_proj" / "scoring-pipeline")
        add(root / "mlflow_proj" / "scoring-pipeline.zip")

    return candidates


def _materialize_scoring_dir(
    path: Path,
) -> Tuple[Path, Optional[tempfile.TemporaryDirectory], str]:
    if path.is_dir():
        resolved = path.resolve()
        return resolved, None, str(resolved)
    if path.is_file() and path.suffix == ".zip":
        temp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(path) as archive:
            archive.extractall(temp_dir.name)
        extracted_root = Path(temp_dir.name)
        subdirs = [p for p in extracted_root.iterdir() if p.is_dir()]
        staging = subdirs[0] if len(subdirs) == 1 else extracted_root
        return staging.resolve(), temp_dir, str(path.resolve())
    raise FileNotFoundError(f"Scoring pipeline path does not exist: {path}")


def _locate_scoring_pipeline(
    raw_value: str,
) -> Tuple[Path, Optional[tempfile.TemporaryDirectory], str]:
    for candidate in _generate_scoring_candidates(raw_value):
        if candidate.exists():
            return _materialize_scoring_dir(candidate)
    raise FileNotFoundError(
        f"Could not locate scoring pipeline directory for value '{raw_value}'."
    )


@contextlib.contextmanager
def _scoring_pipeline(raw_value: str) -> Iterator[Tuple[Path, str]]:
    scoring_dir, temp_dir, marker = _locate_scoring_pipeline(raw_value)
    try:
        yield scoring_dir, marker
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _collect_wheel_paths(scoring_dir: Path) -> List[Path]:
    req_path = scoring_dir / "requirements.txt"
    wheels: List[Path] = []
    seen: set[str] = set()

    if req_path.exists():
        for line in req_path.read_text().splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            if entry.lower().endswith(".whl"):
                wheel_path = (scoring_dir / entry).resolve()
                if wheel_path.exists():
                    key = str(wheel_path)
                    if key not in seen:
                        seen.add(key)
                        wheels.append(wheel_path)

    if not wheels:
        for wheel in sorted(scoring_dir.glob("*.whl")):
            resolved = wheel.resolve()
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                wheels.append(resolved)

    return wheels


def _install_scoring_wheels(scoring_dir: Path, marker: str) -> None:
    if os.environ.get(SCORING_WHEEL_MARKER_ENV) == marker:
        return

    wheel_paths = _collect_wheel_paths(scoring_dir)
    if not wheel_paths:
        raise FileNotFoundError(
            f"No wheel files were found in the scoring pipeline at {scoring_dir}."
        )

    command = [sys.executable, "-m", "pip", "install", "--no-deps"]
    command.extend(str(path) for path in wheel_paths)

    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Failed to install Driverless AI scoring pipeline wheels"
        ) from exc

    os.environ[SCORING_WHEEL_MARKER_ENV] = marker


def _discover_scorer_module_name(scoring_dir: Path) -> str:
    candidates = sorted(scoring_dir.glob("scoring_h2oai_experiment_*.whl"))
    if not candidates:
        raise FileNotFoundError(
            f"Unable to locate scoring_h2oai_experiment_*.whl inside {scoring_dir}"
        )
    return candidates[-1].name.split("-")[0]


def _load_scorer_class(scoring_dir: Path) -> Type[Any]:
    module_name = _discover_scorer_module_name(scoring_dir)
    module = importlib.import_module(module_name)
    scorer_cls = getattr(module, "Scorer", None)
    if scorer_cls is None:
        raise AttributeError(f"Driverless module {module_name} does not expose Scorer")
    return scorer_cls


def _resolve_path(raw_path: str) -> Path:
    """Resolve DBFS-style paths so the CLI can run inside Databricks clusters."""
    if raw_path.startswith("dbfs:/"):
        return Path("/dbfs") / raw_path[len("dbfs:/") :].lstrip("/")
    return Path(raw_path)


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsn"}:
        return pd.read_json(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".jay":
        import datatable as dt  # type: ignore

        return dt.Frame(str(path)).to_pandas()
    raise ValueError(f"Unsupported input format for {path!s}")


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv" or suffix == "":
        df.to_csv(path, index=False)
        return
    if suffix in {".json", ".jsn"}:
        df.to_json(path, orient="split", index=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    if suffix == ".jay":
        import datatable as dt  # type: ignore

        dt.Frame(df).to_jay(str(path))
        return
    raise ValueError(f"Unsupported output format for {path!s}")


def _parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret {value!r} as boolean")


def _score_command(args: argparse.Namespace) -> int:
    input_path = _resolve_path(args.input_path)
    output_path = _resolve_path(args.output_path)
    apply_data_recipes = _parse_bool(args.apply_data_recipes)

    try:
        batch_size = int(args.batch_size)
    except (TypeError, ValueError):
        batch_size = 0
    if batch_size <= 0:
        batch_size = None

    df = _load_frame(input_path)

    with _scoring_pipeline(args.scoring_dir) as (scoring_dir, marker):
        _install_scoring_wheels(scoring_dir, marker)

        # Ensure a prebuilt pyorc wheel is available to avoid source builds in
        # Serving containers. Build it here (inside the project-managed Py 3.8
        # env) and drop the wheel into the scoring directory so the logger will
        # reference it explicitly.
        try:
            if os.environ.get("MLFLOW_DRIVERLESS_BUILD_PYORC", "1").strip() != "0":
                have_pyorc = any(p.name.startswith("pyorc-") and p.suffix == ".whl" for p in scoring_dir.glob("pyorc-*.whl"))
                if not have_pyorc:
                    # Preferred version (configurable), fallback to any
                    pref = os.environ.get("MLFLOW_DRIVERLESS_PYORC_VERSION", "0.9.0").strip()
                    candidates = [f"pyorc=={pref}", "pyorc"] if pref else ["pyorc"]
                    for spec in candidates:
                        cmd = [sys.executable, "-m", "pip", "wheel", spec, "-w", str(scoring_dir)]
                        try:
                            subprocess.check_call(cmd)
                            break
                        except subprocess.CalledProcessError:
                            continue
        except Exception:  # best-effort
            pass
        scorer_cls = _load_scorer_class(scoring_dir)
        scorer = scorer_cls()
        try:
            if batch_size and len(df) > batch_size:
                slices = []
                for start in range(0, len(df), batch_size):
                    stop = start + batch_size
                    slices.append(
                        scorer.score_batch(
                            df.iloc[start:stop], apply_data_recipes=apply_data_recipes
                        )
                    )
                preds = pd.concat(slices, ignore_index=True)
            else:
                preds = scorer.score_batch(df, apply_data_recipes=apply_data_recipes)
        finally:
            try:
                scorer.finish()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_frame(preds, output_path)
    return 0


def _load_json(value: str, default: Optional[Any] = None) -> Any:
    text = (value or "").strip()
    if not text or text == "__NONE__":
        return default
    try:
        decoded = base64.urlsafe_b64decode(text.encode("utf-8"))
        text = decoded.decode("utf-8")
    except Exception:  # value was likely already plain JSON
        pass
    return json.loads(text)


def _log_model_command(args: argparse.Namespace) -> int:
    apply_data_recipes = _parse_bool(args.apply_data_recipes)
    run_id = args.run_id or None
    registered_name = args.registered_model_name or None

    scorer_kwargs: Dict[str, Any] = _load_json(args.scorer_kwargs_json, default={}) or {}
    predict_kwargs: Dict[str, Any] = _load_json(args.predict_kwargs_json, default={}) or {}
    extra_artifacts: Dict[str, str] = _load_json(args.extra_artifacts_json, default={}) or {}
    pip_requirements_list = _load_json(args.pip_requirements_json, default=None)
    if pip_requirements_list is not None and not isinstance(pip_requirements_list, list):
        raise ValueError("pip_requirements_json must decode to a list of strings")
    if pip_requirements_list:
        pip_requirements = [str(item) for item in pip_requirements_list]
    else:
        pip_requirements = None

    python_env_path = args.python_env_path.strip()
    python_env_arg: Optional[str] = (
        python_env_path if python_env_path and python_env_path != "__NONE__" else None
    )

    input_example_df = None
    output_example_df = None
    input_example_path: Optional[Path] = None
    output_example_path: Optional[Path] = None

    if args.input_path and args.input_path != "__NONE__":
        input_example_path = _resolve_path(args.input_path)
        input_example_df = _load_frame(input_example_path)

    if args.output_path and args.output_path != "__NONE__":
        output_example_path = _resolve_path(args.output_path)
        output_example_df = _load_frame(output_example_path)

    extra_artifacts = dict(extra_artifacts)
    if input_example_path and input_example_path.exists():
        extra_artifacts.setdefault(
            "examples/input_example", str(input_example_path)
        )
    if output_example_path and output_example_path.exists():
        extra_artifacts.setdefault(
            "examples/output_example", str(output_example_path)
        )

    with _scoring_pipeline(args.scoring_dir) as (scoring_dir, marker):
        _install_scoring_wheels(scoring_dir, marker)

        # Signal to the helper that we are already running inside the project-managed env.
        os.environ["MLFLOW_DRIVERLESS_PROJECT_MODE"] = "1"

        model_info = log_driverless_scoring_pipeline_in_project(
            scoring_pipeline_dir=str(scoring_dir),
            artifact_path=args.artifact_path,
            run_id=run_id,
            apply_data_recipes=apply_data_recipes,
            scorer_kwargs=scorer_kwargs,
            predict_kwargs=predict_kwargs,
            pip_requirements=pip_requirements,
            python_env=python_env_arg,
            extra_artifacts=extra_artifacts,
            registered_model_name=registered_name,
            input_example_df=input_example_df,
            output_example_df=output_example_df,
            input_example_path=str(input_example_path) if input_example_path else None,
            output_example_path=str(output_example_path) if output_example_path else None,
        )

    payload = {
        "model_uri": model_info.model_uri,
        "run_id": model_info.run_id,
        "artifact_path": model_info.artifact_path,
        "registered_model_name": registered_name,
    }

    mlflow.log_dict(payload, "_driverless/model_info.json")
    print(json.dumps(payload, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Driverless AI scoring utilities for MLflow Projects"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score a dataset with the pipeline")
    score_parser.add_argument("--scoring-dir", default="scoring-pipeline")
    score_parser.add_argument("--input-path", required=True)
    score_parser.add_argument("--output-path", required=True)
    score_parser.add_argument("--apply-data-recipes", default="false")
    score_parser.add_argument("--batch-size", default="0")
    score_parser.set_defaults(func=_score_command)

    log_parser = subparsers.add_parser(
        "log-model", help="Log the scoring pipeline as an MLflow pyfunc model"
    )
    log_parser.add_argument("--scoring-dir", default="scoring-pipeline")
    log_parser.add_argument("--artifact-path", default="driverless_ai_model")
    log_parser.add_argument("--registered-model-name", default="")
    log_parser.add_argument("--run-id", default="")
    log_parser.add_argument("--apply-data-recipes", default="false")
    log_parser.add_argument("--scorer-kwargs-json", default="{}")
    log_parser.add_argument("--predict-kwargs-json", default="{}")
    log_parser.add_argument("--extra-artifacts-json", default="{}")
    log_parser.add_argument("--pip-requirements-json", default="[]")
    log_parser.add_argument("--python-env-path", default="__NONE__")
    log_parser.add_argument("--input-path", default="__NONE__")
    log_parser.add_argument("--output-path", default="__NONE__")
    log_parser.set_defaults(func=_log_model_command)

    return parser


def main(argv: Iterable[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv))
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
