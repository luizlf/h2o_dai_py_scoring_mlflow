"""Runtime tweaks for pyfunc model environments (Python 3.8 on Databricks).

This module is auto-imported by Python when present on sys.path (via 'site').
We use it to:
  1) Provide importlib.resources.files/as_file on Python 3.8 by wiring in the
     importlib-resources backport when needed.
  2) Ensure the model's own pip-installed PySpark is imported instead of the
     Databricks runtime PySpark that targets newer Python. We drop DBR's Spark
     paths from sys.path, letting pip pyspark take precedence.
  3) On Linux, ensure the active environment's lib directory is on
     LD_LIBRARY_PATH so native libraries (e.g., OpenBLAS via conda 'libopenblas')
     are discoverable in Serving containers.
"""

from __future__ import annotations

import os
import sys

# 1) Patch importlib.resources on Py3.8 with the backport's APIs
try:
    import importlib.resources as stdlib_resources  # type: ignore
except Exception:  # pragma: no cover
    stdlib_resources = None  # type: ignore

try:
    import importlib_resources as backport_resources  # type: ignore
except Exception:  # pragma: no cover
    backport_resources = None  # type: ignore

if stdlib_resources is not None and backport_resources is not None:
    if not hasattr(stdlib_resources, "files"):
        try:
            stdlib_resources.files = backport_resources.files  # type: ignore[attr-defined]
        except Exception:
            pass
    if not hasattr(stdlib_resources, "as_file"):
        try:
            stdlib_resources.as_file = backport_resources.as_file  # type: ignore[attr-defined]
        except Exception:
            pass

# 2) Remove Databricks runtime Spark paths so pip pyspark imports cleanly
def _strip_dbr_spark_paths() -> None:
    markers = (
        "/databricks/spark/python",
        "/databricks/python",
    )
    try:
        sys.path[:] = [p for p in sys.path if not any(m in (p or "") for m in markers)]
    except Exception:
        pass

    # Also neutralize inherited env vars that re-inject DBR Spark on child processes
    for key in ("PYTHONPATH", "SPARK_HOME", "PYSPARK_PYTHON", "PYSPARK_DRIVER_PYTHON"):
        if key in os.environ and "/databricks" in os.environ.get(key, ""):
            os.environ.pop(key, None)

_strip_dbr_spark_paths()

# 3) Some runtimes wrap sys.stdout/stderr with objects that don't expose
#    fileno(), while native libs call it. Provide a minimal shim.
def _ensure_streams_have_fileno() -> None:
    import sys as _sys
    import io as _io
    class _WithFileno:
        def __init__(self, base, fd):
            self._b = base
            self._fd = fd
        def fileno(self):
            return self._fd
        def __getattr__(self, name):
            return getattr(self._b, name)
    try:
        if not hasattr(_sys.stdout, 'fileno') or not callable(getattr(_sys.stdout, 'fileno', None)):
            _sys.stdout = _WithFileno(_sys.stdout, 1)  # type: ignore
        if not hasattr(_sys.stderr, 'fileno') or not callable(getattr(_sys.stderr, 'fileno', None)):
            _sys.stderr = _WithFileno(_sys.stderr, 2)  # type: ignore
    except Exception:
        pass

_ensure_streams_have_fileno()

# 4) Some Driverless versions access psutil.RLIM_INFINITY, which is not
#    exposed by newer psutil releases. Provide a compatibility shim by
#    pulling the constant from Python's 'resource' module.
def _ensure_psutil_rlim_infinity() -> None:
    try:
        import psutil  # type: ignore
        if not hasattr(psutil, "RLIM_INFINITY"):
            try:
                import resource as _resource  # type: ignore
                value = getattr(_resource, "RLIM_INFINITY", -1)
            except Exception:
                value = -1
            try:
                setattr(psutil, "RLIM_INFINITY", value)
            except Exception:
                pass
    except Exception:
        pass

_ensure_psutil_rlim_infinity()

# 5) Broader psutil RLIMIT constants shim: define missing RLIMIT_* names as None.
def _ensure_psutil_rlimit_constants() -> None:
    try:
        import psutil  # type: ignore
        names = [
            "RLIM_INFINITY",
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
        ]
        for name in names:
            if not hasattr(psutil, name):
                try:
                    setattr(psutil, name, None)
                except Exception:
                    pass
    except Exception:
        pass

_ensure_psutil_rlimit_constants()

# 4) Ensure the env's lib path is on LD_LIBRARY_PATH for OpenBLAS discovery
def _ensure_ld_library_path() -> None:
    try:
        if os.name != "posix":
            return
        # Prefer CONDA_PREFIX when available, else sys.prefix
        prefixes = []
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            prefixes.append(conda_prefix)
        prefixes.append(sys.prefix)
        lib_dirs = []
        for p in prefixes:
            if not p:
                continue
            cand = os.path.join(p, "lib")
            if os.path.isdir(cand):
                lib_dirs.append(cand)
        if not lib_dirs:
            return
        current = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [x for x in current.split(":") if x]
        changed = False
        for d in lib_dirs:
            if d not in parts:
                parts.append(d)
                changed = True
        if changed:
            os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    except Exception:
        # Best effort only
        pass

_ensure_ld_library_path()


# 5) Diagnostics to help identify missing native libs in Serving
def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[DriverlessDiag] {msg}\n")
    except Exception:
        try:
            print(f"[DriverlessDiag] {msg}")
        except Exception:
            pass


def _list_matching_files(root: str, prefixes: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    try:
        for name in os.listdir(root):
            if any(name.startswith(p) for p in prefixes):
                out.append(os.path.join(root, name))
    except Exception:
        pass
    return out[:10]


def _diagnose_native_libs() -> None:
    try:
        import ctypes
        import ctypes.util
    except Exception:
        return

    try:
        import numpy as _np  # type: ignore

        cfg = getattr(_np, "__config__", None)
        if cfg is not None:
            info = {}
            for key in ("openblas_info", "blas_ilp64_info", "blas_opt_info", "lapack_opt_info"):
                try:
                    info[key] = cfg.get_info(key)
                except Exception:
                    pass
            _log(f"numpy={_np.__version__} build_info={{{k: bool(v) for k,v in info.items()}}}")
    except Exception:
        pass

    conda_prefix = os.environ.get("CONDA_PREFIX")
    prefixes = [p for p in (conda_prefix, sys.prefix) if p]
    lib_dirs = [os.path.join(p, "lib") for p in prefixes if os.path.isdir(os.path.join(p, "lib"))]
    _log(f"sys.prefix={sys.prefix} CONDA_PREFIX={conda_prefix}")
    _log(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH','')}")
    if lib_dirs:
        _log(f"lib_dirs={lib_dirs}")
        for d in lib_dirs:
            matches = _list_matching_files(d, ("libopenblas", "libmagic"))
            if matches:
                _log(f"found in {d}: {matches}")

    def _try_dlopen(libname: str) -> None:
        try:
            path = ctypes.util.find_library(libname)
            _log(f"find_library('{libname}') -> {path}")
            if path:
                ctypes.CDLL(path)
                _log(f"dlopen ok: {path}")
        except Exception as exc:
            _log(f"dlopen failed for {libname}: {exc}")

    for name in ("openblas", "blas", "lapack", "magic"):
        _try_dlopen(name)

    # Do not preload additional OpenBLAS builds from system/conda lib here; preloading
    # (if required) is handled by the model runtime to avoid mixing backends.


if os.environ.get("H2O_DAI_MLFLOW_DIAG", "1").strip() != "0":
    _diagnose_native_libs()
