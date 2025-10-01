"""Runtime tweaks for pyfunc model environments (Python 3.8 on Databricks).

This module is auto-imported by Python when present on sys.path (via 'site').
We use it to:
  1) Provide importlib.resources.files/as_file on Python 3.8 by wiring in the
     importlib-resources backport when needed.
  2) Ensure the model's own pip-installed PySpark is imported instead of the
     Databricks runtime PySpark that targets newer Python. We drop DBR's Spark
     paths from sys.path, letting pip pyspark take precedence.
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
