#!/usr/bin/env python3
"""Build an importable ZIP of the package at the repository root.

This creates a ZIP whose root contains the package directory
`h2o_dai_py_scoring_mlflow/` so it can be imported via zipimport on
Databricks by adding the ZIP to `sys.path`.

Usage:
  python make_zip.py                 # writes dist/h2o_dai_py_scoring_mlflow.zip
  python make_zip.py -o out.zip      # custom output path
  python make_zip.py -p src/..       # custom package source directory
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import os
import zipfile


DEFAULT_SRC = Path("src/h2o_dai_py_scoring_mlflow")
DEFAULT_OUT = Path("dist/h2o_dai_py_scoring_mlflow.zip")

EXCLUDE_NAMES = {
    "__pycache__",
    ".DS_Store",
}
EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def should_exclude(path: Path) -> bool:
    name = path.name
    if name in EXCLUDE_NAMES:
        return True
    if any(name.endswith(suf) for suf in EXCLUDE_SUFFIXES):
        return True
    return False


def build_zip(src_dir: Path, out_zip: Path) -> None:
    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Package directory does not exist: {src_dir}")
    if not (src_dir / "__init__.py").exists():
        raise SystemExit(f"{src_dir} is missing __init__.py; refusing to build.")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    # Recreate shell behavior of `zip -r dist.zip src/h2o_dai_py_scoring_mlflow ...`
    # 1) include a directory entry for each folder
    # 2) include files under `src/h2o_dai_py_scoring_mlflow/` prefix
    root_prefix = Path("src") / src_dir.name
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # add root directory entry once (avoid duplicate when rel_dir is '.')
        zf.writestr(str(root_prefix) + "/", b"")
        for dirpath, dirnames, filenames in os.walk(src_dir):
            dp = Path(dirpath)
            # filter dirs in-place (skip excluded)
            dirnames[:] = [d for d in dirnames if not should_exclude(dp / d)]

            # add directory entry
            rel_dir = dp.relative_to(src_dir)
            arc_dir = (root_prefix / rel_dir)
            # only write non-root directory entries; root already added above
            if str(rel_dir) not in (".", ""):
                zf.writestr(str(arc_dir) + "/", b"")

            for fname in filenames:
                path = dp / fname
                if should_exclude(path):
                    continue
                rel = path.relative_to(src_dir)
                arcname = str(root_prefix / rel)
                zf.write(path, arcname)

    count = 0
    with zipfile.ZipFile(out_zip) as zf:
        count = len(zf.infolist())
    print(f"Wrote {out_zip} ({count} files). Import root: src/{src_dir.name}/")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create an importable package ZIP")
    ap.add_argument("-p", "--package", default=str(DEFAULT_SRC), help="Path to package directory")
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT), help="Output zip path")
    args = ap.parse_args()

    src_dir = Path(args.package).resolve()
    out_zip = Path(args.output).resolve()
    build_zip(src_dir, out_zip)


if __name__ == "__main__":
    main()
