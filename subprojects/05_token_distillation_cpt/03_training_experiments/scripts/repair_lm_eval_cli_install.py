#!/usr/bin/env python3
"""Repair the lm-eval target install used by retention sidecars.

The Clariden target install may contain source files for ``harness.py`` and
``run.py`` but only bytecode for the helper modules imported by that CLI.
Python does not import bytecode from ``__pycache__`` when the source file is
missing, so expose those pyc files at legacy sourceless-module locations and
add the tiny package initializer expected by ``lm_eval.__main__``.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path


ALLOWED_PYC_PACKAGES = (
    "lm_eval",
    "datasets",
    "pyarrow",
    "multiprocess",
    "pandas",
    "dateutil",
    "pytz",
    "accelerate",
    "sklearn",
    "scipy",
    "joblib",
    "tabulate",
    "xxhash",
)
ALLOWED_TOP_LEVEL_MODULES = ("threadpoolctl",)

GLOBAL_MMLU_DEFAULT_UTILS = """from functools import partial


CATEGORIES = ["Business", "Humanities", "Medical", "Other", "STEM", "Social Sciences"]


def process_docs(dataset, category):
    return dataset.filter(lambda x: x["subject_category"] == category)


process_functions = {
    f"process_{category.lower().replace(' ', '_')}": partial(
        process_docs, category=category
    )
    for category in CATEGORIES
}

globals().update(process_functions)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval",
        help="Target-install root that is added to PYTHONPATH.",
    )
    return parser.parse_args()


def ensure_legacy_pyc(pycache_file: Path) -> None:
    module = pycache_file.name.split(".cpython-", maxsplit=1)[0]
    legacy = pycache_file.parent.parent / f"{module}.pyc"
    if legacy.exists():
        return
    rel_src = os.path.relpath(pycache_file, start=legacy.parent)
    legacy.symlink_to(rel_src)


def expose_sourceless_modules(package_root: Path) -> int:
    created = 0
    for pycache in package_root.rglob("__pycache__"):
        latest_by_module: dict[str, Path] = {}
        for pycache_file in pycache.glob("*.cpython-*.pyc"):
            module = pycache_file.name.split(".cpython-", maxsplit=1)[0]
            latest_by_module[module] = pycache_file
        for pycache_file in latest_by_module.values():
            legacy = pycache_file.parent.parent / (
                pycache_file.name.split(".cpython-", maxsplit=1)[0] + ".pyc"
            )
            if not legacy.exists():
                ensure_legacy_pyc(pycache_file)
                created += 1
    return created


def expose_top_level_modules(target: Path) -> int:
    pycache = target / "__pycache__"
    if not pycache.exists():
        return 0
    created = 0
    for module in ALLOWED_TOP_LEVEL_MODULES:
        legacy = target / f"{module}.pyc"
        if legacy.exists():
            continue
        matches = sorted(pycache.glob(f"{module}.cpython-*.pyc"))
        if not matches:
            continue
        legacy.symlink_to(os.path.relpath(matches[-1], start=target))
        created += 1
    return created


def cleanup_disallowed_legacy_pycs(target: Path) -> int:
    removed = 0
    for legacy in target.rglob("*.pyc"):
        if not legacy.is_symlink():
            continue
        try:
            link_target = os.readlink(legacy)
            rel_parts = legacy.relative_to(target).parts
        except OSError:
            continue
        if "__pycache__" not in link_target:
            continue
        if not rel_parts or rel_parts[0] in ALLOWED_PYC_PACKAGES:
            continue
        legacy.unlink()
        removed += 1
    return removed


def restore_global_mmlu_default_utils(target: Path) -> int:
    """Restore source helpers required by YAML !function file imports.

    The task YAMLs import paths such as ``utils.process_business`` through
    ``spec_from_file_location(.../utils.py)``. A sourceless ``utils.pyc`` is not
    enough for that code path, even though normal Python imports can use it.
    """

    default_root = target / "lm_eval" / "tasks" / "global_mmlu" / "default"
    if not default_root.exists():
        return 0
    restored = 0
    for lang_dir in default_root.iterdir():
        if not lang_dir.is_dir():
            continue
        if not any((lang_dir / "__pycache__").glob("utils.cpython-*.pyc")):
            continue
        utils_py = lang_dir / "utils.py"
        if not utils_py.exists() or utils_py.read_text() != GLOBAL_MMLU_DEFAULT_UTILS:
            utils_py.write_text(GLOBAL_MMLU_DEFAULT_UTILS)
            restored += 1
    return restored


def main() -> int:
    args = parse_args()
    target = Path(args.target)
    cli_dir = target / "lm_eval" / "_cli"
    if not (target / "lm_eval" / "__main__.py").is_file():
        raise FileNotFoundError("missing lm_eval/__main__.py")
    if not (cli_dir / "harness.py").is_file():
        raise FileNotFoundError("missing lm_eval/_cli/harness.py")
    if not (cli_dir / "run.py").is_file():
        raise FileNotFoundError("missing lm_eval/_cli/run.py")

    init_py = cli_dir / "__init__.py"
    desired_init = (
        "from lm_eval._cli.harness import HarnessCLI\n\n"
        "__all__ = [\"HarnessCLI\"]\n"
    )
    if not init_py.exists() or init_py.read_text() != desired_init:
        init_py.write_text(desired_init)

    restored_sources = restore_global_mmlu_default_utils(target)
    removed = cleanup_disallowed_legacy_pycs(target)
    created = 0
    for package in ALLOWED_PYC_PACKAGES:
        package_root = target / package
        if package_root.exists():
            created += expose_sourceless_modules(package_root)
    created += expose_top_level_modules(target)

    importlib.invalidate_caches()
    imported = importlib.import_module("lm_eval._cli")
    if not hasattr(imported, "HarnessCLI"):
        raise RuntimeError("lm_eval._cli still does not export HarnessCLI")
    importlib.import_module("lm_eval.__main__")
    from lm_eval.config.evaluate_config import EvaluatorConfig  # noqa: F401

    print(
        f"lm-eval CLI import OK from {target}; "
        f"exposed {created} pyc modules; "
        f"restored {restored_sources} source helpers; "
        f"removed {removed} broad symlinks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
