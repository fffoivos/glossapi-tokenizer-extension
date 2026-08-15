#!/usr/bin/env python3
"""Freeze and verify the only permitted training-time Megatron patch set."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic


UPSTREAM_COMMIT = "c92402e39ef3c8e69ea378a59e79059dc14541f4"
EXTRA_VALID_SHA256 = "2e6810fa8b6c25597ccb3bcb9dc1ff5bf843ead2337e3edde0344605a23ec4c6"
PATCHED_FILES = (
    "megatron/training/arguments.py",
    "megatron/training/training.py",
    "pretrain_gpt.py",
)
HELPERS_MODULE = "megatron.core.datasets.helpers_cpp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=not binary,
    )


def inventory(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in PATCHED_FILES:
        path = root / relative
        require(path.is_file(), f"patched Megatron file missing: {path}")
        rows.append({
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def dataset_helper(root: Path) -> Path:
    candidates = sorted((root / "megatron/core/datasets").glob("helpers_cpp*.so"))
    require(len(candidates) == 1, f"expected exactly one compiled Megatron dataset helper, found {len(candidates)}")
    return candidates[0].resolve()


def helper_import_smoke(root: Path, helper: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    observed = subprocess.check_output(
        [
            sys.executable,
            "-c",
            f"import {HELPERS_MODULE} as helper; print(helper.__file__)",
        ],
        text=True,
        env=environment,
    ).strip()
    require(Path(observed).resolve() == helper, "compiled Megatron dataset helper import resolved to the wrong binary")


def validate_runtime(
    receipt: dict[str, object],
    root: Path,
    patch: Path,
    *,
    require_helpers: bool = False,
) -> None:
    require(receipt.get("schema_version") == "apertus_targeted_training_megatron_v1", "Megatron receipt schema drift")
    require(receipt.get("status") == "frozen", "Megatron receipt is not frozen")
    require(Path(str(receipt.get("output_root", ""))).resolve() == root.resolve(), "Megatron output-root binding drift")
    require(receipt.get("upstream_commit") == UPSTREAM_COMMIT, "Megatron upstream commit drift")
    require(receipt.get("patch_sha256") == EXTRA_VALID_SHA256, "Megatron patch receipt drift")
    require(sha256_file(patch) == EXTRA_VALID_SHA256, "live Megatron patch drift")
    require(git(root, "rev-parse", "HEAD").strip() == UPSTREAM_COMMIT, "live Megatron HEAD drift")
    changed = tuple(sorted(git(root, "diff", "--name-only").splitlines()))
    require(changed == tuple(sorted(PATCHED_FILES)), "live Megatron changed-file set drift")
    require(not git(root, "diff", "--check").strip(), "live Megatron diff-check failed")
    require(not git(root, "ls-files", "--others", "--exclude-standard").strip(), "live Megatron has untracked files")
    diff = git(root, "diff", "--binary", binary=True)
    require(hashlib.sha256(diff).hexdigest() == receipt.get("git_diff_sha256"), "live Megatron diff hash drift")
    observed = inventory(root)
    require(observed == receipt.get("patched_files"), "live Megatron patched-file inventory drift")
    require("extra_valid_datasets_provider" in (root / "pretrain_gpt.py").read_text(encoding="utf-8"), "named validation provider missing")
    require("--extra-valid-data-path" in (root / "megatron/training/arguments.py").read_text(encoding="utf-8"), "named validation CLI missing")
    helpers = receipt.get("dataset_helpers")
    if require_helpers or helpers is not None:
        require(isinstance(helpers, dict), "compiled Megatron dataset-helper receipt missing")
        helper = dataset_helper(root)
        require(helpers.get("module") == HELPERS_MODULE, "Megatron dataset-helper module drift")
        require(helpers.get("binary") == file_binding(helper), "Megatron dataset-helper binary drift")
        require(helpers.get("import_smoke_passed") is True, "Megatron dataset-helper import smoke missing")
        require(isinstance(helpers.get("python_version"), str) and helpers["python_version"], "Megatron dataset-helper Python version missing")
        require(isinstance(helpers.get("python_cache_tag"), str) and helpers["python_cache_tag"], "Megatron dataset-helper cache tag missing")
        require(isinstance(helpers.get("extension_suffix"), str) and helpers["extension_suffix"], "Megatron dataset-helper extension suffix missing")


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable Megatron receipt exists: {args.output}")
    source = args.source_root.resolve()
    root = args.output_root.resolve()
    patch = args.patch.resolve()
    require(source.is_dir() and root.is_dir(), "Megatron source/output root missing")
    require(patch.is_file() and sha256_file(patch) == EXTRA_VALID_SHA256, "named extra-validation patch drift")
    require(git(source, "rev-parse", "HEAD").strip() == UPSTREAM_COMMIT, "source Megatron commit drift")
    require(not git(source, "status", "--porcelain").strip(), "source Megatron checkout is dirty")
    require(git(root, "rev-parse", "HEAD").strip() == UPSTREAM_COMMIT, "output Megatron commit drift")
    changed = tuple(sorted(git(root, "diff", "--name-only").splitlines()))
    require(changed == tuple(sorted(PATCHED_FILES)), "patched Megatron changed-file set drift")
    require(not git(root, "diff", "--check").strip(), "patched Megatron diff-check failed")
    require(not git(root, "ls-files", "--others", "--exclude-standard").strip(), "patched Megatron has untracked files")
    helper = dataset_helper(root)
    helper_import_smoke(root, helper)
    diff = git(root, "diff", "--binary", binary=True)
    payload: dict[str, object] = {
        "schema_version": "apertus_targeted_training_megatron_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_root": str(source),
        "output_root": str(root),
        "upstream_commit": UPSTREAM_COMMIT,
        "patch": file_binding(patch),
        "patch_sha256": EXTRA_VALID_SHA256,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "patched_files": inventory(root),
        "dataset_helpers": {
            "module": HELPERS_MODULE,
            "binary": file_binding(helper),
            "python_version": sys.version.split()[0],
            "python_cache_tag": sys.implementation.cache_tag,
            "extension_suffix": str(sysconfig.get_config_var("EXT_SUFFIX")),
            "import_smoke_passed": True,
        },
        "named_extra_validation_only": True,
        "exact_eval_iteration_patch_applied": False,
        "executing_code_bundle": executing_code_bundle(),
    }
    validate_runtime(payload, root, patch, require_helpers=True)
    write_json_atomic(args.output, payload)
    print(json.dumps({
        "ok": True,
        "output_root": str(root),
        "git_diff_sha256": payload["git_diff_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
