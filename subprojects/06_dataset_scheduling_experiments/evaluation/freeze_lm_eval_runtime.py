#!/usr/bin/env python3
"""Freeze the complete pinned lm-eval target used by the retention suite."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path


IGNORED_PARTS = {"__pycache__"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
EXPECTED_UENV = "pytorch/v2.9.1:v2"
EXPECTED_EXTERNAL = {
    "huggingface-hub": "0.36.0",
    "psutil": "7.1.0",
    "safetensors": "0.6.2",
    "torch": "2.9.1",
    "transformers": "4.57.0",
}
EXPECTED_GLOBAL_MMLU_GROUPS = [
    "global_mmlu_ar",
    "global_mmlu_bn",
    "global_mmlu_de",
    "global_mmlu_en",
    "global_mmlu_es",
    "global_mmlu_fr",
    "global_mmlu_hi",
    "global_mmlu_id",
    "global_mmlu_it",
    "global_mmlu_ja",
    "global_mmlu_ko",
    "global_mmlu_pt",
    "global_mmlu_sw",
    "global_mmlu_yo",
    "global_mmlu_zh",
]


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def parse_lock(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1 or any(marker in line for marker in (";", "[", "]")):
            raise ValueError(f"lock line {line_no} is not one unconditional exact pin")
        name, version = line.split("==", 1)
        key = canonical_name(name.strip())
        if not key or not version.strip() or key in pins:
            raise ValueError(f"invalid or duplicate lock line {line_no}")
        pins[key] = version.strip()
    if pins.get("lm-eval") != "0.4.11" or pins.get("accelerate") != "1.13.0":
        raise ValueError("retention runtime must pin lm-eval 0.4.11 and accelerate 1.13.0")
    return pins


def target_distributions(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(path=[str(root)]):
        name = canonical_name(distribution.metadata["Name"])
        if name in observed:
            raise ValueError(f"duplicate target distribution: {name}")
        observed[name] = distribution.version
    return observed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm-eval-root", type=Path, required=True)
    parser.add_argument("--requirements-lock", type=Path, required=True)
    parser.add_argument("--uenv-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.lm_eval_root.resolve()
    lock = args.requirements_lock.resolve()
    if args.uenv_image != EXPECTED_UENV:
        raise ValueError(f"retention uenv drift: {args.uenv_image}")
    pins = parse_lock(lock)
    distributions = target_distributions(root)
    if distributions != pins:
        missing = sorted(set(pins) - set(distributions))
        extra = sorted(set(distributions) - set(pins))
        changed = sorted(
            name
            for name in set(pins) & set(distributions)
            if pins[name] != distributions[name]
        )
        raise ValueError(
            f"target distributions differ from lock: missing={missing}, extra={extra}, changed={changed}"
        )
    external = {
        name: importlib.metadata.version(name) for name in sorted(EXPECTED_EXTERNAL)
    }
    if external != EXPECTED_EXTERNAL:
        raise ValueError(f"shared uenv distribution drift: {external}")
    package = root / "lm_eval"
    required = [package / "__main__.py", package / "evaluator.py", package / "tasks"]
    if not all(path.exists() for path in required):
        raise ValueError("lm-eval target install is incomplete or unrepaired")
    alias = package / "tasks" / "global_mmlu" / "_global_mmlu.yaml"
    if not alias.is_file():
        raise ValueError("frozen global_mmlu aggregate alias is missing")
    import yaml

    alias_config = yaml.safe_load(alias.read_text(encoding="utf-8"))
    if (
        alias_config.get("group") != "global_mmlu"
        or alias_config.get("task") != EXPECTED_GLOBAL_MMLU_GROUPS
    ):
        raise ValueError("global_mmlu aggregate alias drift")
    files = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() or item.is_symlink()):
        relative_path = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"lm-eval runtime must not contain symlinks: {path}")
        if any(part in IGNORED_PARTS for part in relative_path.parts) or path.suffix in IGNORED_SUFFIXES:
            continue
        relative = relative_path.as_posix()
        digest = sha256_file(path)
        row = {"relative_path": relative, "bytes": path.stat().st_size, "sha256": digest}
        files.append(row)
        aggregate.update((json.dumps(row, sort_keys=True) + "\n").encode())
    if len(files) < 100:
        raise ValueError("lm-eval source/task inventory is unexpectedly small")
    payload = {
        "schema_version": "apertus_mini_lm_eval_runtime_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(root),
        "requirements_lock": {
            "path": str(lock),
            "bytes": lock.stat().st_size,
            "sha256": sha256_file(lock),
        },
        "distributions": distributions,
        "runtime_environment": {
            "uenv_image": args.uenv_image,
            "python": sys.version,
            "external_distributions": external,
        },
        "custom_task_aliases": {
            "global_mmlu": {
                "relative_path": alias.relative_to(root).as_posix(),
                "bytes": alias.stat().st_size,
                "sha256": sha256_file(alias),
                "expands_to": EXPECTED_GLOBAL_MMLU_GROUPS,
                "reason": "official lm-eval 0.4.11 has language groups but no top-level alias",
            }
        },
        "files": files,
        "file_count": len(files),
        "tree_manifest_sha256": aggregate.hexdigest(),
        "task_list": [
            "arc_challenge",
            "arc_easy",
            "hellaswag",
            "winogrande",
            "piqa",
            "mmlu",
            "global_mmlu",
            "xnli",
            "xcopa"
        ],
        "same_suite_as_prior_replay_experiments": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "files": len(files)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
