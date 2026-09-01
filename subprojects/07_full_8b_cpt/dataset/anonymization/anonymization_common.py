#!/usr/bin/env python3
"""Shared, receipt-bound helpers for the full-CPT anonymized derivative."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
BRIDGE_DIR = (
    REPO_ROOT
    / "subprojects/05_token_distillation_cpt/05_training_dataset_bridge/scripts"
)
MASKER_DIR = (
    REPO_ROOT
    / "subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/scripts"
)
for directory in (BRIDGE_DIR, MASKER_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from bridge_common import (  # noqa: E402
    bound_code_sha,
    canonical_sha256,
    load_exclusion_ids,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


OVERLAY_SCHEMA = "full_cpt_anonymization_overlay_v1"
INVENTORY_SCHEMA = "full_cpt_anonymization_task_inventory_v1"
DEDUP_SCHEMA = "full_cpt_postmask_deduplication_v2"
SHARD_SCHEMA = "full_cpt_megatron_shard_v1"


def absolute_receipt(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    selected = path
    path = selected.resolve()
    if selected.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def validate_file_receipt(value: Mapping[str, Any]) -> Path:
    path = Path(str(value.get("path", "")))
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(value.get("bytes", -1)):
        raise ValueError(f"file-size drift: {path}")
    if sha256_file(path) != value.get("sha256"):
        raise ValueError(f"file checksum drift: {path}")
    return path.resolve()


def validate_overlay(path: Path, executing_script: Path) -> dict[str, Any]:
    overlay = read_json(path)
    if (
        overlay.get("schema_version") != OVERLAY_SCHEMA
        or overlay.get("status") != "frozen"
    ):
        raise ValueError("anonymization overlay is not frozen")
    bound_code_sha(overlay, executing_script)
    parent = validate_file_receipt(overlay["parent_input_receipt"])
    parent_value = read_json(parent)
    if parent_value.get("schema_version") != "full_cpt_training_bridge_input_receipt_v1":
        raise ValueError("unsupported parent bridge receipt")
    if canonical_sha256(parent_value.get("tasks", [])) != overlay.get(
        "parent_tasks_sha256"
    ):
        raise ValueError("parent task inventory drift")
    if canonical_sha256(overlay.get("tasks", [])) != overlay.get("tasks_sha256"):
        raise ValueError("derived anonymization task inventory drift")
    validation = overlay.get("validation_manifest")
    if not isinstance(validation, Mapping):
        raise ValueError("overlay does not bind the frozen validation manifest")
    validate_file_receipt(validation)
    return overlay


def load_parent(overlay: Mapping[str, Any]) -> dict[str, Any]:
    return read_json(validate_file_receipt(overlay["parent_input_receipt"]))


def load_overlay_heldouts(
    overlay: Mapping[str, Any], overlay_path: Path, heldout_path: Path
) -> dict[str, Any]:
    heldout_path = heldout_path.resolve()
    heldout = read_json(heldout_path)
    if heldout.get("input_receipt_sha256") != sha256_file(overlay_path):
        raise ValueError("heldout manifest is bound to a different overlay")
    parent = overlay["parent_heldout_manifest"]
    derivation = heldout.get("derivation", {}).get("parent")
    if derivation != parent:
        raise ValueError("heldout derivation parent drift")
    return heldout


def load_task_exclusions(
    task: Mapping[str, Any], heldouts: Mapping[str, Any]
) -> tuple[set[str], dict[str, Any]]:
    if not task.get("requires_heldout_exclusion"):
        if task.get("exclusion_key"):
            raise ValueError("optional heldout exclusions are forbidden")
        return set(), {"required": False}
    key = str(task.get("exclusion_key", ""))
    receipt = heldouts.get("exclusions", {}).get(key)
    if not isinstance(receipt, Mapping):
        raise ValueError(f"missing required heldout exclusion: {key}")
    path = validate_file_receipt(receipt)
    values = load_exclusion_ids(path)
    if len(values) != int(receipt.get("rows", -1)):
        raise ValueError(f"heldout exclusion accounting drift: {key}")
    return values, {
        "required": True,
        "key": key,
        **absolute_receipt(path, rows=len(values)),
    }


def import_parent_builder() -> Any:
    path = BRIDGE_DIR / "build_binary_shard.py"
    spec = importlib.util.spec_from_file_location("full8_parent_binary_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_task_input(task: Mapping[str, Any]) -> Path:
    path = Path(str(task["input_path"]))
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(task["input_bytes"]):
        raise ValueError(f"task input size drift: {path}")
    if sha256_file(path) != task["input_sha256"]:
        raise ValueError(f"task input checksum drift: {path}")
    return path.resolve()


def task_manifest_path(stage_root: Path, task_index: int, kind: str) -> Path:
    return stage_root / kind / f"task_{task_index:05d}.manifest.json"


def task_payload_path(stage_root: Path, task_index: int, suffix: str) -> Path:
    return stage_root / "inventory" / f"task_{task_index:05d}.{suffix}"


__all__ = [
    "DEDUP_SCHEMA",
    "INVENTORY_SCHEMA",
    "OVERLAY_SCHEMA",
    "SHARD_SCHEMA",
    "absolute_receipt",
    "bound_code_sha",
    "canonical_sha256",
    "import_parent_builder",
    "load_overlay_heldouts",
    "load_parent",
    "load_task_exclusions",
    "read_json",
    "sha256_file",
    "task_manifest_path",
    "task_payload_path",
    "utc_now",
    "validate_file_receipt",
    "validate_overlay",
    "validate_task_input",
    "write_json_atomic",
]
