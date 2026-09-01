#!/usr/bin/env python3
"""Validate the narrow immutable-producer compatibility authority."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contract_utils import file_binding, read_json, require


SCHEMA = "apertus_hard_h_to_g_producer_bundle_compatibility_v1"


def validate_authority(value: dict[str, Any], current: dict[str, Any]) -> set[tuple[str, str, str, int, str]]:
    require(value.get("schema_version") == SCHEMA, "producer compatibility schema drift")
    require(value.get("status") == "passed", "producer compatibility authority did not pass")
    require(value.get("executing_code_bundle") == current, "producer compatibility current-bundle drift")
    rows = value.get("producers")
    require(isinstance(rows, list) and rows, "producer compatibility rows missing")
    accepted: set[tuple[str, str, str, int, str]] = set()
    for row in rows:
        require(isinstance(row, dict), "producer compatibility row malformed")
        bundle = row.get("bundle")
        require(isinstance(bundle, dict), "producer compatibility bundle missing")
        receipt = bundle.get("receipt")
        require(isinstance(receipt, dict), "producer compatibility receipt binding missing")
        receipt_path = Path(str(receipt.get("path", "")))
        require(receipt_path.is_file() and receipt == file_binding(receipt_path), "producer compatibility receipt binding drift")
        receipt_value = read_json(receipt_path)
        require(
            receipt_value.get("schema_version") == "apertus_mini_immutable_code_bundle_v1"
            and receipt_value.get("status") == "frozen"
            and receipt_value.get("kind") == "scientific",
            "producer compatibility receipt is not frozen scientific code",
        )
        root = Path(str(bundle.get("root", ""))).resolve()
        tree = str(bundle.get("tree_sha256", ""))
        require(
            root.is_dir()
            and Path(str(receipt_value.get("root", ""))).resolve() == root
            and receipt_value.get("tree_sha256") == tree,
            "producer compatibility bundle identity drift",
        )
        key = (str(root), tree, str(receipt_path.resolve()), int(receipt["bytes"]), str(receipt["sha256"]))
        require(key not in accepted, "duplicate producer compatibility bundle")
        accepted.add(key)
    current_receipt = current.get("receipt")
    require(isinstance(current_receipt, dict), "current bundle receipt binding missing")
    current_key = (
        str(Path(str(current["root"])).resolve()),
        str(current["tree_sha256"]),
        str(Path(str(current_receipt["path"])).resolve()),
        int(current_receipt["bytes"]),
        str(current_receipt["sha256"]),
    )
    require(current_key in accepted, "current bundle is absent from producer compatibility authority")
    return accepted


def load_authority(path: Path, current: dict[str, Any]) -> tuple[dict[str, Any], set[tuple[str, str, str, int, str]]]:
    value = read_json(path)
    return value, validate_authority(value, current)


def require_accepted_producer(
    value: dict[str, Any],
    accepted: set[tuple[str, str, str, int, str]],
    label: str,
) -> None:
    bundle = value.get("executing_code_bundle")
    require(isinstance(bundle, dict), f"{label}: producer code bundle missing")
    receipt = bundle.get("receipt")
    require(isinstance(receipt, dict), f"{label}: producer code receipt binding missing")
    key = (
        str(Path(str(bundle.get("root", ""))).resolve()),
        str(bundle.get("tree_sha256", "")),
        str(Path(str(receipt.get("path", ""))).resolve()),
        int(receipt.get("bytes", -1)),
        str(receipt.get("sha256", "")),
    )
    require(key in accepted, f"{label}: producer bundle is not compatibility-authorized")

