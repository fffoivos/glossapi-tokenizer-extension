#!/usr/bin/env python3
"""Materialize a sparse scientific bundle from immutable base and patch receipts.

This is deliberately not ``cp -a``.  Every unchanged byte is hard-linked from
the verified base inventory; every declared replacement is hard-linked from a
verified patch inventory.  The result is a new tree for a fresh immutable
bundle receipt, while historical frozen runtime assets need not be invented as
Git files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_receipt(path: Path, *, expected_root: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    if (
        value.get("schema_version") != "apertus_mini_immutable_code_bundle_v1"
        or value.get("status") != "frozen"
        or value.get("kind") != "scientific"
        or Path(str(value.get("root", ""))).resolve() != expected_root.resolve()
    ):
        raise ValueError(f"{path}: invalid scientific bundle receipt")
    rows = value.get("files")
    if not isinstance(rows, list) or int(value.get("file_count", -1)) != len(rows):
        raise ValueError(f"{path}: malformed inventory")
    canonical = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != value.get("tree_sha256"):
        raise ValueError(f"{path}: inventory tree hash drift")
    return value


def inventory(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = receipt["files"]
    assert isinstance(rows, list)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed bundle inventory row")
        relative = str(row.get("relative_path", ""))
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in result
        ):
            raise ValueError(f"invalid bundle path: {relative!r}")
        result[relative] = row
    return result


def verify_file(root: Path, relative: str, row: dict[str, Any]) -> Path:
    candidate = root / relative
    resolved = candidate.resolve()
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or (resolved.parent != root.resolve() and root.resolve() not in resolved.parents)
        or candidate.stat().st_size != int(row.get("bytes", -1))
        or sha256_file(candidate) != row.get("sha256")
    ):
        raise ValueError(f"source bundle file drift: {relative}")
    return candidate


def binding(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--patch-receipt", type=Path, required=True)
    parser.add_argument("--replace-path", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    args = parser.parse_args()

    base_root = args.base_root.resolve()
    patch_root = args.patch_root.resolve()
    output_root = args.output_root.resolve()
    materialization_receipt = args.materialization_receipt.resolve()
    if output_root.exists() or materialization_receipt.exists():
        raise FileExistsError("overlay output or receipt already exists")
    base = inventory(read_receipt(args.base_receipt.resolve(), expected_root=base_root))
    patch = inventory(read_receipt(args.patch_receipt.resolve(), expected_root=patch_root))
    replacements = set(args.replace_path)
    if not replacements or any(path not in patch for path in replacements):
        raise ValueError("every replacement path must exist in the patch receipt")

    output_root.mkdir(parents=True)
    linked_base = linked_patch = 0
    for relative, row in base.items():
        if relative in replacements:
            continue
        source = verify_file(base_root, relative, row)
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
        linked_base += 1
    for relative in sorted(replacements):
        source = verify_file(patch_root, relative, patch[relative])
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
        linked_patch += 1

    payload = {
        "schema_version": "apertus_hard_h_to_g_sparse_bundle_overlay_v1",
        "status": "materialized",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_bundle_receipt": binding(args.base_receipt.resolve()),
        "patch_bundle_receipt": binding(args.patch_receipt.resolve()),
        "output_root": str(output_root),
        "replaced_paths": sorted(replacements),
        "hardlinked_base_files": linked_base,
        "hardlinked_patch_files": linked_patch,
    }
    materialization_receipt.parent.mkdir(parents=True, exist_ok=True)
    materialization_receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
