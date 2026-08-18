#!/usr/bin/env python3
"""Verify an immutable code bundle against its receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the content digest without relying on an ambient helper module."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_passing_receipt(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if value.get("schema_version") != "apertus_mini_immutable_code_bundle_v1":
        raise ValueError(f"{path}: unexpected receipt schema")
    if str(value.get("status", "")).lower() not in {
        "pass",
        "passed",
        "completed",
        "frozen",
    }:
        raise ValueError(f"{path}: non-passing receipt status")
    return value


def verify_code_bundle_receipt(path: Path, root: Path, kind: str) -> dict[str, object]:
    """Fail closed on root, inventory, per-file hash, or tree-hash drift.

    This intentionally remains self-contained: workers execute this file with
    ``PYTHONSAFEPATH=1`` and must not import a campaign-specific helper that is
    absent from the committed scientific source tree.
    """

    receipt = read_passing_receipt(path)
    resolved_root = root.resolve()
    rows = receipt.get("files", [])
    if not isinstance(rows, list) or (
        receipt.get("kind") != kind
        or Path(str(receipt.get("root", ""))).resolve() != resolved_root
        or int(receipt.get("file_count", -1)) != len(rows)
        or not rows
    ):
        raise ValueError(f"{kind} bundle receipt/root drift")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{kind} bundle has a malformed file record")
        relative = str(row.get("relative_path", ""))
        unresolved = resolved_root / relative
        candidate = unresolved.resolve()
        if (
            not relative
            or relative in seen
            or unresolved.is_symlink()
            or (candidate.parent != resolved_root and resolved_root not in candidate.parents)
            or not candidate.is_file()
            or candidate.stat().st_size != int(row.get("bytes", -1))
            or sha256_file(candidate) != row.get("sha256")
        ):
            raise ValueError(f"{kind} bundle file drift: {relative}")
        seen.add(relative)
    exclusions = receipt.get("exclusions", {})
    if not isinstance(exclusions, dict):
        raise ValueError(f"{kind} bundle exclusions are malformed")
    ignored_parts = set(exclusions.get("directory_parts", []))
    ignored_suffixes = tuple(exclusions.get("file_suffixes", []))
    observed: set[str] = set()
    for candidate in resolved_root.rglob("*"):
        relative = candidate.relative_to(resolved_root)
        if any(part in ignored_parts for part in relative.parts):
            continue
        if candidate.is_symlink():
            raise ValueError(f"{kind} bundle contains a symlink: {relative.as_posix()}")
        if candidate.is_file() and not candidate.name.endswith(ignored_suffixes):
            observed.add(relative.as_posix())
    if observed != seen:
        raise ValueError(
            f"{kind} bundle inventory drift: missing={sorted(seen - observed)}, "
            f"extra={sorted(observed - seen)}"
        )
    canonical = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != receipt.get("tree_sha256"):
        raise ValueError(f"{kind} bundle tree hash drift")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--kind", choices=("scientific", "efficiency"), required=True)
    args = parser.parse_args()
    value = verify_code_bundle_receipt(args.receipt, args.root, args.kind)
    print(
        json.dumps(
            {"ok": True, "kind": args.kind, "tree_sha256": value["tree_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
