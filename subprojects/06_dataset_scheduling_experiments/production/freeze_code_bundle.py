#!/usr/bin/env python3
"""Freeze an immutable, byte-exact scientific or efficiency code bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from campaign_contract import atomic_write_json, sha256_file


IGNORED_PARTS = {".git", ".pytest_cache", ".venv", "__pycache__"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".DS_Store"}


def inventory(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"code bundle must not contain symlinks: {path}")
        if not path.is_file() or path.name.endswith(tuple(IGNORED_SUFFIXES)):
            continue
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ValueError(f"empty code bundle: {root}")
    return rows


def canonical_tree_sha256(rows: list[dict]) -> str:
    encoded = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("scientific", "efficiency"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    rows = inventory(root)
    payload = {
        "schema_version": "apertus_mini_immutable_code_bundle_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": args.kind,
        "root": str(root),
        "file_count": len(rows),
        "tree_sha256": canonical_tree_sha256(rows),
        "files": rows,
        "exclusions": {
            "directory_parts": sorted(IGNORED_PARTS),
            "file_suffixes": sorted(IGNORED_SUFFIXES),
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "kind": args.kind, "files": len(rows), "tree_sha256": payload["tree_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
