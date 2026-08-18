#!/usr/bin/env python3
"""Freeze an immutable, byte-exact scientific or efficiency code bundle.

The freezer is intentionally self-contained.  It is invoked on a login node
from a clean, committed checkout before its output becomes read-only, so it
must not depend on an untracked helper that is absent from that checkout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


IGNORED_PARTS = {".git", ".pytest_cache", ".venv", "__pycache__"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".DS_Store"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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


def write_json_new(path: Path, value: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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
    payload: dict[str, Any] = {
        "schema_version": "apertus_mini_immutable_code_bundle_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": args.kind,
        "root": str(root),
        "file_count": len(rows),
        "tree_sha256": hashlib.sha256(
            json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "files": rows,
        "exclusions": {
            "directory_parts": sorted(IGNORED_PARTS),
            "file_suffixes": sorted(IGNORED_SUFFIXES),
        },
    }
    write_json_new(args.output, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "kind": args.kind,
                "files": len(rows),
                "tree_sha256": payload["tree_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
