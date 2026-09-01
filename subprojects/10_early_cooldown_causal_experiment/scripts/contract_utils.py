#!/usr/bin/env python3
"""Small fail-closed helpers for the early-cooldown experiment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


ITERATION = re.compile(r"iteration\s+(\d+)\s*/")
METRIC = re.compile(r"(?:^|\|)\s*([^|:]+?)\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_iteration(path: Path, iteration: int) -> dict[str, float]:
    matches = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        found = ITERATION.search(line)
        if found and int(found.group(1)) == iteration:
            matches.append({" ".join(key.lower().split()): float(value) for key, value in METRIC.findall(line)})
    require(len(matches) == 1, f"expected exactly one logged iteration {iteration}, got {len(matches)}")
    return matches[0]


def verify_bound_file(binding: dict[str, Any], *, require_bytes: bool = False) -> Path:
    path = Path(binding["path"]).resolve()
    require(path.is_file(), f"missing bound file: {path}")
    if require_bytes:
        require(path.stat().st_size == int(binding["bytes"]), f"byte-size drift: {path}")
    require(sha256_file(path) == binding["sha256"], f"sha256 drift: {path}")
    return path


def file_binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file(), f"missing receipt input: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, value: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
