#!/usr/bin/env python3
"""Fail closed if a canonical copied analysis artifact has drifted."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "evidence" / "ARTIFACT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checked = 0
    for artifact in payload["artifacts"]:
        path = ROOT / artifact["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != artifact["bytes"]:
            raise ValueError(f"size drift: {path}: {size} != {artifact['bytes']}")
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise ValueError(
                f"sha256 drift: {path}: {actual} != {artifact['sha256']}"
            )
        checked += 1
    print(json.dumps({"ok": True, "checked": checked}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
