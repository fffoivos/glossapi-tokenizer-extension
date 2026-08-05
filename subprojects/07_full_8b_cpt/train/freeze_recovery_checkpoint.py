#!/usr/bin/env python3
"""Freeze the latest clean checkpoint inside a failed full-8B segment."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from contract import atomic_write_json, file_binding, sha256_file


ROW = re.compile(r"iteration\s+(\d+)\s*/[^\n]+", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--segment-start", type=int, required=True)
    parser.add_argument("--segment-end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    marker = args.checkpoint_root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise ValueError("failed attempt produced no checkpoint marker")
    iteration = int(marker.read_text().strip())
    if not args.segment_start < iteration < args.segment_end:
        raise ValueError("latest checkpoint is not an interior recovery point")
    text = args.log.read_text(encoding="utf-8", errors="replace")
    rows = [(int(match.group(1)), match.group(0)) for match in ROW.finditer(text)]
    prefix = "\n".join(row for index, row in rows if index <= iteration)
    if re.search(r"(?:lm loss|grad norm)\s*:\s*(?:nan|inf)", prefix, re.IGNORECASE):
        raise ValueError("nonfinite metric before recovery checkpoint")
    if re.search(r"number of skipped iterations\s*:\s*[1-9]", prefix, re.IGNORECASE):
        raise ValueError("skipped optimizer update before recovery checkpoint")
    directory = args.checkpoint_root / f"iter_{iteration:07d}"
    if not (directory / ".metadata").is_file():
        raise ValueError("recovery checkpoint metadata absent")
    files = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        files.append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    payload = {
        "schema_version": "apertus_full_8b_recovery_checkpoint_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iteration": iteration,
        "tokens": iteration * 4_194_304,
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "checkpoint_directory": str(directory.resolve()),
        "files": len(files),
        "bytes": sum(row["bytes"] for row in files),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "training_log": file_binding(args.log),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "iteration": iteration, "files": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
