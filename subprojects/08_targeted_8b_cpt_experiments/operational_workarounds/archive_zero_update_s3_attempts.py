#!/usr/bin/env python3
"""Archive three checkpoint-free S3 launch failures and reset retry names."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


def binding(path: Path, root: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "relative_path": str(path.relative_to(root)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive-name", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    source = args.run_root / "segments/s3/attempts"
    archive = args.run_root / "segments/s3" / args.archive_name
    receipt = archive.with_suffix(".json")
    if archive.exists() or receipt.exists():
        raise FileExistsError("archive or receipt already exists")
    attempts = sorted(path.name for path in source.iterdir() if path.is_dir())
    if attempts != ["attempt_000001", "attempt_000002", "attempt_000003"]:
        raise ValueError(f"unexpected retry namespace: {attempts}")
    if list(source.rglob("iter_*")) or list(source.rglob("latest_checkpointed_iteration.txt")):
        raise ValueError("refusing to archive an attempt containing a checkpoint")
    for index in (1, 2):
        holder = json.loads((source / f"attempt_{index:06d}/holder.json").read_text())
        if holder.get("status") != "unverifiable_or_insufficient_remaining_time":
            raise ValueError(f"attempt {index} is not the expected pre-training time refusal")
    execution = json.loads((source / "attempt_000003/execution.json").read_text())
    if execution.get("status") != "failed" or execution.get("checkpoint") is not None:
        raise ValueError("attempt 3 is not a checkpoint-free execution failure")
    log = (source / "attempt_000003/train.log").read_text(errors="replace")
    forbidden = ("iteration     2262", "successfully saved checkpoint")
    if any(marker in log for marker in forbidden):
        raise ValueError("attempt 3 reached an optimizer update or checkpoint")
    rows = [binding(path, source) for path in sorted(source.rglob("*")) if path.is_file()]
    os.rename(source, archive)
    source.mkdir(mode=0o750)
    payload = {
        "schema_version": "apertus_zero_update_attempt_archive_v1",
        "status": "passed",
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": args.reason,
        "science_changed": False,
        "checkpoint_or_optimizer_step_observed": False,
        "archived_attempts": attempts,
        "archive_root": str(archive),
        "new_retry_root": str(source),
        "files": rows,
    }
    temporary = receipt.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt)
    print(json.dumps({"status": "passed", "files": len(rows), "archive": str(archive)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
