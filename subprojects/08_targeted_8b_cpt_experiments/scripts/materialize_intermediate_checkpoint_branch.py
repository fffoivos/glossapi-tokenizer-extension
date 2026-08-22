#!/usr/bin/env python3
"""Create a no-copy load view for one saved intermediate DCP checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
from pathlib import Path

from audit_training_checkpoint import parse_training_log
from contract_utils import file_binding, require, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-load-root", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--source-training-log", type=Path, required=True)
    parser.add_argument("--output-load-root", type=Path, required=True)
    parser.add_argument("--output-training-log", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    return parser.parse_args()


def hardlink_tree(source: Path, target: Path) -> tuple[int, int]:
    require(source.is_dir() and not target.exists(), "branch checkpoint path drift")
    target.mkdir(parents=True)
    count = 0
    total_bytes = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        output = target / relative
        if path.is_dir():
            output.mkdir()
            continue
        require(path.is_file() and not path.is_symlink(), f"unexpected checkpoint entry: {path}")
        os.link(path, output)
        source_stat = path.stat()
        output_stat = output.stat()
        require(
            (source_stat.st_dev, source_stat.st_ino, source_stat.st_size)
            == (output_stat.st_dev, output_stat.st_ino, output_stat.st_size),
            f"checkpoint hard-link drift: {relative}",
        )
        count += 1
        total_bytes += source_stat.st_size
    require(count > 0, "checkpoint file inventory is empty")
    return count, total_bytes


def freeze_log_prefix(source: Path, target: Path, update: int) -> dict[str, object]:
    require(source.is_file() and not target.exists(), "branch training-log path drift")
    marker = re.compile(rf"successfully saved checkpoint from iteration\s+{update}\b")
    selected: list[str] = []
    found = False
    with source.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            selected.append(line)
            if marker.search(line):
                found = True
                break
    require(found, "source log lacks the requested save confirmation")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(selected), encoding="utf-8")
    return parse_training_log(target, update)


def main() -> int:
    args = parse_args()
    require(args.update > 0, "branch update must be positive")
    require(not args.output_receipt.exists(), "immutable branch receipt exists")
    source_load = args.source_load_root.resolve()
    source_checkpoint = source_load / f"iter_{args.update:07d}"
    output_load = args.output_load_root
    require(not output_load.exists(), "branch load root exists")
    output_checkpoint = output_load / f"iter_{args.update:07d}"
    count, total_bytes = hardlink_tree(source_checkpoint, output_checkpoint)
    tracker = output_load / "latest_checkpointed_iteration.txt"
    tracker.write_text(f"{args.update}\n", encoding="utf-8")
    log_summary = freeze_log_prefix(
        args.source_training_log.resolve(), args.output_training_log, args.update
    )
    payload = {
        "schema_version": "apertus_hard_h_to_g_intermediate_checkpoint_branch_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "update": args.update,
        "copy_mode": "same_filesystem_hard_links",
        "source_load_root": str(source_load),
        "source_checkpoint_metadata": file_binding(source_checkpoint / ".metadata"),
        "source_checkpoint_common": file_binding(source_checkpoint / "common.pt"),
        "output_load_root": str(output_load.resolve()),
        "output_checkpoint_metadata": file_binding(output_checkpoint / ".metadata"),
        "output_checkpoint_common": file_binding(output_checkpoint / "common.pt"),
        "load_tracker": file_binding(tracker),
        "training_log": file_binding(args.output_training_log),
        "training_log_summary": log_summary,
        "checkpoint_file_count": count,
        "checkpoint_total_bytes": total_bytes,
        "source_and_output_share_inodes": True,
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
