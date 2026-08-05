#!/usr/bin/env python3
"""Freeze receipts for the corrected validation-only full-8B stage overlay."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-stage", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--clean-panel-manifest", type=Path, required=True)
    parser.add_argument("--training-content-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = {
        "lineage": args.stage_root / "stage_lineage.json",
        "pool_corpus": args.stage_root / "inventory/pool_corpus_receipt.json",
        "packed_corpus": args.stage_root / "inventory/packed_corpus_receipt.json",
        "schedule": args.stage_root / "schedules/schedule_manifest.json",
        "validation": args.stage_root / "validation/validation_manifest.json",
        "token_byte_lengths": args.stage_root / "validation/token_utf8_byte_lengths.receipt.json",
        "clean_replay_panel": args.clean_panel_manifest,
        "selected_training_content": args.training_content_receipt,
    }
    values = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    if values["pool_corpus"].get("status") != "completed" or values["packed_corpus"].get("status") != "completed":
        raise ValueError("inherited training receipts are not complete")
    if values["schedule"].get("status") != "completed" or values["validation"].get("status") != "frozen":
        raise ValueError("corrected stage schedule/validation is not complete")
    if values["clean_replay_panel"].get("overlap_audit", {}).get("replacement_panel", {}).get("overlapping_documents") != 0:
        raise ValueError("replacement replay panel overlaps training")
    old = next(row for row in values["validation"]["panels"] if row["name"] == "old_greek")
    if old.get("training_exact_content_overlap_documents") != 0 or old.get("display_name") != "Greek replay retention":
        raise ValueError("validation manifest did not adopt the clean replay panel")
    payload = {
        "schema_version": "apertus_full8b_data_stage_v2",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage_root": str(args.stage_root.resolve()),
        "parent_stage": str(args.parent_stage.resolve()),
        "training_data_identity_changed": False,
        "validation_change": "legacy old_greek ID now points to exact-training-content-disjoint Greek replay",
        "receipts": {name: binding(path) for name, path in paths.items()},
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "stage": str(args.stage_root.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
