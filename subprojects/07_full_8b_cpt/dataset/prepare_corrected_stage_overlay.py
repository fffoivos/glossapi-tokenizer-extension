#!/usr/bin/env python3
"""Create a small successor stage that reuses training data and replaces validation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-stage", type=Path, required=True)
    parser.add_argument("--output-stage", type=Path, required=True)
    args = parser.parse_args()
    if args.output_stage.exists():
        raise FileExistsError(args.output_stage)
    required = (
        "inventory/pool_corpus_receipt.json",
        "inventory/packed_corpus_receipt.json",
        "schedules/schedule_manifest.json",
        "validation/token_utf8_byte_lengths.npy",
        "validation/token_utf8_byte_lengths.receipt.json",
        "validation/neutral_external_modern_greek.manifest.json",
    )
    for relative in required:
        if not (args.parent_stage / relative).is_file():
            raise FileNotFoundError(args.parent_stage / relative)
    for relative in required:
        destination = args.output_stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent_stage / relative, destination)
    lineage = {
        "schema_version": "apertus_full_8b_corrected_stage_lineage_v1",
        "status": "prepared",
        "parent_stage": str(args.parent_stage.resolve()),
        "training_payloads_reused_byte_for_byte": True,
        "change_scope": "make every validation panel exact-training-content disjoint",
    }
    (args.output_stage / "stage_lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "output": str(args.output_stage.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
