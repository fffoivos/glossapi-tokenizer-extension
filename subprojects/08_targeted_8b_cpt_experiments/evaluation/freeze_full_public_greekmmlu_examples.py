#!/usr/bin/env python3
"""Freeze the complete pinned 16,632-question GreekMMLU panel."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from contract_utils import executing_code_bundle, require, write_json_atomic  # noqa: E402

REPO_ID = "dascim/GreekMMLU"
REVISION = "6a03aa06b68beb932fb75edff3a34e50b3674649"
CONFIG = "All"
SPLIT = "test"
COUNT = 16_632


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable output exists: {args.output}")
    from datasets import load_dataset

    dataset = load_dataset(REPO_ID, CONFIG, revision=REVISION, split=SPLIT)
    require(len(dataset) == COUNT, "public GreekMMLU count drift")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(dataset):
        example_id = str(item.get("id") or f"greekmmlu:{index}")
        subject = str(
            item.get("subject") or item.get("sub_category") or item.get("category") or ""
        ).strip()
        level = str(item.get("level") or "").strip()
        require(example_id and subject, f"public GreekMMLU metadata missing at row {index}")
        rows.append(
            {
                "example_id": example_id,
                "row_index": index,
                "subject": subject,
                "educational_level": level or None,
            }
        )
    require(len({row["example_id"] for row in rows}) == COUNT, "duplicate public ids")
    write_json_atomic(
        args.output,
        {
            "schema_version": "apertus_greekmmlu_public_examples_v1",
            "status": "frozen",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "executing_code_bundle": executing_code_bundle(),
            "dataset": {
                "repo_id": REPO_ID,
                "revision": REVISION,
                "config": CONFIG,
                "split": SPLIT,
            },
            "public_count": COUNT,
            "subjects": sorted({row["subject"] for row in rows}),
            "educational_level_populated_count": sum(
                row["educational_level"] is not None for row in rows
            ),
            "examples": rows,
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
