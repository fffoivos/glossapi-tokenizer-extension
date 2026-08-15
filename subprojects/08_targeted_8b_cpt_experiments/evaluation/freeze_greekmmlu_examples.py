#!/usr/bin/env python3
"""Freeze GreekMMLU identifiers and strata at the pinned dataset revision."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from contract_utils import (  # noqa: E402
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    require_file_binding,
    write_json_atomic,
)


REPO_ID = "dascim/GreekMMLU"
REVISION = "6a03aa06b68beb932fb75edff3a34e50b3674649"
CONFIG = "All"
SPLIT = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def clean_ids(path: Path) -> set[str]:
    value = read_json(path)
    require(value.get("status") in {"completed", "frozen", "passed"}, "clean manifest did not pass")
    require(
        value.get("schema_version") == "apertus_mini_greekmmlu_clean_subset_v1"
        and value.get("dataset_repo_id") == REPO_ID
        and value.get("dataset_revision") == REVISION
        and value.get("dataset_config") == CONFIG
        and value.get("dataset_split") == SPLIT
        and value.get("full_count") == 16_632
        and value.get("clean_count") == 16_159,
        "clean GreekMMLU manifest identity drift",
    )
    ids_path = require_file_binding(value.get("clean_example_ids"))
    ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(ids) == 16_159, "clean GreekMMLU id count drift")
    result = set(ids)
    require(len(result) == len(ids), "duplicate clean GreekMMLU ids")
    require(all(item.startswith("greekmmlu:") for item in result), "clean GreekMMLU id namespace drift")
    return result


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable output exists: {args.output}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required in the CSCS evaluation environment") from exc

    wanted = clean_ids(args.clean_manifest)
    dataset = load_dataset(REPO_ID, CONFIG, revision=REVISION, split=SPLIT)
    require(len(dataset) == 16_632, "public GreekMMLU count drift")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(dataset):
        example_id = str(item.get("id") or f"greekmmlu:{index}")
        if example_id not in wanted:
            continue
        subject = str(item.get("subject") or item.get("sub_category") or item.get("category") or "").strip()
        level = str(item.get("level") or "").strip()
        require(subject, f"missing subject: {example_id}")
        rows.append(
            {
                "example_id": example_id,
                "row_index": index,
                "subject": subject,
                "educational_level": level or None,
            }
        )
    require(len(rows) == 16_159, f"clean dataset reconciliation drift: {len(rows)}")
    require({row["example_id"] for row in rows} == wanted, "clean id reconciliation drift")
    payload = {
        "schema_version": "apertus_greekmmlu_clean_examples_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "dataset": {"repo_id": REPO_ID, "revision": REVISION, "config": CONFIG, "split": SPLIT},
        "clean_manifest": file_binding(args.clean_manifest),
        "full_count": 16_632,
        "clean_count": len(rows),
        "subjects": sorted({row["subject"] for row in rows}),
        "educational_level_populated_count": sum(row["educational_level"] is not None for row in rows),
        "examples": rows,
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
