#!/usr/bin/env python3
"""Freeze the parent-packed pool view consumed by the hybrid B schedule."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import copy_file_atomic, file_binding, read_json, require, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pool-receipt", type=Path, required=True)
    parser.add_argument("--hybrid-schedule-manifest", type=Path, required=True)
    parser.add_argument("--packed-output", type=Path, required=True)
    parser.add_argument("--parent-packed-integrity", type=Path, required=True)
    parser.add_argument("--packed-integrity-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable B pool view exists: {args.output}")
    require(not args.packed_output.exists(), f"immutable B packed view exists: {args.packed_output}")
    require(not args.packed_integrity_output.exists(), f"immutable B packed-integrity view exists: {args.packed_integrity_output}")
    parent = read_json(args.parent_pool_receipt)
    schedule = read_json(args.hybrid_schedule_manifest)
    require(parent.get("schema_version") == "apertus_schedule_pool_corpus_v1" and parent.get("status") == "completed", "parent pool receipt drift")
    require(isinstance(parent.get("source_root"), str) and parent["source_root"], "parent source root missing")
    require(schedule.get("schema_version") == "apertus_data_order_schedules_v1" and schedule.get("status") == "completed", "B schedule receipt drift")
    arms = {row["arm_id"]: row for row in schedule["arms"]}
    require(set(arms) == {"D0_mixed"}, "B hybrid schedule must expose only D0_mixed")
    arm = arms["D0_mixed"]
    packed_binding = schedule.get("packed_corpus_receipt", {})
    packed_path = Path(packed_binding.get("path", ""))
    require(file_binding(packed_path)["sha256"] == packed_binding.get("sha256"), "parent packed receipt drift")
    packed_integrity = read_json(args.parent_packed_integrity)
    require(
        packed_integrity.get("schema_version") == "apertus_full_8b_packed_payload_integrity_v1"
        and packed_integrity.get("status") == "passed"
        and packed_integrity.get("packed_receipt", {}).get("sha256") == packed_binding.get("sha256"),
        "parent packed-integrity receipt drift",
    )
    continuation = schedule.get("continuation_contract", {})
    require(continuation.get("parent_prefix_is_byte_exact") is True, "B parent prefix is not exact")
    require(continuation.get("parent_prefix_overlap_selected_sequences") == 0, "B continuation reuses prefix sequences")
    pool_active = {key: int(value) for key, value in arm["pool_active_tokens"].items()}
    require(set(pool_active) == {"H", "G", "F", "O"}, "B pool labels drift")
    active_total = sum(pool_active.values())
    modern = pool_active["H"] + pool_active["G"]
    payload = {
        "schema_version": "apertus_schedule_pool_corpus_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "parent_pool_receipt": file_binding(args.parent_pool_receipt),
        "hybrid_schedule_manifest": file_binding(args.hybrid_schedule_manifest),
        "tokenizer": parent["tokenizer"],
        "source_root": parent.get("source_root"),
        "sorted_training_catalogs": parent["sorted_training_catalogs"],
        "modern_greek": {
            "tokens": modern,
            "hplt_tokens": pool_active["H"],
            "glossapi_non_hplt_tokens": pool_active["G"],
        },
        "integer_79_20_1_geometry": {
            "active_tokens": active_total,
            "modern_greek": modern,
            "foreign_replay": pool_active["F"],
            "old_greek_replay": pool_active["O"],
            "realized_fractions": {
                "modern_greek": str(modern / active_total),
                "foreign_replay": str(pool_active["F"] / active_total),
                "old_greek_replay": str(pool_active["O"] / active_total),
            },
        },
        "invariants": {
            "data_payloads_inherited_from_parent_byte_exactly": True,
            "parent_schedule_prefix_inherited_byte_exactly": True,
            "continuation_uses_only_parent_suffix_sequences": True,
            "global_deduplication_performed": False,
        },
    }
    # The contract is the commit marker. Materialize its two dependencies
    # atomically first, then publish the immutable contract last so a failed
    # copy can never leave a seemingly complete B pool view behind.
    created: list[Path] = []
    try:
        copy_file_atomic(packed_path, args.packed_output)
        created.append(args.packed_output)
        copy_file_atomic(args.parent_packed_integrity, args.packed_integrity_output)
        created.append(args.packed_integrity_output)
        write_json_atomic(args.output, payload)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    print(json.dumps({"ok": True, "active_tokens": active_total}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
