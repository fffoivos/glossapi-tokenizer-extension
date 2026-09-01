#!/usr/bin/env python3
"""Create exact active-token packing tasks from the sorted pool catalogs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


BUCKETS = 128
CATALOG_DTYPE = np.dtype(
    [
        ("pool", "u1"),
        ("task_index", "<u4"),
        ("document_index", "<u4"),
        ("tokens", "<u4"),
        ("identity", "V16"),
        ("order", "V16"),
    ],
    align=False,
)
POOL_ORDER = (
    "hplt_new_greek",
    "non_hplt_new_greek",
    "foreign_replay",
    "old_greek_replay",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = Path(str(path) + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def nearest_ratio_total(modern: int) -> int:
    quotient, remainder = divmod(modern * 100, 79)
    return quotient + int(remainder * 2 >= 79)


def nearest_percent(total: int) -> int:
    quotient, remainder = divmod(total, 100)
    return quotient + int(remainder * 2 >= 100)


def bucket_bounds(rows: np.memmap, bucket: int) -> tuple[int, int]:
    order_first = np.ndarray(
        shape=(rows.size,),
        dtype=np.uint8,
        buffer=rows,
        offset=CATALOG_DTYPE.fields["order"][1],
        strides=(CATALOG_DTYPE.itemsize,),
    )
    start = int(np.searchsorted(order_first, bucket * 2, side="left"))
    end = int(np.searchsorted(order_first, (bucket + 1) * 2, side="left"))
    return start, end


def token_balanced_boundaries(tokens: np.ndarray, parts: int) -> list[int]:
    """Split source-local document rows into nonempty, approximately token-balanced tasks."""
    if tokens.size == 0 or not 1 <= parts <= int(tokens.size):
        raise ValueError("invalid source-local partition shape")
    cumulative = np.cumsum(tokens, dtype=np.uint64)
    total = int(cumulative[-1])
    boundaries = [0]
    for part in range(1, parts):
        target = total * part // parts
        end = int(np.searchsorted(cumulative, target, side="left")) + 1
        end = max(end, boundaries[-1] + 1)
        end = min(end, int(tokens.size) - (parts - part))
        boundaries.append(end)
    boundaries.append(int(tokens.size))
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise RuntimeError("source-local packing partition contains an empty task")
    return boundaries


def materialize_or_verify_selected_catalog(path: Path, selected: np.ndarray) -> str:
    """Freeze a plan-local selected catalog without mutating its source catalog.

    The replay source catalogs can be shared by several experiments.  A selected
    prefix is therefore an output of *this* packing plan, never an artifact next
    to the shared source catalog.  A retry after an interrupted plan build may
    find the exact plan-local payload already present; it is safe to reuse only
    after byte-for-byte verification against the freshly derived selection.
    """
    if path.exists():
        expected_bytes = int(selected.nbytes)
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"existing plan-local selected catalog size drift: {path}")
        existing = np.memmap(path, mode="r", dtype=CATALOG_DTYPE)
        try:
            if existing.shape != selected.shape or not np.array_equal(existing, selected):
                raise ValueError(
                    f"existing plan-local selected catalog content drift: {path}"
                )
        finally:
            del existing
        return "verified_reuse"

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    if temporary.exists():
        raise FileExistsError(f"unreceipted selected catalog partial exists: {temporary}")
    selected.tofile(temporary)
    os.replace(temporary, path)
    return "created"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite packing plan: {args.output}")
    pool_receipt = read_json(args.pool_receipt)
    pool_schema = pool_receipt.get("schema_version")
    if (
        pool_schema
        not in {
            "apertus_mini_schedule_pool_corpus_v1",
            "apertus_schedule_pool_corpus_v1",
        }
        or pool_receipt.get("status") != "completed"
    ):
        raise ValueError("pool corpus receipt is not completed")
    generic_schema = pool_schema == "apertus_schedule_pool_corpus_v1"
    pad_token_id = int(pool_receipt.get("tokenizer", {}).get("pad_token_id", 10))
    modern = int(pool_receipt["modern_greek"]["tokens"])
    total = nearest_ratio_total(modern)
    old_target = nearest_percent(total)
    foreign_target = total - modern - old_target
    targets = {
        "hplt_new_greek": int(pool_receipt["modern_greek"]["hplt_tokens"]),
        "non_hplt_new_greek": int(
            pool_receipt["modern_greek"]["glossapi_non_hplt_tokens"]
        ),
        "foreign_replay": foreign_target,
        "old_greek_replay": old_target,
    }

    tasks: list[dict[str, Any]] = []
    pool_summaries: dict[str, dict[str, Any]] = {}
    for pool_code, pool in enumerate(POOL_ORDER):
        catalog_receipt = pool_receipt["sorted_training_catalogs"][pool]
        catalog_path = Path(catalog_receipt["path"])
        if (
            not catalog_path.is_file()
            or catalog_path.stat().st_size != int(catalog_receipt["bytes"])
            or sha256_file(catalog_path) != catalog_receipt["sha256"]
            or int(catalog_receipt["record_bytes"]) != CATALOG_DTYPE.itemsize
        ):
            raise ValueError(f"sorted catalog drift: {catalog_path}")
        rows = np.memmap(catalog_path, mode="r", dtype=CATALOG_DTYPE)
        target = int(targets[pool])
        pool_task_start = len(tasks)
        if target == int(catalog_receipt["tokens"]):
            selected_end = int(rows.size)
            selected_source_tokens = target
        else:
            seeded_cumulative = np.cumsum(rows["tokens"], dtype=np.uint64)
            selected_end = int(np.searchsorted(seeded_cumulative, target, side="left")) + 1
            selected_source_tokens = int(seeded_cumulative[selected_end - 1])
            del seeded_cumulative
        selected = np.array(rows[:selected_end], copy=True)
        discarded_tail = selected_source_tokens - target
        partial_identity = bytes(selected[-1]["identity"]) if discarded_tail else None
        partial_last = (
            selected["identity"] == np.void(partial_identity)
            if partial_identity is not None
            else np.zeros(selected.size, dtype=bool)
        )
        source_order = np.lexsort(
            (selected["document_index"], selected["task_index"], partial_last)
        )
        selected = selected[source_order]
        if partial_identity is not None and bytes(selected[-1]["identity"]) != partial_identity:
            raise RuntimeError("partial replay document was not placed at the end of IO order")
        io_catalog_path = (
            args.output.parent / "packing_catalogs" / f"{pool}.source_local_selected.catalog45"
        )
        io_catalog_materialization = materialize_or_verify_selected_catalog(
            io_catalog_path, selected
        )
        io_catalog_receipt = {
            "path": str(io_catalog_path),
            "sha256": sha256_file(io_catalog_path),
            "bytes": io_catalog_path.stat().st_size,
            "rows": int(selected.size),
            "tokens": selected_source_tokens,
            "record_bytes": CATALOG_DTYPE.itemsize,
            "selection_source_catalog": dict(catalog_receipt),
            "selection": "seeded_order_prefix_exact_document_set_then_source_task_and_document_index_IO_order",
            "partial_final_selected_document_forced_to_IO_tail": partial_identity is not None,
            "materialization": io_catalog_materialization,
            "scope": "packing_plan_local_not_source_catalog",
        }
        parts = min(BUCKETS, int(selected.size))
        boundaries = token_balanced_boundaries(selected["tokens"], parts)
        remaining_active = target
        for bucket, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            source_tokens = int(selected["tokens"][start:end].astype(np.uint64).sum())
            active = source_tokens if bucket < parts - 1 else source_tokens - discarded_tail
            if active <= 0 or active > remaining_active:
                raise RuntimeError("invalid source-local task active-token quota")
            tasks.append(
                {
                    "task_index": len(tasks),
                    "pool": pool,
                    "pool_code": pool_code,
                    "bucket": bucket,
                    "catalog": dict(io_catalog_receipt),
                    "catalog_row_start": start,
                    "catalog_row_end": end,
                    "selected_document_rows": end - start,
                    "selected_source_tokens": source_tokens,
                    "target_active_tokens": active,
                    "discarded_tail_tokens_in_last_selected_document": source_tokens - active,
                    "output_prefix": f"packed_source_local/{pool}/bucket_{bucket:03d}_text_document",
                }
            )
            remaining_active -= active
        if remaining_active != 0:
            raise RuntimeError(f"source-local task quotas did not exhaust target: {pool}")
        pool_summaries[pool] = {
            "pool_code": pool_code,
            "target_active_tokens": targets[pool],
            "available_tokens": int(catalog_receipt["tokens"]),
            "selected_document_rows": int(selected.size),
            "selected_source_tokens": selected_source_tokens,
            "discarded_selected_tail_tokens": selected_source_tokens - targets[pool],
            "packing_tasks": len(tasks) - pool_task_start,
            "selection": "same_seeded_prefix_document_set_packed_in_source_local_IO_order",
            "packed_sequence_randomization": "deferred_to_packed_sequence_catalog_freeze",
        }
        del rows, selected, source_order, partial_last

    if sum(summary["target_active_tokens"] for summary in pool_summaries.values()) != total:
        raise RuntimeError("integer 79/20/1 target accounting drift")
    payload = {
        "schema_version": (
            "apertus_fixed_sequence_packing_plan_v1"
            if generic_schema
            else "apertus_mini_fixed_sequence_packing_plan_v1"
        ),
        "status": "frozen",
        "pool_corpus_receipt": {
            "path": str(args.pool_receipt.resolve()),
            "sha256": sha256_file(args.pool_receipt),
        },
        "geometry": {
            "sequence_length": 4096,
            "stored_tokens_per_sequence": 4097,
            "bucket_count_per_pool": BUCKETS,
            "pad_token_id": pad_token_id,
            "integer_active_tokens": total,
            "integer_mix": {
                "modern_greek": modern,
                "foreign_replay": foreign_target,
                "old_greek_replay": old_target,
            },
            "realized_fractions": {
                "modern_greek": str(modern / total),
                "foreign_replay": str(foreign_target / total),
                "old_greek_replay": str(old_target / total),
            },
            "rounding_policy": "nearest_total_to_modern_times_100_over_79_then_nearest_one_percent_old_greek",
            "IO_order": "source_task_index_then_document_index_with_at_most_one_partial_selected_document_forced_last",
            "scientific_order": "frozen_seeded_permutation_of_immutable_packed_sequence_ids_after_packing",
        },
        "pools": pool_summaries,
        "tasks": tasks,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"ok": True, "tasks": len(tasks), "active_tokens": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
