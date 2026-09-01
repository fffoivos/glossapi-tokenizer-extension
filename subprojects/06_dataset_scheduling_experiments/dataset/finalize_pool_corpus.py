#!/usr/bin/env python3
"""Reduce partition receipts and freeze exact Mini-tokenized pool capacities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np


getcontext().prec = 60
POOL_CODES = {
    0: "hplt_new_greek",
    1: "non_hplt_new_greek",
    2: "foreign_replay",
    3: "old_greek_replay",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int, required=True)
    parser.add_argument("--expected-training-tasks", type=int, required=True)
    parser.add_argument("--expected-heldout-tasks", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation_root = args.stage_root / "validation" / "partition_groups"
    receipts: list[dict[str, Any]] = []
    identity_arrays: list[np.ndarray] = []
    modern_content_arrays: list[np.ndarray] = []
    modern_identity_arrays: list[np.ndarray] = []
    replay_content_arrays: list[np.ndarray] = []
    catalog_arrays: dict[int, list[np.ndarray]] = defaultdict(list)
    all_tasks: set[str] = set()
    pool_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"documents": 0, "tokens": 0, "shards": 0})
    for index in range(args.expected_groups):
        receipt_path = validation_root / f"{index:04d}.json"
        receipt = read_json(receipt_path)
        if (
            receipt.get("schema_version") != "apertus_mini_partition_group_validation_v2"
            or receipt.get("status") != "completed"
            or int(receipt.get("group_index", -1)) != index
            or not receipt.get("all_generated_payload_sha256_verified")
        ):
            raise ValueError(f"invalid partition validation receipt: {receipt_path}")
        digest = receipt["record_identity_digest"]
        digest_path = Path(digest["path"])
        if (
            not digest_path.is_file()
            or digest_path.stat().st_size != int(digest["bytes"])
            or sha256_file(digest_path) != digest["sha256"]
            or int(digest["bytes"]) != int(digest["rows"]) * 16
        ):
            raise ValueError(f"identity digest drift: {digest_path}")
        identity_array = np.fromfile(digest_path, dtype="V16")
        identity_arrays.append(identity_array)
        content = receipt["content_digest"]
        content_path = Path(content["path"])
        if (
            not content_path.is_file()
            or content_path.stat().st_size != int(content["bytes"])
            or sha256_file(content_path) != content["sha256"]
            or int(content["bytes"]) != int(content["rows"]) * 16
        ):
            raise ValueError(f"content digest drift: {content_path}")
        content_array = np.fromfile(content_path, dtype="V16")
        content_policy = receipt.get("content_uniqueness", {}).get("policy")
        if content_policy == "required_globally_for_modern_greek":
            modern_content_arrays.append(content_array)
            modern_identity_arrays.append(identity_array)
        elif content_policy == "audit_only_preserve_original_training_replay_records":
            replay_content_arrays.append(content_array)
        else:
            raise ValueError(f"unknown content-uniqueness policy: {receipt_path}")
        catalog = receipt["training_catalog"]
        catalog_path = Path(catalog["path"])
        if (
            int(catalog.get("record_bytes", -1)) != CATALOG_DTYPE.itemsize
            or not catalog_path.is_file()
            or catalog_path.stat().st_size != int(catalog["bytes"])
            or sha256_file(catalog_path) != catalog["sha256"]
            or int(catalog["bytes"]) != int(catalog["rows"]) * CATALOG_DTYPE.itemsize
        ):
            raise ValueError(f"training catalog drift: {catalog_path}")
        group_catalog = np.fromfile(catalog_path, dtype=CATALOG_DTYPE)
        for pool_code in np.unique(group_catalog["pool"]):
            code = int(pool_code)
            if code not in POOL_CODES:
                raise ValueError(f"unknown pool code in training catalog: {code}")
            catalog_arrays[code].append(group_catalog[group_catalog["pool"] == code])
        for task_id in receipt["task_ids"]:
            if task_id in all_tasks:
                raise ValueError(f"training task appears in multiple groups: {task_id}")
            all_tasks.add(task_id)
        for shard in receipt["shards"]:
            pool = str(shard["logical_pool"])
            pool_counts[pool]["documents"] += int(shard["documents"])
            pool_counts[pool]["tokens"] += int(shard["tokens"])
            pool_counts[pool]["shards"] += 1
        receipts.append(receipt)
    if len(all_tasks) != args.expected_training_tasks:
        raise ValueError("validated training-task coverage is incomplete")

    identities = np.concatenate(identity_arrays)
    total_identities = int(identities.size)
    unique_identities = int(np.unique(identities).size)
    if unique_identities != total_identities:
        raise ValueError(
            "global duplicate identity or 128-bit digest collision detected: "
            f"{total_identities - unique_identities}"
        )
    modern_contents = np.concatenate(modern_content_arrays)
    modern_identities = np.concatenate(modern_identity_arrays)
    if modern_identities.size != modern_contents.size:
        raise RuntimeError("Modern-Greek identity/content digest alignment drift")
    modern_content_rows = int(modern_contents.size)
    unique_modern_contents = int(np.unique(modern_contents).size)
    modern_duplicate_row_excess = modern_content_rows - unique_modern_contents
    order = np.argsort(modern_contents, kind="stable")
    duplicate_positions = np.flatnonzero(modern_contents[order[1:]] == modern_contents[order[:-1]])
    duplicate_content_digests = sorted(
        {bytes(modern_contents[order[int(position)]]).hex() for position in duplicate_positions}
    )
    duplicate_identity_candidates: dict[str, list[bytes]] = {}
    for digest_hex in duplicate_content_digests:
        target = np.void(bytes.fromhex(digest_hex))
        duplicate_identity_candidates[digest_hex] = [
            bytes(modern_identities[int(position)])
            for position in np.flatnonzero(modern_contents == target)
        ]

    diagnostic_receipt = None
    if duplicate_content_digests:
        diagnostic_path = args.stage_root / "validation" / "modern_content_duplicate_diagnostic.json"
        diagnostic = read_json(diagnostic_path)
        diagnosed = {
            str(row["text_sha256_prefix_128"])
            for row in diagnostic.get("duplicates", [])
        }
        if (
            diagnostic.get("schema_version")
            != "apertus_mini_modern_content_duplicate_diagnostic_v1"
            or diagnostic.get("status") != "completed"
            or diagnosed != set(duplicate_content_digests)
            or int(diagnostic.get("duplicate_row_excess", -1))
            != modern_duplicate_row_excess
        ):
            raise ValueError("Modern-Greek duplicate diagnostic does not bind all collisions")
        diagnostic_receipt = {
            "path": str(diagnostic_path),
            "sha256": sha256_file(diagnostic_path),
            "bytes": diagnostic_path.stat().st_size,
        }

    candidate_bytes = {
        identity
        for identities_for_content in duplicate_identity_candidates.values()
        for identity in identities_for_content
    }
    candidate_array = np.asarray(sorted(candidate_bytes), dtype="V16")
    retained_duplicate_candidates: dict[bytes, dict[str, Any]] = {}
    if candidate_array.size:
        for code, arrays_for_pool in catalog_arrays.items():
            for group_catalog in arrays_for_pool:
                for row in group_catalog[np.isin(group_catalog["identity"], candidate_array)]:
                    identity = bytes(row["identity"])
                    if identity in retained_duplicate_candidates:
                        raise ValueError("duplicate candidate identity appears twice in training catalogs")
                    retained_duplicate_candidates[identity] = {
                        "pool_code": code,
                        "pool": POOL_CODES[code],
                        "task_index": int(row["task_index"]),
                        "document_index": int(row["document_index"]),
                        "tokens": int(row["tokens"]),
                    }

    training_duplicate_exclusions: list[dict[str, Any]] = []
    for digest_hex, identities_for_content in duplicate_identity_candidates.items():
        retained = sorted(
            identity
            for identity in identities_for_content
            if identity in retained_duplicate_candidates
        )
        for identity in retained[1:]:
            training_duplicate_exclusions.append(
                {
                    "content_sha256_prefix_128": digest_hex,
                    "record_identity_sha256_prefix_128": identity.hex(),
                    "deterministic_keep_rule": "keep_lexicographically_smallest_retained_record_identity_sha256_prefix_128",
                    **retained_duplicate_candidates[identity],
                }
            )
    excluded_identity_bytes = {
        bytes.fromhex(row["record_identity_sha256_prefix_128"])
        for row in training_duplicate_exclusions
    }
    excluded_identity_array = np.asarray(sorted(excluded_identity_bytes), dtype="V16")
    replay_contents = np.concatenate(replay_content_arrays)
    replay_content_rows = int(replay_contents.size)
    unique_replay_contents = int(np.unique(replay_contents).size)
    all_contents = np.concatenate((modern_contents, replay_contents))
    all_content_rows = int(all_contents.size)
    unique_all_contents = int(np.unique(all_contents).size)

    sorted_catalog_receipts: dict[str, dict[str, Any]] = {}
    retained_identity_arrays: list[np.ndarray] = []
    raw_pool_counts = {pool: dict(counts) for pool, counts in pool_counts.items()}
    catalog_root = args.stage_root / "catalog"
    catalog_root.mkdir(parents=True, exist_ok=True)
    for code, pool_name in POOL_CODES.items():
        rows = np.concatenate(catalog_arrays[code])
        if int(rows["tokens"].astype(np.uint64).sum()) != pool_counts[pool_name]["tokens"]:
            raise ValueError(f"catalog token accounting drift: {pool_name}")
        if rows.size != pool_counts[pool_name]["documents"]:
            raise ValueError(f"catalog document accounting drift: {pool_name}")
        if excluded_identity_array.size:
            rows = rows[~np.isin(rows["identity"], excluded_identity_array)]
        retained_identity_arrays.append(rows["identity"])
        pool_counts[pool_name]["documents"] = int(rows.size)
        pool_counts[pool_name]["tokens"] = int(rows["tokens"].astype(np.uint64).sum())
        order = np.lexsort((rows["identity"], rows["order"]))
        rows = rows[order]
        if rows.size > 1 and np.any(rows["order"][1:] == rows["order"][:-1]):
            raise ValueError(f"128-bit seeded order collision detected: {pool_name}")
        path = catalog_root / f"{pool_name}.sorted.catalog45"
        temporary = Path(str(path) + ".partial")
        rows.tofile(temporary)
        os.replace(temporary, path)
        sorted_catalog_receipts[pool_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": int(rows.size),
            "tokens": int(rows["tokens"].astype(np.uint64).sum()),
            "record_bytes": CATALOG_DTYPE.itemsize,
            "sort_key": "seeded_order_sha256_prefix_128_then_identity_sha256_prefix_128",
            "schedule_seed": 20260801,
        }
        del rows, order

    retained_identities = np.concatenate(retained_identity_arrays)
    retained_identity_rows = int(retained_identities.size)
    retained_unique_rows = int(np.unique(retained_identities).size)
    if retained_unique_rows != retained_identity_rows:
        raise ValueError("global duplicate retained training identity or 128-bit collision detected")

    heldout_manifests = sorted((args.stage_root / "megatron" / "heldout").glob("*.manifest.json"))
    if len(heldout_manifests) != args.expected_heldout_tasks:
        raise ValueError("heldout binary inventory is incomplete")
    heldout = []
    for path in heldout_manifests:
        value = read_json(path)
        if value.get("status") != "completed" or value.get("kind") != "heldout":
            raise ValueError(f"invalid heldout binary manifest: {path}")
        heldout.append(
            {
                "name": value["heldout_name"],
                "documents": int(value["counts"]["documents"]),
                "tokens": int(value["counts"]["tokens"]),
                "manifest_path": str(path),
                "manifest_sha256": sha256_file(path),
            }
        )

    required_pools = {
        "hplt_new_greek",
        "non_hplt_new_greek",
        "foreign_replay",
        "old_greek_replay",
    }
    if set(pool_counts) != required_pools:
        raise ValueError(f"logical pool inventory drift: {sorted(pool_counts)}")
    hplt = pool_counts["hplt_new_greek"]["tokens"]
    glossapi = pool_counts["non_hplt_new_greek"]["tokens"]
    modern = hplt + glossapi
    total_fractional = Decimal(modern) / Decimal("0.79")
    foreign_required = total_fractional * Decimal("0.20")
    old_required = total_fractional * Decimal("0.01")
    if Decimal(pool_counts["foreign_replay"]["tokens"]) < foreign_required:
        raise ValueError("foreign replay capacity is below the 20% requirement")
    if Decimal(pool_counts["old_greek_replay"]["tokens"]) < old_required:
        raise ValueError("Old-Greek replay capacity is below the 1% requirement")

    output = args.stage_root / "pool_corpus_receipt.json"
    payload = {
        "schema_version": "apertus_mini_schedule_pool_corpus_v1",
        "status": "completed",
        "stage_root": str(args.stage_root.resolve()),
        "tokenizer": read_json(args.stage_root / "input_receipt.json")["tokenizer"],
        "global_identity_proof": {
            "record_algorithm": "unique_over_sha256_prefix_128_of_doc_id_plus_text_sha256",
            "record_rows": total_identities,
            "unique_record_digest_rows": unique_identities,
            "record_duplicates_or_collisions": 0,
            "modern_greek_content_algorithm": "global_exact_content_dedup_over_text_sha256_prefix_128_before_training_catalog_freeze",
            "modern_greek_eligible_content_rows_before_dedup": modern_content_rows,
            "modern_greek_unique_eligible_content_digest_rows": unique_modern_contents,
            "modern_greek_pre_dedup_duplicate_rows_or_collisions": modern_duplicate_row_excess,
            "modern_greek_training_rows_excluded_by_global_content_dedup": len(training_duplicate_exclusions),
            "modern_greek_retained_training_content_rows": pool_counts["hplt_new_greek"]["documents"]
            + pool_counts["non_hplt_new_greek"]["documents"],
            "modern_greek_exact_content_duplicates_or_collisions": 0,
            "modern_greek_duplicate_diagnostic": diagnostic_receipt,
            "replay_content_policy": "audit_only_preserve_original_training_replay_records_and_report_duplicates",
            "replay_content_rows": replay_content_rows,
            "unique_replay_content_digest_rows": unique_replay_contents,
            "replay_exact_content_duplicate_rows_or_collisions": replay_content_rows
            - unique_replay_contents,
            "all_content_rows": all_content_rows,
            "unique_all_content_digest_rows": unique_all_contents,
            "all_exact_content_duplicate_rows_or_collisions": all_content_rows
            - unique_all_contents,
            "partition_groups": args.expected_groups,
            "training_tasks": len(all_tasks),
            "retained_training_identity_rows": retained_identity_rows,
            "unique_retained_training_digest_rows": retained_unique_rows,
        },
        "sorted_training_catalogs": sorted_catalog_receipts,
        "raw_physical_pool_counts_before_global_modern_greek_content_dedup": dict(
            sorted(raw_pool_counts.items())
        ),
        "modern_greek_global_content_dedup_exclusions": training_duplicate_exclusions,
        "logical_pools": dict(sorted(pool_counts.items())),
        "modern_greek": {
            "tokens": modern,
            "hplt_tokens": hplt,
            "glossapi_non_hplt_tokens": glossapi,
            "hplt_fraction": str(Decimal(hplt) / Decimal(modern)),
            "glossapi_non_hplt_fraction": str(Decimal(glossapi) / Decimal(modern)),
        },
        "provisional_79_20_1_geometry": {
            "total_tokens_fractional": str(total_fractional),
            "foreign_replay_tokens_fractional": str(foreign_required),
            "old_greek_replay_tokens_fractional": str(old_required),
            "foreign_capacity_tokens": pool_counts["foreign_replay"]["tokens"],
            "old_greek_capacity_tokens": pool_counts["old_greek_replay"]["tokens"],
            "integer_sequence_quota_pending_packing": True,
        },
        "heldouts": heldout,
        "all_shard_payload_hashes_reverified_in_parallel": True,
    }
    write_json_atomic(output, payload)
    print(json.dumps({"ok": True, "output": str(output), "modern_greek_tokens": modern}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
