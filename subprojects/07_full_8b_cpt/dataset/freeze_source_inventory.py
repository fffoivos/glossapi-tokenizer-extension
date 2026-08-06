#!/usr/bin/env python3
"""Freeze full-run catalogs over the existing production-tokenizer binaries.

This is a metadata-only pass. It never retokenizes text and never rewrites the
source Megatron binaries. The retained ledgers are read in indexed-document
order, so every catalog row remains bound to its original ``.bin/.idx`` row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from collections import defaultdict
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

import numpy as np


getcontext().prec = 60
SEED = 20260801
POOLS = (
    "hplt_new_greek",
    "non_hplt_new_greek",
    "foreign_replay",
    "old_greek_replay",
)
POOL_CODES = {name: index for index, name in enumerate(POOLS)}
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
CONTENT_DTYPE = np.dtype(
    [("content", "V32"), ("pool", "u1"), ("row", "<u8"), ("identity", "V16")],
    align=False,
)
CATALOG_STRUCT = struct.Struct("<BIII16s16s")
CONTENT_STRUCT = struct.Struct("<32sBQ16s")


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


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
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


def classify_manifest(path: Path, manifest: dict[str, Any]) -> str:
    pool = str(manifest.get("pool", ""))
    if pool in POOL_CODES:
        return pool
    text = path.as_posix()
    for candidate in POOLS:
        if f"/{candidate}/" in text:
            return candidate
    raise ValueError(f"unrecognized training pool: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--sanitized-bridge-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bridge_path = args.sanitized_bridge_receipt.resolve()
    bridge = read_json(bridge_path)
    if (
        bridge.get("schema_version") != "full_cpt_sanitized_training_bridge_v1"
        or bridge.get("status") != "completed"
        or Path(bridge.get("stage_root", "")).resolve() != args.source_root.resolve()
        or int(bridge.get("eligibility_audit", {}).get("bytes", -1)) <= 0
    ):
        raise ValueError("source root is not bound to a completed sanitized bridge")
    bridge_counts = bridge["expected_inventory_tokens"]
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")
    megatron_root = args.source_root / "megatron"
    manifest_paths = sorted(
        path
        for phase in ("phase_1", "phase_2")
        for path in (megatron_root / phase).rglob("*.manifest.json")
    )
    if not manifest_paths:
        raise ValueError("no production source manifests found")
    args.output_dir.mkdir(parents=True)
    raw_dir = args.output_dir / "raw"
    catalog_dir = args.output_dir / "catalog"
    raw_dir.mkdir()
    catalog_dir.mkdir()
    raw_paths = {pool: raw_dir / f"{pool}.catalog45" for pool in POOLS}
    raw_handles = {pool: path.open("wb") for pool, path in raw_paths.items()}
    modern_content_path = raw_dir / "modern.content57"
    modern_content = modern_content_path.open("wb")
    pool_counts = defaultdict(lambda: {"documents": 0, "tokens": 0, "tasks": 0})
    tasks: list[dict[str, Any]] = []
    tokenizer_tree_hashes: set[str] = set()
    try:
        for manifest_path in manifest_paths:
            manifest = read_json(manifest_path)
            if manifest.get("status") != "completed" or manifest.get("kind") != "training":
                raise ValueError(f"incomplete source manifest: {manifest_path}")
            documents = int(manifest["counts"]["documents"])
            tokens = int(manifest["counts"]["tokens"])
            if documents == 0:
                if tokens != 0:
                    raise ValueError(f"zero-document manifest carries tokens: {manifest_path}")
                continue
            pool = classify_manifest(manifest_path, manifest)
            outputs = manifest["outputs"]
            for key in ("bin", "idx", "retained_ledger"):
                receipt = outputs[key]
                path = Path(receipt["path"])
                if not path.is_file() or path.stat().st_size != int(receipt["bytes"]):
                    raise ValueError(f"source payload size drift: {path}")
            retained_receipt = outputs["retained_ledger"]
            if int(retained_receipt["rows"]) != documents:
                raise ValueError(f"retained-ledger row drift: {manifest_path}")
            task_index = len(tasks)
            prefix = Path(manifest["output_prefix"]).resolve()
            tasks.append(
                {
                    "task_index": task_index,
                    "pool": pool,
                    "output_prefix": str(prefix),
                    "source_manifest": {
                        "path": str(manifest_path.resolve()),
                        "sha256": sha256_file(manifest_path),
                        "bytes": manifest_path.stat().st_size,
                    },
                    "documents": documents,
                    "tokens": tokens,
                }
            )
            tokenizer_tree_hashes.add(str(manifest["tokenizer_tree_sha256"]))
            ledger_path = Path(retained_receipt["path"])
            observed_documents = 0
            observed_tokens = 0
            pool_row_start = pool_counts[pool]["documents"]
            with ledger_path.open("r", encoding="utf-8") as handle:
                for document_index, line in enumerate(handle):
                    row = json.loads(line)
                    token_count = int(row["tokens"])
                    doc_id = str(row["doc_id"])
                    text_sha = bytes.fromhex(str(row["text_sha256"]))
                    if len(text_sha) != 32 or token_count <= 0:
                        raise ValueError(f"invalid retained row: {ledger_path}:{document_index + 1}")
                    identity = hashlib.sha256(
                        doc_id.encode("utf-8") + b"\0" + text_sha
                    ).digest()[:16]
                    order = hashlib.sha256(
                        SEED.to_bytes(8, "little") + identity
                    ).digest()[:16]
                    raw_handles[pool].write(
                        CATALOG_STRUCT.pack(
                            POOL_CODES[pool],
                            task_index,
                            document_index,
                            token_count,
                            identity,
                            order,
                        )
                    )
                    if pool in {"hplt_new_greek", "non_hplt_new_greek"}:
                        modern_content.write(
                            CONTENT_STRUCT.pack(
                                text_sha,
                                POOL_CODES[pool],
                                pool_row_start + observed_documents,
                                identity,
                            )
                        )
                    observed_documents += 1
                    observed_tokens += token_count
            if observed_documents != documents or observed_tokens != tokens:
                raise ValueError(f"retained-ledger accounting drift: {manifest_path}")
            pool_counts[pool]["documents"] += documents
            pool_counts[pool]["tokens"] += tokens
            pool_counts[pool]["tasks"] += 1
    finally:
        for handle in raw_handles.values():
            handle.close()
        modern_content.close()

    if len(tokenizer_tree_hashes) != 1:
        raise ValueError(f"source tokenizer-tree drift: {sorted(tokenizer_tree_hashes)}")
    modern_before = (
        pool_counts["hplt_new_greek"]["tokens"]
        + pool_counts["non_hplt_new_greek"]["tokens"]
    )
    if modern_before != int(bridge_counts["modern"]):
        raise ValueError(f"Modern-Greek token drift: {modern_before}")
    if pool_counts["foreign_replay"]["tokens"] != int(bridge_counts["foreign"]):
        raise ValueError("foreign-replay token drift")
    if pool_counts["old_greek_replay"]["tokens"] != int(bridge_counts["old_greek"]):
        raise ValueError("Old-Greek replay token drift")

    contents = np.memmap(modern_content_path, mode="r", dtype=CONTENT_DTYPE)
    content_order = np.argsort(contents["content"], kind="stable")
    repeated = np.flatnonzero(
        contents["content"][content_order[1:]] == contents["content"][content_order[:-1]]
    )
    duplicate_groups: list[dict[str, Any]] = []
    excluded_identities: set[bytes] = set()
    if repeated.size:
        starts = sorted({int(index) for index in repeated} | {int(index + 1) for index in repeated})
        group: list[int] = []
        previous: bytes | None = None
        for sorted_position in starts:
            row_index = int(content_order[sorted_position])
            digest = bytes(contents[row_index]["content"])
            if previous is not None and digest != previous:
                identities = sorted(bytes(contents[index]["identity"]) for index in group)
                excluded_identities.update(identities[1:])
                duplicate_groups.append(
                    {"text_sha256": previous.hex(), "rows": len(group), "kept_identity": identities[0].hex()}
                )
                group = []
            group.append(row_index)
            previous = digest
        if group and previous is not None:
            identities = sorted(bytes(contents[index]["identity"]) for index in group)
            excluded_identities.update(identities[1:])
            duplicate_groups.append(
                {"text_sha256": previous.hex(), "rows": len(group), "kept_identity": identities[0].hex()}
            )
    del contents, content_order, repeated

    sorted_catalogs: dict[str, dict[str, Any]] = {}
    retained_counts: dict[str, dict[str, int]] = {}
    all_identity_arrays: list[np.ndarray] = []
    excluded = np.asarray(sorted(excluded_identities), dtype="V16")
    for pool in POOLS:
        rows = np.memmap(raw_paths[pool], mode="r", dtype=CATALOG_DTYPE)
        selected = np.array(rows, copy=True)
        if pool in {"hplt_new_greek", "non_hplt_new_greek"} and excluded.size:
            selected = selected[~np.isin(selected["identity"], excluded)]
        order = np.argsort(selected["order"], kind="stable")
        selected = selected[order]
        if selected.size > 1 and np.any(selected["order"][1:] == selected["order"][:-1]):
            raise ValueError(f"seeded-order digest collision: {pool}")
        output = catalog_dir / f"{pool}.sorted.catalog45"
        selected.tofile(output)
        tokens = int(selected["tokens"].astype(np.uint64).sum())
        retained_counts[pool] = {"documents": int(selected.size), "tokens": tokens}
        all_identity_arrays.append(selected["identity"].copy())
        sorted_catalogs[pool] = {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "rows": int(selected.size),
            "tokens": tokens,
            "record_bytes": CATALOG_DTYPE.itemsize,
            "sort_key": "sha256(seed_le64 || identity)_prefix128",
            "schedule_seed": SEED,
        }
        del rows, selected, order
    identities = np.concatenate(all_identity_arrays)
    if np.unique(identities).size != identities.size:
        raise ValueError("retained source identity appears more than once")

    hplt = retained_counts["hplt_new_greek"]["tokens"]
    glossapi = retained_counts["non_hplt_new_greek"]["tokens"]
    modern = hplt + glossapi
    total = nearest_ratio_total(modern)
    old_target = nearest_percent(total)
    foreign_target = total - modern - old_target
    if retained_counts["foreign_replay"]["tokens"] < foreign_target:
        raise ValueError("foreign replay capacity is below the exact 20% quota")
    if retained_counts["old_greek_replay"]["tokens"] < old_target:
        raise ValueError("Old-Greek replay capacity is below the exact 1% quota")

    heldouts = []
    for path in sorted((megatron_root / "heldout").glob("*.manifest.json")):
        value = read_json(path)
        if value.get("status") != "completed" or value.get("kind") != "heldout":
            raise ValueError(f"invalid heldout manifest: {path}")
        heldouts.append(
            {
                "name": value["heldout_name"],
                "documents": int(value["counts"]["documents"]),
                "tokens": int(value["counts"]["tokens"]),
                "megatron_prefix": value["output_prefix"],
                "manifest_path": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
                "raw_jsonl": value["input"],
            }
        )

    tokenizer_json = args.tokenizer_root / "tokenizer.json"
    payload = {
        "schema_version": "apertus_schedule_pool_corpus_v1",
        "status": "completed",
        "source_root": str(args.source_root.resolve()),
        "sanitized_bridge": {
            "path": str(bridge_path),
            "sha256": sha256_file(bridge_path),
            "bytes": bridge_path.stat().st_size,
            "eligibility_audit": bridge["eligibility_audit"],
            "postmask_dedup": bridge["postmask_dedup"],
        },
        "tokenizer": {
            "root": str(args.tokenizer_root.resolve()),
            "tokenizer_json_sha256": sha256_file(tokenizer_json),
            "source_tokenizer_tree_sha256": next(iter(tokenizer_tree_hashes)),
            "vocab_size": 148_992,
            "pad_token_id": 3,
        },
        "tasks": tasks,
        "sorted_training_catalogs": sorted_catalogs,
        "raw_counts": dict(pool_counts),
        "logical_pools": retained_counts,
        "modern_greek": {
            "tokens": modern,
            "hplt_tokens": hplt,
            "glossapi_non_hplt_tokens": glossapi,
            "hplt_fraction": str(Decimal(hplt) / Decimal(modern)),
            "glossapi_non_hplt_fraction": str(Decimal(glossapi) / Decimal(modern)),
        },
        "global_identity_proof": {
            "retained_rows": int(identities.size),
            "unique_retained_rows": int(identities.size),
            "modern_exact_content_duplicate_groups": len(duplicate_groups),
            "modern_rows_excluded": len(excluded_identities),
            "duplicate_groups": duplicate_groups,
        },
        "integer_79_20_1_geometry": {
            "modern_greek": modern,
            "foreign_replay": foreign_target,
            "old_greek_replay": old_target,
            "active_tokens": total,
        },
        "heldouts": heldouts,
    }
    write_json_atomic(args.output_dir / "pool_corpus_receipt.json", payload)
    print(json.dumps({"ok": True, "active_tokens": total, "tasks": len(tasks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
