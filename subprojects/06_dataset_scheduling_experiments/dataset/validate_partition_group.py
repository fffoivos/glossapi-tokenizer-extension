#!/usr/bin/env python3
"""Validate one physical partition group against its frozen source records.

The tokenization job splits each source file into mutually exclusive physical
partitions.  This validator independently reconstructs the pre-partition
eligible record set, proves that retained plus decontamination-dropped ledgers
cover it exactly once, and re-hashes every generated payload.  A source
``doc_id`` is a cluster key and may legitimately name several distinct text
records, so exact records are identified by ``(doc_id, text_sha256)``.  The
validator emits separate 128-bit record and content digests: the former proves
row coverage and the latter proves global exact-content deduplication.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEDULE_SEED = 20260801
POOL_CODES = {
    "hplt_new_greek": 0,
    "non_hplt_new_greek": 1,
    "foreign_replay": 2,
    "old_greek_replay": 3,
}
CATALOG_RECORD = struct.Struct("<BIII16s16s")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def task_groups(tasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for task in tasks:
        key = (str(task["pool"]), str(task["source_name"]), str(task["input_path"]))
        grouped.setdefault(key, []).append(task)
    result = [sorted(rows, key=lambda row: int(row["task_index"])) for rows in grouped.values()]
    return sorted(result, key=lambda rows: int(rows[0]["task_index"]))


def expected_partition_shape(group: list[dict[str, Any]]) -> list[tuple[int, str]]:
    observed = sorted(
        (
            int(task["phase_partition"]["phase"]),
            str(task["phase_partition"]["logical_pool"]),
        )
        for task in group
    )
    corpus = str(group[0]["phase_partition"]["corpus"])
    if corpus == "new_greek":
        expected = [
            (1, "hplt_new_greek"),
            (2, "hplt_new_greek"),
            (2, "non_hplt_new_greek"),
        ]
    elif corpus == "replay":
        logical = str(group[0]["phase_partition"]["logical_pool"])
        expected = [(1, logical), (2, logical)]
    else:
        raise ValueError(f"unsupported partition corpus: {corpus}")
    if observed != expected:
        raise ValueError(f"physical partition shape drift: {observed} != {expected}")
    return observed


def eligible(row: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    field = str(task.get("filter_field") or "")
    if not field:
        return True
    value = row.get(field)
    if value is None:
        return False
    minimum = task.get("filter_min")
    if minimum is None:
        return True
    try:
        return float(value) >= float(minimum)
    except (TypeError, ValueError):
        return False


def record_identity(doc_id: str, text_sha256: str) -> str:
    if not doc_id or len(text_sha256) != 64:
        raise ValueError("invalid record identity components")
    return f"{doc_id}\0{text_sha256}"


def iter_expected(
    task: Mapping[str, Any], exclusions: set[str], bridge_common: Any
) -> Iterable[tuple[str, str]]:
    import pyarrow.parquet as pq

    columns = [str(task["text_column"])]
    for value in (*task.get("identity_columns", []), task.get("filter_field")):
        if value and str(value) not in columns:
            columns.append(str(value))
    parquet = pq.ParquetFile(str(task["input_path"]))
    row_index = 0
    for batch in parquet.iter_batches(columns=columns, batch_size=4096, use_threads=False):
        data = batch.to_pydict()
        for offset in range(batch.num_rows):
            row = {column: data[column][offset] for column in columns}
            text = row.get(str(task["text_column"]))
            if not eligible(row, task) or not isinstance(text, str) or not text:
                continue
            doc_id = bridge_common.document_key(
                str(task["source_name"]),
                str(task["input_relative"]),
                row_index + offset,
                {
                    str(column): row.get(str(column))
                    for column in task.get("identity_columns", [])
                },
                identity_scope=str(task["identity_scope"]),
            )
            if doc_id not in exclusions:
                text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                yield record_identity(doc_id, text_sha256), text_sha256
        row_index += batch.num_rows


def output_prefix(stage: Path, task: Mapping[str, Any]) -> Path:
    return (stage / "megatron" / str(task["output_prefix"])).resolve()


def validate_manifest(
    stage: Path,
    task: Mapping[str, Any],
    input_sha: str,
    heldout_sha: str,
    bridge_common: Any,
) -> tuple[dict[str, Any], set[str], set[str], bytes, int, int]:
    prefix = output_prefix(stage, task)
    path = Path(str(prefix) + ".manifest.json")
    manifest = read_json(path)
    expected = {
        "schema_version": "full_cpt_megatron_shard_v1",
        "status": "completed",
        "task_id": task["task_id"],
        "task_sha256": bridge_common.canonical_sha256(task),
        "input_receipt_sha256": input_sha,
        "heldout_manifest_sha256": heldout_sha,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest binding drift ({key}): {path}")

    for label, suffix in {
        "bin": ".bin",
        "idx": ".idx",
        "dropped_ledger": ".dropped.jsonl",
        "retained_ledger": ".retained.jsonl",
    }.items():
        receipt = manifest["outputs"][label]
        payload = Path(str(receipt["path"]))
        if payload.resolve() != Path(str(prefix) + suffix).resolve():
            raise ValueError(f"payload path drift: {payload}")
        if (
            not payload.is_file()
            or payload.is_symlink()
            or payload.stat().st_size != int(receipt["bytes"])
            or sha256_file(payload) != receipt["sha256"]
        ):
            raise ValueError(f"payload integrity failure: {payload}")
    sequences, entries, tokens = bridge_common.iter_index_lengths(
        Path(manifest["outputs"]["idx"]["path"])
    )
    counts = manifest["counts"]
    if (sequences, entries, tokens) != (
        int(counts["documents"]),
        int(counts["document_index_entries"]),
        int(counts["tokens"]),
    ):
        raise ValueError(f"index accounting drift: {path}")
    if int(manifest["outputs"]["bin"]["bytes"]) != tokens * 4:
        raise ValueError(f"binary byte accounting drift: {path}")

    identities: set[str] = set()
    contents: set[str] = set()
    source_doc_ids: set[str] = set()
    repeated_source_doc_id_rows = 0
    catalog = bytearray()
    logical_pool = str(task["phase_partition"]["logical_pool"])
    if logical_pool not in POOL_CODES:
        raise ValueError(f"unknown logical pool: {logical_pool}")
    require_unique_content = logical_pool in {
        "hplt_new_greek",
        "non_hplt_new_greek",
    }
    duplicate_content_rows = 0
    ledger_specs = (
        ("retained_ledger", int(counts["documents"])),
        ("dropped_ledger", int(counts["contaminated_rows"])),
    )
    for label, expected_rows in ledger_specs:
        ledger = Path(manifest["outputs"][label]["path"])
        rows = 0
        with ledger.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = str(row.get("doc_id", ""))
                text_sha256 = str(row.get("text_sha256", ""))
                identity = record_identity(doc_id, text_sha256)
                if identity in identities:
                    raise ValueError(f"duplicate exact record inside physical shard: {doc_id}")
                if text_sha256 in contents:
                    duplicate_content_rows += 1
                    if require_unique_content:
                        raise ValueError(
                            f"duplicate exact Modern-Greek content inside physical shard: {text_sha256}"
                        )
                identities.add(identity)
                contents.add(text_sha256)
                if doc_id in source_doc_ids:
                    repeated_source_doc_id_rows += 1
                source_doc_ids.add(doc_id)
                if label == "retained_ledger":
                    tokens = int(row.get("tokens", 0))
                    if tokens <= 0:
                        raise ValueError(f"invalid retained token count: {ledger}")
                    identity_digest = hashlib.sha256(identity.encode("utf-8")).digest()[:16]
                    order_digest = hashlib.sha256(
                        (
                            "apertus-schedule-order-v1\0"
                            f"{SCHEDULE_SEED}\0{logical_pool}\0{identity}"
                        ).encode("utf-8")
                    ).digest()[:16]
                    catalog.extend(
                        CATALOG_RECORD.pack(
                            POOL_CODES[logical_pool],
                            int(task["task_index"]),
                            rows,
                            tokens,
                            identity_digest,
                            order_digest,
                        )
                    )
                rows += 1
        if rows != expected_rows:
            raise ValueError(f"ledger row-count drift: {ledger}")
    if len(identities) != int(counts["candidate_rows"]):
        raise ValueError(f"candidate identity accounting drift: {path}")
    if len(catalog) != int(counts["documents"]) * CATALOG_RECORD.size:
        raise RuntimeError(f"retained catalog accounting drift: {path}")
    return (
        manifest,
        identities,
        contents,
        bytes(catalog),
        repeated_source_doc_id_rows,
        duplicate_content_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--bridge-common", type=Path, required=True)
    parser.add_argument("--group-index", type=int)
    parser.add_argument("--print-group-count", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_receipt = read_json(args.input_receipt)
    groups = task_groups(input_receipt["tasks"])
    if args.print_group_count:
        print(len(groups))
        return 0
    if args.group_index is None or not 0 <= args.group_index < len(groups):
        raise ValueError(f"group index must be in [0,{len(groups)})")
    bridge_common = load_module("partition_validation_bridge_common", args.bridge_common)
    group = groups[args.group_index]
    shape = expected_partition_shape(group)
    representative = group[0]
    for task in group[1:]:
        for key in (
            "input_path",
            "input_sha256",
            "input_bytes",
            "input_rows",
            "text_column",
            "identity_columns",
            "identity_scope",
            "filter_field",
            "filter_min",
            "requires_heldout_exclusion",
            "exclusion_key",
            "exclusion_file",
        ):
            if task.get(key) != representative.get(key):
                raise ValueError(f"partition group input contract drift ({key})")

    input_sha = sha256_file(args.input_receipt)
    heldout_sha = sha256_file(args.heldout_manifest)
    heldout = read_json(args.heldout_manifest)
    if heldout.get("input_receipt_sha256") != input_sha:
        raise ValueError("heldout manifest is bound to a different input receipt")
    exclusions: set[str] = set()
    if representative.get("requires_heldout_exclusion"):
        key = str(representative["exclusion_key"])
        receipt = heldout["exclusions"][key]
        path = (args.stage_root / str(representative["exclusion_file"])).resolve()
        if (
            path != Path(receipt["path"]).resolve()
            or not path.is_file()
            or sha256_file(path) != receipt["sha256"]
        ):
            raise ValueError("heldout exclusion binding drift")
        exclusions = bridge_common.load_exclusion_ids(path)

    observed: set[str] = set()
    observed_contents: set[str] = set()
    manifests: list[dict[str, Any]] = []
    catalog_bytes = bytearray()
    repeated_source_doc_id_rows = 0
    duplicate_content_rows_within_shards = 0
    duplicate_content_hashes_across_partitions: set[str] = set()
    require_unique_content = str(representative["phase_partition"]["corpus"]) == "new_greek"
    for task in group:
        (
            manifest,
            identities,
            contents,
            shard_catalog,
            repeated_rows,
            duplicate_content_rows,
        ) = validate_manifest(args.stage_root, task, input_sha, heldout_sha, bridge_common)
        overlap = observed.intersection(identities)
        if overlap:
            raise ValueError(f"record appears in multiple physical partitions: {next(iter(overlap))}")
        content_overlap = observed_contents.intersection(contents)
        if content_overlap and require_unique_content:
            raise ValueError(
                "exact Modern-Greek content appears in multiple physical partitions: "
                f"{next(iter(content_overlap))}"
            )
        duplicate_content_hashes_across_partitions.update(content_overlap)
        observed.update(identities)
        observed_contents.update(contents)
        manifests.append(manifest)
        catalog_bytes.extend(shard_catalog)
        repeated_source_doc_id_rows += repeated_rows
        duplicate_content_rows_within_shards += duplicate_content_rows

    digest_path = args.stage_root / "validation" / "partition_groups" / f"{args.group_index:04d}.records128"
    content_path = args.stage_root / "validation" / "partition_groups" / f"{args.group_index:04d}.content128"
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_tmp = Path(str(digest_path) + ".partial")
    content_tmp = Path(str(content_path) + ".partial")
    digest_tmp.unlink(missing_ok=True)
    content_tmp.unlink(missing_ok=True)
    expected_count = 0
    with digest_tmp.open("wb") as digest_handle, content_tmp.open("wb") as content_handle:
        for identity, text_sha256 in iter_expected(representative, exclusions, bridge_common):
            if identity not in observed:
                raise ValueError(f"expected record absent from all partitions: {identity}")
            observed.remove(identity)
            digest_handle.write(hashlib.sha256(identity.encode("utf-8")).digest()[:16])
            content_handle.write(bytes.fromhex(text_sha256)[:16])
            expected_count += 1
        digest_handle.flush()
        os.fsync(digest_handle.fileno())
        content_handle.flush()
        os.fsync(content_handle.fileno())
    if observed:
        raise ValueError(f"partition ledger contains unexpected record: {next(iter(observed))}")
    os.replace(digest_tmp, digest_path)
    os.replace(content_tmp, content_path)
    if digest_path.stat().st_size != expected_count * 16:
        raise RuntimeError("record digest byte accounting drift")
    if content_path.stat().st_size != expected_count * 16:
        raise RuntimeError("content digest byte accounting drift")

    catalog_path = args.stage_root / "validation" / "partition_groups" / f"{args.group_index:04d}.catalog45"
    catalog_tmp = Path(str(catalog_path) + ".partial")
    catalog_tmp.write_bytes(catalog_bytes)
    with catalog_tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(catalog_tmp, catalog_path)
    retained_documents = sum(int(manifest["counts"]["documents"]) for manifest in manifests)
    if catalog_path.stat().st_size != retained_documents * CATALOG_RECORD.size:
        raise RuntimeError("training catalog byte accounting drift")

    receipt_path = args.stage_root / "validation" / "partition_groups" / f"{args.group_index:04d}.json"
    receipt = {
        "schema_version": "apertus_mini_partition_group_validation_v2",
        "status": "completed",
        "group_index": args.group_index,
        "input": {
            "pool": representative["pool"],
            "source_name": representative["source_name"],
            "path": representative["input_path"],
            "sha256": representative["input_sha256"],
            "rows": representative["input_rows"],
        },
        "partition_shape": [
            {"phase": phase, "logical_pool": logical_pool}
            for phase, logical_pool in shape
        ],
        "expected_eligible_records": expected_count,
        "missing_records": 0,
        "unexpected_records": 0,
        "duplicate_partition_memberships": 0,
        "reused_source_doc_id_rows_with_distinct_content": repeated_source_doc_id_rows,
        "content_uniqueness": {
            "policy": (
                "required_globally_for_modern_greek"
                if require_unique_content
                else "audit_only_preserve_original_training_replay_records"
            ),
            "duplicate_rows_within_physical_shards": duplicate_content_rows_within_shards,
            "duplicate_hashes_across_physical_partitions": len(
                duplicate_content_hashes_across_partitions
            ),
        },
        "task_ids": [task["task_id"] for task in group],
        "shards": [
            {
                "task_id": manifest["task_id"],
                "logical_pool": task["phase_partition"]["logical_pool"],
                "phase": task["phase_partition"]["phase"],
                "documents": manifest["counts"]["documents"],
                "tokens": manifest["counts"]["tokens"],
                "manifest_path": str(Path(str(output_prefix(args.stage_root, task)) + ".manifest.json")),
                "manifest_sha256": sha256_file(Path(str(output_prefix(args.stage_root, task)) + ".manifest.json")),
            }
            for task, manifest in zip(group, manifests, strict=True)
        ],
        "record_identity_digest": {
            "algorithm": "sha256_prefix_128",
            "path": str(digest_path),
            "sha256": sha256_file(digest_path),
            "bytes": digest_path.stat().st_size,
            "rows": expected_count,
        },
        "content_digest": {
            "algorithm": "text_sha256_prefix_128",
            "path": str(content_path),
            "sha256": sha256_file(content_path),
            "bytes": content_path.stat().st_size,
            "rows": expected_count,
        },
        "training_catalog": {
            "format": "little_endian_BIII_16s_16s_v1",
            "record_bytes": CATALOG_RECORD.size,
            "fields": [
                "pool_code_u8",
                "task_index_u32",
                "document_index_u32",
                "tokens_u32",
                "record_identity_sha256_prefix_128",
                "seeded_order_sha256_prefix_128"
            ],
            "schedule_seed": SCHEDULE_SEED,
            "path": str(catalog_path),
            "sha256": sha256_file(catalog_path),
            "bytes": catalog_path.stat().st_size,
            "rows": retained_documents,
        },
        "all_generated_payload_sha256_verified": True,
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({"ok": True, "group": args.group_index, "identities": expected_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
