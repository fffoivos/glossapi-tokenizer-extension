#!/usr/bin/env python3
"""Reconstruct exact source exposure at every evaluated full-8B checkpoint.

The frozen D0 schedule identifies every packed sequence.  The packing catalogs
identify the exact source-document token intervals inside those sequences.
This program joins the retained Modern-Greek ledgers back to the frozen parquet
metadata, then reports loss-active tokens plus touched/fully-seen documents by
source and by HPLT web-register category.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


CHECKPOINTS = (
    0,
    400,
    1192,
    2384,
    3576,
    4768,
    5960,
    7152,
    8344,
    9536,
    10728,
    11920,
    13112,
    14304,
    14627,
    15496,
    16688,
    17880,
    18284,
)
GLOBAL_BATCH_SEQUENCES = 1024
SEQUENCE_LENGTH = 4096
FILLER_ID = np.uint64(2**64 - 1)
POOL_NAMES = {
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
REGISTER_RE = re.compile(r'"register_level_1"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
HOST_RE = re.compile(r'"host"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')


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


def file_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def stable_code(namespace: str, label: str) -> int:
    value = int.from_bytes(
        hashlib.blake2b(f"{namespace}\0{label}".encode("utf-8"), digest_size=8).digest(),
        "little",
    )
    return value or 1


def document_key(source_dataset: object, source_doc_id: object) -> str:
    components = []
    for key, value in (("source_dataset", source_dataset), ("source_doc_id", source_doc_id)):
        if value is not None and str(value):
            components.append([key, str(value)])
    payload = {
        "contract": "full-cpt-document-identity-v2",
        "source_name": "cleaned_greek_v2",
        "identity_scope": "global",
        "components": components,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "docv2:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def decode_json_string_fragment(value: str) -> str:
    if "\\" not in value:
        return value
    return str(json.loads(f'"{value}"'))


def metadata_label(pattern: re.Pattern[str], raw: object, default: str) -> str:
    if not isinstance(raw, str):
        return default
    match = pattern.search(raw)
    return decode_json_string_fragment(match.group(1)) if match else default


def ledger_records(path: Path, expected: int) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append((str(row["doc_id"]), str(row["raw_text_sha256"])))
    if len(records) != expected:
        raise ValueError(f"ledger row drift: {path}: {len(records)} != {expected}")
    return records


def classify_modern_input(job: Mapping[str, Any]) -> dict[str, Any]:
    """Produce category-code arrays aligned to retained ledger document indices."""

    import pyarrow.parquet as pq

    tasks = list(job["tasks"])
    output_root = Path(str(job["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    task_arrays: dict[int, dict[str, np.ndarray]] = {}
    lookup: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for task in tasks:
        task_index = int(task["task_index"])
        local = ledger_records(Path(str(task["ledger"])), int(task["documents"]))
        for document_index, (doc_id, raw_text_sha256) in enumerate(local):
            lookup[doc_id].append((raw_text_sha256, task_index, document_index))
        count = int(task["documents"])
        task_arrays[task_index] = {
            "source": np.zeros(count, dtype=np.uint64),
            "register": np.zeros(count, dtype=np.uint64),
            "host": np.zeros(count, dtype=np.uint64),
        }

    maps: dict[str, dict[int, str]] = {"source": {}, "register": {}, "host": {}}
    matched: set[tuple[int, int]] = set()
    duplicate_occurrences_resolved = 0
    duplicate_source_rows_not_retained = 0
    duplicate_metadata: dict[tuple[str, str], tuple[str, str, str]] = {}
    parquet = pq.ParquetFile(str(job["input_path"]))
    # A retained identity can be unique in the ledger while multiple raw source
    # rows reuse that identity with different text.  Discover those identities
    # before the metadata pass so text hashes are required exactly where needed.
    raw_identity_counts: dict[str, int] = defaultdict(int)
    for batch in parquet.iter_batches(
        columns=["source_dataset", "source_doc_id"], batch_size=8192, use_threads=False
    ):
        data = batch.to_pydict()
        for source_dataset, source_doc_id in zip(
            data["source_dataset"], data["source_doc_id"], strict=True
        ):
            doc_id = document_key(source_dataset, source_doc_id)
            if doc_id in lookup and raw_identity_counts[doc_id] < 2:
                raw_identity_counts[doc_id] += 1
    ambiguous_doc_ids = {
        doc_id
        for doc_id, values in lookup.items()
        if len(values) > 1 or raw_identity_counts.get(doc_id, 0) > 1
    }
    columns = ["source_dataset", "source_doc_id", "source_metadata_json"]
    if ambiguous_doc_ids:
        columns.append("text")
    for batch in parquet.iter_batches(columns=columns, batch_size=8192, use_threads=False):
        data = batch.to_pydict()
        text_values = data.get("text", [None] * batch.num_rows)
        for source_dataset, source_doc_id, metadata, text in zip(
            data["source_dataset"], data["source_doc_id"], data["source_metadata_json"], text_values, strict=True
        ):
            doc_id = document_key(source_dataset, source_doc_id)
            candidates = lookup.get(doc_id)
            if not candidates:
                continue
            source = str(source_dataset or "Unknown source")
            register = metadata_label(REGISTER_RE, metadata, "Unknown register")
            host = metadata_label(HOST_RE, metadata, "Unknown host")
            if doc_id not in ambiguous_doc_ids:
                if len(candidates) != 1:
                    raise ValueError(f"non-ambiguous identity has {len(candidates)} ledger candidates: {doc_id}")
                _, task_index, document_index = candidates[0]
            else:
                if not isinstance(text, str):
                    raise ValueError(f"duplicate document identity has no source text: {doc_id}")
                raw_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                metadata_key = (doc_id, raw_text_sha256)
                category_tuple = (source, register, host)
                previous_categories = duplicate_metadata.setdefault(metadata_key, category_tuple)
                if previous_categories != category_tuple:
                    raise ValueError(
                        "exact duplicate source rows disagree on category metadata: "
                        f"{job['input_path']}:{doc_id}/{raw_text_sha256}:"
                        f"{previous_categories!r}!={category_tuple!r}"
                    )
                matches = [
                    value
                    for value in candidates
                    if value[0] == raw_text_sha256 and (value[1], value[2]) not in matched
                ]
                if not matches:
                    # Raw rows can reuse a document identity for content that is
                    # absent from the retained training ledger.  Such rows were
                    # not trained on and are excluded from this classification.
                    duplicate_source_rows_not_retained += 1
                    continue
                # Exact duplicate records can share both identity and text.  The
                # retained ledger preserves source order, so consume the next
                # unmatched ledger occurrence deterministically.  We report the
                # count of these occurrence-level resolutions in the receipt.
                _, task_index, document_index = min(matches, key=lambda value: (value[1], value[2]))
                if len(matches) > 1:
                    duplicate_occurrences_resolved += 1
            selected_record = (task_index, document_index)
            if selected_record in matched:
                raise ValueError(f"retained record matched two parquet rows: {doc_id}/{selected_record}")
            matched.add(selected_record)
            for namespace, label in (("source", source), ("register", register), ("host", host)):
                code = stable_code(namespace, label)
                previous = maps[namespace].setdefault(code, label)
                if previous != label:
                    raise ValueError(f"64-bit category collision: {namespace}: {previous!r}/{label!r}")
                task_arrays[task_index][namespace][document_index] = code

    expected = sum(int(task["documents"]) for task in tasks)
    if len(matched) != expected:
        raise ValueError(
            f"retained/parquet join incomplete: {job['input_path']}: {len(matched)} != {expected}"
        )
    outputs = []
    for task in tasks:
        task_index = int(task["task_index"])
        row = {"task_index": task_index, "documents": int(task["documents"]), "arrays": {}}
        for namespace, array in task_arrays[task_index].items():
            if np.any(array == 0):
                raise ValueError(f"unclassified retained rows: task={task_index}/{namespace}")
            path = output_root / f"task_{task_index:05d}.{namespace}.u64"
            temporary = Path(str(path) + ".partial")
            array.tofile(temporary)
            os.replace(temporary, path)
            row["arrays"][namespace] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "rows": int(array.size),
            }
        outputs.append(row)
    return {
        "input_path": str(job["input_path"]),
        "rows": int(job["input_rows"]),
        "matched": len(matched),
        "duplicate_occurrences_resolved": duplicate_occurrences_resolved,
        "duplicate_source_rows_not_retained": duplicate_source_rows_not_retained,
        "outputs": outputs,
        "maps": {namespace: {str(code): label for code, label in values.items()} for namespace, values in maps.items()},
    }


def merge_category_maps(
    target: dict[str, dict[int, str]], incoming: Mapping[str, Mapping[str, str]]
) -> None:
    for namespace, values in incoming.items():
        for raw_code, label in values.items():
            code = int(raw_code)
            previous = target[namespace].setdefault(code, str(label))
            if previous != str(label):
                raise ValueError(f"category hash collision: {namespace}/{code}: {previous!r}/{label!r}")


def update_counter(target: dict[int, int], codes: np.ndarray, weights: np.ndarray | None = None) -> None:
    unique, inverse = np.unique(codes, return_inverse=True)
    values = np.bincount(inverse, weights=weights, minlength=unique.size)
    for code, value in zip(unique, values, strict=True):
        target[int(code)] += int(round(float(value)))


def update_indexed_counter(
    target: dict[int, int],
    unique: np.ndarray,
    inverse: np.ndarray,
    weights: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> None:
    if mask is None:
        local_inverse = inverse
        local_weights = weights
    else:
        local_inverse = inverse[mask]
        local_weights = weights[mask] if weights is not None else None
    values = np.bincount(local_inverse, weights=local_weights, minlength=unique.size)
    for code, value in zip(unique, values, strict=True):
        target[int(code)] += int(round(float(value)))


def doc_seen_tokens(starts: np.ndarray, ends: np.ndarray, seen: np.ndarray) -> np.ndarray:
    first = starts // SEQUENCE_LENGTH
    last = (ends - 1) // SEQUENCE_LENGTH
    prefix = np.concatenate(
        (np.zeros(1, dtype=np.uint64), np.cumsum(seen.astype(np.uint64), dtype=np.uint64))
    )
    result = (prefix[last + 1] - prefix[first]) * np.uint64(SEQUENCE_LENGTH)
    result -= (starts % SEQUENCE_LENGTH).astype(np.uint64) * seen[first].astype(np.uint64)
    right_tail = (SEQUENCE_LENGTH - (ends % SEQUENCE_LENGTH)) % SEQUENCE_LENGTH
    result -= right_tail.astype(np.uint64) * seen[last].astype(np.uint64)
    return result


def summarize_category(
    category_maps: Mapping[str, Mapping[int, str]],
    namespace: str,
    totals: Mapping[str, Mapping[int, int]],
    seen: Mapping[int, Mapping[str, Mapping[int, int]]],
) -> list[dict[str, Any]]:
    rows = []
    labels = category_maps[namespace]
    for code, total_tokens in totals["tokens"].items():
        if code not in labels:
            raise ValueError(f"missing category label: {namespace}/{code}")
        total_documents = int(totals["documents"][code])
        trajectory = []
        for iteration in CHECKPOINTS:
            values = seen[iteration]
            token_value = int(values["tokens"].get(code, 0))
            touched = int(values["touched"].get(code, 0))
            complete = int(values["complete"].get(code, 0))
            trajectory.append(
                {
                    "iteration": iteration,
                    "seen_tokens": token_value,
                    "seen_token_fraction": token_value / total_tokens,
                    "touched_documents": touched,
                    "touched_document_fraction": touched / total_documents,
                    "fully_seen_documents": complete,
                    "fully_seen_document_fraction": complete / total_documents,
                }
            )
        rows.append(
            {
                "code": code,
                "label": labels[code],
                "total_tokens": int(total_tokens),
                "total_documents": total_documents,
                "trajectory": trajectory,
            }
        )
    rows.sort(key=lambda row: (-row["total_tokens"], row["label"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    stage = args.stage_root.resolve()
    schedule_manifest_path = stage / "schedules" / "schedule_manifest.json"
    pool_receipt_path = stage / "inventory" / "pool_corpus_receipt.json"
    packed_receipt_path = stage / "inventory" / "packed_corpus_receipt.json"
    schedule_manifest = read_json(schedule_manifest_path)
    pool_receipt = read_json(pool_receipt_path)
    packed_receipt = read_json(packed_receipt_path)
    arm = next(row for row in schedule_manifest["arms"] if row["arm_id"] == "D0_mixed")
    if int(arm["optimizer_updates"]) != CHECKPOINTS[-1]:
        raise ValueError("checkpoint/schedule terminal iteration drift")
    ids_path = Path(str(arm["sequence_ids"]["path"]))
    if sha256_file(ids_path) != arm["sequence_ids"]["sha256"]:
        raise ValueError("D0 sequence schedule hash drift")
    ids = np.memmap(ids_path, mode="r", dtype=np.uint64)
    if ids.size != int(arm["training_slots"]):
        raise ValueError("D0 schedule slot drift")

    tasks: dict[int, dict[str, Any]] = {}
    modern_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    category_maps: dict[str, dict[int, str]] = {"source": {}, "register": {}, "host": {}}
    for task in pool_receipt["tasks"]:
        task_index = int(task["task_index"])
        manifest = read_json(Path(str(task["source_manifest"]["path"])))
        row = {
            "task_index": task_index,
            "pool": str(task["pool"]),
            "documents": int(task["documents"]),
            "source_name": str(manifest["source_name"]),
            "ledger": str(manifest["outputs"]["retained_ledger"]["path"]),
            "input_path": str(manifest["input"]["path"]),
            "input_rows": int(manifest["input"]["rows"]),
        }
        tasks[task_index] = row
        if row["pool"] in {"hplt_new_greek", "non_hplt_new_greek"}:
            modern_groups[row["input_path"]].append(row)
        else:
            label = row["source_name"]
            code = stable_code("source", label)
            previous = category_maps["source"].setdefault(code, label)
            if previous != label:
                raise ValueError("replay source category collision")
            row["source_code"] = code

    work_root = args.work_root.resolve()
    classifications_root = work_root / "modern_classifications"
    classifications_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "input_path": path,
            "input_rows": rows[0]["input_rows"],
            "tasks": rows,
            "output_root": str(classifications_root),
        }
        for path, rows in sorted(modern_groups.items())
    ]
    classifications = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(classify_modern_input, job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            classifications.append(result)
            merge_category_maps(category_maps, result["maps"])
            print(json.dumps({"phase": "classify", "completed": index, "total": len(futures), "input": result["input_path"]}), flush=True)

    classification_by_task: dict[int, dict[str, Path]] = {}
    for result in classifications:
        for row in result["outputs"]:
            task_index = int(row["task_index"])
            classification_by_task[task_index] = {
                namespace: Path(str(receipt["path"]))
                for namespace, receipt in row["arrays"].items()
            }
    modern_task_count = sum(
        1 for row in tasks.values() if row["pool"] in {"hplt_new_greek", "non_hplt_new_greek"}
    )
    if len(classification_by_task) != modern_task_count:
        raise ValueError("modern task classification coverage drift")

    real_positions = np.flatnonzero(ids != FILLER_ID).astype(np.uint64)
    real_ids = ids[real_positions]
    ranks: dict[tuple[int, int], np.ndarray] = {}
    for record in packed_receipt["packing_task_manifests"]:
        manifest = read_json(Path(str(record["manifest_path"])))
        ranks[(int(manifest["pool_code"]), int(manifest["bucket"]))] = np.full(
            int(manifest["sequence_count"]), np.iinfo(np.uint32).max, dtype=np.uint32
        )
    pool_codes = (real_ids >> np.uint64(62)).astype(np.uint8)
    buckets = ((real_ids >> np.uint64(55)) & np.uint64(0x7F)).astype(np.uint8)
    rows = (real_ids & np.uint64((1 << 55) - 1)).astype(np.uint64)
    group_keys = pool_codes.astype(np.uint16) * np.uint16(128) + buckets.astype(np.uint16)
    group_order = np.argsort(group_keys, kind="stable")
    sorted_keys = group_keys[group_order]
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    for left, right in zip(
        np.concatenate((np.zeros(1, dtype=np.int64), boundaries)),
        np.concatenate((boundaries, np.asarray([sorted_keys.size], dtype=np.int64))),
        strict=True,
    ):
            selected_positions = group_order[int(left):int(right)]
            key = int(sorted_keys[int(left)])
            pool_code, bucket = divmod(key, 128)
            target = ranks[(pool_code, bucket)]
            local_rows = rows[selected_positions]
            if int(local_rows.max()) >= target.size:
                raise ValueError("sequence row outside packed bucket")
            target[local_rows] = real_positions[selected_positions].astype(np.uint32)
    if any(np.any(value == np.iinfo(np.uint32).max) for value in ranks.values()):
        raise ValueError("packed sequence absent from D0 schedule")

    namespaces = ("source", "register")
    totals = {
        namespace: {"tokens": defaultdict(int), "documents": defaultdict(int)}
        for namespace in namespaces
    }
    seen = {
        namespace: {
            iteration: {
                "tokens": defaultdict(int),
                "touched": defaultdict(int),
                "complete": defaultdict(int),
            }
            for iteration in CHECKPOINTS
        }
        for namespace in namespaces
    }
    bucket_receipts = []
    for bucket_index, record in enumerate(packed_receipt["packing_task_manifests"], start=1):
        manifest_path = Path(str(record["manifest_path"]))
        manifest = read_json(manifest_path)
        pool_code = int(manifest["pool_code"])
        pool = POOL_NAMES[pool_code]
        catalog = np.memmap(
            Path(str(manifest["source_catalog"]["path"])), mode="r", dtype=CATALOG_DTYPE
        )[int(manifest["catalog_row_start"]):int(manifest["catalog_row_end"])]
        tokens = np.asarray(catalog["tokens"], dtype=np.uint64).copy()
        discarded = int(manifest["discarded_tail_tokens_in_last_selected_document"])
        if discarded:
            tokens[-1] -= discarded
        if int(tokens.sum(dtype=np.uint64)) != int(manifest["active_tokens"]):
            raise ValueError(f"bucket document token accounting drift: {manifest_path}")
        starts = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(tokens[:-1], dtype=np.uint64)))
        ends = starts + tokens
        task_indices = np.asarray(catalog["task_index"], dtype=np.uint32)
        document_indices = np.asarray(catalog["document_index"], dtype=np.uint32)
        source_codes = np.empty(catalog.size, dtype=np.uint64)
        register_codes = np.full(
            catalog.size, stable_code("register", "Not HPLT"), dtype=np.uint64
        )
        category_maps["register"].setdefault(stable_code("register", "Not HPLT"), "Not HPLT")
        for task_index in np.unique(task_indices):
            selected = task_indices == task_index
            task = tasks[int(task_index)]
            if task["pool"] in {"hplt_new_greek", "non_hplt_new_greek"}:
                arrays = classification_by_task[int(task_index)]
                source_labels = np.memmap(arrays["source"], mode="r", dtype=np.uint64)
                source_codes[selected] = source_labels[document_indices[selected]]
                if task["pool"] == "hplt_new_greek":
                    register_labels = np.memmap(arrays["register"], mode="r", dtype=np.uint64)
                    register_codes[selected] = register_labels[document_indices[selected]]
            else:
                source_codes[selected] = int(task["source_code"])

        category_arrays = {"source": source_codes, "register": register_codes}
        rank = ranks[(pool_code, int(manifest["bucket"]))]
        for namespace, codes in category_arrays.items():
            unique_codes, inverse_codes = np.unique(codes, return_inverse=True)
            token_weights = tokens.astype(np.float64)
            update_indexed_counter(
                totals[namespace]["tokens"], unique_codes, inverse_codes, token_weights
            )
            update_indexed_counter(
                totals[namespace]["documents"], unique_codes, inverse_codes
            )
            for iteration in CHECKPOINTS[1:]:
                is_seen = rank < iteration * GLOBAL_BATCH_SEQUENCES
                token_values = doc_seen_tokens(starts, ends, is_seen)
                first = starts // SEQUENCE_LENGTH
                last = (ends - 1) // SEQUENCE_LENGTH
                prefix = np.concatenate(
                    (np.zeros(1, dtype=np.uint64), np.cumsum(is_seen.astype(np.uint64), dtype=np.uint64))
                )
                sequence_counts = prefix[last + 1] - prefix[first]
                touched = sequence_counts > 0
                complete = sequence_counts == (last - first + 1)
                update_indexed_counter(
                    seen[namespace][iteration]["tokens"],
                    unique_codes,
                    inverse_codes,
                    token_values.astype(np.float64),
                )
                update_indexed_counter(
                    seen[namespace][iteration]["touched"],
                    unique_codes,
                    inverse_codes,
                    mask=touched,
                )
                update_indexed_counter(
                    seen[namespace][iteration]["complete"],
                    unique_codes,
                    inverse_codes,
                    mask=complete,
                )
        bucket_receipts.append(
            {
                "pool": pool,
                "bucket": int(manifest["bucket"]),
                "manifest": file_receipt(manifest_path),
            }
        )
        if bucket_index % 32 == 0:
            print(json.dumps({"phase": "aggregate", "completed": bucket_index, "total": len(packed_receipt["packing_task_manifests"])}), flush=True)

    source_rows = summarize_category(category_maps, "source", totals["source"], seen["source"])
    register_rows = summarize_category(category_maps, "register", totals["register"], seen["register"])
    source_token_total = sum(row["total_tokens"] for row in source_rows)
    if source_token_total != int(packed_receipt["global"]["active_tokens"]):
        raise ValueError(f"global source token accounting drift: {source_token_total}")
    checkpoint_summary = []
    for iteration in CHECKPOINTS:
        source_seen = sum(
            row["trajectory"][CHECKPOINTS.index(iteration)]["seen_tokens"] for row in source_rows
        )
        expected = int(
            np.asarray(
                np.memmap(Path(str(arm["active_tokens"]["path"])), mode="r", dtype=np.uint16)[
                    : iteration * GLOBAL_BATCH_SEQUENCES
                ],
                dtype=np.uint64,
            ).sum()
        )
        if source_seen != expected:
            raise ValueError(f"checkpoint source token accounting drift: {iteration}: {source_seen} != {expected}")
        checkpoint_summary.append(
            {
                "iteration": iteration,
                "schedule_slots": iteration * GLOBAL_BATCH_SEQUENCES,
                "seen_active_tokens": source_seen,
                "corpus_token_fraction": source_seen / source_token_total,
            }
        )

    payload = {
        "schema_version": "full8b_checkpoint_source_exposure_v1",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "seen_tokens": "exact loss-active source-document tokens inside sequences whose D0 schedule position precedes the checkpoint boundary",
            "touched_document": "at least one packed sequence intersecting the source document has been consumed",
            "fully_seen_document": "every packed sequence intersecting the source document has been consumed",
            "hplt_site_type": "HPLT source_metadata_json.register_level_1",
        },
        "checkpoint_summary": checkpoint_summary,
        "sources": source_rows,
        "hplt_register_level_1": [row for row in register_rows if row["label"] != "Not HPLT"],
        "bindings": {
            "stage_root": str(stage),
            "schedule_manifest": file_receipt(schedule_manifest_path),
            "pool_corpus_receipt": file_receipt(pool_receipt_path),
            "packed_corpus_receipt": file_receipt(packed_receipt_path),
            "D0_sequence_ids": file_receipt(ids_path),
            "bucket_manifests": bucket_receipts,
        },
        "classification": {
            "modern_input_files": len(classifications),
            "modern_retained_documents_joined": sum(int(row["matched"]) for row in classifications),
            "exact_duplicate_occurrences_resolved_in_source_order": sum(
                int(row["duplicate_occurrences_resolved"]) for row in classifications
            ),
            "exact_duplicate_source_rows_absent_from_retained_ledger": sum(
                int(row["duplicate_source_rows_not_retained"]) for row in classifications
            ),
            "workers": args.workers,
            "category_hash": "blake2b-64 namespace-NUL-label, collision checked",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "output": str(args.output), "sources": len(source_rows), "registers": len(payload["hplt_register_level_1"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
