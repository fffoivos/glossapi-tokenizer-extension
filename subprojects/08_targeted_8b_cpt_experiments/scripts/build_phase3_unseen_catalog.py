#!/usr/bin/env python3
"""Build Phase-3 catalogs containing only documents unseen in Phases 1 and 2.

Inputs are immutable JSONL catalogs. Every row must carry
``document_key_sha256``, ``document_text_sha256`` and positive ``token_count``.
The realized main-trajectory ledger uses the same identities. A Phase-3
candidate is removed if either its natural-key hash or raw-text hash occurred
in either earlier phase.
Duplicate key/text identities inside Phase 3 fail the build rather than being
silently deduplicated. A separately frozen, exact exception manifest may
exclude a later byte-identical source row when an owner has explicitly
authorized that correction. The retained source-order row and the excluded
row must both match the manifest exactly; all other repetition still fails.

The output preserves source order and contains every eligible row. A capacity
receipt proves that one pass over each pool exceeds its frozen token target;
the downstream GPTDataset builder must therefore use a one-epoch/no-wrap
construction for the requested Phase-3 samples.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from contract_utils import executing_code_bundle, file_binding, require, sha256_file, write_json_atomic


POOLS = ("openarchives", "foreign_replay", "old_greek_replay")
POOL_STREAMS = {
    "openarchives": {"openarchives", "phase3_openarchives_candidates"},
    "foreign_replay": {"foreign"},
    "old_greek_replay": {"old_greek"},
}
SEQUENCE_LENGTH = 4096
# c92402e BlendedMegatronDatasetBuilder asks each component for its normalized
# share of 487,424 samples with its fixed 1.005 construction margin.  Capacity
# must cover that exact one-epoch request, not merely the un-margined 79/20/1
# token target; otherwise GPTDataset silently tiles a second epoch.
COMPONENT_REQUESTED_SAMPLES = {
    "openarchives": 386_991,
    "foreign_replay": 97_973,
    "old_greek_replay": 4_899,
}


def read_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_number}: row is not an object")
            yield line_number, value


def identity(row: dict[str, Any], location: str) -> tuple[str, str]:
    key = str(row.get("document_key_sha256", ""))
    text = str(row.get("document_text_sha256", ""))
    require(len(key) == 64 and len(text) == 64, f"{location}: missing SHA-256 identity")
    return key, text


def phase2_identities(path: Path) -> tuple[set[str], set[str], int]:
    keys: set[str] = set()
    texts: set[str] = set()
    rows = 0
    for line_number, row in read_rows(path):
        key, text = identity(row, f"{path}:{line_number}")
        keys.add(key)
        texts.add(text)
        rows += 1
    require(rows > 0, "Phase-2 realized ledger is empty")
    return keys, texts, rows


def catalog_authority(catalog: Path, receipt_path: Path, source_jsonl: Path, pool: str) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    schema = receipt.get("schema_version")
    require(schema in {"apertus_hard_h_to_g_tokenized_stream_v1", "apertus_hard_h_to_g_document_catalog_v1"}, f"{pool}: candidate catalog receipt schema drift")
    require(receipt.get("status") in {"passed", "frozen"}, f"{pool}: candidate catalog receipt did not pass")
    require(receipt.get("stream") in POOL_STREAMS[pool], f"{pool}: candidate catalog stream drift")
    source_binding = receipt.get("input") if schema == "apertus_hard_h_to_g_tokenized_stream_v1" else receipt.get("source")
    require(isinstance(source_binding, dict) and source_binding == file_binding(source_jsonl), f"{pool}: candidate source binding drift")
    expected_catalog = receipt.get("document_catalog")
    require(isinstance(expected_catalog, dict), f"{pool}: candidate catalog binding missing")
    observed = file_binding(catalog)
    require(
        observed["path"] == expected_catalog.get("path")
        and observed["bytes"] == int(expected_catalog.get("bytes", -1))
        and observed["sha256"] == expected_catalog.get("sha256"),
        f"{pool}: candidate catalog binding drift",
    )
    return receipt


def row_identity(row: dict[str, Any], location: str) -> tuple[str, str]:
    text = row.get("text")
    source_dataset = row.get("source_dataset") or row.get("source")
    source_doc_id = row.get("source_doc_id") or row.get("doc_id") or row.get("id")
    require(isinstance(text, str) and text, f"{location}: source row has no text")
    require(source_dataset not in (None, "") and source_doc_id not in (None, ""), f"{location}: source row has no identity")
    key = hashlib.sha256((str(source_dataset) + "\0" + str(source_doc_id)).encode("utf-8")).hexdigest()
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return key, text_hash


def load_duplicate_exceptions(path: Path | None) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any] | None]:
    """Load an exact, owner-authorized later-row exclusion manifest.

    This is deliberately not a general deduplication switch. Every exception
    names one later source index, the earlier retained source index, both
    document keys, and their shared raw-text hash. The selector verifies all
    of those fields when it encounters the duplicate and fails if an exception
    is unused or does not match its source catalog.
    """
    if path is None:
        return {}, None
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema_version") == "apertus_hard_h_to_g_phase3_duplicate_exceptions_v1", "duplicate-exception schema drift")
    require(value.get("status") == "frozen", "duplicate-exception manifest is not frozen")
    rows = value.get("exceptions")
    require(isinstance(rows, list) and rows, "duplicate-exception manifest is empty")
    by_later_row: dict[tuple[str, int], dict[str, Any]] = {}
    for index, entry in enumerate(rows):
        require(isinstance(entry, dict), f"duplicate exception {index}: not an object")
        pool = str(entry.get("pool", ""))
        later_row = entry.get("exclude_input_row_index")
        retained_row = entry.get("retain_input_row_index")
        require(pool in POOLS, f"duplicate exception {index}: pool drift")
        require(isinstance(later_row, int) and later_row >= 0, f"duplicate exception {index}: later source row invalid")
        require(isinstance(retained_row, int) and 0 <= retained_row < later_row, f"duplicate exception {index}: retained source row invalid")
        for field in ("document_text_sha256", "retain_document_key_sha256", "exclude_document_key_sha256"):
            require(isinstance(entry.get(field), str) and len(str(entry[field])) == 64, f"duplicate exception {index}: {field} invalid")
        key = (pool, later_row)
        require(key not in by_later_row, f"duplicate exception {index}: repeated later source row")
        by_later_row[key] = entry
    return by_later_row, value


def write_selected_jsonl_atomic(
    path: Path,
    source_jsonl: Path,
    selected_rows: list[dict[str, Any]],
) -> tuple[int, int]:
    require(not path.exists(), f"immutable output exists: {path}")
    by_index = {int(row["input_row_index"]): row for row in selected_rows}
    require(len(by_index) == len(selected_rows), f"{path}: repeated selected input row")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    count = 0
    tokens = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            with source_jsonl.open(encoding="utf-8") as source:
                input_row_index = 0
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    selected = by_index.get(input_row_index)
                    if selected is not None:
                        row = json.loads(line)
                        key, text_hash = row_identity(row, f"{source_jsonl}:{line_number}")
                        require(key == selected["document_key_sha256"], f"{source_jsonl}:{line_number}: selected key drift")
                        require(text_hash == selected["document_text_sha256"], f"{source_jsonl}:{line_number}: selected text drift")
                        handle.write(json.dumps({
                            **row,
                            "_phase3_document_key_sha256": key,
                            "_phase3_document_text_sha256": text_hash,
                            "_phase3_source_input_row_index": input_row_index,
                            "_phase3_token_count_including_eod": int(selected["token_count"]),
                            "_phase3_selection": "unseen_by_key_and_raw_text_sha256",
                        }, ensure_ascii=False, sort_keys=True) + "\n")
                        count += 1
                        tokens += int(selected["token_count"])
                    input_row_index += 1
            handle.flush()
            os.fsync(handle.fileno())
        require(count == len(selected_rows), f"{path}: selected rows were absent from source JSONL")
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count, tokens


def eligible_rows(
    path: Path,
    pool: str,
    phase2_keys: set[str],
    phase2_texts: set[str],
    phase3_keys: set[str],
    phase3_texts: set[str],
    counters: Counter[str],
    duplicate_exceptions: dict[tuple[str, int], dict[str, Any]],
    used_duplicate_exceptions: set[tuple[str, int]],
    phase3_text_origins: dict[str, tuple[int, str]],
) -> Iterable[dict[str, Any]]:
    for line_number, row in read_rows(path):
        location = f"{path}:{line_number}"
        require(row.get("pool") == pool, f"{location}: pool drift")
        key, text = identity(row, location)
        tokens = int(row.get("token_count", 0))
        require(tokens > 0, f"{location}: non-positive token_count")
        counters["candidate_rows"] += 1
        counters["candidate_tokens"] += tokens
        if key in phase2_keys or text in phase2_texts:
            counters["phase2_overlap_rows"] += 1
            counters["phase2_overlap_tokens"] += tokens
            continue
        input_row_index = row.get("input_row_index")
        require(isinstance(input_row_index, int) and input_row_index >= 0, f"{location}: input_row_index missing or invalid")
        require(key not in phase3_keys, f"{location}: repeated Phase-3 document key")
        if text in phase3_texts:
            exception_key = (pool, input_row_index)
            exception = duplicate_exceptions.get(exception_key)
            require(exception is not None, f"{location}: repeated Phase-3 document text")
            origin = phase3_text_origins.get(text)
            require(origin is not None, f"{location}: repeated Phase-3 text origin missing")
            require(text == exception["document_text_sha256"], f"{location}: duplicate-exception text drift")
            require(key == exception["exclude_document_key_sha256"], f"{location}: duplicate-exception excluded key drift")
            require(origin == (exception["retain_input_row_index"], exception["retain_document_key_sha256"]), f"{location}: duplicate-exception retained row drift")
            used_duplicate_exceptions.add(exception_key)
            counters["authorized_duplicate_exclusion_rows"] += 1
            counters["authorized_duplicate_exclusion_tokens"] += tokens
            continue
        phase3_keys.add(key)
        phase3_texts.add(text)
        phase3_text_origins[text] = (input_row_index, key)
        yield row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-realized", type=Path, required=True)
    parser.add_argument("--phase2-realized-receipt", type=Path, required=True)
    for pool in POOLS:
        parser.add_argument(f"--{pool.replace('_', '-')}-catalog", type=Path, required=True)
        parser.add_argument(f"--{pool.replace('_', '-')}-catalog-receipt", type=Path, required=True)
        parser.add_argument(f"--{pool.replace('_', '-')}-source-jsonl", type=Path, required=True)
        parser.add_argument(f"--{pool.replace('_', '-')}-target-tokens", type=int, required=True)
    parser.add_argument("--duplicate-exceptions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--packing-reserve-tokens", type=int, default=4096)
    args = parser.parse_args()
    require(not args.output_receipt.exists(), f"immutable receipt exists: {args.output_receipt}")
    realized_receipt = json.loads(args.phase2_realized_receipt.read_text(encoding="utf-8"))
    require(realized_receipt.get("schema_version") == "apertus_hard_h_to_g_realized_document_ledger_v1", "realized-document receipt schema drift")
    require(realized_receipt.get("status") == "passed", "realized-document receipt did not pass")
    require(realized_receipt.get("output") == {**file_binding(args.phase2_realized), "rows": realized_receipt.get("output", {}).get("rows")}, "realized-document ledger binding drift")
    phase2_keys, phase2_texts, phase2_rows = phase2_identities(args.phase2_realized)
    duplicate_exceptions, duplicate_exception_manifest = load_duplicate_exceptions(args.duplicate_exceptions)
    used_duplicate_exceptions: set[tuple[str, int]] = set()
    require(args.packing_reserve_tokens >= 4096, "packing reserve must cover at least one sequence")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    phase3_keys: set[str] = set()
    phase3_texts: set[str] = set()
    phase3_text_origins: dict[str, tuple[int, str]] = {}
    for pool in POOLS:
        source = getattr(args, f"{pool}_catalog")
        source_jsonl = getattr(args, f"{pool}_source_jsonl")
        catalog_receipt_path = getattr(args, f"{pool}_catalog_receipt")
        catalog_receipt = catalog_authority(source, catalog_receipt_path, source_jsonl, pool)
        target = getattr(args, f"{pool}_target_tokens")
        require(target > 0, f"{pool}: target must be positive")
        output = args.output_dir / f"phase3_{pool}.jsonl"
        counters: Counter[str] = Counter()
        selected_rows = list(eligible_rows(
                source, pool, phase2_keys, phase2_texts,
                phase3_keys, phase3_texts, counters, duplicate_exceptions,
                used_duplicate_exceptions, phase3_text_origins,
            ))
        rows, tokens = write_selected_jsonl_atomic(output, source_jsonl, selected_rows)
        require(rows > 0, f"{pool}: no unseen documents")
        component_samples = COMPONENT_REQUESTED_SAMPLES[pool]
        minimum_one_epoch_tokens = component_samples * SEQUENCE_LENGTH + 1
        required_capacity = max(target + args.packing_reserve_tokens, minimum_one_epoch_tokens)
        require(tokens >= required_capacity, f"{pool}: unseen capacity {tokens} below target plus reserve {required_capacity}")
        results[pool] = {
            "candidate_catalog": file_binding(source),
            "candidate_catalog_receipt": file_binding(catalog_receipt_path),
            "candidate_source": file_binding(source_jsonl),
            "candidate_stream": catalog_receipt["stream"],
            "output": {**file_binding(output), "rows": rows, "tokens": tokens},
            "target_tokens": target,
            "component_requested_samples_with_1p005_margin": component_samples,
            "minimum_one_epoch_tokens": minimum_one_epoch_tokens,
            "packing_reserve_tokens": args.packing_reserve_tokens,
            "required_capacity_tokens": required_capacity,
            "capacity_ratio": tokens / required_capacity,
            "accounting": dict(sorted(counters.items())),
        }
    require(used_duplicate_exceptions == set(duplicate_exceptions), "duplicate-exception manifest contains unused or unmatched exception")
    payload = {
        "schema_version": "apertus_phase3_unseen_catalog_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        },
        "executing_code_bundle": executing_code_bundle(),
        "phase2_realized": {
            **file_binding(args.phase2_realized),
            "rows": phase2_rows,
            "unique_keys": len(phase2_keys),
            "unique_texts": len(phase2_texts),
        },
        "phase2_realized_receipt": file_binding(args.phase2_realized_receipt),
        "duplicate_exceptions": (
            {**file_binding(args.duplicate_exceptions), "applied": len(used_duplicate_exceptions)}
            if args.duplicate_exceptions is not None else None
        ),
        "pools": results,
        "phase3_unique_documents": {
            "keys": len(phase3_keys),
            "texts": len(phase3_texts),
        },
        "policy": {
            "separate_phase3_blend": True,
            "cursor_at_update_3218": 0,
            "phase2_key_or_text_overlap_allowed": False,
            "within_phase3_key_or_text_repetition_allowed": False,
            "owner_authorized_exact_later_row_duplicate_exclusions": len(used_duplicate_exceptions),
            "gptdataset_epoch_wrap_allowed": False,
            "component_capacity_covers_c92402e_1p005_builder_margin": True,
        },
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
