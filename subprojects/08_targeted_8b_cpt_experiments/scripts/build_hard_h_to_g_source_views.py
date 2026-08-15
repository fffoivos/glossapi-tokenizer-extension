#!/usr/bin/env python3
"""Build the exact HPLT and OpenArchives source views for the R2 experiment.

The operation is only a row filter. It preserves input-shard order, row
multiplicity, every column and every stored value. It removes exactly:

1. published native-suite training exclusions at their frozen release
   coordinates; and
2. rows whose ``(source_dataset, source_doc_id)`` natural key is already in the
   historical Apertus-original Greek replay pool.

It performs no global or near deduplication.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic


EXPECTED_RELEASE_ROWS = 51_839_746
EXPECTED_HPLT_ROWS = 48_629_460
EXPECTED_OPENARCHIVES_ROWS = 126_597
HPLT_SOURCE = "HPLT/ell_Grek_ge8_no_mt_clean60"
OPENARCHIVES_PATTERN = re.compile(r"^openarchives\.gr")
EXPECTED_GREEK_REPLAY_SHA256 = "93608d67119c104f5036fde4b8193cf63a3b3edc12be382d9ef051b9e0af01ff"
EXPECTED_GREEK_REPLAY_ROWS = 2_223_742


def raw_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_replay_keys(path: Path, receipt_path: Path) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == "apertus_replay_parquet_inspection_v1", "replay inspection schema drift")
    require(receipt.get("status") == "passed", "replay inspection did not pass")
    binding = receipt.get("input", {})
    require(binding.get("sha256") == EXPECTED_GREEK_REPLAY_SHA256, "Greek replay receipt SHA-256 drift")
    require(int(receipt.get("parquet", {}).get("rows", -1)) == EXPECTED_GREEK_REPLAY_ROWS, "Greek replay row-count drift")
    require(path.stat().st_size == int(binding.get("bytes", -1)), "Greek replay byte-size drift")
    require(sha256_file(path) == EXPECTED_GREEK_REPLAY_SHA256, "Greek replay file SHA-256 drift")
    table = pq.read_table(path, columns=["source_dataset", "source_doc_id"])
    require(table.num_rows == EXPECTED_GREEK_REPLAY_ROWS, "Greek replay Parquet row-count drift")
    keys: set[tuple[str, str]] = set()
    null_rows = 0
    for batch in table.to_batches(max_chunksize=262_144):
        datasets = batch.column(0).to_pylist()
        doc_ids = batch.column(1).to_pylist()
        for source_dataset, source_doc_id in zip(datasets, doc_ids, strict=True):
            if source_dataset in (None, "") or source_doc_id in (None, ""):
                null_rows += 1
                continue
            keys.add((str(source_dataset), str(source_doc_id)))
    require(null_rows == 0, f"Greek replay has {null_rows} null/empty natural keys")
    return keys, {
        "rows": table.num_rows,
        "unique_natural_keys": len(keys),
        "duplicate_natural_key_rows": table.num_rows - len(keys),
        "sha256": EXPECTED_GREEK_REPLAY_SHA256,
        "inspection_receipt": file_binding(receipt_path),
    }


def load_native_exclusions(path: Path, receipt_path: Path) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == "apertus_native_suite_training_exclusions_v1", "native exclusion receipt schema drift")
    require(receipt.get("status") == "passed", "native exclusion receipt did not pass")
    expected = receipt.get("exclusion_manifest", {})
    require(path.stat().st_size == int(expected.get("bytes", -1)), "native exclusion manifest byte-size drift")
    require(sha256_file(path) == expected.get("sha256"), "native exclusion manifest SHA-256 drift")
    columns = [
        "dataset_shard", "dataset_row_index", "source_dataset", "source_doc_id",
        "document_text_sha256",
    ]
    table = pq.read_table(path, columns=columns)
    require(table.num_rows == int(expected.get("documents", -1)), "native exclusion document-count drift")
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for row in table.to_pylist():
        shard = str(row["dataset_shard"])
        index = int(row["dataset_row_index"])
        bucket = result.setdefault(shard, {})
        require(index not in bucket, f"duplicate native exclusion coordinate: {shard}:{index}")
        bucket[index] = row
    return result, {
        "documents": table.num_rows,
        "shards": len(result),
        "sha256": expected["sha256"],
        "receipt": file_binding(receipt_path),
    }


def load_validation_exclusions(path: Path, receipt_path: Path) -> tuple[set[str], dict[str, Any]]:
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == "apertus_hard_h_to_g_reused_validation_panels_v1", "validation exclusion receipt schema drift")
    require(receipt.get("status") == "passed", "validation exclusion receipt did not pass")
    expected = receipt.get("training_exclusions", {})
    require(path.stat().st_size == int(expected.get("bytes", -1)), "validation exclusion byte-size drift")
    require(sha256_file(path) == expected.get("sha256"), "validation exclusion SHA-256 drift")
    table = pq.read_table(path, columns=["document_text_sha256"])
    hashes = {str(value) for value in table["document_text_sha256"].to_pylist()}
    require(len(hashes) == table.num_rows == int(expected.get("rows", -1)), "validation exclusion row/count drift")
    return hashes, {
        "exact_text_hashes": len(hashes),
        "sha256": expected["sha256"],
        "receipt": file_binding(receipt_path),
    }


def pool_masks(source_values: list[Any]) -> dict[str, list[bool]]:
    return {
        "hplt": [value == HPLT_SOURCE for value in source_values],
        "openarchives": [isinstance(value, str) and OPENARCHIVES_PATTERN.match(value) is not None for value in source_values],
    }


def write_table_atomic(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        require(pq.read_metadata(temporary).num_rows == table.num_rows, f"output metadata row drift: {path}")
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def process_shard(
    row: dict[str, Any],
    release_root: Path,
    temporary_root: Path,
    replay_keys: set[tuple[str, str]],
    native_by_shard: dict[str, dict[int, dict[str, Any]]],
    validation_hashes: set[str],
) -> dict[str, Any]:
    relative = str(row["path"])
    input_path = release_root / "release" / relative
    require(input_path.is_file(), f"release shard missing: {input_path}")
    require(input_path.stat().st_size == int(row["bytes"]), f"release shard byte-size drift: {relative}")
    # This is the build that consumes every source byte, so verify the manifest
    # digest here instead of trusting only path/size/Parquet metadata.
    require(sha256_file(input_path) == row["sha256"], f"release shard SHA-256 drift: {relative}")
    table = pq.read_table(input_path)
    require(table.num_rows == int(row["rows"]), f"release shard row-count drift: {relative}")
    for name in ("text", "source_dataset", "source_doc_id"):
        require(name in table.column_names, f"release shard lacks {name}: {relative}")
    require("_source_release_shard" not in table.column_names and "_source_release_row_index" not in table.column_names, f"reserved lineage column collision: {relative}")
    table = table.append_column("_source_release_shard", pa.array([relative] * table.num_rows, type=pa.string()))
    table = table.append_column("_source_release_row_index", pa.array(range(table.num_rows), type=pa.int64()))
    source_values = table["source_dataset"].to_pylist()
    doc_ids = table["source_doc_id"].to_pylist()
    texts = table["text"].to_pylist()
    masks = pool_masks(source_values)

    native_rows = native_by_shard.get(relative, {})
    for index, expected in native_rows.items():
        require(0 <= index < table.num_rows, f"native exclusion coordinate out of range: {relative}:{index}")
        require(source_values[index] == expected["source_dataset"], f"native source_dataset drift: {relative}:{index}")
        require(doc_ids[index] == expected["source_doc_id"], f"native source_doc_id drift: {relative}:{index}")
        text = texts[index]
        require(isinstance(text, str), f"native exclusion text is not a string: {relative}:{index}")
        require(raw_text_sha256(text) == expected["document_text_sha256"], f"native raw-text hash drift: {relative}:{index}")

    outputs: dict[str, Any] = {}
    for pool, source_mask in masks.items():
        counters: Counter[str] = Counter()
        keep: list[bool] = []
        for index, in_pool in enumerate(source_mask):
            if not in_pool:
                keep.append(False)
                continue
            counters["source_rows"] += 1
            source_dataset = source_values[index]
            source_doc_id = doc_ids[index]
            require(source_doc_id not in (None, ""), f"{pool} has null/empty source_doc_id: {relative}:{index}")
            native = index in native_rows
            replay = (str(source_dataset), str(source_doc_id)) in replay_keys
            text = texts[index]
            require(isinstance(text, str), f"{pool} has non-string text: {relative}:{index}")
            validation = raw_text_sha256(text) in validation_hashes
            counters["native_exclusion_matches"] += int(native)
            counters["greek_replay_natural_key_matches"] += int(replay)
            counters["validation_exact_text_matches"] += int(validation)
            selected = not (native or replay or validation)
            keep.append(selected)
            counters["kept_rows"] += int(selected)
            counters["removed_rows"] += int(not selected)
            counters["multiple_exclusion_reasons"] += int(sum((native, replay, validation)) > 1)
        if counters["kept_rows"]:
            output_path = temporary_root / pool / relative
            selected_table = table.filter(pa.array(keep, type=pa.bool_()))
            require(selected_table.num_rows == counters["kept_rows"], f"{pool} mask row drift: {relative}")
            write_table_atomic(selected_table, output_path)
            outputs[pool] = {
                "relative_path": str(Path(pool) / relative),
                "input": {
                    "relative_path": relative,
                    "bytes": int(row["bytes"]),
                    "rows": int(row["rows"]),
                    "sha256": row["sha256"],
                },
                "output": file_binding(output_path),
                "counts": dict(counters),
            }
        elif counters["source_rows"]:
            outputs[pool] = {"relative_path": None, "counts": dict(counters)}
    return {"relative_path": relative, "outputs": outputs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-inspection-receipt", type=Path, required=True)
    parser.add_argument("--native-exclusions", type=Path, required=True)
    parser.add_argument("--native-exclusion-receipt", type=Path, required=True)
    parser.add_argument("--greek-replay-parquet", type=Path, required=True)
    parser.add_argument("--replay-inspection-receipt", type=Path, required=True)
    parser.add_argument("--validation-exclusions", type=Path, required=True)
    parser.add_argument("--validation-exclusion-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    require(not args.output_root.exists(), f"immutable output root exists: {args.output_root}")
    require(not args.output_receipt.exists(), f"immutable output receipt exists: {args.output_receipt}")
    require(1 <= args.workers <= 32, "workers must be in [1, 32]")

    manifest_path = args.release_root / "release/manifests/anonymization_manifest.json"
    token_counts_path = args.release_root / "release/manifests/token_counts.json"
    manifest = read_json(manifest_path)
    token_counts = read_json(token_counts_path)
    release_inspection = read_json(args.release_inspection_receipt)
    require(release_inspection.get("schema_version") == "targeted_8b_hf_release_inspection_v1", "release inspection schema drift")
    require(release_inspection.get("status") == "passed", "release inspection did not pass")
    require(release_inspection.get("hf_revision") == "987b8955fcd395c6219e39df9e64715457f69065", "release revision drift")
    require(Path(str(release_inspection.get("release_root", ""))).resolve() == args.release_root.resolve(), "release inspection root drift")
    require(release_inspection["bindings"]["anonymization_manifest"]["sha256"] == sha256_file(manifest_path), "release manifest/inspection drift")
    require(release_inspection["bindings"]["token_counts"]["sha256"] == sha256_file(token_counts_path), "release token-count/inspection drift")
    files = manifest.get("files")
    require(isinstance(files, list) and len(files) == 431, "release file inventory drift")
    require(sum(int(row["rows"]) for row in files) == EXPECTED_RELEASE_ROWS, "release row-count drift")
    require(int(token_counts["source_rows"][HPLT_SOURCE]) == EXPECTED_HPLT_ROWS, "HPLT source population drift")
    require(int(token_counts["source_rows"]["openarchives.gr"]) == EXPECTED_OPENARCHIVES_ROWS, "OpenArchives source population drift")

    replay_keys, replay_summary = load_replay_keys(args.greek_replay_parquet, args.replay_inspection_receipt)
    native_by_shard, native_summary = load_native_exclusions(args.native_exclusions, args.native_exclusion_receipt)
    validation_hashes, validation_summary = load_validation_exclusions(
        args.validation_exclusions, args.validation_exclusion_receipt
    )
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.", suffix=".partial", dir=args.output_root.parent))
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            shard_results = list(executor.map(
                lambda row: process_shard(
                    row, args.release_root, temporary_root, replay_keys, native_by_shard, validation_hashes
                ),
                files,
            ))
        totals: dict[str, Counter[str]] = {"hplt": Counter(), "openarchives": Counter()}
        output_files: dict[str, list[dict[str, Any]]] = {"hplt": [], "openarchives": []}
        for shard in shard_results:
            for pool, result in shard["outputs"].items():
                totals[pool].update(result["counts"])
                if result.get("relative_path") is not None:
                    binding = result["output"]
                    binding["path"] = str((args.output_root / result["relative_path"]).resolve())
                    output_files[pool].append({
                        "relative_path": result["relative_path"],
                        "input": result["input"],
                        "output": binding,
                        "counts": result["counts"],
                    })
        require(totals["hplt"]["source_rows"] == EXPECTED_HPLT_ROWS, "HPLT source-view population drift")
        require(totals["openarchives"]["source_rows"] == EXPECTED_OPENARCHIVES_ROWS, "OpenArchives source-view population drift")
        for pool in totals:
            counter = totals[pool]
            require(counter["source_rows"] == counter["kept_rows"] + counter["removed_rows"], f"{pool} row accounting drift")
            require(counter["kept_rows"] > 0, f"{pool} source view is empty")

        payload = {
            "schema_version": "apertus_hard_h_to_g_source_views_v1",
            "status": "passed",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "nodes": int(os.environ.get("SLURM_NNODES", "0")),
            },
            "executing_code_bundle": executing_code_bundle(),
            "release": {
                "root": str(args.release_root.resolve()),
                "anonymization_manifest": file_binding(manifest_path),
                "token_counts": file_binding(token_counts_path),
                "inspection_receipt": file_binding(args.release_inspection_receipt),
                "files": len(files),
                "rows": EXPECTED_RELEASE_ROWS,
                "all_parquet_sha256_verified_during_build": True,
            },
            "selection": {
                "hplt_source_dataset_exact": HPLT_SOURCE,
                "openarchives_source_dataset_regex": OPENARCHIVES_PATTERN.pattern,
            },
            "native_suite_exclusions": native_summary,
            "greek_replay_anti_join": replay_summary,
            "validation_exact_text_exclusions": validation_summary,
            "pools": {
                pool: {
                    "counts": dict(totals[pool]),
                    "output_files": output_files[pool],
                }
                for pool in ("hplt", "openarchives")
            },
            "invariants": {
                "row_order_preserved_within_input_shards": True,
                "row_multiplicity_preserved_except_named_exclusions": True,
                "all_stored_values_preserved": True,
                "additional_global_deduplication_performed": False,
                "near_deduplication_performed": False,
                "published_native_exclusions_applied_before_e001": True,
                "greek_replay_natural_key_anti_join_applied": True,
                "reused_validation_panel_exact_text_exclusions_applied": True,
                "exact_release_coordinate_columns_added": True,
            },
        }
        write_json_atomic(temporary_root / "source_views_manifest.json", payload)
        os.rename(temporary_root, args.output_root)
        final_receipt = read_json(args.output_root / "source_views_manifest.json")
        write_json_atomic(args.output_receipt, final_receipt)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
