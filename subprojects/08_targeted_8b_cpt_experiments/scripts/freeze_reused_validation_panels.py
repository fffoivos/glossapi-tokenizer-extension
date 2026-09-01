#!/usr/bin/env python3
"""Freeze the corrected full-8B panels and their exact training exclusions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic


EXPECTED_MANIFEST_SHA256 = "a4b1d696adf83b2c691a99565075ba0a70db4074f7fe91fe6fdbab094303c1d9"
EXPECTED_PANELS = [
    "code", "de", "english", "greek_phd", "historical_polytonic", "hplt",
    "math", "neutral_external_modern_greek", "non_hplt", "old_greek",
    "openarchives", "ru", "zh",
]

STANDARD_ROW_SCHEMA = {
    "text_field": "text",
    "doc_id_field": "doc_id",
    "cluster_id_field": None,
    "source_dataset_field": "source_dataset",
}
PANEL_ROW_SCHEMAS = {
    "neutral_external_modern_greek": {
        "text_field": "text",
        "doc_id_field": "source_doc_id",
        "cluster_id_field": "cluster_id",
        "source_dataset_field": "source_id",
    },
}


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_panel_row(panel_name: str, row: dict[str, Any], line_number: int) -> dict[str, str]:
    """Extract identity through a panel-specific, fail-closed schema adapter."""
    schema = PANEL_ROW_SCHEMAS.get(panel_name, STANDARD_ROW_SCHEMA)
    text = row.get(schema["text_field"])
    doc_id = row.get(schema["doc_id_field"])
    source_dataset = row.get(schema["source_dataset_field"])
    cluster_field = schema["cluster_id_field"]
    cluster_id = row.get(cluster_field) if cluster_field else doc_id
    require(isinstance(text, str) and bool(text), f"{panel_name}:{line_number}: missing/empty {schema['text_field']}")
    require(isinstance(doc_id, str) and bool(doc_id), f"{panel_name}:{line_number}: missing/empty {schema['doc_id_field']}")
    require(isinstance(cluster_id, str) and bool(cluster_id), f"{panel_name}:{line_number}: missing/empty {cluster_field or schema['doc_id_field']}")
    require(isinstance(source_dataset, str) and bool(source_dataset), f"{panel_name}:{line_number}: missing/empty {schema['source_dataset_field']}")
    return {
        "text": text,
        "doc_id": doc_id,
        "cluster_id": cluster_id,
        "source_dataset": source_dataset,
    }


def write_parquet_atomic(table: pa.Table, path: Path) -> None:
    require(not path.exists(), f"immutable validation exclusion manifest exists: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        require(pq.read_metadata(temporary).num_rows == table.num_rows, "validation exclusion metadata row drift")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-exclusions", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_receipt.exists(), f"immutable validation receipt exists: {args.output_receipt}")
    require(sha256_file(args.validation_manifest) == EXPECTED_MANIFEST_SHA256, "validation manifest SHA-256 drift")
    manifest = read_json(args.validation_manifest)
    require(manifest.get("schema_version") == "apertus_full_8b_validation_manifest_v1", "validation manifest schema drift")
    require(manifest.get("status") == "frozen", "validation manifest is not frozen")
    require(manifest.get("all_panels_training_exact_content_disjoint") is True, "inherited panel overlap audit did not pass")
    panels = manifest.get("panels")
    require(isinstance(panels, list), "validation panel list missing")
    names = [str(panel["name"]) for panel in panels]
    require(names == EXPECTED_PANELS, f"validation panel inventory/order drift: {names}")

    by_hash: dict[str, dict[str, set[str]]] = defaultdict(lambda: {
        "panels": set(), "doc_ids": set(), "cluster_ids": set(), "source_datasets": set(),
    })
    panel_receipts: list[dict[str, Any]] = []
    total_rows = 0
    for panel in panels:
        name = str(panel["name"])
        raw = panel["raw_jsonl"]
        path = Path(raw["path"])
        binding = file_binding(path)
        require(binding["sha256"] == raw["sha256"] and binding["bytes"] == int(raw["bytes"]), f"{name}: raw panel binding drift")
        rows = 0
        unique_doc_ids: set[str] = set()
        clusters: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                extracted = extract_panel_row(name, row, line_number)
                text = extracted["text"]
                doc_id = extracted["doc_id"]
                digest = text_sha256(text)
                if row.get("text_sha256") is not None:
                    require(row["text_sha256"] == digest, f"{name}:{line_number}: stored text SHA-256 drift")
                cluster_id = extracted["cluster_id"]
                source_dataset = extracted["source_dataset"]
                bucket = by_hash[digest]
                bucket["panels"].add(name)
                bucket["doc_ids"].add(doc_id)
                bucket["cluster_ids"].add(cluster_id)
                if source_dataset:
                    bucket["source_datasets"].add(source_dataset)
                unique_doc_ids.add(doc_id)
                clusters.add(cluster_id)
                rows += 1
        require(rows == int(raw["rows"]) == int(panel["documents"]), f"{name}: validation row-count drift")
        require(len(unique_doc_ids) == rows, f"{name}: duplicate doc_id")
        panel_receipts.append({
            "name": name,
            "raw_jsonl": binding,
            "rows": rows,
            "clusters": len(clusters),
            "bootstrap_unit": panel["bootstrap_unit"],
            "cluster_field": panel.get("cluster_field"),
            "row_schema": PANEL_ROW_SCHEMAS.get(name, STANDARD_ROW_SCHEMA),
            "tokens": int(panel["tokens"]),
        })
        total_rows += rows

    exclusion_rows = [
        {
            "document_text_sha256": digest,
            "panels": sorted(bucket["panels"]),
            "doc_ids": sorted(bucket["doc_ids"]),
            "cluster_ids": sorted(bucket["cluster_ids"]),
            "source_datasets": sorted(bucket["source_datasets"]),
        }
        for digest, bucket in sorted(by_hash.items())
    ]
    write_parquet_atomic(pa.Table.from_pylist(exclusion_rows), args.output_exclusions)
    payload = {
        "schema_version": "apertus_hard_h_to_g_reused_validation_panels_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        },
        "executing_code_bundle": executing_code_bundle(),
        "parent_manifest": file_binding(args.validation_manifest),
        "panels": panel_receipts,
        "counts": {
            "panels": len(panel_receipts),
            "panel_rows": total_rows,
            "unique_exact_text_hashes": len(exclusion_rows),
            "cross_panel_duplicate_rows": total_rows - len(exclusion_rows),
        },
        "training_exclusions": {
            **file_binding(args.output_exclusions),
            "rows": len(exclusion_rows),
            "semantics": "sha256_of_exact_stored_utf8_text",
        },
        "bootstrap_scope": {
            "neutral_external_modern_greek": "explicit_cluster_id",
            "other_panels": "doc_id_proxy_no_explicit_cluster_map",
        },
        "policy": {
            "same_examples_for_both_model_scales": True,
            "reuse_corrected_full8_panels": True,
            "exclude_exact_panel_text_from_every_rebuilt_training_stream": True,
            "additional_deduplication": False,
        },
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
