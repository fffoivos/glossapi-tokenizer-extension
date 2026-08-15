#!/usr/bin/env python3
"""Normalize heterogeneous replay rows for mandatory benchmark scanning.

The frozen adapter config names every selected replay source and maps its text
and identity columns into one scanner schema. No source may opt out based on a
claim of language or source-level disjointness. The normalized JSONL is the
only accepted input to the native-suite plus GreekMMLU content scanner.
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

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


def mapped_value(row: dict[str, Any], mapping: dict[str, Any], name: str) -> Any:
    field = mapping.get(f"{name}_field")
    constant = mapping.get(f"{name}_constant")
    require((field is None) != (constant is None), f"{name}: require exactly one field or constant")
    return row.get(field) if field is not None else constant


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f"{path}:{line_number}: row is not an object")
            yield row


def parquet_rows(path: Path, columns: list[str]) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - CSCS runtime dependency
        raise RuntimeError("pyarrow is required for parquet replay inputs") from error
    parquet = pq.ParquetFile(path)
    require(set(columns).issubset(parquet.schema_arrow.names), f"{path}: adapter column missing")
    for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
        for row in batch.to_pylist():
            yield row


def source_rows(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    path = Path(source["path"])
    require(path.is_file(), f"replay source missing: {path}")
    if source["format"] == "jsonl":
        return jsonl_rows(path)
    require(source["format"] == "parquet", f"unsupported replay format: {source['format']}")
    mapping = source["mapping"]
    columns = sorted({
        value for key, value in mapping.items()
        if key.endswith("_field") and isinstance(value, str)
    })
    return parquet_rows(path, columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_jsonl.exists(), f"immutable output exists: {args.output_jsonl}")
    require(not args.output_receipt.exists(), f"immutable receipt exists: {args.output_receipt}")
    config = read_json(args.adapter_config)
    require(config.get("schema_version") == "apertus_replay_scan_adapter_v1", "adapter schema drift")
    require(config.get("source_level_disjointness_escape_allowed") is False, "replay scan escape enabled")
    sources = config.get("sources")
    require(isinstance(sources, list) and sources, "adapter has no replay sources")
    expected_names = config.get("expected_selected_source_names")
    actual_names = [str(source.get("name", "")) for source in sources]
    require(actual_names == expected_names and len(actual_names) == len(set(actual_names)), "selected replay-source inventory drift")
    expected_content_counts = config.get("expected_content_source_counts")
    require(isinstance(expected_content_counts, dict) and expected_content_counts, "content-source inventory is missing")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{args.output_jsonl.name}.", suffix=".partial", dir=args.output_jsonl.parent)
    temporary = Path(name)
    counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    bindings: dict[str, Any] = {}
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for source in sources:
                source_name = str(source["name"])
                mapping = source["mapping"]
                source_path = Path(source["path"])
                binding = file_binding(source_path)
                require(binding["sha256"] == source["expected_sha256"], f"{source_name}: source SHA-256 drift")
                bindings[source_name] = binding
                source_count = 0
                for row_number, row in enumerate(source_rows(source)):
                    text = mapped_value(row, mapping, "text")
                    source_dataset = mapped_value(row, mapping, "source_dataset")
                    source_doc_id = mapped_value(row, mapping, "source_doc_id")
                    require(isinstance(text, str) and text, f"{source_name}:{row_number}: empty text")
                    require(source_dataset not in (None, ""), f"{source_name}:{row_number}: missing source_dataset")
                    require(source_doc_id not in (None, ""), f"{source_name}:{row_number}: missing source_doc_id")
                    canonical_id = hashlib.sha256(
                        str(source_dataset).encode("utf-8") + b"\0" + str(source_doc_id).encode("utf-8")
                    ).hexdigest()
                    normalized = {
                        "doc_id": canonical_id,
                        "adapter_source": source_name,
                        "adapter_row_index": row_number,
                        "source_dataset": str(source_dataset),
                        "source_doc_id": str(source_doc_id),
                        "document_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "text": text,
                    }
                    output.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
                    source_count += 1
                    content_counts[str(source_dataset)] += 1
                require(source_count == int(source["expected_rows"]), f"{source_name}: row-count drift")
                counts[source_name] = source_count
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, args.output_jsonl)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    require(sum(counts.values()) > 0, "normalized replay scan input is empty")
    require(dict(sorted(content_counts.items())) == expected_content_counts, "replay content-source counts drift")
    payload = {
        "schema_version": "apertus_replay_benchmark_scan_input_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        },
        "executing_code_bundle": executing_code_bundle(),
        "adapter_config": file_binding(args.adapter_config),
        "source_bindings": bindings,
        "rows_by_source": dict(counts),
        "rows_by_content_source": dict(sorted(content_counts.items())),
        "output": {**file_binding(args.output_jsonl), "rows": sum(counts.values())},
        "required_scans": ["native_greek_suite_v1", "greekmmlu_regenerated_v1"],
        "source_level_disjointness_escape_allowed": False,
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
