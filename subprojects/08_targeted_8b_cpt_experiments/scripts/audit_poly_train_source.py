#!/usr/bin/env python3
"""Prove that the resolved existing poly_train artifact matches frozen split facts."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path

import pyarrow.parquet as pq

from contract_utils import file_binding, read_json, require, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable poly source audit exists: {args.output}")
    split = read_json(args.split_manifest)
    expected = split["outputs"]["poly_train"]
    parquet = pq.ParquetFile(args.input)
    require({"source_dataset", "text"}.issubset(parquet.schema_arrow.names), "poly source schema drift")
    rows = 0
    text_chars = 0
    utf8_bytes = 0
    sources: collections.Counter[str] = collections.Counter()
    for batch in parquet.iter_batches(columns=["source_dataset", "text"], batch_size=1024):
        values = batch.to_pydict()
        for source, text_value in zip(values["source_dataset"], values["text"], strict=True):
            text = "" if text_value is None else str(text_value)
            rows += 1
            text_chars += len(text)
            utf8_bytes += len(text.encode("utf-8"))
            sources[str(source)] += 1
    expected_sources = {str(row["source_dataset"]): int(row["len"]) for row in expected["source_counts"]}
    require(rows == int(expected["rows"]), "poly source row-count drift")
    require(text_chars == int(expected["text_chars"]), "poly source character-count drift")
    require(utf8_bytes == int(expected["utf8_bytes"]), "poly source UTF-8 byte-count drift")
    require(dict(sources) == expected_sources, "poly source composition drift")
    value = {
        "schema_version": "targeted_8b_poly_train_source_audit_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": file_binding(args.input),
        "split_manifest": file_binding(args.split_manifest),
        "rows": rows,
        "text_chars": text_chars,
        "utf8_bytes": utf8_bytes,
        "source_counts": dict(sorted(sources.items())),
        "checks": {
            "row_count_matches_frozen_split": True,
            "character_count_matches_frozen_split": True,
            "utf8_byte_count_matches_frozen_split": True,
            "source_composition_matches_frozen_split": True,
        },
    }
    write_json_atomic(args.output, value)
    print(json.dumps({"ok": True, "rows": rows, "sha256": value["input"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
