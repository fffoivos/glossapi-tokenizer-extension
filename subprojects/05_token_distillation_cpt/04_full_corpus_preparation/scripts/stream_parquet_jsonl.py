#!/usr/bin/env python3
"""Stream canonical Parquet rows as `{id,text}` JSONL for the Rust detector."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Sequence


def expand_inputs(values: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(path.rglob("*.parquet"))
        elif any(char in value for char in "*?["):
            paths.extend(Path(item) for item in glob.glob(value, recursive=True))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(value)
    unique = sorted({path.resolve() for path in paths if path.suffix == ".parquet"})
    if not unique:
        raise ValueError("no Parquet inputs resolved")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="source_doc_id")
    parser.add_argument("--source-column", default="source_dataset")
    parser.add_argument("--source-regex")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=250_000)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised on Clariden
        raise RuntimeError("install pyarrow in the Clariden environment") from exc

    source_pattern = re.compile(args.source_regex) if args.source_regex else None
    inputs = expand_inputs(args.input)
    input_inventory = [{"path": str(path), "bytes": path.stat().st_size} for path in inputs]
    emitted = scanned = 0
    seen_ids: set[str] = set()
    id_sequence = hashlib.sha256()
    output = sys.stdout
    for path in inputs:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        required = {args.text_column, args.id_column}
        if source_pattern:
            required.add(args.source_column)
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; has {sorted(names)}")
        columns = [args.id_column, args.text_column]
        if source_pattern:
            columns.append(args.source_column)
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
            values = batch.to_pydict()
            for index, raw_id in enumerate(values[args.id_column]):
                scanned += 1
                if source_pattern and not source_pattern.search(str(values[args.source_column][index] or "")):
                    continue
                if raw_id is None or not str(raw_id):
                    raise ValueError(f"{path}: null/empty {args.id_column} at scanned row {scanned}")
                doc_id = str(raw_id)
                if doc_id in seen_ids:
                    raise ValueError(f"duplicate document id after routing: {doc_id!r}")
                seen_ids.add(doc_id)
                encoded_id = doc_id.encode("utf-8")
                id_sequence.update(len(encoded_id).to_bytes(8, "big"))
                id_sequence.update(encoded_id)
                output.write(
                    json.dumps(
                        {"id": doc_id, "text": str(values[args.text_column][index] or "")},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                emitted += 1
                if args.max_rows and emitted >= args.max_rows:
                    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
                    args.manifest_out.write_text(
                        json.dumps(
                            {
                                "schema_version": "detector_input_stream_v1",
                                "source": args.source_name,
                                "source_regex": args.source_regex,
                                "text_column": args.text_column,
                                "id_column": args.id_column,
                                "source_column": args.source_column,
                                "inputs": input_inventory,
                                "rows_scanned": scanned,
                                "rows_emitted": emitted,
                                "unique_document_ids": len(seen_ids),
                                "id_sequence_sha256": id_sequence.hexdigest(),
                                "bounded_smoke": True,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    print(f"stream_parquet_jsonl: scanned={scanned:,} emitted={emitted:,}", file=sys.stderr)
                    return 0
            if args.progress_every and scanned % args.progress_every < batch.num_rows:
                print(
                    f"stream_parquet_jsonl: scanned={scanned:,} emitted={emitted:,} input={path}",
                    file=sys.stderr,
                    flush=True,
                )
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(
            {
                "schema_version": "detector_input_stream_v1",
                "source": args.source_name,
                "source_regex": args.source_regex,
                "text_column": args.text_column,
                "id_column": args.id_column,
                "source_column": args.source_column,
                "inputs": input_inventory,
                "rows_scanned": scanned,
                "rows_emitted": emitted,
                "unique_document_ids": len(seen_ids),
                "id_sequence_sha256": id_sequence.hexdigest(),
                "bounded_smoke": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"stream_parquet_jsonl: scanned={scanned:,} emitted={emitted:,}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
