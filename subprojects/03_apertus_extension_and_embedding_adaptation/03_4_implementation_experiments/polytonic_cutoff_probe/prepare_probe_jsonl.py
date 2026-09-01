#!/usr/bin/env python3
"""Materialize small, fixed FineWeb-2 probe streams from local parquets."""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq


def export_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    source: str,
    register: str,
    max_docs: int | None,
    max_chars_per_doc: int | None,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(input_path)
    available = set(parquet.schema.names)
    columns = ["text"]
    for optional in ("id", "url", "language", "language_script"):
        if optional in available:
            columns.append(optional)

    docs = 0
    chars = 0
    utf8_bytes = 0
    non_nfc_inputs = 0
    with output_path.open("w", encoding="utf-8") as output:
        for batch in parquet.iter_batches(columns=columns, batch_size=128):
            for row in batch.to_pylist():
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                normalized = unicodedata.normalize("NFC", text)
                if normalized != text:
                    non_nfc_inputs += 1
                if max_chars_per_doc is not None:
                    normalized = normalized[:max_chars_per_doc]
                if not normalized.strip():
                    continue
                payload = {
                    "text": normalized,
                    "source": source,
                    "register": register,
                    "doc_id": row.get("id") or f"{source}:{docs}",
                    "url": row.get("url"),
                    "lang": row.get("language"),
                    "script": row.get("language_script"),
                }
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")
                docs += 1
                chars += len(normalized)
                utf8_bytes += len(normalized.encode("utf-8"))
                if max_docs is not None and docs >= max_docs:
                    break
            if max_docs is not None and docs >= max_docs:
                break

    return {
        "input": str(input_path),
        "output": str(output_path),
        "source": source,
        "register": register,
        "docs": docs,
        "chars": chars,
        "utf8_bytes": utf8_bytes,
        "non_nfc_inputs_normalized": non_nfc_inputs,
        "max_chars_per_doc": max_chars_per_doc,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ancient-train-parquet", type=Path, required=True)
    parser.add_argument("--ancient-test-parquet", type=Path, required=True)
    parser.add_argument("--modern-test-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-docs", type=int, default=None)
    parser.add_argument("--max-ancient-test-docs", type=int, default=1000)
    parser.add_argument("--max-modern-test-docs", type=int, default=1000)
    parser.add_argument(
        "--max-eval-chars-per-doc",
        type=int,
        default=3000,
        help="fixed Unicode-character prefix for both tokenizers; prevents candidate-specific truncation",
    )
    args = parser.parse_args()

    outputs = [
        export_jsonl(
            args.ancient_train_parquet,
            args.output_dir / "ancient_train.jsonl",
            source="FineWeb2/grc_Grek/train",
            register="ancient_polytonic",
            max_docs=args.max_train_docs,
            max_chars_per_doc=None,
        ),
        export_jsonl(
            args.ancient_test_parquet,
            args.output_dir / "ancient_test.jsonl",
            source="FineWeb2/grc_Grek/test",
            register="ancient_polytonic",
            max_docs=args.max_ancient_test_docs,
            max_chars_per_doc=args.max_eval_chars_per_doc,
        ),
        export_jsonl(
            args.modern_test_parquet,
            args.output_dir / "modern_test.jsonl",
            source="FineWeb2/ell_Grek/test",
            register="modern_greek",
            max_docs=args.max_modern_test_docs,
            max_chars_per_doc=args.max_eval_chars_per_doc,
        ),
    ]
    manifest = {
        "schema_version": "polytonic-cutoff-probe-data-v1",
        "normalization": "NFC",
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
