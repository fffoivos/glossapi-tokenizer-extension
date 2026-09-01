#!/usr/bin/env python3
"""Build disjoint ancient/modern streams for new-output-row calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_eval_hashes(path: Path) -> set[str]:
    return {
        text_hash(json.loads(line)["text"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def take_rows(
    path: Path,
    *,
    skip_valid: int,
    limit: int,
    register: str,
    max_chars: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["text"], batch_size=128):
        for row in batch.to_pylist():
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            if seen < skip_valid:
                seen += 1
                continue
            normalized = unicodedata.normalize("NFC", text)[:max_chars]
            if normalized.strip():
                rows.append(
                    {
                        "text": normalized,
                        "register": register,
                        "source": f"FineWeb2/calibration/{register}",
                        "doc_id": f"{register}:{seen}",
                    }
                )
            seen += 1
            if len(rows) >= limit:
                return rows
    raise SystemExit(
        f"{path}: only found {len(rows)} rows after skip={skip_valid}, need {limit}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ancient-train-parquet", type=Path, required=True)
    parser.add_argument("--modern-test-parquet", type=Path, required=True)
    parser.add_argument("--ancient-eval-jsonl", type=Path, required=True)
    parser.add_argument("--modern-eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--docs-per-register", type=int, default=1000)
    parser.add_argument("--modern-skip-valid-docs", type=int, default=1000)
    parser.add_argument("--max-chars-per-doc", type=int, default=3000)
    args = parser.parse_args()

    ancient = take_rows(
        args.ancient_train_parquet,
        skip_valid=0,
        limit=args.docs_per_register,
        register="ancient_polytonic",
        max_chars=args.max_chars_per_doc,
    )
    modern = take_rows(
        args.modern_test_parquet,
        skip_valid=args.modern_skip_valid_docs,
        limit=args.docs_per_register,
        register="modern_greek",
        max_chars=args.max_chars_per_doc,
    )
    eval_hashes = read_eval_hashes(args.ancient_eval_jsonl) | read_eval_hashes(
        args.modern_eval_jsonl
    )
    calibration_hashes = {text_hash(row["text"]) for row in ancient + modern}
    overlaps = sorted(eval_hashes & calibration_hashes)
    if overlaps:
        raise SystemExit(
            f"calibration/eval text overlap detected: {len(overlaps)} hashes"
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for pair in zip(ancient, modern):
            for row in pair:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(args.output_jsonl.read_bytes()).hexdigest()
    report = {
        "schema_version": "polytonic-output-calibration-data-v1",
        "output": str(args.output_jsonl),
        "sha256": digest,
        "bytes": args.output_jsonl.stat().st_size,
        "docs": len(ancient) + len(modern),
        "docs_per_register": {
            "ancient_polytonic": len(ancient),
            "modern_greek": len(modern),
        },
        "max_chars_per_doc": args.max_chars_per_doc,
        "modern_skip_valid_docs": args.modern_skip_valid_docs,
        "eval_text_overlap_count": 0,
        "normalization": "NFC",
    }
    args.manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
