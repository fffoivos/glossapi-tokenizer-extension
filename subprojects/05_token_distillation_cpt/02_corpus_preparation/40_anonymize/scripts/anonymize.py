#!/usr/bin/env python3
"""Anonymize a corpus with the Apertus-parity PII masker, via datatrove.

Format-AGNOSTIC by design: the input form (per-source parquets vs a unified
parquet vs jsonl), the text field, and the id field are all CLI args, because
the canonical-dataset form is a separate decision (see
plan/IMPLEMENTATION.md + the dataset-form note). Point `--input` at whatever
we settle on; nothing here assumes a layout.

Masking logic = pii_formatter.PIIFormatter (wraps pii_masker.mask):
email/IP at exact Apertus parity, IBAN with the Greek fix → <email-pii> /
<ip-pii> / <iban-pii>. Per-doc `pii_count` + `pii_by_type` are written to
metadata for the masked-rate report.

Run (one Clariden normal-partition node):
  python anonymize.py --input <dir> --reader parquet --text-key text \
      --id-key doc_key --output <dir> --tasks 64
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Input folder of shards.")
    p.add_argument("--reader", choices=["jsonl", "parquet"], required=True,
                   help="Input shard format. (Decided by the canonical-dataset form.)")
    p.add_argument("--output", required=True, help="Output folder for masked shards.")
    p.add_argument("--writer", choices=["jsonl", "parquet"], default=None,
                   help="Output format (default: same as --reader).")
    p.add_argument("--text-key", default="text", help="Document text field (default 'text').")
    p.add_argument("--id-key", default="doc_id", help="Document id field (default 'doc_id').")
    p.add_argument("--glob", default=None, help="Optional glob for shard files (e.g. '*.parquet').")
    p.add_argument("--tasks", type=int, default=64, help="Parallel datatrove tasks (CPU cores).")
    p.add_argument("--logging-dir", default=None, help="datatrove logging/stats dir (default <output>/_logs).")
    p.add_argument("--limit", type=int, default=-1, help="Per-task doc limit (smoke; -1 = all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import JsonlReader, ParquetReader
    from datatrove.pipeline.writers import JsonlWriter, ParquetWriter
    from pii_formatter import PIIFormatter

    reader_cls = JsonlReader if args.reader == "jsonl" else ParquetReader
    writer_kind = args.writer or args.reader
    writer_cls = JsonlWriter if writer_kind == "jsonl" else ParquetWriter

    reader_kwargs = dict(text_key=args.text_key, id_key=args.id_key)
    if args.glob:
        reader_kwargs["glob_pattern"] = args.glob
    if args.limit and args.limit > 0:
        reader_kwargs["limit"] = args.limit

    logging_dir = args.logging_dir or str(Path(args.output) / "_logs")

    pipeline = [
        reader_cls(args.input, **reader_kwargs),
        PIIFormatter(add_pii_count_to_metadata=True),
        writer_cls(args.output),
    ]
    print(f"[anonymize] input={args.input} reader={args.reader} text_key={args.text_key} "
          f"id_key={args.id_key} → output={args.output} ({writer_kind}); tasks={args.tasks}",
          flush=True)
    LocalPipelineExecutor(pipeline=pipeline, tasks=args.tasks, workers=args.tasks,
                          logging_dir=logging_dir).run()
    print("[anonymize] done. Per-doc pii_count / pii_by_type written to output metadata; "
          "aggregate the masked-rate report from there.", flush=True)


if __name__ == "__main__":
    main()
