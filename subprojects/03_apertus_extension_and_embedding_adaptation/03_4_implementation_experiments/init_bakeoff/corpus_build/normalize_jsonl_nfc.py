#!/usr/bin/env python3
"""NFC-normalize the text field of a JSONL corpus stream.

The parquet normalizer runs before mix construction, but the final JSONL stream
is what downstream preprocessors and TD coverage scans consume. This pass keeps
unchanged JSONL rows byte-for-byte identical and rewrites only rows whose text
field is not already NFC.
"""

import argparse
import gzip
import hashlib
import json
import os
import unicodedata as ud
from pathlib import Path
from typing import TextIO


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(str(path), mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input_jsonl.is_file():
        raise SystemExit(f"input JSONL not found: {args.input_jsonl}")
    if args.input_jsonl.resolve() == args.output_jsonl.resolve():
        raise SystemExit("refusing in-place JSONL normalization; write a new output path")
    if args.output_jsonl.exists() and not args.overwrite:
        raise SystemExit(f"output already exists, pass --overwrite to replace: {args.output_jsonl}")
    if args.sample_limit < 0:
        raise SystemExit("--sample-limit must be non-negative")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output_jsonl.with_name(f"{args.output_jsonl.name}.tmp.{os.getpid()}")
    tmp_manifest = args.manifest.with_name(f"{args.manifest.name}.tmp.{os.getpid()}")

    rows = 0
    changed_docs = 0
    blank_lines = 0
    missing_or_nonstring_text = 0
    changed_codepoints_before = 0
    changed_codepoints_after = 0
    examples = []
    output_sha256 = hashlib.sha256()

    try:
        with open_text(args.input_jsonl, "rt") as src, open_text(tmp_output, "wt") as dst:
            for line_no, raw_line in enumerate(src, 1):
                if not raw_line.strip():
                    blank_lines += 1
                    dst.write(raw_line)
                    output_sha256.update(raw_line.encode("utf-8"))
                    continue

                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{args.input_jsonl}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise SystemExit(f"{args.input_jsonl}:{line_no}: expected JSON object")

                rows += 1
                text = row.get(args.text_key)
                if not isinstance(text, str):
                    missing_or_nonstring_text += 1
                    out_line = raw_line if raw_line.endswith("\n") else raw_line + "\n"
                else:
                    normalized = ud.normalize("NFC", text)
                    if normalized == text:
                        out_line = raw_line if raw_line.endswith("\n") else raw_line + "\n"
                    else:
                        changed_docs += 1
                        changed_codepoints_before += len(text)
                        changed_codepoints_after += len(normalized)
                        row[args.text_key] = normalized
                        out_line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                        if len(examples) < args.sample_limit:
                            examples.append(
                                {
                                    "line_no": line_no,
                                    "doc_id": row.get("doc_id"),
                                    "source": row.get("source"),
                                    "lang": row.get("lang"),
                                    "before_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
                                    "after_sha1": hashlib.sha1(normalized.encode("utf-8")).hexdigest(),
                                    "before_len": len(text),
                                    "after_len": len(normalized),
                                }
                            )

                dst.write(out_line)
                output_sha256.update(out_line.encode("utf-8"))

        os.replace(str(tmp_output), str(args.output_jsonl))
        manifest = {
            "input_jsonl": str(args.input_jsonl),
            "output_jsonl": str(args.output_jsonl),
            "text_key": args.text_key,
            "rows": rows,
            "blank_lines": blank_lines,
            "missing_or_nonstring_text": missing_or_nonstring_text,
            "non_nfc_docs_before": changed_docs,
            "changed_docs": changed_docs,
            "changed_codepoints_before": changed_codepoints_before,
            "changed_codepoints_after": changed_codepoints_after,
            "input_size_bytes": args.input_jsonl.stat().st_size,
            "output_size_bytes": args.output_jsonl.stat().st_size,
            "output_sha256": output_sha256.hexdigest(),
            "examples": examples,
            "normalization": "unicodedata.normalize('NFC', text)",
        }
        tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(str(tmp_manifest), str(args.manifest))
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    finally:
        for path in (tmp_output, tmp_manifest):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
