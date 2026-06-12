#!/usr/bin/env python3
"""Split the finished replay stream into foreign replay and old-Greek replay.

The replay sweep built one decontaminated/anonymized `replay_only_final.jsonl`
whose internal source weights were the pilot 24/4/2/5 mix. For the peak-LR
sweep we want a different top-level blend:

  79% new Greek + 20% foreign replay + 1% old-Greek replay

Rather than re-running parquet mixing/decontamination, split the existing final
stream by the preserved source label. Everything except
`greek_replay_apertus_original` is foreign replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OLD_GREEK_SOURCE = "greek_replay_apertus_original"


def row_source(row: dict) -> str:
    source = row.get("source")
    if source:
        return str(source)
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("source"):
        return str(metadata["source"])
    return ""


def row_doc_id(row: dict, fallback: str) -> str:
    return str(row.get("doc_id") or row.get("id") or fallback)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--foreign-output", required=True, type=Path)
    ap.add_argument("--old-greek-output", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    args = ap.parse_args()

    args.foreign_output.parent.mkdir(parents=True, exist_ok=True)
    args.old_greek_output.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        "foreign_rows": 0,
        "foreign_chars": 0,
        "old_greek_rows": 0,
        "old_greek_chars": 0,
        "missing_source_rows": 0,
    }
    per_source: dict[str, dict[str, int]] = {}

    with args.input.open(encoding="utf-8") as fin, \
            args.foreign_output.open("w", encoding="utf-8") as f_foreign, \
            args.old_greek_output.open("w", encoding="utf-8") as f_old:
        for row_idx, line in enumerate(fin):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text:
                continue
            source = row_source(row)
            if not source:
                counts["missing_source_rows"] += 1
                source = "missing_source"
            doc_id = row_doc_id(row, f"replay_split_{row_idx}")
            out = {"text": text, "source": source, "doc_id": doc_id}
            bucket = per_source.setdefault(source, {"rows": 0, "chars": 0})
            bucket["rows"] += 1
            bucket["chars"] += len(text)

            if source == OLD_GREEK_SOURCE:
                f_old.write(json.dumps(out, ensure_ascii=False) + "\n")
                counts["old_greek_rows"] += 1
                counts["old_greek_chars"] += len(text)
            else:
                f_foreign.write(json.dumps(out, ensure_ascii=False) + "\n")
                counts["foreign_rows"] += 1
                counts["foreign_chars"] += len(text)

            if (row_idx + 1) % 1_000_000 == 0:
                print(
                    f"split {row_idx + 1:,} rows: "
                    f"foreign={counts['foreign_rows']:,} old_greek={counts['old_greek_rows']:,}",
                    flush=True,
                )

    manifest = {
        "input": str(args.input),
        "foreign_output": str(args.foreign_output),
        "old_greek_output": str(args.old_greek_output),
        "old_greek_source": OLD_GREEK_SOURCE,
        "counts": counts,
        "approx_tokens_by_chars_over_4": {
            "foreign": counts["foreign_chars"] / 4,
            "old_greek": counts["old_greek_chars"] / 4,
        },
        "per_source": per_source,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "manifest": str(args.manifest), **counts}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
