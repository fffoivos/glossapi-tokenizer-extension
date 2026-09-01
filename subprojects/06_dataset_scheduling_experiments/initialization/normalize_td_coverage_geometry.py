#!/usr/bin/env python3
"""Replace decoded-surface TD decompositions with exact appended-merge leaves."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tokenizer_geometry import derive_added_token_base_ids


BASE_VOCAB_SIZE = 131_072
TARGET_VOCAB_SIZE = 148_992


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-jsonl", type=Path, required=True)
    parser.add_argument("--target-tokenizer", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_jsonl, args.output_manifest):
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output}")

    exact = derive_added_token_base_ids(
        args.target_tokenizer / "tokenizer.json",
        base_vocab_size=BASE_VOCAB_SIZE,
        target_vocab_size=TARGET_VOCAB_SIZE,
    )
    rows: dict[int, dict[str, object]] = {}
    changed_ids: list[int] = []
    with args.coverage_jsonl.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{args.coverage_jsonl}:{line_no}: {exc}") from exc
            token_id = row.get("new_token_id") if isinstance(row, dict) else None
            if not isinstance(token_id, int) or token_id not in exact:
                raise SystemExit(
                    f"{args.coverage_jsonl}:{line_no}: unexpected new_token_id {token_id!r}"
                )
            if token_id in rows:
                raise SystemExit(f"duplicate coverage token ID: {token_id}")
            expected = exact[token_id]
            observed = row.get("base_subtoken_ids")
            if observed != expected:
                changed_ids.append(token_id)
                row["decoded_surface_base_subtoken_ids"] = observed
            row["base_subtoken_ids"] = expected
            row["base_subtoken_len"] = len(expected)
            row["base_decomposition_policy"] = (
                "exact_dependency_ordered_appended_merge_dag_leaves"
            )
            rows[token_id] = row

    expected_ids = list(range(BASE_VOCAB_SIZE, TARGET_VOCAB_SIZE))
    if sorted(rows) != expected_ids:
        missing = sorted(set(expected_ids) - set(rows))
        raise SystemExit(f"coverage is not complete; missing IDs include {missing[:10]}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for token_id in expected_ids:
            handle.write(json.dumps(rows[token_id], ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "apertus_mini_td_coverage_geometry_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "input": str(args.coverage_jsonl.resolve()),
        "target_tokenizer": str(args.target_tokenizer.resolve()),
        "output": str(args.output_jsonl.resolve()),
        "row_count": len(rows),
        "changed_decomposition_count": len(changed_ids),
        "changed_token_ids": changed_ids,
        "policy": "exact_dependency_ordered_appended_merge_dag_leaves",
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
