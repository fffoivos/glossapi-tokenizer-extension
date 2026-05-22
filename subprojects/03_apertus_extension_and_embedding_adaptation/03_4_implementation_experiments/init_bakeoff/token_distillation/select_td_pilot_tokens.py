#!/usr/bin/env python3
"""Select bounded Token Distillation smoke and layer-pilot token sets.

Input is the CPU coverage prepass JSONL. The selector is intentionally simple:
eligible tokens must have real emitted-token snippets, then we rank by observed
firings and document coverage so the smoke/pilot starts with the most stable
examples rather than the long tail.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


STATUS_RANK = {
    "enough_100": 4,
    "enough_25": 3,
    "low_20_24": 2,
    "low_lt20": 1,
    "zero": 0,
    "mismatch": -1,
}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: expected JSON object")
            yield row


def as_int(row: Dict[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def min_rank(status: str) -> int:
    if status not in STATUS_RANK:
        raise SystemExit(f"unknown --min-status {status!r}; choices: {sorted(STATUS_RANK)}")
    return STATUS_RANK[status]


def compact(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "new_token_id": row.get("new_token_id"),
        "raw_token": row.get("raw_token"),
        "token_string": row.get("token_string"),
        "base_subtoken_ids": row.get("base_subtoken_ids"),
        "base_subtoken_len": row.get("base_subtoken_len"),
        "extended_firings": row.get("extended_firings"),
        "docs_with_firing": row.get("docs_with_firing"),
        "usable_snippets_25": row.get("usable_snippets_25"),
        "usable_snippets_100": row.get("usable_snippets_100"),
        "status": row.get("status"),
        "recommended_action": row.get("recommended_action"),
        "example_snippet_refs": row.get("example_snippet_refs", []),
    }


def write_id_list(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(f"{row['new_token_id']}\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-size", type=int, default=512)
    parser.add_argument("--layer-pilot-size", type=int, default=2048)
    parser.add_argument("--min-status", default="enough_25", choices=sorted(STATUS_RANK))
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="permit selection even if coverage rows contain status=mismatch",
    )
    args = parser.parse_args()

    if args.smoke_size <= 0 or args.layer_pilot_size <= 0:
        raise SystemExit("pilot sizes must be positive")
    if args.smoke_size > args.layer_pilot_size:
        raise SystemExit("--smoke-size must be <= --layer-pilot-size")

    rows = list(iter_jsonl(args.coverage_jsonl))
    if not rows:
        raise SystemExit(f"empty coverage JSONL: {args.coverage_jsonl}")

    mismatches = [row for row in rows if row.get("status") == "mismatch"]
    strict_no_mismatch = not args.allow_mismatch
    if strict_no_mismatch and mismatches:
        raise SystemExit(f"refusing selection: {len(mismatches)} coverage rows have status=mismatch")

    threshold = min_rank(args.min_status)
    eligible = [
        row
        for row in rows
        if STATUS_RANK.get(str(row.get("status")), -99) >= threshold
    ]
    eligible.sort(
        key=lambda row: (
            STATUS_RANK.get(str(row.get("status")), -99),
            as_int(row, "extended_firings"),
            as_int(row, "docs_with_firing"),
            -as_int(row, "base_subtoken_len"),
            -as_int(row, "new_token_id"),
        ),
        reverse=True,
    )

    smoke = eligible[: args.smoke_size]
    layer_pilot = eligible[: args.layer_pilot_size]

    summary = None
    if args.summary_json and args.summary_json.exists():
        summary = json.loads(args.summary_json.read_text(encoding="utf-8"))

    out = {
        "coverage_jsonl": str(args.coverage_jsonl),
        "summary_json": str(args.summary_json) if args.summary_json else None,
        "prepass_recommended_next_step": (summary or {}).get("recommended_next_step"),
        "min_status": args.min_status,
        "strict_no_mismatch": strict_no_mismatch,
        "total_rows": len(rows),
        "eligible_count": len(eligible),
        "mismatch_count": len(mismatches),
        "smoke_size_requested": args.smoke_size,
        "smoke_size_selected": len(smoke),
        "layer_pilot_size_requested": args.layer_pilot_size,
        "layer_pilot_size_selected": len(layer_pilot),
        "status_counts": (summary or {}).get("status_counts"),
        "top_20_eligible": [compact(row) for row in eligible[:20]],
        "smoke_tokens": [compact(row) for row in smoke],
        "layer_pilot_tokens": [compact(row) for row in layer_pilot],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "td_pilot_token_selection.json"
    smoke_ids_path = args.output_dir / "smoke_token_ids.txt"
    layer_ids_path = args.output_dir / "layer_pilot_token_ids.txt"
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_id_list(smoke_ids_path, smoke)
    write_id_list(layer_ids_path, layer_pilot)

    print(json.dumps({
        "selection_json": str(report_path),
        "smoke_token_ids": str(smoke_ids_path),
        "layer_pilot_token_ids": str(layer_ids_path),
        "eligible_count": len(eligible),
        "smoke_size_selected": len(smoke),
        "layer_pilot_size_selected": len(layer_pilot),
        "prepass_recommended_next_step": out["prepass_recommended_next_step"],
    }, ensure_ascii=False, indent=2))

    if len(smoke) < args.smoke_size:
        return 2
    if len(layer_pilot) < args.layer_pilot_size:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
