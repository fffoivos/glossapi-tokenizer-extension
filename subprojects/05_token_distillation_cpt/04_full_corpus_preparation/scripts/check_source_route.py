#!/usr/bin/env python3
"""Fail closed unless an academic structural audit matches the tracked route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=here / "configs" / "sources.json")
    parser.add_argument("--source", required=True)
    parser.add_argument("--input-scope", choices=("single_source", "canonical_mixed"), required=True)
    parser.add_argument("--source-regex", default="")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--source-column", default="source_dataset")
    parser.add_argument("--allow-shadow", action="store_true")
    parser.add_argument("--segmentation-complete", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.sources.read_text(encoding="utf-8"))
    routes = {row["source_id"]: row for row in config.get("embedded_structural_routes", [])}
    for source in config["sources"]:
        routes.setdefault(
            source["source_id"],
            {
                **source,
                "input_scope": "single_source",
                "source_regex": "",
            },
        )
    route = routes.get(args.source)
    if route is None:
        raise SystemExit(f"ERROR: {args.source!r} has no tracked structural route")
    if args.input_scope != route["input_scope"]:
        raise SystemExit(
            f"ERROR: {args.source}: input_scope={args.input_scope!r}, expected {route['input_scope']!r}"
        )
    expected_regex = route.get("source_regex", "")
    if args.source_regex != expected_regex:
        raise SystemExit(
            f"ERROR: {args.source}: source_regex={args.source_regex!r}, expected exact tracked {expected_regex!r}"
        )
    if args.text_column not in route.get("text_columns", []):
        raise SystemExit(
            f"ERROR: {args.source}: text column {args.text_column!r} not in tracked {route.get('text_columns', [])}"
        )
    if args.id_column not in route.get("id_columns", []):
        raise SystemExit(
            f"ERROR: {args.source}: id column {args.id_column!r} not in tracked {route.get('id_columns', [])}"
        )
    if route["input_scope"] == "canonical_mixed" and args.source_column != route.get("source_column"):
        raise SystemExit(
            f"ERROR: {args.source}: source column {args.source_column!r}, expected {route.get('source_column')!r}"
        )

    policy = route["structural_policy"]
    if policy == "disabled":
        raise SystemExit(f"ERROR: academic structural cleaning is disabled for {args.source}")
    if policy in {"shadow", "shadow_after_segmentation"} and not args.allow_shadow:
        raise SystemExit(f"ERROR: {args.source} is shadow-only; set ALLOW_SHADOW=1 for an audit")
    if policy == "shadow_after_segmentation" and not args.segmentation_complete:
        raise SystemExit(f"ERROR: {args.source} requires segmentation before structural shadow audit")
    if route.get("cleaning_profile") == "academic_sectioned":
        raise SystemExit(
            f"ERROR: {args.source} is raw section-level data; normalize/group it before the whole-document audit"
        )
    if route["input_scope"] == "single_source" and not any(
        str(pattern).lower().endswith(".parquet") for pattern in route.get("include_globs", [])
    ):
        raise SystemExit(
            f"ERROR: {args.source} is not staged as Parquet; its canonical streaming normalizer is still pending"
        )
    print(
        json.dumps(
            {
                "ok": True,
                "source": args.source,
                "input_scope": args.input_scope,
                "source_regex": args.source_regex,
                "text_column": args.text_column,
                "id_column": args.id_column,
                "source_column": args.source_column,
                "cleaning_profile": route.get("cleaning_profile"),
                "structural_policy": policy,
                "mode": "audit_only",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
