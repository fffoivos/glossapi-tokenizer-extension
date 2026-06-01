#!/usr/bin/env python3
"""Derive the Task-1 HPLT-only Greek B1 recipe from the 03 bakeoff recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = json.loads(args.input.read_text(encoding="utf-8"))

    sources = []
    for source in recipe["sources"]:
        bucket = source.get("bucket")
        if bucket != "greek":
            sources.append(source)
            continue
        if source.get("name") != "greek_hplt_clean60":
            continue
        hplt = dict(source)
        hplt["share_within_bucket"] = 1.0
        hplt["weight"] = 0.70
        hplt["comment"] = (
            "Task-1 Vanilla regime diagnostic: HPLT clean60 is the only Greek "
            "source. All other B1 buckets are inherited unchanged from the 03 "
            "bakeoff recipe."
        )
        sources.append(hplt)

    if not any(s.get("name") == "greek_hplt_clean60" for s in sources):
        raise SystemExit("input recipe did not contain greek_hplt_clean60")

    recipe["name"] = "hplt_b1_vanilla_regime"
    recipe["version"] = "v1.0_20260528"
    recipe["description"] = (
        "Task-1 Vanilla CPT regime diagnostic recipe. Top-level shares stay "
        "70% Greek / 24% replay / 4% code / 2% math, but the Greek slice is "
        "restricted to HPLT/ell_Grek_ge8_no_mt_clean60 after the existing "
        "Apertus-overlap and internal-dedup overlay."
    )
    recipe["sources"] = sources
    recipe["validation"] = {
        "expected_weight_sum": 1.0,
        "expected_greek_share": 0.70,
        "expected_replay_share": 0.24,
        "expected_code_share": 0.04,
        "expected_math_share": 0.02,
        "required_greek_sources": ["greek_hplt_clean60"],
        "excluded_greek_sources": "all non-HPLT Greek sources from the 03 bakeoff recipe",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
