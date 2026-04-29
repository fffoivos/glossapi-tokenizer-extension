#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


PATTERNS = {
    "GLYPH": "GLYPH",
    "GLYPH_repeat": "(GLYPH){2,}",
    "hyphenminus": "/hyphenminus",
    "uni_glyph": "/uni[0-9A-Fa-f]{4,}",
    "gid_g": "/g(id)?[0-9]+",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={max(1, int(args.threads))}")
    con.execute("PRAGMA preserve_insertion_order=false")

    summary = []
    for name, pattern in PATTERNS.items():
        row = con.execute(
            """
            SELECT
              count(*) AS rows,
              coalesce(sum(array_length(regexp_extract_all(text, ?))), 0) AS hits
            FROM read_parquet(?)
            WHERE regexp_matches(text, ?)
            """,
            [pattern, str(args.parquet), pattern],
        ).fetchone()
        summary.append({"pattern": name, "rows": int(row[0]), "hits": int(row[1])})

    combined = "|".join(PATTERNS.values())
    samples = []
    rows = con.execute(
        "SELECT text FROM read_parquet(?) WHERE regexp_matches(text, ?) LIMIT ?",
        [str(args.parquet), combined, int(args.sample_limit)],
    ).fetchall()
    for index, (text,) in enumerate(rows, 1):
        body = str(text)
        needle = ""
        position = -1
        for candidate in ["GLYPH", "/hyphenminus", "/uni", "/gid", "/g"]:
            position = body.find(candidate)
            if position >= 0:
                needle = candidate
                break
        lo = max(0, position - 240)
        hi = min(len(body), position + 360)
        samples.append(
            {
                "sample": index,
                "needle": needle,
                "snippet": " ".join(body[lo:hi].split()),
            }
        )

    payload = {
        "parquet": str(args.parquet),
        "summary": summary,
        "samples": samples,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
