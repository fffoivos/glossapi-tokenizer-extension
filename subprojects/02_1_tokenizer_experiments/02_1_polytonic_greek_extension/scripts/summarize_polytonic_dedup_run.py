#!/usr/bin/env python3
"""Summarize a polytonic-source dedup run."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit("summarize_polytonic_dedup_run.py requires duckdb") from exc

    root = args.run_root
    con = duckdb.connect()
    queries = [
        (
            "kept_dropped_by_source",
            f"""
            SELECT source_dataset,
                   count(*) AS input_rows,
                   sum(CASE WHEN decision = 'keep' THEN 1 ELSE 0 END) AS kept_rows,
                   sum(CASE WHEN decision != 'keep' THEN 1 ELSE 0 END) AS dropped_rows
            FROM read_parquet('{root}/final/dedup_decisions.parquet')
            GROUP BY 1
            ORDER BY input_rows DESC
            """,
        ),
        (
            "drops_by_stage",
            f"""
            SELECT source_dataset, decision_stage, count(*) AS rows
            FROM read_parquet('{root}/final/dedup_decisions.parquet')
            WHERE decision != 'keep'
            GROUP BY 1,2
            ORDER BY source_dataset, decision_stage
            """,
        ),
        (
            "family_cross_source",
            f"""
            SELECT family_mixed_source,
                   count(*) AS rows,
                   count(DISTINCT family_id) AS families
            FROM read_parquet('{root}/builder_metadata/dedup_family_membership.parquet')
            GROUP BY 1
            ORDER BY 1
            """,
        ),
        (
            "cross_source_families_top20",
            f"""
            SELECT family_id,
                   family_size,
                   family_source_count,
                   string_agg(DISTINCT source_dataset, ', ' ORDER BY source_dataset) AS sources
            FROM read_parquet('{root}/builder_metadata/dedup_family_membership.parquet')
            WHERE family_mixed_source
            GROUP BY 1,2,3
            ORDER BY family_size DESC, family_id
            LIMIT 20
            """,
        ),
    ]

    for title, sql in queries:
        print(f"## {title}")
        print(con.execute(sql).fetchdf().to_string(index=False))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
