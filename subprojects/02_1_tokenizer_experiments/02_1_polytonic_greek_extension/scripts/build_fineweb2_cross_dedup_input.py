#!/usr/bin/env python3
"""Build input parquets for cross-deduping our corpus against FineWeb-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def sql_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--fineweb", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    data_dir = args.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    ours_out = data_dir / "ours_polytonic_kept.parquet"
    con.execute(
        f"""
        copy (
          select source_dataset, source_doc_id, text, title, author,
                 source_metadata_json,
                 is_historical_or_polytonic,
                 greek_badness_score,
                 mojibake_badness_score,
                 needs_ocr,
                 is_empty,
                 ocr_success,
                 doc_key as original_doc_key
          from read_parquet({sql_quote(args.ours)})
        ) to {sql_quote(ours_out)} (format parquet, compression zstd)
        """
    )

    fine_out = data_dir / "fineweb2_main_grc_Grek.parquet"
    con.execute(
        f"""
        copy (
          select 'fineweb2_main_grc_Grek' as source_dataset,
                 id as source_doc_id,
                 text,
                 '' as title,
                 '' as author,
                 json_object(
                   'url', url,
                   'dump', dump,
                   'date', date,
                   'file_path', file_path,
                   'language_score', language_score,
                   'minhash_cluster_size', minhash_cluster_size,
                   'top_langs', top_langs
                 ) as source_metadata_json,
                 true as is_historical_or_polytonic,
                 0.0::double as greek_badness_score,
                 0.0::double as mojibake_badness_score,
                 false as needs_ocr,
                 coalesce(length(text), 0) = 0 as is_empty,
                 false as ocr_success,
                 id as original_doc_key
          from read_parquet({sql_quote(args.fineweb)})
        ) to {sql_quote(fine_out)} (format parquet, compression zstd)
        """
    )

    summary: dict[str, object] = {
        "ours": str(args.ours),
        "fineweb": str(args.fineweb),
        "outputs": {},
    }
    for p in sorted(data_dir.glob("*.parquet")):
        rows = con.execute(
            f"""
            select source_dataset, count(*) as rows, sum(length(text)) as chars
            from read_parquet({sql_quote(p)})
            group by 1
            """
        ).fetchall()
        summary["outputs"][p.name] = [
            {"source_dataset": row[0], "rows": row[1], "chars": row[2]}
            for row in rows
        ]
    (args.output_root / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
