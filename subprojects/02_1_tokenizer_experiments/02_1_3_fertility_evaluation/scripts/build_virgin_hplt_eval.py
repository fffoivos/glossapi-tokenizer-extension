"""Build a guaranteed-virgin held-out eval slice by anti-joining the HPLT
release against a tokenizer arm's training mix on (source_dataset,
source_doc_id).

Why anti-join on source_doc_id (not text-md5):
  - source_doc_id is unique per HPLT doc and small enough to keep in
    memory as a duckdb temp table
  - text-md5 anti-join over 48 M HPLT rows + 14 M train rows is single-
    threaded in pure Python (~80 min on 1 vCPU) but DuckDB can do the
    doc_id variant in < 1 min on 64 vCPUs

Quality filters applied to the virgin pool:
  - 1,500 <= length(text) <= 80,000 chars (drop very short / very long)
  - greek_badness_score IS NULL OR greek_badness_score < 30 (stricter
    than the wave-2 broad cutoff of 60, so the held-out is cleaner than
    training)
  - polytonic_ratio IS NULL OR polytonic_ratio < 0.05 (keep the modern-
    Greek style C3 was trained on)

For the C3 sweep we used a reservoir sample of 10,000 rows with
seed=20260511.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", type=Path, required=True,
                        help="mix.parquet that fed the tokenizer training")
    parser.add_argument("--hplt-glob", type=str, required=True,
                        help="glob over HPLT clean60 release parquets")
    parser.add_argument("--out-parquet", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_260_511)
    parser.add_argument("--sample-rows", type=int, default=10_000)
    parser.add_argument("--min-chars", type=int, default=1_500)
    parser.add_argument("--max-chars", type=int, default=80_000)
    parser.add_argument("--greek-badness-max", type=float, default=30.0)
    parser.add_argument("--polytonic-ratio-max", type=float, default=0.05)
    parser.add_argument("--source-dataset-prefix", default="HPLT/",
                        help="filter mix doc_ids by source_dataset starting with this")
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_parquet.parent / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute(f"PRAGMA temp_directory='{tmp}'")

    print("[1/4] collecting HPLT doc_ids already in mix", flush=True)
    con.execute(f"""
        CREATE TABLE mix_hplt_ids AS
        SELECT DISTINCT source_doc_id
        FROM read_parquet('{args.mix}')
        WHERE source_dataset LIKE '{args.source_dataset_prefix}%'
    """)
    n_mix = con.execute("SELECT count(*) FROM mix_hplt_ids").fetchone()[0]
    print(f"    mix HPLT distinct doc_ids: {n_mix:,}", flush=True)

    print("[2/4] anti-join + quality filter", flush=True)
    con.execute(f"""
        CREATE TABLE virgin AS
        SELECT
          h.source_dataset, h.source_doc_id, h.text, h.title,
          h.greek_badness_score, h.polytonic_ratio, h.greek_percentage,
          length(h.text) AS chars
        FROM read_parquet('{args.hplt_glob}', union_by_name=true) h
        ANTI JOIN mix_hplt_ids m ON m.source_doc_id = h.source_doc_id
        WHERE h.text IS NOT NULL
          AND length(h.text) BETWEEN {args.min_chars} AND {args.max_chars}
          AND (h.greek_badness_score IS NULL OR h.greek_badness_score < {args.greek_badness_max})
          AND (h.polytonic_ratio IS NULL OR h.polytonic_ratio < {args.polytonic_ratio_max})
    """)
    n_virgin = con.execute("SELECT count(*) FROM virgin").fetchone()[0]
    print(f"    virgin rows after quality + anti-join: {n_virgin:,}", flush=True)

    print(f"[3/4] reservoir-sampling {args.sample_rows} rows", flush=True)
    con.execute(f"""
        CREATE TABLE sample_out AS
        SELECT source_dataset, source_doc_id, text, title,
               greek_badness_score, polytonic_ratio, greek_percentage, chars
        FROM virgin
        USING SAMPLE {args.sample_rows} ROWS (RESERVOIR, {args.seed})
    """)
    n_sample = con.execute("SELECT count(*) FROM sample_out").fetchone()[0]

    print(f"[4/4] writing parquet -> {args.out_parquet}", flush=True)
    con.execute(f"COPY (SELECT * FROM sample_out) TO '{args.out_parquet}' (FORMAT 'parquet')")

    summary = {
        "method": "source_doc_id anti-join (mix HPLT ids vs HPLT release)",
        "mix_parquet": str(args.mix),
        "hplt_glob": args.hplt_glob,
        "out_parquet": str(args.out_parquet),
        "seed": args.seed,
        "sample_target": args.sample_rows,
        "mix_hplt_distinct_doc_ids": n_mix,
        "hplt_rows_virgin_after_quality": n_virgin,
        "sample_rows": n_sample,
    }
    (args.out_parquet.parent / "build_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
