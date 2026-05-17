"""Produce verifiable clean held-out parquets by anti-joining val/test
against train on text-md5.

The splitter in `subprojects/_archive/01_2_training_dataset_mix/scripts/
export_text_budgeted_splits.py` partitions input rows by row index, not
by document/text. When the input mix contains duplicate texts, the
duplicates can land in different splits and contaminate the held-out.

For C3 specifically: train ∩ val = 30 exact text-md5 collisions; train ∩
test = 36. See `docs/C3_CONVERGENCE.md` § Held-out integrity.

This script emits `<src>_clean.parquet` next to the source files for any
parquet whose text matches a row in train.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True,
                        help="path to train.parquet")
    parser.add_argument("--holdouts", nargs="+", type=Path, required=True,
                        help="held-out parquets to clean (val.parquet, test.parquet, etc.)")
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute(f"""
        CREATE TABLE train_md5 AS
        SELECT DISTINCT md5(text) AS h
        FROM read_parquet('{args.train}')
        WHERE text IS NOT NULL
    """)
    summary = {}
    for src in args.holdouts:
        dst = src.with_name(src.stem + "_clean.parquet")
        before = con.execute(f"SELECT count(*) FROM read_parquet('{src}')").fetchone()[0]
        con.execute(f"""
            COPY (
                SELECT s.*
                FROM read_parquet('{src}') s
                WHERE NOT EXISTS (
                    SELECT 1 FROM train_md5 t WHERE t.h = md5(s.text)
                )
            ) TO '{dst}' (FORMAT 'parquet')
        """)
        after = con.execute(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]
        summary[src.name] = {
            "src": str(src),
            "dst": str(dst),
            "rows_before": before,
            "rows_after": after,
            "rows_dropped": before - after,
        }
        print(f"{src.name}: {before} -> {after} (dropped {before - after})")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
