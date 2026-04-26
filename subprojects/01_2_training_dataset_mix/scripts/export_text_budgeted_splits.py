#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic text-only train/val/test parquet splits by character budget.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    # Wave-2 standard exclusion thresholds (THRESHOLDS.yaml § standard_exclusions).
    # `greek_badness_score > 60` and `mojibake_badness_score > 0.1` are dropped
    # (i.e. keep when score is below the cutoff).
    parser.add_argument("--badness-lt", type=float, default=60.0,
                        help="Drop rows with greek_badness_score >= this. Wave-2 default: 60.")
    parser.add_argument("--mojibake-lte", type=float, default=0.1,
                        help="Drop rows with mojibake_badness_score > this. Wave-2 default: 0.1.")
    parser.add_argument("--greek-ratio-gte", type=float, default=0.5,
                        help="Drop rows with charset_greek_ratio < this. Wave-2 default: 0.5.")
    parser.add_argument("--require-non-empty-content", action="store_true", default=True,
                        help="Drop rows with content_chars_kept == 0 (cleaner output is all markers/whitespace). Wave-2 default: on.")
    parser.add_argument("--no-require-non-empty-content", dest="require_non_empty_content",
                        action="store_false",
                        help="Disable the empty-after-comments filter (override wave-2 default).")
    parser.add_argument("--train-chars", type=int, required=True)
    parser.add_argument("--val-chars", type=int, required=True)
    parser.add_argument("--test-chars", type=int, required=True)
    parser.add_argument("--seed-salt", required=True)
    return parser.parse_args()


def data_glob(input_root: Path) -> str:
    root = input_root.resolve()
    if any(root.glob("*.parquet")):
        return str((root / "*.parquet").resolve())
    if any((root / "data").glob("*.parquet")):
        return str((root / "data" / "*.parquet").resolve())
    raise FileNotFoundError(f"No parquet files under {input_root}")


def has_column(root_glob: str, name: str) -> bool:
    for path in sorted(Path(root_glob.rsplit('/', 1)[0]).glob('*.parquet')):
        if name in pq.read_schema(path).names:
            return True
    return False


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    export_root = output_root / "exports"
    manifest_root = output_root / "manifests"
    scripts_root = output_root / "scripts"
    temp_root = output_root / "tmp"
    for path in (export_root, manifest_root, scripts_root, temp_root):
        path.mkdir(parents=True, exist_ok=True)

    shutil.copy2(Path(__file__).resolve(), scripts_root / Path(__file__).name)

    glob_path = data_glob(args.input_root)
    has_chars = has_column(glob_path, "chars")
    has_needs_ocr = has_column(glob_path, "needs_ocr")
    has_ocr_success = has_column(glob_path, "ocr_success")
    has_charset_greek = has_column(glob_path, "charset_greek_ratio")
    has_content_chars_kept = has_column(glob_path, "content_chars_kept")

    chars_expr = "coalesce(src.chars, length(src.text))" if has_chars else "length(src.text)"

    # Wave-2 standard exclusion predicates. The `IS NULL` clauses keep
    # rows whose upstream score is missing; the upstream of this stage
    # (Corpus.clean + post-clean scoring) is responsible for ensuring
    # every row carries a greek_badness_score / mojibake_badness_score
    # before it reaches this filter. If a column is missing entirely
    # from the input parquet, treat as "no signal" → no rejection by
    # that axis (so the filter is robust to schema drift but the
    # upstream pipeline must guarantee the column exists for the
    # filter to actually fire).
    mojibake_pred = "TRUE"
    if args.mojibake_lte is not None:
        mojibake_pred = (
            f"(src.mojibake_badness_score IS NULL "
            f"OR src.mojibake_badness_score <= {args.mojibake_lte})"
        )

    greek_ratio_pred = "TRUE"
    if has_charset_greek and args.greek_ratio_gte is not None:
        greek_ratio_pred = (
            f"(src.charset_greek_ratio IS NULL "
            f"OR src.charset_greek_ratio >= {args.greek_ratio_gte})"
        )

    non_empty_pred = "TRUE"
    if args.require_non_empty_content and has_content_chars_kept:
        non_empty_pred = (
            "(src.content_chars_kept IS NULL OR src.content_chars_kept > 0)"
        )

    if has_needs_ocr and has_ocr_success:
        needs_ocr_pred = "CASE WHEN src.source_dataset='openarchives.gr' THEN coalesce(src.needs_ocr,false)=FALSE ELSE (coalesce(src.needs_ocr,false)=FALSE OR coalesce(src.ocr_success,false)=TRUE) END"
    elif has_needs_ocr:
        needs_ocr_pred = "coalesce(src.needs_ocr,false)=FALSE"
    else:
        needs_ocr_pred = "TRUE"

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={max(1, int(args.threads))}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{temp_root.as_posix()}'")

    glob_sql = sql_quote(glob_path)
    salt_sql = args.seed_salt.replace("'", "''")

    print(json.dumps({"event": "build_filtered_source_docs_start", "input_glob": glob_path}, ensure_ascii=False), flush=True)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE filtered_source_docs AS
        SELECT
          src.source_dataset,
          src.source_doc_id,
          {chars_expr} AS chars,
          md5(src.source_dataset || ':' || src.source_doc_id || ':{salt_sql}') AS stable_key
        FROM read_parquet({glob_sql}, union_by_name=true) AS src
        WHERE src.greek_badness_score < {args.badness_lt}
          AND {mojibake_pred}
          AND {greek_ratio_pred}
          AND {non_empty_pred}
          AND {needs_ocr_pred}
        """
    )
    filtered = con.execute("SELECT count(*) AS rows, coalesce(sum(chars), 0) AS chars FROM filtered_source_docs").fetchone()
    print(json.dumps({"event": "build_filtered_source_docs_done", "rows": int(filtered[0]), "chars": int(filtered[1])}, ensure_ascii=False), flush=True)

    print(json.dumps({"event": "build_assigned_start"}, ensure_ascii=False), flush=True)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE assigned AS
        WITH ranked AS (
          SELECT
            source_dataset,
            source_doc_id,
            chars,
            stable_key,
            sum(chars) OVER (
              ORDER BY stable_key, source_doc_id
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cum_chars
          FROM filtered_source_docs
        )
        SELECT
          source_dataset,
          source_doc_id,
          chars,
          stable_key,
          CASE
            WHEN cum_chars <= {args.val_chars} THEN 'val'
            WHEN cum_chars <= {args.val_chars + args.test_chars} THEN 'test'
            WHEN cum_chars <= {args.val_chars + args.test_chars + args.train_chars} THEN 'train'
            ELSE 'drop'
          END AS split
        FROM ranked
        """
    )
    assigned = con.execute("SELECT split, count(*) AS row_count, coalesce(sum(chars),0) AS char_count FROM assigned GROUP BY split ORDER BY split").fetchall()
    print(json.dumps({"event": "build_assigned_done", "splits": assigned}, ensure_ascii=False), flush=True)

    for split in ("train", "val", "test"):
        manifest_path = manifest_root / f"{split}_manifest.csv"
        export_path = export_root / f"{split}.parquet"
        print(json.dumps({"event": "export_split_start", "split": split}, ensure_ascii=False), flush=True)
        con.execute(
            f"""
            COPY (
              SELECT source_dataset, source_doc_id, chars, stable_key
              FROM assigned
              WHERE split = '{split}'
              ORDER BY stable_key, source_doc_id
            ) TO '{manifest_path.as_posix()}' (HEADER, DELIMITER ',')
            """
        )
        con.execute(
            f"""
            COPY (
              SELECT src.text
              FROM read_parquet({glob_sql}, union_by_name=true) AS src
              JOIN assigned a
                ON src.source_dataset = a.source_dataset
               AND src.source_doc_id = a.source_doc_id
              WHERE a.split = '{split}'
              ORDER BY a.stable_key, a.source_doc_id
            ) TO '{export_path.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
            """
        )
        row = con.execute(f"SELECT count(*), coalesce(sum(chars),0) FROM assigned WHERE split='{split}'").fetchone()
        print(json.dumps({"event": "export_split_done", "split": split, "rows": int(row[0]), "chars": int(row[1]), "path": str(export_path)}, ensure_ascii=False), flush=True)

    summary = {
        "input_root": str(args.input_root.resolve()),
        "input_glob": glob_path,
        "output_root": str(output_root),
        "threads": args.threads,
        "badness_lt": args.badness_lt,
        "mojibake_lte": args.mojibake_lte,
        "greek_ratio_gte": args.greek_ratio_gte,
        "require_non_empty_content": args.require_non_empty_content,
        "has_charset_greek": has_charset_greek,
        "has_content_chars_kept": has_content_chars_kept,
        "train_chars": args.train_chars,
        "val_chars": args.val_chars,
        "test_chars": args.test_chars,
        "seed_salt": args.seed_salt,
        "has_chars": has_chars,
        "has_needs_ocr": has_needs_ocr,
        "has_ocr_success": has_ocr_success,
    }
    for split in ("train", "val", "test"):
        row = con.execute(f"SELECT count(*), coalesce(sum(chars),0) FROM assigned WHERE split='{split}'").fetchone()
        summary[split] = {"rows": int(row[0]), "chars": int(row[1])}
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
