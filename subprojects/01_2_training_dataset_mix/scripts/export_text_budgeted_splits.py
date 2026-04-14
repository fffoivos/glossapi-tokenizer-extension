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
    parser.add_argument("--badness-lt", type=float, default=10.0)
    parser.add_argument("--mojibake-lte", type=float, default=None)
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

    chars_expr = "coalesce(src.chars, length(src.text))" if has_chars else "length(src.text)"
    mojibake_pred = "TRUE"
    if args.mojibake_lte is not None:
        mojibake_pred = f"(src.mojibake_badness_score IS NULL OR src.mojibake_badness_score <= {args.mojibake_lte})"

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
