#!/usr/bin/env python3
# KNOWN ISSUE — partition leaks duplicate texts across splits.
#
# The `stable_key` below hashes `source_dataset:source_doc_id:source_split_row_id:salt`
# so each input row gets an independent split assignment. When the same
# text appears multiple times in the input mix (different rows, e.g.
# near-duplicates that survived upstream dedup), the duplicates can end
# up in different splits. Verified on C3
# (`C3_wave2_broad_glossapi_plus_hplt_50_50`, 2026-05-11): the exported
# `train.parquet` has 29,527 intra-train duplicate texts, and `val`/`test`
# overlap with `train` on 30 and 36 exact text-md5 matches respectively.
#
# Fix path (not yet applied): partition by something that is unique per
# distinct text, e.g.
#     stable_key = md5(coalesce(text, ''))
# or pre-dedup the input mix on text before assignment. In the meantime,
# downstream evaluation slices should anti-join val/test against train
# on `md5(text)` before being used as held-outs — see
# `docs/C3_CONVERGENCE.md` § "Held-out integrity".
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
    parser.add_argument("--allow-missing-badness-scores", action="store_true",
                        help="Keep rows whose badness score is missing. Off by default; production filters fail closed.")
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
    parser.add_argument("--row-group-size", type=int, default=2048,
                        help="Rows per output Parquet row group. Small row groups give continuous-BPE workers enough shards.")
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
    has_greek_badness = has_column(glob_path, "greek_badness_score")
    has_mojibake_badness = has_column(glob_path, "mojibake_badness_score")
    has_charset_greek = has_column(glob_path, "charset_greek_ratio")
    has_content_chars_kept = has_column(glob_path, "content_chars_kept")
    has_source_mix_chars = has_column(glob_path, "source_mix_chars")

    if has_chars:
        chars_expr = "coalesce(src.chars, length(src.text))"
    elif has_source_mix_chars:
        chars_expr = "coalesce(src.source_mix_chars, length(src.text))"
    else:
        chars_expr = "length(src.text)"

    # Wave-2 standard exclusion predicates. In production, missing and
    # empty badness scores are not eligible data: they mean the row has
    # not been scored by the cleaner/noise pass. The explicit opt-in is
    # only for tiny smoke/debug fixtures that intentionally omit scores.
    if not args.allow_missing_badness_scores and not has_greek_badness:
        raise SystemExit("input is missing greek_badness_score; cannot apply production badness filter")
    if args.mojibake_lte is not None and not args.allow_missing_badness_scores and not has_mojibake_badness:
        raise SystemExit("input is missing mojibake_badness_score; cannot apply production badness filter")

    if has_greek_badness:
        greek_score = "try_cast(src.greek_badness_score as DOUBLE)"
        if args.allow_missing_badness_scores:
            greek_badness_pred = f"({greek_score} IS NULL OR {greek_score} < {args.badness_lt})"
        else:
            greek_badness_pred = f"({greek_score} IS NOT NULL AND {greek_score} < {args.badness_lt})"
    else:
        greek_badness_pred = "TRUE"

    mojibake_pred = "TRUE"
    if args.mojibake_lte is not None and has_mojibake_badness:
        mojibake_score = "try_cast(src.mojibake_badness_score as DOUBLE)"
        if args.allow_missing_badness_scores:
            mojibake_pred = f"({mojibake_score} IS NULL OR {mojibake_score} <= {args.mojibake_lte})"
        else:
            mojibake_pred = f"({mojibake_score} IS NOT NULL AND {mojibake_score} <= {args.mojibake_lte})"

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

    print(json.dumps({"event": "build_assigned_rows_start", "input_glob": glob_path}, ensure_ascii=False), flush=True)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE assigned_rows AS
        WITH source_rows AS (
          SELECT
            row_number() OVER () AS source_split_row_id,
            src.*
          FROM read_parquet({glob_sql}, union_by_name=true) AS src
        ),
        filtered_source_rows AS (
          SELECT
            source_split_row_id,
            src.source_dataset,
            src.source_doc_id,
            src.text,
            {chars_expr} AS chars,
            md5(
              coalesce(src.source_dataset, '') || ':' ||
              coalesce(src.source_doc_id, '') || ':' ||
              cast(source_split_row_id AS VARCHAR) || ':{salt_sql}'
            ) AS stable_key
          FROM source_rows AS src
          WHERE {greek_badness_pred}
            AND {mojibake_pred}
            AND {greek_ratio_pred}
            AND {non_empty_pred}
            AND {needs_ocr_pred}
        ),
        ranked AS (
          SELECT
            source_split_row_id,
            source_dataset,
            source_doc_id,
            text,
            chars,
            stable_key,
            sum(chars) OVER (
              ORDER BY stable_key, source_doc_id
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cum_chars
          FROM filtered_source_rows
        )
        SELECT
          source_split_row_id,
          source_dataset,
          source_doc_id,
          text,
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
    assigned = con.execute("SELECT split, count(*) AS row_count, coalesce(sum(chars),0) AS char_count FROM assigned_rows GROUP BY split ORDER BY split").fetchall()
    filtered = con.execute("SELECT count(*) AS rows, coalesce(sum(chars), 0) AS chars FROM assigned_rows").fetchone()
    print(json.dumps({"event": "build_assigned_rows_done", "rows": int(filtered[0]), "chars": int(filtered[1]), "splits": assigned}, ensure_ascii=False), flush=True)

    for split in ("train", "val", "test"):
        manifest_path = manifest_root / f"{split}_manifest.csv"
        export_path = export_root / f"{split}.parquet"
        print(json.dumps({"event": "export_split_start", "split": split}, ensure_ascii=False), flush=True)
        con.execute(
            f"""
            COPY (
              SELECT source_split_row_id, source_dataset, source_doc_id, chars, stable_key
              FROM assigned_rows
              WHERE split = '{split}'
              ORDER BY stable_key, source_split_row_id
            ) TO '{manifest_path.as_posix()}' (HEADER, DELIMITER ',')
            """
        )
        con.execute(
            f"""
            COPY (
              SELECT text
              FROM assigned_rows
              WHERE split = '{split}'
              ORDER BY stable_key, source_split_row_id
            ) TO '{export_path.as_posix()}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {max(1, int(args.row_group_size))})
            """
        )
        row = con.execute(f"SELECT count(*), coalesce(sum(chars),0) FROM assigned_rows WHERE split='{split}'").fetchone()
        print(json.dumps({"event": "export_split_done", "split": split, "rows": int(row[0]), "chars": int(row[1]), "path": str(export_path)}, ensure_ascii=False), flush=True)

    summary = {
        "input_root": str(args.input_root.resolve()),
        "input_glob": glob_path,
        "output_root": str(output_root),
        "threads": args.threads,
        "badness_lt": args.badness_lt,
        "mojibake_lte": args.mojibake_lte,
        "allow_missing_badness_scores": args.allow_missing_badness_scores,
        "greek_ratio_gte": args.greek_ratio_gte,
        "require_non_empty_content": args.require_non_empty_content,
        "has_greek_badness": has_greek_badness,
        "has_mojibake_badness": has_mojibake_badness,
        "has_charset_greek": has_charset_greek,
        "has_content_chars_kept": has_content_chars_kept,
        "has_source_mix_chars": has_source_mix_chars,
        "train_chars": args.train_chars,
        "val_chars": args.val_chars,
        "test_chars": args.test_chars,
        "row_group_size": args.row_group_size,
        "seed_salt": args.seed_salt,
        "has_chars": has_chars,
        "has_needs_ocr": has_needs_ocr,
        "has_ocr_success": has_ocr_success,
    }
    for split in ("train", "val", "test"):
        row = con.execute(f"SELECT count(*), coalesce(sum(chars),0) FROM assigned_rows WHERE split='{split}'").fetchone()
        summary[split] = {"rows": int(row[0]), "chars": int(row[1])}
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
