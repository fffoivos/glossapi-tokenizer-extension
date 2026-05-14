"""Reclean canonical-schema parquets while preserving prior quality scores.

This is the production driver for the wave-3 cleaner/tokenizer iteration.
It keeps every input column, replaces only the text column with cleaned text,
and appends cleaner activity/stat columns. Existing badness scores are never
overwritten. Missing Greek badness can be filled for selected datasets, using
the same Rust noise scorer as the corpus cleaner, before text is rewritten.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import gc
import glob as globmod
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

DROPPED_DATASETS = {
    "HuggingFaceFW/finepdfs-edu",
    "OPUS/OpenSubtitles-el-v2018",
}

SCRIPTS = ["greek", "latin", "french", "spanish", "punctuation", "numbers", "common_symbols"]

FLOAT_COLUMN_TYPES = {
    "greek_percentage": pa.float64(),
    "latin_percentage": pa.float64(),
    "polytonic_ratio": pa.float64(),
    "table_ratio": pa.float64(),
    "greek_badness_score": pa.float64(),
    "mojibake_badness_score": pa.float64(),
}

INT_COLUMN_TYPES = {
    "len_greek": pa.int64(),
}

STRING_COLUMN_TYPES = {
    "quality_method": pa.large_string(),
}

TIMESTAMP_COLUMN_TYPES = {
    "reevaluated_at": pa.timestamp("us", tz="UTC"),
}

NEW_FIELD_SPECS = [
    ("content_chars_kept", pa.uint64()),
    ("chars_dropped_by_line_drop", pa.uint64()),
    ("chars_dropped_by_normalization", pa.uint64()),
    ("chars_dropped_by_per_char_filter", pa.uint64()),
    ("lines_dropped_by_cleaner", pa.uint64()),
    ("marker_chars_passthrough", pa.uint64()),
    ("marker_chars_added", pa.uint64()),
    ("charset_greek_ratio", pa.float64()),
    ("charset_moji_ratio", pa.float64()),
    ("charset_punct_ratio", pa.float64()),
    ("mojibake_noise_ratio", pa.float64()),
    ("rule_a_match_count", pa.uint64()),
    ("rule_b_match_count", pa.uint64()),
    ("residue_line_drop_count", pa.uint64()),
    ("phase_a_fallback_reason", pa.string()),
    ("phase_a_dialect_ambiguous_input", pa.bool_()),
    ("cleaner_chars_before", pa.uint64()),
    ("cleaner_chars_after", pa.uint64()),
]


def value_missing(value: Any) -> bool:
    if value is None or value == "":
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def derive_greek_percentage(existing_value: Any, latin_percentage: Any) -> float | None:
    if not value_missing(existing_value):
        try:
            return float(existing_value)
        except Exception:
            pass
    if not value_missing(latin_percentage):
        try:
            return max(0.0, 100.0 - float(latin_percentage))
        except Exception:
            return None
    return None


def output_field_for(field: pa.Field) -> pa.Field:
    if field.name == "text":
        return pa.field(field.name, pa.large_string(), nullable=True)
    if field.name in FLOAT_COLUMN_TYPES:
        return pa.field(field.name, FLOAT_COLUMN_TYPES[field.name], nullable=True)
    if field.name in INT_COLUMN_TYPES:
        return pa.field(field.name, INT_COLUMN_TYPES[field.name], nullable=True)
    if field.name in STRING_COLUMN_TYPES:
        return pa.field(field.name, STRING_COLUMN_TYPES[field.name], nullable=True)
    if field.name in TIMESTAMP_COLUMN_TYPES:
        return pa.field(field.name, TIMESTAMP_COLUMN_TYPES[field.name], nullable=True)
    return field


def dataset_should_score(dataset: str | None, selected: set[str]) -> bool:
    return bool(dataset and ("*" in selected or dataset in selected))


def first_source_dataset(parquet_file: pq.ParquetFile) -> str | None:
    try:
        first_batch = next(parquet_file.iter_batches(batch_size=1, columns=["source_dataset"]))
    except StopIteration:
        return None
    return first_batch.column("source_dataset")[0].as_py()


def fill_missing_greek_badness(
    rows: list[dict[str, Any]],
    indices: list[int],
    raw_texts: list[str],
    *,
    score_threads: int,
) -> int:
    if not indices:
        return 0

    import glossapi_rs_noise

    metrics_rows = glossapi_rs_noise.score_texts_detailed(raw_texts, score_threads)
    reevaluated_at = dt.datetime.now(dt.timezone.utc)
    filled = 0
    for row_index, metrics in zip(indices, metrics_rows):
        row = rows[row_index]
        if not value_missing(row.get("greek_badness_score")):
            continue
        row["greek_badness_score"] = float(metrics[0])
        if value_missing(row.get("latin_percentage")):
            row["latin_percentage"] = float(metrics[1])
        if value_missing(row.get("table_ratio")):
            row["table_ratio"] = float(metrics[2])
        if value_missing(row.get("polytonic_ratio")):
            row["polytonic_ratio"] = float(metrics[3])
        if value_missing(row.get("len_greek")):
            row["len_greek"] = int(metrics[4])
        if value_missing(row.get("greek_percentage")):
            row["greek_percentage"] = derive_greek_percentage(
                row.get("greek_percentage"), row.get("latin_percentage")
            )
        if value_missing(row.get("quality_method")):
            row["quality_method"] = "glossapi_rs_noise"
        if value_missing(row.get("reevaluated_at")):
            row["reevaluated_at"] = reevaluated_at
        filled += 1
    return filled


def build_output_schema(schema_in: pa.Schema) -> pa.Schema:
    existing_names = set(schema_in.names)
    old_fields = [output_field_for(field) for field in schema_in]
    new_fields = [
        pa.field(name, arrow_type)
        for name, arrow_type in NEW_FIELD_SPECS
        if name not in existing_names
    ]
    return pa.schema(old_fields + new_fields)


def _process_parquet(
    input_path: str,
    output_path: str,
    batch_size: int,
    score_missing_greek_datasets: tuple[str, ...],
    score_threads: int,
    drop_datasets: tuple[str, ...],
) -> dict[str, Any]:
    import glossapi_rs_cleaner as cleaner

    t0 = time.time()
    pf = pq.ParquetFile(input_path)
    schema_in = pf.schema_arrow
    column_names_in = list(schema_in.names)
    selected_score_datasets = set(score_missing_greek_datasets)
    dropped_datasets = set(drop_datasets)

    source_dataset = first_source_dataset(pf)
    if source_dataset in dropped_datasets:
        return {
            "input_path": input_path,
            "output_path": None,
            "status": "skipped_dropped_dataset",
            "source_dataset": source_dataset,
            "rows": 0,
            "elapsed_sec": time.time() - t0,
        }

    out_schema = build_output_schema(schema_in)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    if tmp_output.exists():
        tmp_output.unlink()
    writer = pq.ParquetWriter(tmp_output, out_schema, compression="zstd", use_dictionary=True)

    rows_seen = 0
    rows_written = 0
    rows_empty_input = 0
    rows_greek_badness_scored = 0
    rows_greek_badness_preserved = 0
    rows_greek_badness_missing_after = 0
    rows_mojibake_missing_after = 0
    chars_before_total = 0
    chars_after_total = 0
    should_score = dataset_should_score(source_dataset, selected_score_datasets)

    try:
        pf2 = pq.ParquetFile(input_path)
        for batch in pf2.iter_batches(batch_size=batch_size):
            rows = batch.to_pylist()
            if not rows:
                continue

            new_cols: dict[str, list[Any]] = {name: [] for name, _ in NEW_FIELD_SPECS}
            score_indices: list[int] = []
            score_raw_texts: list[str] = []
            rows_seen += len(rows)

            for row_index, row in enumerate(rows):
                raw_text = row.get("text") or ""
                if should_score and value_missing(row.get("greek_badness_score")):
                    score_indices.append(row_index)
                    score_raw_texts.append(raw_text)
                elif not value_missing(row.get("greek_badness_score")):
                    rows_greek_badness_preserved += 1

                cb = len(raw_text)
                if not raw_text.strip():
                    rows_empty_input += 1
                    cleaned = ""
                    new_cols["content_chars_kept"].append(0)
                    new_cols["chars_dropped_by_line_drop"].append(0)
                    new_cols["chars_dropped_by_normalization"].append(0)
                    new_cols["chars_dropped_by_per_char_filter"].append(0)
                    new_cols["lines_dropped_by_cleaner"].append(0)
                    new_cols["marker_chars_passthrough"].append(0)
                    new_cols["marker_chars_added"].append(0)
                    new_cols["charset_greek_ratio"].append(0.0)
                    new_cols["charset_moji_ratio"].append(0.0)
                    new_cols["charset_punct_ratio"].append(0.0)
                    new_cols["mojibake_noise_ratio"].append(0.0)
                    new_cols["rule_a_match_count"].append(0)
                    new_cols["rule_b_match_count"].append(0)
                    new_cols["residue_line_drop_count"].append(0)
                    new_cols["phase_a_fallback_reason"].append(None)
                    new_cols["phase_a_dialect_ambiguous_input"].append(False)
                    new_cols["cleaner_chars_before"].append(0)
                    new_cols["cleaner_chars_after"].append(0)
                else:
                    charset = cleaner.analyze_charset(raw_text)
                    charset_greek_ratio = float(charset.get("greek_letter_ratio", 0.0) or 0.0)
                    charset_moji_ratio = float(charset.get("moji_residue_ratio", 0.0) or 0.0)
                    charset_punct_ratio = float(charset.get("ascii_punct_ratio", 0.0) or 0.0)

                    cleaned, stats = cleaner.clean_text_with_stats(
                        raw_text, SCRIPTS, None, True, 30, 3, "parser_surgical_verified"
                    )
                    ca = len(cleaned)
                    chars_before_total += cb
                    chars_after_total += ca

                    new_cols["content_chars_kept"].append(int(stats.get("content_chars_kept", 0) or 0))
                    new_cols["chars_dropped_by_line_drop"].append(int(stats.get("chars_dropped_by_line_drop", 0) or 0))
                    new_cols["chars_dropped_by_normalization"].append(int(stats.get("chars_dropped_by_normalization", 0) or 0))
                    new_cols["chars_dropped_by_per_char_filter"].append(int(stats.get("chars_dropped_by_per_char_filter", 0) or 0))
                    new_cols["lines_dropped_by_cleaner"].append(int(stats.get("lines_dropped_count", 0) or 0))
                    new_cols["marker_chars_passthrough"].append(int(stats.get("marker_chars_passthrough", 0) or 0))
                    new_cols["marker_chars_added"].append(int(stats.get("marker_chars_added", 0) or 0))
                    new_cols["charset_greek_ratio"].append(charset_greek_ratio)
                    new_cols["charset_moji_ratio"].append(charset_moji_ratio)
                    new_cols["charset_punct_ratio"].append(charset_punct_ratio)
                    new_cols["mojibake_noise_ratio"].append(charset_moji_ratio + charset_punct_ratio)
                    new_cols["rule_a_match_count"].append(int(stats.get("rule_a_match_count", 0) or 0))
                    new_cols["rule_b_match_count"].append(int(stats.get("rule_b_match_count", 0) or 0))
                    new_cols["residue_line_drop_count"].append(int(stats.get("residue_line_drop_count", 0) or 0))
                    new_cols["phase_a_fallback_reason"].append(stats.get("phase_a_fallback_reason"))
                    new_cols["phase_a_dialect_ambiguous_input"].append(bool(stats.get("phase_a_dialect_ambiguous_input", False)))
                    new_cols["cleaner_chars_before"].append(cb)
                    new_cols["cleaner_chars_after"].append(ca)

                row["text"] = cleaned

            rows_greek_badness_scored += fill_missing_greek_badness(
                rows,
                score_indices,
                score_raw_texts,
                score_threads=score_threads,
            )
            rows_greek_badness_missing_after += sum(
                1 for row in rows if value_missing(row.get("greek_badness_score"))
            )
            rows_mojibake_missing_after += sum(
                1 for row in rows if value_missing(row.get("mojibake_badness_score"))
            )

            out_arrays = []
            for field in out_schema:
                name = field.name
                if name in column_names_in:
                    values = [row.get(name) for row in rows]
                else:
                    values = new_cols[name]
                out_arrays.append(pa.array(values, type=field.type))
            writer.write_batch(pa.RecordBatch.from_arrays(out_arrays, schema=out_schema))
            rows_written += len(rows)

            if rows_seen % (batch_size * 16) == 0:
                gc.collect()
    except Exception:
        writer.close()
        if tmp_output.exists():
            tmp_output.unlink()
        raise

    writer.close()
    tmp_output.replace(output)
    return {
        "input_path": input_path,
        "output_path": output_path,
        "status": "ok",
        "source_dataset": source_dataset,
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "rows_empty_input": rows_empty_input,
        "rows_greek_badness_preserved": rows_greek_badness_preserved,
        "rows_greek_badness_scored": rows_greek_badness_scored,
        "rows_greek_badness_missing_after": rows_greek_badness_missing_after,
        "rows_mojibake_missing_after": rows_mojibake_missing_after,
        "chars_before_total": chars_before_total,
        "chars_after_total": chars_after_total,
        "elapsed_sec": time.time() - t0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--summary-jsonl", type=Path)
    parser.add_argument(
        "--score-missing-greek-badness-dataset",
        action="append",
        default=[],
        help="Dataset whose missing greek_badness_score values should be filled. May be repeated; '*' means all datasets.",
    )
    parser.add_argument("--score-threads-per-worker", type=int, default=1)
    parser.add_argument(
        "--drop-dataset",
        action="append",
        default=sorted(DROPPED_DATASETS),
        help="Source dataset to skip entirely. May be repeated.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    inputs = sorted(globmod.glob(args.input_glob))
    if not inputs:
        print(f"No inputs found at {args.input_glob}", file=sys.stderr)
        return 2

    tasks: list[tuple[str, str]] = []
    for input_path in inputs:
        output_path = args.output_root / Path(input_path).name
        if output_path.exists() and output_path.stat().st_size > 0:
            print(json.dumps({"event": "skip_existing", "input": input_path, "output": str(output_path)}, ensure_ascii=False), flush=True)
            continue
        tasks.append((input_path, str(output_path)))

    print(json.dumps({
        "event": "start",
        "input_count": len(inputs),
        "queued": len(tasks),
        "workers": args.workers,
        "batch_size": args.batch_size,
        "score_missing_greek_badness_datasets": args.score_missing_greek_badness_dataset,
        "score_threads_per_worker": args.score_threads_per_worker,
    }, ensure_ascii=False), flush=True)

    summary_fh = open(args.summary_jsonl, "w", encoding="utf-8") if args.summary_jsonl else None
    results: list[dict[str, Any]] = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    _process_parquet,
                    input_path,
                    output_path,
                    args.batch_size,
                    tuple(args.score_missing_greek_badness_dataset),
                    args.score_threads_per_worker,
                    tuple(args.drop_dataset),
                ): (input_path, output_path)
                for input_path, output_path in tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                input_path, output_path = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "input_path": input_path,
                        "output_path": output_path,
                        "status": "error",
                        "error": repr(exc),
                    }
                results.append(result)
                print(json.dumps({"event": "done", **result}, ensure_ascii=False), flush=True)
                if summary_fh:
                    summary_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    summary_fh.flush()
    finally:
        if summary_fh:
            summary_fh.close()

    error_count = sum(1 for result in results if result.get("status") == "error")
    total_rows = sum(int(result.get("rows_written") or 0) for result in results)
    total_chars_before = sum(int(result.get("chars_before_total") or 0) for result in results)
    total_chars_after = sum(int(result.get("chars_after_total") or 0) for result in results)
    total_greek_scored = sum(int(result.get("rows_greek_badness_scored") or 0) for result in results)
    total_greek_missing_after = sum(int(result.get("rows_greek_badness_missing_after") or 0) for result in results)
    pct_removed = (
        (1.0 - total_chars_after / total_chars_before) * 100.0 if total_chars_before else 0.0
    )
    print(json.dumps({
        "event": "all_done" if error_count == 0 else "done_with_errors",
        "tasks": len(results),
        "errors": error_count,
        "rows_total": total_rows,
        "rows_greek_badness_scored": total_greek_scored,
        "rows_greek_badness_missing_after": total_greek_missing_after,
        "chars_before_total": total_chars_before,
        "chars_after_total": total_chars_after,
        "pct_chars_removed": round(pct_removed, 3),
    }, ensure_ascii=False), flush=True)
    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
