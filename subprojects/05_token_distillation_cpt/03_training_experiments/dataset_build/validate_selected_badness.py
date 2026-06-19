#!/usr/bin/env python3
"""Validate badness-score invariants for the CPT selected Greek pool.

The CPT mix recipes consume the selected parquet by source label, so this guard
keeps the upstream "clean60" contract explicit at the build boundary. It scans
only provenance/score columns and writes a fingerprinted JSON report that later
array jobs can check cheaply.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_THRESHOLD = 60.0
DEFAULT_HPLT_DATASET = "HPLT/ell_Grek_ge8_no_mt_clean60"
REQUIRED_COLUMNS = ("source_dataset", "greek_badness_score")
DOC_ID_COLUMNS = ("source_doc_id", "doc_key", "doc_id", "id")


def _expand_path_spec(spec: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(spec))
    if "$" in expanded:
        raise SystemExit(f"selected path still contains an unexpanded variable: {spec!r}")
    return expanded


def resolve_parquet_paths(spec: str) -> list[Path]:
    expanded = _expand_path_spec(spec)
    if glob.has_magic(expanded):
        paths = [Path(p) for p in glob.glob(expanded)]
    else:
        path = Path(expanded)
        if path.is_dir():
            paths = list(path.rglob("*.parquet"))
        else:
            paths = [path]
    existing = sorted(p.resolve() for p in paths if p.is_file())
    if not existing:
        raise SystemExit(f"selected path matched no parquet files: {expanded}")
    return existing


def fingerprint_paths(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        stat = path.stat()
        out.append(
            {
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return out


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _violates_threshold(score: float, threshold: float, strict_lt: bool) -> bool:
    if strict_lt:
        return score >= threshold
    return score > threshold


def _record_sample(samples: list[dict[str, Any]], limit: int, sample: dict[str, Any]) -> None:
    if len(samples) < limit:
        samples.append(sample)


def _source_summary(
    counts: Counter[str],
    missing: Counter[str],
    invalid: Counter[str],
    over: Counter[str],
    hplt_wrong: Counter[str],
    limit: int,
) -> list[dict[str, Any]]:
    names = [name for name, _ in counts.most_common(limit)]
    return [
        {
            "source_dataset": name,
            "rows": int(counts[name]),
            "missing_badness": int(missing[name]),
            "invalid_badness": int(invalid[name]),
            "badness_out_of_policy": int(over[name]),
            "hplt_wrong_dataset": int(hplt_wrong[name]),
        }
        for name in names
    ]


def validate_selected(args: argparse.Namespace) -> dict[str, Any]:
    source_regex = re.compile(args.source_regex) if args.source_regex else None
    paths = resolve_parquet_paths(args.selected)
    input_files = fingerprint_paths(paths)

    missing_schema: list[dict[str, Any]] = []
    for path in paths:
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        missing = [col for col in REQUIRED_COLUMNS if col not in schema_names]
        if missing:
            missing_schema.append({"path": str(path), "missing_columns": missing})

    policy = {
        "greek_badness_score": f"< {args.threshold}" if args.strict_lt else f"<= {args.threshold}",
        "strict_lt": bool(args.strict_lt),
        "threshold": float(args.threshold),
        "source_regex": args.source_regex,
        "hplt_source_prefix": args.hplt_source_prefix,
        "required_hplt_dataset": args.required_hplt_dataset,
    }

    if missing_schema:
        return {
            "ok": False,
            "selected": _expand_path_spec(args.selected),
            "input_files": input_files,
            "policy": policy,
            "missing_required_columns": missing_schema,
            "rows_scanned": 0,
            "violations": {"missing_required_columns": len(missing_schema)},
            "samples": missing_schema[: args.max_samples],
        }

    source_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    over_counts: Counter[str] = Counter()
    hplt_wrong_counts: Counter[str] = Counter()

    rows_seen = 0
    rows_scanned = 0
    rows_skipped_source_regex = 0
    missing_source = 0
    missing_badness = 0
    invalid_badness = 0
    badness_out_of_policy = 0
    hplt_wrong_dataset = 0
    max_badness: float | None = None
    samples: list[dict[str, Any]] = []

    for path in paths:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        columns = list(REQUIRED_COLUMNS)
        doc_id_col = next((col for col in DOC_ID_COLUMNS if col in available), None)
        if doc_id_col:
            columns.append(doc_id_col)
        file_row_base = 0
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
            data = batch.to_pydict()
            for i in range(batch.num_rows):
                rows_seen += 1
                row_index = file_row_base + i
                source = data["source_dataset"][i]
                source_name = str(source) if source is not None else ""
                if source_regex and not source_regex.search(source_name):
                    rows_skipped_source_regex += 1
                    continue
                rows_scanned += 1
                source_counts[source_name or "<missing>"] += 1
                doc_id = str(data[doc_id_col][i]) if doc_id_col and data[doc_id_col][i] is not None else None

                if not source_name:
                    missing_source += 1
                    _record_sample(
                        samples,
                        args.max_samples,
                        {
                            "reason": "missing_source_dataset",
                            "path": str(path),
                            "row_index": row_index,
                            "doc_id": doc_id,
                        },
                    )

                score_raw = data["greek_badness_score"][i]
                score = _float_or_none(score_raw)
                if score is None:
                    if score_raw is None or (isinstance(score_raw, str) and not score_raw.strip()):
                        missing_badness += 1
                        missing_counts[source_name or "<missing>"] += 1
                        reason = "missing_greek_badness_score"
                    else:
                        invalid_badness += 1
                        invalid_counts[source_name or "<missing>"] += 1
                        reason = "invalid_greek_badness_score"
                    _record_sample(
                        samples,
                        args.max_samples,
                        {
                            "reason": reason,
                            "path": str(path),
                            "row_index": row_index,
                            "source_dataset": source_name,
                            "source_doc_id": doc_id,
                            "greek_badness_score": score_raw,
                        },
                    )
                else:
                    max_badness = score if max_badness is None else max(max_badness, score)
                    if _violates_threshold(score, args.threshold, args.strict_lt):
                        badness_out_of_policy += 1
                        over_counts[source_name or "<missing>"] += 1
                        _record_sample(
                            samples,
                            args.max_samples,
                            {
                                "reason": "greek_badness_score_out_of_policy",
                                "path": str(path),
                                "row_index": row_index,
                                "source_dataset": source_name,
                                "source_doc_id": doc_id,
                                "greek_badness_score": score,
                            },
                        )

                if source_name.startswith(args.hplt_source_prefix) and source_name != args.required_hplt_dataset:
                    hplt_wrong_dataset += 1
                    hplt_wrong_counts[source_name] += 1
                    _record_sample(
                        samples,
                        args.max_samples,
                        {
                            "reason": "hplt_source_dataset_not_clean60",
                            "path": str(path),
                            "row_index": row_index,
                            "source_dataset": source_name,
                            "source_doc_id": doc_id,
                            "required_hplt_dataset": args.required_hplt_dataset,
                        },
                    )
            file_row_base += batch.num_rows

    violations = {
        "missing_source_dataset": int(missing_source),
        "missing_greek_badness_score": int(missing_badness),
        "invalid_greek_badness_score": int(invalid_badness),
        "greek_badness_score_out_of_policy": int(badness_out_of_policy),
        "hplt_source_dataset_not_clean60": int(hplt_wrong_dataset),
    }
    ok = rows_scanned > 0 and all(value == 0 for value in violations.values())
    if rows_scanned == 0:
        violations["no_rows_matched_source_regex"] = 1

    return {
        "ok": ok,
        "selected": _expand_path_spec(args.selected),
        "input_files": input_files,
        "policy": policy,
        "rows_seen": int(rows_seen),
        "rows_scanned": int(rows_scanned),
        "rows_skipped_source_regex": int(rows_skipped_source_regex),
        "distinct_source_datasets": int(len(source_counts)),
        "max_greek_badness_score": max_badness,
        "violations": violations,
        "source_summary": _source_summary(
            source_counts,
            missing_counts,
            invalid_counts,
            over_counts,
            hplt_wrong_counts,
            args.max_source_summary,
        ),
        "samples": samples,
    }


def require_current_report(args: argparse.Namespace) -> int:
    report_path = Path(args.require_current_report)
    if not report_path.is_file():
        print(f"ERROR: badness validation report is missing: {report_path}", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    current_files = fingerprint_paths(resolve_parquet_paths(args.selected))
    expected_policy = {
        "strict_lt": bool(args.strict_lt),
        "threshold": float(args.threshold),
        "source_regex": args.source_regex,
        "hplt_source_prefix": args.hplt_source_prefix,
        "required_hplt_dataset": args.required_hplt_dataset,
    }
    report_policy = report.get("policy") or {}
    policy_mismatches = {
        key: {"report": report_policy.get(key), "expected": value}
        for key, value in expected_policy.items()
        if report_policy.get(key) != value
    }
    if not report.get("ok"):
        print(f"ERROR: badness validation report is not ok: {report_path}", file=sys.stderr)
        return 2
    if report.get("input_files") != current_files:
        print(f"ERROR: badness validation report is stale for selected path: {args.selected}", file=sys.stderr)
        print(f"  report: {report_path}", file=sys.stderr)
        return 2
    if policy_mismatches:
        print(f"ERROR: badness validation report policy mismatch: {json.dumps(policy_mismatches)}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "event": "badness_validation_report_current",
                "report": str(report_path),
                "rows_scanned": report.get("rows_scanned"),
                "selected": report.get("selected"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True, help="Selected Greek parquet file, directory, or glob.")
    parser.add_argument("--output-json", help="Write the validation summary here.")
    parser.add_argument("--require-current-report", help="Check an existing report instead of rescanning.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--strict-lt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-regex", help="Validate only source_dataset values matching this regex.")
    parser.add_argument("--hplt-source-prefix", default="HPLT/")
    parser.add_argument("--required-hplt-dataset", default=DEFAULT_HPLT_DATASET)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--max-source-summary", type=int, default=200)
    args = parser.parse_args()

    if args.require_current_report:
        return require_current_report(args)

    summary = validate_selected(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    if not summary.get("ok"):
        print("ERROR: selected badness validation failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
