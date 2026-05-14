from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        path.write_text("")
        return
    frame.to_csv(path, index=False)


def _pct(series: pd.Series, predicate) -> float:
    if len(series) == 0:
        return 0.0
    return round(float(series.map(predicate).mean()) * 100.0, 2)


def summarize_by(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, subset in frame.groupby(label_col):
        rows.append(
            {
                label_col: label,
                "group_count": int(len(subset)),
                "rows_in_groups": int(subset["group_size"].sum()),
                "median_raw_chars": int(subset["text_chars"].median()),
                "median_chars_no_comments": int(subset["chars_no_comments"].median()),
                "median_chars_no_comments_tables_math": int(subset["chars_no_comments_tables_math"].median()),
                "median_chars_plain_body": int(subset["chars_plain_body"].median()),
                "median_header_count": float(round(subset["header_count"].median(), 2)),
                "median_table_line_count": float(round(subset["table_line_count"].median(), 2)),
                "pct_plain_body_lt_1000": _pct(subset["chars_plain_body"], lambda value: float(value) < 1000),
                "pct_plain_body_lt_3000": _pct(subset["chars_plain_body"], lambda value: float(value) < 3000),
                "pct_zero_headers": _pct(subset["header_count"], lambda value: float(value) == 0),
                "pct_header_ge_3": _pct(subset["header_count"], lambda value: float(value) >= 3),
                "pct_tables_present": _pct(subset["table_line_count"], lambda value: float(value) > 0),
                "pct_comments_present": _pct(subset["html_comment_count"], lambda value: float(value) > 0),
                "pct_math_present": _pct(subset["latex_marker_count"], lambda value: float(value) > 0),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["rows_in_groups", "group_count", label_col], ascending=[False, False, True])


def build_report(
    *,
    run_id: str,
    bucket_summary: pd.DataFrame,
    subreason_summary: pd.DataFrame,
    reviewed_summary: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"# OA Text Shape Research: {run_id}")
    lines.append("")
    lines.append("## What These Metrics Mean")
    lines.append("- `chars_no_comments`: shared-text length after removing HTML comments and collapsing whitespace")
    lines.append("- `chars_no_comments_tables_math`: shared-text length after additionally removing table lines and LaTeX-like math")
    lines.append("- `chars_plain_body`: shared-text length after also removing markdown headers")
    lines.append("- `header_count`: number of markdown-style headers (`#`, `##`, etc.)")
    lines.append("")
    lines.append("## Summary By Deterministic Resolution Status")
    for row in bucket_summary.to_dict(orient="records"):
        lines.append(
            f"- {row['deterministic_resolution_status']}: groups={row['group_count']}, rows={row['rows_in_groups']}, "
            f"median_plain_chars={row['median_chars_plain_body']}, median_headers={row['median_header_count']}, "
            f"pct_plain_lt_1000={row['pct_plain_body_lt_1000']}%, pct_tables={row['pct_tables_present']}%, pct_comments={row['pct_comments_present']}%"
        )
    lines.append("")
    lines.append("## Summary By OA Suspicious Subreason")
    for row in subreason_summary.to_dict(orient="records"):
        lines.append(
            f"- {row['suspicious_subreason']}: groups={row['group_count']}, rows={row['rows_in_groups']}, "
            f"median_plain_chars={row['median_chars_plain_body']}, median_headers={row['median_header_count']}, "
            f"pct_plain_lt_1000={row['pct_plain_body_lt_1000']}%, pct_math={row['pct_math_present']}%"
        )
    if not reviewed_summary.empty:
        lines.append("")
        lines.append("## Summary On Reviewed Top-12 Semantic Slice")
        for row in reviewed_summary.to_dict(orient="records"):
            lines.append(
                f"- {row['high_level_content_type']}: groups={row['group_count']}, rows={row['rows_in_groups']}, "
                f"median_plain_chars={row['median_chars_plain_body']}, median_headers={row['median_header_count']}, "
                f"pct_plain_lt_1000={row['pct_plain_body_lt_1000']}%"
            )
    return "\n".join(lines) + "\n"


def analyze_openarchives_text_shape_research(run_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    analysis_root = (output_dir or (run_root / "analysis")).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)

    profiles = pd.read_csv(analysis_root / "oa_semantic_resolution_group_profiles.csv")
    bucket_summary = summarize_by(profiles, "deterministic_resolution_status")
    subreason_summary = summarize_by(profiles, "suspicious_subreason")

    reviewed_summary = pd.DataFrame()
    reviewed_csv = analysis_root / "oa_high_level_content_review_top12_20260327.csv"
    if reviewed_csv.exists():
        reviewed = pd.read_csv(reviewed_csv)
        reviewed_profiles = profiles.merge(reviewed[["group_hash", "high_level_content_type"]], on="group_hash", how="inner")
        if not reviewed_profiles.empty:
            reviewed_summary = summarize_by(reviewed_profiles, "high_level_content_type")

    bucket_summary_path = analysis_root / "oa_text_shape_summary_by_bucket.csv"
    subreason_summary_path = analysis_root / "oa_text_shape_summary_by_subreason.csv"
    reviewed_summary_path = analysis_root / "oa_text_shape_summary_top12_reviewed.csv"
    report_path = analysis_root / "oa_text_shape_research_report.md"

    write_csv(bucket_summary, bucket_summary_path)
    write_csv(subreason_summary, subreason_summary_path)
    write_csv(reviewed_summary, reviewed_summary_path)
    report_path.write_text(
        build_report(
            run_id=str(run_root.name),
            bucket_summary=bucket_summary,
            subreason_summary=subreason_summary,
            reviewed_summary=reviewed_summary,
        )
    )

    payload = {
        "run_id": str(run_root.name),
        "analysis_root": str(analysis_root),
        "oa_text_shape_summary_by_bucket_path": str(bucket_summary_path),
        "oa_text_shape_summary_by_subreason_path": str(subreason_summary_path),
        "oa_text_shape_summary_top12_reviewed_path": str(reviewed_summary_path),
        "oa_text_shape_research_report_path": str(report_path),
    }
    (analysis_root / "oa_text_shape_research_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Research OA shared-text shape metrics after stripping comments, tables, and math.")
    parser.add_argument("--run-root", required=True, type=Path, help="Path to the completed exact-stage run root")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory; defaults to <run_root>/analysis")
    args = parser.parse_args()
    payload = analyze_openarchives_text_shape_research(args.run_root, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
