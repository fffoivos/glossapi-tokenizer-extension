from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from analyze_openarchives_suspicious_exact_groups import load_suspicious_groups, write_csv


csv.field_size_limit(10**9)

TOKEN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "και",
    "στην",
    "στη",
    "στο",
    "των",
    "του",
    "της",
    "των",
    "των",
    "τον",
    "την",
    "ένα",
    "μια",
    "για",
    "από",
    "with",
    "from",
    "into",
    "using",
    "study",
    "analysis",
    "της",
    "στης",
}


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value)).casefold()
    return WHITESPACE_RE.sub(" ", text).strip()


def tokenize(value: str | None) -> list[str]:
    normalized = normalize_text(value)
    tokens = []
    for token in TOKEN_RE.findall(normalized):
        if len(token) < 4 or token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def score_member_row(row: dict[str, str]) -> dict[str, Any]:
    text_tokens = set(tokenize(row.get("text")))
    title_tokens = sorted(set(tokenize(row.get("title"))))
    author_tokens = sorted(set(tokenize(row.get("author"))))
    matched_title = sorted(token for token in title_tokens if token in text_tokens)
    matched_author = sorted(token for token in author_tokens if token in text_tokens)
    title_overlap = len(matched_title) / len(title_tokens) if title_tokens else 0.0
    author_overlap = len(matched_author) / len(author_tokens) if author_tokens else 0.0
    combined = (title_overlap * 0.75) + (author_overlap * 0.25)
    return {
        "doc_key": str(row.get("doc_key") or ""),
        "source_doc_id": str(row.get("source_doc_id") or ""),
        "oa_collection_slug": str(row.get("oa_collection_slug") or "unknown"),
        "title": str(row.get("title") or ""),
        "author": str(row.get("author") or ""),
        "title_overlap": title_overlap,
        "author_overlap": author_overlap,
        "combined_score": combined,
        "title_token_count": len(title_tokens),
        "author_token_count": len(author_tokens),
        "matched_title_tokens": "|".join(matched_title),
        "matched_author_tokens": "|".join(matched_author),
    }


def choose_best_candidate(rows: list[dict[str, str]]) -> dict[str, Any]:
    scores = [score_member_row(row) for row in rows]
    return sorted(
        scores,
        key=lambda score: (
            -float(score["combined_score"]),
            -float(score["title_overlap"]),
            -float(score["author_overlap"]),
            str(score["oa_collection_slug"]),
            str(score["source_doc_id"]),
        ),
    )[0]


def has_plausible_origin_match(score: dict[str, Any]) -> bool:
    if float(score["title_overlap"]) >= 0.6:
        return True
    if float(score["author_overlap"]) >= 0.5:
        return True
    if float(score["combined_score"]) >= 0.55:
        return True
    if float(score["title_overlap"]) >= 0.4 and float(score["author_overlap"]) > 0:
        return True
    return False


def classify_group_resolution(subreason: str, best_score: dict[str, Any] | None) -> tuple[str, str, str]:
    if subreason == "image_markup_mixed_text":
        return (
            "generic_artifact_group",
            "historical_scrape_image_markup_capture",
            "artifact",
        )
    if subreason == "latex_or_formula_stub":
        return (
            "generic_artifact_group",
            "historical_scrape_formula_stub_capture",
            "artifact",
        )
    if subreason in {
        "author_denies_access_notice",
        "author_access_notice",
        "electronic_access_unavailable_notice",
        "specific_work_unavailable_notice",
        "embargo_release_notice",
    }:
        return (
            "generic_notice_group",
            "historical_scrape_repository_notice_capture",
            "notice",
        )
    if best_score is not None and has_plausible_origin_match(best_score):
        return (
            "likely_true_file_in_group",
            "probable_partial_scraper_attachment_error",
            "content",
        )
    return (
        "no_plausible_true_file_in_group",
        "probable_historical_scraper_attachment_error",
        "content",
    )


def load_suspicious_members(summary: dict[str, Any], suspicious_groups: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    members_path = Path(summary["oa_meaningful_exact_members_path"])
    wanted = set(suspicious_groups["group_hash"].astype(str))
    grouped: dict[str, list[dict[str, str]]] = {}
    with members_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            group_hash = str(row["group_hash"])
            if group_hash not in wanted:
                continue
            bucket = grouped.setdefault(group_hash, [])
            bucket.append(
                {
                    "group_hash": group_hash,
                    "doc_key": str(row.get("doc_key") or ""),
                    "source_doc_id": str(row.get("source_doc_id") or ""),
                    "oa_collection_slug": str(row.get("oa_collection_slug") or "unknown"),
                    "title": str(row.get("title") or ""),
                    "author": str(row.get("author") or ""),
                    "text": str(row.get("text") or ""),
                    "is_kept": str(row.get("is_kept") or ""),
                }
            )
    return grouped


def build_group_audit(suspicious_groups: pd.DataFrame, grouped_members: dict[str, list[dict[str, str]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in suspicious_groups.to_dict(orient="records"):
        group_hash = str(group["group_hash"])
        members = grouped_members.get(group_hash, [])
        best_score = choose_best_candidate(members) if members else None
        origin_resolution, scraper_marker, evidence_tier = classify_group_resolution(
            str(group["suspicious_subreason"]),
            best_score,
        )
        rows.append(
            {
                "group_hash": group_hash,
                "group_size": int(group["group_size"]),
                "suspicious_subreason": str(group["suspicious_subreason"]),
                "origin_resolution": origin_resolution,
                "scraper_error_marker": scraper_marker,
                "evidence_tier": evidence_tier,
                "best_candidate_doc_key": str(best_score["doc_key"]) if best_score else "",
                "best_candidate_source_doc_id": str(best_score["source_doc_id"]) if best_score else "",
                "best_candidate_collection": str(best_score["oa_collection_slug"]) if best_score else "",
                "best_candidate_title": str(best_score["title"]) if best_score else "",
                "best_candidate_author": str(best_score["author"]) if best_score else "",
                "best_title_overlap": round(float(best_score["title_overlap"]), 4) if best_score else 0.0,
                "best_author_overlap": round(float(best_score["author_overlap"]), 4) if best_score else 0.0,
                "best_combined_score": round(float(best_score["combined_score"]), 4) if best_score else 0.0,
                "matched_title_tokens": str(best_score["matched_title_tokens"]) if best_score else "",
                "matched_author_tokens": str(best_score["matched_author_tokens"]) if best_score else "",
                "collections_in_group": str(group["collections_in_group"]),
                "unique_title_count": int(group["unique_title_count"]),
                "unique_author_count": int(group["unique_author_count"]),
                "representative_title": str(group["representative_title"] or ""),
                "representative_text_preview": str(group["representative_text_preview"] or ""),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "group_hash",
                "group_size",
                "suspicious_subreason",
                "origin_resolution",
                "scraper_error_marker",
                "evidence_tier",
                "best_candidate_doc_key",
                "best_candidate_source_doc_id",
                "best_candidate_collection",
                "best_candidate_title",
                "best_candidate_author",
                "best_title_overlap",
                "best_author_overlap",
                "best_combined_score",
                "matched_title_tokens",
                "matched_author_tokens",
                "collections_in_group",
                "unique_title_count",
                "unique_author_count",
                "representative_title",
                "representative_text_preview",
            ]
        )
    return frame.sort_values(
        ["origin_resolution", "group_size", "best_combined_score", "group_hash"],
        ascending=[True, False, False, True],
    )


def build_row_markers(group_audit: pd.DataFrame, grouped_members: dict[str, list[dict[str, str]]]) -> pd.DataFrame:
    audit_map = {str(row["group_hash"]): row for row in group_audit.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    for group_hash, members in grouped_members.items():
        audit_row = audit_map[group_hash]
        for member in members:
            row_issue = audit_row["scraper_error_marker"]
            if audit_row["origin_resolution"] == "likely_true_file_in_group":
                if member["doc_key"] == audit_row["best_candidate_doc_key"]:
                    row_issue = "candidate_true_file_for_shared_text"
                else:
                    row_issue = "candidate_misattached_row"
            elif audit_row["origin_resolution"] == "no_plausible_true_file_in_group":
                row_issue = "no_true_file_detected_in_group"
            rows.append(
                {
                    "group_hash": group_hash,
                    "source_doc_id": member["source_doc_id"],
                    "doc_key": member["doc_key"],
                    "oa_collection_slug": member["oa_collection_slug"],
                    "title": member["title"],
                    "author": member["author"],
                    "origin_resolution": audit_row["origin_resolution"],
                    "scraper_error_marker": audit_row["scraper_error_marker"],
                    "row_issue_marker": row_issue,
                    "best_candidate_doc_key": audit_row["best_candidate_doc_key"],
                    "best_candidate_source_doc_id": audit_row["best_candidate_source_doc_id"],
                    "best_candidate_title": audit_row["best_candidate_title"],
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "group_hash",
                "source_doc_id",
                "doc_key",
                "oa_collection_slug",
                "title",
                "author",
                "origin_resolution",
                "scraper_error_marker",
                "row_issue_marker",
                "best_candidate_doc_key",
                "best_candidate_source_doc_id",
                "best_candidate_title",
            ]
        )
    return frame.sort_values(["origin_resolution", "group_hash", "row_issue_marker", "source_doc_id"], ascending=[True, True, True, True])


def build_resolution_summary(group_audit: pd.DataFrame) -> pd.DataFrame:
    if group_audit.empty:
        return pd.DataFrame(columns=["origin_resolution", "group_count", "rows_in_groups", "dropped_rows"])
    summary = group_audit.groupby(["origin_resolution", "scraper_error_marker"], as_index=False).agg(
        group_count=("group_hash", "count"),
        rows_in_groups=("group_size", "sum"),
    )
    summary["dropped_rows"] = summary["rows_in_groups"] - summary["group_count"]
    return summary.sort_values(["rows_in_groups", "group_count", "origin_resolution"], ascending=[False, False, True])


def build_report(run_id: str, resolution_summary: pd.DataFrame, group_audit: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append(f"# OA Suspicious Origin Audit: {run_id}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("- choose the best in-group metadata match to the shared text when possible")
    lines.append("- emit a stable scraper-error marker when the shared text does not plausibly belong to any row in the group")
    lines.append("")
    lines.append("## Resolution Summary")
    for row in resolution_summary.to_dict(orient="records"):
        lines.append(
            f"- {row['origin_resolution']} / {row['scraper_error_marker']}: groups={int(row['group_count'])}, rows={int(row['rows_in_groups'])}, dropped_rows={int(row['dropped_rows'])}"
        )
    lines.append("")
    lines.append("## Largest Groups With No Plausible In-Group Origin")
    unresolved = group_audit[group_audit["origin_resolution"] == "no_plausible_true_file_in_group"].head(12)
    for row in unresolved.to_dict(orient="records"):
        lines.append(
            f"- group={row['group_hash'][:16]}..., size={int(row['group_size'])}, collections={row['collections_in_group']}, "
            f"repr_title={row['representative_title']!r}, best_score={float(row['best_combined_score']):.3f}"
        )
    if unresolved.empty:
        lines.append("- none")
    lines.append("")
    lines.append("## Largest Groups With A Plausible In-Group Origin")
    resolved = group_audit[group_audit["origin_resolution"] == "likely_true_file_in_group"].head(12)
    for row in resolved.to_dict(orient="records"):
        lines.append(
            f"- group={row['group_hash'][:16]}..., size={int(row['group_size'])}, best_title={row['best_candidate_title']!r}, "
            f"collection={row['best_candidate_collection']}, best_score={float(row['best_combined_score']):.3f}, matched_title_tokens={row['matched_title_tokens']!r}"
        )
    if resolved.empty:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def analyze_openarchives_suspicious_origin_audit(run_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    summary, suspicious_groups, default_analysis_root = load_suspicious_groups(run_root)
    analysis_root = (output_dir or default_analysis_root).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)

    grouped_members = load_suspicious_members(summary, suspicious_groups)
    group_audit = build_group_audit(suspicious_groups, grouped_members)
    row_markers = build_row_markers(group_audit, grouped_members)
    resolution_summary = build_resolution_summary(group_audit)

    group_audit_path = analysis_root / "oa_suspicious_origin_group_audit.csv"
    row_markers_path = analysis_root / "oa_suspicious_origin_row_markers.csv"
    resolution_summary_path = analysis_root / "oa_suspicious_origin_resolution_summary.csv"
    report_path = analysis_root / "oa_suspicious_origin_report.md"

    write_csv(group_audit, group_audit_path)
    write_csv(row_markers, row_markers_path)
    write_csv(resolution_summary, resolution_summary_path)
    report_path.write_text(build_report(str(summary["run_id"]), resolution_summary, group_audit))

    payload = {
        "run_id": str(summary["run_id"]),
        "analysis_root": str(analysis_root),
        "oa_suspicious_origin_group_audit_path": str(group_audit_path),
        "oa_suspicious_origin_row_markers_path": str(row_markers_path),
        "oa_suspicious_origin_resolution_summary_path": str(resolution_summary_path),
        "oa_suspicious_origin_report_path": str(report_path),
    }
    (analysis_root / "oa_suspicious_origin_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit suspicious OA exact groups for likely true file selection and scraper-error markers.")
    parser.add_argument("--run-root", required=True, type=Path, help="Path to a completed exact-stage run root")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory; defaults to <run_root>/analysis")
    args = parser.parse_args()
    payload = analyze_openarchives_suspicious_origin_audit(args.run_root, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
