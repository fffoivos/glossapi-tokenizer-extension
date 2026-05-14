from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from analyze_openarchives_suspicious_exact_groups import classify_suspicious_subreason, load_suspicious_groups, write_csv
from shared_text_shape_metrics import build_shared_text_shape_metrics, normalize_text


csv.field_size_limit(10**9)

SEMANTIC_METADATA_KEYWORDS = (
    "περιγραφ",
    "abstract",
    "summary",
    "subject",
    "θέμα",
    "keyword",
    "λέξεις",
    "επιστημον",
    "type",
    "ημερομην",
    "date",
    "creator",
    "contributor",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def classify_content_size_hint(profile: dict[str, Any]) -> str:
    chars = int(profile["chars_plain_body"])
    headers = int(profile["header_count"])

    if chars < 500:
        return "tiny_stub"
    if chars < 8_000 and headers <= 4:
        return "short_note_or_brief_article"
    if chars < 60_000 and headers <= 20:
        return "article_sized"
    if chars < 180_000 and headers <= 60:
        return "long_article_or_short_thesis"
    if chars < 500_000 and headers <= 180:
        return "thesis_sized"
    return "book_or_manual_sized"


def has_notice_like_shape(profile: dict[str, Any]) -> bool:
    """Keep notice bucketing narrow so long structured documents are not misclassified."""
    return (
        int(profile["chars_plain_body"]) <= 1_000
        and int(profile["header_count"]) == 0
        and int(profile["table_line_count"]) == 0
        and int(profile["latex_marker_count"]) == 0
        and int(profile["image_markup_count"]) == 0
        and int(profile["html_comment_count"]) == 0
    )


def classify_deterministic_resolution_status(text: str, suspicious_subreason: str, profile: dict[str, Any]) -> str:
    full_text_subreason = classify_suspicious_subreason(text[:10_000])
    notice_like = has_notice_like_shape(profile)

    if suspicious_subreason in {
        "author_denies_access_notice",
        "author_access_notice",
        "embargo_release_notice",
        "electronic_access_unavailable_notice",
        "specific_work_unavailable_notice",
    } or full_text_subreason in {
        "author_denies_access_notice",
        "author_access_notice",
        "embargo_release_notice",
        "electronic_access_unavailable_notice",
        "specific_work_unavailable_notice",
    }:
        if notice_like:
            return "clear_notice_like_output"

    chars = int(profile["chars_plain_body"])
    words = int(profile["plain_body_word_count"])
    headers = int(profile["header_count"])
    chars_no_comments = int(profile["chars_no_comments"])

    if chars == 0 and chars_no_comments > 0:
        return "clear_no_body_after_stripping"
    if chars < 120 and words < 25 and headers == 0:
        return "clear_tiny_flat_output"
    return "needs_semantic_review"


def classify_semantic_review_path(deterministic_resolution_status: str, group_size: int, max_rows_per_packet: int) -> str:
    if deterministic_resolution_status != "needs_semantic_review":
        return "skip_clear_deterministic_output"
    if group_size <= max_rows_per_packet:
        return "single_packet"
    return "split_group_batches"


def parse_source_metadata(value: str | None) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    raw = str(value).strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_unparsed_source_metadata_json": raw[:10_000]}
    if isinstance(parsed, dict):
        return parsed
    return {"_source_metadata_value": parsed}


def select_semantic_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key, value in metadata.items():
        key_norm = unicodedata.normalize("NFC", str(key)).casefold()
        if any(keyword in key_norm for keyword in SEMANTIC_METADATA_KEYWORDS):
            selected[str(key)] = value
    return selected


def load_group_members(summary: dict[str, Any], suspicious_groups: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    members_path = Path(summary["oa_meaningful_exact_members_path"])
    wanted = set(suspicious_groups["group_hash"].astype(str))
    grouped: dict[str, list[dict[str, Any]]] = {}
    with members_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            group_hash = str(row["group_hash"])
            if group_hash not in wanted:
                continue
            metadata = parse_source_metadata(row.get("source_metadata_json"))
            grouped.setdefault(group_hash, []).append(
                {
                    "doc_key": str(row.get("doc_key") or ""),
                    "source_doc_id": str(row.get("source_doc_id") or ""),
                    "oa_collection_slug": str(row.get("oa_collection_slug") or "unknown"),
                    "title": str(row.get("title") or ""),
                    "author": str(row.get("author") or ""),
                    "text": str(row.get("text") or ""),
                    "metadata_preview": str(row.get("metadata_preview") or ""),
                    "source_metadata": metadata,
                    "semantic_metadata": select_semantic_metadata(metadata),
                }
            )
    return grouped


def build_group_profile_frame(
    suspicious_groups: pd.DataFrame,
    grouped_members: dict[str, list[dict[str, Any]]],
    *,
    max_rows_per_packet: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in suspicious_groups.to_dict(orient="records"):
        group_hash = str(group["group_hash"])
        members = grouped_members[group_hash]
        shared_text = normalize_text(members[0]["text"])
        profile = build_shared_text_shape_metrics(shared_text)
        content_size_hint = classify_content_size_hint(profile)
        deterministic_status = classify_deterministic_resolution_status(shared_text, str(group["suspicious_subreason"]), profile)
        semantic_review_path = classify_semantic_review_path(deterministic_status, int(group["group_size"]), max_rows_per_packet)
        rows.append(
            {
                "group_hash": group_hash,
                "group_size": int(group["group_size"]),
                "collections_in_group": str(group["collections_in_group"]),
                "suspicious_subreason": str(group["suspicious_subreason"]),
                "representative_title": str(group["representative_title"] or ""),
                "unique_title_count": int(group["unique_title_count"]),
                "unique_author_count": int(group["unique_author_count"]),
                "text_chars": int(profile["raw_chars"]),
                "raw_collapsed_chars": int(profile["raw_collapsed_chars"]),
                "word_count": int(profile["word_count"]),
                "header_count": int(profile["header_count"]),
                "table_line_count": int(profile["table_line_count"]),
                "latex_marker_count": int(profile["latex_marker_count"]),
                "image_markup_count": int(profile["image_markup_count"]),
                "html_comment_count": int(profile["html_comment_count"]),
                "abstract_marker_count": int(profile["abstract_marker_count"]),
                "bibliography_marker_count": int(profile["bibliography_marker_count"]),
                "chars_no_comments": int(profile["chars_no_comments"]),
                "chars_no_comments_tables": int(profile["chars_no_comments_tables"]),
                "chars_no_comments_tables_math": int(profile["chars_no_comments_tables_math"]),
                "chars_plain_body": int(profile["chars_plain_body"]),
                "plain_body_word_count": int(profile["plain_body_word_count"]),
                "removed_by_comments_chars": int(profile["removed_by_comments_chars"]),
                "removed_by_tables_chars": int(profile["removed_by_tables_chars"]),
                "removed_by_math_chars": int(profile["removed_by_math_chars"]),
                "removed_by_headers_chars": int(profile["removed_by_headers_chars"]),
                "alpha_char_ratio_plain_body": float(profile["alpha_char_ratio_plain_body"]),
                "content_size_hint": content_size_hint,
                "deterministic_resolution_status": deterministic_status,
                "semantic_review_path": semantic_review_path,
                "shared_text_preview": shared_text[:500],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "group_hash",
                "group_size",
                "collections_in_group",
                "suspicious_subreason",
                "representative_title",
                "unique_title_count",
                "unique_author_count",
                "text_chars",
                "raw_collapsed_chars",
                "word_count",
                "header_count",
                "table_line_count",
                "latex_marker_count",
                "image_markup_count",
                "html_comment_count",
                "abstract_marker_count",
                "bibliography_marker_count",
                "chars_no_comments",
                "chars_no_comments_tables",
                "chars_no_comments_tables_math",
                "chars_plain_body",
                "plain_body_word_count",
                "removed_by_comments_chars",
                "removed_by_tables_chars",
                "removed_by_math_chars",
                "removed_by_headers_chars",
                "alpha_char_ratio_plain_body",
                "content_size_hint",
                "deterministic_resolution_status",
                "semantic_review_path",
                "shared_text_preview",
            ]
        )
    return frame.sort_values(
        ["deterministic_resolution_status", "text_chars", "group_size", "group_hash"],
        ascending=[True, False, False, True],
    )


def build_summary_frame(group_profiles: pd.DataFrame) -> pd.DataFrame:
    if group_profiles.empty:
        return pd.DataFrame(columns=["deterministic_resolution_status", "content_size_hint", "group_count", "rows_in_groups", "implied_dropped_rows"])
    summary = group_profiles.groupby(["deterministic_resolution_status", "content_size_hint"], as_index=False).agg(
        group_count=("group_hash", "count"),
        rows_in_groups=("group_size", "sum"),
    )
    summary["implied_dropped_rows"] = summary["rows_in_groups"] - summary["group_count"]
    return summary.sort_values(["rows_in_groups", "group_count", "deterministic_resolution_status"], ascending=[False, False, True])


def gemini_prompt_contract(*, split_group_batch: bool) -> dict[str, Any]:
    problem_statement = (
        "We have a group of dataset rows that share the same extracted document text but have conflicting metadata. "
        "Our goal is to identify which row, if any, is the true owner of the shared text so we can keep that row and drop the others."
    )
    instructions = [
        "Use only the shared text and the structured metadata rows in this packet.",
        "Look for direct semantic matches from the text into metadata fields such as title, author, abstract, subject, keywords, description, and date.",
        "Return `best_candidate_in_batch` only when one row is a clearly better owner of the shared text than the others.",
        "Return `no_plausible_candidate_in_batch` if none of the rows plausibly owns the shared text.",
        "Return `multiple_plausible_candidates_in_batch` only if more than one row in this packet plausibly owns the same text.",
    ]
    if split_group_batch:
        instructions.append(
            "This packet is only one batch from a larger group, so the true owner may be absent. If no row in this batch fits well, say so."
        )
    response_schema = {
        "decision": "best_candidate_in_batch | no_plausible_candidate_in_batch | multiple_plausible_candidates_in_batch",
        "best_candidate_row_indices": ["1-based batch row indices; empty if none"],
        "confidence": "low | medium | high",
        "reasoning_summary": "short explanation grounded in text-to-metadata evidence",
        "matched_signals": [
            "flat list of cues such as title phrases, author names, abstract overlap, topic overlap, or clear contradiction"
        ],
    }
    return {
        "problem_statement": problem_statement,
        "instructions": instructions,
        "response_schema": response_schema,
    }


def build_packet_rows(members: list[dict[str, Any]], start_index: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, member in enumerate(members, start=1):
        rows.append(
            {
                "row_index_in_batch": offset,
                "row_index_in_group": start_index + offset,
                "source_doc_id": member["source_doc_id"],
                "oa_collection_slug": member["oa_collection_slug"],
                "title": member["title"],
                "author": member["author"],
                "semantic_metadata": member["semantic_metadata"],
                "source_metadata": member["source_metadata"],
                "metadata_preview": member["metadata_preview"],
            }
        )
    return rows


def write_gemini_packets(
    run_root: Path,
    group_profiles: pd.DataFrame,
    grouped_members: dict[str, list[dict[str, Any]]],
    *,
    max_rows_per_packet: int,
) -> tuple[Path, pd.DataFrame]:
    packet_dir = run_root / "analysis" / "oa_gemini_resolution_packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    packet_rows: list[dict[str, Any]] = []
    packet_index: list[dict[str, Any]] = []
    eligible = group_profiles[group_profiles["semantic_review_path"].isin({"single_packet", "split_group_batches"})].copy()
    for group in eligible.to_dict(orient="records"):
        group_hash = str(group["group_hash"])
        members = grouped_members[group_hash]
        shared_text = normalize_text(members[0]["text"])
        if group["semantic_review_path"] == "single_packet":
            batches = [members]
            mode = "single_batch"
        else:
            batches = [members[i : i + max_rows_per_packet] for i in range(0, len(members), max_rows_per_packet)]
            mode = "split_group_batch"
        total_batches = len(batches)
        for batch_index, batch_members in enumerate(batches, start=1):
            packet = {
                "packet_family_id": f"{group_hash[:16]}",
                "group_hash": group_hash,
                "group_size": int(group["group_size"]),
                "packet_mode": mode,
                "batch_index": batch_index,
                "batch_count": total_batches,
                "max_rows_per_packet": max_rows_per_packet,
                "collections_in_group": str(group["collections_in_group"]),
                "suspicious_subreason": str(group["suspicious_subreason"]),
                "deterministic_resolution_status": str(group["deterministic_resolution_status"]),
                "content_size_hint": str(group["content_size_hint"]),
                "shared_text_profile": {
                    "text_chars": int(group["text_chars"]),
                    "chars_no_comments": int(group["chars_no_comments"]),
                    "chars_no_comments_tables_math": int(group["chars_no_comments_tables_math"]),
                    "chars_plain_body": int(group["chars_plain_body"]),
                    "word_count": int(group["word_count"]),
                    "header_count": int(group["header_count"]),
                    "table_line_count": int(group["table_line_count"]),
                    "latex_marker_count": int(group["latex_marker_count"]),
                    "abstract_marker_count": int(group["abstract_marker_count"]),
                    "bibliography_marker_count": int(group["bibliography_marker_count"]),
                },
                "prompt_contract": gemini_prompt_contract(split_group_batch=(mode == "split_group_batch")),
                "shared_text": shared_text,
                "rows": build_packet_rows(batch_members, start_index=(batch_index - 1) * max_rows_per_packet),
            }
            packet_name = f"{len(packet_index)+1:04d}_{group_hash[:16]}_b{batch_index:02d}.json"
            packet_path = packet_dir / packet_name
            packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2))
            packet_index.append(
                {
                    "packet_id": len(packet_index) + 1,
                    "packet_path": str(packet_path),
                    "packet_mode": mode,
                    "batch_index": batch_index,
                    "batch_count": total_batches,
                    "group_hash": group_hash,
                    "group_size": int(group["group_size"]),
                    "rows_in_packet": len(batch_members),
                    "deterministic_resolution_status": str(group["deterministic_resolution_status"]),
                    "content_size_hint": str(group["content_size_hint"]),
                    "representative_title": str(group["representative_title"]),
                }
            )
            packet_rows.append(packet_index[-1])
    index_path = packet_dir / "index.json"
    index_path.write_text(json.dumps(packet_index, ensure_ascii=False, indent=2))
    frame = pd.DataFrame(packet_rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "packet_id",
                "packet_path",
                "packet_mode",
                "batch_index",
                    "batch_count",
                    "group_hash",
                    "group_size",
                    "rows_in_packet",
                    "deterministic_resolution_status",
                    "content_size_hint",
                    "representative_title",
                ]
        )
    return index_path, frame


def build_report(
    *,
    run_id: str,
    group_profiles: pd.DataFrame,
    summary_frame: pd.DataFrame,
    packet_index_frame: pd.DataFrame,
    max_rows_per_packet: int,
) -> str:
    lines: list[str] = []
    lines.append(f"# OA Semantic Resolution Pipeline Prep: {run_id}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("- classify suspicious OA exact groups by neutral deterministic shape/status")
    lines.append("- make only strong deterministic calls and leave the rest for semantic review")
    lines.append("- prepare structured packets for later semantic review and possible Gemini owner-resolution")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- max_rows_per_packet: `{max_rows_per_packet}`")
    lines.append("")
    lines.append("## Deterministic Summary")
    for row in summary_frame.to_dict(orient="records"):
        lines.append(
            f"- {row['deterministic_resolution_status']} / {row['content_size_hint']}: "
            f"groups={int(row['group_count'])}, rows={int(row['rows_in_groups'])}, dropped_rows={int(row['implied_dropped_rows'])}"
        )
    lines.append("")
    eligible = group_profiles[group_profiles["semantic_review_path"].isin({"single_packet", "split_group_batches"})]
    split_count = int((eligible["semantic_review_path"] == "split_group_batches").sum()) if not eligible.empty else 0
    lines.append("## Semantic Review Packet Plan")
    lines.append(f"- eligible_groups: `{len(eligible)}`")
    lines.append(f"- split_groups: `{split_count}`")
    lines.append(f"- packet_files: `{len(packet_index_frame)}`")
    lines.append("")
    lines.append("## Largest Semantic-Review Groups")
    largest = eligible.sort_values(["group_size", "text_chars", "group_hash"], ascending=[False, False, True]).head(12)
    for row in largest.to_dict(orient="records"):
        lines.append(
            f"- group={row['group_hash'][:16]}..., size={int(row['group_size'])}, "
            f"status={row['deterministic_resolution_status']}, size_hint={row['content_size_hint']}, "
            f"plain_chars={int(row['chars_plain_body'])}, headers={int(row['header_count'])}, repr_title={row['representative_title']!r}"
        )
    if largest.empty:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def prepare_openarchives_semantic_resolution_pipeline(
    run_root: Path,
    *,
    output_dir: Path | None = None,
    max_rows_per_packet: int = 8,
) -> dict[str, Any]:
    summary, suspicious_groups, default_analysis_root = load_suspicious_groups(run_root)
    analysis_root = (output_dir or default_analysis_root).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)

    grouped_members = load_group_members(summary, suspicious_groups)
    group_profiles = build_group_profile_frame(suspicious_groups, grouped_members, max_rows_per_packet=max_rows_per_packet)
    summary_frame = build_summary_frame(group_profiles)
    index_path, packet_index_frame = write_gemini_packets(run_root.resolve(), group_profiles, grouped_members, max_rows_per_packet=max_rows_per_packet)

    group_profiles_path = analysis_root / "oa_semantic_resolution_group_profiles.csv"
    summary_path = analysis_root / "oa_semantic_resolution_summary.csv"
    packet_index_csv_path = analysis_root / "oa_gemini_resolution_packet_index.csv"
    report_path = analysis_root / "oa_semantic_resolution_report.md"

    write_csv(group_profiles, group_profiles_path)
    write_csv(summary_frame, summary_path)
    write_csv(packet_index_frame, packet_index_csv_path)
    report_path.write_text(
        build_report(
            run_id=str(summary["run_id"]),
            group_profiles=group_profiles,
            summary_frame=summary_frame,
            packet_index_frame=packet_index_frame,
            max_rows_per_packet=max_rows_per_packet,
        )
    )

    payload = {
        "run_id": str(summary["run_id"]),
        "analysis_root": str(analysis_root),
        "oa_semantic_resolution_group_profiles_path": str(group_profiles_path),
        "oa_semantic_resolution_summary_path": str(summary_path),
        "oa_gemini_resolution_packet_index_json_path": str(index_path),
        "oa_gemini_resolution_packet_index_csv_path": str(packet_index_csv_path),
        "oa_semantic_resolution_report_path": str(report_path),
        "max_rows_per_packet": int(max_rows_per_packet),
    }
    (analysis_root / "oa_semantic_resolution_pipeline_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic OA semantic-resolution analysis and Gemini packetization.")
    parser.add_argument("--run-root", required=True, type=Path, help="Path to a completed exact-stage run root")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory; defaults to <run_root>/analysis")
    parser.add_argument("--max-rows-per-packet", type=int, default=8, help="Maximum rows to include in one Gemini packet before splitting")
    args = parser.parse_args()
    payload = prepare_openarchives_semantic_resolution_pipeline(
        run_root=args.run_root,
        output_dir=args.output_dir,
        max_rows_per_packet=args.max_rows_per_packet,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
