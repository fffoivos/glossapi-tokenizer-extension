from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        path.write_text("")
        return
    frame.to_csv(path, index=False)


def load_packets(run_root: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    packet_dir = run_root / "analysis" / "oa_llm_review_packets"
    index_path = packet_dir / "index.json"
    packet_index = load_json(index_path)
    by_id: dict[int, dict[str, Any]] = {}
    by_group_hash: dict[str, dict[str, Any]] = {}
    for entry in packet_index:
        packet = load_json(Path(entry["packet_path"]))
        by_id[int(packet["packet_id"])] = packet
        by_group_hash[str(packet["group_hash"])] = packet
    return packet_index, by_id, by_group_hash


def load_heuristic_audit(run_root: Path) -> dict[str, dict[str, str]]:
    path = run_root / "analysis" / "oa_suspicious_origin_group_audit.csv"
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return {str(row["group_hash"]): row for row in reader}


def resolve_candidate_member(packet: dict[str, Any], review: dict[str, Any]) -> dict[str, Any] | None:
    row_index = review.get("best_candidate_row_index")
    if row_index is None:
        return None
    index = int(row_index)
    members = packet.get("members", [])
    if index < 1 or index > len(members):
        raise ValueError(f"packet {packet['packet_id']} candidate row index {index} out of range")
    return dict(members[index - 1])


def normalize_resolution(value: str) -> str:
    allowed = {
        "no_plausible_in_group_owner",
        "best_candidate_in_group_owner",
    }
    if value not in allowed:
        raise ValueError(f"unsupported llm_owner_resolution: {value}")
    return value


def heuristic_owner_resolution(row: dict[str, str]) -> str:
    if row.get("origin_resolution") == "likely_true_file_in_group":
        return "best_candidate_in_group_owner"
    return "no_plausible_in_group_owner"


def build_review_frame(
    packet_index: list[dict[str, Any]],
    packets_by_id: dict[int, dict[str, Any]],
    heuristic_by_group: dict[str, dict[str, str]],
    review_payload: dict[str, Any],
) -> pd.DataFrame:
    packet_index_map = {int(entry["packet_id"]): entry for entry in packet_index}
    rows: list[dict[str, Any]] = []
    for review in review_payload["reviews"]:
        packet_id = int(review["packet_id"])
        packet = packets_by_id[packet_id]
        packet_index_entry = packet_index_map[packet_id]
        heuristic = heuristic_by_group[str(packet["group_hash"])]
        llm_resolution = normalize_resolution(str(review["llm_owner_resolution"]))
        candidate = resolve_candidate_member(packet, review)
        heuristic_resolution = heuristic_owner_resolution(heuristic)
        heuristic_agrees_owner_resolution = heuristic_resolution == llm_resolution
        heuristic_candidate_agreement = False
        if llm_resolution == "best_candidate_in_group_owner" and candidate is not None:
            heuristic_candidate_agreement = (
                str(heuristic.get("best_candidate_source_doc_id") or "") == str(candidate.get("source_doc_id") or "")
            )
        rows.append(
            {
                "review_batch_id": str(review_payload["review_batch_id"]),
                "packet_id": packet_id,
                "group_hash": str(packet["group_hash"]),
                "group_size": int(packet["group_size"]),
                "collections_in_group": str(packet["collections_in_group"]),
                "suspicious_subreason": str(packet["suspicious_subreason"]),
                "packet_path": str(packet_index_entry["packet_path"]),
                "heuristic_origin_resolution": str(heuristic["origin_resolution"]),
                "heuristic_best_candidate_source_doc_id": str(heuristic.get("best_candidate_source_doc_id") or ""),
                "heuristic_best_candidate_title": str(heuristic.get("best_candidate_title") or ""),
                "llm_reviewer": str(review["reviewer"]),
                "llm_owner_resolution": llm_resolution,
                "llm_shared_text_type": str(review["llm_shared_text_type"]),
                "llm_confidence": str(review["confidence"]),
                "llm_best_candidate_row_index": int(review["best_candidate_row_index"]) if review.get("best_candidate_row_index") is not None else "",
                "llm_best_candidate_source_doc_id": str(candidate.get("source_doc_id") or "") if candidate else "",
                "llm_best_candidate_collection": str(candidate.get("oa_collection_slug") or "") if candidate else "",
                "llm_best_candidate_title": str(candidate.get("title") or "") if candidate else "",
                "llm_best_candidate_author": str(candidate.get("author") or "") if candidate else "",
                "llm_reasoning_summary": str(review["reasoning_summary"]),
                "heuristic_agrees_owner_resolution": heuristic_agrees_owner_resolution,
                "heuristic_agrees_candidate": heuristic_candidate_agreement,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "review_batch_id",
                "packet_id",
                "group_hash",
                "group_size",
                "collections_in_group",
                "suspicious_subreason",
                "packet_path",
                "heuristic_origin_resolution",
                "heuristic_best_candidate_source_doc_id",
                "heuristic_best_candidate_title",
                "llm_reviewer",
                "llm_owner_resolution",
                "llm_shared_text_type",
                "llm_confidence",
                "llm_best_candidate_row_index",
                "llm_best_candidate_source_doc_id",
                "llm_best_candidate_collection",
                "llm_best_candidate_title",
                "llm_best_candidate_author",
                "llm_reasoning_summary",
                "heuristic_agrees_owner_resolution",
                "heuristic_agrees_candidate",
            ]
        )
    return frame.sort_values(["packet_id"], ascending=[True])


def build_summary_frame(review_frame: pd.DataFrame) -> pd.DataFrame:
    if review_frame.empty:
        return pd.DataFrame(
            columns=[
                "llm_owner_resolution",
                "llm_shared_text_type",
                "group_count",
                "rows_in_groups",
                "implied_dropped_rows",
                "heuristic_resolution_matches",
                "heuristic_resolution_mismatches",
            ]
        )
    summary = review_frame.groupby(["llm_owner_resolution", "llm_shared_text_type"], as_index=False).agg(
        group_count=("group_hash", "count"),
        rows_in_groups=("group_size", "sum"),
        heuristic_resolution_matches=("heuristic_agrees_owner_resolution", "sum"),
    )
    summary["implied_dropped_rows"] = summary["rows_in_groups"] - summary["group_count"]
    summary["heuristic_resolution_mismatches"] = summary["group_count"] - summary["heuristic_resolution_matches"]
    return summary.sort_values(["rows_in_groups", "group_count", "llm_owner_resolution"], ascending=[False, False, True])


def build_report(run_id: str, review_payload: dict[str, Any], review_frame: pd.DataFrame, summary_frame: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append(f"# OA LLM Semantic Review: {run_id}")
    lines.append("")
    lines.append("## Batch")
    lines.append(f"- review_batch_id: `{review_payload['review_batch_id']}`")
    lines.append(f"- reviewed_groups: `{len(review_frame)}`")
    lines.append("")
    lines.append("## Summary")
    for row in summary_frame.to_dict(orient="records"):
        lines.append(
            f"- {row['llm_owner_resolution']} / {row['llm_shared_text_type']}: "
            f"groups={int(row['group_count'])}, rows={int(row['rows_in_groups'])}, dropped_rows={int(row['implied_dropped_rows'])}, "
            f"heuristic_matches={int(row['heuristic_resolution_matches'])}, heuristic_mismatches={int(row['heuristic_resolution_mismatches'])}"
        )
    lines.append("")
    mismatch_rows = review_frame[~review_frame["heuristic_agrees_owner_resolution"]]
    lines.append("## Heuristic Mismatches")
    if mismatch_rows.empty:
        lines.append("- none")
    else:
        for row in mismatch_rows.to_dict(orient="records"):
            lines.append(
                f"- packet={int(row['packet_id'])}, group={row['group_hash'][:16]}..., heuristic={row['heuristic_origin_resolution']}, "
                f"llm={row['llm_owner_resolution']}, shared_text_type={row['llm_shared_text_type']}, "
                f"llm_best_candidate={row['llm_best_candidate_title']!r}"
            )
    lines.append("")
    lines.append("## Packet Decisions")
    for row in review_frame.to_dict(orient="records"):
        candidate_part = ""
        if row["llm_owner_resolution"] == "best_candidate_in_group_owner":
            candidate_part = f", best_candidate={row['llm_best_candidate_title']!r}"
        lines.append(
            f"- packet={int(row['packet_id'])}, group={row['group_hash'][:16]}..., resolution={row['llm_owner_resolution']}, "
            f"text_type={row['llm_shared_text_type']}, reviewer={row['llm_reviewer']}, confidence={row['llm_confidence']}{candidate_part}"
        )
        lines.append(f"  reason: {row['llm_reasoning_summary']}")
    return "\n".join(lines) + "\n"


def integrate_openarchives_llm_semantic_review(
    run_root: Path,
    review_input: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    review_input = review_input.resolve()
    analysis_root = (output_dir or (run_root / "analysis")).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)

    packet_index, packets_by_id, _ = load_packets(run_root)
    heuristic_by_group = load_heuristic_audit(run_root)
    review_payload = load_json(review_input)
    review_frame = build_review_frame(packet_index, packets_by_id, heuristic_by_group, review_payload)
    summary_frame = build_summary_frame(review_frame)

    stem = str(review_payload["review_batch_id"])
    review_csv_path = analysis_root / f"{stem}.csv"
    summary_csv_path = analysis_root / f"{stem}_summary.csv"
    report_path = analysis_root / f"{stem}_report.md"
    json_path = analysis_root / f"{stem}_summary.json"

    write_csv(review_frame, review_csv_path)
    write_csv(summary_frame, summary_csv_path)
    report_path.write_text(build_report(str(run_root.name), review_payload, review_frame, summary_frame))

    payload = {
        "run_id": str(run_root.name),
        "review_batch_id": stem,
        "review_input_path": str(review_input),
        "review_csv_path": str(review_csv_path),
        "summary_csv_path": str(summary_csv_path),
        "report_path": str(report_path),
        "reviewed_group_count": int(len(review_frame)),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LLM semantic review judgments for OA exact groups into analysis artifacts.")
    parser.add_argument("--run-root", required=True, type=Path, help="Path to completed exact-stage run root")
    parser.add_argument("--review-input", required=True, type=Path, help="Path to structured LLM review input JSON")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory; defaults to <run_root>/analysis")
    args = parser.parse_args()
    payload = integrate_openarchives_llm_semantic_review(args.run_root, args.review_input, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
