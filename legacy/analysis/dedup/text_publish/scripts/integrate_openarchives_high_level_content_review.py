from __future__ import annotations

import argparse
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


def load_packets(run_root: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    packet_dir = run_root / "analysis" / "oa_llm_review_packets"
    packet_index = load_json(packet_dir / "index.json")
    by_id = {int(entry["packet_id"]): load_json(Path(entry["packet_path"])) for entry in packet_index}
    return packet_index, by_id


def build_review_frame(packet_index: list[dict[str, Any]], packets_by_id: dict[int, dict[str, Any]], review_payload: dict[str, Any]) -> pd.DataFrame:
    packet_index_map = {int(entry["packet_id"]): entry for entry in packet_index}
    rows: list[dict[str, Any]] = []
    for review in review_payload["reviews"]:
        packet_id = int(review["packet_id"])
        packet = packets_by_id[packet_id]
        packet_entry = packet_index_map[packet_id]
        rows.append(
            {
                "review_batch_id": str(review_payload["review_batch_id"]),
                "packet_id": packet_id,
                "group_hash": str(packet["group_hash"]),
                "group_size": int(packet["group_size"]),
                "collections_in_group": str(packet["collections_in_group"]),
                "packet_path": str(packet_entry["packet_path"]),
                "reviewer": str(review["reviewer"]),
                "high_level_content_type": str(review["high_level_content_type"]),
                "is_substantive_content": bool(review["is_substantive_content"]),
                "confidence": str(review["confidence"]),
                "reasoning_summary": str(review["reasoning_summary"]),
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
                "packet_path",
                "reviewer",
                "high_level_content_type",
                "is_substantive_content",
                "confidence",
                "reasoning_summary",
            ]
        )
    return frame.sort_values(["packet_id"], ascending=[True])


def build_summary_frame(review_frame: pd.DataFrame) -> pd.DataFrame:
    if review_frame.empty:
        return pd.DataFrame(columns=["high_level_content_type", "is_substantive_content", "group_count", "rows_in_groups", "implied_dropped_rows"])
    summary = review_frame.groupby(["high_level_content_type", "is_substantive_content"], as_index=False).agg(
        group_count=("group_hash", "count"),
        rows_in_groups=("group_size", "sum"),
    )
    summary["implied_dropped_rows"] = summary["rows_in_groups"] - summary["group_count"]
    return summary.sort_values(["rows_in_groups", "group_count", "high_level_content_type"], ascending=[False, False, True])


def build_report(run_id: str, review_payload: dict[str, Any], review_frame: pd.DataFrame, summary_frame: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append(f"# OA High-Level Content Review: {run_id}")
    lines.append("")
    lines.append("## Batch")
    lines.append(f"- review_batch_id: `{review_payload['review_batch_id']}`")
    lines.append(f"- reviewed_groups: `{len(review_frame)}`")
    lines.append("")
    lines.append("## Summary")
    for row in summary_frame.to_dict(orient="records"):
        lines.append(
            f"- {row['high_level_content_type']} / substantive={bool(row['is_substantive_content'])}: "
            f"groups={int(row['group_count'])}, rows={int(row['rows_in_groups'])}, dropped_rows={int(row['implied_dropped_rows'])}"
        )
    lines.append("")
    lines.append("## Packet Decisions")
    for row in review_frame.to_dict(orient="records"):
        lines.append(
            f"- packet={int(row['packet_id'])}, group={row['group_hash'][:16]}..., type={row['high_level_content_type']}, "
            f"substantive={bool(row['is_substantive_content'])}, reviewer={row['reviewer']}, confidence={row['confidence']}"
        )
        lines.append(f"  reason: {row['reasoning_summary']}")
    return "\n".join(lines) + "\n"


def integrate_openarchives_high_level_content_review(run_root: Path, review_input: Path, output_dir: Path | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    analysis_root = (output_dir or (run_root / "analysis")).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)
    packet_index, packets_by_id = load_packets(run_root)
    review_payload = load_json(review_input.resolve())
    review_frame = build_review_frame(packet_index, packets_by_id, review_payload)
    summary_frame = build_summary_frame(review_frame)

    stem = str(review_payload["review_batch_id"])
    review_csv_path = analysis_root / f"{stem}.csv"
    summary_csv_path = analysis_root / f"{stem}_summary.csv"
    report_path = analysis_root / f"{stem}_report.md"
    summary_json_path = analysis_root / f"{stem}_summary.json"

    write_csv(review_frame, review_csv_path)
    write_csv(summary_frame, summary_csv_path)
    report_path.write_text(build_report(str(run_root.name), review_payload, review_frame, summary_frame))

    payload = {
        "run_id": str(run_root.name),
        "review_batch_id": stem,
        "review_input_path": str(review_input.resolve()),
        "review_csv_path": str(review_csv_path),
        "summary_csv_path": str(summary_csv_path),
        "report_path": str(report_path),
        "reviewed_group_count": int(len(review_frame)),
    }
    summary_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge subagent high-level OA content-type review into durable analysis artifacts.")
    parser.add_argument("--run-root", required=True, type=Path, help="Path to completed exact-stage run root")
    parser.add_argument("--review-input", required=True, type=Path, help="Path to structured high-level review JSON")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory; defaults to <run_root>/analysis")
    args = parser.parse_args()
    payload = integrate_openarchives_high_level_content_review(args.run_root, args.review_input, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
