from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


csv.field_size_limit(10**9)

WHITESPACE_RE = re.compile(r"\s+")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        df = pd.DataFrame(columns=df.columns)
    df.to_csv(path, index=False)


def normalize_preview(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value)).casefold()
    return WHITESPACE_RE.sub(" ", text).strip()


def classify_suspicious_subreason(text_preview: str | None) -> str:
    preview = str(text_preview or "")
    normalized = normalize_preview(preview)

    if (
        "πρόσβαση" in normalized
        and ("συγγραφέ" in normalized or "συγγραφε" in normalized)
        and ("διατριβ" in normalized or "μεταπτυχια" in normalized or "πλήρες κείμενο" in normalized or "πληρες κειμενο" in normalized)
    ):
        if "δεν επιτρέπει" in normalized:
            return "author_denies_access_notice"
        if "επιτρέπει" in normalized and "από" in normalized:
            return "embargo_release_notice"
        return "author_access_notice"
    if "η πρόσβαση στην ηλεκτρονική" in normalized and "δεν είναι δυνατή" in normalized:
        return "electronic_access_unavailable_notice"
    if "η πρόσβαση στη συγκεκριμένη εργασία" in normalized and "δεν είναι δυνατή" in normalized:
        return "specific_work_unavailable_notice"
    if "<!-- image -->" in preview or "<img" in preview:
        return "image_markup_mixed_text"
    if "$$" in preview or "\\begin{" in preview:
        return "latex_or_formula_stub"
    return "other_content_collision"


def load_suspicious_groups(run_root: Path) -> tuple[dict[str, Any], pd.DataFrame, Path]:
    run_root = run_root.resolve()
    analysis_root = run_root / "analysis"
    summary_path = analysis_root / "oa_meaningful_exact_summary.json"
    summary = load_json(summary_path)
    group_details_path = Path(summary["oa_meaningful_exact_group_details_path"])
    groups = pd.read_csv(group_details_path)
    groups = groups[groups["analysis_bucket"] == "likely_text_metadata_attachment_problem"].copy()
    groups["suspicious_subreason"] = groups["representative_text_preview"].map(classify_suspicious_subreason)
    return summary, groups, analysis_root


def build_subreason_summary(groups: pd.DataFrame) -> pd.DataFrame:
    if groups.empty:
        return pd.DataFrame(columns=["suspicious_subreason", "group_count", "rows_in_groups", "dropped_rows"])
    summary = groups.groupby("suspicious_subreason", as_index=False).agg(
        group_count=("group_hash", "count"),
        rows_in_groups=("group_size", "sum"),
    )
    summary["dropped_rows"] = summary["rows_in_groups"] - summary["group_count"]
    return summary.sort_values(["rows_in_groups", "group_count", "suspicious_subreason"], ascending=[False, False, True])


def build_collection_summary(groups: pd.DataFrame, members_path: Path) -> pd.DataFrame:
    wanted = set(groups["group_hash"].astype(str))
    group_to_subreason = dict(zip(groups["group_hash"], groups["suspicious_subreason"]))
    rows: list[dict[str, Any]] = []
    with members_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            group_hash = str(row["group_hash"])
            if group_hash not in wanted:
                continue
            rows.append(
                {
                    "group_hash": group_hash,
                    "oa_collection_slug": row["oa_collection_slug"],
                    "is_kept": str(row["is_kept"]).lower() == "true",
                    "suspicious_subreason": group_to_subreason[group_hash],
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["suspicious_subreason", "oa_collection_slug", "rows_in_groups", "dropped_rows", "groups_with_collection"]
        )
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["suspicious_subreason", "oa_collection_slug"], as_index=False).agg(
        rows_in_groups=("group_hash", "count"),
        dropped_rows=("is_kept", lambda s: int((~s).sum())),
        groups_with_collection=("group_hash", "nunique"),
    )
    return summary.sort_values(
        ["suspicious_subreason", "rows_in_groups", "dropped_rows", "oa_collection_slug"],
        ascending=[True, False, False, True],
    )


def build_examples(groups: pd.DataFrame, *, per_subreason: int = 12) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for subreason, grp in groups.groupby("suspicious_subreason"):
        frames.append(
            grp.sort_values(
                ["group_size", "collection_count", "unique_title_count", "group_hash"],
                ascending=[False, False, False, True],
            ).head(per_subreason)
        )
    if not frames:
        return pd.DataFrame(columns=list(groups.columns))
    return pd.concat(frames, ignore_index=True)


def build_member_examples(groups: pd.DataFrame, members_path: Path, *, max_groups_per_subreason: int = 5) -> pd.DataFrame:
    chosen_groups = (
        groups.sort_values(["group_size", "group_hash"], ascending=[False, True])
        .groupby("suspicious_subreason", as_index=False, group_keys=False)
        .head(max_groups_per_subreason)
    )
    wanted = set(chosen_groups["group_hash"].astype(str))
    group_to_subreason = dict(zip(chosen_groups["group_hash"], chosen_groups["suspicious_subreason"]))
    rows: list[dict[str, Any]] = []
    with members_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            group_hash = str(row["group_hash"])
            if group_hash not in wanted:
                continue
            rows.append(
                {
                    "suspicious_subreason": group_to_subreason[group_hash],
                    "group_hash": group_hash,
                    "oa_collection_slug": row["oa_collection_slug"],
                    "is_kept": row["is_kept"],
                    "source_doc_id": row["source_doc_id"],
                    "title": row["title"],
                    "author": row["author"],
                    "text_preview": row["text_preview"],
                    "metadata_preview": row["metadata_preview"],
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["suspicious_subreason", "group_hash", "oa_collection_slug", "is_kept", "source_doc_id", "title", "author", "text_preview", "metadata_preview"])
    return frame.sort_values(["suspicious_subreason", "group_hash", "is_kept", "oa_collection_slug", "source_doc_id"], ascending=[True, True, False, True, True])


def build_report(
    *,
    run_id: str,
    subreason_summary: pd.DataFrame,
    collection_summary: pd.DataFrame,
    examples: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"# OA Suspicious Exact Groups: {run_id}")
    lines.append("")
    lines.append("## Definitions")
    lines.append("- suspicious groups here are the OA exact-match groups previously labeled `likely_text_metadata_attachment_problem`")
    lines.append("- this report splits that bucket into concrete text-pattern subreasons")
    lines.append("")
    lines.append("## Subreason Summary")
    for row in subreason_summary.to_dict(orient="records"):
        lines.append(
            f"- {row['suspicious_subreason']}: groups={int(row['group_count'])}, rows={int(row['rows_in_groups'])}, dropped_rows={int(row['dropped_rows'])}"
        )
    lines.append("")
    lines.append("## Most Affected Collections By Subreason")
    for subreason in subreason_summary["suspicious_subreason"].tolist():
        lines.append(f"### {subreason}")
        subset = collection_summary[collection_summary["suspicious_subreason"] == subreason].head(10)
        for row in subset.to_dict(orient="records"):
            lines.append(
                f"- {row['oa_collection_slug']}: rows={int(row['rows_in_groups'])}, dropped_rows={int(row['dropped_rows'])}, groups={int(row['groups_with_collection'])}"
            )
        if subset.empty:
            lines.append("- none")
    lines.append("")
    lines.append("## Largest Example Groups")
    for subreason in subreason_summary["suspicious_subreason"].tolist():
        lines.append(f"### {subreason}")
        subset = examples[examples["suspicious_subreason"] == subreason].head(8)
        for row in subset.to_dict(orient="records"):
            lines.append(
                f"- group={row['group_hash'][:16]}..., size={int(row['group_size'])}, collections={row['collections_in_group']}, "
                f"titles={int(row['unique_title_count'])}, authors={int(row['unique_author_count'])}, repr_title={row['representative_title']!r}"
            )
        if subset.empty:
            lines.append("- none")
    return "\n".join(lines) + "\n"


def analyze_openarchives_suspicious_exact_groups(run_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    summary, groups, default_analysis_root = load_suspicious_groups(run_root)
    analysis_root = (output_dir or default_analysis_root).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)

    members_path = Path(summary["oa_meaningful_exact_members_path"])
    subreason_summary = build_subreason_summary(groups)
    collection_summary = build_collection_summary(groups, members_path)
    examples = build_examples(groups)
    member_examples = build_member_examples(groups, members_path)

    subreason_summary_path = analysis_root / "oa_suspicious_exact_subreason_summary.csv"
    collection_summary_path = analysis_root / "oa_suspicious_exact_collection_summary.csv"
    examples_path = analysis_root / "oa_suspicious_exact_examples.csv"
    member_examples_path = analysis_root / "oa_suspicious_exact_member_examples.csv"
    report_path = analysis_root / "oa_suspicious_exact_report.md"

    write_csv(subreason_summary, subreason_summary_path)
    write_csv(collection_summary, collection_summary_path)
    write_csv(examples, examples_path)
    write_csv(member_examples, member_examples_path)
    report_path.write_text(
        build_report(
            run_id=str(summary["run_id"]),
            subreason_summary=subreason_summary,
            collection_summary=collection_summary,
            examples=examples,
        )
    )

    payload = {
        "run_id": str(summary["run_id"]),
        "analysis_root": str(analysis_root),
        "oa_suspicious_exact_subreason_summary_path": str(subreason_summary_path),
        "oa_suspicious_exact_collection_summary_path": str(collection_summary_path),
        "oa_suspicious_exact_examples_path": str(examples_path),
        "oa_suspicious_exact_member_examples_path": str(member_examples_path),
        "oa_suspicious_exact_report_path": str(report_path),
    }
    (analysis_root / "oa_suspicious_exact_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Split OA suspicious exact groups into concrete text-pattern subreasons.")
    parser.add_argument("--run-root", required=True, type=Path, help="Path to a completed exact-stage run root")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory; defaults to <run_root>/analysis")
    args = parser.parse_args()
    payload = analyze_openarchives_suspicious_exact_groups(args.run_root, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
