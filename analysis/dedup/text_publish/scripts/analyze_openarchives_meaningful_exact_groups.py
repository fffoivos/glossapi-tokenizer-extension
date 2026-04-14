from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from analyze_openarchives_internal_collection_patterns import load_oa_internal_groups
from sample_openarchives_internal_exact_groups import (
    build_group_summary,
    build_member_export,
    build_oa_collection_map,
    build_source_file_map,
    lookup_rows,
    load_json,
    write_csv,
)


LOW_INFORMATION_REASONS = {
    "missing_text",
    "empty",
    "image_placeholder_only",
    "html_comment_only",
    "markup_or_symbol_only",
}

LIKELY_TRUE_DUPLICATE_REASONS = {
    "literal_duplicate_same_metadata",
    "literal_duplicate_cross_collection",
    "normalization_only_text_duplicate",
    "same_title_author_text_variation",
}

LIKELY_SAME_WORK_METADATA_VARIATION_REASONS = {
    "same_text_title_author_variation",
    "same_text_author_title_variation",
    "same_title_text_duplicate",
}

LIKELY_TEXT_METADATA_ATTACHMENT_PROBLEM_REASONS = {
    "same_text_metadata_variation",
    "exact_text_duplicate_metadata_mixed",
}


def classify_meaningful_bucket(derived_reason: str) -> str:
    if derived_reason in LIKELY_TRUE_DUPLICATE_REASONS:
        return "likely_true_duplicate"
    if derived_reason in LIKELY_SAME_WORK_METADATA_VARIATION_REASONS:
        return "likely_same_work_metadata_variation"
    if derived_reason in LIKELY_TEXT_METADATA_ATTACHMENT_PROBLEM_REASONS:
        return "likely_text_metadata_attachment_problem"
    return "uncertain_meaningful_group"


def load_meaningful_group_details(run_root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, strict_groups, oa_groups = load_oa_internal_groups(run_root)
    docs_exact = pq.read_table(Path(summary["docs_exact_path"])).to_pandas()
    snapshot_manifest = pq.read_table(Path(summary["snapshot_manifest_path"])).to_pandas()

    file_map = build_source_file_map(Path(summary["snapshot_manifest_path"]))
    oa_files = file_map.get("openarchives.gr", [])
    oa_collection_map = build_oa_collection_map(oa_files)
    member_doc_ids = set(
        strict_groups.loc[
            strict_groups["group_hash"].isin(oa_groups["group_hash"])
            & (strict_groups["member_source_dataset"] == "openarchives.gr"),
            "member_source_doc_id",
        ].astype(str)
    )
    source_rows = lookup_rows(oa_files, member_doc_ids)
    all_members = build_member_export(
        oa_groups,
        docs_exact,
        snapshot_manifest,
        oa_collection_map,
        source_rows,
    )
    all_groups = build_group_summary(oa_groups, all_members)
    meaningful_groups = all_groups[~all_groups["derived_reason"].isin(LOW_INFORMATION_REASONS)].copy()
    meaningful_groups["analysis_bucket"] = meaningful_groups["derived_reason"].map(classify_meaningful_bucket)
    meaningful_members = all_members[all_members["group_hash"].isin(meaningful_groups["group_hash"])].copy()
    meaningful_members = meaningful_members.merge(
        meaningful_groups[["group_hash", "derived_reason", "analysis_bucket"]],
        how="left",
        on="group_hash",
    )
    return summary, all_groups, meaningful_groups, meaningful_members


def build_bucket_summary(meaningful_groups: pd.DataFrame) -> pd.DataFrame:
    if meaningful_groups.empty:
        return pd.DataFrame(columns=["analysis_bucket", "group_count", "rows_in_groups", "kept_rows", "dropped_rows"])
    summary = meaningful_groups.groupby("analysis_bucket", as_index=False).agg(
        group_count=("group_hash", "count"),
        rows_in_groups=("group_size", "sum"),
    )
    summary["kept_rows"] = summary["group_count"]
    summary["dropped_rows"] = summary["rows_in_groups"] - summary["kept_rows"]
    return summary.sort_values(["rows_in_groups", "group_count", "analysis_bucket"], ascending=[False, False, True])


def build_reason_summary(meaningful_groups: pd.DataFrame) -> pd.DataFrame:
    if meaningful_groups.empty:
        return pd.DataFrame(
            columns=["analysis_bucket", "derived_reason", "group_count", "rows_in_groups", "kept_rows", "dropped_rows"]
        )
    summary = meaningful_groups.groupby(["analysis_bucket", "derived_reason"], as_index=False).agg(
        group_count=("group_hash", "count"),
        rows_in_groups=("group_size", "sum"),
    )
    summary["kept_rows"] = summary["group_count"]
    summary["dropped_rows"] = summary["rows_in_groups"] - summary["kept_rows"]
    return summary.sort_values(
        ["rows_in_groups", "group_count", "analysis_bucket", "derived_reason"],
        ascending=[False, False, True, True],
    )


def build_collection_bucket_summary(meaningful_members: pd.DataFrame) -> pd.DataFrame:
    if meaningful_members.empty:
        return pd.DataFrame(
            columns=["analysis_bucket", "oa_collection_slug", "rows_in_groups", "kept_rows", "dropped_rows", "groups_with_collection"]
        )
    summary = meaningful_members.groupby(["analysis_bucket", "oa_collection_slug"], as_index=False).agg(
        rows_in_groups=("source_doc_id", "count"),
        kept_rows=("is_kept", lambda s: int(s.sum())),
        dropped_rows=("is_kept", lambda s: int((~s).sum())),
        groups_with_collection=("group_hash", "nunique"),
    )
    return summary.sort_values(
        ["analysis_bucket", "rows_in_groups", "dropped_rows", "oa_collection_slug"],
        ascending=[True, False, False, True],
    )


def build_cross_collection_bucket_summary(meaningful_members: pd.DataFrame) -> pd.DataFrame:
    pair_rows: list[dict[str, Any]] = []
    for (_, bucket), grp in meaningful_members.groupby(["group_hash", "analysis_bucket"]):
        counts = grp["oa_collection_slug"].value_counts().to_dict()
        slugs = sorted(counts)
        if len(slugs) < 2:
            continue
        for i, source_a in enumerate(slugs):
            for source_b in slugs[i + 1 :]:
                pair_rows.append(
                    {
                        "analysis_bucket": bucket,
                        "source_a": source_a,
                        "source_b": source_b,
                        "group_count": 1,
                        "pair_rows_in_groups": counts[source_a] + counts[source_b],
                    }
                )
    if not pair_rows:
        return pd.DataFrame(columns=["analysis_bucket", "source_a", "source_b", "group_count", "pair_rows_in_groups"])
    summary = pd.DataFrame(pair_rows).groupby(["analysis_bucket", "source_a", "source_b"], as_index=False).sum()
    return summary.sort_values(
        ["analysis_bucket", "group_count", "pair_rows_in_groups", "source_a", "source_b"],
        ascending=[True, False, False, True, True],
    )


def build_bucket_examples(meaningful_groups: pd.DataFrame, *, per_bucket: int = 12) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for bucket, grp in meaningful_groups.groupby("analysis_bucket"):
        sample = grp.sort_values(
            ["group_size", "unique_title_count", "unique_author_count", "group_hash"],
            ascending=[False, False, False, True],
        ).head(per_bucket)
        frames.append(sample)
    if not frames:
        return pd.DataFrame(columns=list(meaningful_groups.columns))
    return pd.concat(frames, ignore_index=True)


def build_report(
    *,
    run_id: str,
    total_oa_groups: int,
    excluded_noise_groups: int,
    excluded_noise_rows: int,
    bucket_summary: pd.DataFrame,
    reason_summary: pd.DataFrame,
    collection_bucket_summary: pd.DataFrame,
    cross_bucket_summary: pd.DataFrame,
    examples: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"# OA Meaningful Exact Groups: {run_id}")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- OA-only strict exact groups total: {total_oa_groups}")
    lines.append(f"- low-information groups excluded as noise: {excluded_noise_groups}")
    lines.append(f"- OA rows in excluded noise groups: {excluded_noise_rows}")
    lines.append("")
    lines.append("## Bucket Definitions")
    lines.append("- likely_true_duplicate: same work with matching title/author or only normalization-format differences")
    lines.append("- likely_same_work_metadata_variation: same exact text, but one metadata field varies")
    lines.append("- likely_text_metadata_attachment_problem: same exact text, but both title and author conflict or metadata is otherwise strongly mixed")
    lines.append("- uncertain_meaningful_group: meaningful exact group that did not fit the above rules cleanly")
    lines.append("")
    lines.append("## Bucket Summary")
    for row in bucket_summary.to_dict(orient="records"):
        lines.append(
            f"- {row['analysis_bucket']}: groups={int(row['group_count'])}, rows={int(row['rows_in_groups'])}, dropped_rows={int(row['dropped_rows'])}"
        )
    lines.append("")
    lines.append("## Reason Summary")
    for row in reason_summary.to_dict(orient="records"):
        lines.append(
            f"- {row['analysis_bucket']} / {row['derived_reason']}: groups={int(row['group_count'])}, rows={int(row['rows_in_groups'])}, dropped_rows={int(row['dropped_rows'])}"
        )
    lines.append("")
    lines.append("## Most Affected Collections By Bucket")
    for bucket in bucket_summary["analysis_bucket"].tolist():
        lines.append(f"### {bucket}")
        subset = collection_bucket_summary[collection_bucket_summary["analysis_bucket"] == bucket].head(10)
        for row in subset.to_dict(orient="records"):
            lines.append(
                f"- {row['oa_collection_slug']}: rows={int(row['rows_in_groups'])}, dropped_rows={int(row['dropped_rows'])}, groups={int(row['groups_with_collection'])}"
            )
        if subset.empty:
            lines.append("- none")
    lines.append("")
    lines.append("## Top Cross-Collection Pairs By Bucket")
    for bucket in bucket_summary["analysis_bucket"].tolist():
        lines.append(f"### {bucket}")
        subset = cross_bucket_summary[cross_bucket_summary["analysis_bucket"] == bucket].head(10)
        for row in subset.to_dict(orient="records"):
            lines.append(
                f"- {row['source_a']} <-> {row['source_b']}: groups={int(row['group_count'])}, rows={int(row['pair_rows_in_groups'])}"
            )
        if subset.empty:
            lines.append("- none")
    lines.append("")
    lines.append("## Largest Example Groups")
    for bucket in bucket_summary["analysis_bucket"].tolist():
        lines.append(f"### {bucket}")
        subset = examples[examples["analysis_bucket"] == bucket].head(8)
        for row in subset.to_dict(orient="records"):
            lines.append(
                f"- group={row['group_hash'][:16]}..., size={int(row['group_size'])}, reason={row['derived_reason']}, "
                f"collections={row['collections_in_group']}, titles={int(row['unique_title_count'])}, "
                f"authors={int(row['unique_author_count'])}, raw_hashes={int(row['unique_raw_content_hash_count'])}, "
                f"repr_title={row['representative_title']!r}"
            )
        if subset.empty:
            lines.append("- none")
    return "\n".join(lines) + "\n"


def analyze_openarchives_meaningful_exact_groups(run_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    analysis_root = (output_dir or (run_root / "analysis")).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)

    summary, all_groups, meaningful_groups, meaningful_members = load_meaningful_group_details(run_root)
    bucket_summary = build_bucket_summary(meaningful_groups)
    reason_summary = build_reason_summary(meaningful_groups)
    collection_bucket_summary = build_collection_bucket_summary(meaningful_members)
    cross_bucket_summary = build_cross_collection_bucket_summary(meaningful_members)
    examples = build_bucket_examples(meaningful_groups)

    excluded_noise = all_groups[all_groups["derived_reason"].isin(LOW_INFORMATION_REASONS)].copy()

    group_details_path = analysis_root / "oa_meaningful_exact_group_details.csv"
    members_path = analysis_root / "oa_meaningful_exact_members.csv"
    bucket_summary_path = analysis_root / "oa_meaningful_exact_bucket_summary.csv"
    reason_summary_path = analysis_root / "oa_meaningful_exact_reason_summary.csv"
    collection_bucket_summary_path = analysis_root / "oa_meaningful_exact_collection_bucket_summary.csv"
    cross_bucket_summary_path = analysis_root / "oa_meaningful_exact_cross_collection_bucket_summary.csv"
    examples_path = analysis_root / "oa_meaningful_exact_examples.csv"
    report_path = analysis_root / "oa_meaningful_exact_report.md"

    write_csv(meaningful_groups, group_details_path)
    write_csv(meaningful_members, members_path)
    write_csv(bucket_summary, bucket_summary_path)
    write_csv(reason_summary, reason_summary_path)
    write_csv(collection_bucket_summary, collection_bucket_summary_path)
    write_csv(cross_bucket_summary, cross_bucket_summary_path)
    write_csv(examples, examples_path)
    report_path.write_text(
        build_report(
            run_id=str(summary["run_id"]),
            total_oa_groups=int(len(all_groups)),
            excluded_noise_groups=int(len(excluded_noise)),
            excluded_noise_rows=int(excluded_noise["group_size"].sum()) if not excluded_noise.empty else 0,
            bucket_summary=bucket_summary,
            reason_summary=reason_summary,
            collection_bucket_summary=collection_bucket_summary,
            cross_bucket_summary=cross_bucket_summary,
            examples=examples,
        )
    )

    payload = {
        "run_id": str(summary["run_id"]),
        "analysis_root": str(analysis_root),
        "oa_meaningful_exact_group_details_path": str(group_details_path),
        "oa_meaningful_exact_members_path": str(members_path),
        "oa_meaningful_exact_bucket_summary_path": str(bucket_summary_path),
        "oa_meaningful_exact_reason_summary_path": str(reason_summary_path),
        "oa_meaningful_exact_collection_bucket_summary_path": str(collection_bucket_summary_path),
        "oa_meaningful_exact_cross_collection_bucket_summary_path": str(cross_bucket_summary_path),
        "oa_meaningful_exact_examples_path": str(examples_path),
        "oa_meaningful_exact_report_path": str(report_path),
    }
    (analysis_root / "oa_meaningful_exact_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify the non-noise OA exact duplicate groups into policy buckets.")
    parser.add_argument("--run-root", required=True, type=Path, help="Path to a completed exact-stage run root")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory; defaults to <run_root>/analysis")
    args = parser.parse_args()
    payload = analyze_openarchives_meaningful_exact_groups(args.run_root, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
