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

from sample_openarchives_internal_exact_groups import (
    build_oa_collection_map,
    build_source_file_map,
    classify_low_information_text,
    load_json,
    lookup_rows,
    write_csv,
)


def load_oa_internal_groups(run_root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary = load_json(run_root / "stage_01_exact" / "summary.json")
    strict_groups = pq.read_table(Path(summary["strict_exact_groups_path"])).to_pandas()
    per_group = strict_groups.groupby("group_hash", as_index=False).agg(
        group_size=("group_size", "max"),
        source_count=("member_source_dataset", "nunique"),
        oa_rows=("member_source_dataset", lambda s: int((s == "openarchives.gr").sum())),
    )
    oa_groups = per_group[(per_group["source_count"] == 1) & (per_group["oa_rows"] > 1)].copy()
    kept_map = strict_groups[["group_hash", "kept_doc_key", "member_doc_key", "member_source_doc_id"]].drop_duplicates()
    kept_map = kept_map[kept_map["member_doc_key"] == kept_map["kept_doc_key"]][
        ["group_hash", "kept_doc_key", "member_source_doc_id"]
    ].rename(columns={"member_source_doc_id": "kept_source_doc_id"})
    oa_groups = oa_groups.merge(kept_map, how="left", on="group_hash")
    return summary, strict_groups, oa_groups


def load_collection_members(summary: dict[str, Any], strict_groups: pd.DataFrame, oa_groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_map = build_source_file_map(Path(summary["snapshot_manifest_path"]))
    oa_files = file_map.get("openarchives.gr", [])
    oa_collection_map = build_oa_collection_map(oa_files)

    representative_rows = lookup_rows(oa_files, set(oa_groups["kept_source_doc_id"].dropna().astype(str)))
    representative_rows["low_information_reason"] = representative_rows["text"].map(classify_low_information_text)
    oa_groups = oa_groups.merge(
        representative_rows[["source_doc_id", "low_information_reason"]],
        how="left",
        left_on="kept_source_doc_id",
        right_on="source_doc_id",
    ).drop(columns=["source_doc_id"], errors="ignore")

    members = strict_groups[strict_groups["group_hash"].isin(oa_groups["group_hash"])].copy()
    members = members[members["member_source_dataset"] == "openarchives.gr"][
        ["group_hash", "member_source_doc_id"]
    ]
    members = members.merge(oa_collection_map, how="left", left_on="member_source_doc_id", right_on="source_doc_id")
    members["oa_collection_slug"] = members["oa_collection_slug"].fillna("unknown")
    return oa_groups, members


def build_collection_summary(oa_groups: pd.DataFrame, members: pd.DataFrame, *, meaningful_only: bool) -> pd.DataFrame:
    target_groups = oa_groups[oa_groups["low_information_reason"].isna()].copy() if meaningful_only else oa_groups.copy()
    if target_groups.empty:
        return pd.DataFrame(
            columns=[
                "oa_collection_slug",
                "member_rows",
                "group_memberships",
                "groups_with_collection",
            ]
        )

    target_members = members[members["group_hash"].isin(target_groups["group_hash"])].copy()
    if target_members.empty:
        return pd.DataFrame(columns=["oa_collection_slug", "member_rows", "group_memberships", "groups_with_collection"])

    summary = target_members.groupby("oa_collection_slug", as_index=False).agg(
        member_rows=("member_source_doc_id", "count"),
        groups_with_collection=("group_hash", "nunique"),
    )
    summary["group_memberships"] = summary["member_rows"]
    return summary.sort_values(["member_rows", "groups_with_collection", "oa_collection_slug"], ascending=[False, False, True])


def build_pair_summary(oa_groups: pd.DataFrame, members: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meaningful_groups = set(oa_groups.loc[oa_groups["low_information_reason"].isna(), "group_hash"].astype(str))
    target_members = members[members["group_hash"].isin(meaningful_groups)].copy()
    pair_rows: list[dict[str, Any]] = []
    for _, grp in target_members.groupby("group_hash"):
        counts = grp["oa_collection_slug"].value_counts().to_dict()
        slugs = sorted(counts)
        for i, source_a in enumerate(slugs):
            for source_b in slugs[i:]:
                pair_rows.append(
                    {
                        "source_a": source_a,
                        "source_b": source_b,
                        "group_count": 1,
                        "member_rows": counts[source_a] if source_a == source_b else counts[source_a] + counts[source_b],
                    }
                )
    if not pair_rows:
        empty = pd.DataFrame(columns=["source_a", "source_b", "group_count", "member_rows"])
        return empty, empty

    pair_df = pd.DataFrame(pair_rows).groupby(["source_a", "source_b"], as_index=False).sum()
    cross = pair_df[pair_df["source_a"] != pair_df["source_b"]].copy()
    within = pair_df[pair_df["source_a"] == pair_df["source_b"]].copy()
    within = within.rename(columns={"source_a": "oa_collection_slug"}).drop(columns=["source_b"])
    cross = cross.sort_values(["group_count", "member_rows", "source_a", "source_b"], ascending=[False, False, True, True])
    within = within.sort_values(["group_count", "member_rows", "oa_collection_slug"], ascending=[False, False, True])
    return cross, within


def build_reason_summary(oa_groups: pd.DataFrame) -> pd.DataFrame:
    frame = oa_groups.copy()
    frame["reason_bucket"] = frame["low_information_reason"].fillna("meaningful_or_mixed")
    summary = frame.groupby("reason_bucket", as_index=False).agg(
        group_count=("group_hash", "count"),
        member_rows=("group_size", "sum"),
    )
    return summary.sort_values(["member_rows", "group_count", "reason_bucket"], ascending=[False, False, True])


def build_report(
    *,
    run_id: str,
    population_size: int,
    collection_summary_all: pd.DataFrame,
    collection_summary_meaningful: pd.DataFrame,
    cross_pairs: pd.DataFrame,
    within_pairs: pd.DataFrame,
    reason_summary: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"# OA Internal Exact Collection Patterns: {run_id}")
    lines.append("")
    lines.append("## Population")
    lines.append(f"- OA-only strict exact groups: {population_size}")
    lines.append("")
    lines.append("## Reason Mix")
    for row in reason_summary.to_dict(orient="records"):
        lines.append(
            f"- {row['reason_bucket']}: groups={int(row['group_count'])}, member_rows={int(row['member_rows'])}"
        )
    lines.append("")
    lines.append("## Most Affected Collections")
    for row in collection_summary_all.head(15).to_dict(orient="records"):
        lines.append(
            f"- {row['oa_collection_slug']}: member_rows={int(row['member_rows'])}, groups={int(row['groups_with_collection'])}"
        )
    lines.append("")
    lines.append("## Most Affected Collections In Meaningful Groups")
    for row in collection_summary_meaningful.head(15).to_dict(orient="records"):
        lines.append(
            f"- {row['oa_collection_slug']}: member_rows={int(row['member_rows'])}, groups={int(row['groups_with_collection'])}"
        )
    lines.append("")
    lines.append("## Top Meaningful Cross-Collection Pairs")
    for row in cross_pairs.head(20).to_dict(orient="records"):
        lines.append(
            f"- {row['source_a']} <-> {row['source_b']}: groups={int(row['group_count'])}, pair_member_rows={int(row['member_rows'])}"
        )
    lines.append("")
    lines.append("## Top Meaningful Within-Collection Duplicates")
    for row in within_pairs.head(15).to_dict(orient="records"):
        lines.append(
            f"- {row['oa_collection_slug']}: groups={int(row['group_count'])}, member_rows={int(row['member_rows'])}"
        )
    return "\n".join(lines) + "\n"


def analyze_openarchives_internal_patterns(run_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    analysis_root = (output_dir or (run_root / "analysis")).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)

    summary, strict_groups, oa_groups = load_oa_internal_groups(run_root)
    oa_groups, members = load_collection_members(summary, strict_groups, oa_groups)

    collection_summary_all = build_collection_summary(oa_groups, members, meaningful_only=False)
    collection_summary_meaningful = build_collection_summary(oa_groups, members, meaningful_only=True)
    cross_pairs, within_pairs = build_pair_summary(oa_groups, members)
    reason_summary = build_reason_summary(oa_groups)

    collection_summary_all_path = analysis_root / "oa_internal_collection_summary_all.csv"
    collection_summary_meaningful_path = analysis_root / "oa_internal_collection_summary_meaningful.csv"
    cross_pairs_path = analysis_root / "oa_internal_cross_collection_pairs_meaningful.csv"
    within_pairs_path = analysis_root / "oa_internal_within_collection_pairs_meaningful.csv"
    reason_summary_path = analysis_root / "oa_internal_reason_summary.csv"
    report_path = analysis_root / "oa_internal_collection_patterns_report.md"

    write_csv(collection_summary_all, collection_summary_all_path)
    write_csv(collection_summary_meaningful, collection_summary_meaningful_path)
    write_csv(cross_pairs, cross_pairs_path)
    write_csv(within_pairs, within_pairs_path)
    write_csv(reason_summary, reason_summary_path)
    report_path.write_text(
        build_report(
            run_id=str(summary["run_id"]),
            population_size=int(len(oa_groups)),
            collection_summary_all=collection_summary_all,
            collection_summary_meaningful=collection_summary_meaningful,
            cross_pairs=cross_pairs,
            within_pairs=within_pairs,
            reason_summary=reason_summary,
        )
    )

    payload = {
        "run_id": str(summary["run_id"]),
        "analysis_root": str(analysis_root),
        "oa_internal_collection_summary_all_path": str(collection_summary_all_path),
        "oa_internal_collection_summary_meaningful_path": str(collection_summary_meaningful_path),
        "oa_internal_cross_collection_pairs_meaningful_path": str(cross_pairs_path),
        "oa_internal_within_collection_pairs_meaningful_path": str(within_pairs_path),
        "oa_internal_reason_summary_path": str(reason_summary_path),
        "oa_internal_collection_patterns_report_path": str(report_path),
    }
    (analysis_root / "oa_internal_collection_patterns_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OA internal exact duplicate patterns by collection slug.")
    parser.add_argument("--run-root", required=True, type=Path, help="Path to a completed exact-stage run root")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory; defaults to <run_root>/analysis")
    args = parser.parse_args()
    payload = analyze_openarchives_internal_patterns(args.run_root, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
