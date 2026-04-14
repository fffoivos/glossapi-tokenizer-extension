from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_OA_DROP_LIST_SUMMARY = Path(
    "/home/foivos/data/glossapi_work/analysis/dedup/metadata_openarchives/runs/drop_list_20260325T125005473654Z/summary.json"
)
DEFAULT_PHD_CROSS_COLLECTION_SUMMARY = Path(
    "/home/foivos/data/glossapi_work/analysis/dedup/metadata_openarchives/runs/phdtheses_cross_collection_20260325T180424Z/summary.json"
)
DEFAULT_PERGAMOS_PHD_RECOVERY_SUMMARY = Path(
    "/home/foivos/data/glossapi_work/analysis/dedup/metadata_openarchives/runs/pergamos_phd_recovery_20260325T190000Z/summary.json"
)
DEFAULT_PERGAMOS_PHD_REVIEW_SUMMARY = Path(
    "/home/foivos/data/glossapi_work/analysis/dedup/metadata_openarchives/runs/pergamos_phd_review_bucket_20260325T192500Z/summary.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        df = pd.DataFrame(columns=df.columns)
    df.to_csv(path, index=False)


def build_openarchives_collection_map(input_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in sorted(input_root.glob("openarchives.gr.part-*.parquet")):
        table = pq.read_table(path, columns=["source_doc_id", "source_metadata_json"])
        payload = table.to_pylist()
        for row in payload:
            source_doc_id = row.get("source_doc_id")
            if not source_doc_id:
                continue
            collection_slug = "unknown"
            metadata_json = row.get("source_metadata_json")
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {}
                if isinstance(metadata, dict):
                    value = metadata.get("collection_slug")
                    if value:
                        collection_slug = str(value)
            rows.append(
                {
                    "source_doc_id": str(source_doc_id),
                    "oa_collection_slug": collection_slug,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["source_doc_id", "oa_collection_slug"])
    return frame.drop_duplicates(subset=["source_doc_id"], keep="first")


def query_frame(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def parquet_glob_sql(path: Path) -> str:
    escaped = str(path).replace("'", "''")
    return f"read_parquet('{escaped}')"


def create_group_view(con: duckdb.DuckDBPyConnection, view_name: str, path: Path) -> None:
    parquet_file = pq.ParquetFile(path)
    if parquet_file.schema_arrow.names:
        con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM {parquet_glob_sql(path)}")
        return
    con.execute(
        f"""
        CREATE VIEW {view_name} AS
        SELECT
            CAST(NULL AS VARCHAR) AS group_hash,
            CAST(NULL AS BIGINT) AS group_size,
            CAST(NULL AS VARCHAR) AS kept_doc_key,
            CAST(NULL AS VARCHAR) AS member_doc_key,
            CAST(NULL AS VARCHAR) AS member_source_dataset,
            CAST(NULL AS VARCHAR) AS member_source_doc_id,
            CAST(NULL AS BOOLEAN) AS dropped
        WHERE FALSE
        """
    )


def scalar_count(df: pd.DataFrame, **filters: object) -> int:
    if df.empty:
        return 0
    mask = pd.Series([True] * len(df))
    for key, value in filters.items():
        mask &= df[key] == value
    if "dropped_rows" in df.columns:
        return int(df.loc[mask, "dropped_rows"].sum())
    if "oa_member_rows" in df.columns:
        return int(df.loc[mask, "oa_member_rows"].sum())
    return int(mask.sum())


def build_report(
    *,
    run_summary: dict[str, Any],
    source_summary: pd.DataFrame,
    oa_collection_summary: pd.DataFrame,
    final_drop_by_kept_source: pd.DataFrame,
    oa_drop_by_kept_source: pd.DataFrame,
    oa_cross_source_overlap: pd.DataFrame,
    metadata_comparison: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"# Exact Stage Analysis: {run_summary['run_id']}")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- Total rows: {run_summary['total_rows']}")
    lines.append(f"- Kept after exact: {run_summary['kept_after_exact_rows']}")
    lines.append(f"- Strict dropped rows: {run_summary['strict']['dropped_rows']}")
    lines.append(f"- Relaxed dropped rows: {run_summary['relaxed']['dropped_rows']}")
    lines.append("")
    lines.append("## Most Affected Source Datasets")
    top_sources = source_summary.sort_values(
        ["total_dropped_rows", "strict_dropped_rows", "source_dataset"],
        ascending=[False, False, True],
    ).head(10)
    for row in top_sources.to_dict(orient="records"):
        lines.append(
            f"- {row['source_dataset']}: dropped={int(row['total_dropped_rows'])}, "
            f"strict={int(row['strict_dropped_rows'])}, relaxed_only={int(row['relaxed_only_dropped_rows'])}, "
            f"kept={int(row['kept_after_exact_rows'])}"
        )
    if not oa_collection_summary.empty:
        lines.append("")
        lines.append("## OpenArchives Collections Most Affected")
        top_collections = oa_collection_summary.sort_values(
            ["total_dropped_rows", "oa_collection_slug"],
            ascending=[False, True],
        ).head(10)
        for row in top_collections.to_dict(orient="records"):
            lines.append(
                f"- {row['oa_collection_slug']}: dropped={int(row['total_dropped_rows'])}, "
                f"strict={int(row['strict_dropped_rows'])}, relaxed_only={int(row['relaxed_only_dropped_rows'])}, "
                f"kept={int(row['kept_after_exact_rows'])}"
            )
    if not final_drop_by_kept_source.empty:
        lines.append("")
        lines.append("## Largest Final Drop Flows")
        top_flows = final_drop_by_kept_source.sort_values(
            ["dropped_rows", "dropped_source_dataset", "kept_source_dataset"],
            ascending=[False, True, True],
        ).head(10)
        for row in top_flows.to_dict(orient="records"):
            lines.append(
                f"- {row['dropped_source_dataset']} -> {row['kept_source_dataset']}: "
                f"{int(row['dropped_rows'])} dropped rows ({row['stage']})"
            )
    lines.append("")
    lines.append("## Metadata Comparison")
    oa_exact = metadata_comparison["exact_published"]["openarchives"]
    phd_exact = metadata_comparison["exact_published"]["phd"]
    oa_meta = metadata_comparison["metadata_baselines"]["openarchives_raw_drop_list"]
    phd_meta = metadata_comparison["metadata_baselines"]["phd_cross_collection"]
    lines.append(
        f"- Kallipos: metadata raw OA drop-list={oa_meta['kallipos_raw_drop_rows']} vs "
        f"published exact OA->Apothetirio_Kallipos drops={oa_exact['kallipos_to_kallipos_source_drops']} "
        f"and OA memberships in exact cross-source groups={oa_exact['kallipos_to_kallipos_source_overlap_rows']}."
    )
    lines.append(
        f"- Pergamos: metadata raw OA drop-list={oa_meta['pergamos_raw_drop_rows']} vs "
        f"published exact OA->Apothetirio_Pergamos drops={oa_exact['pergamos_to_pergamos_source_drops']} "
        f"and OA memberships in exact cross-source groups={oa_exact['pergamos_to_pergamos_source_overlap_rows']}."
    )
    lines.append(
        f"- OA phdtheses collection vs greek_phd: metadata outside-collection OA matches={phd_meta['matched_oa_outside_phdtheses']} "
        f"and exact title+author Pergamos->phdtheses baseline={phd_meta['pergamos_title_author_rows']} before recovery tiers; "
        f"published exact OA phdtheses -> greek_phd drops={phd_exact['oa_phdtheses_to_greek_phd_drops']}, "
        f"OA Pergamos -> greek_phd drops={phd_exact['oa_pergamos_to_greek_phd_drops']}, "
        f"and greek_phd -> openarchives.gr drops={phd_exact['greek_phd_to_openarchives_drops']}."
    )
    if not oa_drop_by_kept_source.empty:
        top_oa_flows = oa_drop_by_kept_source.sort_values(
            ["dropped_rows", "oa_collection_slug", "kept_source_dataset"],
            ascending=[False, True, True],
        ).head(10)
        lines.append("")
        lines.append("## Largest OpenArchives Collection Drop Flows")
        for row in top_oa_flows.to_dict(orient="records"):
            lines.append(
                f"- {row['oa_collection_slug']} -> {row['kept_source_dataset']}: "
                f"{int(row['dropped_rows'])} dropped rows ({row['stage']})"
            )
    if not oa_cross_source_overlap.empty:
        top_overlap = oa_cross_source_overlap.sort_values(
            ["oa_member_rows", "oa_collection_slug", "counterpart_source_dataset"],
            ascending=[False, True, True],
        ).head(10)
        lines.append("")
        lines.append("## Largest Exact Cross-Source OA Overlaps")
        for row in top_overlap.to_dict(orient="records"):
            lines.append(
                f"- {row['oa_collection_slug']} <-> {row['counterpart_source_dataset']}: "
                f"{int(row['oa_member_rows'])} OA rows in exact {row['stage']} overlap groups"
            )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- source_summary.csv")
    lines.append("- final_drop_by_kept_source.csv")
    lines.append("- openarchives_collection_summary.csv")
    lines.append("- openarchives_drop_by_kept_source.csv")
    lines.append("- openarchives_cross_source_overlap.csv")
    lines.append("- metadata_comparison.json")
    return "\n".join(lines) + "\n"


def analyze_exact_run(run_root: Path, output_dir: Path) -> dict[str, Any]:
    stage_root = run_root / "stage_01_exact"
    run_summary_path = stage_root / "summary.json"
    if not run_summary_path.exists():
        raise FileNotFoundError(f"exact-stage summary is missing: {run_summary_path}")
    run_summary = load_json(run_summary_path)
    input_root = Path(run_summary["input_root"])
    docs_path = Path(run_summary["docs_exact_path"])
    strict_groups_path = Path(run_summary["strict_exact_groups_path"])
    relaxed_groups_path = Path(run_summary["relaxed_exact_groups_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW docs AS SELECT * FROM {parquet_glob_sql(docs_path)}")
    create_group_view(con, "strict_groups", strict_groups_path)
    create_group_view(con, "relaxed_groups", relaxed_groups_path)

    source_summary = query_frame(
        con,
        """
        SELECT
            source_dataset,
            COUNT(*) AS total_rows,
            SUM(CASE WHEN strict_dropped THEN 1 ELSE 0 END) AS strict_dropped_rows,
            SUM(CASE WHEN NOT strict_dropped AND COALESCE(relaxed_dropped, FALSE) THEN 1 ELSE 0 END) AS relaxed_only_dropped_rows,
            SUM(CASE WHEN NOT kept_after_exact THEN 1 ELSE 0 END) AS total_dropped_rows,
            SUM(CASE WHEN kept_after_exact THEN 1 ELSE 0 END) AS kept_after_exact_rows
        FROM docs
        GROUP BY source_dataset
        ORDER BY total_dropped_rows DESC, source_dataset
        """,
    )
    write_csv(source_summary, output_dir / "source_summary.csv")

    final_drop_by_kept_source = query_frame(
        con,
        """
        WITH strict_stage AS (
            SELECT
                'exact_strict' AS stage,
                d.source_dataset AS dropped_source_dataset,
                k.source_dataset AS kept_source_dataset,
                COUNT(*) AS dropped_rows
            FROM docs AS d
            JOIN docs AS k
              ON k.doc_key = d.strict_kept_doc_key
            WHERE d.strict_dropped
            GROUP BY 1, 2, 3
        ),
        relaxed_stage AS (
            SELECT
                'exact_relaxed' AS stage,
                d.source_dataset AS dropped_source_dataset,
                k.source_dataset AS kept_source_dataset,
                COUNT(*) AS dropped_rows
            FROM docs AS d
            JOIN docs AS k
              ON k.doc_key = d.relaxed_kept_doc_key
            WHERE NOT d.strict_dropped
              AND COALESCE(d.relaxed_dropped, FALSE)
            GROUP BY 1, 2, 3
        )
        SELECT *
        FROM strict_stage
        UNION ALL
        SELECT *
        FROM relaxed_stage
        ORDER BY dropped_rows DESC, dropped_source_dataset, kept_source_dataset
        """,
    )
    write_csv(final_drop_by_kept_source, output_dir / "final_drop_by_kept_source.csv")

    oa_map = build_openarchives_collection_map(input_root)
    write_csv(oa_map, output_dir / "openarchives_collection_map.csv")
    if oa_map.empty:
        oa_collection_summary = pd.DataFrame(
            columns=[
                "oa_collection_slug",
                "total_rows",
                "strict_dropped_rows",
                "relaxed_only_dropped_rows",
                "total_dropped_rows",
                "kept_after_exact_rows",
            ]
        )
        oa_drop_by_kept_source = pd.DataFrame(columns=["stage", "oa_collection_slug", "kept_source_dataset", "dropped_rows"])
        oa_cross_source_overlap = pd.DataFrame(columns=["stage", "oa_collection_slug", "counterpart_source_dataset", "oa_member_rows"])
    else:
        con.register("oa_map", oa_map)
        oa_collection_summary = query_frame(
            con,
            """
            SELECT
                m.oa_collection_slug,
                COUNT(*) AS total_rows,
                SUM(CASE WHEN d.strict_dropped THEN 1 ELSE 0 END) AS strict_dropped_rows,
                SUM(CASE WHEN NOT d.strict_dropped AND COALESCE(d.relaxed_dropped, FALSE) THEN 1 ELSE 0 END) AS relaxed_only_dropped_rows,
                SUM(CASE WHEN NOT d.kept_after_exact THEN 1 ELSE 0 END) AS total_dropped_rows,
                SUM(CASE WHEN d.kept_after_exact THEN 1 ELSE 0 END) AS kept_after_exact_rows
            FROM docs AS d
            JOIN oa_map AS m
              ON m.source_doc_id = d.source_doc_id
            WHERE d.source_dataset = 'openarchives.gr'
            GROUP BY m.oa_collection_slug
            ORDER BY total_dropped_rows DESC, m.oa_collection_slug
            """,
        )
        write_csv(oa_collection_summary, output_dir / "openarchives_collection_summary.csv")

        oa_drop_by_kept_source = query_frame(
            con,
            """
            WITH strict_stage AS (
                SELECT
                    'exact_strict' AS stage,
                    m.oa_collection_slug,
                    k.source_dataset AS kept_source_dataset,
                    COUNT(*) AS dropped_rows
                FROM docs AS d
                JOIN oa_map AS m
                  ON m.source_doc_id = d.source_doc_id
                JOIN docs AS k
                  ON k.doc_key = d.strict_kept_doc_key
                WHERE d.source_dataset = 'openarchives.gr'
                  AND d.strict_dropped
                GROUP BY 1, 2, 3
            ),
            relaxed_stage AS (
                SELECT
                    'exact_relaxed' AS stage,
                    m.oa_collection_slug,
                    k.source_dataset AS kept_source_dataset,
                    COUNT(*) AS dropped_rows
                FROM docs AS d
                JOIN oa_map AS m
                  ON m.source_doc_id = d.source_doc_id
                JOIN docs AS k
                  ON k.doc_key = d.relaxed_kept_doc_key
                WHERE d.source_dataset = 'openarchives.gr'
                  AND NOT d.strict_dropped
                  AND COALESCE(d.relaxed_dropped, FALSE)
                GROUP BY 1, 2, 3
            )
            SELECT *
            FROM strict_stage
            UNION ALL
            SELECT *
            FROM relaxed_stage
            ORDER BY dropped_rows DESC, oa_collection_slug, kept_source_dataset
            """,
        )
        write_csv(oa_drop_by_kept_source, output_dir / "openarchives_drop_by_kept_source.csv")

        oa_cross_source_overlap = query_frame(
            con,
            """
            WITH strict_overlap AS (
                SELECT
                    'exact_strict' AS stage,
                    m.oa_collection_slug,
                    other.member_source_dataset AS counterpart_source_dataset,
                    COUNT(DISTINCT oa.member_doc_key) AS oa_member_rows
                FROM strict_groups AS oa
                JOIN strict_groups AS other
                  ON other.group_hash = oa.group_hash
                 AND other.member_source_dataset <> 'openarchives.gr'
                JOIN oa_map AS m
                  ON m.source_doc_id = oa.member_source_doc_id
                WHERE oa.member_source_dataset = 'openarchives.gr'
                GROUP BY 1, 2, 3
            ),
            relaxed_overlap AS (
                SELECT
                    'exact_relaxed' AS stage,
                    m.oa_collection_slug,
                    other.member_source_dataset AS counterpart_source_dataset,
                    COUNT(DISTINCT oa.member_doc_key) AS oa_member_rows
                FROM relaxed_groups AS oa
                JOIN relaxed_groups AS other
                  ON other.group_hash = oa.group_hash
                 AND other.member_source_dataset <> 'openarchives.gr'
                JOIN oa_map AS m
                  ON m.source_doc_id = oa.member_source_doc_id
                WHERE oa.member_source_dataset = 'openarchives.gr'
                GROUP BY 1, 2, 3
            )
            SELECT *
            FROM strict_overlap
            UNION ALL
            SELECT *
            FROM relaxed_overlap
            ORDER BY oa_member_rows DESC, oa_collection_slug, counterpart_source_dataset
            """,
        )
        write_csv(oa_cross_source_overlap, output_dir / "openarchives_cross_source_overlap.csv")

    oa_metadata_summary = optional_json(DEFAULT_OA_DROP_LIST_SUMMARY) or {}
    phd_cross_summary = optional_json(DEFAULT_PHD_CROSS_COLLECTION_SUMMARY) or {}
    pergamos_phd_recovery = optional_json(DEFAULT_PERGAMOS_PHD_RECOVERY_SUMMARY) or {}
    pergamos_phd_review = optional_json(DEFAULT_PERGAMOS_PHD_REVIEW_SUMMARY) or {}

    metadata_comparison = {
        "run_id": run_summary["run_id"],
        "metadata_baselines": {
            "openarchives_raw_drop_list": {
                "kallipos_raw_drop_rows": int(
                    (oa_metadata_summary.get("origins") or {}).get("kallipos", {}).get("drop_rows", 0)
                ),
                "pergamos_raw_drop_rows": int(
                    (oa_metadata_summary.get("origins") or {}).get("pergamos", {}).get("drop_rows", 0)
                ),
                "raw_total_drop_rows": int(oa_metadata_summary.get("drop_rows_total", 0)),
            },
            "phd_cross_collection": {
                "matched_oa_outside_phdtheses": int(phd_cross_summary.get("matched_oa_outside_phdtheses", 0)),
                "pergamos_title_author_rows": int(pergamos_phd_recovery.get("exact_title_author_rows", 0)),
                "pergamos_exact_title_unique_fallback_total": int(
                    pergamos_phd_recovery.get("total_rows_recovered_if_exact_title_unique_fallback_added", 0)
                ),
                "pergamos_after_reduced_title_unique": int(
                    pergamos_phd_review.get("after_reduced_title_unique", 0)
                ),
                "pergamos_after_all_tested_layers": int(
                    pergamos_phd_review.get("after_all_tested_layers", 0)
                ),
            },
        },
        "exact_published": {
            "openarchives": {
                "kallipos_to_kallipos_source_drops": scalar_count(
                    oa_drop_by_kept_source,
                    oa_collection_slug="kallipos",
                    kept_source_dataset="Apothetirio_Kallipos",
                ),
                "pergamos_to_pergamos_source_drops": scalar_count(
                    oa_drop_by_kept_source,
                    oa_collection_slug="pergamos",
                    kept_source_dataset="Apothetirio_Pergamos",
                ),
                "kallipos_to_kallipos_source_overlap_rows": scalar_count(
                    oa_cross_source_overlap,
                    oa_collection_slug="kallipos",
                    counterpart_source_dataset="Apothetirio_Kallipos",
                ),
                "pergamos_to_pergamos_source_overlap_rows": scalar_count(
                    oa_cross_source_overlap,
                    oa_collection_slug="pergamos",
                    counterpart_source_dataset="Apothetirio_Pergamos",
                ),
            },
            "phd": {
                "oa_phdtheses_to_greek_phd_drops": scalar_count(
                    oa_drop_by_kept_source,
                    oa_collection_slug="phdtheses",
                    kept_source_dataset="greek_phd",
                ),
                "oa_pergamos_to_greek_phd_drops": scalar_count(
                    oa_drop_by_kept_source,
                    oa_collection_slug="pergamos",
                    kept_source_dataset="greek_phd",
                ),
                "oa_phdtheses_to_greek_phd_overlap_rows": scalar_count(
                    oa_cross_source_overlap,
                    oa_collection_slug="phdtheses",
                    counterpart_source_dataset="greek_phd",
                ),
                "oa_pergamos_to_greek_phd_overlap_rows": scalar_count(
                    oa_cross_source_overlap,
                    oa_collection_slug="pergamos",
                    counterpart_source_dataset="greek_phd",
                ),
                "greek_phd_to_openarchives_drops": scalar_count(
                    final_drop_by_kept_source,
                    dropped_source_dataset="greek_phd",
                    kept_source_dataset="openarchives.gr",
                ),
            },
        },
    }

    metadata_comparison_path = output_dir / "metadata_comparison.json"
    metadata_comparison_path.write_text(json.dumps(metadata_comparison, ensure_ascii=False, indent=2))

    report_path = output_dir / "report.md"
    report_path.write_text(
        build_report(
            run_summary=run_summary,
            source_summary=source_summary,
            oa_collection_summary=oa_collection_summary,
            final_drop_by_kept_source=final_drop_by_kept_source,
            oa_drop_by_kept_source=oa_drop_by_kept_source,
            oa_cross_source_overlap=oa_cross_source_overlap,
            metadata_comparison=metadata_comparison,
        )
    )

    summary = {
        "run_id": run_summary["run_id"],
        "run_root": str(run_root),
        "analysis_root": str(output_dir),
        "report_path": str(report_path),
        "source_summary_path": str(output_dir / "source_summary.csv"),
        "final_drop_by_kept_source_path": str(output_dir / "final_drop_by_kept_source.csv"),
        "openarchives_collection_summary_path": str(output_dir / "openarchives_collection_summary.csv"),
        "openarchives_drop_by_kept_source_path": str(output_dir / "openarchives_drop_by_kept_source.csv"),
        "openarchives_cross_source_overlap_path": str(output_dir / "openarchives_cross_source_overlap.csv"),
        "metadata_comparison_path": str(metadata_comparison_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a completed exact-stage text dedup run.")
    parser.add_argument("--run-root", required=True, type=Path, help="Path to the completed dedup run root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for analysis artifacts; defaults to <run_root>/analysis",
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output_dir = (args.output_dir or (run_root / "analysis")).resolve()
    payload = analyze_exact_run(run_root=run_root, output_dir=output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
