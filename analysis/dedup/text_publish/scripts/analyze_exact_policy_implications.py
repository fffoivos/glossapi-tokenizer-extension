from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq


SOURCE_POLICY = {
    "openarchives.gr": {
        "role": "aggregator",
        "rank": 50,
        "reason": "OpenArchives is an aggregator mirror; when it overlaps with a dedicated source, prefer the dedicated source over the aggregator copy.",
    },
    "greek_phd": {
        "role": "canonical_thesis",
        "rank": 5,
        "reason": "greek_phd is the dedicated thesis corpus and is the cleaner provenance anchor for thesis material.",
    },
    "Apothetirio_Kallipos": {
        "role": "standalone_repository",
        "rank": 10,
        "reason": "Apothetirio_Kallipos is the standalone repository source and should outrank aggregator mirrors of the same material.",
    },
    "Apothetirio_Pergamos": {
        "role": "standalone_repository",
        "rank": 10,
        "reason": "Apothetirio_Pergamos is the standalone repository source and should outrank aggregator mirrors of the same material.",
    },
    "opengov.gr-diaboyleuseis": {
        "role": "canonical_government",
        "rank": 0,
        "reason": "The direct opengov consultation corpus has stronger provenance than mirrored aggregator copies.",
    },
    "eurlex-greek-legislation": {
        "role": "canonical_legal",
        "rank": 0,
        "reason": "The Eur-Lex legal corpus is the canonical legal source and should outrank mirrored or OCR-derived copies.",
    },
    "ellinika_dedomena_europaikou_koinovouliou": {
        "role": "canonical_institutional",
        "rank": 5,
        "reason": "The dedicated European Parliament corpus is the more structured provenance anchor than generic PDF mixtures.",
    },
    "openbook_gr": {
        "role": "curated_book_corpus",
        "rank": 15,
        "reason": "openbook_gr is a dedicated curated corpus and should outrank generic PDF mixtures when texts are exact duplicates.",
    },
    "HuggingFaceFW/finepdfs-edu": {
        "role": "pdf_mixture",
        "rank": 60,
        "reason": "finepdfs-edu is a generic PDF mixture and should usually lose to a direct structured source when exact duplicates exist.",
    },
    "HuggingFaceFW/finewiki": {
        "role": "reference_corpus",
        "rank": 30,
        "reason": "finewiki is a self-contained reference corpus; most exact duplicates here are internal rather than provenance conflicts.",
    },
}

IMAGE_PLACEHOLDER_RE = re.compile(
    r"(?:<!--\s*image\s*-->|!\[[^\]]*\]\([^)]*\)|<img\b[^>]*>|\[\s*image\s*\]|\(\s*image\s*\))",
    flags=re.IGNORECASE,
)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SYMBOL_SPACE_RE = re.compile(r"[\s\-\–\—\.\,\:\;\!\?\"'“”‘’\(\)\[\]\{\}/\\|_*+=~`<>·•…]+")


def policy_for(source_dataset: str) -> dict[str, Any]:
    return SOURCE_POLICY.get(
        source_dataset,
        {
            "role": "unknown",
            "rank": 40,
            "reason": "No explicit provenance preference is encoded for this source pair; keep/drop should be reviewed manually.",
        },
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        df = pd.DataFrame(columns=df.columns)
    df.to_csv(path, index=False)


def numericize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return df


def load_group_frame(path: Path) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    if not parquet_file.schema_arrow.names:
        return pd.DataFrame(
            columns=[
                "group_hash",
                "group_size",
                "kept_doc_key",
                "member_doc_key",
                "member_source_dataset",
                "member_source_doc_id",
                "dropped",
            ]
        )
    return pq.read_table(path).to_pandas()


def build_source_file_map(snapshot_manifest_path: Path) -> dict[str, list[Path]]:
    manifest = pq.read_table(snapshot_manifest_path, columns=["source_dataset", "file_path"]).to_pandas()
    file_map: dict[str, list[Path]] = {}
    for row in manifest.to_dict(orient="records"):
        source_dataset = str(row["source_dataset"])
        file_path = Path(str(row["file_path"]))
        bucket = file_map.setdefault(source_dataset, [])
        if file_path not in bucket:
            bucket.append(file_path)
    return file_map


def lookup_doc_texts(snapshot_manifest_path: Path, lookup_docs: pd.DataFrame) -> pd.DataFrame:
    if lookup_docs.empty:
        return pd.DataFrame(columns=["source_dataset", "source_doc_id", "title", "text"])

    file_map = build_source_file_map(snapshot_manifest_path)
    needed_by_source: dict[str, set[str]] = {}
    for row in lookup_docs[["source_dataset", "source_doc_id"]].drop_duplicates().to_dict(orient="records"):
        needed_by_source.setdefault(str(row["source_dataset"]), set()).add(str(row["source_doc_id"]))

    found_rows: list[dict[str, Any]] = []
    for source_dataset, needed_ids in needed_by_source.items():
        remaining = set(needed_ids)
        if not remaining:
            continue
        for file_path in file_map.get(source_dataset, []):
            parquet_file = pq.ParquetFile(file_path)
            available_columns = set(parquet_file.schema_arrow.names)
            if "source_doc_id" not in available_columns or "text" not in available_columns:
                continue
            columns = ["source_doc_id", "text"]
            if "title" in available_columns:
                columns.append("title")
            for batch in parquet_file.iter_batches(columns=columns, batch_size=2048):
                for row in batch.to_pylist():
                    source_doc_id = row.get("source_doc_id")
                    if source_doc_id not in remaining:
                        continue
                    found_rows.append(
                        {
                            "source_dataset": source_dataset,
                            "source_doc_id": str(source_doc_id),
                            "title": row.get("title"),
                            "text": row.get("text"),
                        }
                    )
                    remaining.remove(str(source_doc_id))
                if not remaining:
                    break
            if not remaining:
                break
    return pd.DataFrame(found_rows)


def classify_low_information_text(text: str | None) -> str | None:
    if text is None:
        return "missing_text"
    stripped = text.strip()
    if not stripped:
        return "empty"
    if not IMAGE_PLACEHOLDER_RE.sub("", stripped).strip():
        return "image_placeholder_only"
    without_comments = HTML_COMMENT_RE.sub("", stripped)
    if not without_comments.strip():
        return "html_comment_only"
    without_markup = HTML_TAG_RE.sub("", without_comments)
    if not SYMBOL_SPACE_RE.sub("", without_markup):
        return "markup_or_symbol_only"
    return None


def build_low_information_groups(
    *,
    snapshot_manifest_path: Path,
    strict_groups_path: Path,
    relaxed_groups_path: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for stage, path in (
        ("strict", strict_groups_path),
        ("relaxed", relaxed_groups_path),
    ):
        frame = load_group_frame(path)
        if frame.empty:
            continue
        grouped = (
            frame.groupby("group_hash", as_index=False)
            .agg(
                group_size=("group_size", "max"),
                source_count=("member_source_dataset", "nunique"),
                sources_in_group=("member_source_dataset", lambda series: "|".join(sorted({str(v) for v in series}))),
            )
        )
        grouped = grouped[grouped["source_count"] > 1].copy()
        if grouped.empty:
            continue
        representative_rows = frame.loc[frame["member_doc_key"] == frame["kept_doc_key"]].copy()
        representative_rows = representative_rows[
            ["group_hash", "member_source_dataset", "member_source_doc_id"]
        ].drop_duplicates(subset=["group_hash"], keep="first")
        if len(representative_rows) < len(grouped):
            fallback = (
                frame.sort_values(["group_hash", "member_source_dataset", "member_source_doc_id"])
                .groupby("group_hash", as_index=False)
                .first()[["group_hash", "member_source_dataset", "member_source_doc_id"]]
            )
            representative_rows = pd.concat([representative_rows, fallback], ignore_index=True)
            representative_rows = representative_rows.drop_duplicates(subset=["group_hash"], keep="first")
        stage_frame = grouped.merge(representative_rows, how="left", on="group_hash")
        stage_frame["stage"] = stage
        stage_frame = stage_frame.rename(
            columns={
                "member_source_dataset": "source_dataset",
                "member_source_doc_id": "source_doc_id",
            }
        )
        frames.append(stage_frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "stage",
                "group_hash",
                "group_size",
                "source_count",
                "sources_in_group",
                "source_dataset",
                "source_doc_id",
                "representative_title",
                "representative_text_chars",
                "low_information_reason",
                "text_preview",
            ]
        )

    representatives = pd.concat(frames, ignore_index=True)
    text_lookup = lookup_doc_texts(
        snapshot_manifest_path,
        representatives[["source_dataset", "source_doc_id"]].drop_duplicates(),
    )
    merged = representatives.merge(
        text_lookup,
        how="left",
        on=["source_dataset", "source_doc_id"],
    )
    merged["representative_text_chars"] = merged["text"].map(lambda value: len(value) if isinstance(value, str) else 0)
    merged["low_information_reason"] = merged["text"].map(classify_low_information_text)
    merged["representative_title"] = merged["title"]
    merged["text_preview"] = merged["text"].map(lambda value: (value or "")[:200].replace("\n", "\\n"))
    result = merged[
        [
            "stage",
            "group_hash",
            "group_size",
            "source_count",
            "sources_in_group",
            "source_dataset",
            "source_doc_id",
            "representative_title",
            "representative_text_chars",
            "low_information_reason",
            "text_preview",
        ]
    ].copy()
    return result[result["low_information_reason"].notna()].sort_values(
        ["group_size", "stage", "group_hash"],
        ascending=[False, True, True],
    )


def register_hash_view(con: duckdb.DuckDBPyConnection, view_name: str, hashes: set[str]) -> None:
    frame = pd.DataFrame({"group_hash": sorted(hashes)})
    if frame.empty:
        frame = pd.DataFrame({"group_hash": pd.Series(dtype="string")})
    con.register(view_name, frame)


def create_filtered_views(
    con: duckdb.DuckDBPyConnection,
    *,
    strict_low_info_hashes: set[str],
    relaxed_low_info_hashes: set[str],
) -> None:
    register_hash_view(con, "strict_low_info_groups", strict_low_info_hashes)
    register_hash_view(con, "relaxed_low_info_groups", relaxed_low_info_hashes)
    con.execute(
        """
        CREATE OR REPLACE VIEW strict_groups_filtered AS
        SELECT * FROM strict_groups
        WHERE group_hash NOT IN (SELECT group_hash FROM strict_low_info_groups)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW relaxed_groups_filtered AS
        SELECT * FROM relaxed_groups
        WHERE group_hash NOT IN (SELECT group_hash FROM relaxed_low_info_groups)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW strict_groups_low_info AS
        SELECT * FROM strict_groups
        WHERE group_hash IN (SELECT group_hash FROM strict_low_info_groups)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW relaxed_groups_low_info AS
        SELECT * FROM relaxed_groups
        WHERE group_hash IN (SELECT group_hash FROM relaxed_low_info_groups)
        """
    )


def query_pair_overlap(
    con: duckdb.DuckDBPyConnection,
    *,
    strict_view: str,
    relaxed_view: str,
) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH strict_pairs AS (
            SELECT
                a.member_source_dataset AS source_a,
                b.member_source_dataset AS source_b,
                COUNT(DISTINCT a.group_hash) AS strict_shared_groups,
                COUNT(DISTINCT a.member_doc_key) AS strict_source_a_rows,
                COUNT(DISTINCT b.member_doc_key) AS strict_source_b_rows
            FROM {strict_view} AS a
            JOIN {strict_view} AS b
              ON a.group_hash = b.group_hash
             AND a.member_source_dataset < b.member_source_dataset
            GROUP BY 1, 2
        ),
        relaxed_pairs AS (
            SELECT
                a.member_source_dataset AS source_a,
                b.member_source_dataset AS source_b,
                COUNT(DISTINCT a.group_hash) AS relaxed_shared_groups,
                COUNT(DISTINCT a.member_doc_key) AS relaxed_source_a_rows,
                COUNT(DISTINCT b.member_doc_key) AS relaxed_source_b_rows
            FROM {relaxed_view} AS a
            JOIN {relaxed_view} AS b
              ON a.group_hash = b.group_hash
             AND a.member_source_dataset < b.member_source_dataset
            GROUP BY 1, 2
        )
        SELECT
            COALESCE(s.source_a, r.source_a) AS source_a,
            COALESCE(s.source_b, r.source_b) AS source_b,
            COALESCE(strict_shared_groups, 0) AS strict_shared_groups,
            COALESCE(relaxed_shared_groups, 0) AS relaxed_shared_groups,
            COALESCE(strict_source_a_rows, 0) + COALESCE(relaxed_source_a_rows, 0) AS source_a_rows_in_shared_groups,
            COALESCE(strict_source_b_rows, 0) + COALESCE(relaxed_source_b_rows, 0) AS source_b_rows_in_shared_groups
        FROM strict_pairs AS s
        FULL OUTER JOIN relaxed_pairs AS r
          ON s.source_a = r.source_a
         AND s.source_b = r.source_b
        """
    ).df()


def query_directed_cross_drops(
    con: duckdb.DuckDBPyConnection,
    *,
    strict_hash_filter: str,
    relaxed_hash_filter: str,
) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH strict_cross AS (
            SELECT d.source_dataset AS dropped_source, k.source_dataset AS kept_source, COUNT(*) AS strict_drops
            FROM docs AS d
            JOIN docs AS k
              ON k.doc_key = d.strict_kept_doc_key
            WHERE d.strict_dropped = 1
              AND d.source_dataset <> k.source_dataset
              AND d.exact_strict_hash {strict_hash_filter}
            GROUP BY 1, 2
        ),
        relaxed_cross AS (
            SELECT d.source_dataset AS dropped_source, k.source_dataset AS kept_source, COUNT(*) AS relaxed_drops
            FROM docs AS d
            JOIN docs AS k
              ON k.doc_key = d.relaxed_kept_doc_key
            WHERE d.strict_dropped = 0
              AND COALESCE(d.relaxed_dropped, 0) = 1
              AND d.source_dataset <> k.source_dataset
              AND d.exact_relaxed_hash {relaxed_hash_filter}
            GROUP BY 1, 2
        )
        SELECT
            COALESCE(s.dropped_source, r.dropped_source) AS dropped_source,
            COALESCE(s.kept_source, r.kept_source) AS kept_source,
            COALESCE(strict_drops, 0) AS strict_drops,
            COALESCE(relaxed_drops, 0) AS relaxed_drops,
            COALESCE(strict_drops, 0) + COALESCE(relaxed_drops, 0) AS total_drops
        FROM strict_cross AS s
        FULL OUTER JOIN relaxed_cross AS r
          ON s.dropped_source = r.dropped_source
         AND s.kept_source = r.kept_source
        """
    ).df()


def recommendation_for_pair(source_a: str, source_b: str) -> tuple[str, str, str]:
    if source_a == source_b:
        return (source_a, source_b, "Internal duplicates; keep the best representative within the same source.")
    policy_a = policy_for(source_a)
    policy_b = policy_for(source_b)
    roles = {policy_a["role"], policy_b["role"]}
    if "aggregator" in roles and (
        "canonical_government" in roles
        or "canonical_legal" in roles
        or "canonical_institutional" in roles
        or "canonical_thesis" in roles
        or "standalone_repository" in roles
        or "curated_book_corpus" in roles
    ):
        if policy_a["role"] == "aggregator":
            return (
                source_b,
                source_a,
                "Prefer the direct dedicated source over the OpenArchives aggregator mirror for this pair.",
            )
        return (
            source_a,
            source_b,
            "Prefer the direct dedicated source over the OpenArchives aggregator mirror for this pair.",
        )
    if "pdf_mixture" in roles and len(roles) > 1:
        if policy_a["role"] == "pdf_mixture":
            return (
                source_b,
                source_a,
                "Prefer the more structured source over the generic PDF mixture when the text is an exact duplicate.",
            )
        return (
            source_a,
            source_b,
            "Prefer the more structured source over the generic PDF mixture when the text is an exact duplicate.",
        )
    if policy_a["rank"] < policy_b["rank"]:
        return (
            source_a,
            source_b,
            f"Prefer `{source_a}` ({policy_a['role']}) over `{source_b}` ({policy_b['role']}) on provenance grounds.",
        )
    if policy_b["rank"] < policy_a["rank"]:
        return (
            source_b,
            source_a,
            f"Prefer `{source_b}` ({policy_b['role']}) over `{source_a}` ({policy_a['role']}) on provenance grounds.",
        )
    return ("review", "review", "No strong provenance preference is encoded for this pair; review the pair manually.")


def build_source_influence(con: duckdb.DuckDBPyConnection, analysis_root: Path) -> pd.DataFrame:
    source_summary = pd.read_csv(analysis_root / "source_summary.csv")
    source_summary = numericize(
        source_summary,
        [
            "total_rows",
            "strict_dropped_rows",
            "relaxed_only_dropped_rows",
            "total_dropped_rows",
            "kept_after_exact_rows",
        ],
    )
    meaningful_drop_flows = query_directed_cross_drops(
        con,
        strict_hash_filter="NOT IN (SELECT group_hash FROM strict_low_info_groups)",
        relaxed_hash_filter="NOT IN (SELECT group_hash FROM relaxed_low_info_groups)",
    )
    meaningful_drop_flows = numericize(meaningful_drop_flows, ["total_drops"])
    low_info_drop_flows = query_directed_cross_drops(
        con,
        strict_hash_filter="IN (SELECT group_hash FROM strict_low_info_groups)",
        relaxed_hash_filter="IN (SELECT group_hash FROM relaxed_low_info_groups)",
    )
    low_info_drop_flows = numericize(low_info_drop_flows, ["total_drops"])

    internal = (
        pd.read_csv(analysis_root / "final_drop_by_kept_source.csv")
        .pipe(numericize, ["dropped_rows"])
        .query("dropped_source_dataset == kept_source_dataset")
        .groupby("dropped_source_dataset", as_index=False)["dropped_rows"]
        .sum()
        .rename(columns={"dropped_source_dataset": "source_dataset", "dropped_rows": "internal_dropped_rows"})
    )
    cross_out = (
        meaningful_drop_flows.groupby("dropped_source", as_index=False)["total_drops"]
        .sum()
        .rename(columns={"dropped_source": "source_dataset", "total_drops": "cross_source_dropped_rows"})
    )
    cross_in = (
        meaningful_drop_flows.groupby("kept_source", as_index=False)["total_drops"]
        .sum()
        .rename(columns={"kept_source": "source_dataset", "total_drops": "rows_kept_from_other_sources"})
    )
    low_info_cross_out = (
        low_info_drop_flows.groupby("dropped_source", as_index=False)["total_drops"]
        .sum()
        .rename(columns={"dropped_source": "source_dataset", "total_drops": "low_information_cross_source_dropped_rows"})
    )

    frame = source_summary.merge(internal, how="left", on="source_dataset")
    frame = frame.merge(cross_out, how="left", on="source_dataset")
    frame = frame.merge(cross_in, how="left", on="source_dataset")
    frame = frame.merge(low_info_cross_out, how="left", on="source_dataset")
    frame = frame.fillna(0)
    frame["drop_rate_pct"] = (frame["total_dropped_rows"] / frame["total_rows"]) * 100.0
    frame["internal_share_pct"] = frame.apply(
        lambda row: (row["internal_dropped_rows"] / row["total_dropped_rows"] * 100.0) if row["total_dropped_rows"] else 0.0,
        axis=1,
    )
    frame["cross_source_share_pct"] = frame.apply(
        lambda row: (row["cross_source_dropped_rows"] / row["total_dropped_rows"] * 100.0) if row["total_dropped_rows"] else 0.0,
        axis=1,
    )
    frame["low_information_cross_source_share_pct"] = frame.apply(
        lambda row: (row["low_information_cross_source_dropped_rows"] / row["total_dropped_rows"] * 100.0)
        if row["total_dropped_rows"]
        else 0.0,
        axis=1,
    )
    frame["role"] = frame["source_dataset"].map(lambda value: policy_for(str(value))["role"])
    return frame.sort_values(["total_dropped_rows", "drop_rate_pct", "source_dataset"], ascending=[False, False, True])


def build_pair_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    pair_overlap = query_pair_overlap(
        con,
        strict_view="strict_groups_filtered",
        relaxed_view="relaxed_groups_filtered",
    )
    low_info_pair_overlap = query_pair_overlap(
        con,
        strict_view="strict_groups_low_info",
        relaxed_view="relaxed_groups_low_info",
    )

    directed = query_directed_cross_drops(
        con,
        strict_hash_filter="NOT IN (SELECT group_hash FROM strict_low_info_groups)",
        relaxed_hash_filter="NOT IN (SELECT group_hash FROM relaxed_low_info_groups)",
    )
    low_info_directed = query_directed_cross_drops(
        con,
        strict_hash_filter="IN (SELECT group_hash FROM strict_low_info_groups)",
        relaxed_hash_filter="IN (SELECT group_hash FROM relaxed_low_info_groups)",
    )

    flow_map = {
        (str(row["dropped_source"]), str(row["kept_source"])): int(row["total_drops"])
        for row in directed.to_dict(orient="records")
    }
    low_info_flow_map = {
        (str(row["dropped_source"]), str(row["kept_source"])): int(row["total_drops"])
        for row in low_info_directed.to_dict(orient="records")
    }
    low_info_pair_map = {
        (str(row["source_a"]), str(row["source_b"])): row
        for row in low_info_pair_overlap.to_dict(orient="records")
    }

    filtered_pair_map = {
        (str(row["source_a"]), str(row["source_b"])): row
        for row in pair_overlap.to_dict(orient="records")
    }

    rows: list[dict[str, Any]] = []
    for source_a, source_b in sorted(set(filtered_pair_map) | set(low_info_pair_map)):
        row = filtered_pair_map.get(
            (source_a, source_b),
            {
                "strict_shared_groups": 0,
                "relaxed_shared_groups": 0,
                "source_a_rows_in_shared_groups": 0,
                "source_b_rows_in_shared_groups": 0,
            },
        )
        keep_source, drop_source, rationale = recommendation_for_pair(source_a, source_b)
        low_info_row = low_info_pair_map.get((source_a, source_b), {})
        rows.append(
            {
                "source_a": source_a,
                "source_b": source_b,
                "strict_shared_groups": int(row["strict_shared_groups"]),
                "relaxed_shared_groups": int(row["relaxed_shared_groups"]),
                "source_a_rows_in_shared_groups": int(row["source_a_rows_in_shared_groups"]),
                "source_b_rows_in_shared_groups": int(row["source_b_rows_in_shared_groups"]),
                "total_rows_in_shared_groups": int(row["source_a_rows_in_shared_groups"]) + int(row["source_b_rows_in_shared_groups"]),
                "current_drops_a_to_b": int(flow_map.get((source_a, source_b), 0)),
                "current_drops_b_to_a": int(flow_map.get((source_b, source_a), 0)),
                "excluded_low_information_groups": int(low_info_row.get("strict_shared_groups", 0))
                + int(low_info_row.get("relaxed_shared_groups", 0)),
                "excluded_low_information_rows_a": int(low_info_row.get("source_a_rows_in_shared_groups", 0)),
                "excluded_low_information_rows_b": int(low_info_row.get("source_b_rows_in_shared_groups", 0)),
                "excluded_low_information_drops_a_to_b": int(low_info_flow_map.get((source_a, source_b), 0)),
                "excluded_low_information_drops_b_to_a": int(low_info_flow_map.get((source_b, source_a), 0)),
                "recommended_keep_source": keep_source,
                "recommended_drop_source": drop_source,
                "recommendation_rationale": rationale,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["current_total_cross_drops"] = frame["current_drops_a_to_b"] + frame["current_drops_b_to_a"]
    frame["excluded_low_information_total_cross_drops"] = (
        frame["excluded_low_information_drops_a_to_b"] + frame["excluded_low_information_drops_b_to_a"]
    )
    return frame.sort_values(
        ["total_rows_in_shared_groups", "current_total_cross_drops", "source_a", "source_b"],
        ascending=[False, False, True, True],
    )


def build_oa_collection_pairs(con: duckdb.DuckDBPyConnection, analysis_root: Path) -> pd.DataFrame:
    oa_map = pd.read_csv(analysis_root / "openarchives_collection_map.csv")
    if oa_map.empty:
        return pd.DataFrame()
    con.register("oa_collection_map", oa_map)

    overlaps = con.execute(
        """
        WITH oa_members AS (
            SELECT
                sg.group_hash,
                COALESCE(map.oa_collection_slug, 'unknown') AS oa_collection_slug,
                sg.member_doc_key
            FROM strict_groups_filtered AS sg
            LEFT JOIN oa_collection_map AS map
              ON map.source_doc_id = sg.member_source_doc_id
            WHERE sg.member_source_dataset = 'openarchives.gr'
        ),
        counterparts AS (
            SELECT DISTINCT
                group_hash,
                member_source_dataset AS counterpart_source_dataset
            FROM strict_groups_filtered
            WHERE member_source_dataset <> 'openarchives.gr'
        )
        SELECT
            oa_collection_slug,
            counterpart_source_dataset,
            COUNT(DISTINCT member_doc_key) AS strict_overlap_rows
        FROM oa_members
        JOIN counterparts USING (group_hash)
        GROUP BY 1, 2
        """
    ).df()
    overlaps = numericize(overlaps, ["strict_overlap_rows"])

    low_info_overlaps = con.execute(
        """
        WITH oa_members AS (
            SELECT
                sg.group_hash,
                COALESCE(map.oa_collection_slug, 'unknown') AS oa_collection_slug,
                sg.member_doc_key
            FROM strict_groups_low_info AS sg
            LEFT JOIN oa_collection_map AS map
              ON map.source_doc_id = sg.member_source_doc_id
            WHERE sg.member_source_dataset = 'openarchives.gr'
        ),
        counterparts AS (
            SELECT DISTINCT
                group_hash,
                member_source_dataset AS counterpart_source_dataset
            FROM strict_groups_low_info
            WHERE member_source_dataset <> 'openarchives.gr'
        )
        SELECT
            oa_collection_slug,
            counterpart_source_dataset,
            COUNT(DISTINCT member_doc_key) AS excluded_low_information_oa_rows
        FROM oa_members
        JOIN counterparts USING (group_hash)
        GROUP BY 1, 2
        """
    ).df()
    low_info_overlaps = numericize(low_info_overlaps, ["excluded_low_information_oa_rows"])

    drops = con.execute(
        """
        WITH strict_cross AS (
            SELECT
                COALESCE(map.oa_collection_slug, 'unknown') AS oa_collection_slug,
                kept.source_dataset AS counterpart_source_dataset,
                COUNT(*) AS strict_drops
            FROM docs AS dropped
            JOIN docs AS kept
              ON kept.doc_key = dropped.strict_kept_doc_key
            LEFT JOIN oa_collection_map AS map
              ON map.source_doc_id = dropped.source_doc_id
            WHERE dropped.strict_dropped = 1
              AND dropped.source_dataset = 'openarchives.gr'
              AND kept.source_dataset <> 'openarchives.gr'
              AND dropped.exact_strict_hash NOT IN (SELECT group_hash FROM strict_low_info_groups)
            GROUP BY 1, 2
        ),
        relaxed_cross AS (
            SELECT
                COALESCE(map.oa_collection_slug, 'unknown') AS oa_collection_slug,
                kept.source_dataset AS counterpart_source_dataset,
                COUNT(*) AS relaxed_drops
            FROM docs AS dropped
            JOIN docs AS kept
              ON kept.doc_key = dropped.relaxed_kept_doc_key
            LEFT JOIN oa_collection_map AS map
              ON map.source_doc_id = dropped.source_doc_id
            WHERE dropped.strict_dropped = 0
              AND COALESCE(dropped.relaxed_dropped, 0) = 1
              AND dropped.source_dataset = 'openarchives.gr'
              AND kept.source_dataset <> 'openarchives.gr'
              AND dropped.exact_relaxed_hash NOT IN (SELECT group_hash FROM relaxed_low_info_groups)
            GROUP BY 1, 2
        )
        SELECT
            COALESCE(s.oa_collection_slug, r.oa_collection_slug) AS oa_collection_slug,
            COALESCE(s.counterpart_source_dataset, r.counterpart_source_dataset) AS counterpart_source_dataset,
            COALESCE(strict_drops, 0) + COALESCE(relaxed_drops, 0) AS current_oa_drops_to_counterpart
        FROM strict_cross AS s
        FULL OUTER JOIN relaxed_cross AS r
          ON s.oa_collection_slug = r.oa_collection_slug
         AND s.counterpart_source_dataset = r.counterpart_source_dataset
        """
    ).df()
    drops = numericize(drops, ["current_oa_drops_to_counterpart"])

    frame = overlaps.merge(drops, how="outer", on=["oa_collection_slug", "counterpart_source_dataset"])
    frame = frame.merge(low_info_overlaps, how="outer", on=["oa_collection_slug", "counterpart_source_dataset"])
    frame = frame.fillna(0)
    recommendations: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        oa_collection_slug = str(row["oa_collection_slug"])
        counterpart = str(row["counterpart_source_dataset"])
        keep_source, drop_source, rationale = recommendation_for_pair("openarchives.gr", counterpart)
        recommendations.append(
            {
                "oa_collection_slug": oa_collection_slug,
                "counterpart_source_dataset": counterpart,
                "oa_rows_in_shared_groups": int(row["strict_overlap_rows"]),
                "current_oa_drops_to_counterpart": int(row["current_oa_drops_to_counterpart"]),
                "excluded_low_information_oa_rows": int(row["excluded_low_information_oa_rows"]),
                "recommended_keep_source": keep_source,
                "recommended_drop_source": drop_source,
                "recommendation_rationale": rationale,
            }
        )
    result = pd.DataFrame(recommendations)
    if result.empty:
        return result
    return result.sort_values(
        ["oa_rows_in_shared_groups", "current_oa_drops_to_counterpart", "oa_collection_slug", "counterpart_source_dataset"],
        ascending=[False, False, True, True],
    )


def build_metadata_comparison(analysis_root: Path, oa_collection_pairs: pd.DataFrame) -> pd.DataFrame:
    metadata = load_json(analysis_root / "metadata_comparison.json")
    oa_to_greek_phd = int(
        oa_collection_pairs.loc[
            oa_collection_pairs["counterpart_source_dataset"] == "greek_phd",
            "current_oa_drops_to_counterpart",
        ].sum()
    )
    rows = [
        {
            "comparison_case": "OA kallipos vs standalone Kallipos",
            "metadata_signal": "raw OA metadata drop-list",
            "metadata_count": int(metadata["metadata_baselines"]["openarchives_raw_drop_list"]["kallipos_raw_drop_rows"]),
            "exact_count": int(metadata["exact_published"]["openarchives"]["kallipos_to_kallipos_source_drops"]),
            "interpretation": "Metadata identifies many repository duplicates, but exact published text found none. Exact text equality is too strict to replace the metadata workflow here.",
        },
        {
            "comparison_case": "OA pergamos vs standalone Pergamos",
            "metadata_signal": "raw OA metadata drop-list",
            "metadata_count": int(metadata["metadata_baselines"]["openarchives_raw_drop_list"]["pergamos_raw_drop_rows"]),
            "exact_count": int(metadata["exact_published"]["openarchives"]["pergamos_to_pergamos_source_drops"]),
            "interpretation": "Again, metadata sees many mirrored records while exact published text sees zero direct duplicates. The text layer is not reproducing the metadata baseline.",
        },
        {
            "comparison_case": "OA phdtheses/pergamos targeted thesis dedup vs greek_phd",
            "metadata_signal": "title/author/year thesis matching",
            "metadata_count": int(metadata["metadata_baselines"]["phd_cross_collection"]["pergamos_after_all_tested_layers"]),
            "exact_count": int(metadata["exact_published"]["phd"]["oa_phdtheses_to_greek_phd_drops"])
            + int(metadata["exact_published"]["phd"]["oa_pergamos_to_greek_phd_drops"]),
            "interpretation": "The targeted thesis metadata baseline is much stronger than exact text equality for these OA collections. Stage 1 exact dedup is not a substitute for that metadata work.",
        },
        {
            "comparison_case": "Other OA collections vs greek_phd",
            "metadata_signal": "not the main metadata target; placeholder-only overlaps excluded",
            "metadata_count": 0,
            "exact_count": oa_to_greek_phd,
            "interpretation": "Meaningful exact text overlap with greek_phd exists after excluding low-information placeholder groups, but it is concentrated in OA collections like Pandemos, IKEE_AUT, and ntua rather than the earlier targeted phdtheses/pergamos metadata cases.",
        },
    ]
    return pd.DataFrame(rows)


def build_report(
    *,
    run_id: str,
    exact_summary: dict[str, Any],
    source_influence: pd.DataFrame,
    pair_summary: pd.DataFrame,
    oa_collection_pairs: pd.DataFrame,
    low_information_groups: pd.DataFrame,
    metadata_comparison: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"# Exact Dedup Policy Analysis: {run_id}")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- Total rows: {exact_summary['total_rows']}")
    lines.append(f"- Kept after exact: {exact_summary['kept_after_exact_rows']}")
    lines.append(f"- Strict dropped rows: {exact_summary['strict']['dropped_rows']}")
    lines.append(f"- Relaxed dropped rows: {exact_summary['relaxed']['dropped_rows']}")
    lines.append("")
    lines.append("## Source Influence")
    for row in source_influence.head(10).to_dict(orient="records"):
        lines.append(
            f"- {row['source_dataset']}: dropped={int(row['total_dropped_rows'])} "
            f"({row['drop_rate_pct']:.2f}% of source), internal_share={row['internal_share_pct']:.2f}%, "
            f"meaningful_cross_source_share={row['cross_source_share_pct']:.2f}%, "
            f"low_info_cross_source_share={row['low_information_cross_source_share_pct']:.2f}%, "
            f"role={row['role']}"
        )
    lines.append("")
    lines.append("Pattern: exact-stage loss is dominated by internal repetition inside `openarchives.gr`, `eurlex-greek-legislation`, and `HuggingFaceFW/finewiki`. Cross-source exact overlap exists, but the policy view now excludes low-information placeholder groups before ranking provenance-sensitive pairs.")
    lines.append("")
    lines.append("## Excluded Low-Information Cross-Source Groups")
    lines.append(
        f"- Excluded groups: {len(low_information_groups)}"
    )
    lines.append(
        f"- Excluded member rows: {int(low_information_groups['group_size'].sum()) if not low_information_groups.empty else 0}"
    )
    if not low_information_groups.empty:
        for row in low_information_groups.head(8).to_dict(orient="records"):
            lines.append(
                f"- {row['stage']} {row['sources_in_group']}: group_size={int(row['group_size'])}, "
                f"reason={row['low_information_reason']}, preview={row['text_preview']!r}"
            )
    lines.append("")
    lines.append("## Cross-Source Pairs")
    meaningful_pairs = pair_summary[pair_summary["total_rows_in_shared_groups"] > 0]
    if meaningful_pairs.empty:
        lines.append("- No meaningful cross-source pairs remain after excluding low-information groups.")
    else:
        for row in meaningful_pairs.head(8).to_dict(orient="records"):
            lines.append(
                f"- {row['source_a']} <-> {row['source_b']}: shared_rows={int(row['total_rows_in_shared_groups'])}, "
                f"shared_groups(strict={int(row['strict_shared_groups'])}, relaxed={int(row['relaxed_shared_groups'])}), "
                f"current_drops={int(row['current_total_cross_drops'])}, "
                f"excluded_low_info_groups={int(row['excluded_low_information_groups'])}, "
                f"recommended_keep={row['recommended_keep_source']}"
            )
    lines.append("")
    lines.append("Pattern: the earlier `openarchives.gr` versus `opengov.gr-diaboyleuseis` signal was an artifact of placeholder-only extraction failures such as repeated `<!-- image -->`. After excluding those low-information groups, the meaningful provenance-sensitive pairs are much smaller and more plausible.")
    lines.append("")
    lines.append("## Grounded Drop-Side Recommendations")
    for row in meaningful_pairs.head(8).to_dict(orient="records"):
        if row["recommended_keep_source"] == "review":
            continue
        lines.append(
            f"- Prefer `{row['recommended_keep_source']}` over `{row['recommended_drop_source']}` for the pair "
            f"`{row['source_a']}` / `{row['source_b']}`. Reason: {row['recommendation_rationale']}"
        )
    lines.append("")
    lines.append("## OpenArchives Collection-Level Findings")
    meaningful_oa_pairs = oa_collection_pairs[
        (oa_collection_pairs["oa_rows_in_shared_groups"] > 0)
        | (oa_collection_pairs["current_oa_drops_to_counterpart"] > 0)
    ]
    for row in meaningful_oa_pairs.head(10).to_dict(orient="records"):
        lines.append(
            f"- {row['oa_collection_slug']} vs {row['counterpart_source_dataset']}: "
            f"OA rows in shared groups={int(row['oa_rows_in_shared_groups'])}, "
            f"current OA drops to counterpart={int(row['current_oa_drops_to_counterpart'])}, "
            f"excluded low-info OA rows={int(row['excluded_low_information_oa_rows'])}, "
            f"recommended_keep={row['recommended_keep_source']}"
        )
    lines.append("")
    lines.append("Pattern: OpenArchives exact overlap with `greek_phd` is not mainly coming from the earlier targeted `phdtheses` / `pergamos` metadata cases. After excluding low-information groups, it is concentrated in other OA collections such as `Pandemos`, `IKEE_AUT`, and `ntua`.")
    lines.append("")
    lines.append("## Exact Vs Metadata")
    for row in metadata_comparison.to_dict(orient="records"):
        lines.append(
            f"- {row['comparison_case']}: metadata={int(row['metadata_count'])}, exact={int(row['exact_count'])}. "
            f"{row['interpretation']}"
        )
    lines.append("")
    lines.append("Conclusion: exact dedup is useful for cheap, high-confidence row removal, but provenance conclusions must ignore low-information placeholder collisions. Even after that correction, exact text dedup does not reproduce the earlier metadata-driven academic dedup work. For OA / repository / thesis overlaps, metadata and provenance-aware pair policies remain necessary, and Stage 2 near-dup is likely required to bridge the gap.")
    return "\n".join(lines) + "\n"


def analyze_policy(run_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    analysis_root = (output_dir or (run_root / "analysis")).resolve()
    exact_summary = load_json(run_root / "stage_01_exact" / "summary.json")
    low_information_groups = build_low_information_groups(
        snapshot_manifest_path=Path(exact_summary["snapshot_manifest_path"]),
        strict_groups_path=Path(exact_summary["strict_exact_groups_path"]),
        relaxed_groups_path=Path(exact_summary["relaxed_exact_groups_path"]),
    )
    strict_low_info_hashes = set(
        low_information_groups.loc[low_information_groups["stage"] == "strict", "group_hash"].astype(str)
    )
    relaxed_low_info_hashes = set(
        low_information_groups.loc[low_information_groups["stage"] == "relaxed", "group_hash"].astype(str)
    )

    con = duckdb.connect()
    con.execute(f"CREATE VIEW docs AS SELECT * FROM read_parquet('{exact_summary['docs_exact_path']}')")
    con.execute(f"CREATE VIEW strict_groups AS SELECT * FROM read_parquet('{exact_summary['strict_exact_groups_path']}')")
    con.execute(f"CREATE VIEW relaxed_groups AS SELECT * FROM read_parquet('{exact_summary['relaxed_exact_groups_path']}')")
    create_filtered_views(
        con,
        strict_low_info_hashes=strict_low_info_hashes,
        relaxed_low_info_hashes=relaxed_low_info_hashes,
    )

    source_influence = build_source_influence(con, analysis_root)
    pair_summary = build_pair_summary(con)
    oa_collection_pairs = build_oa_collection_pairs(con, analysis_root)
    metadata_comparison = build_metadata_comparison(analysis_root, oa_collection_pairs)

    source_influence_path = analysis_root / "policy_source_influence.csv"
    pair_summary_path = analysis_root / "policy_pair_summary.csv"
    oa_collection_pairs_path = analysis_root / "policy_openarchives_collection_pairs.csv"
    metadata_comparison_path = analysis_root / "policy_metadata_vs_exact.csv"
    low_information_groups_path = analysis_root / "policy_low_information_groups.csv"
    report_path = analysis_root / "policy_report.md"

    write_csv(source_influence, source_influence_path)
    write_csv(pair_summary, pair_summary_path)
    write_csv(oa_collection_pairs, oa_collection_pairs_path)
    write_csv(metadata_comparison, metadata_comparison_path)
    write_csv(low_information_groups, low_information_groups_path)
    report_path.write_text(
        build_report(
            run_id=str(exact_summary["run_id"]),
            exact_summary=exact_summary,
            source_influence=source_influence,
            pair_summary=pair_summary,
            oa_collection_pairs=oa_collection_pairs,
            low_information_groups=low_information_groups,
            metadata_comparison=metadata_comparison,
        )
    )

    payload = {
        "run_id": str(exact_summary["run_id"]),
        "analysis_root": str(analysis_root),
        "policy_source_influence_path": str(source_influence_path),
        "policy_pair_summary_path": str(pair_summary_path),
        "policy_openarchives_collection_pairs_path": str(oa_collection_pairs_path),
        "policy_metadata_vs_exact_path": str(metadata_comparison_path),
        "policy_low_information_groups_path": str(low_information_groups_path),
        "policy_report_path": str(report_path),
    }
    (analysis_root / "policy_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exact dedup policy implications and metadata comparisons.")
    parser.add_argument("--run-root", required=True, type=Path, help="Path to a completed exact-stage run root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory; defaults to <run_root>/analysis",
    )
    args = parser.parse_args()
    payload = analyze_policy(run_root=args.run_root, output_dir=args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
