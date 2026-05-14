from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


IMAGE_PLACEHOLDER_RE = re.compile(
    r"(?:<!--\s*image\s*-->|!\[[^\]]*\]\([^)]*\)|<img\b[^>]*>|\[\s*image\s*\]|\(\s*image\s*\))",
    flags=re.IGNORECASE,
)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SYMBOL_SPACE_RE = re.compile(r"[\s\-\–\—\.\,\:\;\!\?\"'“”‘’\(\)\[\]\{\}/\\|_*+=~`<>·•…]+")
WHITESPACE_RE = re.compile(r"\s+")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        df = pd.DataFrame(columns=df.columns)
    df.to_csv(path, index=False)


def normalize_loose_text(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value)).casefold()
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


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


def build_oa_collection_map(file_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in file_paths:
        parquet_file = pq.ParquetFile(path)
        if "source_metadata_json" not in parquet_file.schema_arrow.names:
            continue
        for batch in parquet_file.iter_batches(columns=["source_doc_id", "source_metadata_json"], batch_size=2048):
            for row in batch.to_pylist():
                source_doc_id = row.get("source_doc_id")
                if not source_doc_id:
                    continue
                metadata_json = row.get("source_metadata_json")
                collection_slug = "unknown"
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                    except json.JSONDecodeError:
                        metadata = {}
                    if isinstance(metadata, dict) and metadata.get("collection_slug"):
                        collection_slug = str(metadata["collection_slug"])
                rows.append({"source_doc_id": str(source_doc_id), "oa_collection_slug": collection_slug})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["source_doc_id", "oa_collection_slug"])
    return frame.drop_duplicates(subset=["source_doc_id"], keep="first")


def lookup_rows(file_paths: list[Path], needed_ids: set[str]) -> pd.DataFrame:
    columns = [
        "source_doc_id",
        "text",
        "title",
        "author",
        "source_metadata_json",
        "greek_badness_score",
        "mojibake_badness_score",
        "needs_ocr",
        "is_empty",
        "ocr_success",
        "is_historical_or_polytonic",
    ]
    rows: list[dict[str, Any]] = []
    remaining = set(needed_ids)
    for file_path in file_paths:
        if not remaining:
            break
        parquet_file = pq.ParquetFile(file_path)
        available = [column for column in columns if column in parquet_file.schema_arrow.names]
        for batch in parquet_file.iter_batches(columns=available, batch_size=2048):
            for row in batch.to_pylist():
                source_doc_id = row.get("source_doc_id")
                if source_doc_id not in remaining:
                    continue
                payload = {column: row.get(column) for column in available}
                payload["source_doc_id"] = str(source_doc_id)
                rows.append(payload)
                remaining.remove(str(source_doc_id))
            if not remaining:
                break
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame


def derive_group_reason(member_frame: pd.DataFrame) -> str:
    texts = [str(value or "") for value in member_frame["text"].tolist()]
    low_info_reasons = {classify_low_information_text(text) for text in texts}
    low_info_reasons.discard(None)
    if low_info_reasons:
        return sorted(low_info_reasons)[0]

    title_keys = {normalize_loose_text(value) for value in member_frame["title"].tolist() if normalize_loose_text(value)}
    author_keys = {normalize_loose_text(value) for value in member_frame["author"].tolist() if normalize_loose_text(value)}
    content_hashes = {str(value) for value in member_frame["content_hash_raw"].tolist() if value}
    collection_slugs = {str(value) for value in member_frame["oa_collection_slug"].tolist() if value}

    same_title = len(title_keys) <= 1
    same_author = len(author_keys) <= 1
    same_raw_content = len(content_hashes) <= 1
    multi_collection = len(collection_slugs) > 1

    if same_raw_content and same_title and same_author and not multi_collection:
        return "literal_duplicate_same_metadata"
    if same_raw_content and same_title and same_author and multi_collection:
        return "literal_duplicate_cross_collection"
    if same_raw_content and same_title and not same_author:
        return "same_text_title_author_variation"
    if same_raw_content and not same_title and same_author:
        return "same_text_author_title_variation"
    if same_raw_content and not same_title and not same_author:
        return "same_text_metadata_variation"
    if not same_raw_content and same_title and same_author:
        return "normalization_only_text_duplicate"
    if same_title and same_author:
        return "same_title_author_text_variation"
    if same_title:
        return "same_title_text_duplicate"
    return "exact_text_duplicate_metadata_mixed"


def build_group_summary(sample_groups: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in sample_groups.to_dict(orient="records"):
        group_hash = str(row["group_hash"])
        group_members = members[members["group_hash"] == group_hash].copy()
        title_keys = {normalize_loose_text(value) for value in group_members["title"].tolist() if normalize_loose_text(value)}
        author_keys = {normalize_loose_text(value) for value in group_members["author"].tolist() if normalize_loose_text(value)}
        collections = Counter(str(value) for value in group_members["oa_collection_slug"].fillna("unknown").tolist())
        content_hashes = {str(value) for value in group_members["content_hash_raw"].tolist() if value}
        representative = (
            group_members[group_members["doc_key"] == row["kept_doc_key"]].head(1)
            if row["kept_doc_key"]
            else group_members.head(1)
        )
        representative_text = ""
        representative_title = None
        if not representative.empty:
            representative_text = str(representative.iloc[0].get("text") or "")
            representative_title = representative.iloc[0].get("title")
        rows.append(
            {
                "group_hash": group_hash,
                "group_size": int(row["group_size"]),
                "kept_doc_key": row["kept_doc_key"],
                "kept_source_doc_id": row["kept_source_doc_id"],
                "collection_count": int(len(collections)),
                "collections_in_group": "|".join(f"{key}:{value}" for key, value in sorted(collections.items())),
                "unique_title_count": int(len(title_keys)),
                "unique_author_count": int(len(author_keys)),
                "unique_raw_content_hash_count": int(len(content_hashes)),
                "derived_reason": derive_group_reason(group_members),
                "representative_title": representative_title,
                "representative_text_chars": len(representative_text),
                "representative_text_preview": representative_text[:240].replace("\n", "\\n"),
            }
        )
    return pd.DataFrame(rows).sort_values(["group_size", "group_hash"], ascending=[False, True])


def build_member_export(
    sample_groups: pd.DataFrame,
    docs_exact: pd.DataFrame,
    snapshot_manifest: pd.DataFrame,
    oa_collection_map: pd.DataFrame,
    source_rows: pd.DataFrame,
) -> pd.DataFrame:
    group_members = docs_exact[docs_exact["exact_strict_hash"].isin(sample_groups["group_hash"])].copy()
    group_members = group_members[group_members["source_dataset"] == "openarchives.gr"].copy()
    group_members = group_members.merge(
        sample_groups[["group_hash", "kept_doc_key"]].rename(columns={"group_hash": "exact_strict_hash"}),
        how="left",
        on="exact_strict_hash",
    )
    group_members = group_members.merge(
        snapshot_manifest[["doc_key", "content_hash_raw"]],
        how="left",
        on="doc_key",
        suffixes=("", "_manifest"),
    )
    if "content_hash_raw_manifest" in group_members.columns:
        group_members["content_hash_raw"] = group_members["content_hash_raw"].fillna(group_members["content_hash_raw_manifest"])
        group_members = group_members.drop(columns=["content_hash_raw_manifest"])
    group_members = group_members.merge(oa_collection_map, how="left", on="source_doc_id")
    group_members = group_members.merge(
        source_rows,
        how="left",
        on="source_doc_id",
        suffixes=("", "_source"),
    )
    group_members["oa_collection_slug"] = group_members["oa_collection_slug"].fillna("unknown")
    group_members["is_kept"] = group_members["doc_key"] == group_members["kept_doc_key"]
    group_members["normalized_title_key"] = group_members["title"].map(normalize_loose_text)
    group_members["normalized_author_key"] = group_members["author"].map(normalize_loose_text)
    group_members["low_information_reason"] = group_members["text"].map(classify_low_information_text)
    group_members["text_preview"] = group_members["text"].map(lambda value: (value or "")[:240].replace("\n", "\\n"))
    group_members["metadata_preview"] = group_members["source_metadata_json"].map(lambda value: (value or "")[:400].replace("\n", "\\n"))
    group_members["group_hash"] = group_members["exact_strict_hash"]
    return group_members[
        [
            "group_hash",
            "exact_strict_hash",
            "strict_group_size",
            "is_kept",
            "doc_key",
            "source_doc_id",
            "oa_collection_slug",
            "content_hash_raw",
            "title",
            "author",
            "normalized_title_key",
            "normalized_author_key",
            "greek_badness_score",
            "mojibake_badness_score",
            "needs_ocr",
            "is_empty",
            "ocr_success",
            "is_historical_or_polytonic",
            "raw_text_chars",
            "strict_text_chars",
            "text",
            "source_metadata_json",
            "low_information_reason",
            "text_preview",
            "metadata_preview",
        ]
    ].sort_values(["exact_strict_hash", "is_kept", "source_doc_id"], ascending=[True, False, True])


def build_report(*, run_id: str, sample_size: int, seed: int, population_size: int, sampled_groups: pd.DataFrame, sampled_members: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append(f"# OA Internal Exact-Duplicate Sample: {run_id}")
    lines.append("")
    lines.append("## Sampling")
    lines.append(f"- Population: {population_size} OA-only exact duplicate groups")
    lines.append(f"- Sampled groups: {sample_size}")
    lines.append(f"- Seed: {seed}")
    lines.append("")
    lines.append("## Sample Reason Mix")
    reason_counts = sampled_groups["derived_reason"].value_counts()
    for reason, count in reason_counts.items():
        lines.append(f"- {reason}: {int(count)} groups")
    lines.append("")
    lines.append("## Sampled Groups")
    for row in sampled_groups.head(20).to_dict(orient="records"):
        lines.append(
            f"- group={row['group_hash'][:16]}..., size={int(row['group_size'])}, reason={row['derived_reason']}, "
            f"collections={row['collections_in_group']}, titles={int(row['unique_title_count'])}, "
            f"authors={int(row['unique_author_count'])}, raw_hashes={int(row['unique_raw_content_hash_count'])}, "
            f"repr_title={row['representative_title']!r}"
        )
    low_info_members = sampled_members[sampled_members["low_information_reason"].notna()]
    if not low_info_members.empty:
        lines.append("")
        lines.append("## Low-Information Warning")
        lines.append(
            f"- {int(low_info_members['exact_strict_hash'].nunique())} sampled groups contain low-information text such as placeholders or markup-only content."
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- oa_internal_exact_sample_groups.csv")
    lines.append("- oa_internal_exact_sample_members.csv")
    return "\n".join(lines) + "\n"


def analyze_oa_internal_exact_sample(
    *,
    run_root: Path,
    sample_size: int,
    seed: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    stage_root = run_root / "stage_01_exact"
    summary = load_json(stage_root / "summary.json")
    analysis_root = (output_dir or (run_root / "analysis")).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)

    strict_groups = pq.read_table(Path(summary["strict_exact_groups_path"])).to_pandas()
    docs_exact = pq.read_table(Path(summary["docs_exact_path"])).to_pandas()
    snapshot_manifest = pq.read_table(Path(summary["snapshot_manifest_path"])).to_pandas()

    per_group = (
        strict_groups.groupby("group_hash", as_index=False)
        .agg(
            group_size=("group_size", "max"),
            source_count=("member_source_dataset", "nunique"),
            oa_rows=("member_source_dataset", lambda series: int((series == "openarchives.gr").sum())),
        )
    )
    oa_groups = per_group[(per_group["source_count"] == 1) & (per_group["oa_rows"] > 1)].copy()
    if oa_groups.empty:
        raise ValueError("no OA-only exact duplicate groups found in strict_exact_groups")

    kept_map = (
        strict_groups[["group_hash", "kept_doc_key", "member_doc_key", "member_source_doc_id"]]
        .drop_duplicates()
    )
    kept_map = kept_map[kept_map["member_doc_key"] == kept_map["kept_doc_key"]][
        ["group_hash", "kept_doc_key", "member_source_doc_id"]
    ].rename(columns={"member_source_doc_id": "kept_source_doc_id"})
    oa_groups = oa_groups.merge(kept_map, how="left", on="group_hash")

    rng = random.Random(seed)
    population = oa_groups.to_dict(orient="records")
    actual_sample_size = min(sample_size, len(population))
    sampled = pd.DataFrame(rng.sample(population, actual_sample_size))
    sampled = sampled.sort_values(["group_size", "group_hash"], ascending=[False, True]).reset_index(drop=True)

    file_map = build_source_file_map(Path(summary["snapshot_manifest_path"]))
    oa_files = file_map.get("openarchives.gr", [])
    oa_collection_map = build_oa_collection_map(oa_files)
    sample_doc_ids = set(
        strict_groups.loc[
            strict_groups["group_hash"].isin(sampled["group_hash"]) & (strict_groups["member_source_dataset"] == "openarchives.gr"),
            "member_source_doc_id",
        ].astype(str)
    )
    source_rows = lookup_rows(oa_files, sample_doc_ids)

    sampled_members = build_member_export(
        sampled,
        docs_exact,
        snapshot_manifest,
        oa_collection_map,
        source_rows,
    )
    sampled_groups = build_group_summary(sampled, sampled_members)

    groups_path = analysis_root / "oa_internal_exact_sample_groups.csv"
    members_path = analysis_root / "oa_internal_exact_sample_members.csv"
    report_path = analysis_root / "oa_internal_exact_sample_report.md"

    write_csv(sampled_groups, groups_path)
    write_csv(sampled_members, members_path)
    report_path.write_text(
        build_report(
            run_id=str(summary["run_id"]),
            sample_size=actual_sample_size,
            seed=seed,
            population_size=len(oa_groups),
            sampled_groups=sampled_groups,
            sampled_members=sampled_members,
        )
    )

    payload = {
        "run_id": str(summary["run_id"]),
        "sample_size": actual_sample_size,
        "seed": seed,
        "population_size": int(len(oa_groups)),
        "analysis_root": str(analysis_root),
        "oa_internal_exact_sample_groups_path": str(groups_path),
        "oa_internal_exact_sample_members_path": str(members_path),
        "oa_internal_exact_sample_report_path": str(report_path),
    }
    (analysis_root / "oa_internal_exact_sample_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample random OpenArchives-only exact duplicate groups for manual audit.")
    parser.add_argument("--run-root", required=True, type=Path, help="Path to a completed exact-stage run root")
    parser.add_argument("--sample-size", type=int, default=25, help="Number of OA duplicate groups to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory; defaults to <run_root>/analysis")
    args = parser.parse_args()
    payload = analyze_oa_internal_exact_sample(
        run_root=args.run_root,
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
