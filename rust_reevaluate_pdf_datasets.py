#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import zstandard as zstd

import glossapi_rs_noise


WORK_ROOT = Path(os.environ.get("GLOSSAPI_WORK_ROOT", str(Path(__file__).resolve().parent)))
RAW_ROOT = Path(os.environ.get("GLOSSAPI_RAW_ROOT", "/home/foivos/data/glossapi_raw"))
OUT_ROOT = WORK_ROOT / "reeval"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_stem(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "doc"


def build_sxolika() -> pd.DataFrame:
    path = RAW_ROOT / "hf" / "Sxolika_vivlia" / "Schoolbook_clean.parquet"
    df = pd.read_parquet(path).copy()
    df["source_dataset"] = "Sxolika_vivlia"
    df["source_doc_id"] = df.index.map(lambda i: f"sxolika_{i:06d}")
    df["filename"] = df["source_doc_id"] + ".md"
    df["text"] = df["text"].fillna("").astype(str)
    cols = ["source_dataset", "source_doc_id", "filename", "text", "titlos", "taxis", "mathima", "kateuthinsi"]
    return df[cols]


def build_pergamos() -> pd.DataFrame:
    section_path = RAW_ROOT / "hf" / "Apothetirio_Pergamos" / "sections_with_annotation_n_metadata.parquet"
    meta_path = RAW_ROOT / "hf" / "Apothetirio_Pergamos" / "metadata.parquet"
    sec = pd.read_parquet(section_path).copy()
    meta = pd.read_parquet(meta_path).copy()
    sec["section"] = sec["section"].fillna("").astype(str)
    sec = sec.sort_values(["filename", "id", "row_id"], kind="stable")
    grouped = (
        sec.groupby("filename", as_index=False)
        .agg(
            text=("section", lambda s: "\n\n".join(x.strip() for x in s if str(x).strip())),
            section_count=("section", "size"),
            predicted_section_values=("predicted_section", lambda s: json.dumps([x for x in s.dropna().tolist()])),
            header_values=("header", lambda s: json.dumps([x for x in s.dropna().tolist()])),
        )
    )
    merged = grouped.merge(meta, on="filename", how="left")
    merged["source_dataset"] = "Apothetirio_Pergamos"
    merged["source_doc_id"] = merged["filename"].astype(str)
    merged["filename"] = merged["filename"].astype(str).str.replace(r"\.pdf$", ".md", regex=True)
    return merged


def build_kallipos() -> pd.DataFrame:
    section_path = RAW_ROOT / "hf" / "Apothetirio_Kallipos" / "Dataset_Kallipos.parquet"
    meta_path = RAW_ROOT / "hf" / "Apothetirio_Kallipos" / "kallipos_repository_metadata_fixed.parquet"
    sec = pd.read_parquet(section_path).copy()
    meta = pd.read_parquet(meta_path).copy()
    sec["section"] = sec["section"].fillna("").astype(str)
    sec = sec.sort_values(["filename", "id", "row_id"], kind="stable")
    grouped = (
        sec.groupby("filename", as_index=False)
        .agg(
            text=("section", lambda s: "\n\n".join(x.strip() for x in s if str(x).strip())),
            section_count=("section", "size"),
            predicted_section_values=("predicted_section", lambda s: json.dumps([x for x in s.dropna().tolist()])),
            header_values=("header", lambda s: json.dumps([x for x in s.dropna().tolist()])),
            document_type_from_sections=("document_type", lambda s: next((x for x in s if pd.notna(x)), None)),
        )
    )
    merged = grouped.merge(meta, on="filename", how="left")
    merged["source_dataset"] = "Apothetirio_Kallipos"
    merged["source_doc_id"] = merged["filename"].astype(str)
    merged["filename"] = merged["filename"].astype(str).str.replace(r"\.pdf$", ".md", regex=True)
    return merged


def build_openbook() -> pd.DataFrame:
    root = RAW_ROOT / "mozilla" / "openbook_gr" / "openbook.gr"
    meta = pd.read_parquet(root / "metadata.parquet").copy()
    meta["source_dataset"] = "openbook_gr"
    meta["source_doc_id"] = meta["filename"].astype(str).str.replace(r"\.md$", "", regex=True)
    meta["text"] = meta["filename"].apply(lambda name: (root / "dataset" / str(name)).read_text(encoding="utf-8"))
    return meta


def build_95k() -> pd.DataFrame:
    path = RAW_ROOT / "hf" / "95k_deigma_ellinikis" / "95k.csv"
    df = pd.read_csv(path).copy()
    df["source_dataset"] = "95k_deigma_ellinikis"
    df["source_doc_id"] = df.index.map(lambda i: f"deigma95k_{i:06d}")
    df["filename"] = df["source_doc_id"] + ".md"
    df["text"] = df["text"].fillna("").astype(str)
    return df


def build_first_1k() -> pd.DataFrame:
    path = RAW_ROOT / "hf" / "1000_prwta_xronia_ellhnikhs" / "1k_texts.parquet"
    df = pd.read_parquet(path).copy()
    df["source_dataset"] = "1000_prwta_xronia_ellhnikhs"
    df["source_doc_id"] = df.index.map(lambda i: f"first1k_{i:06d}")
    df["filename"] = df["source_doc_id"] + ".md"
    df["text"] = df["text"].fillna("").astype(str)
    return df


def build_ekklisiastika() -> pd.DataFrame:
    path = RAW_ROOT / "hf" / "Ekklisiastika_Keimena" / "litourgical_texts.parquet"
    df = pd.read_parquet(path).copy()
    df["source_dataset"] = "Ekklisiastika_Keimena"
    df["source_doc_id"] = df.index.map(lambda i: f"ekkl_{i:06d}")
    df["filename"] = df["source_doc_id"] + ".md"
    df["text"] = df["texts"].fillna("").astype(str)
    return df


def build_gutenberg() -> pd.DataFrame:
    path = RAW_ROOT / "hf" / "Ellinika_Keimena_Project_Gutenberg" / "gutenberg_clean.parquet"
    df = pd.read_parquet(path).copy()
    df["source_dataset"] = "Ellinika_Keimena_Project_Gutenberg"
    fallback_ids = pd.Series(df.index, index=df.index).astype(str)
    df["source_doc_id"] = df["Text Number"].astype("string").fillna(fallback_ids).astype(str)
    df["filename"] = df["source_doc_id"].map(safe_stem) + ".md"
    df["text"] = df["Text"].fillna("").astype(str)
    return df


def build_wikisource() -> pd.DataFrame:
    path = RAW_ROOT / "hf" / "Wikisource_Greek_texts" / "wikisource_greek_deduped.parquet"
    df = pd.read_parquet(path).copy()
    df["source_dataset"] = "Wikisource_Greek_texts"
    df["source_doc_id"] = df.index.map(lambda i: f"wikisource_{i:06d}")
    df["filename"] = df["source_doc_id"] + ".md"
    df["text"] = df["text"].fillna("").astype(str)
    return df


def build_dimodis() -> pd.DataFrame:
    path = RAW_ROOT / "hf" / "dimodis_logotexnia" / "dimodous_mathimata.parquet"
    df = pd.read_parquet(path).copy()
    df["source_dataset"] = "dimodis_logotexnia"
    df["source_doc_id"] = df.index.map(lambda i: f"dimodis_{i:06d}")
    df["filename"] = df["source_doc_id"] + ".md"
    df["text"] = df["text"].fillna("").astype(str)
    return df


def build_europarl_greek() -> pd.DataFrame:
    section_path = RAW_ROOT / "hf" / "ellinika_dedomena_europaikou_koinovouliou" / "ellhnika_dedomena_europaikou_koinovouliou.parquet"
    sec = pd.read_parquet(section_path).copy()
    sec["section"] = sec["section"].fillna("").astype(str)
    sec = sec.sort_values(["filename", "id", "row_id"], kind="stable")
    grouped = (
        sec.groupby("filename", as_index=False)
        .agg(
            text=("section", lambda s: "\n\n".join(x.strip() for x in s if str(x).strip())),
            section_count=("section", "size"),
            predicted_section_values=("predicted_section", lambda s: json.dumps([x for x in s.dropna().tolist()])),
            processing_stage_values=("processing_stage", lambda s: json.dumps([x for x in s.dropna().tolist()])),
        )
    )
    grouped["source_dataset"] = "ellinika_dedomena_europaikou_koinovouliou"
    grouped["source_doc_id"] = grouped["filename"].astype(str)
    grouped["filename"] = grouped["filename"].astype(str) + ".md"
    return grouped


def build_klasikh() -> pd.DataFrame:
    path = RAW_ROOT / "hf" / "klasikh_arx_ell_grammateia" / "Classic_AG_texts_v2.parquet"
    df = pd.read_parquet(path).copy()
    df["source_dataset"] = "klasikh_arx_ell_grammateia"
    df["source_doc_id"] = df.index.map(lambda i: f"klassiki_{i:06d}")
    df["filename"] = df["source_doc_id"] + ".md"
    df["text"] = df["text"].fillna("").astype(str)
    return df


def build_opengov() -> pd.DataFrame:
    db_path = RAW_ROOT / "hf" / "opengov.gr-diaboyleuseis" / "deliberation_data_gr_MIGRATED_FRESH_20250602170747.db"
    query = """
        SELECT
            id,
            title,
            url,
            type,
            status,
            COALESCE(NULLIF(TRIM(content_cleaned), ''), NULLIF(TRIM(processed_text), ''), NULLIF(TRIM(content), '')) AS text
        FROM documents
        WHERE COALESCE(NULLIF(TRIM(content_cleaned), ''), NULLIF(TRIM(processed_text), ''), NULLIF(TRIM(content), '')) IS NOT NULL
    """
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(query, con)
    df["source_dataset"] = "opengov.gr-diaboyleuseis"
    df["source_doc_id"] = df["id"].astype(int).map(lambda i: f"opengov_{i:06d}")
    df["filename"] = df["source_doc_id"] + ".md"
    df["text"] = df["text"].fillna("").astype(str)
    return df


def build_eurlex() -> pd.DataFrame:
    zip_path = RAW_ROOT / "hf" / "eurlex-greek-legislation" / "cleaned_dataset_md.zip"
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.endswith(".md"):
                continue
            name = Path(info.filename).name
            source_doc_id = Path(name).stem
            with zf.open(info) as handle:
                text = handle.read().decode("utf-8", errors="replace")
            rows.append(
                {
                    "source_dataset": "eurlex-greek-legislation",
                    "source_doc_id": source_doc_id,
                    "filename": name,
                    "text": text,
                }
            )
    return pd.DataFrame(rows)


DATASET_BUILDERS = {
    "1000_prwta_xronia_ellhnikhs": build_first_1k,
    "95k_deigma_ellinikis": build_95k,
    "Sxolika_vivlia": build_sxolika,
    "Apothetirio_Pergamos": build_pergamos,
    "Apothetirio_Kallipos": build_kallipos,
    "Ekklisiastika_Keimena": build_ekklisiastika,
    "Ellinika_Keimena_Project_Gutenberg": build_gutenberg,
    "Wikisource_Greek_texts": build_wikisource,
    "dimodis_logotexnia": build_dimodis,
    "ellinika_dedomena_europaikou_koinovouliou": build_europarl_greek,
    "eurlex-greek-legislation": build_eurlex,
    "klasikh_arx_ell_grammateia": build_klasikh,
    "opengov.gr-diaboyleuseis": build_opengov,
    "openbook_gr": build_openbook,
}


def materialize_exact_openarchives() -> tuple[Path, Path]:
    root = RAW_ROOT / "hf" / "openarchives.gr" / "data" / "openarchives"
    rows = []
    for jsonl_path in sorted(root.glob("shard_*/*.jsonl.zst")):
        dctx = zstd.ZstdDecompressor()
        with jsonl_path.open("rb") as fh, dctx.stream_reader(fh) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                record = json.loads(line)
                pipeline = record.get("pipeline_metadata") or {}
                source = record.get("source_metadata") or {}
                base_name = str(record.get("filename") or source.get("filename_base") or record.get("doc_id") or "")
                md_name = safe_stem(base_name) + ".md"
                rows.append(
                    {
                        "source_dataset": "openarchives.gr",
                        "source_doc_id": str(record.get("doc_id") or base_name),
                        "filename": str(source.get("filename") or md_name),
                        "md_filename": md_name,
                        "source_jsonl": str(jsonl_path),
                        "repository_collection": source.get("collection_slug"),
                        "document_type": source.get("type"),
                        "language_code": source.get("language_code"),
                        "greek_badness_score": pipeline.get("greek_badness_score"),
                        "mojibake_badness_score": pipeline.get("mojibake_badness_score"),
                        "latin_percentage": pipeline.get("latin_percentage"),
                        "polytonic_ratio": pipeline.get("polytonic_ratio"),
                        "char_count_no_comments": pipeline.get("char_count_no_comments"),
                        "is_empty": pipeline.get("is_empty"),
                        "filter": pipeline.get("filter"),
                        "needs_ocr": pipeline.get("needs_ocr"),
                        "ocr_success": pipeline.get("ocr_success"),
                    }
                )
    return write_exact_exports("openarchives.gr", pd.DataFrame(rows), quality_method="existing_pipeline_exact")


def materialize_exact_greek_phd() -> tuple[Path, Path]:
    root = RAW_ROOT / "mozilla" / "greek_phd" / "phd-theses-corpus" / "contents"
    rows = []
    for jsonl_path in sorted(root.glob("*.jsonl.zst")):
        dctx = zstd.ZstdDecompressor()
        with jsonl_path.open("rb") as fh, dctx.stream_reader(fh) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                record = json.loads(line)
                filename = str(record.get("filename") or record.get("doc_id") or "")
                md_name = safe_stem(filename) + ".md"
                rows.append(
                    {
                        "source_dataset": "greek_phd",
                        "source_doc_id": str(record.get("doc_id") or filename),
                        "filename": filename,
                        "md_filename": md_name,
                        "source_jsonl": str(jsonl_path),
                        "language": record.get("language"),
                        "filetype": record.get("filetype"),
                        "greek_badness_score": record.get("greek_badness_score"),
                        "mojibake_badness_score": record.get("mojibake_badness_score"),
                        "latin_percentage": record.get("latin_percentage"),
                        "polytonic_ratio": record.get("polytonic_ratio"),
                        "char_count_no_comments": record.get("char_count_no_comments"),
                        "is_empty": record.get("is_empty"),
                        "filter": record.get("filter"),
                        "needs_ocr": record.get("needs_ocr"),
                        "ocr_success": record.get("ocr_success"),
                    }
                )
    return write_exact_exports("greek_phd", pd.DataFrame(rows), quality_method="existing_pipeline_exact")


EXACT_DATASET_EXPORTERS = {
    "greek_phd": materialize_exact_greek_phd,
    "openarchives.gr": materialize_exact_openarchives,
}


def write_markdown_docs(df: pd.DataFrame, md_dir: Path) -> pd.DataFrame:
    ensure_dir(md_dir)
    rows = []
    for row in df.to_dict("records"):
        file_name = safe_stem(str(row["filename"]))
        if not file_name.endswith(".md"):
            file_name += ".md"
        path = md_dir / file_name
        path.write_text((row.get("text") or "").strip() + "\n", encoding="utf-8")
        row["md_path"] = str(path)
        row["md_filename"] = file_name
        rows.append(row)
    return pd.DataFrame(rows)


def score_directory(md_dir: Path) -> pd.DataFrame:
    results = glossapi_rs_noise.score_markdown_directory_detailed(str(md_dir), None)
    rows = []
    for row in results:
        path, score, latin_pct, table_ratio, poly_ratio, len_greek = row[:6]
        rows.append(
            {
                "md_path": str(path),
                "greek_badness_score": float(score),
                "latin_percentage": float(latin_pct),
                "table_ratio": float(table_ratio),
                "polytonic_ratio": float(poly_ratio),
                "len_greek": int(len_greek),
            }
        )
    return pd.DataFrame(rows)


def write_exact_exports(dataset_name: str, df: pd.DataFrame, quality_method: str) -> tuple[Path, Path]:
    dataset_root = OUT_ROOT / dataset_name
    ensure_dir(dataset_root)
    df = df.copy()
    df["quality_method"] = quality_method
    df["reevaluated_at"] = pd.Timestamp.now("UTC")
    docs_path = dataset_root / "document_level.parquet"
    scores_path = dataset_root / "document_quality.parquet"
    df.to_parquet(docs_path, index=False)
    score_cols = [
        "source_dataset",
        "source_doc_id",
        "md_filename",
        "greek_badness_score",
        "latin_percentage",
        "len_greek",
        "polytonic_ratio",
        "quality_method",
        "reevaluated_at",
        "mojibake_badness_score",
        "is_empty",
        "filter",
        "needs_ocr",
        "ocr_success",
    ]
    available_score_cols = [col for col in score_cols if col in df.columns]
    df[available_score_cols].to_parquet(scores_path, index=False)
    return docs_path, scores_path


def evaluate_dataset(dataset_name: str, keep_markdown: bool) -> tuple[Path, Path]:
    builder = DATASET_BUILDERS[dataset_name]
    df = builder().copy()
    dataset_root = OUT_ROOT / dataset_name
    md_dir = dataset_root / "markdown"
    ensure_dir(dataset_root)
    if md_dir.exists():
        shutil.rmtree(md_dir)
    docs = write_markdown_docs(df, md_dir)
    scores = score_directory(md_dir)
    merged = docs.merge(scores, on="md_path", how="left")
    merged["quality_method"] = "glossapi_rs_noise"
    merged["reevaluated_at"] = pd.Timestamp.now("UTC")
    docs_path = dataset_root / "document_level.parquet"
    scores_path = dataset_root / "document_quality.parquet"
    merged.to_parquet(docs_path, index=False)
    merged[
        [
            "source_dataset",
            "source_doc_id",
            "md_filename",
            "greek_badness_score",
            "latin_percentage",
            "table_ratio",
            "polytonic_ratio",
            "quality_method",
            "reevaluated_at",
        ]
    ].to_parquet(scores_path, index=False)
    if not keep_markdown:
        shutil.rmtree(md_dir)
    return docs_path, scores_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASET_BUILDERS) + list(EXACT_DATASET_EXPORTERS),
        choices=sorted(set(DATASET_BUILDERS) | set(EXACT_DATASET_EXPORTERS)),
    )
    parser.add_argument("--keep-markdown", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUT_ROOT)
    for dataset_name in args.datasets:
        if dataset_name in DATASET_BUILDERS:
            docs_path, scores_path = evaluate_dataset(dataset_name, keep_markdown=args.keep_markdown)
        else:
            docs_path, scores_path = EXACT_DATASET_EXPORTERS[dataset_name]()
        print(f"{dataset_name}\n  docs={docs_path}\n  scores={scores_path}")


if __name__ == "__main__":
    main()
