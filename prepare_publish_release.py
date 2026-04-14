from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import glossapi_rs_noise
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from assemble_hf_release import build_row_counts, build_validation
from glossapi_corpus_cli import pipeline


DEFAULT_INPUT_ROOT = Path("/home/foivos/data/glossapi_work/hf_release")
DEFAULT_OUTPUT_ROOT = Path("/home/foivos/data/glossapi_work/hf_release_publish")
WORK_ROOT = Path(__file__).resolve().parent
DEFAULT_BUILDER_SCRIPT_ROOT = WORK_ROOT / "nanochat_glossapi_en_vs_el"
DEFAULT_DEDUP_OVERLAY_ROOT = WORK_ROOT / "analysis" / "dedup" / "hf_uploads" / "base_repo_overlay_20260328"
BADNESS_THRESHOLD = 25.0
EMPTY_STRING = pa.scalar("", type=pa.large_string())
WORD_PATTERN = r"\S+"
WHITESPACE_PATTERN = r"\s+"

RESCORE_DATASETS = {
    "openarchives.gr",
    "greek_phd",
    "HuggingFaceFW/finewiki",
    "HuggingFaceFW/finepdfs-edu",
    "AI-team-UoA/greek_legal_code",
}

KEEP_KEYS_BY_DATASET = {
    "Ellinika_Keimena_Project_Gutenberg": ["Author Year", "Translator", "Translation Year", "Variety"],
    "Apothetirio_Pergamos": ["department", "submission_date", "year", "language", "abstract", "permanent_url", "supervisors"],
    "Apothetirio_Kallipos": [
        "document_type",
        "Υπότιτλος",
        "Θεματικές Κατηγορίες",
        "Λέξεις-κλειδιά",
        "Περίληψη",
        "Τύπος",
        "Γλώσσα",
        "DOI",
        "Βιβλιογραφική Αναφορά",
    ],
    "ellinika_dedomena_europaikou_koinovouliou": ["doc_type", "year", "link_url", "preferred_format", "preferred_url"],
    "eurlex-greek-legislation": ["document_url", "el_html_link", "category_id", "category_title", "category_acts_count"],
    "opengov.gr-diaboyleuseis": [
        "type",
        "url",
        "consultations.title",
        "ministries.name",
        "consultations.start_date",
        "consultations.end_date",
        "consultations.total_comments",
        "consultations.accepted_comments",
    ],
    "openbook_gr": ["url", "created_at", "modified_at", "issues"],
    "openarchives.gr": [
        "type",
        "collection_slug",
        "language_code",
        "Θέμα",
        "Περιγραφή",
        "Επιστημονικό πεδίο",
        "Σχολή/Τμήμα/Ινστιτούτο",
        "Τύπος",
        "Χρονολογία",
        "Ημερομηνία έκδοσης",
        "Συντελεστής",
        "Πάροχος",
        "Δικαιώματα",
        "Αποθετήριο / συλλογή",
        "Επιμέρους συλλογή",
    ],
    "greek_phd": [
        "year",
        "department",
        "university",
        "abstract_el",
        "scientific_field",
        "scientific_field_level1",
        "scientific_field_level2",
        "scientific_field_level3",
        "keywords",
        "doi",
        "language",
        "title_en",
        "date_accepted",
        "handle_url",
        "license",
    ],
    "HuggingFaceFW/finewiki": ["url", "date_modified", "wikidata_id", "page_id", "wikiname", "in_language", "version"],
    "HuggingFaceFW/finepdfs-edu": [
        "url",
        "date",
        "dump",
        "language",
        "full_doc_lid",
        "full_doc_lid_score",
        "page_average_lid",
        "page_average_lid_score",
        "token_count",
        "fw_edu_scores",
    ],
}


def default_workers() -> int:
    return max(1, min(6, os.cpu_count() or 1))


def default_clean_threads(workers: int) -> int:
    cpu_total = os.cpu_count() or 1
    return max(1, cpu_total // max(1, workers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the corrected publish snapshot from the staged local HF release.")
    workers = default_workers()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--badness-threshold", type=float, default=BADNESS_THRESHOLD)
    parser.add_argument("--workers", type=int, default=workers)
    parser.add_argument("--clean-threads", type=int, default=default_clean_threads(workers))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--builder-script-root", type=Path, default=DEFAULT_BUILDER_SCRIPT_ROOT)
    parser.add_argument("--dedup-overlay-root", type=Path, default=DEFAULT_DEDUP_OVERLAY_ROOT)
    return parser.parse_args()


def detect_dataset_name(path: Path) -> str:
    parquet = pq.ParquetFile(path)
    batch = parquet.iter_batches(batch_size=1, columns=["source_dataset"])
    row = next(batch).to_pylist()[0]
    return str(row["source_dataset"])


def select_payload_for_dataset(dataset_name: str, raw_payload: str | None) -> str | None:
    if raw_payload is None:
        return None
    parsed = pipeline.maybe_json_loads(raw_payload)
    keep_keys = KEEP_KEYS_BY_DATASET.get(dataset_name)
    if keep_keys is None:
        return pipeline.metadata_json(parsed)
    return pipeline.metadata_json(pipeline.select_metadata_fields(parsed, keep_keys))


def row_needs_latest_clean(dataset_name: str, row: dict[str, Any]) -> bool:
    if dataset_name in RESCORE_DATASETS:
        return True
    if row.get("quality_method") != "glossapi_rs_noise":
        return True
    return row.get("greek_badness_score") is None


def rescore_rows_latest_clean(rows: list[dict[str, Any]], clean_threads: int) -> list[dict[str, Any]]:
    if not rows:
        return rows
    with tempfile.TemporaryDirectory(prefix="publish_clean_") as tmpdir:
        tmp_root = Path(tmpdir)
        filename_map: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(rows):
            name = f"doc_{index:06d}.md"
            (tmp_root / name).write_text(pipeline.clean_text(row.get("text")) + "\n", encoding="utf-8")
            filename_map[name] = row
        scored = glossapi_rs_noise.score_markdown_directory_detailed(str(tmp_root), clean_threads)
        reevaluated_at = pd.Timestamp.now("UTC")
        for score_row in scored:
            metrics = pipeline.decode_noise_score_row(score_row)
            row = filename_map[Path(str(metrics["md_path"])).name]
            row["greek_badness_score"] = float(metrics["greek_badness_score"])
            row["latin_percentage"] = float(metrics["latin_percentage"])
            row["table_ratio"] = float(metrics["table_ratio"])
            row["polytonic_ratio"] = float(metrics["polytonic_ratio"])
            row["len_greek"] = int(metrics["len_greek"])
            row["quality_method"] = "glossapi_rs_noise"
            row["reevaluated_at"] = reevaluated_at
    return rows


def keep_publish_row(row: dict[str, Any], badness_threshold: float) -> bool:
    if not pipeline.clean_text(row.get("text")):
        return False
    badness = row.get("greek_badness_score")
    if badness is None:
        return False
    try:
        return float(badness) < float(badness_threshold)
    except Exception:
        return False


def transform_rows(
    dataset_name: str,
    rows: list[dict[str, Any]],
    badness_threshold: float,
    clean_threads: int,
) -> list[dict[str, Any]]:
    if any(row_needs_latest_clean(dataset_name, row) for row in rows):
        rows = rescore_rows_latest_clean(rows, clean_threads)
    transformed: list[dict[str, Any]] = []
    for row in rows:
        row["source_metadata_json"] = select_payload_for_dataset(dataset_name, row.get("source_metadata_json"))
        row["is_empty"] = False if pipeline.clean_text(row.get("text")) else True
        if not keep_publish_row(row, badness_threshold):
            continue
        transformed.append(row)
    return transformed


def write_transformed_parquet(
    src: Path,
    dest: Path,
    badness_threshold: float,
    batch_size: int,
    clean_threads: int,
) -> tuple[str, int]:
    dataset_name = detect_dataset_name(src)
    parquet = pq.ParquetFile(src)
    writer: pq.ParquetWriter | None = None
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    if tmp_dest.exists():
        tmp_dest.unlink()
    row_count = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            rows = transform_rows(dataset_name, batch.to_pylist(), badness_threshold, clean_threads)
            if not rows:
                continue
            frame = pipeline.finalize_frame(pd.DataFrame(rows))
            table = pa.Table.from_pandas(frame, schema=pipeline.CANONICAL_ARROW_SCHEMA, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_dest, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += len(frame)
    except Exception:
        if writer is not None:
            writer.close()
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise
    if writer is not None:
        writer.close()
        tmp_dest.replace(dest)
    return dataset_name, row_count


def _normalized_text_array(text_values: pa.Array) -> pa.Array:
    casted = pc.cast(text_values, pa.large_string())
    return pc.if_else(pc.is_null(casted), EMPTY_STRING, casted)


def summarize_text_metrics(text_values: pa.Array) -> dict[str, int]:
    normalized = _normalized_text_array(text_values)
    words = pc.fill_null(pc.count_substring_regex(normalized, pattern=WORD_PATTERN), 0)
    chars = pc.fill_null(pc.utf8_length(normalized), 0)
    non_whitespace = pc.replace_substring_regex(normalized, pattern=WHITESPACE_PATTERN, replacement="")
    non_whitespace_chars = pc.fill_null(pc.utf8_length(non_whitespace), 0)
    utf8_bytes = pc.fill_null(pc.binary_length(pc.cast(normalized, pa.large_binary())), 0)
    return {
        "row_count": int(len(normalized)),
        "chars": int(pc.sum(chars).as_py() or 0),
        "non_whitespace_chars": int(pc.sum(non_whitespace_chars).as_py() or 0),
        "utf8_bytes": int(pc.sum(utf8_bytes).as_py() or 0),
        "approx_word_count": int(pc.sum(words).as_py() or 0),
    }


def build_prepare_manifest(output_root: Path) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    totals = {
        "row_count": 0,
        "chars": 0,
        "non_whitespace_chars": 0,
        "utf8_bytes": 0,
        "approx_word_count": 0,
    }
    for path in sorted((output_root / "data").glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        metrics = {
            "row_count": 0,
            "chars": 0,
            "non_whitespace_chars": 0,
            "utf8_bytes": 0,
            "approx_word_count": 0,
        }
        dataset_name: str | None = None
        for batch in parquet.iter_batches(batch_size=1024, columns=["source_dataset", "text"]):
            if dataset_name is None:
                dataset_name = str(batch.column(0).to_pylist()[0])
            batch_metrics = summarize_text_metrics(batch.column(1))
            for key in metrics:
                metrics[key] += int(batch_metrics[key])
        if dataset_name is None:
            continue
        entry = {
            "source_dataset": dataset_name,
            "path": str(Path("data") / path.name),
            **metrics,
        }
        sources.append(entry)
        for key in totals:
            totals[key] += int(metrics[key])
    manifest = {
        "source_count": int(len(sources)),
        "totals": totals,
        "sources": sources,
    }
    (output_root / "prepare_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def stage_extra_builder_artifacts(
    output_root: Path,
    *,
    builder_script_root: Path,
    dedup_overlay_root: Path | None,
) -> None:
    scripts_dir = output_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for src, dest_name in (
        (builder_script_root / "prepare_glossapi_greek_experiment_data.py", "prepare_glossapi_greek_experiment_data.py"),
        (builder_script_root / "summarize_glossapi_greek_experiment_data.py", "summarize_glossapi_greek_experiment_data.py"),
        (WORK_ROOT / "glossapi_corpus_cli" / "text_dedup.py", "text_dedup.py"),
    ):
        if src.exists():
            shutil.copy2(src, scripts_dir / dest_name)
    if dedup_overlay_root is not None:
        dedup_source = dedup_overlay_root / "dedup_metadata"
        dedup_dest = output_root / "dedup_metadata"
        if dedup_source.exists():
            if dedup_dest.exists():
                shutil.rmtree(dedup_dest)
            shutil.copytree(dedup_source, dedup_dest)


def ensure_clean_output_root(output_root: Path) -> Path:
    if output_root.exists():
        shutil.rmtree(output_root)
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


def write_readme(output_root: Path, row_counts: pd.DataFrame) -> None:
    total_rows = int(row_counts["row_count"].sum())
    dedup_latest_path = output_root / "dedup_metadata" / "latest.json"
    scripts_dir = output_root / "scripts"
    lines = [
        "---",
        "language:",
        "- el",
        "license: other",
        "pretty_name: Glossapi Greek Nanochat Pretraining Dataset",
        "size_categories:",
        "- 100K<n<1M",
        "task_categories:",
        "- text-generation",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: data/*.parquet",
        "---",
        "",
        "# Glossapi Greek Nanochat Pretraining Dataset",
        "",
        "This dataset is a unified Greek pretraining corpus assembled from the local GlossAPI collection plus three selected external Hugging Face sources.",
        "The current prepared snapshot is intended to be used directly as a source-ready corpus for lightweight mix builders.",
        "",
        "## Scope",
        "",
        f"- Total rows: `{total_rows}`",
        f"- Included source datasets: `{len(row_counts)}`",
        "- Explicitly excluded from the canonical corpus: `95k_deigma_ellinikis`",
        f"- Published rows are filtered to non-empty text and `greek_badness_score < {BADNESS_THRESHOLD:g}` after applying the latest clean outputs where needed.",
        "",
    ]
    if dedup_latest_path.exists():
        latest = json.loads(dedup_latest_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "## Dedup Metadata",
                "",
                "This repo now also carries builder-facing dedup metadata under:",
                "",
                "- `dedup_metadata/latest.json`",
                f"- `{latest['path']}/`",
                "",
                "The base corpus rows in `data/*.parquet` are unchanged. Dedup metadata are published as extra artifacts for downstream builders and should not be interpreted as an in-place mutation of the base dataset.",
                "",
                "The intended builder flow is:",
                "",
                "1. load base rows from `data/*.parquet`",
                "2. join dedup metadata by `doc_key` or `(source_dataset, source_doc_id)`",
                "3. apply builder-time dedup policy such as `annotate`, `drop_intra`, or `drop_intra_and_inter`",
                "",
                "The `configs` block in this card keeps the default dataset loading path restricted to `data/*.parquet`, so the extra dedup parquet files do not become part of the default row view.",
                "",
            ]
        )
    lines.extend(
        [
        "## Canonical Columns",
        "",
        ]
    )
    lines.extend(f"- `{column}`" for column in pipeline.CANONICAL_COLUMNS)
    lines.extend(
        [
            "",
            "## Metadata Notes",
            "",
            "- `title` and `author` are the only normalized source metadata columns.",
            "- `source_metadata_json` is narrowed to the selected source fields from `CONTENT_METADATA_INVENTORY.md`.",
            "- Placeholder URL values such as `na` and `n/a` are normalized to null.",
            "",
            "## Prepared-Source Notes",
            "",
            "- `row_counts.csv` records staged row and file counts per source dataset.",
            "- `prepare_manifest.json` records staged file-level text metrics for downstream planning.",
            "",
            "## Quality Notes",
            "",
            "- The published snapshot is stricter than the broad local canonical corpus.",
            "- Filtering is currently based on non-empty text plus low badness.",
            "- `greek_percentage` is kept as information, not as a filtering condition.",
            "- Quality metrics remain top-level columns and are not embedded in `source_metadata_json`.",
            "",
            "## Included Source Row Counts",
            "",
            "| source_dataset | row_count | file_count |",
            "| --- | --- | --- |",
        ]
    )
    for row in row_counts.itertuples(index=False):
        lines.append(f"| {row.source_dataset} | {row.row_count} | {row.file_count} |")
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "See `BUILD_REPLICATION.md` in this repo for the exact staging and refresh workflow used for this prepared snapshot.",
        ]
    )
    if (scripts_dir / "prepare_glossapi_greek_experiment_data.py").exists():
        lines.extend(
            [
                "",
                "## Experiment Reconstruction",
                "",
                "The repo also includes a deterministic build script for the nanochat experiment subset:",
                "",
                "- `scripts/prepare_glossapi_greek_experiment_data.py`",
                "",
                "The companion summary helper is:",
                "",
                "- `scripts/summarize_glossapi_greek_experiment_data.py`",
            ]
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_replication_doc(output_root: Path) -> None:
    text = """# Build Replication

This publish snapshot was prepared locally from the staged release workspace.

## Preconditions

- Raw corpora live under `/home/foivos/data/glossapi_raw`
- Reevaluation outputs live under `/home/foivos/data/glossapi_work/reeval`
- The staged broad release exists at `/home/foivos/data/glossapi_work/hf_release`
- Python environment is available at `/home/foivos/data/glossapi_work/.venv`

## 1. Prepare the publish snapshot

```bash
source /home/foivos/data/glossapi_work/.venv/bin/activate
python /home/foivos/data/glossapi_work/prepare_publish_release.py --input-root /home/foivos/data/glossapi_work/hf_release --output-root /home/foivos/data/glossapi_work/hf_release_publish
```

This step:

- narrows `source_metadata_json` to the selected source keep-set
- normalizes placeholder URLs like `na` to null
- reruns the latest clean outputs for datasets that are missing current clean metadata
- filters all published rows to non-empty text and low badness
- stages the builder scripts under `scripts/`
- carries the published dedup overlay under `dedup_metadata/` when available
- rewrites `row_counts.csv`, `validation_summary.csv`, `prepare_manifest.json`, and the dataset card

## 2. Publish to Hugging Face

```bash
export HF_TOKEN=...
source /home/foivos/data/glossapi_work/.venv/bin/activate
python /home/foivos/data/glossapi_work/publish_hf_release.py --release-root /home/foivos/data/glossapi_work/hf_release_publish --repo-id <username>/glossapi-greek-nanochat-pretraining-dataset --private
```
"""
    (output_root / "BUILD_REPLICATION.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_data_root = args.input_root / "data"
    output_data_root = ensure_clean_output_root(args.output_root)
    sources = sorted(input_data_root.glob("*.parquet"))
    with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                write_transformed_parquet,
                src,
                output_data_root / src.name,
                args.badness_threshold,
                args.batch_size,
                args.clean_threads,
            ): src.name
            for src in sources
        }
        for future in cf.as_completed(futures):
            dataset_name, row_count = future.result()
            print(f"{dataset_name}: {row_count}", flush=True)
    row_counts = build_row_counts(args.output_root)
    validation = build_validation(args.output_root)
    validation.to_csv(args.output_root / "validation_summary.csv", index=False)
    stage_extra_builder_artifacts(
        args.output_root,
        builder_script_root=args.builder_script_root.resolve(),
        dedup_overlay_root=args.dedup_overlay_root.resolve() if args.dedup_overlay_root else None,
    )
    build_prepare_manifest(args.output_root)
    write_readme(args.output_root, row_counts)
    write_replication_doc(args.output_root)
    print(f"prepared publish release at {args.output_root}")
    print(f"total rows: {int(row_counts['row_count'].sum())}")


if __name__ == "__main__":
    main()
