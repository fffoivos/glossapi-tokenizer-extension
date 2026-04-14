from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from glossapi_corpus_cli import pipeline


DEFAULT_LOCAL_DATA = Path("/home/foivos/data/glossapi_work/unified_corpus/data")
DEFAULT_EXTERNAL_DATA = Path("/home/foivos/data/glossapi_work/_external_builds/data")
DEFAULT_RELEASE_ROOT = Path("/home/foivos/data/glossapi_work/hf_release")
DEFAULT_MAX_PARQUET_BYTES = 4_500_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage the Hugging Face release directory for the unified GlossAPI corpus.")
    parser.add_argument("--local-data-root", type=Path, default=DEFAULT_LOCAL_DATA)
    parser.add_argument("--external-data-root", type=Path, default=DEFAULT_EXTERNAL_DATA)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--max-parquet-bytes", type=int, default=DEFAULT_MAX_PARQUET_BYTES)
    return parser.parse_args()


def ensure_clean_release_root(release_root: Path) -> Path:
    data_root = release_root / "data"
    if release_root.exists():
        shutil.rmtree(release_root)
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


def copy_small_parquet(src: Path, dest: Path) -> list[Path]:
    shutil.copy2(src, dest)
    return [dest]


def split_large_parquet(src: Path, data_root: Path, max_parquet_bytes: int) -> list[Path]:
    parquet = pq.ParquetFile(src)
    total_rows = parquet.metadata.num_rows
    approx_rows_per_part = max(1, int(total_rows * (max_parquet_bytes / src.stat().st_size)))
    batch_size = max(128, min(2048, approx_rows_per_part))
    part_index = 0
    current_rows = 0
    writer: pq.ParquetWriter | None = None
    created: list[Path] = []

    def open_writer(index: int) -> tuple[Path, pq.ParquetWriter]:
        part_path = data_root / f"{src.stem}.part-{index:05d}.parquet"
        created.append(part_path)
        return part_path, pq.ParquetWriter(part_path, parquet.schema_arrow, compression="zstd")

    _, writer = open_writer(part_index)
    for batch in parquet.iter_batches(batch_size=batch_size):
        if current_rows and current_rows + batch.num_rows > approx_rows_per_part:
            writer.close()
            part_index += 1
            current_rows = 0
            _, writer = open_writer(part_index)
        writer.write_table(pa.Table.from_batches([batch], schema=parquet.schema_arrow))
        current_rows += batch.num_rows
    writer.close()
    return created


def stage_parquet_files(local_data_root: Path, external_data_root: Path, release_root: Path, max_parquet_bytes: int) -> list[Path]:
    data_root = ensure_clean_release_root(release_root)
    created: list[Path] = []
    source_paths = sorted(local_data_root.glob("*.parquet")) + sorted(external_data_root.glob("*.parquet"))
    for src in source_paths:
        dest = data_root / src.name
        if src.stat().st_size > max_parquet_bytes:
            created.extend(split_large_parquet(src, data_root, max_parquet_bytes))
        else:
            created.extend(copy_small_parquet(src, dest))
    return created


def source_dataset_for_parquet(path: Path) -> str:
    frame = pd.read_parquet(path, columns=["source_dataset"])
    return str(frame["source_dataset"].iloc[0])


def build_row_counts(release_root: Path) -> pd.DataFrame:
    rows_by_dataset: dict[str, int] = defaultdict(int)
    file_count_by_dataset: dict[str, int] = defaultdict(int)
    for path in sorted((release_root / "data").glob("*.parquet")):
        dataset_name = source_dataset_for_parquet(path)
        rows_by_dataset[dataset_name] += pq.ParquetFile(path).metadata.num_rows
        file_count_by_dataset[dataset_name] += 1
    frame = pd.DataFrame(
        [
            {
                "source_dataset": dataset_name,
                "row_count": rows_by_dataset[dataset_name],
                "file_count": file_count_by_dataset[dataset_name],
            }
            for dataset_name in sorted(rows_by_dataset)
        ]
    )
    frame.to_csv(release_root / "row_counts.csv", index=False)
    return frame


def build_validation(release_root: Path) -> pd.DataFrame:
    rows = pipeline.validate_canonical_corpus(release_root)
    frame = pd.DataFrame(rows).sort_values(["source_dataset"])
    frame.to_csv(release_root / "validation_summary.csv", index=False)
    return frame


def markdown_table_from_csv(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_dataset_card(release_root: Path, row_counts: pd.DataFrame) -> None:
    total_rows = int(row_counts["row_count"].sum())
    row_table = markdown_table_from_csv(row_counts[["source_dataset", "row_count", "file_count"]])
    readme = f"""---
language:
- el
license: other
pretty_name: Glossapi Greek Nanochat Pretraining Dataset
size_categories:
- 100K<n<1M
task_categories:
- text-generation
---

# Glossapi Greek Nanochat Pretraining Dataset

This dataset is a unified Greek pretraining corpus assembled from the local GlossAPI collection plus four selected external sources:

- `HuggingFaceFW/finewiki` (`el`)
- `HuggingFaceFW/finepdfs-edu` (`ell_Grek`)
- `AI-team-UoA/greek_legal_code`
- `OPUS/OpenSubtitles-el-v2018` (monolingual Greek subtitles)

The upload is the canonical full corpus for local filtering and mix construction. It is intended to support Greek pretraining experiments, especially dataset selection and text-only export for a Greek nanochat-style workflow.

## Scope

- Total rows: `{total_rows}`
- Included source datasets: `{len(row_counts)}`
- Explicitly excluded from the canonical corpus: `95k_deigma_ellinikis`

## Canonical Columns

- `source_dataset`
- `source_doc_id`
- `text`
- `title`
- `author`
- `source_metadata_json`
- `is_historical_or_polytonic`
- `contains_math`
- `contains_latex`
- `greek_percentage`
- `latin_percentage`
- `polytonic_ratio`
- `table_ratio`
- `greek_badness_score`
- `len_greek`
- `mojibake_badness_score`
- `needs_ocr`
- `is_empty`
- `filter`
- `ocr_success`
- `quality_method`
- `reevaluated_at`

`title` and `author` are the only normalized source metadata fields promoted to top-level canonical columns. All other source metadata stays in `source_metadata_json` under original field names when available.

## Quality Notes

- The canonical upload is intentionally broad. CLI-side filtering is expected for training subsets.
- `openarchives.gr` stricter filtering such as `needs_ocr == false` plus `greek_badness_score < 25` is a downstream mix choice, not a canonical-upload exclusion.
- `greek_percentage` is derived where needed, typically from existing Latin-percentage signals.
- Rust badness scores are more reliable for modern Greek cleanliness than for older/polytonic/liturgical corpora.

## Metadata Notes

- Pipeline-generated filenames and synthetic processing identifiers are excluded from `source_metadata_json`.
- Some datasets have weak metadata or unresolved joins. Good text is still kept even when metadata is sparse.
- `contains_math` and `contains_latex` are content-analysis flags, not source metadata.

## Included Source Row Counts

{row_table}

## Intended Use

- Greek pretraining experiments
- Corpus filtering and mixture design
- Text-only export for nanochat-style training pipelines

## Limitations

- This dataset mixes modern, historical, polytonic, legal, academic, educational, and repository text.
- Source licenses are mixed and remain governed by the upstream datasets and repositories.
- Not every source has equally strong metadata quality or equally strong OCR/noise diagnostics.

## Rebuild

See `BUILD_REPLICATION.md` in this repo for the exact local staging workflow used to produce this upload.
"""
    (release_root / "README.md").write_text(readme, encoding="utf-8")


def write_replication_doc(release_root: Path) -> None:
    text = """# Build Replication

This release was staged locally from the existing GlossAPI workspace.

## Preconditions

- Raw corpora live under `/home/foivos/data/glossapi_raw`
- Reevaluation outputs live under `/home/foivos/data/glossapi_work/reeval`
- Python environment is available at `/home/foivos/venvs/glossapi-corpus-clean`

## 1. Build the canonical corpus shards

```bash
/home/foivos/venvs/glossapi-corpus-clean/bin/python -m glossapi_corpus_cli.cli build --output-root /home/foivos/data/glossapi_work/unified_corpus --no-include-external --workers 4
/home/foivos/venvs/glossapi-corpus-clean/bin/python - <<'PY'
from glossapi_corpus_cli.pipeline import build_dataset_to_parquet
for name in ['HuggingFaceFW/finewiki', 'HuggingFaceFW/finepdfs-edu', 'AI-team-UoA/greek_legal_code', 'OPUS/OpenSubtitles-el-v2018']:
    print(build_dataset_to_parquet(name, '/home/foivos/data/glossapi_work/_external_builds', False))
PY
```

## 2. Stage the Hugging Face release directory

```bash
/home/foivos/venvs/glossapi-corpus-clean/bin/python /home/foivos/Projects/glossapi-tokenizer-extension/assemble_hf_release.py
```

This step:

- copies the 15 local canonical source shards
- adds the 4 selected external source shards
- splits oversized Parquet files into smaller upload-safe parts
- writes `row_counts.csv`
- writes `validation_summary.csv`
- writes the dataset card `README.md`

## 3. Publish to Hugging Face

```bash
export HF_TOKEN=...
/home/foivos/venvs/glossapi-corpus-clean/bin/python /home/foivos/Projects/glossapi-tokenizer-extension/publish_hf_release.py \
  --release-root /home/foivos/data/glossapi_work/hf_release \
  --repo-id <username>/glossapi-greek-nanochat-pretraining-dataset \
  --private
```

If you are publishing from a remote high-uplink machine, rebuild there from the same source corpora instead of first transferring the finished local release directory.
"""
    (release_root / "BUILD_REPLICATION.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    stage_parquet_files(args.local_data_root, args.external_data_root, args.release_root, args.max_parquet_bytes)
    row_counts = build_row_counts(args.release_root)
    build_validation(args.release_root)
    write_dataset_card(args.release_root, row_counts)
    write_replication_doc(args.release_root)
    print(f"staged release at {args.release_root}")
    print(f"total rows: {int(row_counts['row_count'].sum())}")


if __name__ == "__main__":
    main()
