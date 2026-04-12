#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

TOKENIZER_REPO_ROOT = Path("/home/foivos/Projects/glossapi-tokenizer-extension")
HF_RELEASE_ROOT = Path("/home/foivos/data/glossapi_work/hf_release_publish")
GLOSSAPI_WORK_ROOT = Path("/home/foivos/data/glossapi_work")
DEFAULT_HPLT_BASE_URL = "https://data.hplt-project.org/three/sorted/ell_Grek/"
DEFAULT_DATASET_NAME = "HPLT/ell_Grek_ge8_no_mt"
DEFAULT_REPO_ID = "fffoivos/glossapi-greek-nanochat-pretraining-dataset"
DEFAULT_PART_ROWS = 200_000
DEFAULT_BATCH_SIZE = 512
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))
SCHEMA_SOURCE = HF_RELEASE_ROOT / "data" / "openarchives.gr.part-00000.parquet"

sys.path.insert(0, str(TOKENIZER_REPO_ROOT / "subprojects" / "01_hplt_filtering" / "scripts"))
sys.path.insert(0, str(GLOSSAPI_WORK_ROOT))
sys.path.insert(0, str(HF_RELEASE_ROOT))

import hplt_web_register as hplt_register  # noqa: E402
from glossapi_corpus_cli import pipeline  # noqa: E402
from assemble_hf_release import build_row_counts, build_validation  # noqa: E402
from prepare_publish_release import build_prepare_manifest, write_readme  # noqa: E402


@dataclass
class ShardResult:
    shard: str
    quality_bin: int
    rows_seen: int
    rows_kept: int
    rows_skipped_mt: int
    rows_skipped_badness: int
    rows_skipped_empty: int
    rows_written: int
    part_files: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a filtered HPLT slice in the canonical HF source-parquet schema and optionally upload it.")
    parser.add_argument("--release-root", type=Path, default=HF_RELEASE_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--hplt-base-url", default=DEFAULT_HPLT_BASE_URL)
    parser.add_argument("--quality-min", type=int, default=8)
    parser.add_argument("--exclude-main-register", action="append", default=["MT"])
    parser.add_argument("--require-filter", default=None, help="Optional exact HPLT filter value to keep, e.g. keep")
    parser.add_argument("--only-shards", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--rows-per-part", type=int, default=DEFAULT_PART_ROWS)
    parser.add_argument("--quality-mode", choices=["score_only", "corpus_clean"], default="corpus_clean")
    parser.add_argument("--greek-badness-max", type=float, default=60.0)
    parser.add_argument("--clean-num-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--log-every-rows", type=int, default=100000)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--commit-message", default="Add filtered HPLT Greek slice (>=8, no MT)")
    parser.add_argument("--commit-description", default="Adds the filtered HPLT Greek source parquet slice and refreshes release metadata files.")
    return parser.parse_args()


def list_shards(base_url: str, quality_min: int) -> list[str]:
    html = urlopen(base_url).read().decode("utf-8", errors="replace")
    names = sorted(set(re.findall(r'href="([0-9]+_[0-9]+\.jsonl\.zst)"', html)))
    if not names:
        raise RuntimeError(f"No shard links found at {base_url}")
    selected = [name for name in names if shard_quality_bin(name) >= quality_min]
    if not selected:
        raise RuntimeError(f"No shards matched quality_min={quality_min} at {base_url}")
    return selected


def shard_quality_bin(name: str) -> int:
    return int(name.split("_", 1)[0])


def stream_hplt_rows(url: str) -> Iterable[dict[str, Any]]:
    quoted = shlex.quote(url)
    cmd = f"curl -L --silent {quoted} | zstd -dc"
    proc = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if line.strip():
                yield json.loads(line)
    finally:
        proc.kill()
        proc.wait()


def normalize_html_lang(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(item) for item in value if item not in (None, "")]
        return cleaned or None
    return str(value)


def register_labels(web_register: dict[str, float] | None) -> tuple[str | None, str | None, str | None, str | None, float | None, float | None]:
    main_code, main_score = hplt_register.top_main_label(web_register)
    sub_code, sub_score = hplt_register.top_sub_label(web_register)
    parent_code, canonical_sub_code = hplt_register.category_tuple_codes_from_sub_label(sub_code)
    if main_code is not None and parent_code == main_code and canonical_sub_code is not None:
        return (
            main_code,
            hplt_register.label_name(main_code),
            canonical_sub_code,
            hplt_register.label_name(canonical_sub_code),
            float(main_score),
            float(sub_score),
        )
    if main_code is not None:
        return main_code, hplt_register.label_name(main_code), None, None, float(main_score), None
    return None, None, None, None, None, None


def build_source_metadata(row: dict[str, Any], *, quality_bin: int) -> str | None:
    url = row.get("u")
    host = urlparse(url).netloc if url else None
    raw_web_register = row.get("web-register") or {}
    main_code, main_label, sub_code, sub_label, main_score, sub_score = register_labels(raw_web_register)
    payload = {
        "url": url,
        "host": host,
        "content_type": row.get("c"),
        "timestamp": row.get("ts"),
        "crawl_id": row.get("crawl_id"),
        "quality_bin": quality_bin,
        "filter": row.get("filter"),
        "html_lang": normalize_html_lang(row.get("html_lang")),
        "lang": row.get("lang"),
        "prob": row.get("prob"),
        "cluster_size": row.get("cluster_size"),
        "register_level_1_code": main_code,
        "register_level_1": main_label,
        "register_level_1_score": main_score,
        "register_level_2_code": sub_code,
        "register_level_2": sub_label,
        "register_level_2_score": sub_score,
        "web_register": raw_web_register or None,
        "doc_scores": row.get("doc_scores"),
        "seg_langs": row.get("seg_langs"),
    }
    return pipeline.metadata_json(payload)


def build_base_row(raw_row: dict[str, Any], *, dataset_name: str, shard: str, quality_bin: int) -> dict[str, Any] | None:
    text = pipeline.clean_text(raw_row.get("text"))
    if not text:
        return None
    row_id = pipeline.clean_text(raw_row.get("id"))
    main_code, _, _, _, _, _ = register_labels(raw_row.get("web-register") or {})
    return {
        "source_dataset": dataset_name,
        "source_doc_id": f"hplt::{shard}::{row_id or 'row'}",
        "text": text,
        "title": None,
        "author": None,
        "source_metadata_json": build_source_metadata(raw_row, quality_bin=quality_bin),
        "is_historical_or_polytonic": False,
        "contains_math": pipeline.contains_math(text),
        "contains_latex": pipeline.contains_latex(text),
        "greek_percentage": None,
        "latin_percentage": None,
        "polytonic_ratio": None,
        "table_ratio": None,
        "greek_badness_score": None,
        "mojibake_badness_score": None,
        "needs_ocr": False,
        "is_empty": False,
        "filter": None,
        "ocr_success": None,
        "quality_method": None,
        "reevaluated_at": None,
        "_top_main_code": main_code,
    }


def load_target_schema(target_schema_path: Path) -> pa.Schema:
    if target_schema_path.exists():
        return pq.read_schema(target_schema_path)
    return pipeline.CANONICAL_ARROW_SCHEMA


def score_rows_with_corpus_clean(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    clean_num_threads: int,
    greek_badness_max: float,
) -> tuple[list[dict[str, Any]], int]:
    from glossapi import Corpus  # lazy import; only needed on worker builds

    with tempfile.TemporaryDirectory(prefix=f"corpus_clean_{dataset_name.replace('/', '_')}_") as tmpdir:
        tmp_root = Path(tmpdir)
        corpus = Corpus(input_dir=tmp_root / "input", output_dir=tmp_root / "output")
        filename_map: dict[str, dict[str, Any]] = {}
        for idx, row in enumerate(rows):
            stem = f"doc_{idx:06d}"
            md_path = corpus.markdown_dir / f"{stem}.md"
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(row["text"].strip() + "\n", encoding="utf-8")
            filename_map[f"{stem}.pdf"] = row
        corpus.clean(
            input_dir=corpus.markdown_dir,
            drop_bad=False,
            write_cleaned_files=False,
            num_threads=clean_num_threads,
        )
        parquet_path = corpus.output_dir / "download_results" / "download_results.parquet"
        frame = pd.read_parquet(parquet_path)
        kept: list[dict[str, Any]] = []
        dropped_badness = 0
        for record in frame.to_dict("records"):
            filename = str(record.get("filename") or "")
            row = filename_map.get(filename)
            if row is None:
                continue
            for field in (
                "greek_percentage",
                "latin_percentage",
                "polytonic_ratio",
                "table_ratio",
                "greek_badness_score",
                "len_greek",
                "mojibake_badness_score",
                "needs_ocr",
                "is_empty",
                "filter",
                "ocr_success",
                "quality_method",
                "reevaluated_at",
            ):
                if field in record:
                    row[field] = record[field]
            if row.get("quality_method") in (None, ""):
                row["quality_method"] = "corpus.clean"
            score = row.get("greek_badness_score")
            if score not in (None, "") and float(score) > greek_badness_max:
                dropped_badness += 1
                continue
            kept.append(row)
    return kept, dropped_badness


def score_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    quality_mode: str,
    clean_num_threads: int,
    greek_badness_max: float,
) -> tuple[list[dict[str, Any]], int]:
    if not rows:
        return [], 0
    if quality_mode == "score_only":
        scored = pipeline.batch_score_missing_quality(rows, dataset_name)
        dropped_badness = 0
        if greek_badness_max is not None:
            kept = []
            for item in scored:
                score = item.get("greek_badness_score")
                if score not in (None, "") and float(score) > greek_badness_max:
                    dropped_badness += 1
                    continue
                kept.append(item)
            scored = kept
        return scored, dropped_badness
    return score_rows_with_corpus_clean(
        rows,
        dataset_name=dataset_name,
        clean_num_threads=clean_num_threads,
        greek_badness_max=greek_badness_max,
    )


class PartWriter:
    def __init__(self, *, data_root: Path, dataset_name: str, shard: str, rows_per_part: int, target_schema: pa.Schema, target_columns: list[str]):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.shard = shard.replace('.jsonl.zst', '')
        self.rows_per_part = rows_per_part
        self.target_schema = target_schema
        self.target_columns = target_columns
        self.writer: pq.ParquetWriter | None = None
        self.current_rows = 0
        self.part_index = 0
        self.created: list[Path] = []

    def _next_path(self) -> Path:
        stem = self.dataset_name.replace('/', '__')
        return self.data_root / f"{stem}.{self.shard}.part-{self.part_index:05d}.parquet"

    def _open_writer(self) -> None:
        path = self._next_path()
        self.created.append(path)
        self.writer = pq.ParquetWriter(path, self.target_schema, compression='zstd')
        self.current_rows = 0
        self.part_index += 1

    def write_rows(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        frame = pipeline.finalize_frame(pd.DataFrame(rows))
        frame = frame[self.target_columns]
        table = pa.Table.from_pandas(frame, schema=self.target_schema, preserve_index=False)
        if self.writer is None or self.current_rows + len(frame) > self.rows_per_part:
            self.close()
            self._open_writer()
        assert self.writer is not None
        self.writer.write_table(table)
        self.current_rows += len(frame)
        return len(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.current_rows = 0


def process_shard(
    shard: str,
    *,
    base_url: str,
    dataset_name: str,
    quality_min: int,
    exclude_main_registers: set[str],
    require_filter: str | None,
    batch_size: int,
    rows_per_part: int,
    data_root: Path,
    target_schema_path: Path,
    quality_mode: str,
    greek_badness_max: float,
    clean_num_threads: int,
    max_docs: int | None,
    max_chars: int | None,
    log_every_rows: int,
) -> ShardResult:
    quality_bin = shard_quality_bin(shard)
    if quality_bin < quality_min:
        return ShardResult(shard, quality_bin, 0, 0, 0, 0, 0, 0, [])
    target_schema = load_target_schema(target_schema_path)
    target_columns = list(target_schema.names)
    writer = PartWriter(
        data_root=data_root,
        dataset_name=dataset_name,
        shard=shard,
        rows_per_part=rows_per_part,
        target_schema=target_schema,
        target_columns=target_columns,
    )
    url = base_url.rstrip('/') + '/' + shard
    batch: list[dict[str, Any]] = []
    rows_seen = rows_kept = rows_skipped_mt = rows_skipped_badness = rows_skipped_empty = rows_written = 0
    chars_kept = 0
    for raw_row in stream_hplt_rows(url):
        rows_seen += 1
        if require_filter is not None and raw_row.get('filter') != require_filter:
            continue
        row = build_base_row(raw_row, dataset_name=dataset_name, shard=shard, quality_bin=quality_bin)
        if row is None:
            rows_skipped_empty += 1
            continue
        if row.pop('_top_main_code', None) in exclude_main_registers:
            rows_skipped_mt += 1
            continue
        batch.append(row)
        rows_kept += 1
        chars_kept += len(row['text'])
        if log_every_rows > 0 and rows_seen % log_every_rows == 0:
            print(json.dumps({
                'event': 'shard_progress',
                'shard': shard,
                'quality_bin': quality_bin,
                'rows_seen': rows_seen,
                'rows_kept': rows_kept,
                'rows_skipped_mt': rows_skipped_mt,
                'rows_skipped_badness': rows_skipped_badness,
                'rows_skipped_empty': rows_skipped_empty,
                'rows_written_so_far': rows_written,
            }, ensure_ascii=False), flush=True)
        if len(batch) >= batch_size:
            scored, dropped_badness = score_rows(
                batch,
                dataset_name=dataset_name,
                quality_mode=quality_mode,
                clean_num_threads=clean_num_threads,
                greek_badness_max=greek_badness_max,
            )
            rows_skipped_badness += dropped_badness
            for item in scored:
                item['is_historical_or_polytonic'] = pipeline.derive_historical_flag(dataset_name, polytonic_ratio=item.get('polytonic_ratio'))
                item['is_empty'] = False if pipeline.clean_text(item.get('text')) else True
                item['needs_ocr'] = False
            rows_written += writer.write_rows(scored)
            batch = []
        if max_docs is not None and rows_kept >= max_docs:
            break
        if max_chars is not None and chars_kept >= max_chars:
            break
    if batch:
        scored, dropped_badness = score_rows(
            batch,
            dataset_name=dataset_name,
            quality_mode=quality_mode,
            clean_num_threads=clean_num_threads,
            greek_badness_max=greek_badness_max,
        )
        rows_skipped_badness += dropped_badness
        for item in scored:
            item['is_historical_or_polytonic'] = pipeline.derive_historical_flag(dataset_name, polytonic_ratio=item.get('polytonic_ratio'))
            item['is_empty'] = False if pipeline.clean_text(item.get('text')) else True
            item['needs_ocr'] = False
        rows_written += writer.write_rows(scored)
    writer.close()
    return ShardResult(
        shard=shard,
        quality_bin=quality_bin,
        rows_seen=rows_seen,
        rows_kept=rows_kept,
        rows_skipped_mt=rows_skipped_mt,
        rows_skipped_badness=rows_skipped_badness,
        rows_skipped_empty=rows_skipped_empty,
        rows_written=rows_written,
        part_files=[str(path) for path in writer.created],
    )


def remove_existing_dataset_parts(data_root: Path, dataset_name: str) -> list[Path]:
    stem = dataset_name.replace('/', '__')
    removed = []
    for path in sorted(data_root.glob(f'{stem}*.parquet')):
        removed.append(path)
        path.unlink()
    return removed


def refresh_release_metadata(release_root: Path) -> dict[str, Any]:
    row_counts = build_row_counts(release_root)
    validation = build_validation(release_root)
    validation.to_csv(release_root / 'validation_summary.csv', index=False)
    prepare_manifest = build_prepare_manifest(release_root)
    write_readme(release_root, row_counts)
    return {
        'row_counts_csv': str(release_root / 'row_counts.csv'),
        'validation_summary_csv': str(release_root / 'validation_summary.csv'),
        'prepare_manifest_json': str(release_root / 'prepare_manifest.json'),
        'readme': str(release_root / 'README.md'),
        'row_count_total': int(row_counts['row_count'].sum()),
        'source_dataset_count': int(len(row_counts)),
        'prepare_manifest_source_count': int(prepare_manifest.get('source_count', 0)),
    }


def stage_patch_root(release_root: Path, part_files: list[Path]) -> Path:
    patch_root = Path(tempfile.mkdtemp(prefix='hplt_hf_patch_'))
    (patch_root / 'data').mkdir(parents=True, exist_ok=True)
    for path in part_files:
        shutil.copy2(path, patch_root / 'data' / path.name)
    for name in ['row_counts.csv', 'validation_summary.csv', 'prepare_manifest.json', 'README.md']:
        shutil.copy2(release_root / name, patch_root / name)
    return patch_root


def upload_patch(patch_root: Path, repo_id: str, commit_message: str, commit_description: str) -> None:
    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        repo_type='dataset',
        folder_path=patch_root,
        commit_message=commit_message,
        commit_description=commit_description,
    )


def main() -> None:
    args = parse_args()
    release_root = args.release_root.resolve()
    data_root = release_root / 'data'
    data_root.mkdir(parents=True, exist_ok=True)
    summary_json = args.summary_json or (release_root / 'hplt_build_summary_ge8_no_mt.json')

    shards = args.only_shards or list_shards(args.hplt_base_url, args.quality_min)
    shards = [name for name in shards if shard_quality_bin(name) >= args.quality_min]
    removed = remove_existing_dataset_parts(data_root, args.dataset_name)

    worker_count = max(1, min(args.workers, len(shards)))
    results: list[ShardResult] = []
    if worker_count == 1:
        for shard in shards:
            results.append(
                process_shard(
                    shard,
                    base_url=args.hplt_base_url,
                    dataset_name=args.dataset_name,
                    quality_min=args.quality_min,
                    exclude_main_registers=set(args.exclude_main_register),
                    require_filter=args.require_filter,
                    batch_size=args.batch_size,
                    rows_per_part=args.rows_per_part,
                    data_root=data_root,
                    target_schema_path=SCHEMA_SOURCE,
                    quality_mode=args.quality_mode,
                    greek_badness_max=args.greek_badness_max,
                    clean_num_threads=args.clean_num_threads,
                    max_docs=args.max_docs,
                    max_chars=args.max_chars,
                    log_every_rows=args.log_every_rows,
                )
            )
    else:
        with cf.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    process_shard,
                    shard,
                    base_url=args.hplt_base_url,
                    dataset_name=args.dataset_name,
                    quality_min=args.quality_min,
                    exclude_main_registers=set(args.exclude_main_register),
                    require_filter=args.require_filter,
                    batch_size=args.batch_size,
                    rows_per_part=args.rows_per_part,
                    data_root=data_root,
                    target_schema_path=SCHEMA_SOURCE,
                    quality_mode=args.quality_mode,
                    greek_badness_max=args.greek_badness_max,
                    clean_num_threads=args.clean_num_threads,
                    max_docs=args.max_docs,
                    max_chars=args.max_chars,
                    log_every_rows=args.log_every_rows,
                )
                for shard in shards
            ]
            for future in cf.as_completed(futures):
                result = future.result()
                print(json.dumps({'event': 'shard_complete', **asdict(result)}, ensure_ascii=False), flush=True)
                results.append(result)
    results.sort(key=lambda item: (item.quality_bin, item.shard))
    part_files = [Path(path) for result in results for path in result.part_files]
    print(json.dumps({'event': 'metadata_refresh_start', 'release_root': str(release_root), 'part_file_count': len(part_files)}, ensure_ascii=False), flush=True)
    metadata_summary = refresh_release_metadata(release_root)
    print(json.dumps({'event': 'metadata_refresh_done', 'row_count_total': metadata_summary['row_count_total'], 'source_dataset_count': metadata_summary['source_dataset_count']}, ensure_ascii=False), flush=True)
    summary = {
        'dataset_name': args.dataset_name,
        'repo_id': args.repo_id,
        'release_root': str(release_root),
        'quality_min': args.quality_min,
        'exclude_main_register': sorted(set(args.exclude_main_register)),
        'require_filter': args.require_filter,
        'quality_mode': args.quality_mode,
        'greek_badness_max': args.greek_badness_max,
        'clean_num_threads': args.clean_num_threads,
        'removed_existing_parts': [str(path) for path in removed],
        'shards': [asdict(item) for item in results],
        'part_file_count': len(part_files),
        'rows_written_total': sum(item.rows_written for item in results),
        'metadata_refresh': metadata_summary,
        'upload_performed': not args.no_upload,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'summary', 'summary_json': str(summary_json), 'rows_written_total': summary['rows_written_total'], 'part_file_count': len(part_files)}, ensure_ascii=False), flush=True)
    if not args.no_upload:
        patch_root = stage_patch_root(release_root, part_files)
        print(json.dumps({'event': 'upload_start', 'patch_root': str(patch_root), 'repo_id': args.repo_id}, ensure_ascii=False), flush=True)
        upload_patch(patch_root, args.repo_id, args.commit_message, args.commit_description)
        print(json.dumps({'event': 'upload_done', 'repo_id': args.repo_id}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
