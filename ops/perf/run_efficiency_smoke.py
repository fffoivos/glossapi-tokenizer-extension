#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from glossapi_corpus_cli import pipeline, text_dedup  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight efficiency smoke checks for the canonical tokenizer pipeline repo.")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--target", choices=["all", "near_candidates", "mix_streaming"], default="all")
    parser.add_argument("--candidate-docs", type=int, default=512)
    parser.add_argument("--candidate-group-size", type=int, default=32)
    parser.add_argument("--candidate-workers", type=int, default=16)
    parser.add_argument("--candidate-max-total-rss-mb", type=float, default=None)
    parser.add_argument("--mix-rows", type=int, default=12000)
    parser.add_argument("--mix-max-rss-mb", type=float, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


class LinuxProcessSampler:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_self_rss_kb = 0
        self.peak_total_child_rss_kb = 0
        self.peak_self_pss_kb = 0
        self.peak_total_child_pss_kb = 0
        self.peak_child_count = 0

    def _read_rss_kb(self, pid: int) -> int:
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except FileNotFoundError:
            return 0
        return 0

    def _read_pss_kb(self, pid: int) -> int:
        try:
            with open(f"/proc/{pid}/smaps_rollup", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("Pss:"):
                        return int(line.split()[1])
        except FileNotFoundError:
            return 0
        except PermissionError:
            return 0
        return 0

    def _process_table(self) -> dict[int, int]:
        table: dict[int, int] = {}
        proc_root = Path("/proc")
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                with open(entry / "stat", encoding="utf-8") as handle:
                    raw = handle.read()
            except (FileNotFoundError, ProcessLookupError):
                continue
            try:
                _prefix, rest = raw.split(")", 1)
                parts = rest.strip().split()
                ppid = int(parts[1])
            except (ValueError, IndexError):
                continue
            table[int(entry.name)] = ppid
        return table

    def _descendant_pids(self) -> list[int]:
        process_table = self._process_table()
        descendants: list[int] = []
        queue: list[int] = [self.pid]
        while queue:
            parent_pid = queue.pop()
            child_pids = [pid for pid, ppid in process_table.items() if ppid == parent_pid]
            descendants.extend(child_pids)
            queue.extend(child_pids)
        return descendants

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_self_rss_kb = max(self.peak_self_rss_kb, self._read_rss_kb(self.pid))
            self.peak_self_pss_kb = max(self.peak_self_pss_kb, self._read_pss_kb(self.pid))
            child_pids = self._descendant_pids()
            total_child_rss = sum(self._read_rss_kb(child_pid) for child_pid in child_pids)
            total_child_pss = sum(self._read_pss_kb(child_pid) for child_pid in child_pids)
            self.peak_child_count = max(self.peak_child_count, len(child_pids))
            self.peak_total_child_rss_kb = max(self.peak_total_child_rss_kb, total_child_rss)
            self.peak_total_child_pss_kb = max(self.peak_total_child_pss_kb, total_child_pss)
            time.sleep(0.1)

    def __enter__(self) -> "LinuxProcessSampler":
        if Path("/proc").exists():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def make_canonical_row(source_dataset: str, source_doc_id: str, text: str, *, needs_ocr: bool = False) -> dict[str, object]:
    return {
        "source_dataset": source_dataset,
        "source_doc_id": source_doc_id,
        "text": text,
        "title": None,
        "author": None,
        "source_metadata_json": None,
        "is_historical_or_polytonic": False,
        "contains_math": False,
        "contains_latex": False,
        "greek_percentage": None,
        "latin_percentage": None,
        "polytonic_ratio": None,
        "table_ratio": None,
        "greek_badness_score": 0.0,
        "mojibake_badness_score": 0.0,
        "needs_ocr": needs_ocr,
        "is_empty": False,
        "filter": "keep",
        "ocr_success": True,
        "quality_method": None,
        "reevaluated_at": None,
    }


def write_canonical_snapshot(output_root: Path, rows: list[dict[str, object]]) -> None:
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame = pipeline.finalize_frame(frame)
    table = pa.Table.from_pandas(
        frame[pipeline.CANONICAL_COLUMNS],
        schema=pipeline.CANONICAL_ARROW_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, data_root / "part-00000.parquet", compression="zstd")


def write_source_mix_config(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")


def run_mix_streaming_smoke(work_root: Path, *, rows: int, max_rss_mb: float | None) -> dict[str, object]:
    release_root = work_root / "mix_streaming" / "release"
    mix_path = work_root / "mix_streaming" / "mix.parquet"
    mix_config_path = work_root / "mix_streaming" / "source_mix.json"
    generated_rows: list[dict[str, object]] = []
    for idx in range(rows):
        if idx % 4 == 0:
            dataset = "HPLT/ell_Grek_ge8_no_mt_clean60"
            text = "η" * 900
        else:
            dataset = "openarchives.gr" if idx % 2 == 0 else "alpha"
            text = "α" * 220
        generated_rows.append(make_canonical_row(dataset, f"doc-{idx:06d}", text, needs_ocr=(dataset == "openarchives.gr" and idx % 40 == 0)))
    write_canonical_snapshot(release_root, generated_rows)
    write_source_mix_config(
        mix_config_path,
        [
            {
                "name": "all_non_hplt",
                "exclude_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
                "fraction": 1.0,
                "fraction_mode": "of_group",
            },
            {
                "name": "hplt",
                "include_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
                "fraction": 0.30,
                "fraction_mode": "of_total",
            },
        ],
    )
    streaming_expected = pipeline.build_mix_should_stream(
        mix_output_path=mix_path,
        dedup_metadata_root=None,
        dedup_action="ignore",
        source_mix_config_path=mix_config_path,
    )
    started = time.perf_counter()
    with LinuxProcessSampler(os.getpid()) as sampler:
        payload = pipeline.build_mix_export(
            output_root=release_root,
            mix_output_path=mix_path,
            exclude_needs_ocr_sources=["openarchives.gr"],
            source_mix_config_path=mix_config_path,
        )
    elapsed = time.perf_counter() - started
    peak_rss_mb = sampler.peak_self_rss_kb / 1024.0
    if max_rss_mb is not None and peak_rss_mb > max_rss_mb:
        raise RuntimeError(f"mix_streaming peak RSS exceeded limit: {peak_rss_mb:.1f} MB > {max_rss_mb:.1f} MB")
    return {
        "target": "mix_streaming",
        "rows_input": rows,
        "rows_output": int(payload["rows_kept"]),
        "streaming_expected": bool(streaming_expected),
        "elapsed_seconds": round(elapsed, 3),
        "peak_self_rss_mb": round(peak_rss_mb, 3),
        "mix_output_path": str(mix_path),
        "source_mix_summary_path": str(mix_path.with_suffix(".source_mix_summary.json")),
    }


def build_signatures_stage(stage_root: Path, *, docs: int, group_size: int, bands: int, rows_per_band: int) -> None:
    stage_root.mkdir(parents=True, exist_ok=True)
    signature_rows: list[dict[str, object]] = []
    bucket_rows_by_band: dict[int, list[dict[str, object]]] = {band_index: [] for band_index in range(bands)}
    num_perm = bands * rows_per_band
    for idx in range(docs):
        group_id = idx // group_size
        doc_key = f"doc-{idx:06d}"
        signature = [int(group_id + band_index) for band_index in range(num_perm)]
        signature_rows.append(
            {
                "doc_key": doc_key,
                "source_dataset": "synthetic",
                "source_doc_id": doc_key,
                "file_path": "synthetic.parquet",
                "row_group_index": 0,
                "row_index_in_file": idx,
                "token_count": 512,
                "char_count": 4096,
                "near_text_chars": 4096,
                "shingle_count": 256,
                "shingle_mode": "token",
                "shingle_size": 5,
                "signature": signature,
            }
        )
        for band_index in range(bands):
            bucket_rows_by_band[band_index].append(
                {
                    "doc_key": doc_key,
                    "band_index": band_index,
                    "bucket_hash": f"band-{band_index:02d}-group-{group_id:04d}",
                    "token_count": 512,
                    "char_count": 4096,
                    "shingle_mode": "token",
                }
            )
    pq.write_table(pa.Table.from_pylist(signature_rows, schema=text_dedup.SIGNATURE_SCHEMA), stage_root / "signatures.parquet")
    text_dedup.build_signature_matrix_artifacts(stage_root / "signatures.parquet", num_perm=num_perm)
    for band_index, rows_for_band in bucket_rows_by_band.items():
        band_dir = stage_root / "shards" / "lsh_buckets" / f"band_{band_index:02d}"
        band_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(rows_for_band, schema=text_dedup.LSH_BUCKET_SCHEMA),
            band_dir / "synthetic.parquet",
        )


def run_near_candidate_efficiency_smoke(
    work_root: Path,
    *,
    docs: int,
    group_size: int,
    requested_workers: int,
    max_total_rss_mb: float | None,
) -> dict[str, object]:
    state_root = work_root / "near_candidates" / "state"
    run_root = work_root / "near_candidates" / "run"
    stage_root = run_root / "stage_02_near"
    bands = 32
    rows_per_band = 4
    num_perm = bands * rows_per_band
    build_signatures_stage(stage_root, docs=docs, group_size=group_size, bands=bands, rows_per_band=rows_per_band)
    config = text_dedup.pipeline_config_payload(
        input_root=work_root / "near_candidates" / "input",
        state_root=state_root,
        run_root=run_root,
        greek_diacritic_policy=text_dedup.DEFAULT_GREEK_DIACRITIC_POLICY,
        exact_only=False,
        max_workers=requested_workers,
        minhash_threshold=text_dedup.DEFAULT_NEAR_THRESHOLD,
        num_perm=num_perm,
        bands=bands,
        rows_per_band=rows_per_band,
        shingle_mode="token",
        shingle_size=5,
        large_component_threshold=text_dedup.DEFAULT_LARGE_COMPONENT_THRESHOLD,
        max_bucket_size=text_dedup.DEFAULT_MAX_BUCKET_SIZE,
    )
    run_id, digest = text_dedup.ensure_run_config(run_root=run_root, config=config, resume=False)
    conn = text_dedup.connect_db(state_root)
    try:
        text_dedup.init_run(conn, run_id=run_id, config=config, config_digest=digest)
        started = time.perf_counter()
        with LinuxProcessSampler(os.getpid()) as sampler:
            summary = text_dedup._run_near_candidate_stage(
                conn,
                run_id=run_id,
                run_root=run_root,
                state_root=state_root,
                config=config,
                minhash_threshold=text_dedup.DEFAULT_NEAR_THRESHOLD,
                bands=bands,
                rows_per_band=rows_per_band,
                max_workers=requested_workers,
                max_bucket_size=text_dedup.DEFAULT_MAX_BUCKET_SIZE,
            )
        elapsed = time.perf_counter() - started
    finally:
        conn.close()
    peak_total_child_rss_mb = sampler.peak_total_child_rss_kb / 1024.0
    peak_total_child_pss_mb = sampler.peak_total_child_pss_kb / 1024.0
    if max_total_rss_mb is not None and peak_total_child_rss_mb > max_total_rss_mb:
        raise RuntimeError(
            f"near_candidates peak child RSS exceeded limit: {peak_total_child_rss_mb:.1f} MB > {max_total_rss_mb:.1f} MB"
        )
    return {
        "target": "near_candidates",
        "docs_input": docs,
        "group_size": group_size,
        "requested_workers": requested_workers,
        "capped_workers": min(requested_workers, text_dedup.near_candidate_worker_cap(), bands),
        "start_method": text_dedup.candidate_process_pool_context().get_start_method(),
        "elapsed_seconds": round(elapsed, 3),
        "peak_child_count": int(sampler.peak_child_count),
        "peak_total_child_rss_mb": round(peak_total_child_rss_mb, 3),
        "peak_total_child_pss_mb": round(peak_total_child_pss_mb, 3),
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    work_root = args.work_root.resolve()
    if work_root.exists():
        for child in work_root.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    work_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "repo_root": str(REPO_ROOT),
        "work_root": str(work_root),
    }
    if args.target in {"all", "mix_streaming"}:
        results["mix_streaming"] = run_mix_streaming_smoke(
            work_root,
            rows=args.mix_rows,
            max_rss_mb=args.mix_max_rss_mb,
        )
    if args.target in {"all", "near_candidates"}:
        results["near_candidates"] = run_near_candidate_efficiency_smoke(
            work_root,
            docs=args.candidate_docs,
            group_size=args.candidate_group_size,
            requested_workers=args.candidate_workers,
            max_total_rss_mb=args.candidate_max_total_rss_mb,
        )
    summary_json = args.summary_json or (work_root / "efficiency_summary.json")
    summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
