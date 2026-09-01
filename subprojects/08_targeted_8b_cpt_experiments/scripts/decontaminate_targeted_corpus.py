#!/usr/bin/env python3
"""Apply the canonical GreekMMLU policy to the anonymized HF release schema.

The canonical implementation is loaded from an immutable code bundle and owns
all normalization and matching decisions.  This adapter changes only the
ledger identity: the public release has source_dataset/source_doc_id but no
pre-anonymization stable_uid or acquisition_source_id columns.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from contract_utils import file_binding, require, write_json_atomic


_CANONICAL: ModuleType | None = None
_INDEX: Any = None


def load_canonical(path: Path) -> ModuleType:
    module_dir = str(path.resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("target8_canonical_decontam", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import canonical decontaminator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ledger_schema() -> pa.Schema:
    return pa.schema(
        [
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("input_text_sha256", pa.string()),
            ("action", pa.string()),
            ("reason", pa.string()),
            ("benchmark_matches_json", pa.string()),
        ]
    )


def worker_init(canonical_path: str, index: Any) -> None:
    global _CANONICAL, _INDEX
    _CANONICAL = load_canonical(Path(canonical_path))
    _INDEX = index


def _temp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def process_file(task: tuple[str, str, str, str, str]) -> dict[str, Any]:
    input_name, relative, output_root, dropped_root, ledger_root = task
    input_path = Path(input_name)
    output_path = Path(output_root) / relative
    dropped_path = Path(dropped_root) / relative
    ledger_path = Path(ledger_root) / relative
    canonical = _CANONICAL
    if canonical is None or _INDEX is None:
        raise RuntimeError("decontamination worker was not initialized")
    parquet = pq.ParquetFile(input_path)
    required = {"source_dataset", "source_doc_id", "text"}
    missing = required - set(parquet.schema_arrow.names)
    require(not missing, f"{input_path}: missing required columns: {sorted(missing)}")
    for path in (output_path, dropped_path, ledger_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_temp, dropped_temp, ledger_temp = map(_temp, (output_path, dropped_path, ledger_path))
    output_writer = pq.ParquetWriter(output_temp, parquet.schema_arrow, compression="zstd")
    dropped_writer = pq.ParquetWriter(dropped_temp, parquet.schema_arrow, compression="zstd")
    action_writer = pq.ParquetWriter(ledger_temp, ledger_schema(), compression="zstd")
    counts: collections.Counter[str] = collections.Counter()
    try:
        for batch in parquet.iter_batches(batch_size=512, use_threads=False):
            kept_rows: list[dict[str, Any]] = []
            dropped_rows: list[dict[str, Any]] = []
            actions: list[dict[str, Any]] = []
            for row in batch.to_pylist():
                text = str(row.get("text") or "")
                action, reason, evidence = canonical.match_document(text, _INDEX)
                actions.append(
                    {
                        "source_dataset": str(row.get("source_dataset") or ""),
                        "source_doc_id": str(row.get("source_doc_id") or ""),
                        "input_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "action": action,
                        "reason": reason,
                        "benchmark_matches_json": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                    }
                )
                counts["input"] += 1
                if action == "drop":
                    dropped_rows.append(row)
                    counts["dropped"] += 1
                else:
                    kept_rows.append(row)
                    counts["kept"] += 1
                    if evidence:
                        counts["audit_candidates"] += 1
            if kept_rows:
                output_writer.write_table(pa.Table.from_pylist(kept_rows, schema=parquet.schema_arrow))
            if dropped_rows:
                dropped_writer.write_table(pa.Table.from_pylist(dropped_rows, schema=parquet.schema_arrow))
            if actions:
                action_writer.write_table(pa.Table.from_pylist(actions, schema=ledger_schema()))
    except BaseException:
        output_writer.close()
        dropped_writer.close()
        action_writer.close()
        for path in (output_temp, dropped_temp, ledger_temp):
            path.unlink(missing_ok=True)
        raise
    output_writer.close()
    dropped_writer.close()
    action_writer.close()
    for source, destination in (
        (output_temp, output_path),
        (dropped_temp, dropped_path),
        (ledger_temp, ledger_path),
    ):
        os.replace(source, destination)
    return {
        "relative_path": relative,
        "counts": dict(counts),
        "output": file_binding(output_path),
        "dropped": file_binding(dropped_path),
        "ledger": file_binding(ledger_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-script", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dropped", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    require(args.workers >= 1, "--workers must be positive")
    require(not args.manifest.exists(), f"immutable manifest exists: {args.manifest}")
    for root in (args.output, args.dropped, args.ledger):
        require(not root.exists(), f"refusing to reuse output root: {root}")
    canonical = load_canonical(args.canonical_script)
    index, benchmark = canonical.load_benchmark_index(
        args.queries_jsonl,
        args.benchmark_manifest,
        k=canonical.DEFAULT_K,
        min_coverage=canonical.DEFAULT_MIN_COVERAGE,
        minhash_threshold=canonical.DEFAULT_MINHASH_THRESHOLD,
        min_matched_grams=canonical.DEFAULT_MIN_MATCHED_GRAMS,
        max_gap_tokens=canonical.DEFAULT_MAX_GAP,
    )
    files = sorted(args.input.rglob("*.parquet"))
    require(files, f"no Parquet files below {args.input}")
    tasks = [
        (str(path), path.relative_to(args.input).as_posix(), str(args.output), str(args.dropped), str(args.ledger))
        for path in files
    ]
    created_roots: list[Path] = []
    try:
        for root in (args.output, args.dropped, args.ledger):
            root.mkdir(parents=True)
            created_roots.append(root)
        context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=worker_init,
            initargs=(str(args.canonical_script.resolve()), index),
        ) as executor:
            files_receipts = list(executor.map(process_file, tasks, chunksize=1))
        totals: collections.Counter[str] = collections.Counter()
        for receipt in files_receipts:
            totals.update(receipt["counts"])
        require(totals["input"] == totals["kept"] + totals["dropped"], "row accounting mismatch")
        payload = {
            "schema_version": "targeted_8b_greekmmlu_decontamination_v1",
            "status": "completed",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "dropped": str(args.dropped.resolve()),
            "ledger": str(args.ledger.resolve()),
            "canonical_implementation": file_binding(args.canonical_script),
            "benchmark": benchmark,
            "policy": {
                "policy_version": canonical.POLICY_VERSION,
                "normalization": "NFKC+strip_combining_marks+casefold+unicode_word_tokens_v1",
                "k": canonical.DEFAULT_K,
                "min_coverage": canonical.DEFAULT_MIN_COVERAGE,
                "minhash_threshold": canonical.DEFAULT_MINHASH_THRESHOLD,
                "minhash_permutations": canonical.MINHASH_PERMUTATIONS,
                "min_matched_grams": canonical.DEFAULT_MIN_MATCHED_GRAMS,
                "max_gap_tokens": canonical.DEFAULT_MAX_GAP,
                "answer_only_action": "audit_only",
            },
            "identity": ["source_dataset", "source_doc_id", "input_text_sha256"],
            "invariants": {
                "text_transformed": False,
                "global_deduplication_performed": False,
                "near_deduplication_performed": False,
                "only_high_confidence_greekmmlu_matches_removed": True,
            },
            "workers": args.workers,
            "counts": dict(totals),
            "files": sorted(files_receipts, key=lambda row: row["relative_path"]),
        }
        write_json_atomic(args.manifest, payload)
    except BaseException:
        for root in reversed(created_roots):
            shutil.rmtree(root, ignore_errors=True)
        try:
            args.manifest.parent.rmdir()
        except OSError:
            pass
        raise
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": dict(totals)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
