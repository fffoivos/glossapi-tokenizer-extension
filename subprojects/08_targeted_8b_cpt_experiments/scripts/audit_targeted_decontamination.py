#!/usr/bin/env python3
"""Independently reconcile and rescan a targeted GreekMMLU decontamination."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow.parquet as pq

from contract_utils import file_binding, read_json, require, write_json_atomic
from decontaminate_targeted_corpus import load_canonical


Identity = tuple[str, str, str]
_CANONICAL: ModuleType | None = None
_INDEX: Any = None


def declared_counts_match(totals: collections.Counter[str], declared: dict[str, Any]) -> bool:
    """Match canonical counts, treating an omitted sparse zero as zero."""
    return (
        totals["input"] == int(declared.get("input", -1))
        and totals["kept"] == int(declared.get("kept", -1))
        and totals["dropped"] == int(declared.get("dropped", 0))
        and totals["ledger"] == totals["input"]
    )


def parquet_identities(path: Path) -> collections.Counter[Identity]:
    result: collections.Counter[Identity] = collections.Counter()
    parquet = pq.ParquetFile(path)
    require(
        {"source_dataset", "source_doc_id", "text"}.issubset(parquet.schema_arrow.names),
        f"identity/text columns missing in {path}",
    )
    for batch in parquet.iter_batches(columns=["source_dataset", "source_doc_id", "text"], batch_size=2048, use_threads=False):
        for row in batch.to_pylist():
            text = str(row.get("text") or "")
            result[
                (
                    str(row.get("source_dataset") or ""),
                    str(row.get("source_doc_id") or ""),
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            ] += 1
    return result


def parquet_ledger(path: Path) -> tuple[collections.Counter[Identity], collections.Counter[str]]:
    result: collections.Counter[Identity] = collections.Counter()
    actions: collections.Counter[str] = collections.Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, use_threads=False):
        for row in batch.to_pylist():
            result[(str(row["source_dataset"]), str(row["source_doc_id"]), str(row["input_text_sha256"]))] += 1
            actions[str(row["action"])] += 1
    return result, actions


def rescan_worker_init(canonical_path: str, index: Any) -> None:
    global _CANONICAL, _INDEX
    _CANONICAL = load_canonical(Path(canonical_path))
    _INDEX = index


def rescan_file(path_string: str) -> list[dict[str, Any]]:
    canonical = _CANONICAL
    if canonical is None or _INDEX is None:
        raise RuntimeError("audit worker was not initialized")
    path = Path(path_string)
    matches: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=["source_dataset", "source_doc_id", "text"],
        batch_size=512,
        use_threads=False,
    ):
        for row in batch.to_pylist():
            text = str(row.get("text") or "")
            action, reason, evidence = canonical.match_document(text, _INDEX)
            if action == "drop":
                matches.append(
                    {
                        "identity": (
                            str(row.get("source_dataset") or ""),
                            str(row.get("source_doc_id") or ""),
                            hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        ),
                        "reason": reason,
                        "evidence": evidence,
                    }
                )
                if len(matches) >= 20:
                    return matches
    return matches


def reconcile_and_rescan(task: dict[str, Any]) -> dict[str, Any]:
    relative = task["relative_path"]
    input_string = task["input"]
    kept_string = task["output"]["path"]
    dropped_string = task["dropped"]["path"]
    ledger_string = task["ledger"]["path"]
    for label in ("output", "dropped", "ledger"):
        require(file_binding(Path(task[label]["path"])) == task[label], f"{relative}: {label} binding drift")
    input_ids = parquet_identities(Path(input_string))
    kept_ids = parquet_identities(Path(kept_string))
    dropped_ids = parquet_identities(Path(dropped_string))
    ledger_ids, actions = parquet_ledger(Path(ledger_string))
    require(input_ids == kept_ids + dropped_ids, f"{relative}: input != kept + dropped")
    require(input_ids == ledger_ids, f"{relative}: ledger identities != input")
    require(actions["keep"] == sum(kept_ids.values()), f"{relative}: ledger kept count mismatch")
    require(actions["drop"] == sum(dropped_ids.values()), f"{relative}: ledger dropped count mismatch")
    return {
        "relative_path": relative,
        "input": sum(input_ids.values()),
        "kept": sum(kept_ids.values()),
        "dropped": sum(dropped_ids.values()),
        "ledger": sum(ledger_ids.values()),
        "actions": dict(actions),
        "remaining": rescan_file(kept_string),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-script", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    require(not args.receipt.exists(), f"immutable audit receipt exists: {args.receipt}")
    require(args.workers >= 1, "--workers must be positive")
    manifest = read_json(args.manifest)
    require(manifest.get("status") == "completed", "decontamination manifest is not completed")
    input_root = Path(manifest["input"])
    file_rows = manifest.get("files", [])
    require(file_rows, "decontamination manifest has no file rows")
    tasks: list[dict[str, Any]] = []
    for row in file_rows:
        relative = str(row.get("relative_path", ""))
        relative_path = Path(relative)
        require(
            relative
            and not relative_path.is_absolute()
            and ".." not in relative_path.parts,
            f"unsafe relative path: {relative}",
        )
        paths = {
            "input": input_root / relative_path,
            "output": Path(row["output"]["path"]),
            "dropped": Path(row["dropped"]["path"]),
            "ledger": Path(row["ledger"]["path"]),
        }
        require(paths["input"].is_file(), f"missing decontamination input: {paths['input']}")
        tasks.append(
            {
                "relative_path": relative,
                "input": str(paths["input"]),
                "output": row["output"],
                "dropped": row["dropped"],
                "ledger": row["ledger"],
            }
        )

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
    totals: collections.Counter[str] = collections.Counter()
    remaining: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks)),
        mp_context=mp.get_context("fork"),
        initializer=rescan_worker_init,
        initargs=(str(args.canonical_script.resolve()), index),
    ) as executor:
        for result in executor.map(reconcile_and_rescan, tasks, chunksize=1):
            for key in ("input", "kept", "dropped", "ledger"):
                totals[key] += int(result[key])
            remaining.extend(result["remaining"])
    remaining = remaining[:20]
    require(not remaining, f"kept corpus still contains high-confidence matches: {remaining}")
    declared = manifest.get("counts", {})
    require(declared_counts_match(totals, declared), "aggregate shard accounting/manifest drift")

    payload = {
        "schema_version": "targeted_8b_decontamination_audit_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decontamination_manifest": file_binding(args.manifest),
        "canonical_implementation": file_binding(args.canonical_script),
        "benchmark": benchmark,
        "counts": {
            "input": totals["input"],
            "kept": totals["kept"],
            "dropped": totals["dropped"],
            "ledger": totals["ledger"],
            "remaining_high_confidence_matches": 0,
        },
        "checks": {
            "input_equals_kept_plus_dropped_by_exact_identity_multiplicity": True,
            "ledger_equals_input_by_exact_identity_multiplicity": True,
            "ledger_actions_reconcile": True,
            "kept_corpus_independently_rescanned": True,
            "kept_rescan_is_parallel_and_read_only": True,
            "identity_reconciliation_is_parallel_and_shard_bounded": True,
            "no_high_confidence_match_remains": True,
            "no_deduplication_performed": True,
        },
        "workers": args.workers,
        "files": len(tasks),
    }
    write_json_atomic(args.receipt, payload)
    print(json.dumps(payload["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
