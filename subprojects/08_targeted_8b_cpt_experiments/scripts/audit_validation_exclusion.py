#!/usr/bin/env python3
"""Reconcile validation exclusions per shard and prove zero heldout overlap."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from contract_utils import file_binding, read_json, require, write_json_atomic
from exclude_frozen_validation_content import freeze_heldout_hashes


Identity = tuple[str, str, str]
_HELDOUT_HASHES: set[str] = set()


def identities(path: Path) -> collections.Counter[Identity]:
    rows: collections.Counter[Identity] = collections.Counter()
    parquet = pq.ParquetFile(path)
    require(
        {"source_dataset", "source_doc_id", "text"}.issubset(parquet.schema_arrow.names),
        f"identity/text columns missing: {path}",
    )
    for batch in parquet.iter_batches(
        columns=["source_dataset", "source_doc_id", "text"], batch_size=2048, use_threads=False
    ):
        for row in batch.to_pylist():
            text_hash = hashlib.sha256(str(row.get("text") or "").encode("utf-8")).hexdigest()
            rows[(str(row.get("source_dataset") or ""), str(row.get("source_doc_id") or ""), text_hash)] += 1
    return rows


def ledger(path: Path) -> tuple[collections.Counter[Identity], collections.Counter[str]]:
    rows: collections.Counter[Identity] = collections.Counter()
    actions: collections.Counter[str] = collections.Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, use_threads=False):
        for row in batch.to_pylist():
            identity = (str(row["source_dataset"]), str(row["source_doc_id"]), str(row["input_text_sha256"]))
            rows[identity] += 1
            actions[str(row["action"])] += 1
    return rows, actions


def worker_init(heldout_hashes: set[str]) -> None:
    global _HELDOUT_HASHES
    _HELDOUT_HASHES = heldout_hashes


def reconcile_shard(task: dict[str, Any]) -> dict[str, Any]:
    relative = task["relative_path"]
    for label in ("kept", "excluded", "ledger"):
        require(file_binding(Path(task[label]["path"])) == task[label], f"{relative}: {label} binding drift")
    input_rows = identities(Path(task["input"]))
    kept_rows = identities(Path(task["kept"]["path"]))
    excluded_rows = identities(Path(task["excluded"]["path"]))
    ledger_rows, actions = ledger(Path(task["ledger"]["path"]))
    require(input_rows == kept_rows + excluded_rows, f"{relative}: input != kept + excluded")
    require(input_rows == ledger_rows, f"{relative}: ledger != input")
    require(actions["keep"] == sum(kept_rows.values()), f"{relative}: kept ledger count mismatch")
    require(actions["exclude"] == sum(excluded_rows.values()), f"{relative}: excluded ledger count mismatch")
    kept_overlap = sum(count for identity, count in kept_rows.items() if identity[2] in _HELDOUT_HASHES)
    excluded_nonoverlap = sum(count for identity, count in excluded_rows.items() if identity[2] not in _HELDOUT_HASHES)
    require(kept_overlap == 0, f"{relative}: kept rows overlap frozen validation: {kept_overlap}")
    require(excluded_nonoverlap == 0, f"{relative}: non-validation rows were excluded: {excluded_nonoverlap}")
    return {
        "relative_path": relative,
        "input": sum(input_rows.values()),
        "kept": sum(kept_rows.values()),
        "excluded": sum(excluded_rows.values()),
        "ledger": sum(ledger_rows.values()),
        "kept_overlap": kept_overlap,
        "excluded_nonoverlap": excluded_nonoverlap,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    require(not args.receipt.exists(), f"immutable audit receipt exists: {args.receipt}")
    require(args.workers >= 1, "--workers must be positive")
    manifest = read_json(args.exclusion_manifest)
    require(manifest.get("status") == "completed", "exclusion manifest is not completed")
    validation_path = Path(manifest["validation_manifest"]["path"])
    heldout_hashes, panels = freeze_heldout_hashes(validation_path)
    input_root = Path(manifest["input"])
    file_rows = manifest.get("files", [])
    require(file_rows, "exclusion manifest has no file rows")
    tasks: list[dict[str, Any]] = []
    for row in file_rows:
        relative = str(row.get("relative_path", ""))
        relative_path = Path(relative)
        require(
            relative and not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"unsafe relative path: {relative}",
        )
        input_path = input_root / relative_path
        require(input_path.is_file(), f"missing exclusion input: {input_path}")
        tasks.append(
            {
                "relative_path": relative,
                "input": str(input_path),
                "kept": row["kept"],
                "excluded": row["excluded"],
                "ledger": row["ledger"],
            }
        )

    totals: collections.Counter[str] = collections.Counter()
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks)),
        mp_context=mp.get_context("fork"),
        initializer=worker_init,
        initargs=(heldout_hashes,),
    ) as executor:
        for result in executor.map(reconcile_shard, tasks, chunksize=1):
            for key in ("input", "kept", "excluded", "ledger", "kept_overlap", "excluded_nonoverlap"):
                totals[key] += int(result[key])
    declared = manifest.get("counts", {})
    require(
        totals["input"] == int(declared.get("input", -1))
        and totals["kept"] == int(declared.get("kept", -1))
        and totals["excluded"] == int(declared.get("excluded", -1))
        and totals["ledger"] == totals["input"],
        "aggregate shard accounting/manifest drift",
    )
    payload = {
        "schema_version": "targeted_8b_validation_exclusion_audit_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "exclusion_manifest": file_binding(args.exclusion_manifest),
        "validation_manifest": file_binding(validation_path),
        "validation_panels": len(panels),
        "counts": {
            "input": totals["input"],
            "kept": totals["kept"],
            "excluded": totals["excluded"],
            "ledger": totals["ledger"],
            "kept_validation_exact_content_overlap": totals["kept_overlap"],
            "excluded_non_validation_content": totals["excluded_nonoverlap"],
        },
        "checks": {
            "exact_identity_multiplicities_reconcile": True,
            "ledger_actions_reconcile": True,
            "kept_zero_exact_validation_overlap": True,
            "every_exclusion_is_exact_validation_content": True,
            "non_validation_duplicates_preserved": True,
            "deduplication_performed": False,
            "identity_reconciliation_is_parallel_and_shard_bounded": True,
        },
        "workers": args.workers,
        "files": len(tasks),
    }
    write_json_atomic(args.receipt, payload)
    print(json.dumps(payload["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
