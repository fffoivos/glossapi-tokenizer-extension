#!/usr/bin/env python3
"""Run the Stage-35 Rust diagnostic over the exact v3 masked review sample.

``profile_dataset_quality_rust.py`` has a legacy review-sample adapter whose
input contracts are intentionally incompatible with Agent 1 v3.  This narrow
adapter does not pretend that the v3 packet is a legacy packet.  It consumes
only ``agent1_v3_masked_review_sample_v1`` rows, validates their immutable
receipt, then calls the same receipt-bound GlossAPI Rust noise/cleaner runtime
and batch implementation used by the full scan.

The output Parquet contains diagnostics and hashes only; it never persists the
masked review-copy text or any canonical text.  It is diagnostic evidence, not
an admission decision or cleaning action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

import agent1_v3_review as review  # noqa: E402
import agent1_v3_review_evidence as evidence  # noqa: E402
import profile_dataset_quality_rust as quality  # noqa: E402


QUALITY_CONTRACT_SCHEMA = "agent1_v3_masked_review_sample_quality_contract_v1"
QUALITY_SUMMARY_SCHEMA = "agent1_v3_masked_review_sample_quality_summary_v1"
QUALITY_HANDOFF_SCHEMA = "agent1_v3_masked_review_sample_quality_handoff_v1"
TEXT_VARIANT = "high_precision_identifier_masked_review_sample"


def _require_sha256(label: str, value: Any) -> str:
    return evidence.require_sha256(label, value)


def _read_adapter_rows(sample_path: Path, receipt_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the narrow sample adapter without ever consulting canonical data."""

    receipt = evidence.read_json(receipt_path, label="v3 masked review sample receipt")
    if receipt.get("schema_version") != evidence.MASKED_SAMPLE_RECEIPT_SCHEMA or receipt.get("status") != "passed":
        raise ValueError("v3 masked review sample receipt schema/status drift")
    evidence.validate_self_hash(receipt, field="receipt_sha256", label="v3 masked review sample receipt")
    if receipt.get("raw_corpus_included") is not False or receipt.get("text_variant") != TEXT_VARIANT:
        raise ValueError("v3 masked review sample receipt privacy/text variant drift")
    output = receipt.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("v3 masked review sample receipt lacks output binding")
    evidence.binding_matches(output, evidence.file_binding(sample_path), label="v3 masked review sample output")
    rows = evidence.read_jsonl(sample_path, label="v3 masked review sample")
    if output.get("rows") != len(rows) or receipt.get("primary_sample_count") != len(rows):
        raise ValueError("v3 masked review sample row-count drift")
    inventory = receipt.get("primary_sample_inventory")
    if not isinstance(inventory, list) or len(inventory) != len(rows):
        raise ValueError("v3 masked review sample inventory coverage drift")
    expected_fields = {
        "schema_version",
        "sample_id",
        "source_id",
        "source_dataset",
        "source_revision",
        "source_route",
        "sampling_stratum",
        "original_text_sha256",
        "review_copy_sha256",
        "review_request_sha256",
        "text_variant",
        "review_copy",
    }
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    actual_inventory: list[dict[str, Any]] = []
    for number, row in enumerate(rows, start=1):
        if set(row) != expected_fields:
            raise ValueError(f"v3 masked review sample:{number}: unexpected row fields")
        if row.get("schema_version") != evidence.MASKED_SAMPLE_SCHEMA or row.get("text_variant") != TEXT_VARIANT:
            raise ValueError(f"v3 masked review sample:{number}: schema/text variant drift")
        sample_id = _require_sha256(f"v3 masked review sample:{number}.sample_id", row.get("sample_id"))
        if sample_id in seen:
            raise ValueError("v3 masked review sample repeats primary sample")
        seen.add(sample_id)
        for name in ("original_text_sha256", "review_copy_sha256", "review_request_sha256"):
            _require_sha256(f"v3 masked review sample:{number}.{name}", row.get(name))
        if row.get("source_route") not in review.ALLOWED_ROUTES:
            raise ValueError(f"v3 masked review sample:{number}: unsupported source route")
        if row.get("sampling_stratum") not in review.STRATA:
            raise ValueError(f"v3 masked review sample:{number}: unsupported sampling stratum")
        for name in ("source_id", "source_dataset", "source_revision"):
            if not isinstance(row.get(name), str) or not str(row[name]).strip():
                raise ValueError(f"v3 masked review sample:{number}: invalid {name}")
        text = row.get("review_copy")
        if not isinstance(text, str):
            raise ValueError(f"v3 masked review sample:{number}: review_copy must be text")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != row.get("review_copy_sha256"):
            raise ValueError(f"v3 masked review sample:{number}: review-copy hash drift")
        actual_inventory.append(
            {
                "sample_id": sample_id,
                "source_id": str(row["source_id"]),
                "sampling_stratum": str(row["sampling_stratum"]),
                "original_text_sha256": str(row["original_text_sha256"]),
                "review_copy_sha256": str(row["review_copy_sha256"]),
                "review_request_sha256": str(row["review_request_sha256"]),
            }
        )
        normalized.append(dict(row))
    if actual_inventory != inventory or receipt.get("primary_sample_inventory_sha256") != evidence.sha256_json(inventory):
        raise ValueError("v3 masked review sample inventory hash/content drift")
    return normalized, receipt


def _quality_rows(rows: Sequence[Mapping[str, Any]], sample_path: Path) -> list[dict[str, Any]]:
    """Adapt v3 masked rows to the shared Rust batch API without legacy schemas."""

    sample_binding = evidence.file_binding(sample_path)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source_id = str(row["source_id"])
        result.append(
            {
                "source_id": source_id,
                "source_dataset": str(row["source_dataset"]),
                # V3 packets intentionally do not expose raw upstream
                # repository identifiers. Group diagnostics by source_id and
                # make the adapter choice explicit in the summary.
                "source_repo_id": source_id,
                "source_revision": str(row["source_revision"]),
                "stable_uid": str(row["sample_id"]),
                "normalized_text_sha256": str(row["original_text_sha256"]),
                "profile_text_sha256": str(row["review_copy_sha256"]),
                "profile_text_variant": TEXT_VARIANT,
                "input_shard_path": sample_path.name,
                "input_shard_sha256": str(sample_binding["sha256"]),
                "input_row_index": index,
                "private_data_true": False,
                "corrected_version_present": False,
                "text": str(row["review_copy"]),
            }
        )
    return result


def _write_json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    evidence.write_json_no_replace(path, value)


def run_quality(
    *,
    sample_path: Path,
    sample_receipt_path: Path,
    build_receipt: Path,
    expected_commit: str,
    output_dir: Path,
    scratch_dir: Path,
    summary_path: Path,
    handoff_path: Path,
    batch_size: int,
    threads: int,
    quantile_sample_size: int,
) -> dict[str, Any]:
    if batch_size < 1 or threads < 1 or quantile_sample_size < 100:
        raise ValueError("batch-size/threads must be positive and quantile-sample-size must be at least 100")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"immutable masked-sample quality output already exists: {output_dir}")
    for path in (summary_path, handoff_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"immutable masked-sample quality metadata already exists: {path}")
    rows, sample_receipt = _read_adapter_rows(sample_path, sample_receipt_path)
    runtime = quality.validate_runtime_receipt(build_receipt, expected_commit)
    quality_rows = _quality_rows(rows, sample_path)
    output_root = quality.prepare_secure_directory(output_dir, context="v3 masked review sample quality output")
    scratch_root = quality.prepare_secure_directory(scratch_dir, context="v3 masked review sample quality scratch")
    if any(output_root.iterdir()):  # pragma: no cover - guard for unusual race after no-exists check
        raise ValueError("v3 masked review sample quality output unexpectedly became non-empty")
    source_counts = Counter(str(row["source_id"]) for row in rows)
    source_routes: dict[str, str] = {}
    for row in rows:
        source_id = str(row["source_id"])
        route = str(row["source_route"])
        previous = source_routes.setdefault(source_id, route)
        if previous != route:
            raise ValueError(f"{source_id}: masked review sample mixes declared source routes")
    contract: dict[str, Any] = {
        "schema_version": QUALITY_CONTRACT_SCHEMA,
        "scan_mode": "exact_v3_masked_review_sample",
        "diagnostic_only": True,
        "admission_decision": "not_evaluated_in_stage35",
        "sample": {
            "artifact": evidence.file_binding(sample_path),
            "receipt": evidence.file_binding(sample_receipt_path),
            "primary_samples": len(rows),
            "primary_sample_inventory_sha256": str(sample_receipt["primary_sample_inventory_sha256"]),
            "text_variant": TEXT_VARIANT,
            "raw_corpus_included": False,
        },
        "glossapi": {
            "build_receipt": evidence.file_binding(build_receipt),
            "expected_commit": expected_commit,
            "runtime_modules": sorted(str(item.get("name")) for item in runtime.receipt.get("modules", [])),
        },
        "implementation": {
            "quality_adapter_script_sha256": evidence.sha256_file(Path(__file__).resolve()),
            "shared_rust_profiler_script_sha256": evidence.sha256_file(Path(quality.__file__).resolve()),
            "batch_size": batch_size,
            "threads": threads,
            "quantile_sample_size": quantile_sample_size,
        },
        "source_counts": dict(sorted(source_counts.items())),
        "source_routes": dict(sorted(source_routes.items())),
        "source_repository_grouping": "source_id_only_v3_packet_does_not_expose_raw_upstream_repository_id",
        "write_cleaned_files": False,
    }
    contract["contract_sha256"] = evidence.sha256_json(contract)
    contract_path = output_root / "contract.json"
    _write_json_no_replace(contract_path, contract)
    shard = quality.ShardBinding(
        source_id="exact_v3_masked_review_sample",
        path=sample_path.resolve(),
        relative_path=f"review-sample/{sample_path.name}",
        bytes=sample_path.stat().st_size,
        sha256=evidence.sha256_file(sample_path),
        rows=len(quality_rows),
    )
    batch_receipts: list[dict[str, Any]] = []
    for batch_index, row_start in enumerate(range(0, len(quality_rows), batch_size)):
        batch_receipts.append(
            quality.process_batch(
                rows=quality_rows[row_start : row_start + batch_size],
                shard=shard,
                batch_index=batch_index,
                row_start=row_start,
                output_root=output_root,
                scratch_root=scratch_root,
                contract_sha256=str(contract["contract_sha256"]),
                runtime=runtime,
                threads=threads,
            )
        )
    document_output, global_summary, source_summaries = quality.consolidate_batches(
        batch_receipts, output_root=output_root, reservoir_size=quantile_sample_size
    )
    checkpoint_inventory = [
        {
            "receipt": dict(item["receipt"]),
            "output": dict(item["output"]),
            "batch_index": int(item["batch_index"]),
            "row_start": int(item["row_start"]),
            "row_end_exclusive": int(item["row_end_exclusive"]),
        }
        for item in sorted(batch_receipts, key=lambda item: int(item["batch_index"]))
    ]
    if sum(int(item["row_end_exclusive"]) - int(item["row_start"]) for item in checkpoint_inventory) != len(rows):
        raise ValueError("v3 masked review sample Rust checkpoint coverage drift")
    summary: dict[str, Any] = {
        "schema_version": QUALITY_SUMMARY_SCHEMA,
        "status": "passed",
        "scan_mode": "exact_v3_masked_review_sample",
        "diagnostic_only": True,
        "created_at": evidence.utc_now(),
        "contract": evidence.file_binding(contract_path),
        "sample": {
            "artifact": evidence.file_binding(sample_path),
            "receipt": evidence.file_binding(sample_receipt_path),
            "primary_samples": len(rows),
            "primary_sample_inventory_sha256": str(sample_receipt["primary_sample_inventory_sha256"]),
            "text_variant": TEXT_VARIANT,
            "raw_corpus_included": False,
        },
        "glossapi": {
            "build_receipt": evidence.file_binding(build_receipt),
            "commit": expected_commit,
            "runtime_modules": sorted(str(item.get("name")) for item in runtime.receipt.get("modules", [])),
        },
        "document_output": evidence.file_binding(output_root / f"{quality.DOCUMENT_SCHEMA}.parquet"),
        "global": global_summary,
        "source_summaries": source_summaries,
        "checkpoint_closure": {
            "count": len(checkpoint_inventory),
            "rows": len(rows),
            "inventory_sha256": evidence.sha256_json(checkpoint_inventory),
            "inventory": checkpoint_inventory,
        },
        "source_repository_grouping": "source_id_only_v3_packet_does_not_expose_raw_upstream_repository_id",
        "metric_scope": (
            "Exact Stage-30 primary review sample after high-precision direct-identifier masking. "
            "Metrics are diagnostic evidence only and are not population estimates or admission decisions."
        ),
        "admission_decision": "not_evaluated_in_stage35",
    }
    if summary["document_output"]["bytes"] != document_output["bytes"] or summary["document_output"]["sha256"] != document_output["sha256"]:
        raise ValueError("v3 masked review sample consolidated document receipt drift")
    summary["summary_sha256"] = evidence.sha256_json(summary)
    _write_json_no_replace(summary_path, summary)
    handoff: dict[str, Any] = {
        "schema_version": QUALITY_HANDOFF_SCHEMA,
        "status": "passed",
        "created_at": evidence.utc_now(),
        "summary": evidence.file_binding(summary_path),
        "document_output": evidence.file_binding(output_root / f"{quality.DOCUMENT_SCHEMA}.parquet"),
        "sample_receipt": evidence.file_binding(sample_receipt_path),
        "glossapi_build_receipt": evidence.file_binding(build_receipt),
        "diagnostic_only": True,
        "raw_corpus_included": False,
        "admission_decision": "not_evaluated_in_stage35",
    }
    handoff["handoff_sha256"] = evidence.sha256_json(handoff)
    _write_json_no_replace(handoff_path, handoff)
    return {"summary": summary, "handoff": handoff}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sample", type=Path, required=True)
    result.add_argument("--sample-receipt", type=Path, required=True)
    result.add_argument("--build-receipt", type=Path, required=True)
    result.add_argument("--expected-commit", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--scratch-dir", type=Path, required=True)
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument("--handoff", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--threads", type=int, default=16)
    result.add_argument("--quantile-sample-size", type=int, default=1024)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = run_quality(
        sample_path=args.sample,
        sample_receipt_path=args.sample_receipt,
        build_receipt=args.build_receipt,
        expected_commit=args.expected_commit,
        output_dir=args.output_dir,
        scratch_dir=args.scratch_dir,
        summary_path=args.summary,
        handoff_path=args.handoff,
        batch_size=args.batch_size,
        threads=args.threads,
        quantile_sample_size=args.quantile_sample_size,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "summary_sha256": result["summary"]["summary_sha256"],
                "handoff_sha256": result["handoff"]["handoff_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
