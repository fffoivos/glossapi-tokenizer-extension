#!/usr/bin/env python3
"""Verify and freeze the exact historical public-GreekMMLU evaluator."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic
from producer_bundle_compatibility import load_authority, require_accepted_producer


EXPECTED_REVISION = "cfdd0e7b00761a736be660867bf3d09733e24a92"


def validate_contract(value: dict[str, Any]) -> None:
    require(value.get("schema_version") == "apertus_legacy_public_greekmmlu_v1", "legacy evaluator schema drift")
    require(value.get("status") == "frozen", "legacy evaluator contract is not frozen")
    require(value["code"]["revision"] == EXPECTED_REVISION, "legacy evaluator revision drift")
    dataset = value["dataset"]
    require(dataset["repo_id"] == "dascim/GreekMMLU", "legacy evaluator dataset drift")
    require(dataset["revision"] == "6a03aa06b68beb932fb75edff3a34e50b3674649", "legacy evaluator revision drift")
    require((dataset["config"], dataset["split"], dataset["questions"]) == ("All", "test", 16632), "legacy evaluator panel drift")
    require(dataset["historical_code_did_not_forward_revision"] is True, "historical loader caveat hidden")
    require(dataset["historical_june_revision_is_unrecoverable"] is True, "unrecoverable dataset drift hidden")
    require(dataset["execution_input"] == "pre_materialized_snapshot_at_pinned_revision", "legacy input is not pinned")
    require(dataset["snapshot_must_reconcile_to_surviving_20260731_full_query_set"] is True, "legacy query reconciliation disabled")
    require(dataset["unresolved_dataset_drift_is_a_named_reconstruction_difference"] is True, "legacy dataset drift unnamed")
    invocation = value["invocation"]
    require(invocation["benchmarks"] == "greekmmlu" and invocation["sample_size"] == 0, "legacy evaluator panel drift")
    require(invocation["dtype"] == "bfloat16", "legacy evaluator dtype drift")
    require(invocation["candidate_batch_size"] == invocation["example_batch_size"] == 16, "legacy evaluator batching drift")
    wrapper = value["compatibility_wrapper"]
    require(wrapper["scoring_arithmetic_must_remain_at_cfdd0e7b"] is True, "legacy scoring arithmetic may drift")
    require(wrapper["only_dataset_loading_may_be_replaced_by_the_frozen_snapshot"] is True, "legacy wrapper scope drift")
    require(wrapper["parity_against_unmodified_loader_on_the_pinned_revision_required"] is True, "legacy loader parity disabled")
    historical = value["historical_reference"]
    require(historical["final_correct"] == 9969 and historical["best_correct"] == 9973, "historical correct-count drift")
    require(historical["denominator"] == 16632, "historical denominator drift")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--query-receipt", type=Path, required=True)
    parser.add_argument("--loader-parity-receipt", type=Path, required=True)
    parser.add_argument("--snapshot-adapter", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable output exists: {args.output}")
    contract = read_json(args.contract)
    validate_contract(contract)
    current = executing_code_bundle()
    compatibility, accepted_producers = load_authority(args.producer_compatibility, current)
    sources = {}
    for name, expected in contract["code"]["files"].items():
        path = args.source_root / name
        require(path.is_file(), f"legacy evaluator source missing: {path}")
        digest = sha256_file(path)
        require(digest == expected, f"legacy evaluator source drift: {name}")
        sources[name] = file_binding(path)
    query = read_json(args.query_receipt)
    require(query.get("schema_version") == "apertus_frozen_greekmmlu_queries_receipt_v1" and query.get("status") == "passed", "legacy GreekMMLU snapshot receipt drift")
    snapshot = query.get("queries")
    require(isinstance(snapshot, dict) and snapshot.get("rows") == snapshot.get("unique_example_ids") == 16_632, "legacy GreekMMLU snapshot count drift")
    snapshot_path = Path(str(snapshot.get("path", "")))
    require(snapshot_path.is_file() and snapshot == {**file_binding(snapshot_path), "rows": 16_632, "unique_example_ids": 16_632}, "legacy GreekMMLU snapshot binding drift")
    parity = read_json(args.loader_parity_receipt)
    require(parity.get("schema_version") == "apertus_legacy_public_greekmmlu_loader_parity_v1" and parity.get("status") == "passed", "legacy loader-parity receipt drift")
    require(parity.get("query_receipt") == file_binding(args.query_receipt) and parity.get("snapshot") == file_binding(snapshot_path), "legacy loader-parity snapshot binding drift")
    require(isinstance(parity.get("checks"), dict) and all(parity["checks"].values()), "legacy loader-parity checks incomplete")
    for label, value in (("snapshot", query), ("loader parity", parity)):
        require_accepted_producer(value, accepted_producers, f"legacy {label}")
    require(args.snapshot_adapter.is_file(), "legacy snapshot adapter missing")
    payload = {
        "schema_version": "apertus_legacy_public_greekmmlu_receipt_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": current,
        "contract": file_binding(args.contract),
        "source_root": str(args.source_root.resolve()),
        "sources": sources,
        "snapshot_query_receipt": file_binding(args.query_receipt),
        "snapshot": file_binding(snapshot_path),
        "loader_parity_receipt": file_binding(args.loader_parity_receipt),
        "snapshot_adapter": file_binding(args.snapshot_adapter),
        "producer_bundle_compatibility": file_binding(args.producer_compatibility),
        "loader_change_scope": "dataset_loading_only",
        "code_revision": EXPECTED_REVISION,
        "clean_panel_is_scientific_primary": True,
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
