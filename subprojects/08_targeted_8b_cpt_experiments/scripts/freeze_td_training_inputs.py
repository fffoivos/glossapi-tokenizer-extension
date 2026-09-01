#!/usr/bin/env python3
"""Freeze the historical TD token order against regenerated clean snippets."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterator

from contract_utils import (
    executing_code_bundle,
    file_binding,
    require,
    require_file_binding,
    require_receipt,
    write_json_atomic,
)


def rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_no}: expected object")
            yield value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--coverage-jsonl", type=Path, required=True)
    parser.add_argument("--coverage-summary", type=Path, required=True)
    parser.add_argument("--snippets-jsonl", type=Path, required=True)
    parser.add_argument("--coverage-seed", type=int, required=True)
    parser.add_argument("--dataset-authority", type=Path, required=True)
    parser.add_argument("--hplt-stage-b-receipt", type=Path, required=True)
    parser.add_argument("--openarchives-stage-b-receipt", type=Path, required=True)
    parser.add_argument("--output-token-ids", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not args.output_token_ids.exists() and not args.output_receipt.exists(), "immutable TD input output exists")
    manifest = json.loads(args.historical_manifest.read_text(encoding="utf-8"))
    selected = manifest.get("selected_token_ids")
    require(isinstance(selected, list) and all(isinstance(value, int) for value in selected), "historical selected-token ledger missing")
    require(len(selected) == 17_392 and len(set(selected)) == len(selected), "historical selected-token ledger drift")
    require(all(131_072 <= value < 148_480 for value in selected), "selected token outside modern extension")
    require(manifest.get("target_layer") == 11 and manifest.get("trained_token_count") == 17_377, "historical TD manifest drift")
    summary = json.loads(args.coverage_summary.read_text(encoding="utf-8"))
    require(summary.get("new_id_start") == 131_072 and summary.get("new_id_end") == 148_480, "coverage vocabulary range drift")
    require(summary.get("target_extended_tokens") == 2_000_000_000, "coverage token budget drift")
    require(summary.get("snippets_per_token") == 100 and summary.get("snippet_token_radius") == 50, "coverage snippet geometry drift")
    require(summary.get("require_nfc") is False, "TD coverage must preserve the exact Stage-B Unicode bytes")
    non_nfc_docs = int(summary.get("non_nfc_docs", -1))
    docs_seen = int(summary.get("docs_seen", 0))
    require(non_nfc_docs >= 0 and docs_seen > 0 and non_nfc_docs <= docs_seen, "TD coverage NFC audit drift")
    require(args.coverage_seed == 20260523, "coverage seed drift")
    execution = summary.get("encoding_execution")
    require(
        isinstance(execution, dict)
        and execution.get("mode") == "ordered_multiprocess_encode_batches_with_sequential_parity_guard"
        and int(execution.get("workers", -1)) == 16
        and int(execution.get("max_batches_in_flight", -1)) == 32
        and int(execution.get("parity_documents", -1)) == 256
        and execution.get("new_token_position_filter")
        == "numpy_flatnonzero_contiguous_id_range_preserving_ascending_order"
        and execution.get("snippet_materialization")
        == "deferred_until_exact_pinned_reservoir_slot_is_selected"
        and execution.get("scientific_state_update_order") == "identical_to_pinned_sequential_reference",
        "coverage batched-encoding parity authority drift",
    )
    dataset_authority = require_receipt(
        args.dataset_authority,
        schemas={"apertus_hard_h_to_g_dataset_authority_v1"},
    )
    stage_receipts = {}
    expected_inputs = []
    for stream, receipt_path in (
        ("hplt", args.hplt_stage_b_receipt),
        ("openarchives", args.openarchives_stage_b_receipt),
    ):
        require_file_binding(dataset_authority["stage_b"][stream], expected_path=receipt_path)
        stage = require_receipt(receipt_path, schemas={"apertus_hard_h_to_g_stage_b_stream_v1"})
        require(stage.get("stream") == stream, f"Stage-B stream drift: {stream}")
        output = stage.get("output")
        require(isinstance(output, dict), f"Stage-B output binding missing: {stream}")
        input_path = require_file_binding(output, verify_sha256=False)
        expected_inputs.append(str(input_path))
        stage_receipts[stream] = file_binding(receipt_path)
    require(summary.get("inputs") == expected_inputs, "TD coverage input order or identity drift")

    coverage: dict[int, dict] = {}
    for row in rows(args.coverage_jsonl):
        token_id = row.get("new_token_id")
        require(isinstance(token_id, int) and 131_072 <= token_id < 148_480, "coverage token-id drift")
        require(token_id not in coverage, f"duplicate coverage token {token_id}")
        coverage[token_id] = row
    missing = sorted(set(selected) - set(coverage))
    require(not missing, f"selected tokens missing coverage rows: {missing[:10]}")

    snippet_counts = {token_id: 0 for token_id in selected}
    snippet_rows = 0
    for row in rows(args.snippets_jsonl):
        token_id = row.get("new_token_id")
        if token_id in snippet_counts:
            snippet_counts[token_id] += 1
        snippet_rows += 1
    at_least_25 = sum(count >= 25 for count in snippet_counts.values())
    require(at_least_25 / len(selected) >= 0.99, "regenerated snippets cannot meet the predeclared 99% TD coverage floor")

    args.output_token_ids.parent.mkdir(parents=True, exist_ok=True)
    args.output_token_ids.write_text("".join(f"{token_id}\n" for token_id in selected), encoding="utf-8")
    receipt = {
        "schema_version": "apertus_td_training_inputs_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "named_reconstruction_differences": [
            "regenerated TD snippet text from benchmark-clean Modern-Greek stream",
            "exact Stage-B Unicode bytes retained for TD coverage instead of adding a non-historical NFC training transform",
        ],
        "historical_target_layer": 11,
        "target_1p5b_layer": 6,
        "target_1p5b_hidden_state_index": 6,
        "selected_token_count": len(selected),
        "selected_token_order_preserved": True,
        "coverage_row_count": len(coverage),
        "snippet_row_count": snippet_rows,
        "tokens_with_at_least_25_candidate_snippets": at_least_25,
        "candidate_coverage_fraction": at_least_25 / len(selected),
        "historical_manifest": file_binding(args.historical_manifest),
        "coverage_jsonl": file_binding(args.coverage_jsonl),
        "coverage_summary": file_binding(args.coverage_summary),
        "snippets_jsonl": file_binding(args.snippets_jsonl),
        "coverage_command_contract": {
            "target_extended_tokens": 2_000_000_000,
            "candidate_snippets_per_token": 100,
            "snippet_token_radius": 50,
            "seed": args.coverage_seed,
            "require_nfc": False,
        },
        "unicode_normalization_audit": {
            "input_bytes_transformed": False,
            "normalization_form_required": None,
            "documents_seen": docs_seen,
            "non_nfc_documents": non_nfc_docs,
            "non_nfc_document_fraction": non_nfc_docs / docs_seen,
            "policy": "audit_and_preserve_exact_stage_b_text",
        },
        "coverage_encoding_execution": execution,
        "dataset_authority": file_binding(args.dataset_authority),
        "stage_b_receipts": stage_receipts,
        "ordered_training_streams": expected_inputs,
        "token_ids": file_binding(args.output_token_ids),
        "executing_code_bundle": executing_code_bundle(),
    }
    write_json_atomic(args.output_receipt, receipt)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
