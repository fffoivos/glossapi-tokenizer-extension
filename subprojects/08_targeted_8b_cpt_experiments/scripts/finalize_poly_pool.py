#!/usr/bin/env python3
"""Freeze release-internal polytonic source datasets after required exclusions."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import file_binding, read_json, require, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-source-audit", type=Path, required=True)
    parser.add_argument("--source-extraction-manifest", type=Path, required=True)
    parser.add_argument("--source-token-receipt", type=Path, required=True)
    parser.add_argument("--decontamination-manifest", type=Path, required=True)
    parser.add_argument("--decontamination-audit", type=Path, required=True)
    parser.add_argument("--decontamination-dropped-tokens", type=Path, required=True)
    parser.add_argument("--validation-exclusion-manifest", type=Path, required=True)
    parser.add_argument("--validation-exclusion-audit", type=Path, required=True)
    parser.add_argument("--validation-excluded-tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable poly pool receipt exists: {args.output}")
    source_audit = read_json(args.release_source_audit)
    extraction = read_json(args.source_extraction_manifest)
    source = read_json(args.source_token_receipt)
    decontam = read_json(args.decontamination_manifest)
    decontam_audit = read_json(args.decontamination_audit)
    dropped = read_json(args.decontamination_dropped_tokens)
    exclusion = read_json(args.validation_exclusion_manifest)
    exclusion_audit = read_json(args.validation_exclusion_audit)
    excluded = read_json(args.validation_excluded_tokens)
    require(
        source_audit.get("schema_version") == "targeted_8b_release_polytonic_source_audit_v1"
        and source_audit.get("status") == "passed"
        and source_audit.get("selection_authority") == "pinned_hf_release_only",
        "release polytonic source audit is not passed",
    )
    require(
        extraction.get("schema_version") == "targeted_8b_source_extraction_v1"
        and extraction.get("status") == "passed"
        and extraction.get("sources") == source_audit.get("source_datasets"),
        "release polytonic extraction/source-audit binding drift",
    )
    require(
        Path(extraction.get("release_root", "")).resolve()
        == Path(source_audit.get("release_root", "")).resolve(),
        "release polytonic extraction release-root drift",
    )
    require(
        extraction.get("upstream", {}).get("anonymization_manifest")
        == source_audit.get("anonymization_manifest")
        and extraction.get("upstream", {}).get("token_counts")
        == source_audit.get("token_counts_manifest"),
        "release polytonic extraction upstream-manifest drift",
    )
    require(
        source.get("schema_version") == "targeted_8b_release_polytonic_token_receipt_v1"
        and source.get("status") == "passed"
        and source.get("selection_authority") == "pinned_hf_release_only"
        and source.get("source_datasets") == source_audit.get("source_datasets"),
        "release polytonic source token receipt is not passed",
    )
    require(
        source.get("release_source_selection_audit", {}) == file_binding(args.release_source_audit),
        "poly token count/source-audit binding drift",
    )
    extraction_data = (args.source_extraction_manifest.parent / "data").resolve()
    require(
        Path(source.get("input", "")).resolve() == extraction_data,
        "poly token receipt did not count the release extraction",
    )
    extracted_files = sorted(
        (row["output"] for row in extraction.get("outputs", [])),
        key=lambda value: value["path"],
    )
    counted_files = sorted(source.get("input_files", []), key=lambda value: value["path"])
    require(
        extracted_files and counted_files == extracted_files,
        "poly token receipt files do not byte-match release extraction outputs",
    )
    require(decontam.get("status") == "completed" and decontam_audit.get("status") == "passed", "poly decontamination is not audited")
    require(exclusion.get("status") == "completed" and exclusion_audit.get("status") == "passed", "poly validation exclusion is not audited")
    pre_rows = int(source["rows"])
    pre_training = int(source["training_tokens"])
    decontam_dropped_rows = int(decontam["counts"].get("dropped", 0))
    validation_excluded_rows = int(exclusion["counts"].get("excluded", 0))
    require(pre_rows == int(extraction["rows"]), "poly source token/extraction row count drift")
    require(
        Path(decontam.get("input", "")).resolve() == args.source_extraction_manifest.parent.resolve(),
        "poly decontamination input is not the release extraction root",
    )
    require(pre_rows == int(decontam["counts"]["input"]), "poly input row drift")
    require(int(dropped["rows"]) == decontam_dropped_rows, "poly decontamination token receipt row drift")
    require(int(exclusion["counts"]["input"]) == int(decontam["counts"]["kept"]), "poly validation input drift")
    require(int(excluded["rows"]) == validation_excluded_rows, "poly validation token receipt row drift")
    final_rows = int(exclusion["counts"]["kept"])
    final_training = pre_training - int(dropped["training_tokens"]) - int(excluded["training_tokens"])
    require(final_rows == pre_rows - decontam_dropped_rows - validation_excluded_rows, "final poly rows do not reconcile")
    require(final_training > 0, "final poly token total is not positive")
    payload = {
        "schema_version": "targeted_8b_release_polytonic_pool_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "release_source_audit": file_binding(args.release_source_audit),
        "source_extraction_manifest": file_binding(args.source_extraction_manifest),
        "source_token_receipt": file_binding(args.source_token_receipt),
        "decontamination_manifest": file_binding(args.decontamination_manifest),
        "decontamination_audit": file_binding(args.decontamination_audit),
        "decontamination_dropped_tokens": file_binding(args.decontamination_dropped_tokens),
        "validation_exclusion_manifest": file_binding(args.validation_exclusion_manifest),
        "validation_exclusion_audit": file_binding(args.validation_exclusion_audit),
        "validation_excluded_tokens": file_binding(args.validation_excluded_tokens),
        "counts": {
            "pre_rows": pre_rows,
            "pre_training_tokens": pre_training,
            "greekmmlu_removed_rows": decontam_dropped_rows,
            "greekmmlu_removed_training_tokens": int(dropped["training_tokens"]),
            "validation_excluded_rows": validation_excluded_rows,
            "validation_excluded_training_tokens": int(excluded["training_tokens"]),
            "final_rows": final_rows,
            "final_training_tokens": final_training,
        },
        "training_data": str(Path(exclusion["output"]).resolve()),
        "invariants": {
            "all_release_selected_polytonic_source_rows_consumed_once": True,
            "pinned_hf_release_only": True,
            "only_greekmmlu_and_frozen_validation_content_removed": True,
            "no_deduplication_performed": True,
            "row_multiplicity_otherwise_preserved": True,
            "exact_production_tokenizer_arithmetic": True,
        },
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
