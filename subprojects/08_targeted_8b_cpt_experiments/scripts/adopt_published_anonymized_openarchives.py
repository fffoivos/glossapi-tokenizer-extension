#!/usr/bin/env python3
"""Bind a Phase-3 stream to its already-published v2 anonymization receipt.

This adapter performs no text processing.  The selected OpenArchives stream
already descends from the published v2 release; the adapter verifies that
receipt chain and makes it explicit for the subsequent catalog builder.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    require_file_binding,
    require_receipt,
    write_json_atomic,
)


def receipt_binding_matches(path: Path, binding: object, label: str) -> None:
    require(isinstance(binding, dict), f"{label} binding missing")
    require_file_binding(binding, expected_path=path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--prepared-receipt", type=Path, required=True)
    parser.add_argument("--source-view-receipt", type=Path, required=True)
    parser.add_argument("--anonymization-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()

    require(not args.output_receipt.exists(), f"immutable output exists: {args.output_receipt}")
    prepared = require_receipt(
        args.prepared_receipt,
        schemas={"apertus_hard_h_to_g_prepared_greek_stream_v1"},
        statuses={"passed"},
    )
    require(prepared.get("pool") == "openarchives", "prepared stream pool drift")
    clean = prepared.get("clean")
    require(isinstance(clean, dict), "prepared stream clean binding missing")
    require_file_binding(clean, expected_path=args.input_jsonl, verify_sha256=False)

    raw_receipt_path = Path(str(prepared.get("selected_mix_receipt", {}).get("path", "")))
    receipt_binding_matches(raw_receipt_path, prepared.get("selected_mix_receipt"), "prepared raw receipt")
    raw = require_receipt(
        raw_receipt_path,
        schemas={"apertus_hard_h_to_g_openarchives_candidate_source_v1"},
        statuses={"passed"},
    )
    source_view_binding = raw.get("source_view_receipt")
    receipt_binding_matches(args.source_view_receipt, source_view_binding, "raw source-view receipt")
    source_view = require_receipt(
        args.source_view_receipt,
        schemas={"apertus_hard_h_to_g_source_views_v1"},
        statuses={"passed"},
    )
    release = source_view.get("release")
    require(isinstance(release, dict), "source view release binding missing")
    receipt_binding_matches(args.anonymization_manifest, release.get("anonymization_manifest"), "published anonymization manifest")
    anonymization = read_json(args.anonymization_manifest)
    require(anonymization.get("status") == "passed", "published anonymization manifest did not pass")
    require(int(anonymization.get("counts", {}).get("rows", 0)) > 0, "published anonymization manifest row count missing")

    payload = {
        "schema_version": "apertus_hard_h_to_g_published_anonymized_stream_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stream": "phase3_openarchives_candidates",
        "mode": "published_v2_authority_no_reaudit",
        "executing_code_bundle": executing_code_bundle(),
        "output": {**clean, "path": str(args.input_jsonl.resolve())},
        "prepared_receipt": file_binding(args.prepared_receipt),
        "raw_candidate_receipt": file_binding(raw_receipt_path),
        "source_view_receipt": file_binding(args.source_view_receipt),
        "published_anonymization_manifest": file_binding(args.anonymization_manifest),
        "invariants": {
            "published_anonymization_receipt_accepted": True,
            "additional_text_scan_performed": False,
            "text_transformed": False,
            "additional_deduplication": False,
            "row_order_preserved": True,
            "row_multiplicity_preserved": True,
        },
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
