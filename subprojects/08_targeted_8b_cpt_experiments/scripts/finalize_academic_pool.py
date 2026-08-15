#!/usr/bin/env python3
"""Freeze exact post-GreekMMLU, post-heldout academic pool totals."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import file_binding, read_json, require, write_json_atomic


SOURCES = ("openarchives.gr", "greek_phd")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-token-counts", type=Path, required=True)
    parser.add_argument("--decontamination-manifest", type=Path, required=True)
    parser.add_argument("--decontamination-audit", type=Path, required=True)
    parser.add_argument("--decontamination-dropped-tokens", type=Path, required=True)
    parser.add_argument("--validation-exclusion-manifest", type=Path, required=True)
    parser.add_argument("--validation-exclusion-audit", type=Path, required=True)
    parser.add_argument("--validation-excluded-tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable academic receipt exists: {args.output}")
    release = read_json(args.release_token_counts)
    decontam = read_json(args.decontamination_manifest)
    decontam_audit = read_json(args.decontamination_audit)
    decontam_tokens = read_json(args.decontamination_dropped_tokens)
    exclusion = read_json(args.validation_exclusion_manifest)
    exclusion_audit = read_json(args.validation_exclusion_audit)
    exclusion_tokens = read_json(args.validation_excluded_tokens)
    require(release.get("status") == "passed", "release token counts are not passed")
    require(decontam.get("status") == "completed" and decontam_audit.get("status") == "passed", "decontamination is not audited")
    require(exclusion.get("status") == "completed" and exclusion_audit.get("status") == "passed", "validation exclusion is not audited")
    pre_rows = sum(int(release["source_rows"][source]) for source in SOURCES)
    pre_text = sum(int(release["source_text_tokens"][source]) for source in SOURCES)
    pre_training = pre_text + pre_rows
    require(pre_rows == int(decontam["counts"]["input"]), "academic input row drift")
    require(
        int(decontam_tokens["rows"]) == int(decontam["counts"].get("dropped", 0)),
        "decontamination dropped-row token receipt drift",
    )
    require(int(exclusion["counts"]["input"]) == int(decontam["counts"]["kept"]), "validation exclusion input drift")
    require(int(exclusion_tokens["rows"]) == int(exclusion["counts"]["excluded"]), "validation excluded-row token receipt drift")
    final_rows = int(exclusion["counts"]["kept"])
    removed_training = int(decontam_tokens["training_tokens"]) + int(exclusion_tokens["training_tokens"])
    final_training = pre_training - removed_training
    require(final_rows == pre_rows - int(decontam_tokens["rows"]) - int(exclusion_tokens["rows"]), "final academic rows do not reconcile")
    require(final_training > 0, "final academic token total is not positive")
    payload = {
        "schema_version": "targeted_8b_academic_pool_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": list(SOURCES),
        "release_token_counts": file_binding(args.release_token_counts),
        "decontamination_manifest": file_binding(args.decontamination_manifest),
        "decontamination_audit": file_binding(args.decontamination_audit),
        "decontamination_dropped_tokens": file_binding(args.decontamination_dropped_tokens),
        "validation_exclusion_manifest": file_binding(args.validation_exclusion_manifest),
        "validation_exclusion_audit": file_binding(args.validation_exclusion_audit),
        "validation_excluded_tokens": file_binding(args.validation_excluded_tokens),
        "counts": {
            "pre_rows": pre_rows,
            "pre_text_tokens": pre_text,
            "pre_training_tokens": pre_training,
            "greekmmlu_removed_rows": int(decontam_tokens["rows"]),
            "greekmmlu_removed_training_tokens": int(decontam_tokens["training_tokens"]),
            "validation_excluded_rows": int(exclusion_tokens["rows"]),
            "validation_excluded_training_tokens": int(exclusion_tokens["training_tokens"]),
            "final_rows": final_rows,
            "final_training_tokens": final_training,
        },
        "training_data": str(Path(exclusion["output"]).resolve()),
        "invariants": {
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
