#!/usr/bin/env python3
"""Freeze aggregate decontamination and selected-pool receipts for experiment A."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from contract_utils import file_binding, read_json, require, write_json_atomic


def checked_component(name: str, manifest_path: Path, audit_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    require(manifest.get("schema_version") == "targeted_8b_greekmmlu_decontamination_v1", f"{name} decontamination schema drift")
    require(manifest.get("status") == "completed", f"{name} decontamination incomplete")
    require(audit.get("schema_version") == "targeted_8b_decontamination_audit_v1", f"{name} audit schema drift")
    require(audit.get("status") == "passed", f"{name} audit did not pass")
    require(int(audit["counts"]["remaining_high_confidence_matches"]) == 0, f"{name} still matches GreekMMLU")
    require(audit["checks"].get("no_deduplication_performed") is True, f"{name} audit lacks no-dedup proof")
    counts = manifest["counts"]
    # Canonical decontamination manifests are sparse: a zero-drop corpus may
    # omit the ``dropped`` key entirely.  Treat only that omission as zero;
    # input and kept remain mandatory evidence fields.
    require(
        int(counts["input"]) == int(counts["kept"]) + int(counts.get("dropped", 0)),
        f"{name} row accounting drift",
    )
    return manifest, audit


def checked_validation(name: str, manifest_path: Path, audit_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    require(manifest.get("status") == "completed", f"{name} validation exclusion incomplete")
    require(audit.get("schema_version") == "targeted_8b_validation_exclusion_audit_v1", f"{name} validation audit schema drift")
    require(audit.get("status") == "passed", f"{name} validation exclusion audit did not pass")
    checks = audit.get("checks", {})
    require(checks.get("kept_zero_exact_validation_overlap") is True, f"{name} validation content remains")
    require(checks.get("deduplication_performed") is False, f"{name} validation gate changed multiplicity")
    require(checks.get("non_validation_duplicates_preserved") is True, f"{name} validation gate lacks multiplicity proof")
    return manifest, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--academic-pool-receipt", type=Path, required=True)
    parser.add_argument("--poly-pool-receipt", type=Path, required=True)
    parser.add_argument("--hplt-decontamination-manifest", type=Path, required=True)
    parser.add_argument("--hplt-decontamination-audit", type=Path, required=True)
    parser.add_argument("--hplt-validation-manifest", type=Path, required=True)
    parser.add_argument("--hplt-validation-audit", type=Path, required=True)
    parser.add_argument("--pool-corpus-receipt", type=Path, required=True)
    parser.add_argument("--decontamination-output", type=Path, required=True)
    parser.add_argument("--selected-pools-output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.decontamination_output.exists(), f"immutable receipt exists: {args.decontamination_output}")
    require(not args.selected_pools_output.exists(), f"immutable receipt exists: {args.selected_pools_output}")
    academic = read_json(args.academic_pool_receipt)
    poly = read_json(args.poly_pool_receipt)
    pool = read_json(args.pool_corpus_receipt)
    require(academic.get("status") == "passed" and poly.get("status") == "passed", "component pool receipt did not pass")
    require(pool.get("schema_version") == "apertus_schedule_pool_corpus_v1" and pool.get("status") == "completed", "pool corpus is incomplete")

    component_paths = {
        "academic": (
            Path(academic["decontamination_manifest"]["path"]),
            Path(academic["decontamination_audit"]["path"]),
            Path(academic["validation_exclusion_manifest"]["path"]),
            Path(academic["validation_exclusion_audit"]["path"]),
        ),
        "polytonic": (
            Path(poly["decontamination_manifest"]["path"]),
            Path(poly["decontamination_audit"]["path"]),
            Path(poly["validation_exclusion_manifest"]["path"]),
            Path(poly["validation_exclusion_audit"]["path"]),
        ),
        "hplt": (
            args.hplt_decontamination_manifest,
            args.hplt_decontamination_audit,
            args.hplt_validation_manifest,
            args.hplt_validation_audit,
        ),
    }
    total_input = total_kept = total_dropped = 0
    bindings: dict[str, Any] = {}
    for name, (decontam_path, decontam_audit_path, validation_path, validation_audit_path) in component_paths.items():
        manifest, _ = checked_component(name, decontam_path, decontam_audit_path)
        checked_validation(name, validation_path, validation_audit_path)
        total_input += int(manifest["counts"]["input"])
        total_kept += int(manifest["counts"]["kept"])
        total_dropped += int(manifest["counts"].get("dropped", 0))
        bindings[name] = {
            "decontamination_manifest": file_binding(decontam_path),
            "decontamination_audit": file_binding(decontam_audit_path),
            "validation_exclusion_manifest": file_binding(validation_path),
            "validation_exclusion_audit": file_binding(validation_audit_path),
        }
    require(total_input == total_kept + total_dropped, "aggregate decontamination accounting drift")
    decontamination = {
        "schema_version": "targeted_8b_a_decontamination_summary_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "components": bindings,
        "counts": {
            "input": total_input,
            "kept": total_kept,
            "dropped": total_dropped,
            "remaining_high_confidence_matches": 0,
        },
        "no_global_deduplication": True,
        "independent_postscan_passed_for_every_component": True,
    }
    academic_tokens = int(academic["counts"]["final_training_tokens"])
    poly_tokens = int(poly["counts"]["final_training_tokens"])
    modern = pool["modern_greek"]
    require(int(modern["hplt_tokens"]) == academic_tokens, "HPLT active target is not equal to academic")
    require(int(modern["glossapi_non_hplt_tokens"]) == academic_tokens + poly_tokens, "non-HPLT target drift")
    invariants = pool.get("invariants", {})
    require(invariants.get("new_modern_rows_deduplicated") is False, "pool corpus reports a second deduplication")
    require(invariants.get("new_modern_row_multiplicity_preserved") is True, "pool corpus lacks multiplicity proof")
    wrote_decontamination = False
    try:
        write_json_atomic(args.decontamination_output, decontamination)
        wrote_decontamination = True
        selected = {
            "schema_version": "targeted_8b_a_selected_pools_v1",
            "status": "passed",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "decontamination_summary": file_binding(args.decontamination_output),
            "academic_pool_receipt": file_binding(args.academic_pool_receipt),
            "poly_pool_receipt": file_binding(args.poly_pool_receipt),
            "pool_corpus_receipt": file_binding(args.pool_corpus_receipt),
            "pool_active_tokens": {
                "academic": academic_tokens,
                "hplt": academic_tokens,
                "polytonic": poly_tokens,
            },
            "no_global_deduplication": True,
            "greekmmlu_postscan_high_confidence_matches": 0,
            "frozen_validation_exact_content_overlaps": 0,
            "row_multiplicity_otherwise_preserved": True,
        }
        write_json_atomic(args.selected_pools_output, selected)
    except BaseException:
        if wrote_decontamination:
            args.decontamination_output.unlink(missing_ok=True)
        raise
    print(json.dumps({"ok": True, "pool_active_tokens": selected["pool_active_tokens"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
