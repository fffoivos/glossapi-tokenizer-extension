#!/usr/bin/env python3
"""Prove both endpoint checkpoints consumed the same frozen Phase-1/2 trajectory."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from freeze_phase_blend_cache import validate_receipt as validate_phase_cache
from producer_bundle_compatibility import load_authority, require_accepted_producer


EXPECTED_CACHE_SAMPLES = {1: 2_315_264, 2: 979_968}


def validate_cache(
    path: Path,
    phase: int,
    *,
    accepted_producers: set[tuple[str, str, str, int, str]],
) -> dict[str, Any]:
    value = read_json(path)
    data_path = Path(str(value.get("data_path_spec", {}).get("path", "")))
    cache_root = Path(str(value.get("cache_root", "")))
    accepted_code_bundles = {
        (root, tree)
        for root, tree, _receipt, _bytes, _sha256 in accepted_producers
    }
    validate_phase_cache(
        value,
        phase=phase,
        data_path_spec=data_path,
        cache_root=cache_root,
        accepted_code_bundles=accepted_code_bundles,
    )
    require_accepted_producer(value, accepted_producers, f"Phase-{phase} cache receipt")
    return value


def bound_json(binding: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    require(isinstance(binding, dict), f"{label} binding missing")
    path = Path(str(binding.get("path", "")))
    require(path.is_file() and binding == file_binding(path), f"{label} binding drift")
    return path, read_json(path)


def validate_cache_lineage(
    path: Path,
    canonical_path: Path,
    *,
    phase: int,
    accepted_producers: set[tuple[str, str, str, int, str]],
) -> dict[str, Any]:
    """Prove a scale-specific cache contains the canonical training cache unchanged.

    The 1.5B runs use qualification-overlay cache roots so validation indices can
    coexist with the canonical training indices.  The training files are seeded
    as hardlinks from the canonical cache; requiring receipt identity here would
    reject that proven superset even though the optimizer sees the same samples.
    """

    canonical = validate_cache(
        canonical_path, phase, accepted_producers=accepted_producers
    )
    observed = validate_cache(path, phase, accepted_producers=accepted_producers)
    if file_binding(path) == file_binding(canonical_path):
        return {
            "cache_receipt": file_binding(path),
            "canonical_cache_receipt": file_binding(canonical_path),
            "lineage_kind": "canonical_identity",
        }

    if (
        observed.get("cache_tree_sha256") == canonical.get("cache_tree_sha256")
        and observed.get("cache_files") == canonical.get("cache_files")
        and observed.get("data_path_spec") == canonical.get("data_path_spec")
        and observed.get("data_path_tokens") == canonical.get("data_path_tokens")
        and Path(str(observed.get("cache_root", ""))).resolve()
        == Path(str(canonical.get("cache_root", ""))).resolve()
    ):
        return {
            "cache_receipt": file_binding(path),
            "canonical_cache_receipt": file_binding(canonical_path),
            "lineage_kind": "canonical_content_identity",
        }

    build_path, build = bound_json(observed.get("blend_manifest"), "overlay cache-build")
    require_accepted_producer(build, accepted_producers, "overlay cache-build")
    require(
        build.get("schema_version") == "apertus_hard_h_to_g_phase_cache_build_v1"
        and build.get("status") == "passed"
        and int(build.get("phase", -1)) == phase
        and Path(str(build.get("cache_root", ""))).resolve()
        == Path(str(observed.get("cache_root", ""))).resolve()
        and build.get("source_cache_receipt") == file_binding(canonical_path),
        f"Phase-{phase} overlay cache-build lineage drift",
    )
    overlay_path, overlay = bound_json(
        build.get("qualification_overlay_receipt"), "qualification overlay"
    )
    require_accepted_producer(overlay, accepted_producers, "qualification overlay")
    materialization = overlay.get("materialization", {})
    source_inventory = materialization.get("source_live_inventory_at_adoption", {})
    require(
        overlay.get("schema_version") == "apertus_hard_h_to_g_phase_cache_overlay_v1"
        and int(overlay.get("phase", -1)) == phase
        and Path(str(overlay.get("overlay_root", ""))).resolve()
        == Path(str(observed.get("cache_root", ""))).resolve()
        and materialization.get("source_cache_receipt") == file_binding(canonical_path)
        and materialization.get("methods")
        == {"byte_copy": 0, "hardlink": len(canonical.get("cache_files", []))}
        and source_inventory.get("missing_relative_paths") == []
        and source_inventory.get("changed_relative_paths") == []
        and source_inventory.get("added_relative_paths") == []
        and int(source_inventory.get("observed_file_count", -1))
        == len(canonical.get("cache_files", []))
        and source_inventory.get("observed_tree_sha256")
        == canonical.get("cache_tree_sha256"),
        f"Phase-{phase} qualification-overlay provenance drift",
    )
    canonical_files = {
        row["relative_path"]: (int(row["bytes"]), str(row["sha256"]))
        for row in canonical.get("cache_files", [])
    }
    observed_files = {
        row["relative_path"]: (int(row["bytes"]), str(row["sha256"]))
        for row in observed.get("cache_files", [])
    }
    seed_files = {
        row["relative_path"]: (int(row["bytes"]), str(row["sha256"]))
        for row in overlay.get("seed_files", [])
    }
    require(
        canonical_files and seed_files == canonical_files
        and all(observed_files.get(name) == identity for name, identity in canonical_files.items()),
        f"Phase-{phase} canonical training-cache files are not an exact overlay subset",
    )
    return {
        "cache_receipt": file_binding(path),
        "canonical_cache_receipt": file_binding(canonical_path),
        "lineage_kind": "hardlinked_canonical_superset",
        "cache_build_receipt": file_binding(build_path),
        "qualification_overlay_receipt": file_binding(overlay_path),
    }


def validate_endpoint_permit(
    path: Path,
    *,
    scale: str,
    canonical_phase1_path: Path,
    canonical_phase2_path: Path,
    accepted_producers: set[tuple[str, str, str, int, str]],
) -> dict[str, Any]:
    permit = read_json(path)
    require(permit.get("schema_version") == "apertus_hard_h_to_g_checkpoint_permit_v2", f"{scale}: checkpoint permit schema drift")
    require(
        permit.get("status") == "passed"
        and permit.get("scale") == scale
        and permit.get("source_phase") == 2
        and permit.get("update") == 3218,
        f"{scale}: endpoint permit identity drift",
    )
    require_accepted_producer(permit, accepted_producers, f"{scale} checkpoint permit")
    phase2_path, _ = bound_json(
        permit.get("source_phase_cache_receipt"), f"{scale} Phase-2 cache"
    )
    phase2_lineage = validate_cache_lineage(
        phase2_path,
        canonical_phase2_path,
        phase=2,
        accepted_producers=accepted_producers,
    )
    audit_binding = permit.get("checkpoint_audit")
    require(isinstance(audit_binding, dict), f"{scale}: checkpoint audit binding missing")
    audit_path = Path(str(audit_binding.get("path", "")))
    require(audit_path.is_file() and audit_binding == file_binding(audit_path), f"{scale}: checkpoint audit binding drift")
    audit = read_json(audit_path)
    require_accepted_producer(audit, accepted_producers, f"{scale} checkpoint audit")
    require(
        audit.get("schema_version") == "apertus_hard_h_to_g_checkpoint_state_audit_v1"
        and audit.get("status") == "passed"
        and audit.get("scale") == scale
        and audit.get("source_phase") == 2
        and audit.get("update") == 3218,
        f"{scale}: checkpoint audit identity drift",
    )
    require(
        audit.get("source_phase_cache_receipt") == file_binding(phase2_path),
        f"{scale}: audit cache binding drift",
    )
    require(
        audit.get("data_cursor") == {
            "global_consumed_samples": 3_295_232,
            "phase_local_consumed_samples": 979_968,
            "phase_start_update": 2261,
        },
        f"{scale}: endpoint data cursor drift",
    )
    preflight_path, preflight = bound_json(
        audit.get("segment_preflight"), f"{scale} Phase-2 segment preflight"
    )
    require_accepted_producer(preflight, accepted_producers, f"{scale} Phase-2 segment preflight")
    require(
        preflight.get("schema_version") == "apertus_hard_h_to_g_train_segment_preflight_v1"
        and preflight.get("status") == "passed"
        and preflight.get("scale") == scale
        and preflight.get("phase") == 2
        and preflight.get("start_update") == 2261
        and preflight.get("exit_update") == 3218
        and preflight.get("phase_cache_receipt") == file_binding(phase2_path),
        f"{scale}: Phase-2 segment preflight drift",
    )
    phase1_path, _ = bound_json(
        preflight.get("source_phase_cache_receipt"), f"{scale} Phase-1 cache"
    )
    phase1_lineage = validate_cache_lineage(
        phase1_path,
        canonical_phase1_path,
        phase=1,
        accepted_producers=accepted_producers,
    )
    phase1_permit_path, phase1_permit = bound_json(
        preflight.get("checkpoint_permit"), f"{scale} update-2261 checkpoint permit"
    )
    require_accepted_producer(
        phase1_permit, accepted_producers, f"{scale} update-2261 checkpoint permit"
    )
    require(
        phase1_permit.get("schema_version") == "apertus_hard_h_to_g_checkpoint_permit_v2"
        and phase1_permit.get("status") == "passed"
        and phase1_permit.get("scale") == scale
        and phase1_permit.get("source_phase") == 1
        and phase1_permit.get("update") == 2261
        and phase1_permit.get("source_phase_cache_receipt") == file_binding(phase1_path),
        f"{scale}: update-2261 checkpoint-permit lineage drift",
    )
    return {
        "endpoint_checkpoint_permit": file_binding(path),
        "endpoint_checkpoint_audit": file_binding(audit_path),
        "phase2_segment_preflight": file_binding(preflight_path),
        "phase1_checkpoint_permit": file_binding(phase1_permit_path),
        "phase1_cache_lineage": phase1_lineage,
        "phase2_cache_lineage": phase2_lineage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-cache-receipt", type=Path, required=True)
    parser.add_argument("--phase2-cache-receipt", type=Path, required=True)
    parser.add_argument("--realized-ledger-receipt", type=Path, required=True)
    parser.add_argument("--8b-checkpoint-permit", type=Path, required=True)
    parser.add_argument("--1p5b-checkpoint-permit", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable cross-scale ledger authority exists: {args.output}")
    current_bundle = executing_code_bundle()
    _, accepted_producers = load_authority(args.producer_compatibility, current_bundle)
    validate_cache(args.phase1_cache_receipt, 1, accepted_producers=accepted_producers)
    validate_cache(args.phase2_cache_receipt, 2, accepted_producers=accepted_producers)
    cache_bindings = {
        1: file_binding(args.phase1_cache_receipt),
        2: file_binding(args.phase2_cache_receipt),
    }
    ledger = read_json(args.realized_ledger_receipt)
    require_accepted_producer(ledger, accepted_producers, "realized document ledger")
    require(ledger.get("schema_version") == "apertus_hard_h_to_g_realized_document_ledger_v1", "realized-ledger schema drift")
    require(ledger.get("status") == "passed", "realized-ledger did not pass")
    rows = ledger.get("cache_receipts")
    require(isinstance(rows, list) and len(rows) == 2, "realized-ledger cache set drift")
    observed = {
        int(row["phase"]): (row.get("receipt"), int(row.get("consumed_samples", -1)))
        for row in rows if isinstance(row, dict)
    }
    require(
        observed == {
            phase: (cache_bindings[phase], EXPECTED_CACHE_SAMPLES[phase])
            for phase in (1, 2)
        },
        "realized-ledger Phase-1/2 cache/sample binding drift",
    )
    trajectory_sha = str(ledger.get("realized_sample_trajectory_sha256", ""))
    require(len(trajectory_sha) == 64, "realized sample trajectory SHA-256 missing")
    scale_lineage = {}
    scale_lineage["8b"] = validate_endpoint_permit(
        args.__dict__["8b_checkpoint_permit"], scale="8b",
        canonical_phase1_path=args.phase1_cache_receipt,
        canonical_phase2_path=args.phase2_cache_receipt,
        accepted_producers=accepted_producers,
    )
    scale_lineage["1p5b"] = validate_endpoint_permit(
        args.__dict__["1p5b_checkpoint_permit"], scale="1p5b",
        canonical_phase1_path=args.phase1_cache_receipt,
        canonical_phase2_path=args.phase2_cache_receipt,
        accepted_producers=accepted_producers,
    )
    payload = {
        "schema_version": "apertus_hard_h_to_g_cross_scale_ledger_match_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": current_bundle,
        "phase_cache_receipts": {str(phase): binding for phase, binding in cache_bindings.items()},
        "realized_ledger_receipt": file_binding(args.realized_ledger_receipt),
        "producer_bundle_compatibility": file_binding(args.producer_compatibility),
        "realized_sample_trajectory_sha256": trajectory_sha,
        "endpoint_checkpoint_permits": {
            "8b": file_binding(args.__dict__["8b_checkpoint_permit"]),
            "1p5b": file_binding(args.__dict__["1p5b_checkpoint_permit"]),
        },
        "scale_cache_lineage": scale_lineage,
        "matched_through_update": 3218,
        "global_consumed_samples": 3_295_232,
        "phase_2_local_consumed_samples": 979_968,
        "invariants": {
            "both_scales_bind_canonical_or_proven_hardlink_superset_phase_caches": True,
            "both_scales_bind_the_same_exact_index_derived_trajectory": True,
            "both_checkpoint_audits_restore_the_same_phase_local_cursor": True,
            "quota_inference_used": False,
        },
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
