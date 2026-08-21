#!/usr/bin/env python3
"""Freeze a role-agnostic authority for the narrowly audited producer bundles."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    sha256_file,
    write_json_atomic,
)
from producer_bundle_compatibility import SCHEMA, validate_authority

# These are the only paths that changed between the already-running v14/v15
# producers and the replacement bundle. None changes query construction,
# source-view extraction, validation-panel selection, decontamination,
# anonymization, tokenization, packing, model geometry, or optimization.
ALLOWED_CHANGED_PATHS = frozenset(
    {
        # Operational bundle tooling only: absolute-path workers run with
        # PYTHONSAFEPATH=1, so the verifier and freezer are intentionally
        # self-contained.  No scientific input changes.
        "subprojects/06_dataset_scheduling_experiments/production/freeze_code_bundle.py",
        "subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py",
        # The deployment patch changes only the historical launch wrapper's
        # uenv mounting syntax (uenv-run-around-srun to the CSCS-supported
        # srun --uenv/--view form).  Keep this exact generated path auditable;
        # never broaden it to frozen_training_tools/**.
        "frozen_training_tools/bakeoff_training/bakeoff_train.sbatch",
        "frozen_td_tools/_common.py",
        # The v52+ bundle carries the already-approved historical 8B LR
        # decision at the exact path consumed by freeze_lr_selection.py.  This
        # is evidence packaging only; it does not change optimizer behavior.
        "subprojects/05_token_distillation_cpt/PRODUCTION_LR_DECISION_20260613.md",
        "subprojects/08_targeted_8b_cpt_experiments/ULTRACODE_R2_REMEDIATION_20260814.md",
        "subprojects/08_targeted_8b_cpt_experiments/HARD_H_TO_G_8B_1P5B_REPLICATION_PLAN_20260814.md",
        "subprojects/08_targeted_8b_cpt_experiments/README.md",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/build_1p5b_td_init_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/build_1p5b_td_init_normal.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/build_modern_mix_selection_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/build_phase_gptdataset_cache_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/materialize_phase_cache_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/build_td_snippets_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/build_td_snippets_xfer.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/build_td_xfer_runtime.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/adopt_canonical_pre_main_data_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/diagnose_1p5b_td_row_norms_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/deploy_targeted_bundle.sh",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_hard_h_to_g_evaluation_contracts_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_owner_authorization_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_pre_main_artifact_manifest_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_pre_main_launch_gate_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_prelaunch_benchmark_contract_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_modern_mix_recipes_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_pre_main_data_authorities_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/export_realized_document_ledger_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_producer_bundle_compatibility_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/prepare_init_geometry_debug.sbatch",
        # Debug-only release orchestration for the already-frozen 1.5B TD
        # initialization. It merely chains corrected HF geometry and the exact
        # HF -> Megatron -> HF round-trip verifier; it does not produce or
        # rewrite any training-data payload.
        "subprojects/08_targeted_8b_cpt_experiments/clariden/prepare_1p5b_init_release_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/prepare_training_megatron_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/prove_uenv10_srun_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/prove_nested_same_uenv_srun_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/roundtrip_td_init_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/run_1p5b_td_init_common.sh",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/run_native_suite_replay_scan_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/run_phase3_resume_smoke.sbatch",
        # Issue #128: the exact-profile benchmark still runs the unchanged
        # multi-node torchrun workload.  These two experiment-owned paths only
        # classify the known post-checkpoint elastic-rendezvous teardown after
        # proving the expected finite row and complete DCP metadata; restart
        # parity remains mandatory and every other launcher failure is fatal.
        "subprojects/08_targeted_8b_cpt_experiments/clariden/run_prelaunch_benchmark.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/train_hard_h_to_g_segment.sbatch",
        # These wrappers now consume the exact compiled Megatron receipt
        # instead of silently selecting the retired pre-helper receipt.  The
        # tokenized payloads are unchanged; this closes future load/config
        # drift in preparation and Phase-3 continuation paths.
        "subprojects/08_targeted_8b_cpt_experiments/clariden/tokenize_h2g_stream_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/tokenize_phase3_stream_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/configs/1p5b_tokenizer_compatibility_v1.json",
        "subprojects/08_targeted_8b_cpt_experiments/configs/hard_h_to_g_allocation_v1.json",
        # The resized profiles remain within the fixed global batch and use
        # first-allocation qualification. These are declaration/evidence
        # assets only; they do not change prepared data or the recipe.
        "subprojects/08_targeted_8b_cpt_experiments/configs/hard_h_to_g_allocation_v3_minimum_defensible.json",
        "subprojects/08_targeted_8b_cpt_experiments/ALLOCATION_GEOMETRY_DECISION_20260818.md",
        "subprojects/08_targeted_8b_cpt_experiments/runtime_compat/sitecustomize.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/adopt_canonical_pre_main_data.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/audit_training_checkpoint.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/build_checkpoint_permit.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/build_training_run_permit.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/check_dcp_metadata_compat.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/diagnose_1p5b_td_row_norms.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_artifact_manifest.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_hard_h_to_g_contract.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_owner_authorization.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_legacy_public_evaluator.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_online_validation_binaries.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/build_phase_gptdataset_cache.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/build_retok_reference_init.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/contract_utils.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/finalize_init_roundtrip.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/finalize_lr_pilot_arm.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/finalize_profile_benchmark.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/finalize_training_megatron.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_phase_blend_cache.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_lr_selection.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_prelaunch_benchmark_contract.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/materialize_historical_tokenizer.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/materialize_phase_cache.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_pre_main_data_authorities.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_producer_bundle_compatibility.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_replay_source_inventory.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_td_training_inputs.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/producer_bundle_compatibility.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/preflight_train_segment.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/prepare_training_geometry_hf.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/patch_bakeoff_uenv10_srun.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/patch_bakeoff_runtime_compat.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/run_canonical_train_segment.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/run_in_allocation_profile_qualification.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/workaround_parameterized_profile_qualification.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/workaround_accept_intentional_torchrun_teardown.py",
        # Contract compilation is a control-plane adaptation only. It reads
        # existing immutable continuation manifests and passes their frozen
        # segment identities to the canonical runner; it is not a producer of
        # data, validation, tokenization, or initialization artifacts.
        "subprojects/08_targeted_8b_cpt_experiments/scripts/workaround_rebind_resized_continuation_contract.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/td_coverage_prepass_batched.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/verify_td_initialization.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/verify_td_xfer_runtime.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/verify_prelaunch_benchmark_contract.py",
        "subprojects/08_targeted_8b_cpt_experiments/configs/td_xfer_runtime_requirements_v1.txt",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/freeze_greekmmlu_examples.py",
        "subprojects/08_targeted_8b_cpt_experiments/tests/test_hard_h_to_g_contracts.py",
        "subprojects/08_targeted_8b_cpt_experiments/tests/test_phase_cache_isolation.py",
        "subprojects/08_targeted_8b_cpt_experiments/tests/test_canonical_train_adapter.py",
        "subprojects/08_targeted_8b_cpt_experiments/tests/test_r2_orchestration.py",
        # Cross-scale evaluation, canonical-campaign and authorization closure
        # added after the last accepted data producer. These paths do not
        # participate in the already-frozen source selection, Stage-B,
        # tokenization or Phase-1/2 cache payloads. They are predeclared here
        # so those exact payload receipts can be adopted without pretending
        # the newer evaluation/control bundle produced them.
        "subprojects/08_targeted_8b_cpt_experiments/1P5B_TD_ROW_NORM_DIAGNOSTIC_20260815.md",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/build_canonical_campaign_contracts_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/export_checkpoint_for_evaluation_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/finalize_canonical_runtime_qualification_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/finalize_lr_pilot_arm_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/finalize_matched_study_evidence_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/finalize_profile_benchmark_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_1p5b_td_policy_authorization_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_cross_scale_realized_ledger_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_cross_scale_sentinel_authority_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_extension_artifact_manifest_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_extension_launch_gate_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_phase3_authority_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_post_checkpoint_authorities_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_lr_selection_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_profile_promotion_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_production_timing_and_allocation_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_training_run_permit_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/run_frozen_greekmmlu_4node_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/run_greekmmlu_shard.sh",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/run_legacy_public_greekmmlu_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/run_native_suite_checkpoint_4node_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/run_offline_panels_4node_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/run_per_document_panel_group.sh",
        "subprojects/08_targeted_8b_cpt_experiments/clariden/validate_and_inspect_debug.sbatch",
        "subprojects/08_targeted_8b_cpt_experiments/configs/1p5b_td_acceptance_policy_v2.json",
        "subprojects/08_targeted_8b_cpt_experiments/configs/hard_h_to_g_replication_v1.json",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/aggregate_frozen_greekmmlu.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/bind_native_suite_checkpoint.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/canonical_evidence.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/checkpoint_export_contract.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/checkpoint_export_receipt.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/finalize_legacy_public_greekmmlu.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/finalize_matched_study_evidence.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/run_checkpoint_export_evaluator.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/run_greekmmlu_calibration_evaluator.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/run_greekmmlu_evaluator.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/run_greekmmlu_fallback_evaluator.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/run_greekmmlu_plateau_evaluator.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/run_legacy_public_greekmmlu_evaluator.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/run_native_suite_evaluator.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/run_offline_panels_evaluator.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/score_frozen_greekmmlu_shard.py",
        "subprojects/08_targeted_8b_cpt_experiments/evaluation/verify_exact_checkpoint_weight_mapping.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/build_canonical_campaign_contracts.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/build_canonical_qualification_context.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/build_existing_8b_qualification_context.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/evaluate_td_objective.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_1p5b_td_policy_authorization.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_cross_scale_sentinel_authority.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_production_timing_and_allocation.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_statistical_decision_contract.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_submission_dry_run.py",
        "subprojects/08_targeted_8b_cpt_experiments/scripts/promote_canonical_runtime.py",
        "subprojects/08_targeted_8b_cpt_experiments/tests/test_canonical_campaign_contracts.py",
        "subprojects/08_targeted_8b_cpt_experiments/tests/test_canonical_qualification_evidence.py",
        "subprojects/08_targeted_8b_cpt_experiments/tests/test_cross_scale_sentinel_authority.py",
        "subprojects/08_targeted_8b_cpt_experiments/tests/test_greekmmlu_evaluation_gates.py",
        "subprojects/08_targeted_8b_cpt_experiments/tests/test_matched_study_statistics.py",
    }
)


def verified_bundle(path: Path) -> tuple[dict[str, Any], dict[str, tuple[int, str]]]:
    value = read_json(path)
    require(
        value.get("schema_version") == "apertus_mini_immutable_code_bundle_v1"
        and value.get("status") == "frozen"
        and value.get("kind") == "scientific",
        f"invalid producer bundle receipt: {path}",
    )
    root = Path(str(value.get("root", ""))).resolve()
    rows = value.get("files")
    require(root.is_dir() and isinstance(rows, list) and rows, f"producer bundle root/files missing: {path}")
    require(int(value.get("file_count", -1)) == len(rows), f"producer bundle file count drift: {path}")
    inventory: dict[str, tuple[int, str]] = {}
    for row in rows:
        relative = str(row.get("relative_path", ""))
        candidate = (root / relative).resolve()
        require(
            relative
            and relative not in inventory
            and (candidate.parent == root or root in candidate.parents)
            and candidate.is_file()
            and not candidate.is_symlink()
            and candidate.stat().st_size == int(row.get("bytes", -1))
            and sha256_file(candidate) == row.get("sha256"),
            f"producer bundle file drift: {relative}",
        )
        inventory[relative] = (int(row["bytes"]), str(row["sha256"]))
    canonical = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
    require(hashlib.sha256(canonical).hexdigest() == value.get("tree_sha256"), f"producer bundle tree drift: {path}")
    return value, inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-receipt", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable producer compatibility authority exists: {args.output}")
    current = executing_code_bundle()
    current_receipt = Path(str(current["receipt"]["path"]))
    receipt_paths = [path.resolve() for path in args.producer_receipt]
    require(current_receipt.resolve() in receipt_paths, "producer list omits the executing bundle")
    require(len(receipt_paths) == len(set(receipt_paths)), "duplicate producer receipt path")
    bundles = [(*verified_bundle(path), path) for path in receipt_paths]
    current_index = receipt_paths.index(current_receipt.resolve())
    current_value, current_inventory, _ = bundles[current_index]
    producers = []
    observed_changed_paths: set[str] = set()
    for value, inventory, path in bundles:
        changed = sorted(
            relative
            for relative in set(current_inventory) | set(inventory)
            if current_inventory.get(relative) != inventory.get(relative)
        )
        unexpected = set(changed) - ALLOWED_CHANGED_PATHS
        require(not unexpected, f"producer bundle has unaudited changed paths: {sorted(unexpected)}")
        observed_changed_paths.update(changed)
        rows = []
        for relative in changed:
            rows.append(
                {
                    "relative_path": relative,
                    "producer": (
                        {"bytes": inventory[relative][0], "sha256": inventory[relative][1]}
                        if relative in inventory
                        else None
                    ),
                    "current": (
                        {"bytes": current_inventory[relative][0], "sha256": current_inventory[relative][1]}
                        if relative in current_inventory
                        else None
                    ),
                }
            )
        producers.append(
            {
                "bundle": {
                    "root": str(Path(str(value["root"])).resolve()),
                    "tree_sha256": str(value["tree_sha256"]),
                    "receipt": file_binding(path),
                },
                "changed_files_against_current": rows,
            }
        )
    payload = {
        "schema_version": SCHEMA,
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": current,
        "current_bundle": {
            "root": str(Path(str(current_value["root"])).resolve()),
            "tree_sha256": str(current_value["tree_sha256"]),
            "receipt": file_binding(current_receipt),
        },
        "producers": producers,
        "allowed_changed_paths": sorted(ALLOWED_CHANGED_PATHS),
        "observed_changed_paths": sorted(observed_changed_paths),
        "invariants": {
            "all_bundle_receipts_fully_reverified": True,
            "all_changed_paths_predeclared": True,
            "data_payloads_are_reused_by_exact_binding_not_rewritten": True,
            "arbitrary_historical_bundles_are_not_accepted": True,
        },
    }
    validate_authority(payload, current)
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
