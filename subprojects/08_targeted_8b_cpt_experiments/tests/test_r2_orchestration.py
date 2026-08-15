from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_production_timing_and_allocation import (
    MINIMUM_RUNTIME_FRACTION,
    future_json_binding,
    successor_rows,
)
from freeze_hard_h_to_g_contract import role_semantics_match
from patch_bakeoff_uenv10_srun import (
    NEW as UENV10_NEW,
)
from patch_bakeoff_uenv10_srun import (
    OLD as UENV10_OLD,
)
from patch_bakeoff_uenv10_srun import (
    patch as patch_uenv10,
)
from preflight_train_segment import expected_authorization_stage


def test_future_json_binding_matches_atomic_json_encoding(tmp_path: Path) -> None:
    value = {"z": 1, "a": "Greek Ελληνικά"}
    path = tmp_path / "receipt.json"
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    binding = future_json_binding(path, value)
    assert binding["bytes"] == path.stat().st_size
    import hashlib

    assert binding["sha256"] == hashlib.sha256(payload).hexdigest()


def test_successor_schedule_obeys_bounded_holder_arithmetic() -> None:
    segments = [
        {"segment_id": index, "conservative_wall_seconds": seconds}
        for index, seconds in enumerate((12_000, 11_000, 5_000, 12_500, 4_000, 4_000))
    ]
    rows = successor_rows(
        {"8b": {"segments": segments}, "1p5b": {"segments": segments}}
    )
    assert len(rows) == 10
    for row in rows:
        assert (
            row["target_runtime_seconds"]
            + row["maximum_hold_seconds"]
            + row["reserve_seconds"]
            == 43_200
        )
        assert row["source_trigger_seconds"] > 0
        source = segments[row["source_segment_id"]]
        assert (
            source["conservative_wall_seconds"] - row["source_trigger_seconds"]
            <= row["maximum_hold_seconds"]
        )


def test_allocation_schedule_requires_false_debug_timer_invariant(tmp_path: Path) -> None:
    timing = tmp_path / "timing.json"
    timing.write_text("{}\n", encoding="utf-8")
    import hashlib

    binding = {
        "path": str(timing.resolve()),
        "bytes": timing.stat().st_size,
        "sha256": hashlib.sha256(timing.read_bytes()).hexdigest(),
    }
    receipt = {
        "schema_version": "apertus_hard_h_to_g_allocation_schedule_v1",
        "scale": "8b",
        "allocation_seconds": 43_200,
        "reserve_seconds": 1_200,
        "maximum_pending_delayed_successors": 1,
        "timing_receipt": binding,
        "successors": [{"scale": "8b"} for _ in range(5)],
        "invariants": {
            "direct_normal_holder": True,
            "debug_timer_for_normal_holder": False,
            "source_trigger_uses_conservative_source_wall_time": True,
            "holder_verifies_checkpoint_permit_and_target_cache": True,
            "holder_requires_target_runtime_plus_reserve": True,
            "at_most_one_delayed_successor": True,
            "sbatch_test_only_passed_without_manifest_mutation": True,
        },
    }
    assert role_semantics_match("allocation_schedule", receipt, expected_scale="8b")
    receipt["invariants"]["debug_timer_for_normal_holder"] = True
    assert not role_semantics_match("allocation_schedule", receipt, expected_scale="8b")


def test_minimum_runtime_fraction_is_a_lower_tripwire_not_handoff_budget() -> None:
    assert MINIMUM_RUNTIME_FRACTION == 0.75
    source = (ROOT / "scripts/freeze_production_timing_and_allocation.py").read_text(
        encoding="utf-8"
    )
    assert '"minimum_train_seconds": max(' in source
    assert '"conservative_wall_seconds": conservative' in source
    assert '"minimum_runtime_is_lower_completion_tripwire": True' in source
    assert '"conservative_wall_is_upper_allocation_budget": True' in source


def test_8b_profile_is_reused_and_1p5b_is_qualified_in_first_allocation() -> None:
    permit = (ROOT / "scripts/build_training_run_permit.py").read_text(
        encoding="utf-8"
    )
    permit_wrapper = (
        ROOT / "clariden/freeze_training_run_permit_debug.sbatch"
    ).read_text(encoding="utf-8")
    timing = (
        ROOT / "scripts/freeze_production_timing_and_allocation.py"
    ).read_text(encoding="utf-8")
    timing_wrapper = (
        ROOT / "clariden/freeze_production_timing_and_allocation_debug.sbatch"
    ).read_text(encoding="utf-8")
    for text in (permit, permit_wrapper, timing, timing_wrapper):
        assert "producer-compatibility" in text or "H2G_PRODUCER_COMPATIBILITY" in text
    assert "require_accepted_producer(profile_authority" in permit
    assert "require_accepted_producer(lr_selection" in permit
    assert "1.5B profile candidate" in timing
    assert "predeclared_first_allocation_bound_then_measured_eta_refresh" in timing


def test_1p5b_lr_is_fixed_to_8b_recipe_without_pilot_arms() -> None:
    experiment = json.loads(
        (ROOT / "configs/hard_h_to_g_replication_v1.json").read_text(encoding="utf-8")
    )
    policy = experiment["learning_rate_1p5b"]
    assert policy["mode"] == "fixed_matched_8b_recipe"
    assert policy["peak_lr"] == "5.5e-5"
    assert policy["floor_lr"] == "5.5e-6"
    assert policy["lr_pilot_runs"] == 0
    wrapper = (ROOT / "clariden/freeze_lr_selection_debug.sbatch").read_text(
        encoding="utf-8"
    )
    assert "LR pilot receipts are forbidden" in wrapper
    assert "--arm" not in wrapper


def test_phase2_restart_smoke_is_bound_and_production_override_is_cleared() -> None:
    benchmark = (ROOT / "clariden/run_prelaunch_benchmark.sbatch").read_text(
        encoding="utf-8"
    )
    production = (ROOT / "clariden/train_hard_h_to_g_segment.sbatch").read_text(
        encoding="utf-8"
    )
    assert "phase2_uninterrupted" in benchmark
    assert "phase2_resumed" in benchmark
    assert "H2G_PHASE_START_UPDATE_OVERRIDE=1" in benchmark
    assert (
        "unset H2G_PHASE_START_UPDATE_OVERRIDE H2G_PRELAUNCH_PHASE2_RESTART_SMOKE"
        in production
    )


def test_each_training_stage_requires_the_matching_authorization_gate() -> None:
    assert (
        expected_authorization_stage(phase=1, start_update=0, resume_smoke=False)
        == "pre_main"
    )
    assert (
        expected_authorization_stage(phase=2, start_update=2261, resume_smoke=False)
        == "pre_main"
    )
    assert (
        expected_authorization_stage(phase=3, start_update=3218, resume_smoke=False)
        == "pre_extension"
    )
    assert (
        expected_authorization_stage(phase=3, start_update=3456, resume_smoke=False)
        == "pre_second_extension"
    )
    assert (
        expected_authorization_stage(phase=3, start_update=3218, resume_smoke=True)
        == "pre_main"
    )
    assert (
        expected_authorization_stage(phase=3, start_update=3456, resume_smoke=True)
        == "pre_extension"
    )
    production = (ROOT / "clariden/train_hard_h_to_g_segment.sbatch").read_text(
        encoding="utf-8"
    )
    smoke = (ROOT / "clariden/run_phase3_resume_smoke.sbatch").read_text(
        encoding="utf-8"
    )
    assert "H2G_AUTHORIZATION_GATE:?" in production
    assert '--authorization-gate "$H2G_AUTHORIZATION_GATE"' in production
    assert "H2G_AUTHORIZATION_GATE" in smoke


def test_launch_manifest_is_bound_to_explicit_producer_compatibility() -> None:
    freezer = (ROOT / "scripts/freeze_hard_h_to_g_contract.py").read_text(
        encoding="utf-8"
    )
    builder = (ROOT / "scripts/freeze_artifact_manifest.py").read_text(encoding="utf-8")
    assert "--producer-compatibility" in freezer
    assert (
        'manifest.get("producer_bundle_compatibility") == file_binding(args.producer_compatibility)'
        in freezer
    )
    assert "require_accepted_producer(receipt, accepted" in builder


def test_owner_authorization_is_last_and_cannot_be_self_certified_early() -> None:
    owner = (ROOT / "scripts/freeze_owner_authorization.py").read_text(encoding="utf-8")
    wrapper = (
        ROOT / "clariden/freeze_pre_main_artifact_manifest_debug.sbatch"
    ).read_text(encoding="utf-8")
    assert "all_non_owner_stage_gates_passed_first" in owner
    assert "production_launch_authorized_by_scale" in owner
    assert "immutable experiment contract must remain unauthorized" in owner
    assert "H2G_ARTIFACT_MANIFEST_OUTPUT" in wrapper
    assert "owner_production_authorization=$H2G_OWNER_AUTHORIZATION" in wrapper


def test_pre_main_operational_gates_are_scale_scoped_but_extensions_remain_joint() -> (
    None
):
    manifest = (
        ROOT / "clariden/freeze_pre_main_artifact_manifest_debug.sbatch"
    ).read_text(encoding="utf-8")
    launch = (ROOT / "clariden/freeze_pre_main_launch_gate_debug.sbatch").read_text(
        encoding="utf-8"
    )
    timing = (
        ROOT / "clariden/freeze_production_timing_and_allocation_debug.sbatch"
    ).read_text(encoding="utf-8")
    preflight = (ROOT / "scripts/preflight_train_segment.py").read_text(
        encoding="utf-8"
    )
    for text in (manifest, launch, timing):
        assert "H2G_SCALE" in text
    assert '--scale "$H2G_SCALE"' in manifest
    assert '--scale "$H2G_SCALE"' in launch
    assert '--scale "$H2G_SCALE"' in timing
    assert 'scale if expected_stage == "pre_main" else None' in preflight


def test_extension_manifests_and_launch_gates_are_joint_and_fail_closed() -> None:
    manifest = (
        ROOT / "clariden/freeze_extension_artifact_manifest_debug.sbatch"
    ).read_text(encoding="utf-8")
    launch = (ROOT / "clariden/freeze_extension_launch_gate_debug.sbatch").read_text(
        encoding="utf-8"
    )
    for text in (manifest, launch):
        assert "#SBATCH --partition=debug" in text
        assert '"${SLURM_NNODES:-0}" == 1' in text
        assert "forbid H2G_SCALE" in text
        assert "verify_code_bundle.py" in text
    assert "pre_extension:pre_authorization" in manifest
    assert "owner_extension_authorization" in manifest
    assert "pre_second_extension:final" in manifest
    assert "artifact_manifest_${H2G_GATE_STAGE}.json" in launch
    assert "launch_gate_${H2G_GATE_STAGE}.json" in launch
    assert '--gate-stage "$H2G_GATE_STAGE"' in launch


def test_phase_cache_and_checkpoint_paths_match_canonical_outputs() -> None:
    phase3 = (ROOT / "clariden/freeze_phase3_authority_debug.sbatch").read_text(
        encoding="utf-8"
    )
    ledger = (
        ROOT / "clariden/freeze_cross_scale_realized_ledger_debug.sbatch"
    ).read_text(encoding="utf-8")
    paired = (
        ROOT / "clariden/freeze_post_checkpoint_authorities_debug.sbatch"
    ).read_text(encoding="utf-8")
    assert "$H2G_STAGE_ROOT/receipts/phase_3_blend_cache.json" in phase3
    assert "$H2G_STAGE_ROOT/receipts/phase3_unseen_catalog.json" in phase3
    assert "$H2G_STAGE_ROOT/receipts/tokenized_phase3_openarchives.json" in phase3
    assert '"${H2G_PHASE1_CACHE_RECEIPT:?' in ledger
    assert '"${H2G_PHASE2_CACHE_RECEIPT:?' in ledger
    assert '--phase1-cache-receipt "$H2G_PHASE1_CACHE_RECEIPT"' in ledger
    assert '--phase2-cache-receipt "$H2G_PHASE2_CACHE_RECEIPT"' in ledger
    assert "H2G_8B_UPDATE_3218_PERMIT" in ledger
    assert "H2G_1P5B_UPDATE_3218_PERMIT" in ledger
    assert "$H2G_STAGE_ROOT/receipts/phase1_phase2_realized_documents.json" in ledger
    assert "H2G_8B_SOURCE_RECEIPT" in paired
    assert "H2G_1P5B_SOURCE_RECEIPT" in paired
    assert "data/phases/phase3/cache_receipt.json" not in phase3


def test_phase3_tokenizer_derives_megatron_root_from_bound_receipt() -> None:
    wrapper = (ROOT / "clariden/tokenize_phase3_stream_debug.sbatch").read_text(
        encoding="utf-8"
    )
    assert 'value.get("output_root"' in wrapper
    assert "megatron_training_c92402e_extra_valid\"" not in wrapper
    assert '"$H2G_STAGE_ROOT"/tools/*' in wrapper


def test_first_allocation_qualification_has_no_nested_sbatch() -> None:
    adapter = (
        ROOT / "scripts/run_in_allocation_profile_qualification.py"
    ).read_text(encoding="utf-8")
    assert "run_prelaunch_benchmark.sbatch" in adapter
    assert "profile_benchmark.json" in adapter
    assert "apertus_gate_context_v1" in adapter
    assert '["sbatch"' not in adapter


def test_existing_8b_profile_adapter_rechecks_slurm_accounting() -> None:
    adapter = (
        ROOT / "scripts/build_existing_8b_qualification_context.py"
    ).read_text(encoding="utf-8")
    assert '"sacct", "-X"' in adapter
    assert '"exact_nodes": row[3] == "16"' in adapter
    assert "require_accepted_producer" in adapter
    assert "apertus_gate_context_v1" in adapter
    assert '["sbatch"' not in adapter


def test_td_coverage_audits_non_nfc_without_transforming_training_text() -> None:
    scanner = (ROOT / "scripts/td_coverage_prepass_batched.py").read_text(
        encoding="utf-8"
    )
    freezer = (ROOT / "scripts/freeze_td_training_inputs.py").read_text(
        encoding="utf-8"
    )
    wrapper = (ROOT / "clariden/build_td_snippets_xfer.sbatch").read_text(
        encoding="utf-8"
    )
    assert 'if text != ud.normalize("NFC", text):' in scanner
    assert 'if args.require_nfc and text != ud.normalize("NFC", text):' not in scanner
    assert "--no-require-nfc" in wrapper
    assert '"input_bytes_transformed": False' in freezer
    assert '"policy": "audit_and_preserve_exact_stage_b_text"' in freezer


def test_prelaunch_benchmark_accepts_only_audited_historical_producers() -> None:
    freezer = (ROOT / "scripts/freeze_prelaunch_benchmark_contract.py").read_text(
        encoding="utf-8"
    )
    preflight = (ROOT / "scripts/verify_prelaunch_benchmark_contract.py").read_text(
        encoding="utf-8"
    )
    wrapper = (
        ROOT / "clariden/freeze_prelaunch_benchmark_contract_debug.sbatch"
    ).read_text(encoding="utf-8")
    online = (ROOT / "scripts/freeze_online_validation_binaries.py").read_text(
        encoding="utf-8"
    )
    assert "--producer-compatibility" in freezer
    assert "H2G_PRODUCER_COMPATIBILITY" in wrapper
    assert "load_authority(compatibility_path, current)" in preflight
    assert "require_accepted_producer(cache, accepted_producers" in preflight
    assert "phase_cache_overlay_receipt" in freezer
    assert "validate_overlay_receipt" in preflight
    assert "accepted_code_bundles=accepted_code_bundles" in preflight
    assert "accepted_code_bundles: set[tuple[str, str]] | None" in online


def test_uenv10_patch_mounts_the_image_in_both_srun_launch_modes(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "trainer.sbatch"
    trainer.write_text(("prefix\n" + UENV10_OLD + "one\n") * 2, encoding="utf-8")
    patch_uenv10(trainer)
    patched = trainer.read_text(encoding="utf-8")
    assert patched.count(UENV10_NEW) == 2
    assert UENV10_OLD not in patched
    patch_uenv10(trainer)


def test_uenv10_srun_smoke_is_routed_to_debug() -> None:
    smoke = (ROOT / "clariden/prove_uenv10_srun_debug.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=debug" in smoke
    assert "srun --uenv=pytorch/v2.9.1:v2 --view=default" in smoke
    assert "command -v torchrun" in smoke


def test_resume_checkpoint_loads_parent_root_not_iteration_directory() -> None:
    preflight = (ROOT / "scripts/preflight_train_segment.py").read_text(
        encoding="utf-8"
    )
    permit = (ROOT / "scripts/build_checkpoint_permit.py").read_text(encoding="utf-8")
    assert (
        'checkpoint_root = args.load_checkpoint / f"iter_{args.start_update:07d}"'
        in preflight
    )
    assert '"load_root": str(load_root)' in permit
    assert '"load_tracker": file_binding(tracker)' in permit


def test_legacy_compatibility_path_changes_only_dataset_loading() -> None:
    adapter = (ROOT / "scripts/run_legacy_greekmmlu_snapshot_eval.py").read_text(
        encoding="utf-8"
    )
    freezer = (ROOT / "scripts/freeze_legacy_public_evaluator.py").read_text(
        encoding="utf-8"
    )
    assert "legacy._load_dataset = snapshot_loader" in adapter
    assert '"loader_change_scope": "dataset_loading_only"' in freezer
    assert "loader-parity receipt drift" in freezer
