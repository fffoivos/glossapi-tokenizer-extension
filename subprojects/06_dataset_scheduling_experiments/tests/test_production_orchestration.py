from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from array import array
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_contract():
    path = ROOT / "production" / "campaign_contract.py"
    spec = importlib.util.spec_from_file_location("campaign_contract_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_checkpoint_export():
    path = ROOT / "evaluation" / "checkpoint_eval_export_receipt.py"
    spec = importlib.util.spec_from_file_location("checkpoint_export_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_initial_checkpoint_export():
    path = ROOT / "evaluation" / "prepare_initial_checkpoint_hf_export.py"
    spec = importlib.util.spec_from_file_location("initial_checkpoint_export_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProductionOrchestrationTests(unittest.TestCase):
    def test_evaluation_runtime_override_is_receipt_and_hash_bound(self):
        contract = load_contract()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "campaign.json"
            manifest.write_text("{}\n")
            original_scientific = root / "scientific-v50"
            original_megatron = root / "megatron-v50"
            replacement_scientific = root / "scientific-v51"
            replacement_megatron = root / "megatron-v51"
            for path in (
                original_scientific,
                original_megatron,
                replacement_scientific,
                replacement_megatron,
            ):
                path.mkdir()
            runtime_file = replacement_scientific / "controller.py"
            runtime_file.write_text("print('fixed')\n")
            megatron_file = replacement_megatron / "saver.py"
            megatron_file.write_text("print('semantic parity')\n")
            rows = [
                {
                    "relative_path": "controller.py",
                    "bytes": runtime_file.stat().st_size,
                    "sha256": contract.sha256_file(runtime_file),
                }
            ]
            bundle_receipt = root / "bundle.json"
            bundle_receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "apertus_mini_immutable_code_bundle_v1",
                        "status": "frozen",
                        "kind": "scientific",
                        "root": str(replacement_scientific),
                        "file_count": 1,
                        "tree_sha256": hashlib.sha256(
                            json.dumps(
                                rows, separators=(",", ":"), sort_keys=True
                            ).encode()
                        ).hexdigest(),
                        "files": rows,
                    }
                )
            )
            diagnostic = root / "diagnostic.json"
            diagnostic.write_text(
                json.dumps(
                    {
                        "schema_version": "apertus_mini_conversion_semantic_parity_diagnostic_v1",
                        "status": "passed",
                    }
                )
            )
            receipt = root / "recovery.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "apertus_mini_evaluation_runtime_recovery_v1",
                        "status": "frozen",
                        "evaluation_namespace": "fp32_v1",
                        "authoritative_evaluation_dtype": "float32",
                        "campaign_manifest": {
                            "path": str(manifest),
                            "sha256": contract.sha256_file(manifest),
                        },
                        "overrides": {
                            "scientific_bundle": {
                                "from_path": str(original_scientific),
                                "to_path": str(replacement_scientific),
                            },
                            "megatron_dir": {
                                "from_path": str(original_megatron),
                                "to_path": str(replacement_megatron),
                            },
                        },
                        "runtime_files": [
                            {
                                "path": str(runtime_file),
                                "sha256": contract.sha256_file(runtime_file),
                            },
                            {
                                "path": str(megatron_file),
                                "sha256": contract.sha256_file(megatron_file),
                            },
                        ],
                        "scientific_bundle_receipt": {
                            "path": str(bundle_receipt),
                            "sha256": contract.sha256_file(bundle_receipt),
                        },
                        "semantic_parity_diagnostics": {
                            "path": str(diagnostic),
                            "sha256": contract.sha256_file(diagnostic),
                        },
                    }
                )
            )
            campaign = {
                "assets": {
                    "scientific_bundle": str(original_scientific),
                    "megatron_dir": str(original_megatron),
                }
            }
            environment = {
                "SCIENTIFIC_BUNDLE": str(replacement_scientific),
                "EVALUATION_MEGATRON_DIR": str(replacement_megatron),
                "OPERATIONAL_RECOVERY_RECEIPT": str(receipt),
                "EVALUATION_NAMESPACE": "fp32_v1",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                observed = contract.resolve_evaluation_runtime(manifest, campaign)
                self.assertEqual(
                    observed,
                    (replacement_scientific.resolve(), replacement_megatron.resolve()),
                )
                runtime_file.write_text("drift\n")
                with self.assertRaises(ValueError):
                    contract.resolve_evaluation_runtime(manifest, campaign)

    def test_frozen_geometry_and_counts(self):
        contract = load_contract()
        self.assertEqual(contract.TOTAL_ITERATIONS, 38_496)
        self.assertEqual(contract.SEGMENT_BOUNDARY, 19_456)
        self.assertEqual(contract.EXPECTED_CHECKPOINTS_PER_ARM, 83)
        self.assertEqual(contract.EXPECTED_GREEKMMLU_TOTAL, 415)
        self.assertEqual(contract.SCHEDULED_TOKEN_SLOTS, 2_097_152 * 38_496)
        self.assertEqual(len(contract.ARMS), 5)

    def test_checkpoint_plan_requires_exact_83_points(self):
        contract = load_contract()
        values = list(range(81)) + [38_000, 38_496]
        plan = {"checkpoint_rows": [{"iteration": value} for value in values]}
        self.assertEqual(len(contract.checkpoint_iterations(plan)), 83)
        plan["checkpoint_rows"][-1]["iteration"] = 38_495
        with self.assertRaises(ValueError):
            contract.checkpoint_iterations(plan)

    def test_checkpoint_plan_must_bind_current_matrix_and_schedule(self):
        contract = load_contract()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = root / "schedule.json"
            matrix = root / "matrix.json"
            matrix.write_text("{}\n")
            sequence_receipts = {}
            for arm_id, before, source_pool, destination_pool in (
                ("D1_hard_h_to_g", 1000, 0, 1),
                ("D2_hard_g_to_h", 2000, 1, 0),
            ):
                ids_path = root / f"{arm_id}.u64"
                values = array(
                    "Q",
                    [
                        (source_pool << 62) | index
                        for index in range(before * 512)
                    ]
                    + [(destination_pool << 62) | (before * 512)],
                )
                with ids_path.open("wb") as handle:
                    values.tofile(handle)
                sequence_receipts[arm_id] = {
                    "path": str(ids_path.resolve()),
                    "bytes": ids_path.stat().st_size,
                    "sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
                }
            schedule.write_text(
                json.dumps(
                    {
                        "schema_version": "apertus_mini_five_data_order_schedules_v1",
                        "status": "completed",
                        "arms": [
                            {
                                "arm_id": arm_id,
                                **(
                                    {"sequence_ids": sequence_receipts[arm_id]}
                                    if arm_id in sequence_receipts
                                    else {}
                                ),
                            }
                            for arm_id in contract.ARMS
                        ],
                    }
                )
                + "\n"
            )
            rows = []
            required = {0, 800, 19_456, int(0.8 * 38_496), 38_496}
            required.update(range(512, 38_496 + 1, 512))
            required.update({1000, 1001, 2000, 2001})
            self.assertEqual(len(required), 83)
            for iteration in sorted(required):
                reasons = []
                if iteration == 0:
                    reasons.append("initial_checkpoint")
                if iteration == 800:
                    reasons.append("after_warmup")
                if iteration == 19_456:
                    reasons.append("normal_partition_segment_boundary")
                if iteration == int(0.8 * 38_496):
                    reasons.append("cooldown_start")
                if iteration == 38_496:
                    reasons.append("raw_final_endpoint")
                if iteration and iteration % 512 == 0:
                    reasons.append("regular_512_step_cadence")
                for arm_id, before in (
                    ("D1_hard_h_to_g", 1000),
                    ("D2_hard_g_to_h", 2000),
                ):
                    if iteration == before:
                        reasons.append(f"matched_{arm_id}_pre_transition")
                    if iteration == before + 1:
                        reasons.append(f"matched_{arm_id}_post_transition")
                rows.append(
                    {
                        "iteration": iteration,
                        "nominal_consumed_tokens": iteration * 512 * 4096,
                        "reasons": reasons,
                        "all_arms": list(contract.ARMS),
                        "full_state_checkpoint_required": True,
                        "fast_source_conditioned_panel_required": True,
                        "native_greekmmlu_required": True,
                        "native_greekmmlu_metrics": [
                            "official_zero_shot_accuracy",
                            "multiple_choice_cross_entropy_from_frozen_normalized_choice_scores",
                            "correct_answer_continuation_bpb",
                        ],
                        "same_frozen_evaluator_contract": True,
                    }
                )

            def receipt(path):
                return {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema_version": "apertus_mini_checkpoint_evaluation_plan_v1",
                        "status": "frozen",
                        "schedule_manifest": receipt(schedule),
                        "experiment_matrix": receipt(matrix),
                        "optimizer_steps": 38_496,
                        "checkpoint_count_per_arm": 83,
                        "native_greekmmlu_evaluations_total": 415,
                        "greekmmlu_origin": "natively_authored_greek",
                        "greekmmlu_dataset": {
                            "repo_id": "dascim/GreekMMLU",
                            "revision": "6a03aa06b68beb932fb75edff3a34e50b3674649",
                            "config": "All",
                            "split": "test",
                        },
                        "hard_transitions": {
                            arm_id: {
                                "first_destination_sequence_slot": before * 512,
                                "optimizer_update_containing_first_destination_sequence": before + 1,
                                "checkpoint_immediately_before": before,
                                "checkpoint_after_first_complete_transition_update": before + 1,
                            }
                            for arm_id, before in (
                                ("D1_hard_h_to_g", 1000),
                                ("D2_hard_g_to_h", 2000),
                            )
                        },
                        "checkpoint_rows": rows,
                    }
                )
                + "\n"
            )
            contract.verify_checkpoint_plan(plan, schedule, matrix)
            matrix.write_text('{"changed": true}\n')
            with self.assertRaisesRegex(ValueError, "source experiment matrix"):
                contract.verify_checkpoint_plan(plan, schedule, matrix)

    def test_atomic_json_replace_is_valid(self):
        contract = load_contract()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            contract.atomic_write_json(path, {"status": "first"})
            contract.atomic_write_json(path, {"status": "second"}, exclusive=False)
            self.assertEqual(json.loads(path.read_text()), {"status": "second"})
            self.assertFalse(list(Path(directory).glob("*.partial")))

    def test_shell_entrypoints_parse(self):
        paths = [
            ROOT / "clariden" / "run_production_arm_segment.sh",
            ROOT / "clariden" / "train_five_arm_segment.sbatch",
            ROOT / "clariden" / "watch_checkpoint_evaluations.sbatch",
            ROOT / "clariden" / "freeze_segment_checkpoint.sbatch",
            ROOT / "clariden" / "gate_segment.sbatch",
            ROOT / "clariden" / "supervise_production_segment.sbatch",
            ROOT / "clariden" / "run_full_endpoint_validation.sbatch",
            ROOT / "clariden" / "run_greek_endpoint_wave.sbatch",
            ROOT / "clariden" / "run_retention_endpoint_wave.sbatch",
            ROOT / "clariden" / "run_retention_endpoint_one.sh",
            ROOT / "clariden" / "finalize_core_campaign_evidence.sbatch",
            ROOT / "clariden" / "finalize_campaign_evidence.sbatch",
            ROOT / "clariden" / "run_tied_td_pilots.sbatch",
            ROOT / "clariden" / "evaluate_tied_td_pilots.sbatch",
            ROOT / "clariden" / "run_full_tied_td.sbatch",
            ROOT / "clariden" / "verify_full_tied_td.sbatch",
            ROOT / "clariden" / "convert_full_td_to_megatron.sbatch",
            ROOT / "clariden" / "finalize_td_conversion_receipt.sbatch",
            ROOT / "clariden" / "finalize_tied_td_initialization.sbatch",
            ROOT / "clariden" / "submit_tied_td_pipeline.sh",
            ROOT / "clariden" / "run_common_stability_smoke.sbatch",
            ROOT / "clariden" / "run_prelaunch_smoke_arm.sh",
            ROOT / "clariden" / "run_five_arm_prelaunch_smoke.sbatch",
            ROOT / "clariden" / "submit_production_campaign.sh",
            ROOT / "clariden" / "finalize_prelaunch_campaign.sh",
            ROOT / "clariden" / "build_neutral_external_heldout.sbatch",
            ROOT / "clariden" / "build_validation_manifest.sbatch",
            ROOT / "clariden" / "build_packed_training_td_assets.sbatch",
            ROOT / "clariden" / "build_token_byte_lengths.sbatch",
            ROOT / "clariden" / "build_production_megatron.sbatch",
            ROOT / "clariden" / "prepare_neutral_external_source.sbatch",
            ROOT / "clariden" / "build_neutral_candidate_signatures.sbatch",
            ROOT / "clariden" / "match_neutral_minhash_bucket.sbatch",
            ROOT / "clariden" / "finalize_neutral_cross_dedup.sbatch",
            ROOT / "clariden" / "submit_neutral_external_pipeline.sh",
            ROOT / "clariden" / "build_lm_eval_runtime.sbatch",
            ROOT / "clariden" / "finalize_static_prelaunch_evidence.sbatch",
            ROOT / "training" / "runtime_patches" / "apply_production_megatron_patches.sh",
        ]
        for path in paths:
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_tied_td_submitter_is_dry_run_and_confirmation_gated(self):
        text = (ROOT / "clariden" / "submit_tied_td_pipeline.sh").read_text()
        self.assertIn("DRY_RUN=${DRY_RUN:-1}", text)
        self.assertIn("CONFIRM_GPU_LAUNCH=MINI_TIED_TD", text)
        self.assertIn('dependency="afterok:$pilots"', text)
        self.assertIn('dependency="afterok:$conversion"', text)
        self.assertIn("TD_CANONICAL_ADAPTER", text)
        self.assertIn("TD_FAIR_METRICS_SCRIPT", text)
        self.assertNotIn("$MINI_CODE_BUNDLE/../03_", text)

    def test_full_td_requires_fvt_macro_non_regression(self):
        text = (ROOT / "clariden" / "run_full_tied_td.sbatch").read_text()
        self.assertIn('d["fvt_macro_non_regression_gate_passed"] is True', text)
        self.assertIn('d["selected_delta_macro_bpb_vs_fvt"]', text)
        launch_gate = (ROOT / "production" / "finalize_launch_gate_set.py").read_text()
        self.assertIn('initialization.get("fvt_macro_non_regression_gate_passed") is True', launch_gate)
        self.assertIn('initialization.get("selected_delta_macro_bpb_vs_fvt", 1.0)', launch_gate)

    def test_host_side_runtime_python_is_pinned(self):
        paths = (
            "finalize_tied_td_initialization.sbatch",
            "finalize_td_conversion_receipt.sbatch",
            "build_production_megatron.sbatch",
            "run_common_stability_smoke.sbatch",
            "run_five_arm_prelaunch_smoke.sbatch",
            "run_prelaunch_smoke_arm.sh",
            "finalize_prelaunch_campaign.sh",
            "train_five_arm_segment.sbatch",
            "supervise_production_segment.sbatch",
            "run_production_arm_segment.sh",
            "watch_checkpoint_evaluations.sbatch",
            "freeze_segment_checkpoint.sbatch",
            "gate_segment.sbatch",
            "run_initial_validation.sbatch",
            "run_full_endpoint_validation.sbatch",
            "finalize_campaign_evidence.sbatch",
            "finalize_core_campaign_evidence.sbatch",
            "run_greek_endpoint_wave.sbatch",
            "run_retention_endpoint_wave.sbatch",
            "run_checkpoint_native_greekmmlu_wave.sbatch",
            "run_checkpoint_native_greekmmlu_one.sh",
            "submit_production_campaign.sh",
            "submit_tied_td_pipeline.sh",
        )
        for name in paths:
            with self.subTest(name=name):
                text = (ROOT / "clariden" / name).read_text()
                self.assertIn("HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}", text)

    def test_production_megatron_patches_are_self_contained_and_pinned(self):
        root = ROOT / "training" / "runtime_patches"
        applier = (root / "apply_production_megatron_patches.sh").read_text()
        extra = root / "megatron_extra_valid_c92402e.patch"
        exact = root / "megatron_exact_eval_iterations_c92402e.patch"
        self.assertTrue(extra.is_file())
        self.assertTrue(exact.is_file())
        self.assertNotIn("../05_token_distillation_cpt", applier)
        self.assertIn(hashlib.sha256(extra.read_bytes()).hexdigest(), applier)
        self.assertIn(hashlib.sha256(exact.read_bytes()).hexdigest(), applier)
        self.assertIn("git clone --quiet --no-hardlinks", applier)

    def test_td_conversion_propagates_pythonpath_inside_uenv(self):
        text = (ROOT / "clariden" / "convert_full_td_to_megatron.sbatch").read_text()
        self.assertEqual(text.count('uenv run "$UENV_IMAGE" --view=default --'), 7)
        self.assertEqual(
            text.count(
                'uenv run "$UENV_IMAGE" --view=default -- env PYTHONPATH="$PYTHONPATH"'
            ),
            7,
        )
        self.assertIn(
            "--ckpt-format torch --ckpt-convert-format torch_dist",
            text,
        )
        inverse = (ROOT / "initialization" / "torch_to_torchdist.py").read_text()
        self.assertNotIn('"ckpt_format": "torch"', inverse)
        self.assertNotIn('"ckpt_convert_format": "torch_dist"', inverse)
        self.assertEqual(text.count('echo release > "$INITIAL_ROOT/latest_checkpointed_iteration.txt"'), 1)
        self.assertEqual(
            text.count(
                'echo release > "$ROUNDTRIP_TORCH_ROOT/latest_checkpointed_iteration.txt"'
            ),
            1,
        )
        self.assertIn('--load-dir "$ROUNDTRIP_TORCH_ROOT"', text)
        conversion_finalizer = (
            ROOT / "initialization" / "finalize_td_megatron_conversion.py"
        ).read_text()
        initialization_finalizer = (
            ROOT / "initialization" / "finalize_mini_td_initialization.py"
        ).read_text()
        self.assertIn('latest.read_text().strip() != "release"', conversion_finalizer)
        self.assertIn('"release_tree": tree(release)', conversion_finalizer)
        self.assertNotIn('iteration_zero_tree', conversion_finalizer)
        self.assertIn('verify_tree(initial_root / "release"', initialization_finalizer)
        self.assertNotIn('initial_root / "iter_0000000"', initialization_finalizer)

    def test_production_lr_is_manifest_bound(self):
        driver = (ROOT / "clariden" / "run_production_arm_segment.sh").read_text()
        aggregate = (ROOT / "clariden" / "train_five_arm_segment.sbatch").read_text()
        builder = (ROOT / "production" / "build_campaign_manifest.py").read_text()
        self.assertIn('PEAK_LR=${contract[9]}', driver)
        self.assertIn('--lr "$PEAK_LR" --min-lr "$MIN_LR"', driver)
        self.assertNotIn('PEAK_LR="${PEAK_LR:-', aggregate)
        self.assertIn('--lr-selection-receipt', builder)

    def test_patched_megatron_runtime_is_receipt_bound(self):
        builder = (ROOT / "production" / "build_campaign_manifest.py").read_text()
        preflight = (ROOT / "production" / "preflight_segment.py").read_text()
        finalizer = (ROOT / "production" / "finalize_launch_gate_set.py").read_text()
        closure = (ROOT / "clariden" / "finalize_prelaunch_campaign.sh").read_text()
        for text in (builder, preflight, finalizer, closure):
            self.assertIn("megatron_runtime_receipt", text.lower())
        self.assertIn("apertus_mini_patched_megatron_runtime_v1", builder)
        self.assertIn("apertus_mini_patched_megatron_runtime_v1", preflight)
        self.assertIn("apertus_mini_patched_megatron_runtime_v1", finalizer)

    def test_lm_eval_runtime_reconstruction_is_exact_and_fully_receipted(self):
        builder = (ROOT / "clariden" / "build_lm_eval_runtime.sbatch").read_text()
        freezer = (ROOT / "evaluation" / "freeze_lm_eval_runtime.py").read_text()
        lock = (ROOT / "evaluation" / "lm_eval_runtime_requirements_0_4_11.txt").read_text()
        self.assertIn("--no-deps", builder)
        self.assertEqual(builder.count('env PYTHONPATH="$'), 2)
        self.assertIn("lm-eval==0.4.11", lock)
        self.assertIn("accelerate==1.13.0", lock)
        self.assertIn('EXPECTED_UENV = "pytorch/v2.9.1:v2"', freezer)
        self.assertIn("target_distributions(root)", freezer)
        self.assertIn("external_distributions", freezer)
        alias = (ROOT / "evaluation" / "lm_eval_custom_tasks" / "global_mmlu.yaml").read_text()
        self.assertIn("group: global_mmlu", alias)
        self.assertEqual(alias.count("  - global_mmlu_"), 15)
        self.assertIn("official runtime unexpectedly defines global_mmlu alias", builder)

    def test_campaign_uses_canonical_release_initial_checkpoint(self):
        builder = (ROOT / "production" / "build_campaign_manifest.py").read_text()
        self.assertIn('latest.read_text(encoding="utf-8").strip() != "release"', builder)
        self.assertIn('initial_checkpoint_root.resolve() / "release"', builder)
        self.assertNotIn('initial_checkpoint_root.resolve() / "iter_0000000"', builder)

    def test_prelaunch_gate_accepts_receipted_phase_names(self):
        finalizer = (ROOT / "production" / "finalize_launch_gate_set.py").read_text()
        self.assertIn('"initial_to_64"', finalizer)
        self.assertIn('"resume_64_to_128"', finalizer)
        self.assertNotIn('row.get("phase") in {"initial", "resume"}', finalizer)

    def test_stability_smoke_decision_includes_validation_and_retention(self):
        runner = (ROOT / "clariden" / "run_common_stability_smoke.sbatch").read_text()
        evaluator = (ROOT / "training" / "evaluate_common_stability_smoke.py").read_text()
        self.assertIn("run_initial_validation", runner)
        self.assertIn("run_endpoint_validation", runner)
        self.assertIn('--initial-validation-log "$SMOKE_ROOT/initial_validation.log"', runner)
        self.assertIn('--endpoint-validation-log "$output/endpoint_validation.log"', runner)
        self.assertIn("--skip-train --eval-iters 1", runner)
        self.assertNotIn("--eval-only", runner)
        self.assertIn("retention_panels_within_predeclared_relative_margin", evaluator)
        self.assertIn("added_token_target_loss_present_and_stable_on_four_greek_probes", evaluator)

    def test_stability_fallback_reuses_failed_candidate_and_changes_only_lr(self):
        text = (ROOT / "clariden" / "run_common_stability_fallback.sbatch").read_text()
        self.assertIn('[[ "$candidate_verdict" -eq 1 ]]', text)
        self.assertIn("--lr 1.5e-4 --min-lr 1.5e-5", text)
        self.assertIn('--driver-log "$CANDIDATE_DRIVER"', text)
        self.assertIn('--fallback-receipt "$output/receipt.json"', text)
        self.assertNotIn("--lr 3e-4 --min-lr 3e-5", text)

    def test_production_driver_uses_full_run_anchors(self):
        text = (ROOT / "clariden" / "run_production_arm_segment.sh").read_text()
        required = (
            "TOTAL_ITERATIONS=38496",
            "TRAIN_SAMPLES=19709952",
            "LR_WARMUP_SAMPLES=409600",
            "LR_WSD_DECAY_SAMPLES=3941990",
            '--ademamix-beta3-warmup "$TOTAL_ITERATIONS"',
            '--ademamix-alpha-warmup "$TOTAL_ITERATIONS"',
            '--exit-interval "$END_ITERATION"',
            "MINI_SCHEDULE_EVAL_ITERATIONS",
            '--lr "$PEAK_LR" --min-lr "$MIN_LR"',
        )
        for value in required:
            self.assertIn(value, text)
        self.assertNotIn("BENCHMARK_STEPS", text)
        self.assertNotIn('PEAK_LR:-', text)
        self.assertIn('EVAL_ITERS=${EVAL_ITERS:-1}', text)

    def test_training_validation_disables_goldfish_and_stratifies(self):
        text = (ROOT / "training" / "pretrain_scheduled_gpt.py").read_text()
        self.assertIn("goldfish_loss=False", text)
        self.assertIn('reporting["base-token target loss"]', text)
        self.assertIn('reporting["added-token target loss"]', text)
        self.assertNotIn('if float(reduced[1]) > 0:', text)
        self.assertNotIn('if float(reduced[3]) > 0:', text)

    def test_failed_attempt_checkpoint_recovery_is_explicit_and_audited(self):
        freezer = (ROOT / "production" / "freeze_segment_checkpoint.py").read_text()
        wrapper = (ROOT / "clariden" / "freeze_segment_checkpoint.sbatch").read_text()
        self.assertIn('"--failed-attempt-recovery"', freezer)
        self.assertIn('"failed_attempt_recovery": args.failed_attempt_recovery', freezer)
        self.assertIn('"segment_state_sha256"', freezer)
        self.assertIn('FAILED_ATTEMPT_RECOVERY', wrapper)

    def test_goldfish_audit_accepts_python_digit_separators(self):
        text = (ROOT / "training" / "audit_goldfish_added_token_uniformity.py").read_text()
        self.assertIn('([\\d_]+)', text)
        self.assertEqual(text.count('.replace("_", "")'), 2)

    def test_submitter_is_dry_run_by_default_and_gated(self):
        text = (ROOT / "clariden" / "submit_production_campaign.sh").read_text()
        self.assertIn("DRY_RUN=${DRY_RUN:-1}", text)
        self.assertIn("CONFIRM_GPU_LAUNCH=FIVE_ARM_MINI_CPT", text)
        self.assertIn('dependency="after:$train0"', text)
        self.assertIn("supervise_production_segment.sbatch", text)
        self.assertNotIn("train1=$(sbatch", text)

    def test_prelaunch_closure_cannot_submit_gpu_work(self):
        text = (ROOT / "clariden" / "finalize_prelaunch_campaign.sh").read_text()
        self.assertIn("DRY_RUN=1", text)
        self.assertNotIn("CONFIRM_GPU_LAUNCH", text)
        self.assertIn("finalize_launch_gate_set.py", text)
        self.assertIn("freeze_code_bundle.py", text)
        freezer = (ROOT / "production" / "freeze_code_bundle.py").read_text()
        self.assertIn('".venv"', freezer)

    def test_recovery_keeps_full_segment_evaluation_inventory(self):
        text = (ROOT / "production" / "watch_checkpoint_evaluations.py").read_text()
        self.assertIn("nominal_start < int(value) <= end", text)
        self.assertIn('elif iteration <= start and recovery_roots is not None:', text)
        self.assertIn('"recovery_start": start', text)

    def test_only_explicit_infrastructure_states_retry(self):
        text = (ROOT / "production" / "supervise_production_segment.py").read_text()
        self.assertIn('RETRYABLE_INFRA = {"BOOT_FAIL", "NODE_FAIL", "PREEMPTED", "REVOKED", "TIMEOUT"}', text)
        self.assertIn("MAX_ATTEMPT = 2", text)
        self.assertNotIn('RETRYABLE_INFRA = {"FAILED"', text)

    def test_evaluation_watcher_recognizes_preemption_and_revocation(self):
        text = (ROOT / "production" / "watch_checkpoint_evaluations.py").read_text()
        self.assertIn('"PREEMPTED"', text)
        self.assertIn('"REVOKED"', text)
        self.assertIn('line.strip().split()[0].split("+")[0]', text)

    def test_xfer_controllers_reuse_the_pinned_python_interpreter(self):
        watcher = (ROOT / "production" / "watch_checkpoint_evaluations.py").read_text()
        supervisor = (ROOT / "production" / "supervise_production_segment.py").read_text()
        self.assertIn("sys.executable", watcher)
        self.assertNotIn('[\n                "python3",', watcher)
        self.assertEqual(supervisor.count("sys.executable"), 4)

    def test_watcher_binds_completed_receipts_to_arm_and_finite_metrics(self):
        text = (ROOT / "production" / "watch_checkpoint_evaluations.py").read_text()
        self.assertIn('pipeline.get("arm_id") == arm', text)
        self.assertIn('int(metrics.get("n", -1)) == 16_632', text)
        self.assertIn("math.isfinite(metric)", text)
        self.assertIn('sha256_file(export_path) == checkpoint.get("export_receipt_sha256")', text)
        self.assertIn('value.get("evaluation_namespace") == namespace', text)
        self.assertIn("AUTHORITATIVE_EVALUATION_DTYPE", text)

    def test_segment_gate_requires_both_validation_modalities(self):
        gate = (ROOT / "production" / "gate_segment.py").read_text()
        supervisor = (ROOT / "production" / "supervise_production_segment.py").read_text()
        self.assertIn("segment_source_validation_gate_v1", gate)
        self.assertIn("all_metrics_finite", gate)
        self.assertIn("source_validation_bindings", gate)
        self.assertIn("audit_segment_source_validation.py", supervisor)

    def test_authoritative_greekmmlu_isolated_and_float32(self):
        runner = (ROOT / "clariden" / "run_checkpoint_native_greekmmlu.sbatch").read_text()
        finalizer = (ROOT / "evaluation" / "finalize_checkpoint_greekmmlu_receipt.py").read_text()
        self.assertIn("--dtype float32", runner)
        self.assertIn("EVALUATION_NAMESPACE", runner)
        self.assertIn('metadata.get("dtype") != "float32"', finalizer)
        self.assertIn('"evaluation_namespace": args.evaluation_namespace', finalizer)

    def test_watcher_reuses_valid_lanes_and_retries_only_missing_arms(self):
        text = (ROOT / "production" / "watch_checkpoint_evaluations.py").read_text()
        self.assertIn("completed_receipts: dict[str, Path]", text)
        self.assertIn("missing_arms = [arm for arm in ARMS if arm not in completed_receipts]", text)
        self.assertIn('"requested_arms": missing_arms', text)

    def test_evaluation_controllers_cover_a_full_segment_and_validation_lag(self):
        watcher = (ROOT / "production" / "watch_checkpoint_evaluations.py").read_text()
        watcher_sbatch = (ROOT / "clariden" / "watch_checkpoint_evaluations.sbatch").read_text()
        supervisor_sbatch = (ROOT / "clariden" / "supervise_production_segment.sbatch").read_text()
        self.assertIn('os.environ.get("WATCH_MAX_SECONDS", "86_000")', watcher)
        self.assertIn("#SBATCH --time=24:00:00", watcher_sbatch)
        self.assertIn("#SBATCH --time=24:00:00", supervisor_sbatch)

    def test_iteration_zero_evaluation_uses_the_canonical_release_checkpoint(self):
        watcher = (ROOT / "production" / "watch_checkpoint_evaluations.py").read_text()
        export = (ROOT / "evaluation" / "checkpoint_eval_export_receipt.py").read_text()
        self.assertIn('checkpoint_directory = "release" if iteration == 0', watcher)
        self.assertIn('checkpoint_name = "release" if args.iteration == 0', export)
        self.assertIn('tracker_value = "release" if args.iteration == 0', export)

    def test_iteration_zero_checkpoint_view_is_a_valid_release_view(self):
        checkpoint_export = load_checkpoint_export()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            release = source / "release"
            release.mkdir(parents=True)
            (release / ".metadata").write_text("metadata\n")
            (release / "common.pt").write_bytes(b"checkpoint")
            output = root / "output"
            args = type(
                "Args",
                (),
                {
                    "source_checkpoint_root": source,
                    "iteration": 0,
                    "output_root": output,
                },
            )()
            checkpoint_export.prepare(args)
            self.assertEqual(
                (output / "source_view" / "latest_checkpointed_iteration.txt").read_text(),
                "release\n",
            )
            self.assertEqual(
                (output / "source_view" / "release").resolve(), release.resolve()
            )
            receipt = json.loads((output / "source_checkpoint_receipt.json").read_text())
            self.assertEqual(receipt["iteration"], 0)
            self.assertEqual(Path(receipt["source_iteration"]), release.resolve())

    def test_extra_evaluation_attempt_is_iteration_scoped_and_receipt_bound(self):
        watcher = (ROOT / "production" / "watch_checkpoint_evaluations.py").read_text()
        self.assertIn('os.environ.get("EVALUATION_ATTEMPT_LIMIT_OVERRIDES"', watcher)
        self.assertIn('os.environ.get("OPERATIONAL_RECOVERY_RECEIPT"', watcher)
        self.assertIn('receipt.get(\n            "evaluation_attempt_limit_overrides"', watcher)
        self.assertIn(
            "attempt_limits.get(iteration, DEFAULT_MAX_ATTEMPTS)", watcher
        )
        self.assertIn("not DEFAULT_MAX_ATTEMPTS < limit <= 5", watcher)

    def test_iteration_zero_intermediate_checkpoint_is_normalized_as_release(self):
        conversion = (
            ROOT / "clariden" / "convert_checkpoint_for_native_greekmmlu.sbatch"
        ).read_text()
        self.assertIn('[[ "$SOURCE_ITERATION" == 0 ]]', conversion)
        self.assertIn('ln -s iter_0000000 "$intermediate_root/release"', conversion)
        self.assertIn(
            "printf 'release\\n' > \"$intermediate_root/latest_checkpointed_iteration.txt\"",
            conversion,
        )

    def test_iteration_zero_uses_receipt_bound_canonical_td_hf(self):
        worker = (ROOT / "clariden" / "run_checkpoint_native_greekmmlu_one.sh").read_text()
        export = (ROOT / "evaluation" / "prepare_initial_checkpoint_hf_export.py").read_text()
        self.assertIn('if [[ "$SOURCE_ITERATION" == 0 ]]', worker)
        self.assertIn("prepare_initial_checkpoint_hf_export.py", worker)
        self.assertIn('"receipt_bound_canonical_td_hf_reference"', export)
        self.assertIn('"hf_roundtrip_model_bitwise_identical": True', export)
        self.assertIn('"rope_scaling": {"rope_type": "default"}', export)

    def test_backlog_accelerator_is_partition_only_and_fail_closed(self):
        text = (
            ROOT / "production" / "accelerate_checkpoint_evaluation_backlog.py"
        ).read_text()
        self.assertIn('"Partition": "normal"', text)
        self.assertIn('"TimeLimit": "00:30:00"', text)
        self.assertIn('"NumNodes": "1-1"', text)
        self.assertIn('"gres/gpu=4"', text)
        self.assertIn('"Partition=debug"', text)
        self.assertIn("max_active_debug_evaluations", text)
        self.assertIn("<= 4", text)
        self.assertNotIn('"scancel"', text)
        self.assertNotIn('"sbatch"', text)

        sbatch = (
            ROOT / "clariden" / "accelerate_checkpoint_evaluation_backlog.sbatch"
        ).read_text()
        self.assertIn('MAX_ACTIVE_DEBUG_EVALUATIONS:-1', sbatch)

    def test_backlog_batch_changes_packing_only(self):
        builder = (
            ROOT / "production" / "build_checkpoint_evaluation_backlog_batches.py"
        ).read_text()
        runner = (
            ROOT / "clariden" / "run_checkpoint_evaluation_backlog_batch.sbatch"
        ).read_text()
        self.assertIn('"scientific_execution_unchanged": True', builder)
        self.assertIn('"wave_manifest_sha256"', builder)
        self.assertIn("#SBATCH --nodes=4", runner)
        self.assertIn("#SBATCH --partition=debug", runner)
        self.assertIn("EVAL_DTYPE=float32", runner)
        self.assertIn("base+=16", runner)
        self.assertIn("run_checkpoint_native_greekmmlu_one.sh", runner)

        controller = (
            ROOT / "clariden" / "run_checkpoint_evaluation_backlog_controller.sbatch"
        ).read_text()
        self.assertIn("--nodes=4 --time=00:22:00", controller)
        self.assertIn("expected between one and five backlog manifests", controller)
        self.assertIn("wait_terminal", controller)
        self.assertIn("EVAL_DTYPE=float32", controller)

    def test_export_gate_records_runtime_parity_and_requires_exact_mapping(self):
        text = (ROOT / "evaluation" / "checkpoint_eval_export_receipt.py").read_text()
        converter = (
            ROOT / "clariden" / "convert_checkpoint_for_native_greekmmlu.sbatch"
        ).read_text()
        verifier = (
            ROOT / "evaluation" / "verify_exact_checkpoint_weight_mapping.py"
        ).read_text()
        self.assertIn('"mean_kl_divergence": 1.0e-4', text)
        self.assertIn('"mean_total_variation": 5.0e-3', text)
        self.assertIn('"mean_top_token_logprob_abs_difference": 1.0e-2', text)
        self.assertIn('"p99_top_token_logprob_abs_difference": 7.5e-2', text)
        self.assertIn('"p999_top_token_logprob_abs_difference": 2.0e-1', text)
        self.assertIn(
            '"parity_gate_policy": "exact_parameter_mapping_with_runtime_diagnostics_v3"',
            text,
        )
        self.assertIn('"all_mapped_parameter_tensors_bit_exact": True', text)
        self.assertIn('"runtime_numerical_sensitivity_detected"', text)
        self.assertIn('"parity_acceptance_path"', text)
        self.assertIn('"bit_exact_parameter_mapping"', text)
        self.assertIn("verify_exact_checkpoint_weight_mapping.py", converter)
        self.assertIn("exact_weight_mapping_receipt.json", converter)
        self.assertIn('"all_source_parameters_covered": True', verifier)
        self.assertIn('"all_hf_tensors_accounted_for": True', verifier)
        self.assertIn('"all_mapped_parameter_tensors_bit_exact": True', verifier)


if __name__ == "__main__":
    unittest.main()
