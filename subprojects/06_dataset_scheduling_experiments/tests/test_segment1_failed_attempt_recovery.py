#!/usr/bin/env python3
"""Regression checks for the Segment-1 final-microbatch recovery."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Segment1FailedAttemptRecoveryTests(unittest.TestCase):
    def test_stratified_metric_keys_are_present_in_every_microbatch(self):
        text = (ROOT / "training" / "pretrain_scheduled_gpt.py").read_text()
        self.assertIn('reporting["base-token target loss"] =', text)
        self.assertIn('reporting["added-token target loss"] =', text)
        self.assertNotIn('if float(reduced[1]) > 0:', text)
        self.assertNotIn('if float(reduced[3]) > 0:', text)

    def test_pad_only_microbatch_uses_safe_backward_denominator(self):
        text = (ROOT / "training" / "pretrain_scheduled_gpt.py").read_text()
        self.assertIn(
            "backward_num_tokens = torch.clamp_min(local_num_tokens, 1)", text
        )
        self.assertIn("return loss, backward_num_tokens, reporting", text)
        self.assertNotIn("--check-for-nan-in-loss-and-grad", (
            ROOT / "clariden" / "run_production_arm_segment.sh"
        ).read_text())

    def test_failed_attempt_checkpoint_freeze_is_explicit_and_receipted(self):
        freezer = (ROOT / "production" / "freeze_segment_checkpoint.py").read_text()
        wrapper = (ROOT / "clariden" / "freeze_segment_checkpoint.sbatch").read_text()
        self.assertIn('"--failed-attempt-recovery"', freezer)
        self.assertIn('"failed_attempt_recovery": args.failed_attempt_recovery', freezer)
        self.assertIn('"segment_state_sha256"', freezer)
        self.assertIn('FAILED_ATTEMPT_RECOVERY', wrapper)
        self.assertIn('"--load-view-root"', freezer)
        self.assertIn('"latest_checkpointed_iteration.txt"', freezer)
        self.assertIn('"checkpoint_source_root"', freezer)
        self.assertIn("LOAD_VIEW_ROOT", wrapper)

    def test_zero_exit_numerical_failure_can_freeze_an_earlier_clean_checkpoint(self):
        freezer = (ROOT / "production" / "freeze_segment_checkpoint.py").read_text()
        wrapper = (ROOT / "clariden" / "freeze_segment_checkpoint.sbatch").read_text()
        self.assertIn('"--numerical-failure-recovery"', freezer)
        self.assertIn('"numerical_failure_recovery": args.numerical_failure_recovery', freezer)
        self.assertIn("NUMERICAL_FAILURE_RECOVERY", wrapper)

    def test_recovery_runtime_bundle_is_verified_before_gpu_launch(self):
        preflight = (ROOT / "production" / "preflight_segment.py").read_text()
        launcher = (ROOT / "clariden" / "train_five_arm_segment.sbatch").read_text()
        arm_launcher = (
            ROOT / "clariden" / "run_production_arm_segment.sh"
        ).read_text()
        self.assertIn('"--runtime-scientific-bundle-receipt"', preflight)
        self.assertIn("verify_code_bundle_receipt(", preflight)
        self.assertIn("RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT", launcher)
        self.assertIn(
            'preflight.get("runtime_scientific_bundle") or assets["scientific_bundle"]',
            arm_launcher,
        )
        self.assertIn("resume checkpoint marker drift", preflight)
        self.assertIn("resume checkpoint view target drift", preflight)

    def test_supervisor_preserves_verified_training_bundle_on_infra_retry(self):
        supervisor = (
            ROOT / "production" / "supervise_production_segment.py"
        ).read_text()
        if "resolve_evaluation_runtime" not in supervisor:
            self.skipTest("legacy training bundle does not provide the evaluation supervisor")
        self.assertIn("TRAINING_SCIENTIFIC_BUNDLE", supervisor)
        self.assertIn("TRAINING_SCIENTIFIC_BUNDLE_RECEIPT", supervisor)
        self.assertIn("verify_code_bundle_receipt(", supervisor)
        self.assertIn("RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT=", supervisor)

    def test_source_validation_attempt_authority_is_explicit(self):
        audit = (
            ROOT / "evaluation" / "audit_segment_source_validation.py"
        ).read_text()
        supervisor = (
            ROOT / "production" / "supervise_production_segment.py"
        ).read_text()
        wrapper = (
            ROOT / "clariden" / "supervise_production_segment.sbatch"
        ).read_text()
        self.assertIn('"--authoritative-attempt-boundary"', audit)
        self.assertIn('"post_boundary_attempt"', audit)
        self.assertIn('"attempt_authority"', audit)
        self.assertIn("SOURCE_VALIDATION_ATTEMPT_BOUNDARY", supervisor)
        self.assertIn("RECOVERY_CONTROLLER_BUNDLE_RECEIPT", supervisor)
        self.assertIn("RECOVERY_CONTROLLER_BUNDLE", wrapper)
        self.assertIn('"--attempt-authority-through"', audit)
        self.assertIn("SOURCE_VALIDATION_ATTEMPT_AUTHORITY", supervisor)


if __name__ == "__main__":
    unittest.main()
