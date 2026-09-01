#!/usr/bin/env python3
"""Regression checks for authoritative source-validation attempt selection."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Segment1SourceAuthorityTests(unittest.TestCase):
    def test_audit_selects_attempts_around_an_explicit_boundary(self):
        audit = (
            ROOT / "evaluation" / "audit_segment_source_validation.py"
        ).read_text()
        self.assertIn('"--authoritative-attempt-boundary"', audit)
        self.assertIn('"--post-boundary-attempt"', audit)
        self.assertIn('"attempt_authority"', audit)
        self.assertIn('"--attempt-authority-through"', audit)
        self.assertIn('"inclusive_upper_bound_attempts"', audit)

    def test_recovery_supervisor_is_bundle_and_boundary_bound(self):
        supervisor = (
            ROOT / "production" / "supervise_production_segment.py"
        ).read_text()
        wrapper = (
            ROOT / "clariden" / "supervise_production_segment.sbatch"
        ).read_text()
        self.assertIn("SOURCE_VALIDATION_ATTEMPT_BOUNDARY", supervisor)
        self.assertIn("RECOVERY_CONTROLLER_BUNDLE_RECEIPT", supervisor)
        self.assertIn("RECOVERY_CONTROLLER_BUNDLE", wrapper)
        self.assertIn("SOURCE_VALIDATION_ATTEMPT_AUTHORITY", supervisor)
        self.assertIn('re.split(r"[;,]", authority)', supervisor)

    def test_core_trajectory_uses_the_same_attempt_authority(self):
        collector = (
            ROOT / "evaluation" / "collect_validation_trajectory.py"
        ).read_text()
        wrapper = (
            ROOT / "clariden" / "finalize_core_campaign_evidence.sbatch"
        ).read_text()
        self.assertIn('"--attempt-authority-through"', collector)
        self.assertIn('"attempt_authority"', collector)
        self.assertIn("SOURCE_VALIDATION_ATTEMPT_AUTHORITY", wrapper)


if __name__ == "__main__":
    unittest.main()
