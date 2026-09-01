#!/usr/bin/env python3
"""Small regression checks for release-index invariants."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from assemble_trajectory_release_metadata import (
    CHECKPOINTS,
    NATIVE_COUNTS,
    PRIVATE_FULL_MIX_REPOSITORY_ID,
    PUBLIC_TRAIN_REPOSITORY_ID,
    frozen_dataset_upload,
    parse_metrics,
)


class ReleaseIndexTest(unittest.TestCase):
    def test_checkpoint_branch_mapping_is_complete_and_unique(self) -> None:
        self.assertEqual(len(CHECKPOINTS), 18)
        self.assertEqual(len({row[0] for row in CHECKPOINTS}), 18)
        self.assertEqual(len({row[2] for row in CHECKPOINTS}), 18)
        self.assertEqual(CHECKPOINTS[-1][2], "main")

    def test_parse_metrics_selects_only_aggregate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            fields = ["benchmark", "subject", "n", "accuracy", "choice_nll", "correct_answer_bpb", "balanced_accuracy", "binary_macro_f1"]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for benchmark, count in NATIVE_COUNTS.items():
                    writer.writerow({"benchmark": benchmark, "subject": "__all__", "n": count, "accuracy": 0.5, "choice_nll": 1.0, "correct_answer_bpb": 0.2, "balanced_accuracy": "", "binary_macro_f1": ""})
                    writer.writerow({"benchmark": benchmark, "subject": "sub", "n": 1, "accuracy": 0.0, "choice_nll": 3.0, "correct_answer_bpb": 0.9, "balanced_accuracy": "", "binary_macro_f1": ""})
                writer.writerow({"benchmark": "oyxoy_nli_exact_set", "subject": "__all__", "n": 1748, "accuracy": 0.0, "choice_nll": "", "correct_answer_bpb": "", "balanced_accuracy": "", "binary_macro_f1": ""})
            parsed = parse_metrics(path)
            self.assertEqual(set(parsed), set(NATIVE_COUNTS))
            self.assertEqual(parsed["demosqa"]["accuracy"], 0.5)

    def test_frozen_dataset_upload_binds_visibility_and_immutable_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.json"
            revision = "a" * 40
            path.write_text(json.dumps({
                "schema_version": "apertus_full8_frozen_dataset_hf_upload_v1",
                "status": "completed",
                "repo_id": PUBLIC_TRAIN_REPOSITORY_ID,
                "revision": revision,
                "training_must_pin_revision": revision,
                "private": False,
            }), encoding="utf-8")
            bound = frozen_dataset_upload(path, repo_id=PUBLIC_TRAIN_REPOSITORY_ID, private=False)
            self.assertEqual(bound["revision"], revision)
            with self.assertRaises(ValueError):
                frozen_dataset_upload(path, repo_id=PRIVATE_FULL_MIX_REPOSITORY_ID, private=True)


if __name__ == "__main__":
    unittest.main()
