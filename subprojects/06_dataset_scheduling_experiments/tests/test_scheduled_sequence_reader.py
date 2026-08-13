from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "training" / "scheduled_sequence_reader.py"
SPEC = importlib.util.spec_from_file_location("scheduled_sequence_reader", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
import sys

sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_receipt(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


class ScheduledSequenceReaderTests(unittest.TestCase):
    def _fixture(self, root: Path, *, scheduled_active: int = 4096) -> Path:
        payload = np.arange(4097, dtype=np.int32)
        binary = root / "bucket.bin"
        active = root / "bucket.active.u16"
        payload.tofile(binary)
        np.asarray([4096], dtype=np.uint16).tofile(active)
        bucket_manifest = root / "bucket.manifest.json"
        bucket_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "apertus_mini_fixed_sequence_bucket_v1",
                    "status": "completed",
                    "pool_code": 2,
                    "bucket": 3,
                    "pad_token_id": 10,
                    "sequence_count": 1,
                    "stored_tokens_per_sequence": 4097,
                    "sequence_length": 4096,
                    "outputs": {
                        "bin": file_receipt(binary),
                        "active_counts": file_receipt(active),
                    },
                }
            )
            + "\n"
        )
        packed = root / "packed.json"
        packed.write_text(
            json.dumps(
                {
                    "schema_version": "apertus_mini_packed_sequence_corpus_v1",
                    "status": "completed",
                    "packing_task_manifests": [
                        {"manifest_path": str(bucket_manifest), "pool": "foreign", "bucket": 3}
                    ],
                }
            )
            + "\n"
        )
        sequence_id = (2 << 62) | (3 << 55)
        ids = root / "D0.ids"
        counts = root / "D0.active"
        np.asarray([sequence_id, MODULE.FILLER_ID], dtype=np.uint64).tofile(ids)
        np.asarray([scheduled_active, 0], dtype=np.uint16).tofile(counts)
        arm = {
            "arm_id": "D0_mixed",
            "training_slots": 2,
            "sequence_ids": file_receipt(ids),
            "active_tokens": file_receipt(counts),
        }
        schedule = root / "schedule_manifest.json"
        schedule.write_text(
            json.dumps(
                {
                    "schema_version": "apertus_mini_five_data_order_schedules_v1",
                    "status": "completed",
                    "common_contract": {
                        "same_exact_sequence_multiset": True,
                        "same_replay_sequence_ids_at_same_global_positions": True,
                    },
                    "packed_corpus_receipt": {"path": str(packed), "sha256": sha(packed)},
                    "arms": [arm],
                }
            )
            + "\n"
        )
        return schedule

    def test_resolves_stable_sequence_id_to_exact_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schedule = self._fixture(Path(tmp))
            reader = MODULE.ScheduledSequenceReader(schedule, "D0_mixed")
            payload = reader[0]
            self.assertFalse(payload.filler)
            self.assertEqual(payload.active_tokens, 4096)
            np.testing.assert_array_equal(payload.tokens, np.arange(4097, dtype=np.int64))

    def test_filler_is_loss_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schedule = self._fixture(Path(tmp))
            payload = MODULE.ScheduledSequenceReader(schedule, "D0_mixed")[1]
            self.assertTrue(payload.filler)
            self.assertEqual(payload.active_tokens, 0)
            self.assertEqual(int(payload.tokens.sum()), 0)

    def test_accepts_terminal_active_cap_without_rewriting_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schedule = self._fixture(Path(tmp), scheduled_active=4095)
            reader = MODULE.ScheduledSequenceReader(schedule, "D0_mixed")
            payload = reader[0]
            self.assertEqual(payload.active_tokens, 4095)
            np.testing.assert_array_equal(payload.tokens, np.arange(4097, dtype=np.int64))

    def test_rejects_zero_active_non_filler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schedule = self._fixture(Path(tmp), scheduled_active=0)
            reader = MODULE.ScheduledSequenceReader(schedule, "D0_mixed")
            with self.assertRaisesRegex(ValueError, "invalid scheduled active-token cap"):
                reader[0]

    def test_rejects_active_cap_larger_than_packed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schedule = self._fixture(Path(tmp), scheduled_active=4097)
            reader = MODULE.ScheduledSequenceReader(schedule, "D0_mixed")
            with self.assertRaisesRegex(ValueError, "invalid scheduled active-token cap"):
                reader[0]

    def test_stationary_schedule_reads_multiple_packed_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = self._fixture(root)
            legacy_value = json.loads(legacy.read_text())
            packed_one = root / "packed.json"
            binary_two = root / "bucket_two.bin"
            active_two = root / "bucket_two.active.u16"
            (np.arange(4097, dtype=np.int32) + 10000).tofile(binary_two)
            np.asarray([4000], dtype=np.uint16).tofile(active_two)
            bucket_two = root / "bucket_two.manifest.json"
            bucket_two.write_text(
                json.dumps(
                    {
                        "schema_version": "apertus_fixed_sequence_bucket_v1",
                        "status": "completed",
                        "pool_code": 3,
                        "bucket": 4,
                        "pad_token_id": 10,
                        "sequence_count": 1,
                        "stored_tokens_per_sequence": 4097,
                        "sequence_length": 4096,
                        "outputs": {
                            "bin": file_receipt(binary_two),
                            "active_counts": file_receipt(active_two),
                        },
                    }
                )
                + "\n"
            )
            packed_two = root / "packed_two.json"
            packed_two.write_text(
                json.dumps(
                    {
                        "schema_version": "apertus_packed_sequence_corpus_v1",
                        "status": "completed",
                        "packing_task_manifests": [
                            {
                                "manifest_path": str(bucket_two),
                                "manifest_sha256": sha(bucket_two),
                                "pool": "old_greek_replay",
                                "bucket": 4,
                            }
                        ],
                    }
                )
                + "\n"
            )
            ids = root / "stationary.ids"
            active = root / "stationary.active"
            sequence_one = (2 << 62) | (3 << 55)
            sequence_two = (3 << 62) | (4 << 55)
            np.asarray([sequence_one, sequence_two], dtype=np.uint64).tofile(ids)
            np.asarray([4095, 3999], dtype=np.uint16).tofile(active)
            stationary = root / "stationary.json"
            stationary.write_text(
                json.dumps(
                    {
                        "schema_version": "apertus_cpt_stationary_replay_schedule_v1",
                        "status": "completed",
                        "schedule_id": "stationary",
                        "sequence_count": 2,
                        "sequence_ids": file_receipt(ids),
                        "active_tokens_per_sequence": file_receipt(active),
                        "packed_corpus_receipts": [
                            {"path": str(packed_one), "sha256": sha(packed_one)},
                            {"path": str(packed_two), "sha256": sha(packed_two)},
                        ],
                        "checks": {
                            "every_selected_sequence_scheduled_once": True,
                            "payload_not_rewritten": True,
                        },
                    }
                )
                + "\n"
            )
            self.assertIn("packed_corpus_receipt", legacy_value)
            reader = MODULE.ScheduledSequenceReader(stationary, "stationary")
            self.assertEqual(reader[0].active_tokens, 4095)
            self.assertEqual(reader[1].active_tokens, 3999)
            self.assertEqual(int(reader[1].tokens[0]), 10000)


if __name__ == "__main__":
    unittest.main()
