from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset"))

from validate_partition_group import record_identity  # noqa: E402


class PartitionRecordIdentityTests(unittest.TestCase):
    def test_distinct_content_under_same_cluster_id_is_distinct_record(self) -> None:
        first = record_identity("docv2:cluster", "a" * 64)
        second = record_identity("docv2:cluster", "b" * 64)
        self.assertNotEqual(first, second)

    def test_record_identity_rejects_malformed_content_hash(self) -> None:
        with self.assertRaises(ValueError):
            record_identity("docv2:cluster", "short")
