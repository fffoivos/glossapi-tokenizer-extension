#!/usr/bin/env python3
"""Unit tests for deterministic native-Greek evaluation sharding."""

from __future__ import annotations

import unittest

from run_checkpoint_suite import select_frozen_rows


class FrozenRowShardingTest(unittest.TestCase):
    def test_two_shards_are_disjoint_and_complete(self) -> None:
        rows = [
            {"benchmark": "wic", "example_id": f"w{i}"} for i in range(11)
        ] + [{"benchmark": "other", "example_id": f"o{i}"} for i in range(4)]
        left = select_frozen_rows(rows, {"wic"}, 0, 2, 0)["wic"]
        right = select_frozen_rows(rows, {"wic"}, 1, 2, 0)["wic"]
        left_ids = {row["example_id"] for row in left}
        right_ids = {row["example_id"] for row in right}
        self.assertFalse(left_ids & right_ids)
        self.assertEqual(left_ids | right_ids, {f"w{i}" for i in range(11)})
        self.assertEqual((len(left), len(right)), (6, 5))

    def test_limit_is_applied_within_each_selected_shard(self) -> None:
        rows = [{"benchmark": "wic", "example_id": f"w{i}"} for i in range(10)]
        selected = select_frozen_rows(rows, {"wic"}, 1, 2, 2)
        self.assertEqual([row["example_id"] for row in selected["wic"]], ["w1", "w3"])

    def test_ten_shards_cover_rows_once(self) -> None:
        rows = [{"benchmark": "wsd", "example_id": f"d{i}"} for i in range(103)]
        shard_ids = [
            {
                row["example_id"]
                for row in select_frozen_rows(rows, {"wsd"}, index, 10, 0)["wsd"]
            }
            for index in range(10)
        ]
        self.assertEqual(sum(len(ids) for ids in shard_ids), 103)
        self.assertEqual(set().union(*shard_ids), {f"d{i}" for i in range(103)})
        for index, ids in enumerate(shard_ids):
            self.assertTrue(all(int(example_id[1:]) % 10 == index for example_id in ids))


if __name__ == "__main__":
    unittest.main()
