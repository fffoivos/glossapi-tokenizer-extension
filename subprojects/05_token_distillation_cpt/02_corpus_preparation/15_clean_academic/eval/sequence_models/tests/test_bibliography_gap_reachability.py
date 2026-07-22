import numpy as np

from sequence_models.bibliography_gap_reachability import classify_oracle_misses


def test_reachability_reasons_partition_missed_gold_lines() -> None:
    gold = np.asarray([True, True, True, False, True, True, True, True, True])
    oracle = np.asarray([False, True, False, False, False, False, True, True, False])
    seeds = np.asarray([False, True, False, False, False, False, True, True, False])
    reason = classify_oracle_misses(gold, oracle, seeds, np.arange(9))
    assert reason.tolist() == [1, -1, 1, -1, 2, 2, -1, -1, 3]


def test_internal_miss_between_two_seeds_is_distinct_from_edges() -> None:
    gold = np.ones(5, dtype=bool)
    oracle = np.asarray([False, True, False, True, False])
    seeds = np.asarray([False, True, False, True, False])
    reason = classify_oracle_misses(gold, oracle, seeds, np.arange(5))
    assert reason.tolist() == [2, -1, 4, -1, 3]


def test_physical_break_creates_separate_gold_blocks() -> None:
    gold = np.ones(4, dtype=bool)
    oracle = np.zeros(4, dtype=bool)
    seeds = np.asarray([True, True, False, False])
    reason = classify_oracle_misses(gold, oracle, seeds, np.asarray([0, 1, 100, 101]))
    assert reason.tolist() == [4, 4, 0, 0]
