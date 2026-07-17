import numpy as np

from sequence_models.bibliography_gap_connect_table import (
    GapPair,
    gap_length_bucket,
    select_pairs,
    typed_heading_barrier,
)


def _pair(index: int, target: int, *, source: str = "oa", fold: int = 0) -> GapPair:
    return GapPair(
        document_index=index,
        document_id=f"d{index}",
        work_id=f"w{index}",
        source=source,
        fold=fold,
        left=0,
        right=4,
        target=target,
        length_bucket="3-5",
    )


def test_gap_length_buckets_cover_variable_sequences() -> None:
    assert gap_length_bucket(1) == "1"
    assert gap_length_bucket(4) == "3-5"
    assert gap_length_bucket(61) == "61-120"
    assert gap_length_bucket(201) == ">200"


def test_pair_sampling_keeps_negatives_and_caps_positives_deterministically() -> None:
    pairs = [_pair(0, 0), *(_pair(index, 1) for index in range(1, 20))]
    first = select_pairs(
        pairs, positive_to_negative=3, minimum_positive_per_group=2, seed=7,
    )
    second = select_pairs(
        pairs, positive_to_negative=3, minimum_positive_per_group=2, seed=7,
    )
    assert first == second
    assert sum(pair.target == 0 for pair in first) == 1
    assert sum(pair.target == 1 for pair in first) == 3


def test_only_main_and_non_bibliography_headings_are_barriers() -> None:
    probability = np.asarray([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.3, 0.2, 0.4],
    ], dtype=np.float32)
    assert typed_heading_barrier(probability, threshold=0.5).tolist() == [True, False, True, False]
