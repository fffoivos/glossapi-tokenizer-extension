import numpy as np

from sequence_models.bibliography_gap_candidates import (
    CandidateContext,
    GapCandidate,
    cap_nonbib_by_work,
    enumerate_component_gaps,
    normalize_boundary_weights,
    sample_nonbib_spans,
    seed_components,
)


CONTEXT = CandidateContext(0, "d1", "w1", "greek_phd", 0)


def test_seed_components_compress_exactly_adjacent_predictions() -> None:
    probability = np.asarray([0.8, 0.9, 0.0, 0.7, 0.8], dtype=np.float32)
    components = seed_components(
        probability,
        np.full(5, 100),
        np.arange(5),
        np.zeros(5, dtype=bool),
        threshold=0.25,
        seed_length_limit=330,
    )
    assert components == [(0, 1), (3, 4)]


def test_false_seed_component_creates_a_real_negative_gap() -> None:
    candidates, _ = enumerate_component_gaps(
        context=CONTEXT,
        entry_probability=np.asarray([0.8, 0.0, 0.7], dtype=np.float32),
        char_lengths=np.full(3, 100),
        abs_indices=np.arange(3),
        gold_bib=np.asarray([True, False, False]),
        typed_heading_barrier=np.zeros(3, dtype=bool),
        exact_scope=np.zeros(3, dtype=bool),
        thresholds=(0.25,),
    )
    assert len(candidates) == 1
    assert candidates[0].target == 0
    assert candidates[0].regime == "deployment_real"
    assert candidates[0].gap_length == 1


def test_higher_threshold_fragments_one_gold_block_into_a_positive_gap() -> None:
    candidates, _ = enumerate_component_gaps(
        context=CONTEXT,
        entry_probability=np.asarray([0.8, 0.2, 0.7], dtype=np.float32),
        char_lengths=np.full(3, 100),
        abs_indices=np.arange(3),
        gold_bib=np.ones(3, dtype=bool),
        typed_heading_barrier=np.zeros(3, dtype=bool),
        exact_scope=np.zeros(3, dtype=bool),
        thresholds=(0.60,),
    )
    assert len(candidates) == 1
    assert candidates[0].target == 1
    assert candidates[0].regime == "threshold_ladder"


def test_threshold_duplicates_are_one_boundary_with_all_thresholds() -> None:
    candidates, _ = enumerate_component_gaps(
        context=CONTEXT,
        entry_probability=np.asarray([0.8, 0.0, 0.7], dtype=np.float32),
        char_lengths=np.full(3, 100),
        abs_indices=np.arange(3),
        gold_bib=np.ones(3, dtype=bool),
        typed_heading_barrier=np.zeros(3, dtype=bool),
        exact_scope=np.zeros(3, dtype=bool),
        thresholds=(0.10, 0.25, 0.60),
    )
    assert len(candidates) == 1
    assert candidates[0].generation_thresholds == (0.1, 0.25, 0.6)
    assert candidates[0].regime == "deployment_real"


def test_typed_heading_negative_becomes_two_weighted_miss_variants() -> None:
    candidates, _ = enumerate_component_gaps(
        context=CONTEXT,
        entry_probability=np.asarray([0.8, 0.0, 0.0, 0.7], dtype=np.float32),
        char_lengths=np.full(4, 100),
        abs_indices=np.arange(4),
        gold_bib=np.asarray([True, False, False, True]),
        typed_heading_barrier=np.asarray([False, True, False, False]),
        exact_scope=np.zeros(4, dtype=bool),
        thresholds=(0.25,),
    )
    assert {row.synthetic_kind for row in candidates} == {
        "heading_probability_masked",
        "heading_line_removed",
    }
    assert sum(row.base_weight for row in normalize_boundary_weights(candidates)) == 0.5
    removed = next(row for row in candidates if row.synthetic_kind == "heading_line_removed")
    assert removed.removed_offsets == (1,)


def test_exact_scope_is_never_ablated() -> None:
    candidates, _ = enumerate_component_gaps(
        context=CONTEXT,
        entry_probability=np.asarray([0.8, 0.0, 0.7], dtype=np.float32),
        char_lengths=np.full(3, 100),
        abs_indices=np.arange(3),
        gold_bib=np.asarray([True, False, True]),
        typed_heading_barrier=np.zeros(3, dtype=bool),
        exact_scope=np.asarray([False, True, False]),
        thresholds=(0.25,),
    )
    assert candidates == []


def test_nonbib_spans_are_interior_and_match_requested_length() -> None:
    rows = sample_nonbib_spans(
        context=CONTEXT,
        entry_probability=np.asarray([0.0, 0.0, 0.4, 0.1, 0.0, 0.0]),
        abs_indices=np.arange(6),
        gold_bib=np.zeros(6, dtype=bool),
        typed_heading_barrier=np.zeros(6, dtype=bool),
        exact_scope=np.zeros(6, dtype=bool),
        target_lengths=(2,),
    )
    assert {row.regime for row in rows} == {"hard_nonbib", "easy_nonbib"}
    assert all(row.gap_length == 2 and row.virtual_boundaries for row in rows)
    assert next(row for row in rows if row.regime == "hard_nonbib").entry_max >= next(
        row for row in rows if row.regime == "easy_nonbib"
    ).entry_max


def test_random_span_cap_is_per_work_and_kind() -> None:
    rows = [
        GapCandidate(CandidateContext(i, f"d{i}", "same-work", "oa", 0), 0, 2, 0, kind, (),
                     synthetic_kind=f"{kind}_span", virtual_boundaries=True)
        for i in range(3)
        for kind in ("hard_nonbib", "easy_nonbib")
    ]
    selected = cap_nonbib_by_work(rows)
    assert len(selected) == 2
    assert {row.regime for row in selected} == {"hard_nonbib", "easy_nonbib"}
