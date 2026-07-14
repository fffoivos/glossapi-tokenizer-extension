from sequence_models.bibliography_component_false_review import _sample_indices


def test_sample_indices_cover_start_middle_and_end_with_context() -> None:
    assert _sample_indices(10, 30, 50, 1) == [9, 10, 11, 19, 20, 21, 29, 30, 31]
