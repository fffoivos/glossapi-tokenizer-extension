import numpy as np

from sequence_models.bibliography_gap_dataset_presentation import (
    _case,
    _display_indices,
    select_presentation_rows,
)


def _metadata() -> tuple[dict[str, object], ...]:
    rows = []
    for source in ("greek_phd", "kallipos", "openarchives"):
        for target in (0, 1):
            for index in range(12):
                rows.append({
                    "source": source,
                    "fold": index % 5,
                    "model_line_count": (1, 4, 12, 40)[index % 4],
                    "variant_id": f"{source}:{target}:{index}",
                })
    return tuple(rows)


def test_presentation_selection_is_balanced_and_deterministic() -> None:
    metadata = _metadata()
    targets = np.asarray([
        int(str(row["variant_id"]).split(":")[1]) for row in metadata
    ], dtype=np.uint8)
    rows = np.arange(len(metadata), dtype=np.int64)
    first = select_presentation_rows(
        metadata, targets, rows, per_source_label=5, seed=17
    )
    second = select_presentation_rows(
        metadata, targets, rows, per_source_label=5, seed=17
    )
    assert np.array_equal(first, second)
    assert len(first) == 3 * 2 * 5
    observed = {
        (str(metadata[index]["source"]), int(targets[index]))
        for index in first
    }
    assert len(observed) == 6
    assert all(
        sum(
            str(metadata[index]["source"]) == source and int(targets[index]) == target
            for index in first
        ) == 5
        for source, target in observed
    )


def test_long_gap_display_keeps_endpoints_and_marks_omission() -> None:
    indices = _display_indices(
        5, 56, line_count=80, context=2, maximum_gap=10
    )
    assert indices[:3] == [3, 4, 5]
    assert indices[-3:] == [56, 57, 58]
    assert indices.count(None) == 1
    assert [value for value in indices if value is not None and 5 < value < 56] == [
        6, 7, 8, 9, 10, 51, 52, 53, 54, 55
    ]


def test_display_context_is_clipped_at_document_end() -> None:
    assert _display_indices(
        2, 4, line_count=5, context=5, maximum_gap=10
    ) == [0, 1, 2, 3, 4]


def test_case_marks_only_the_gap_as_training_span() -> None:
    raw_lines = [
        {"abs_idx": index * 2, "text": f"line {index}", "label": "BIB" if 2 <= index <= 5 else "O"}
        for index in range(8)
    ]
    row = {
        "variant_id": "v",
        "boundary_group_id": "b",
        "document_id": "d",
        "work_id": "w",
        "source": "greek_phd",
        "fold": 0,
        "left_local_index": 2,
        "right_local_index": 5,
        "left_abs_idx": 4,
        "right_abs_idx": 10,
        "regime": "threshold_ladder",
        "generation_thresholds": [0.4],
        "model_line_count": 2,
        "entry_mean": 0.1,
        "entry_max": 0.2,
    }
    case = _case(
        7, row, 1, {"lines": raw_lines}, context=1, maximum_gap=10
    )
    roles = [line["role"] for line in case["lines"]]
    assert roles == [
        "context", "left_anchor", "training_span", "training_span",
        "right_anchor", "context",
    ]
    assert case["gap_silver_labels"] == {"BIB": 2}
    assert case["training_label"] == "BIB / CONNECT"
