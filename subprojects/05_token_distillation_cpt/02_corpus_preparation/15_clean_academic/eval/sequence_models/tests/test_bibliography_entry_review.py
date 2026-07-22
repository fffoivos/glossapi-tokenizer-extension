from __future__ import annotations

from sequence_models.bibliography_entry_review import select_source_balanced_cases


def _cases(count: int, source: str, offset: int) -> list[dict[str, object]]:
    return [
        {
            "case_id": f"{source}-{index}",
            "source": source,
            "risk_score": float(count - index + offset),
        }
        for index in range(count)
    ]


def test_source_balanced_case_selection() -> None:
    rows = _cases(60, "a", 0) + _cases(60, "b", 100) + _cases(60, "c", 200)
    selected = select_source_balanced_cases(rows, 120)
    counts = {source: sum(row["source"] == source for row in selected) for source in "abc"}
    assert counts == {"a": 40, "b": 40, "c": 40}
    assert [row["review_order"] for row in selected] == list(range(120))


def test_short_source_is_filled_from_remaining_sources() -> None:
    rows = _cases(5, "a", 0) + _cases(80, "b", 100) + _cases(80, "c", 200)
    selected = select_source_balanced_cases(rows, 120)
    assert len(selected) == 120
    assert sum(row["source"] == "a" for row in selected) == 5
