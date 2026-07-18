from __future__ import annotations

import numpy as np
import pytest

from sequence_models.bibliography_deterministic_roles import ROLE_NAMES
from sequence_models.bibliography_evolution_header_role_generation import (
    HEADER_ROLES,
    analyze_document,
    combine_header_roles,
)


def test_text_only_generation_ignores_annotation_fields() -> None:
    base = [
        {"text": "References", "abs_idx": 10, "label": "O"},
        {"text": "Primary Sources", "abs_idx": 11, "label": "BIB"},
        {"text": "# List of Tables", "abs_idx": 12, "label": "BIB"},
    ]
    changed = [
        {**row, "label": "TOC", "target": 999, "original_label": "UNKNOWN"}
        for row in base
    ]
    first = analyze_document(("doc", base))
    second = analyze_document(("doc", changed))
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    roles = combine_header_roles(first[1], first[2])
    assert roles.tolist() == [
        HEADER_ROLES.index("BIB_HEADER"),
        HEADER_ROLES.index("BIB_SUBHEADER"),
        HEADER_ROLES.index("NON_BIB_HEADER"),
    ]


def test_combiner_is_perpendicular_and_fails_on_conflict() -> None:
    bibliography = np.asarray([0, 1, 2, 0], dtype=np.uint8)
    negatives = np.zeros((4, len(ROLE_NAMES)), dtype=np.uint8)
    negatives[3, ROLE_NAMES.index("exact_negative_scope_heading")] = 1
    combined = combine_header_roles(bibliography, negatives)
    assert combined.tolist() == [0, 1, 2, 3]

    negatives[1, ROLE_NAMES.index("exact_negative_scope_heading")] = 1
    with pytest.raises(ValueError, match="conflict"):
        combine_header_roles(bibliography, negatives)


def test_combiner_rejects_misalignment_and_unknown_roles() -> None:
    with pytest.raises(ValueError, match="align"):
        combine_header_roles(
            np.zeros(2, dtype=np.uint8),
            np.zeros((3, len(ROLE_NAMES)), dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="unknown"):
        combine_header_roles(
            np.asarray([4], dtype=np.uint8),
            np.zeros((1, len(ROLE_NAMES)), dtype=np.uint8),
        )
