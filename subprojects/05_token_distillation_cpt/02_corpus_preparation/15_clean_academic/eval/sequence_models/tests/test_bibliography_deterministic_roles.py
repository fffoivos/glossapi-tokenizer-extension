from sequence_models.bibliography_deterministic_roles import (
    ROLE_NAMES,
    _analyze_document,
    _role_index,
)


def test_role_index_assigns_one_structural_owner() -> None:
    assert ROLE_NAMES[_role_index(("BIB2_NEGATIVE_FIGURE_CAPTION",))] == "figure_caption"
    assert ROLE_NAMES[_role_index(("BIB2_NEGATIVE_STATISTICAL_TABLE",))] == "table_or_equation"
    assert ROLE_NAMES[_role_index(("BIB2_NEGATIVE_NOTES_HEADING",))] == "exact_negative_scope_heading"
    assert ROLE_NAMES[_role_index(("BIB2_NEGATIVE_NONSTRUCTURAL_MARKDOWN_HEADING",))] == "generic_markdown_heading"
    assert ROLE_NAMES[_role_index(("BIB2_NEGATIVE_FOOTNOTE",))] == "footnote"
    assert ROLE_NAMES[_role_index(("BIB2_NEGATIVE_INLINE_CITATION_PROSE",))] == "running_or_enumerated_prose"


def test_auxiliary_headings_are_exact_scope_not_generic_markdown() -> None:
    _, roles, _ = _analyze_document(
        (
            "doc",
            [
                {"text": "## List of figures"},
                {"text": "## ΣΧΕΤΙΖΟΜΕΝΑ ΧΝΑΡΙΑ"},
                {"text": "## Unknown bibliography subdivision"},
            ],
        )
    )
    exact = ROLE_NAMES.index("exact_negative_scope_heading")
    generic = ROLE_NAMES.index("generic_markdown_heading")
    assert roles[:, exact].tolist() == [1, 1, 0]
    assert roles[:, generic].tolist() == [0, 0, 1]
