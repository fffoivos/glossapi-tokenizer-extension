from sequence_models.bibliography_deterministic_roles import (
    ROLE_NAMES,
    _role_index,
)


def test_role_index_assigns_one_structural_owner() -> None:
    assert ROLE_NAMES[_role_index(("BIB2_NEGATIVE_FIGURE_CAPTION",))] == "figure_caption"
    assert ROLE_NAMES[_role_index(("BIB2_NEGATIVE_STATISTICAL_TABLE",))] == "table_or_equation"
    assert ROLE_NAMES[_role_index(("BIB2_NEGATIVE_NOTES_HEADING",))] == "negative_section_heading"
    assert ROLE_NAMES[_role_index(("BIB2_NEGATIVE_FOOTNOTE",))] == "footnote"
    assert ROLE_NAMES[_role_index(("BIB2_NEGATIVE_INLINE_CITATION_PROSE",))] == "running_or_enumerated_prose"
