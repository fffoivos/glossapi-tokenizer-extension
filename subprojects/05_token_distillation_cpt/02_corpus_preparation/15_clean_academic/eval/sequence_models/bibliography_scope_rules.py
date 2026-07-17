"""Dependency-light exact bibliography exclusion scope rules."""

from __future__ import annotations

import re
from typing import Sequence

from .deterministic_structure import _ATX_HEADING, _heading_key


AUXILIARY_SCOPE_HEADINGS = {
    "abbreviations",
    "list of abbreviations",
    "list of figures",
    "list of illustrations",
    "list of tables",
    "related links",
    "related material",
    "related resources",
    "συντομογραφιες",
    "καταλογος συντομογραφιων",
    "καταλογος εικονων",
    "καταλογος πινακων",
    "καταλογος πινακων και προελευση εικονων",
    "καταλογος σχηματων",
    "λιστα επιλεγμενων παραλλαγων",
    "σχετικοι συνδεσμοι",
    "σχετιζομενο υλικο",
    "σχετιζομενα χναρια",
}
BODY_CITATION_SCOPE_HEADINGS = {
    "examples",
    "why",
    "γιατι",
    "παραδειγματα",
    "παρα∆ειγματα",
}
AUXILIARY_SCOPE_PREFIXES = (
    "list of selected variants:",
    "λιστα επιλεγμενων παραλλαγων:",
)
# These headings name a format, not necessarily a non-bibliography semantic
# scope.  They stay negative heading lines but cannot veto the following rows.
AMBIGUOUS_FORMAT_HEADINGS = {
    "abbreviations",
    "list of abbreviations",
    "συντομογραφιες",
    "καταλογος συντομογραφιων",
}
SECTION_VETO_HEADINGS = AUXILIARY_SCOPE_HEADINGS - AMBIGUOUS_FORMAT_HEADINGS


def normalized_scope_heading_key(text: str) -> str:
    key = _heading_key(text)
    return re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", key).strip()


def is_exact_non_bibliography_scope_heading(text: str) -> bool:
    key = normalized_scope_heading_key(text)
    return (
        key in SECTION_VETO_HEADINGS
        or key in BODY_CITATION_SCOPE_HEADINGS
        or any(key.startswith(prefix) for prefix in AUXILIARY_SCOPE_PREFIXES)
    )


def is_persistent_archive_scope_heading(text: str) -> bool:
    key = normalized_scope_heading_key(text)
    return any(key.startswith(prefix) for prefix in AUXILIARY_SCOPE_PREFIXES)


def is_archive_type_subheading(text: str) -> bool:
    key = normalized_scope_heading_key(text)
    return bool(re.match(r"^(?:ατ|at)(?:/atu)?\s+\d", key, re.IGNORECASE))


def auxiliary_scope_mask(texts: Sequence[str]) -> list[bool]:
    """Return the exact active negative scope for an aligned document."""

    result = []
    active_atx_scope = False
    persistent_archive_scope = False
    for text in texts:
        auxiliary_heading = is_exact_non_bibliography_scope_heading(text)
        if _ATX_HEADING.match(text):
            if auxiliary_heading:
                active_atx_scope = True
                persistent_archive_scope = is_persistent_archive_scope_heading(text)
            elif not (
                persistent_archive_scope and is_archive_type_subheading(text)
            ):
                active_atx_scope = False
                persistent_archive_scope = False
        result.append(active_atx_scope or auxiliary_heading)
    return result
