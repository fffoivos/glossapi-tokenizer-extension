"""Adversarial and metamorphic checks for deterministic structure detection.

The examples in this module are synthetic.  They deliberately exercise only
the public, dependency-free rule surface and never read STRUCT2K (including
its sealed/test partitions) or any corpus artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.deterministic_structure import (  # noqa: E402
    BibRole,
    StructureKind,
    TocRole,
    analyze_bib_line,
    analyze_bib_lines,
    analyze_toc_line,
    analyze_toc_lines,
    decode_bib_blocks,
    decode_toc_blocks,
    detect_structure,
)


TOC_ENTRIES = [
    "1. Εισαγωγή ........ 1",
    "2. Θεωρία ........ 7",
    "3. Μέθοδος ........ 15",
    "4. Αποτελέσματα ........ 29",
]

AUTHOR_YEAR_ENTRIES = [
    "Smith, J. (2018). First title. London: Press.",
    "Brown, K. (2019). Second title. London: Press.",
    "Jones, P. (2020). Third title. London: Press.",
    "White, R. (2021). Fourth title. London: Press.",
]


def _only_span(lines: list[str], kind: StructureKind):
    decision = detect_structure(lines, n_physical_lines=max(20, len(lines)))
    matching = [span for span in decision.spans if span.kind == kind]
    assert len(matching) == 1
    return matching[0]


def test_toc_typed_line_insertion_and_deletion_preserve_anchor_membership() -> None:
    baseline = _only_span(["ΠΕΡΙΕΧΟΜΕΝΑ", *TOC_ENTRIES[:2]], StructureKind.TOC)

    inserted_lines = [
        "ΠΕΡΙΕΧΟΜΕΝΑ",
        TOC_ENTRIES[0],
        "ΜΕΡΟΣ ΠΡΩΤΟ",
        TOC_ENTRIES[1],
    ]
    inserted = _only_span(inserted_lines, StructureKind.TOC)
    assert inserted.bridged_line_indices == (2,)
    assert tuple(
        inserted_lines[index] for index in inserted.supporting_line_indices
    ) == (
        "ΠΕΡΙΕΧΟΜΕΝΑ",
        *TOC_ENTRIES[:2],
    )

    restored_lines = inserted_lines[:2] + inserted_lines[3:]
    restored = _only_span(restored_lines, StructureKind.TOC)
    assert restored.supporting_line_indices == baseline.supporting_line_indices
    assert restored.bridged_line_indices == ()


def test_toc_gap_budget_is_bounded_by_both_line_count_and_tokens() -> None:
    two_short_gaps = analyze_toc_lines(
        [
            "ΠΕΡΙΕΧΟΜΕΝΑ",
            TOC_ENTRIES[0],
            "ΜΕΡΟΣ ΠΡΩΤΟ",
            "ΚΥΡΙΑ ΕΝΟΤΗΤΑ",
            TOC_ENTRIES[1],
        ]
    )
    accepted = decode_toc_blocks(two_short_gaps, n_physical_lines=20)
    assert len(accepted) == 1
    assert accepted[0].bridged_line_indices == (2, 3)

    three_short_gaps = analyze_toc_lines(
        [
            "ΠΕΡΙΕΧΟΜΕΝΑ",
            TOC_ENTRIES[0],
            "ΜΕΡΟΣ ΠΡΩΤΟ",
            "ΚΥΡΙΑ ΕΝΟΤΗΤΑ",
            "ΔΕΥΤΕΡΗ ΥΠΟΕΝΟΤΗΤΑ",
            TOC_ENTRIES[1],
        ]
    )
    assert decode_toc_blocks(three_short_gaps, n_physical_lines=20) == ()

    oversized_gap = " ".join(["ΛΕΞΗ"] * 41) + " ........"
    assert analyze_toc_line(oversized_gap).role == TocRole.POSSIBLE_CONTINUATION
    over_token_budget = analyze_toc_lines(
        ["ΠΕΡΙΕΧΟΜΕΝΑ", TOC_ENTRIES[0], oversized_gap, oversized_gap, TOC_ENTRIES[1]]
    )
    assert decode_toc_blocks(over_token_budget, n_physical_lines=20) == ()


def test_hard_barrier_insertion_splits_independent_toc_blocks() -> None:
    body_heading = "## ΚΕΦΑΛΑΙΟ 2"
    assert analyze_toc_line(body_heading).hard_negative
    lines = [
        "ΠΕΡΙΕΧΟΜΕΝΑ",
        *TOC_ENTRIES[:2],
        body_heading,
        "TABLE OF CONTENTS",
        *TOC_ENTRIES[2:],
    ]
    spans = decode_toc_blocks(analyze_toc_lines(lines), n_physical_lines=30)
    assert [(span.start_line_index, span.end_line_index) for span in spans] == [
        (0, 2),
        (4, 6),
    ]
    assert spans[0].terminator_line_index == 3


def test_repeated_page_furniture_is_bridged_only_inside_the_budget() -> None:
    repeated_header = "ΠΑΝΕΠΙΣΤΗΜΙΟ ΑΘΗΝΩΝ"
    evidence = analyze_toc_lines(
        ["ΠΕΡΙΕΧΟΜΕΝΑ", TOC_ENTRIES[0], repeated_header, TOC_ENTRIES[1]]
    )
    span = decode_toc_blocks(evidence, n_physical_lines=20)[0]
    assert span.bridged_line_indices == (2,)

    repeated_footer = "— 12 —"
    assert analyze_toc_line(repeated_footer).role == TocRole.OTHER
    blocked = analyze_toc_lines(
        ["ΠΕΡΙΕΧΟΜΕΝΑ", TOC_ENTRIES[0], repeated_footer, TOC_ENTRIES[1]]
    )
    assert decode_toc_blocks(blocked, n_physical_lines=20) == ()

    repeated_heading = analyze_toc_lines(
        ["ΠΕΡΙΕΧΟΜΕΝΑ", TOC_ENTRIES[0], "ΠΕΡΙΕΧΟΜΕΝΑ", TOC_ENTRIES[1]]
    )
    assert len(decode_toc_blocks(repeated_heading, n_physical_lines=20)) == 1


def test_mixed_blank_and_continuation_gap_shares_one_two_line_budget() -> None:
    evidence = analyze_toc_lines(
        [
            "ΠΕΡΙΕΧΟΜΕΝΑ",
            TOC_ENTRIES[0],
            "",
            "",
            "ΜΕΡΟΣ ΠΡΩΤΟ ........",
            "ΚΥΡΙΑ ΕΝΟΤΗΤΑ ........",
            TOC_ENTRIES[1],
        ]
    )

    assert decode_toc_blocks(evidence, n_physical_lines=20) == ()


@pytest.mark.parametrize("scope_heading", ["ΔΗΜΟΣΙΕΥΣΕΙΣ", "ΣΗΜΕΙΩΣΕΙΣ"])
def test_cv_and_notes_suppression_survives_neutral_insertions_and_deletions(
    scope_heading: str,
) -> None:
    baseline = [scope_heading, *AUTHOR_YEAR_ENTRIES]
    assert decode_bib_blocks(analyze_bib_lines(baseline)) == ()

    inserted = baseline[:2] + ["ουδέτερη γραμμή"] + baseline[2:]
    assert decode_bib_blocks(analyze_bib_lines(inserted)) == ()
    assert inserted[:2] + inserted[3:] == baseline

    rearmed = [*inserted, "BIBLIOGRAPHY", *AUTHOR_YEAR_ENTRIES[:2]]
    spans = decode_bib_blocks(analyze_bib_lines(rearmed))
    assert len(spans) == 1
    assert spans[0].start_line_index == len(inserted)


@pytest.mark.parametrize(
    ("heading", "entries", "expected_style"),
    [
        ("ΒΙΒΛΙΟΓΡΑΦΙΑ", AUTHOR_YEAR_ENTRIES[:2], "author_year"),
        (
            "REFERENCES",
            [
                "[1] Smith, J. (2020). Alpha. https://doi.org/10.1234/alpha",
                "[2] Brown, K. (2021). Beta. https://doi.org/10.1234/beta",
            ],
            "numbered",
        ),
        (
            "ΝΟΜΟΘΕΣΙΑ",
            [
                "Ν. 1234/2019. Ελληνική νομοθεσία.",
                "Ν. 4567/2020. Ελληνική νομοθεσία.",
            ],
            "legal",
        ),
    ],
)
def test_bibliography_citation_families_confirm_only_coherent_blocks(
    heading: str, entries: list[str], expected_style: str
) -> None:
    evidence = analyze_bib_lines([heading, *entries])
    assert all(
        expected_style in item.citation_styles
        for item in evidence[1:]
        if item.role == BibRole.STRONG_ENTRY_START
    )
    spans = decode_bib_blocks(evidence)
    assert len(spans) == 1
    assert spans[0].line_indices == tuple(range(3))


@pytest.mark.parametrize(
    ("heading", "kind"),
    [
        ("Πίνακας Περιεχομένων", "toc"),
        ("TABLE OF CONTENTS", "toc"),
        ("Βιβλιογραφία", "bib"),
        ("BIBLIOGRAPHY", "bib"),
    ],
)
def test_greek_and_english_headings_have_equivalent_local_roles(
    heading: str, kind: str
) -> None:
    if kind == "toc":
        assert analyze_toc_line(heading).role == TocRole.HEADING
    else:
        assert analyze_bib_line(heading).role == BibRole.HEADING


def test_conflicts_remain_fail_closed_after_prefix_insertion() -> None:
    ambiguous_entries = [
        "[1] Smith (2018). First title ........ 7",
        "[2] Brown (2019). Second title ........ 15",
        "[3] Jones (2020). Third title ........ 29",
        "[4] White (2021). Fourth title ........ 40",
    ]
    baseline = detect_structure(
        ["ΠΕΡΙΕΧΟΜΕΝΑ", *ambiguous_entries], n_physical_lines=20
    )
    assert baseline.spans == ()
    assert len(baseline.conflicts) == 1

    prefixed = detect_structure(
        ["ordinary preface", "ΠΕΡΙΕΧΟΜΕΝΑ", *ambiguous_entries],
        n_physical_lines=20,
    )
    assert prefixed.spans == ()
    assert len(prefixed.conflicts) == 1
    assert prefixed.conflicts[0].overlapping_line_indices == (2, 3, 4, 5)

    # Deleting one entry removes the four-entry headerless bibliography seed;
    # the independent ToC proposal may then be returned instead of a conflict.
    reduced = detect_structure(
        ["ΠΕΡΙΕΧΟΜΕΝΑ", *ambiguous_entries[:3]], n_physical_lines=20
    )
    assert reduced.conflicts == ()
    assert [span.kind for span in reduced.spans] == [StructureKind.TOC]


@pytest.mark.parametrize(
    "lines",
    [
        [
            "1. Το 2018 ξεκίνησε το πρώτο έργο.",
            "2. Το 2019 συνεχίστηκε η δεύτερη φάση.",
            "3. Το 2020 ολοκληρώθηκε η τρίτη φάση.",
            "4. Το 2021 άρχισε το νέο πρόγραμμα.",
        ],
        [
            "Smith (2020) argues that this policy failed in practice.",
            "Jones (2021) explains the different outcome in Greece.",
            "Brown (2019) instead supports the older interpretation.",
            "White (2022) rejects both accounts in the final analysis.",
        ],
    ],
)
def test_headerless_chronologies_and_literature_review_prose_are_not_bibliographies(
    lines: list[str],
) -> None:
    assert decode_bib_blocks(analyze_bib_lines(lines)) == ()


@pytest.mark.parametrize("heading", ["Sources", "ΠΗΓΕΣ"])
def test_ambiguous_sources_heading_does_not_authorize_a_chronology(
    heading: str,
) -> None:
    chronology = [
        heading,
        "1. Το 2018 ξεκίνησε το πρώτο έργο.",
        "2. Το 2019 συνεχίστηκε η δεύτερη φάση.",
    ]

    assert decode_bib_blocks(analyze_bib_lines(chronology)) == ()
