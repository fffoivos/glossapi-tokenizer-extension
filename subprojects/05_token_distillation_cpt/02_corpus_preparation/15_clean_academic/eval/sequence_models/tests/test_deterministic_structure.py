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
    evaluate_r0_section,
)


def _r0_rows(numbers: list[int]) -> list[str]:
    return [
        f"| Ενότητα {index} | {number} |" for index, number in enumerate(numbers, 1)
    ]


class TestFrozenR0:
    def test_missing_and_non_table_text_are_negative(self) -> None:
        assert evaluate_r0_section(None) == (0, [])
        assert evaluate_r0_section(float("nan")) == (0, [])
        assert evaluate_r0_section("Εισαγωγή ........ 7") == (0, [])

    def test_nearly_all_table_shortcut_accepts_four_non_decreasing_rows(self) -> None:
        text = "\n".join(_r0_rows([1, 2, 2, 4]))
        assert evaluate_r0_section(text) == (1, [1, 2, 2, 4])

    def test_four_page_run_does_not_satisfy_normal_seven_page_threshold(self) -> None:
        # Seven numbers disable the minimum-count exit, while two prose lines
        # disable the nearly-all-table shortcut.  The longest increasing run is
        # only four, which catches an accidental double sequence increment.
        text = "\n".join(
            _r0_rows([1, 2, 3, 4, 1, 0, 2]) + ["ordinary prose", "more prose"]
        )
        assert evaluate_r0_section(text) == (0, [1, 2, 3, 4, 1, 0, 2])

    def test_normal_seven_page_sequence_and_broken_row_merge_match_r0(self) -> None:
        lines = _r0_rows([1, 2, 3, 4, 5, 6])
        lines += ["| Τελική ενότητα", "συνέχεια | 7 |", "ordinary", "ordinary"]
        assert evaluate_r0_section("\n".join(lines)) == (1, [1, 2, 3, 4, 5, 6, 7])

    def test_r0_keeps_legacy_page_ceiling(self) -> None:
        text = "\n".join(f"| Ενότητα Άλφα | {page} |" for page in [401, 402, 403, 404])
        assert evaluate_r0_section(text) == (0, [])


class TestTocLocalEvidence:
    @pytest.mark.parametrize(
        ("text", "page", "page_kind"),
        [
            ("1. Εισαγωγή ........ 7", 7, "arabic"),
            ("| 1.2 Μεθοδολογία | 34 |", 34, "arabic"),
            ("Πρόλογος ........ ix", 9, "roman"),
            ("Ά Μέρος ........ 417", 417, "arabic"),
        ],
    )
    def test_strong_entries_cover_plain_table_roman_unicode_and_long_books(
        self, text: str, page: int, page_kind: str
    ) -> None:
        evidence = analyze_toc_line(text)
        assert evidence.role == TocRole.STRONG_ENTRY
        assert evidence.page_number == page
        assert evidence.page_kind == page_kind
        assert not evidence.hard_negative

    @pytest.mark.parametrize("tail", ["IIII", "IIV", "civil"])
    def test_noncanonical_roman_word_tails_are_not_pages(self, tail: str) -> None:
        evidence = analyze_toc_line(f"Πρόλογος ........ {tail}")

        assert evidence.page_number is None
        assert evidence.page_kind is None

    def test_exact_accented_heading_is_recognised(self) -> None:
        evidence = analyze_toc_line("## Πίνακας Περιεχομένων")
        assert evidence.role == TocRole.HEADING
        assert evidence.reason_codes == ("TOC_EXACT_HEADING",)

    @pytest.mark.parametrize(
        "text",
        [
            "| Περιφέρεια | 17,4% |",
            "| Έτος | 2024 |",
            "1. Ο αιτών υποχρεούται να προσκομίσει τα δικαιολογητικά.",
            "Κανονικό κείμενο που αναπτύσσει ένα επιχείρημα σε αρκετές λέξεις και "
            "συνεχίζει ως ολοκληρωμένη παράγραφος με σαφές νόημα και τελική στίξη.",
        ],
    )
    def test_statistical_year_and_prose_lines_are_hard_negatives(
        self, text: str
    ) -> None:
        evidence = analyze_toc_line(text)
        assert evidence.role == TocRole.HARD_OTHER
        assert evidence.hard_negative
        assert any("NEGATIVE" in reason for reason in evidence.reason_codes)

    def test_single_line_analyser_validates_its_coordinate(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            analyze_toc_line("ΠΕΡΙΕΧΟΜΕΝΑ", -1)


class TestTocCoherence:
    def test_heading_needs_two_confirming_strong_entries(self) -> None:
        incomplete = analyze_toc_lines(["ΠΕΡΙΕΧΟΜΕΝΑ", "1. Εισαγωγή ........ 7"])
        complete = analyze_toc_lines(
            ["ΠΕΡΙΕΧΟΜΕΝΑ", "1. Εισαγωγή ........ 7", "2. Μέθοδος ........ 15"]
        )
        assert decode_toc_blocks(incomplete, n_physical_lines=10) == ()
        assert len(decode_toc_blocks(complete, n_physical_lines=10)) == 1

    def test_identical_ambiguous_line_is_rescued_only_inside_a_coherent_block(
        self,
    ) -> None:
        ambiguous = "ΜΕΡΟΣ ΠΡΩΤΟ"
        isolated = analyze_toc_lines([ambiguous])
        contextual = analyze_toc_lines(
            [
                "ΠΕΡΙΕΧΟΜΕΝΑ",
                "1. Εισαγωγή ........ 7",
                ambiguous,
                "2. Μέθοδος ........ 15",
            ]
        )
        assert isolated[0].role == contextual[2].role == TocRole.POSSIBLE_CONTINUATION
        assert decode_toc_blocks(isolated, n_physical_lines=10) == ()
        span = decode_toc_blocks(contextual, n_physical_lines=10)[0]
        assert contextual[2].line_index in span.bridged_line_indices

    def test_headerless_dense_run_is_front_limited_and_format_coherent(self) -> None:
        lines = [
            "1. Εισαγωγή ........ 7",
            "2. Μέθοδος ........ 15",
            "3. Αποτελέσματα ........ 29",
            "4. Συμπεράσματα ........ 40",
        ]
        front = analyze_toc_lines(lines, start_line_index=5)
        back = analyze_toc_lines(lines, start_line_index=70)
        assert (
            decode_toc_blocks(front, n_physical_lines=200)[0].seed_kind
            == "toc_headerless_dense_run"
        )
        assert decode_toc_blocks(back, n_physical_lines=200) == ()

    def test_hard_negative_splits_blocks_and_allows_a_later_independent_block(
        self,
    ) -> None:
        body = (
            "Κανονικό κείμενο που αναπτύσσει ένα επιχείρημα σε αρκετές λέξεις και "
            "συνεχίζει ως ολοκληρωμένη παράγραφος με σαφές νόημα και τελική στίξη."
        )
        evidence = analyze_toc_lines(
            [
                "ΠΕΡΙΕΧΟΜΕΝΑ",
                "1. Εισαγωγή ........ 7",
                "2. Μέθοδος ........ 15",
                body,
                "ΠΕΡΙΕΧΟΜΕΝΑ",
                "1. Πίνακες ........ 2",
                "2. Σχήματα ........ 6",
            ]
        )
        spans = decode_toc_blocks(evidence, n_physical_lines=20)
        assert [(span.start_line_index, span.end_line_index) for span in spans] == [
            (0, 2),
            (4, 6),
        ]
        assert spans[0].terminator_line_index == 3

    def test_three_soft_gap_lines_cannot_be_blindly_bridged(self) -> None:
        evidence = analyze_toc_lines(
            [
                "ΠΕΡΙΕΧΟΜΕΝΑ",
                "1. Εισαγωγή ........ 7",
                "ΜΕΡΟΣ ΠΡΩΤΟ",
                "ΚΥΡΙΑ ΕΝΟΤΗΤΑ",
                "ΔΕΥΤΕΡΗ ΥΠΟΕΝΟΤΗΤΑ",
                "2. Μέθοδος ........ 15",
            ]
        )
        assert decode_toc_blocks(evidence, n_physical_lines=20) == ()


class TestBibliographyLocalEvidence:
    def test_greek_author_year_publisher_is_a_multi_family_entry(self) -> None:
        evidence = analyze_bib_line(
            "Παπαδόπουλος, Α. (2019). Η γλωσσική εκπαίδευση. Αθήνα: Εκδόσεις Πατάκη."
        )
        assert evidence.role == BibRole.STRONG_ENTRY_START
        assert {"author", "year", "publisher_place"}.issubset(
            evidence.evidence_families
        )
        assert "author_year" in evidence.citation_styles

    def test_numbered_doi_entry_is_strong_and_explained(self) -> None:
        evidence = analyze_bib_line(
            "[12] Smith, J. (2021). A title. https://doi.org/10.1234/example"
        )
        assert evidence.role == BibRole.STRONG_ENTRY_START
        assert evidence.entry_number == 12
        assert {"numbered", "web"}.issubset(evidence.citation_styles)

    def test_inline_citation_prose_and_cv_heading_are_hard_negatives(self) -> None:
        prose = analyze_bib_line(
            "Όπως υποστηρίζει ο Παπαδόπουλος (2019), η εκπαίδευση είναι σημαντική και "
            "το επιχείρημα αναπτύσσεται αναλυτικά στην ακόλουθη παράγραφο."
        )
        cv = analyze_bib_line("ΔΗΜΟΣΙΕΥΣΕΙΣ")
        assert prose.role == cv.role == BibRole.HARD_OTHER
        assert "BIB_NEGATIVE_INLINE_CITATION_PROSE" in prose.reason_codes
        assert "BIB_NEGATIVE_CV_PUBLICATIONS_HEADING" in cv.reason_codes

    def test_procedural_legal_enumeration_is_not_a_numbered_reference(self) -> None:
        evidence = analyze_bib_line(
            "1. Ο αιτών υποχρεούται να προσκομίσει τα απαιτούμενα δικαιολογητικά."
        )
        assert evidence.role == BibRole.HARD_OTHER
        assert "BIB_NEGATIVE_LEGAL_PROCEDURE" in evidence.reason_codes

    def test_typed_subheading_and_wrapped_publisher_tail_are_not_seeds(self) -> None:
        subheading = analyze_bib_line("ΔΙΚΤΥΟΓΡΑΦΙΑ")
        continuation = analyze_bib_line("  Αθήνα: Εκδόσεις Πατάκη.")
        assert subheading.role == BibRole.SUBHEADING
        assert subheading.citation_styles == ("web",)
        assert continuation.role == BibRole.POSSIBLE_CONTINUATION


class TestBibliographyCoherence:
    @staticmethod
    def _entries() -> list[str]:
        return [
            "Smith, J. (2018). First title. London: Press.",
            "Brown, K. (2019). Second title. London: Press.",
            "Jones, P. (2020). Third title. London: Press.",
            "White, R. (2021). Fourth title. London: Press.",
        ]

    def test_heading_requires_confirmation_and_never_extends_blindly_to_eof(
        self,
    ) -> None:
        assert decode_bib_blocks(analyze_bib_lines(["ΒΙΒΛΙΟΓΡΑΦΙΑ"])) == ()
        lines = ["ΒΙΒΛΙΟΓΡΑΦΙΑ", *self._entries()[:2], "Ordinary body resumes here"]
        span = decode_bib_blocks(analyze_bib_lines(lines))[0]
        assert (span.start_line_index, span.end_line_index) == (0, 2)
        assert span.terminator_line_index == 3
        assert 3 not in span.line_indices

    def test_identical_continuation_is_contextual_not_an_independent_block(
        self,
    ) -> None:
        continuation = "  Αθήνα: Εκδόσεις Πατάκη."
        isolated = analyze_bib_lines([continuation])
        contextual = analyze_bib_lines(
            ["ΒΙΒΛΙΟΓΡΑΦΙΑ", self._entries()[0], continuation, self._entries()[1]]
        )
        assert isolated[0].role == contextual[2].role == BibRole.POSSIBLE_CONTINUATION
        assert decode_bib_blocks(isolated) == ()
        span = decode_bib_blocks(contextual)[0]
        assert contextual[2].line_index in span.bridged_line_indices

    def test_four_style_compatible_entries_seed_a_headerless_block(self) -> None:
        spans = decode_bib_blocks(analyze_bib_lines(self._entries() + ["Body"]))
        assert len(spans) == 1
        assert spans[0].seed_kind == "bib_headerless_dense_run"
        assert spans[0].line_indices == (0, 1, 2, 3)

    def test_cv_publication_context_suppresses_headerless_detection(self) -> None:
        evidence = analyze_bib_lines(["ΔΗΜΟΣΙΕΥΣΕΙΣ", *self._entries()])
        assert decode_bib_blocks(evidence) == ()

    def test_cv_scope_remains_denied_beyond_twenty_five_entries(self) -> None:
        long_publication_list = [
            f"Surname{index}, A. ({1980 + index}). Title {index}. London: Press."
            for index in range(30)
        ]
        evidence = analyze_bib_lines(["ΔΗΜΟΣΙΕΥΣΕΙΣ", *long_publication_list])
        assert decode_bib_blocks(evidence) == ()

    def test_notes_scope_denies_reference_like_footnotes_until_a_body_heading(
        self,
    ) -> None:
        note_entries = [
            f"Smith, J. ({1990 + index}). Note source {index}. London: Press."
            for index in range(10)
        ]
        denied = analyze_bib_lines(["ΣΗΜΕΙΩΣΕΙΣ", *note_entries])
        assert decode_bib_blocks(denied) == ()
        reopened = analyze_bib_lines(
            ["ΣΗΜΕΙΩΣΕΙΣ", *note_entries, "ΚΕΦΑΛΑΙΟ 3", *self._entries()]
        )
        spans = decode_bib_blocks(reopened)
        assert len(spans) == 1
        assert spans[0].start_line_index == 12

    def test_explicit_bibliography_heading_rearms_after_cv_context(self) -> None:
        evidence = analyze_bib_lines(
            ["ΔΗΜΟΣΙΕΥΣΕΙΣ", self._entries()[0], "ΒΙΒΛΙΟΓΡΑΦΙΑ", *self._entries()[1:3]]
        )
        spans = decode_bib_blocks(evidence)
        assert len(spans) == 1
        assert spans[0].start_line_index == 2

    def test_unconfirmed_bibliography_heading_does_not_rearm_cv_context(self) -> None:
        evidence = analyze_bib_lines(
            [
                "ΔΗΜΟΣΙΕΥΣΕΙΣ",
                "ΒΙΒΛΙΟΓΡΑΦΙΑ",
                *[f"ordinary spacer {index}" for index in range(12)],
                *self._entries(),
            ]
        )
        assert decode_bib_blocks(evidence) == ()

    def test_one_strong_entry_plus_two_typed_continuations_confirms_heading(
        self,
    ) -> None:
        evidence = analyze_bib_lines(
            [
                "ΒΙΒΛΙΟΓΡΑΦΙΑ",
                self._entries()[0],
                "  Αθήνα: Εκδόσεις Πατάκη.",
                "  https://example.org/catalogue-entry",
            ]
        )
        spans = decode_bib_blocks(evidence)
        assert len(spans) == 1
        assert spans[0].line_indices == (0, 1, 2, 3)

    def test_hard_body_heading_splits_two_formal_bibliography_blocks(self) -> None:
        evidence = analyze_bib_lines(
            [
                "ΒΙΒΛΙΟΓΡΑΦΙΑ",
                *self._entries()[:2],
                "ΚΕΦΑΛΑΙΟ 2 Μεθοδολογία",
                "REFERENCES",
                *self._entries()[2:],
            ]
        )
        spans = decode_bib_blocks(evidence)
        assert [(span.start_line_index, span.end_line_index) for span in spans] == [
            (0, 2),
            (4, 6),
        ]

    @pytest.mark.parametrize(
        ("heading", "entries"),
        [
            ("ΠΗΓΕΣ", _entries.__func__()[:2]),
            ("SOURCES", _entries.__func__()[:2]),
            (
                "ΔΙΚΤΥΟΓΡΑΦΙΑ",
                [
                    "[1] Smith (2020). https://example.org/a",
                    "[2] Brown (2021). https://example.org/b",
                ],
            ),
            (
                "ΝΟΜΟΘΕΣΙΑ",
                [
                    "Ν. 1234/2019. Ελληνική νομοθεσία.",
                    "Ν. 4567/2020. Ελληνική νομοθεσία.",
                ],
            ),
            (
                "ΝΟΜΟΛΟΓΙΑ",
                ["Ν. 1234/2019. Απόφαση αναφοράς.", "Ν. 4567/2020. Απόφαση αναφοράς."],
            ),
        ],
    )
    def test_conditional_formal_anchors_require_matching_entries(
        self, heading: str, entries: list[str]
    ) -> None:
        spans = decode_bib_blocks(analyze_bib_lines([heading, *entries]))
        assert len(spans) == 1
        assert spans[0].start_line_index == 0

    @pytest.mark.parametrize("heading", ["ΔΙΚΤΥΟΓΡΑΦΙΑ", "ΝΟΜΟΘΕΣΙΑ", "ΝΟΜΟΛΟΓΙΑ"])
    def test_typed_anchor_rejects_nonmatching_author_year_entries(
        self, heading: str
    ) -> None:
        assert (
            decode_bib_blocks(analyze_bib_lines([heading, *self._entries()[:2]])) == ()
        )

    @pytest.mark.parametrize("heading", ["ΠΗΓΕΣ", "SOURCES"])
    def test_sources_anchor_rejects_unstructured_body_lines(self, heading: str) -> None:
        assert (
            decode_bib_blocks(
                analyze_bib_lines([heading, "Official website", "Additional material"])
            )
            == ()
        )


def test_combined_entry_point_returns_evidence_and_proposals_not_actions() -> None:
    decision = detect_structure(
        ["ΠΕΡΙΕΧΟΜΕΝΑ", "1. Εισαγωγή ........ 7", "2. Μέθοδος ........ 15"],
        n_physical_lines=20,
    )
    assert len(decision.toc_evidence) == len(decision.bib_evidence) == 3
    assert [span.kind for span in decision.spans] == [StructureKind.TOC]
    assert decision.conflicts == ()
    assert not hasattr(decision.spans[0], "delete")


def test_combined_entry_point_withholds_overlapping_toc_and_bib_candidates() -> None:
    lines = [
        "ΠΕΡΙΕΧΟΜΕΝΑ",
        "[1] Smith (2018). First title ........ 7",
        "[2] Brown (2019). Second title ........ 15",
        "[3] Jones (2020). Third title ........ 29",
        "[4] White (2021). Fourth title ........ 40",
    ]
    decision = detect_structure(lines, n_physical_lines=20)
    assert decision.spans == ()
    assert len(decision.conflicts) == 1
    conflict = decision.conflicts[0]
    assert conflict.toc_span.kind == StructureKind.TOC
    assert conflict.bib_span.kind == StructureKind.BIB
    assert conflict.overlapping_line_indices == (1, 2, 3, 4)


def test_decoders_reject_unsorted_or_duplicate_coordinates() -> None:
    evidence = (
        analyze_toc_line("1. Εισαγωγή ........ 7", 1),
        analyze_toc_line("2. Μέθοδος ........ 15", 1),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        decode_toc_blocks(evidence, n_physical_lines=10)


def test_combined_entry_point_rejects_an_uncovered_physical_extent() -> None:
    with pytest.raises(ValueError, match="does not cover"):
        detect_structure(["one", "two"], start_line_index=5, n_physical_lines=6)
