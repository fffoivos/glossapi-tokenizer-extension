from __future__ import annotations

import sys
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.bibliography_v2 import (  # noqa: E402
    analyze_bibliography_line_v2,
    analyze_bibliography_lines_v2,
    decode_bibliography_blocks_v2,
    extract_bibliography_features,
)
from sequence_models.deterministic_structure import BibRole  # noqa: E402


def test_explicit_english_journal_features_are_independently_counted() -> None:
    features = extract_bibliography_features(
        'Smith, J. A. & Brown, Ph. (2021). "A title." Nat. Chem. Biol. '
        "2006, 2, pp. 241-243. https://doi.org/10.1234/example"
    )

    assert features.initial_count >= 3
    assert features.initial_sequence_count >= 1
    assert features.ampersand_count == 1
    assert features.quoted_span_count == 1
    assert features.dotted_sequence_count >= 1
    assert features.journal_year_volume_count == 1
    assert features.page_marker_count == 1
    assert features.page_range_count == 1
    assert features.url_count == 1
    assert features.doi_count == 1


def test_explicit_greek_publisher_editor_and_page_features() -> None:
    features = extract_bibliography_features(
        "Παπαδόπουλος, Α. Ι. (επιμ.). (2019). «Ένας τίτλος». "
        "Αθήνα: Εκδόσεις Πατάκη, σσ. 12–19."
    )

    assert features.initial_count == 2
    assert features.editor_term_count == 1
    assert features.quoted_span_count == 1
    assert features.place_name_count == 1
    assert features.place_publisher_shape_count == 1
    assert features.publisher_term_count >= 1
    assert features.page_marker_count == 1
    assert features.page_range_count == 1


def test_dates_thesis_isbn_and_issn_are_separate_features() -> None:
    features = extract_bibliography_features(
        "Doctoral dissertation, 17-05-2020, accessed 3 March 2021. "
        "ISBN 978-0-306-40615-7. ISSN 2049-3630."
    )

    assert features.thesis_term_count == 1
    assert features.numeric_date_count == 1
    assert features.month_date_count == 1
    assert features.access_date_count == 1
    assert features.isbn_count == 1
    assert features.issn_count == 1


def test_undotted_biomedical_author_lists_are_detected_as_names() -> None:
    evidence = analyze_bibliography_line_v2(
        "Rancic Z, Pecoraro F, Pfammatter T, Banzic I, Klein H (2012). "
        "The use of a stent graft. Angiology 63:634-637."
    )

    assert evidence.features.name_initial_pair_count >= 4
    assert evidence.role == BibRole.STRONG_ENTRY_START
    assert "BIB2_REPEATED_NAME_INITIAL_PAIRS" in evidence.reason_codes

    prose = extract_bibliography_features(
        "These ordinary words as a, simple example in a, sentence are lowercase."
    )
    assert prose.name_initial_pair_count == 0


def test_non_citation_markdown_table_rows_are_hard_barriers() -> None:
    row = analyze_bibliography_line_v2(
        "| Γεώργιος Ζωγράφος : Καθηγητής Χειρουργικής Ε. Κ. Π. Α. |"
    )
    separator = analyze_bibliography_line_v2("|-----------------------|")

    assert row.role == separator.role == BibRole.HARD_OTHER
    assert row.hard_negative and separator.hard_negative
    assert "BIB2_NEGATIVE_NONCITATION_TABLE_ROW" in row.reason_codes


def test_long_numbered_prose_is_not_an_anchor_without_citation_support() -> None:
    evidence = analyze_bibliography_line_v2(
        "1. Knowledge is the first property of products and services, and this "
        "ordinary explanatory list item develops a long argument without any "
        "publication coordinate or bibliographic date."
    )

    assert evidence.role not in {
        BibRole.STRONG_ENTRY_START,
        BibRole.WEAK_ENTRY_START,
    }


def test_inline_author_year_prose_retains_the_conservative_veto() -> None:
    evidence = analyze_bibliography_line_v2(
        "Όπως υποστηρίζει ο Παπαδόπουλος (2019), η ανάλυση αυτή συνεχίζεται "
        "στο επόμενο τμήμα του κεφαλαίου."
    )

    assert evidence.role == BibRole.HARD_OTHER
    assert evidence.hard_negative
    assert "BIB2_NEGATIVE_INLINE_CITATION_PROSE" in evidence.reason_codes


def test_neutral_fragment_is_rescued_only_between_citation_anchors() -> None:
    entries = [
        "Smith, J. A. (2018). First title. London: Academic Press.",
        "and",
        "Brown, K. B. (2019). Second title. London: Academic Press.",
        "Jones, P. C. (2020). Third title. London: Academic Press.",
    ]
    evidence = analyze_bibliography_lines_v2(entries)
    spans = decode_bibliography_blocks_v2(evidence)

    assert len(spans) == 1
    assert spans[0].seed_kind == "bib2_headerless_dense_run"
    assert spans[0].line_indices == (0, 1, 2, 3)
    assert spans[0].bridged_line_indices == (1,)
    assert decode_bibliography_blocks_v2(analyze_bibliography_lines_v2(["and"])) == ()


def test_heading_allows_two_anchors_but_not_one() -> None:
    one = analyze_bibliography_lines_v2(
        ["ΒΙΒΛΙΟΓΡΑΦΙΑ", "Smith, J. A. (2018). First title. London: Academic Press."]
    )
    two = analyze_bibliography_lines_v2(
        [
            "ΒΙΒΛΙΟΓΡΑΦΙΑ",
            "Smith, J. A. (2018). First title. London: Academic Press.",
            "Brown, K. B. (2019). Second title. London: Academic Press.",
        ]
    )

    assert decode_bibliography_blocks_v2(one) == ()
    assert decode_bibliography_blocks_v2(two)[0].line_indices == (0, 1, 2)


def test_long_body_line_breaks_anchor_clusters() -> None:
    body = (
        "This chapter develops a complete and deliberately long argument about the "
        "subject, explains the method, reports the observations, and then continues "
        "with ordinary expository prose that should never be bridged."
    )
    evidence = analyze_bibliography_lines_v2(
        [
            "Smith, J. A. (2018). First title. London: Academic Press.",
            body,
            "Brown, K. B. (2019). Second title. London: Academic Press.",
            "Jones, P. C. (2020). Third title. London: Academic Press.",
        ]
    )

    assert decode_bibliography_blocks_v2(evidence) == ()
