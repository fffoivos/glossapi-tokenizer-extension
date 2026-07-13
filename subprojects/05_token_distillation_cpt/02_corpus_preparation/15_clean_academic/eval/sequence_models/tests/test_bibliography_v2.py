from __future__ import annotations

import sys
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.bibliography_v2 import (  # noqa: E402
    BibliographyFeatureReview,
    analyze_bibliography_line_v2,
    analyze_bibliography_lines_v2,
    decode_bibliography_blocks_v2,
    extract_bibliography_feature_review,
    extract_bibliography_features,
)
from sequence_models.deterministic_structure import BibRole  # noqa: E402


def test_explicit_english_journal_features_use_specific_ownership() -> None:
    features = extract_bibliography_features(
        'Smith, J. A. & Brown, Ph. (2021). "A title." Nat. Chem. Biol. '
        "2006, 2, pp. 241-243. https://doi.org/10.1234/example"
    )

    assert features.initial_count >= 3
    assert features.ampersand_count == 1
    assert features.quoted_span_count == 1
    assert features.dotted_sequence_count >= 1
    assert features.journal_year_volume_count == 1
    assert features.page_marker_count == 1
    assert features.page_range_count == 1
    assert features.url_count == 0  # The more-specific DOI owns the URL span.
    assert features.doi_count == 1


def test_initials_are_at_most_two_letters_and_do_not_overlap_dotted_words() -> None:
    features = extract_bibliography_features("I. Ph. Pro. Nat. Chem. Biol.")

    assert features.initial_count == 2
    assert features.dotted_word_count == 0
    assert features.dotted_sequence_count == 1


def test_inverted_author_counts_every_author_and_includes_all_initials() -> None:
    text = "Lewis, M.A., Haviland-Jones, J.M., & Barrett, L.F. (2008). Handbook."
    review = extract_bibliography_feature_review(text)
    inverted = [
        match for match in review.matches if match.feature == "inverted_author_count"
    ]

    assert review.features.inverted_author_count == 3
    assert [match.text.rstrip() for match in inverted] == [
        "Lewis, M.A.",
        "Haviland-Jones, J.M.",
        "Barrett, L.F.",
    ]
    assert review.features.initial_count == 6


def test_european_author_scripts_and_diacritics_are_supported() -> None:
    names = (
        "García Márquez, G. G.",  # Romance, Latin
        "de Gaulle, C.",  # Romance surname particle
        "Țepeș, V.",  # Romanian
        "Dvořák, A.",  # Czech
        "Łukasiewicz, J.",  # Polish
        "Šimić, M.",  # Croatian
        "Župančič, O.",  # Slovene
        "Čiarnienė, R.",  # Baltic Latin; the source's unsplit form
        "Достоевский, Ф. М.",  # Russian Cyrillic
        "Шевченко, Т. Г.",  # Ukrainian Cyrillic
        "Његош, П. П.",  # Serbian Cyrillic
        "Παπαδόπουλος, Ι.",  # monotonic Greek
        "Ἀριστοτέλης, Ἀ.",  # polytonic Greek
    )

    for name in names:
        review = extract_bibliography_feature_review(f"- {name} (2001). Title.")
        inverted = [
            match.text.strip()
            for match in review.matches
            if match.feature == "inverted_author_count"
        ]
        assert len(inverted) == 1, (name, review)
        assert inverted[0] == name


def test_ocr_split_lithuanian_authors_on_document_47416_line_1142() -> None:
    text = (
        "- 4) Č iarnien ė ,  R.  and  Vienažindien ė ,  M.  (2012),  "
        "'Lean manufacturing: Theory and Practice', Economics and Management, "
        "Vol. 17 (2), pp. 1-7."
    )
    review = extract_bibliography_feature_review(text)
    inverted = [
        match.text.strip()
        for match in review.matches
        if match.feature == "inverted_author_count"
    ]

    assert inverted == ["Č iarnien ė ,  R.", "Vienažindien ė ,  M."]
    assert review.features.inverted_author_count == 2


def test_undotted_biomedical_citation_variants_have_specific_owners() -> None:
    methods = extract_bibliography_feature_review(
        "8. Gotto AM, Pownall HJ, Havel RJ. 1986. Introduction to the plasma "
        "lipoproteins. In Methods in Enzymology. Segrst P, Albers JJ (eds). "
        "Academic Press, London, pp 3-41."
    )
    clinics = extract_bibliography_feature_review(
        "23. Tamir I, Heiss G, Glueck CJ, Christensen B, Kwiterovich P, "
        "Rifkind BM. 1981. Lipid and lipoprotein distributions in white children "
        "ages 619 years. The Lipid Research Clinics Program Prevalence Study. "
        "J. Clin. Dis. 34:27-39."
    )
    metabolic = extract_bibliography_feature_review(
        "33. Nikkila EA. 1983. Familial lipoprotein lipase deficiency. In "
        "Metabolic Basis of Inherited Diseas 5 th edn. Stanbury JB (ed). "
        "McGraw Hill, New York, pp 622-642."
    )

    def matches(review: BibliographyFeatureReview, feature: str) -> list[str]:
        return [
            match.text
            for match in review.matches
            if match.feature == feature
        ]

    assert methods.features.name_initial_pair_count == 5
    assert "Enzymology." in matches(methods, "dotted_word_count")
    assert "(eds)" in matches(methods, "editor_term_count")
    assert "pp " in matches(methods, "page_marker_count")

    assert clinics.features.name_initial_pair_count == 6
    assert matches(clinics, "article_page_range_count") == ["34:27-39"]
    assert clinics.features.page_range_count == 0

    assert metabolic.features.name_initial_pair_count == 2
    assert "(ed)" in matches(metabolic, "editor_term_count")
    assert matches(metabolic, "edition_term_count") == ["5 th edn"]
    assert "pp " in matches(metabolic, "page_marker_count")
    assert matches(metabolic, "page_range_count") == ["622-642"]


def test_direct_author_hypothesis_beats_spurious_inverted_fragments() -> None:
    text = (
        "- [126] C. Avin and G. Ercal, 'On the cover time of random geometric "
        "graphs,' in Automata, Languages and Programming (L. Caires, G. F. "
        "Italiano, L. Monteiro, C. Palamidessi, and M. Yung, eds.), (Berlin, "
        "Heidelberg), pp. 677-689, Springer Berlin Heidelberg, 2005."
    )
    review = extract_bibliography_feature_review(text)
    direct = [
        match.text
        for match in review.matches
        if match.feature == "direct_author_count"
    ]

    assert review.features.direct_author_count == 7
    assert review.features.inverted_author_count == 0
    assert direct == [
        "C. Avin",
        "G. Ercal",
        "L. Caires",
        "G. F. Italiano",
        "L. Monteiro",
        "C. Palamidessi",
        "M. Yung",
    ]


def test_specific_terms_own_broad_dotted_and_proper_word_spans() -> None:
    review = extract_bibliography_feature_review(
        "Editor. Eds. Vol. 12, Σελ. 3, Press. Nat. Chem. Biol."
    )
    dotted = [
        match.text
        for match in review.matches
        if match.feature == "dotted_word_count"
    ]
    proper = [
        match.text
        for match in review.matches
        if match.feature == "proper_name_word_count"
    ]

    assert review.features.editor_term_count == 2
    assert review.features.volume_marker_count == 1
    assert review.features.page_marker_count == 1
    assert review.features.publisher_term_count == 1
    assert review.features.dotted_sequence_count == 1
    assert dotted == []
    assert proper == []


def test_proper_words_never_end_at_a_dot() -> None:
    review = extract_bibliography_feature_review("Alpha. Beta Gamma")
    proper = [
        match.text
        for match in review.matches
        if match.feature == "proper_name_word_count"
    ]

    assert proper == ["Beta", "Gamma"]
    assert all(
        review.normalized_text[match.end :].lstrip(" \t").startswith(".") is False
        for match in review.matches
        if match.feature == "proper_name_word_count"
    )


def test_dates_pages_and_years_do_not_duplicate_numeric_spans() -> None:
    citation = extract_bibliography_features(
        "Published 2004. J Immunol 173:1535-1548."
    )
    named_date = extract_bibliography_features("17-19 November 2005")
    year_range = extract_bibliography_features("1980-1990")

    assert citation.year_count == 1
    assert citation.article_page_range_count == 1
    assert citation.page_range_count == 0
    assert named_date.month_date_count == 1
    assert named_date.year_count == 0
    assert named_date.page_range_count == 0
    assert named_date.numbered_entry_count == 0
    assert year_range.year_count == 2
    assert year_range.page_range_count == 0
    assert year_range.numbered_entry_count == 0


def test_article_page_range_owns_the_complete_lipics_coordinate() -> None:
    text = "vol. 66, pp. 44:1-44:14, Schloss Dagstuhl, 2017."
    review = extract_bibliography_feature_review(text)
    article_pages = [
        match
        for match in review.matches
        if match.feature == "article_page_range_count"
    ]

    assert review.features.article_page_range_count == 1
    assert review.features.page_range_count == 0
    assert len(article_pages) == 1
    assert article_pages[0].text == "44:1-44:14"
    assert article_pages[0].start == text.index("44:1")


def test_editor_abbreviations_do_not_claim_edition_or_hyphenated_trans() -> None:
    features = extract_bibliography_features(
        "All-trans-retinoic acid. 2nd ed. Editor. Trans."
    )

    assert features.edition_term_count == 1
    assert features.editor_term_count == 2


def test_inline_author_page_is_not_a_journal_volume() -> None:
    features = extract_bibliography_features(
        "Σύμφωνα με τον Gerler (2013: 275), ένα από τα βασικά χαρακτηριστικά "
        "μιας κρίσης είναι το γεγονός ότι γίνεται αντιληπτή αρνητικά."
    )

    assert features.prose_lead_count == 1
    assert features.journal_year_volume_count == 0
    assert features.year_count == 1


def test_decimal_and_thousands_ranges_are_not_dates_or_page_ranges() -> None:
    features = extract_bibliography_features(
        "7.34 [7.28-7.44], density 1,006-1,019, odds 1.03-3.78."
    )

    assert features.numeric_date_count == 0
    assert features.page_range_count == 0
    assert features.numbered_entry_count == 0


def test_late_journal_abbreviation_is_not_a_direct_author() -> None:
    features = extract_bibliography_features(
        "Acta. Obstet. Gynecol P. Scand. 5213 55"
    )

    assert features.direct_author_count == 0


def test_numbered_entry_uses_first_non_special_character() -> None:
    positives = (
        "1 Lewis, M.A. (2008). Handbook.",
        "[2] Lewis, M.A. (2008). Handbook.",
        "— (3) Lewis, M.A. (2008). Handbook.",
        "*** #4: Lewis, M.A. (2008). Handbook.",
    )
    for text in positives:
        assert extract_bibliography_features(text).numbered_entry_count == 1

    assert extract_bibliography_features(
        "Lewis, M.A. discussed 4 examples."
    ).numbered_entry_count == 0


def test_review_matches_have_exact_offsets_and_count_parity() -> None:
    text = (
        'Smith, J. A. & Brown, Ph. (2021). "A title." Nat. Chem. Biol. '
        "2006, 2, pp. 241-243. https://doi.org/10.1234/example"
    )
    review = extract_bibliography_feature_review(text)
    by_feature: dict[str, list] = {}
    for match in review.matches:
        by_feature.setdefault(match.feature, []).append(match)
        assert review.normalized_text[match.start : match.end] == match.text
    for feature, count in review.features.as_dict().items():
        if feature != "token_count":
            assert len(by_feature.get(feature, [])) == count
    doi = by_feature["doi_count"][0]
    assert doi.text == "https://doi.org/10.1234/example"
    assert (doi.start, doi.end) == (
        review.normalized_text.index("https://doi.org"),
        len(review.normalized_text),
    )
    page_range = by_feature["page_range_count"][0]
    assert page_range.text == "241-243"


def test_review_offsets_account_for_outer_markdown_table_bars() -> None:
    review = extract_bibliography_feature_review(
        " | Andreou, S. A. (1987). A method. Water Journal, 2, 3-8. | "
    )
    inverted = next(
        match for match in review.matches if match.feature == "inverted_author_count"
    )
    assert inverted.start == review.normalized_text.index("Andreou")
    assert inverted.text.startswith("Andreou, S.")
    table = next(match for match in review.matches if match.feature == "table_row_count")
    assert table.text.startswith("|") and table.text.endswith("|")


def test_explicit_greek_publisher_editor_and_page_features() -> None:
    features = extract_bibliography_features(
        "Παπαδόπουλος, Α. Ι. (επιμ.). (2019). «Ένας τίτλος». "
        "Αθήνα: Εκδόσεις Πατάκη, σσ. 12–19."
    )

    assert features.initial_count == 2
    assert features.editor_term_count == 1
    assert features.quoted_span_count == 1
    assert features.place_name_count == 0  # Owned by the place–publisher shape.
    assert features.place_publisher_shape_count == 1
    assert features.publisher_term_count == 1  # The separate publisher name remains.
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


def test_generic_concept_colon_is_not_a_place_publisher_shape() -> None:
    features = extract_bibliography_features(
        "1. Knowledge: This is an ordinary explanation in a numbered list."
    )

    assert features.place_publisher_shape_count == 0


def test_figure_caption_with_year_and_source_url_is_a_hard_negative() -> None:
    evidence = analyze_bibliography_line_v2(
        "Εικ.7.1 P. Gauguin, Οβίρι, 1891-1893. Πηγή: https://example.org/image"
    )

    assert evidence.role == BibRole.HARD_OTHER
    assert "BIB2_NEGATIVE_FIGURE_CAPTION" in evidence.reason_codes


def test_citation_table_rows_are_allowed_but_separators_remain_barriers() -> None:
    citation = analyze_bibliography_line_v2(
        "| Andreou, S. A., Marks, D. H. (1987). A method. Water Journal, 2, 3-8. |"
    )
    separator = analyze_bibliography_line_v2("|-----------------------|")

    assert citation.role in {BibRole.STRONG_ENTRY_START, BibRole.WEAK_ENTRY_START}
    assert not citation.hard_negative
    assert separator.role == BibRole.HARD_OTHER


def test_extended_chapter_heading_and_language_subheading_are_typed() -> None:
    heading = analyze_bibliography_line_v2("ΒΙΒΛΙΟΓΡΑΦΙΑ ΚΕΦΑΛΑΙΟΥ Γ.")
    subheading = analyze_bibliography_line_v2("Ελληνόγλωσσες Βιβλιογραφικές Πηγές")

    assert heading.role == BibRole.HEADING
    assert subheading.role == BibRole.SUBHEADING


def test_inline_citation_prose_retains_the_conservative_veto() -> None:
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
