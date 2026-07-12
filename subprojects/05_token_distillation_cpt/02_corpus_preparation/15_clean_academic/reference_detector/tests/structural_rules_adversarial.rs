//! Synthetic adversarial/metamorphic checks for the public structural rules.
//!
//! These tests never read corpus, STRUCT2K, or sealed/test artifacts.

use reference_detector::{
    structural_detect, BibStyle, LineRole, ReasonCode, StructuralConfig, StructuralDecision,
    StructureKind,
};

const TOC_1: &str = "1. Εισαγωγή ........ 1";
const TOC_2: &str = "2. Θεωρία ........ 7";

const AY_1: &str = "Smith, J. (2018). First title. London: Press.";
const AY_2: &str = "Brown, K. (2019). Second title. London: Press.";
const AY_3: &str = "Jones, P. (2020). Third title. London: Press.";
const AY_4: &str = "White, R. (2021). Fourth title. London: Press.";

fn detect(lines: &[&str]) -> StructuralDecision {
    structural_detect(
        "synthetic-doc",
        "adversarial-test",
        &lines.join("\n"),
        &StructuralConfig::default(),
    )
}

fn only_span(
    result: &StructuralDecision,
    kind: StructureKind,
) -> &reference_detector::StructureSpan {
    let matching: Vec<_> = result
        .spans
        .iter()
        .filter(|span| span.kind == kind)
        .collect();
    assert_eq!(matching.len(), 1, "unexpected spans: {:?}", result.spans);
    matching[0]
}

#[test]
fn toc_typed_line_insertion_and_deletion_preserve_anchor_membership() {
    let baseline = detect(&["ΠΕΡΙΕΧΟΜΕΝΑ", TOC_1, TOC_2]);
    let baseline_span = only_span(&baseline, StructureKind::Toc);
    assert_eq!(baseline_span.supporting_lines, vec![0, 1, 2]);

    let inserted = detect(&["ΠΕΡΙΕΧΟΜΕΝΑ", TOC_1, "ΜΕΡΟΣ ΠΡΩΤΟ ........", TOC_2]);
    let inserted_span = only_span(&inserted, StructureKind::Toc);
    assert_eq!(inserted_span.supporting_lines, vec![0, 1, 3]);
    assert_eq!(inserted_span.bridged_lines, vec![2]);

    let restored = detect(&["ΠΕΡΙΕΧΟΜΕΝΑ", TOC_1, TOC_2]);
    let restored_span = only_span(&restored, StructureKind::Toc);
    assert_eq!(
        restored_span.supporting_lines,
        baseline_span.supporting_lines
    );
    assert!(restored_span.bridged_lines.is_empty());
}

#[test]
fn soft_gap_line_budget_accepts_two_and_rejects_three() {
    let accepted = detect(&[
        "ΠΕΡΙΕΧΟΜΕΝΑ",
        TOC_1,
        "ΜΕΡΟΣ ΠΡΩΤΟ ........",
        "ΚΥΡΙΑ ΕΝΟΤΗΤΑ ........",
        TOC_2,
    ]);
    let span = only_span(&accepted, StructureKind::Toc);
    assert_eq!(span.bridged_lines, vec![2, 3]);

    let rejected = detect(&[
        "ΠΕΡΙΕΧΟΜΕΝΑ",
        TOC_1,
        "ΜΕΡΟΣ ΠΡΩΤΟ ........",
        "ΚΥΡΙΑ ΕΝΟΤΗΤΑ ........",
        "ΔΕΥΤΕΡΗ ΥΠΟΕΝΟΤΗΤΑ ........",
        TOC_2,
    ]);
    assert!(rejected
        .spans
        .iter()
        .all(|span| span.kind != StructureKind::Toc));
}

#[test]
fn mixed_blank_and_continuation_gap_uses_one_total_two_line_budget() {
    let result = detect(&[
        "ΠΕΡΙΕΧΟΜΕΝΑ",
        TOC_1,
        "",
        "",
        "ΜΕΡΟΣ ΠΡΩΤΟ ........",
        "ΚΥΡΙΑ ΕΝΟΤΗΤΑ ........",
        TOC_2,
    ]);
    assert!(result
        .spans
        .iter()
        .all(|span| span.kind != StructureKind::Toc));
}

#[test]
fn soft_gap_token_budget_rejects_two_oversized_bibliography_tails() {
    let long_tail = format!("{} https://example.org/item", "x ".repeat(40));
    let text = format!("BIBLIOGRAPHY\n{AY_1}\n{long_tail}\n{long_tail}\n{AY_2}");
    let result = structural_detect(
        "synthetic-doc",
        "adversarial-test",
        &text,
        &StructuralConfig::default(),
    );
    assert_eq!(
        result.line_evidence[2].bib_role,
        LineRole::PossibleContinuation
    );
    assert!(result
        .spans
        .iter()
        .all(|span| span.kind != StructureKind::Bibliography));
}

#[test]
fn hard_barrier_insertion_splits_independent_toc_blocks() {
    let result = detect(&[
        "ΠΕΡΙΕΧΟΜΕΝΑ",
        TOC_1,
        TOC_2,
        "## ΚΕΦΑΛΑΙΟ 2",
        "TABLE OF CONTENTS",
        "3. Μέθοδος ........ 15",
        "4. Αποτελέσματα ........ 29",
    ]);
    let spans: Vec<_> = result
        .spans
        .iter()
        .filter(|span| span.kind == StructureKind::Toc)
        .collect();
    assert_eq!(spans.len(), 2);
    assert_eq!((spans[0].line_start, spans[0].line_end), (0, 2));
    assert_eq!(spans[0].terminated_by, Some(3));
    assert!(spans[0]
        .reason_codes
        .contains(&ReasonCode::TerminatedByHardBarrier));
    assert_eq!((spans[1].line_start, spans[1].line_end), (4, 6));
}

#[test]
fn repeated_structural_headers_are_anchors_but_page_footers_are_not() {
    let repeated_heading = detect(&["ΠΕΡΙΕΧΟΜΕΝΑ", TOC_1, "ΠΕΡΙΕΧΟΜΕΝΑ", TOC_2]);
    let span = only_span(&repeated_heading, StructureKind::Toc);
    assert_eq!(span.supporting_lines, vec![0, 1, 2, 3]);

    let footer = detect(&["ΠΕΡΙΕΧΟΜΕΝΑ", TOC_1, "— 12 —", TOC_2]);
    assert_eq!(footer.line_evidence[2].toc_role, LineRole::Other);
    assert!(footer
        .spans
        .iter()
        .all(|span| span.kind != StructureKind::Toc));

    let furniture_only = detect(&[
        "ΠΑΝΕΠΙΣΤΗΜΙΟ ΑΘΗΝΩΝ",
        "— 1 —",
        "ΠΑΝΕΠΙΣΤΗΜΙΟ ΑΘΗΝΩΝ",
        "— 2 —",
    ]);
    assert!(furniture_only.spans.is_empty());
}

#[test]
fn cv_and_notes_scopes_survive_neutral_insertions_and_explicitly_rearm() {
    for scope in ["ΔΗΜΟΣΙΕΥΣΕΙΣ", "ΣΗΜΕΙΩΣΕΙΣ"] {
        let suppressed = detect(&[scope, AY_1, "ουδέτερη γραμμή", AY_2, AY_3, AY_4]);
        assert!(suppressed
            .spans
            .iter()
            .all(|span| span.kind != StructureKind::Bibliography));
        assert!(suppressed.line_evidence[3]
            .reason_codes
            .iter()
            .any(|reason| {
                matches!(
                    reason,
                    ReasonCode::HardCvSection | ReasonCode::HardNotesSection
                )
            }));

        let rearmed = detect(&[
            scope,
            AY_1,
            "ουδέτερη γραμμή",
            AY_2,
            AY_3,
            AY_4,
            "BIBLIOGRAPHY",
            AY_1,
            AY_2,
        ]);
        let span = only_span(&rearmed, StructureKind::Bibliography);
        assert_eq!((span.line_start, span.line_end), (6, 8));
    }
}

#[test]
fn bibliography_citation_families_confirm_coherent_blocks() {
    let author_year = detect(&["ΒΙΒΛΙΟΓΡΑΦΙΑ", AY_1, AY_2]);
    assert_eq!(
        author_year.line_evidence[1].bib_style,
        Some(BibStyle::AuthorYear)
    );
    assert_eq!(
        only_span(&author_year, StructureKind::Bibliography).line_end,
        2
    );

    let numeric = detect(&[
        "REFERENCES",
        "[1] Smith, J. (2020). Alpha. https://doi.org/10.1234/alpha",
        "[2] Brown, K. (2021). Beta. https://doi.org/10.1234/beta",
    ]);
    assert_eq!(numeric.line_evidence[1].bib_style, Some(BibStyle::Numeric));
    assert_eq!(only_span(&numeric, StructureKind::Bibliography).line_end, 2);

    let legal = detect(&[
        "ΝΟΜΟΘΕΣΙΑ",
        "Ν. 1234/2019. Ελληνική νομοθεσία.",
        "Ν. 4567/2020. Ελληνική νομοθεσία.",
    ]);
    assert_eq!(legal.line_evidence[1].bib_style, Some(BibStyle::Legal));
    assert_eq!(only_span(&legal, StructureKind::Bibliography).line_end, 2);
}

#[test]
fn greek_and_english_headings_have_equivalent_local_roles() {
    for heading in ["Πίνακας Περιεχομένων", "TABLE OF CONTENTS"] {
        let result = detect(&[heading]);
        assert_eq!(result.line_evidence[0].toc_role, LineRole::Heading);
    }
    for heading in ["Βιβλιογραφία", "BIBLIOGRAPHY"] {
        let result = detect(&[heading]);
        assert_eq!(result.line_evidence[0].bib_role, LineRole::Heading);
    }
}

#[test]
fn conflicts_remain_fail_closed_after_prefix_insertion() {
    let ambiguous = [
        "[1] Smith (2018). First title ........ 7",
        "[2] Brown (2019). Second title ........ 15",
        "[3] Jones (2020). Third title ........ 29",
        "[4] White (2021). Fourth title ........ 40",
    ];
    let baseline = detect(&[
        "ΠΕΡΙΕΧΟΜΕΝΑ",
        ambiguous[0],
        ambiguous[1],
        ambiguous[2],
        ambiguous[3],
    ]);
    assert!(baseline.spans.is_empty());
    assert_eq!(baseline.conflicts.len(), 1);
    assert_eq!(
        baseline.conflicts[0].reason_code,
        ReasonCode::ConflictFailClosed
    );

    let prefixed = detect(&[
        "ordinary preface",
        "ΠΕΡΙΕΧΟΜΕΝΑ",
        ambiguous[0],
        ambiguous[1],
        ambiguous[2],
        ambiguous[3],
    ]);
    assert!(prefixed.spans.is_empty());
    assert_eq!(prefixed.conflicts.len(), 1);
    assert_eq!(prefixed.conflicts[0].toc_line_start, 1);
    assert_eq!(prefixed.conflicts[0].bib_line_start, 2);

    let reduced = detect(&["ΠΕΡΙΕΧΟΜΕΝΑ", ambiguous[0], ambiguous[1], ambiguous[2]]);
    assert!(reduced.conflicts.is_empty());
    assert_eq!(reduced.spans.len(), 1);
    assert_eq!(reduced.spans[0].kind, StructureKind::Toc);
}

#[test]
fn headerless_chronologies_and_literature_review_prose_are_not_bibliographies() {
    let chronology = detect(&[
        "1. Το 2018 ξεκίνησε το πρώτο έργο.",
        "2. Το 2019 συνεχίστηκε η δεύτερη φάση.",
        "3. Το 2020 ολοκληρώθηκε η τρίτη φάση.",
        "4. Το 2021 άρχισε το νέο πρόγραμμα.",
    ]);
    assert!(chronology
        .spans
        .iter()
        .all(|span| span.kind != StructureKind::Bibliography));

    let review = detect(&[
        "Smith (2020) argues that this policy failed in practice.",
        "Jones (2021) explains the different outcome in Greece.",
        "Brown (2019) instead supports the older interpretation.",
        "White (2022) rejects both accounts in the final analysis.",
    ]);
    assert!(review
        .spans
        .iter()
        .all(|span| span.kind != StructureKind::Bibliography));
}

#[test]
fn ambiguous_sources_heading_does_not_authorize_a_chronology() {
    for heading in ["Sources", "ΠΗΓΕΣ"] {
        let result = detect(&[
            heading,
            "1. Το 2018 ξεκίνησε το πρώτο έργο.",
            "2. Το 2019 συνεχίστηκε η δεύτερη φάση.",
        ]);
        assert!(result
            .spans
            .iter()
            .all(|span| span.kind != StructureKind::Bibliography));
    }
}
