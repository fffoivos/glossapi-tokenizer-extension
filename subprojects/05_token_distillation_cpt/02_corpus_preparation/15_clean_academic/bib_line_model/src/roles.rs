//! The deterministic negative-role taxonomy — 9 of the signal TCN's 10 inputs.
//!
//! Three layers, ported in the order they call each other:
//!
//! 1. `deterministic_structure.analyze_bib_line` — the base verdict. Only its
//!    `hard_negative` flag and `reason_codes` matter here; the scoring, styles and
//!    entry/year extraction feed other consumers and are computed only where a
//!    hard-negative branch depends on them.
//! 2. `bibliography_v2.analyze_bibliography_line_v2` — wraps the base, adds four
//!    hard-negative branches of its own, and can *override* two of the base's.
//! 3. `bibliography_deterministic_roles._analyze_document` — turns the reason codes
//!    into one of eight mutually exclusive roles by substring match (`_role_index`).
//!
//! The substring matching in step 3 is order-sensitive: a line carrying both
//! `STATISTICAL_TABLE` and `FOOTNOTE` is table, because "TABLE" is tested first. The
//! cascade below preserves that order literally.

use crate::features::Line;
use crate::patterns::PATTERNS;
use crate::shape::py_strip;
use crate::structure::heading_key;
use crate::unicode as u;
use once_cell::sync::Lazy;
use std::collections::HashSet;

/// `ROLE_NAMES` in `bibliography_deterministic_roles`.
pub const ROLE_NAMES: [&str; 8] = [
    "figure_caption",
    "table_or_equation",
    "exact_negative_scope_heading",
    "generic_markdown_heading",
    "footnote",
    "running_or_enumerated_prose",
    "legal_procedure",
    "other_explicit_negative",
];

pub const N_ROLES: usize = ROLE_NAMES.len();

/// `stripped.endswith((".", "!", "?", ";", "·"))` in the running-prose test.
///
/// This `·` is **U+00B7 MIDDLE DOT** — the opposite choice from the otherwise
/// identical-looking set in `line_shape`, which uses U+0387 GREEK ANO TELEIA. Here
/// U+00B7 is the reachable one, because the text has already been NFKC-normalized
/// and NFKC maps U+0387 onto U+00B7. Two files, the same glyph, different
/// codepoints, and opposite consequences; both are transcribed as escapes.
const SENTENCE_END: [char; 5] = ['\u{2e}', '\u{21}', '\u{3f}', '\u{3b}', '\u{b7}'];

fn lexicon(name: &str) -> &'static HashSet<String> {
    static CACHE: Lazy<std::collections::HashMap<String, HashSet<String>>> = Lazy::new(|| {
        PATTERNS
            .lexicons
            .all
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect()
    });
    CACHE
        .get(name)
        .unwrap_or_else(|| panic!("lexicon {name} missing from patterns.json"))
}

#[inline]
fn matches_at_start(name: &str, text: &str) -> bool {
    matches!(PATTERNS.get(name).find(text), Ok(Some(m)) if m.start() == 0)
}

#[inline]
fn searches(name: &str, text: &str) -> bool {
    PATTERNS.get(name).is_match(text).unwrap_or(false)
}

/// `_token_count` — `len(re.findall(r"\S+", value))`, where `\S` is the complement
/// of Python's `str.isspace()`.
fn token_count(value: &str) -> usize {
    let mut count = 0usize;
    let mut in_run = false;
    for ch in value.chars() {
        if u::py_isspace(ch) {
            in_run = false;
        } else if !in_run {
            in_run = true;
            count += 1;
        }
    }
    count
}

/// What the role stage needs from `analyze_bib_line`.
struct BaseVerdict {
    hard_negative: bool,
    /// Only the codes containing "NEGATIVE"; the v2 overrides compare this as a set
    /// and `_role_index` only ever matches negative markers.
    negative_codes: Vec<&'static str>,
}

/// `analyze_bib_line`, reduced to its hard-negative verdict.
///
/// The non-negative branches (HEADING, SUBHEADING, the scored entry roles) all
/// return `hard_negative=False`, so they collapse to a single "not negative" answer.
fn base_verdict(line: &Line) -> BaseVerdict {
    let none = || BaseVerdict {
        hard_negative: false,
        negative_codes: Vec::new(),
    };
    let stripped = py_strip(&line.normalized);
    if stripped.is_empty() {
        return none();
    }
    let key = heading_key(stripped);
    if lexicon("bib_headings").contains(&key) || lexicon("bib_subheadings").contains(&key) {
        return none();
    }
    if lexicon("cv_headings").contains(&key) {
        return BaseVerdict {
            hard_negative: true,
            negative_codes: vec!["BIB_NEGATIVE_CV_PUBLICATIONS_HEADING"],
        };
    }
    if lexicon("notes_headings").contains(&key) {
        return BaseVerdict {
            hard_negative: true,
            negative_codes: vec!["BIB_NEGATIVE_NOTES_HEADING"],
        };
    }
    if matches_at_start("DS_ATX_HEADING", stripped) {
        return BaseVerdict {
            hard_negative: true,
            negative_codes: vec!["BIB_NEGATIVE_NONSTRUCTURAL_MARKDOWN_HEADING"],
        };
    }

    let folded = crate::structure::fold(stripped);
    let mut hard: Vec<&'static str> = Vec::new();
    if matches_at_start("DS_BODY_HEADING", &folded) {
        hard.push("BIB_NEGATIVE_BODY_HEADING");
    }
    // The pipe test is a whole-line shape, not a pattern: two or more pipes plus any
    // digit is a statistical table row.
    if searches("DS_STATISTICAL", stripped)
        || (stripped.matches('|').count() >= 2 && stripped.chars().any(|c| c.is_ascii_digit()))
    {
        hard.push("BIB_NEGATIVE_STATISTICAL_TABLE");
    }
    if searches("DS_EQUATION", stripped) {
        hard.push("BIB_NEGATIVE_EQUATION");
    }
    let numbered = matches_at_start("DS_BIB_NUMBER", stripped);
    if matches_at_start("DS_FOOTNOTE", stripped) && !numbered {
        hard.push("BIB_NEGATIVE_FOOTNOTE");
    }
    if searches("DS_INLINE_CITATION", &folded) {
        hard.push("BIB_NEGATIVE_INLINE_CITATION_PROSE");
    }
    if matches_at_start("DS_NARRATIVE_AUTHOR_YEAR", stripped) {
        hard.push("BIB_NEGATIVE_NARRATIVE_AUTHOR_YEAR_PROSE");
    }
    if matches_at_start("DS_LEGAL_START", &folded) && !searches("DS_BIB_LEGAL", stripped) {
        hard.push("BIB_NEGATIVE_LEGAL_PROCEDURE");
    }

    // The running-prose test needs the author/number/legal evidence, so compute just
    // those three rather than the whole entry analysis.
    let number_end = PATTERNS
        .get("DS_BIB_NUMBER")
        .find(stripped)
        .ok()
        .flatten()
        .filter(|m| m.start() == 0)
        .map(|m| m.end());
    let author_text = match number_end {
        Some(end) => &stripped[end..],
        None => stripped,
    };
    let author_inverted = matches_at_start("DS_BIB_AUTHOR_COMMA", author_text);
    let author_year = matches_at_start("DS_BIB_AUTHOR_YEAR", author_text);
    let author = author_inverted || author_year;
    let legal = searches("DS_BIB_LEGAL", stripped);
    let tokens = token_count(stripped);

    if tokens >= 16 && !author && number_end.is_none() && !legal {
        let sentence_like = stripped
            .chars()
            .next_back()
            .map_or(false, |c| SENTENCE_END.contains(&c))
            || stripped.matches(',').count() >= 2;
        if sentence_like {
            hard.push("BIB_NEGATIVE_RUNNING_PROSE");
        }
    }

    BaseVerdict {
        hard_negative: !hard.is_empty(),
        negative_codes: hard,
    }
}

/// `_has_date_evidence`.
fn has_date_evidence(line: &Line) -> bool {
    use crate::features::*;
    let c = &line.counts;
    c[F_YEAR] > 0
        || c[F_NO_DATE] > 0
        || c[F_NUMERIC_DATE] > 0
        || c[F_MONTH_DATE] > 0
        || c[F_JOURNAL_YEAR_VOLUME] > 0
}

/// The role index, or `None` when the line is not a hard negative.
///
/// This is `analyze_bibliography_line_v2`'s hard-negative cascade followed by
/// `_role_index`, fused: the intermediate reason-code tuples exist only to be
/// substring-matched, so each branch resolves straight to its role.
pub fn negative_role(text: &str, line: &Line) -> Option<usize> {
    use crate::features::*;
    let c = &line.counts;
    let stripped = py_strip(&line.normalized);
    if stripped.is_empty() {
        return None;
    }
    // `stripped.lstrip("#").strip()` — strips only '#', then whitespace.
    let heading_text = py_strip(stripped.trim_start_matches('#'));

    let key = heading_key(stripped);
    if lexicon("bib_headings").contains(&key) || lexicon("bib_subheadings").contains(&key) {
        return None;
    }
    let full = |name: &str, s: &str| match PATTERNS.get(name).find(s) {
        Ok(Some(m)) => m.start() == 0 && m.end() == s.len(),
        _ => false,
    };
    if full("_BIB_EXTENDED_HEADING", heading_text) || full("_BIB_EXTENDED_SUBHEADING", heading_text)
    {
        return None;
    }
    // `_BIB_HEADING_WORD.fullmatch(NFKD(stripped).casefold().replace("́", ""))`
    // — note this is NFKD + casefold with only the combining acute removed, which is
    // not the same normalisation `_heading_key` performs.
    if full("_BIB_HEADING_WORD", &crate::structure::nfkd_casefold_deacute(stripped)) {
        return None;
    }

    let strong_table_citation = c[F_TABLE_ROW] > 0
        && has_date_evidence(line)
        && (c[F_INVERTED_AUTHOR] > 0 || c[F_NAME_INITIAL_PAIR] >= 2);
    let identifier_specific = c[F_DOI] > 0 || c[F_ISBN] > 0 || c[F_ISSN] > 0;

    // Resolve to a role directly; `_role_index` would reach the same answer by
    // substring-matching the code these branches emit.
    let auxiliary = || {
        if lexicon("auxiliary_scope_headings").contains(&heading_key(text)) {
            Some(2)
        } else {
            None
        }
    };

    if c[F_TABLE_ROW] > 0 && !strong_table_citation && !identifier_specific {
        // BIB2_NEGATIVE_NONCITATION_TABLE_ROW contains "TABLE" -> role 1.
        return auxiliary().or(Some(1));
    }
    let author_specific = c[F_INVERTED_AUTHOR] > 0 || c[F_NAME_INITIAL_PAIR] >= 2;
    if matches_at_start("_FIGURE_CAPTION_START", stripped) {
        return auxiliary().or(Some(0));
    }
    let tokens = crate::features::token_count(&line.normalized);
    if tokens >= 22
        && matches_at_start("_ENUMERATED_PROSE_START", stripped)
        && !author_specific
        && !identifier_specific
    {
        // BIB2_NEGATIVE_LONG_ENUMERATED_PROSE -> role 5.
        return auxiliary().or(Some(5));
    }

    let base = base_verdict(line);
    if !base.hard_negative {
        return None;
    }
    let only = |code: &str| base.negative_codes.len() == 1 && base.negative_codes[0] == code;
    let override_running_prose = only("BIB_NEGATIVE_RUNNING_PROSE")
        && (c[F_NAME_INITIAL_PAIR] >= 2
            || c[F_DOI] > 0
            || c[F_ISBN] > 0
            || (c[F_JOURNAL_YEAR_VOLUME] > 0
                && (c[F_PAGE_MARKER] > 0 || c[F_ARTICLE_PAGE_RANGE] > 0 || c[F_PAGE_RANGE] > 0))
            || ((c[F_PUBLISHER_TERM] > 0 || c[F_PLACE_PUBLISHER_SHAPE] > 0)
                && (has_date_evidence(line)
                    || c[F_PAGE_MARKER] > 0
                    || c[F_ARTICLE_PAGE_RANGE] > 0
                    || c[F_PAGE_RANGE] > 0)));
    let override_statistical_table =
        strong_table_citation && only("BIB_NEGATIVE_STATISTICAL_TABLE");
    if override_running_prose || override_statistical_table {
        return None;
    }
    auxiliary().or_else(|| Some(role_index(&base.negative_codes)))
}

/// `_role_index` — first marker wins, in this order.
fn role_index(codes: &[&str]) -> usize {
    let joined = codes.join(" ");
    if joined.contains("FIGURE_CAPTION") {
        return 0;
    }
    if joined.contains("TABLE") || joined.contains("EQUATION") {
        return 1;
    }
    for marker in [
        "CV_PUBLICATIONS_HEADING",
        "NOTES_HEADING",
        "AUXILIARY_HEADING",
        "BODY_HEADING",
    ] {
        if joined.contains(marker) {
            return 2;
        }
    }
    if joined.contains("NONSTRUCTURAL_MARKDOWN_HEADING") {
        return 3;
    }
    if joined.contains("FOOTNOTE") {
        return 4;
    }
    for marker in [
        "RUNNING_PROSE",
        "INLINE_CITATION_PROSE",
        "NARRATIVE_AUTHOR_YEAR_PROSE",
        "LONG_ENUMERATED_PROSE",
    ] {
        if joined.contains(marker) {
            return 5;
        }
    }
    if joined.contains("LEGAL_PROCEDURE") {
        return 6;
    }
    7
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::analyze;

    fn role_of(text: &str) -> Option<usize> {
        negative_role(text, &analyze(text))
    }

    #[test]
    fn role_names_are_the_reference_order() {
        assert_eq!(ROLE_NAMES[0], "figure_caption");
        assert_eq!(ROLE_NAMES[7], "other_explicit_negative");
    }

    #[test]
    fn table_wins_over_footnote_by_order() {
        // _role_index tests "TABLE" before "FOOTNOTE"; order is load-bearing.
        assert_eq!(role_index(&["BIB_NEGATIVE_STATISTICAL_TABLE", "BIB_NEGATIVE_FOOTNOTE"]), 1);
        assert_eq!(role_index(&["BIB_NEGATIVE_FOOTNOTE"]), 4);
        assert_eq!(role_index(&["BIB_NEGATIVE_LEGAL_PROCEDURE"]), 6);
        assert_eq!(role_index(&["BIB_SOMETHING_ELSE"]), 7);
    }

    #[test]
    fn bibliography_headings_are_not_negative() {
        assert_eq!(role_of("## ΒΙΒΛΙΟΓΡΑΦΙΑ"), None);
        assert_eq!(role_of(""), None);
    }

    #[test]
    fn a_bare_markdown_heading_is_a_generic_heading() {
        assert_eq!(role_of("## Εισαγωγή"), Some(3));
    }

    #[test]
    fn token_count_splits_on_python_whitespace() {
        assert_eq!(token_count("  a  bb   ccc "), 3);
        assert_eq!(token_count(""), 0);
    }
}
