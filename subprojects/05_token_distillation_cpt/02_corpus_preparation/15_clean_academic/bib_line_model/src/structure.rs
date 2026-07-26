//! The bibliography-heading lexicon match — `structure:bib_heading_lexicon`.
//!
//! Port of `bibliography_nextgen_table.py::bib_heading_lexicon_match`, which asks
//! `deterministic_structure.analyze_bib_line` whether a line is a HEADING or a
//! SUBHEADING and, if not, retries a few rewrites of it.
//!
//! `analyze_bib_line` is 200-odd lines, but only two of its branches can return
//! HEADING or SUBHEADING and both are exact membership tests of `_heading_key(...)`
//! in a frozen lexicon — every other branch yields a different role. So the whole
//! dependency reduces to `_heading_key` plus two sets, and the rest of
//! `deterministic_structure` (the TOC and block decoders) is not on this path.
//!
//! `_heading_key` needs `str.casefold()`, which Rust does not have. It is not
//! interchangeable with `to_lowercase` here: Rust's `str::to_lowercase`
//! special-cases word-final sigma to ς, whereas casefold maps every sigma to σ —
//! and `_fold` depends on that uniformity, because it re-derives the final form
//! itself with an explicit `σ(?=\W|$)` rewrite. Folding therefore goes through
//! Python's own per-codepoint casefold map.

use crate::patterns::PATTERNS;
use crate::unicode::TABLES;
use once_cell::sync::Lazy;
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;

/// Union of `_BIB_HEADINGS` and `_BIB_SUBHEADINGS`: the keys for which
/// `analyze_bib_line` returns HEADING or SUBHEADING.
static HEADING_OR_SUBHEADING: Lazy<HashSet<String>> = Lazy::new(|| {
    let mut set: HashSet<String> = HashSet::new();
    set.extend(PATTERNS.lexicons.bib_headings.iter().cloned());
    set.extend(PATTERNS.lexicons.bib_subheadings.iter().cloned());
    set
});

static EXTRA_BIB_SUBHEADINGS: Lazy<HashSet<String>> =
    Lazy::new(|| PATTERNS.lexicons.extra_bib_subheadings.iter().cloned().collect());

/// Python `str.casefold()`, from Python's own table.
fn casefold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match TABLES.casefold.get(&(ch as u32)) {
            Some(folded) => out.push_str(folded),
            None => out.push(ch),
        }
    }
    out
}

/// `_fold` — casefold, NFKD, drop combining marks, then restore word-final sigma.
fn fold(value: &str) -> String {
    let decomposed: String = casefold(value).nfkd().collect();
    let stripped: String = decomposed
        .chars()
        .filter(|ch| !TABLES.combining.contains(*ch))
        .collect();
    PATTERNS
        .get("DS_FINAL_SIGMA")
        .replace_all(&stripped, "ς")
        .into_owned()
}

/// `_heading_key` — strip decorative leading/trailing punctuation, fold, and
/// collapse whitespace runs.
pub fn heading_key(value: &str) -> String {
    let stripped = PATTERNS.get("DS_HEADING_STRIP").replace_all(value, "");
    let folded = fold(&stripped);
    PATTERNS
        .get("DS_WHITESPACE_RUN")
        .replace_all(&folded, " ")
        .trim()
        .to_string()
}

/// Whether `analyze_bib_line(text).role` is HEADING or SUBHEADING.
pub fn is_heading_or_subheading(text: &str) -> bool {
    let normalized = crate::features::normalize(text);
    let stripped = crate::shape::py_strip(&normalized);
    if stripped.is_empty() {
        // The blank branch returns POSSIBLE_CONTINUATION before any lexicon test.
        return false;
    }
    HEADING_OR_SUBHEADING.contains(&heading_key(stripped))
}

/// `bib_heading_lexicon_match`.
///
/// `extra_lexicon` is true in the deployed configuration.
pub fn bib_heading_lexicon_match(text: &str, extra_lexicon: bool) -> bool {
    if is_heading_or_subheading(text) {
        return true;
    }
    let body = PATTERNS.get("TABLE_ATX_PREFIX").replace_all(text, "").into_owned();
    let section_stripped = PATTERNS
        .get("TABLE_SECTION_PREFIX")
        .replace_all(text, "")
        .into_owned();
    let wrappers = PATTERNS.get("TABLE_WRAPPERS");
    let candidates = [
        section_stripped.clone(),
        wrappers.replace_all(&body, "").into_owned(),
        wrappers.replace_all(&section_stripped, "").into_owned(),
    ];

    let body_stripped = crate::shape::py_strip(&body);
    for candidate in &candidates {
        let candidate = crate::shape::py_strip(candidate);
        // A rewrite that changed nothing has already been tested as `text`.
        if candidate.is_empty() || candidate == body_stripped {
            continue;
        }
        if is_heading_or_subheading(&format!("## {candidate}")) {
            return true;
        }
    }
    if extra_lexicon {
        if EXTRA_BIB_SUBHEADINGS.contains(&heading_key(&body)) {
            return true;
        }
        for candidate in &candidates {
            if EXTRA_BIB_SUBHEADINGS.contains(&heading_key(candidate)) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexicons_loaded() {
        assert!(!HEADING_OR_SUBHEADING.is_empty());
        assert!(!EXTRA_BIB_SUBHEADINGS.is_empty());
    }

    #[test]
    fn folds_greek_accents_and_final_sigma() {
        // casefold + NFKD + drop combining => unaccented; final sigma restored.
        assert_eq!(heading_key("ΒΙΒΛΙΟΓΡΑΦΙΑ"), "βιβλιογραφια");
        assert_eq!(heading_key("## Πηγές"), "πηγες");
    }

    #[test]
    fn plain_heading_matches() {
        assert!(bib_heading_lexicon_match("## ΒΙΒΛΙΟΓΡΑΦΙΑ", true));
        assert!(bib_heading_lexicon_match("## References", true));
    }

    #[test]
    fn section_numbered_heading_matches_via_rewrite() {
        // `_heading_key` strips an ATX prefix but not a section number, so this only
        // matches through the _SECTION_PREFIX candidate.
        assert!(bib_heading_lexicon_match("## 5. References", true));
    }

    #[test]
    fn ordinary_prose_does_not_match() {
        assert!(!bib_heading_lexicon_match(
            "Σύμφωνα με τα διαθέσιμα στοιχεία, το μερίδιο αυξήθηκε.",
            true
        ));
        assert!(!bib_heading_lexicon_match("", true));
    }
}
