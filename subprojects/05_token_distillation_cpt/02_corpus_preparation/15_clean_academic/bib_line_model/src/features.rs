//! Deterministic per-line features — the 35 counts the line model consumes.
//!
//! Port of `eval/sequence_models/bibliography_v2.py::_feature_spans`. The order of
//! `FEATURE_NAMES` is load-bearing: it is the column order of the feature matrix,
//! so it must match `bibliography_feature_explorer.FEATURE_SPECS` exactly.
//!
//! Two properties of the Python original the port has to keep:
//!
//! * **Span ownership.** Detectors do not count independently; a span claimed by a
//!   stronger detector is removed from the weaker ones (`_without_overlaps`). So a
//!   year inside an access-date is not also a standalone year. Porting the regexes
//!   without the arbitration would inflate the counts.
//! * **Counts are span counts**, not match counts — overlapping matches of the same
//!   detector collapse.
//!
//! Progress is measured by `tests/fixture_parity.rs` against values generated from
//! the Python extractor; unimplemented features return 0 and show up there as gaps.

use once_cell::sync::Lazy;
use regex::Regex;

/// Column order of the count block. Must equal `FEATURE_SPECS` in
/// `bibliography_feature_explorer.py`.
pub const FEATURE_NAMES: [&str; 35] = [
    "year_count",
    "no_date_count",
    "numeric_date_count",
    "month_date_count",
    "access_date_count",
    "url_count",
    "doi_count",
    "isbn_count",
    "issn_count",
    "initial_count",
    "proper_name_word_count",
    "inverted_author_count",
    "name_initial_pair_count",
    "direct_author_count",
    "ampersand_count",
    "numbered_entry_count",
    "quoted_span_count",
    "editor_term_count",
    "thesis_term_count",
    "in_container_count",
    "edition_term_count",
    "dotted_word_count",
    "dotted_sequence_count",
    "volume_marker_count",
    "volume_shape_count",
    "journal_year_volume_count",
    "page_marker_count",
    "article_page_range_count",
    "page_range_count",
    "publisher_term_count",
    "place_name_count",
    "place_publisher_shape_count",
    "punctuation_count",
    "prose_lead_count",
    "table_row_count",
];

pub const N_FEATURES: usize = FEATURE_NAMES.len();

#[inline]
fn index_of(name: &str) -> usize {
    FEATURE_NAMES
        .iter()
        .position(|n| *n == name)
        .expect("unknown feature name")
}

// --- detectors that do not participate in span arbitration ---------------

/// `structure:table_row` — a markdown table row. Python counts this as a
/// whole-line property, not a span.
static TABLE_ROW: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*\|").unwrap());

/// Punctuation is counted per character, after stronger detectors have claimed
/// their spans (`bibliography_v2.py` builds it last, from the residue).
const PUNCTUATION: &str = ".,;:()[]«»“”\"";

/// Compute the count row for one line.
///
/// Currently a partial port: features whose detector is not yet written return 0
/// and are reported as gaps by the parity test rather than silently passing.
pub fn line_counts(text: &str) -> [u32; N_FEATURES] {
    let mut counts = [0u32; N_FEATURES];

    if TABLE_ROW.is_match(text) {
        counts[index_of("table_row_count")] = 1;
    }

    // NOTE: punctuation in Python is the residue after span arbitration, so this
    // is only correct once the claiming detectors exist. Left unset deliberately
    // — a wrong value is worse than a reported gap.

    let _ = PUNCTUATION;
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_order_is_stable() {
        assert_eq!(FEATURE_NAMES.len(), 35);
        assert_eq!(FEATURE_NAMES[0], "year_count");
        assert_eq!(FEATURE_NAMES[N_FEATURES - 1], "table_row_count");
    }

    #[test]
    fn table_row_is_detected() {
        let c = line_counts("| Βιβλιογραφία .......... 325 |");
        assert_eq!(c[index_of("table_row_count")], 1);
        let c = line_counts("Alchian, A.A. (1972). Production.");
        assert_eq!(c[index_of("table_row_count")], 0);
    }
}
