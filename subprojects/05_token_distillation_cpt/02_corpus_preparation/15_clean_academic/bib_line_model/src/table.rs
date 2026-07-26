//! The 126-column feature row the line model consumes.
//!
//! Port of the row assembly in `bibliography_nextgen_table.py`. Column order is the
//! contract (`bibliography-nextgen-full-table-v3`) and is asserted against the
//! deployed receipt's `feature_names`:
//!
//! ```text
//!   0..10    probability:   entry, signal_tcn, connector, continuation_specialist,
//!                           continuation, filler, bib_header, bib_subheader,
//!                           non_bib_header, other
//!  10..45    presence:      count > 0, for each of the 35 count features
//!  45..80    log1p:         log1p(count)
//!  80..114   shape:         line_shape
//! 114..121   gap:           gap summaries
//! 121..126   structure:     markdown_heading, image_marker, table_row, rule_line,
//!                           bib_heading_lexicon
//! ```
//!
//! v3 deliberately stops there: the 12 citation-grammar columns measured +0.00125 AP
//! (noise) and the owner's decision was to deploy without them, so this port targets
//! 126 columns and does not carry them.
//!
//! The ten leading probabilities come from the upstream models. Everything from
//! column 10 on is deterministic and computable from the line alone, which is what
//! `deterministic_row` returns — it is the part that can be checked against the
//! deployed `features.npy` without loading a single fitted model.

use crate::features::{self, N_FEATURES};
use crate::gaps::{self, N_GAPS, N_STRUCTURE};
use crate::shape::{self, N_SHAPE};

/// Where the deterministic block starts: after the ten upstream probabilities.
pub const N_PROBABILITY: usize = 10;
pub const N_COLUMNS: usize = N_PROBABILITY + 2 * N_FEATURES + N_SHAPE + N_GAPS + N_STRUCTURE;

pub const PROBABILITY_NAMES: [&str; N_PROBABILITY] = [
    "entry",
    "signal_tcn",
    "connector",
    "continuation_specialist",
    "continuation",
    "filler",
    "bib_header",
    "bib_subheader",
    "non_bib_header",
    "other",
];

/// The full column-name list, in contract order.
pub fn feature_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_COLUMNS);
    names.extend(PROBABILITY_NAMES.iter().map(|n| format!("probability:{n}")));
    names.extend(features::FEATURE_NAMES.iter().map(|n| format!("presence:{n}")));
    names.extend(features::FEATURE_NAMES.iter().map(|n| format!("log1p:{n}")));
    names.extend(shape::LINE_SHAPE_NAMES.iter().map(|n| format!("shape:{n}")));
    names.extend(gaps::GAP_SUMMARY_NAMES.iter().map(|n| format!("gap:{n}")));
    names.extend(gaps::STRUCTURE_NAMES.iter().map(|n| format!("structure:{n}")));
    names
}

/// Columns 10..126 for one raw line — everything that does not need a fitted model.
///
/// `bib_heading_lexicon` is passed in rather than computed here because it depends on
/// the deterministic-structure lexicon, which is a separate stage.
pub fn deterministic_row(text: &str, bib_heading_lexicon: bool) -> Vec<f32> {
    let counts = features::line_counts(text);
    let mut row = Vec::with_capacity(N_COLUMNS - N_PROBABILITY);

    // presence, then log1p — Python builds them from the same count vector, in that
    // order, as two separate blocks rather than interleaved.
    row.extend(counts.iter().map(|c| if *c > 0 { 1.0f32 } else { 0.0f32 }));
    // `np.log1p` on a float32 array computes in float32.
    row.extend(counts.iter().map(|c| (*c as f32).ln_1p()));
    row.extend(shape::line_shape(text));
    row.extend(gaps::line_gaps(text));
    row.extend(gaps::line_structure(text, bib_heading_lexicon));
    debug_assert_eq!(row.len(), N_COLUMNS - N_PROBABILITY);
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_count_matches_the_v3_contract() {
        assert_eq!(N_COLUMNS, 126);
        assert_eq!(feature_names().len(), 126);
    }

    #[test]
    fn names_are_in_contract_order() {
        let n = feature_names();
        assert_eq!(n[0], "probability:entry");
        assert_eq!(n[9], "probability:other");
        assert_eq!(n[10], "presence:year_count");
        assert_eq!(n[45], "log1p:year_count");
        assert_eq!(n[80], "shape:char_length");
        assert_eq!(n[114], "gap:unmatched_fraction");
        assert_eq!(n[121], "structure:markdown_heading");
        assert_eq!(n[125], "structure:bib_heading_lexicon");
    }

    #[test]
    fn deterministic_row_is_the_tail_of_the_contract() {
        let row = deterministic_row("Smith, J. (2020). A title. Athens: Press.", false);
        assert_eq!(row.len(), 116);
        assert!(row.iter().all(|v| v.is_finite()));
    }
}
