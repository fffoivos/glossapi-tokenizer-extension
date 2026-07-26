//! The context builder — the 106 columns between the table and the line model.
//!
//! Port of `bibliography_nextgen_models.build_context_features`. The line model is
//! fitted on **232** features, not the 126 of the table: eight probability signals
//! summarised over three radii in both directions (max and mean each), plus ten
//! document- and segment-level positions.
//!
//! Everything it reads is already verified, so this stage is pure rearrangement —
//! but three details decide whether it is the right rearrangement:
//!
//! * Windows never cross a **physical gap**; the summaries run per segment, while the
//!   document-relative positions are measured against the whole document.
//! * The above-window is `[index-radius, index)` and the below-window is
//!   `(index, index+radius]` — the line itself is in neither.
//! * The anchor distances are a running scan clamped at 31, not the bounded search
//!   `_nearest_anchor` performs for the connector; same idea, different function.

use crate::gaps::pairwise_sum;
use crate::table::N_COLUMNS;

/// `CONTEXT_SIGNALS` — column indices into the 126-wide row, in order.
/// entry, signal_tcn, continuation_specialist, continuation, filler,
/// bib_header, bib_subheader, non_bib_header.
pub const CONTEXT_SIGNALS: [usize; 8] = [0, 1, 3, 4, 5, 6, 7, 8];
/// `CONTEXT_RADII`.
pub const CONTEXT_RADII: [usize; 3] = [1, 3, 8];
/// `structure:markdown_heading`.
const MARKDOWN_HEADING: usize = 121;
/// `probability:entry`.
const ENTRY: usize = 0;
const ENTRY_THRESHOLD: f32 = 0.25;

pub const N_CONTEXT: usize = CONTEXT_SIGNALS.len() * 2 * CONTEXT_RADII.len() * 2 + 10;
pub const N_LINE_MODEL_FEATURES: usize = N_COLUMNS + N_CONTEXT;

#[inline]
fn max_of(values: &[f32]) -> f32 {
    values.iter().copied().fold(0.0f32, f32::max)
}

#[inline]
fn mean_of(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        (pairwise_sum(&values.iter().map(|v| *v as f64).collect::<Vec<_>>()) / values.len() as f64)
            as f32
    }
}

/// Append the 106 context columns to every row of one document.
///
/// `rows` is the document's 126-column table; `abs_indices` its absolute line
/// indices, used only to find the physical segment boundaries.
pub fn build_context(rows: &[Vec<f32>], abs_indices: &[i64]) -> Vec<Vec<f32>> {
    let doc_length = rows.len();
    let doc_denominator = (doc_length.max(2) - 1) as f64;
    let log_doc_length = ((doc_length.max(1)) as f64).ln_1p();

    let mut out: Vec<Vec<f32>> = rows
        .iter()
        .map(|r| {
            let mut v = Vec::with_capacity(N_LINE_MODEL_FEATURES);
            v.extend_from_slice(r);
            v
        })
        .collect();

    for (segment_start, segment_end) in crate::tcn::physical_segments(abs_indices) {
        let local = &rows[segment_start..segment_end];
        let len = local.len();
        let entry: Vec<f32> = local.iter().map(|r| r[ENTRY]).collect();
        let markdown: Vec<f32> = local.iter().map(|r| r[MARKDOWN_HEADING]).collect();

        // Running distance to the nearest anchor either side, clamped at 31.
        let mut above_distance = vec![31.0f32; len];
        let mut previous: Option<usize> = None;
        for index in 0..len {
            if entry[index] >= ENTRY_THRESHOLD {
                previous = Some(index);
            }
            if let Some(p) = previous {
                above_distance[index] = (index - p).min(31) as f32;
            }
        }
        let mut below_distance = vec![31.0f32; len];
        let mut following: Option<usize> = None;
        for index in (0..len).rev() {
            if entry[index] >= ENTRY_THRESHOLD {
                following = Some(index);
            }
            if let Some(f) = following {
                below_distance[index] = (f - index).min(31) as f32;
            }
        }

        // Per-signal columns, materialised once per segment rather than per line.
        let signals: Vec<Vec<f32>> = CONTEXT_SIGNALS
            .iter()
            .map(|c| local.iter().map(|r| r[*c]).collect())
            .collect();

        for index in 0..len {
            let document_offset = segment_start + index;
            let remaining = doc_length - document_offset - 1;
            let mut values: Vec<f32> = Vec::with_capacity(N_CONTEXT);
            for signal in &signals {
                for direction in [-1i32, 1] {
                    for radius in CONTEXT_RADII {
                        let (low, high) = if direction < 0 {
                            (index.saturating_sub(radius), index)
                        } else {
                            ((index + 1).min(len), (index + radius + 1).min(len))
                        };
                        let selected = &signal[low..high];
                        values.push(max_of(selected));
                        values.push(mean_of(selected));
                    }
                }
            }
            values.extend([
                above_distance[index],
                below_distance[index],
                max_of(&markdown[index.saturating_sub(3)..index]),
                max_of(&markdown[(index + 1).min(len)..(index + 4).min(len)]),
                (document_offset as f64 / doc_denominator) as f32,
                (remaining as f64 / doc_denominator) as f32,
                ((document_offset as f64).ln_1p() / log_doc_length) as f32,
                ((remaining as f64).ln_1p() / log_doc_length) as f32,
                log_doc_length as f32,
                (index as f64 / (len.max(2) - 1) as f64) as f32,
            ]);
            debug_assert_eq!(values.len(), N_CONTEXT);
            out[segment_start + index].extend(values);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_matches_the_fitted_line_model() {
        assert_eq!(N_CONTEXT, 106);
        assert_eq!(N_LINE_MODEL_FEATURES, 232);
    }

    #[test]
    fn context_columns_are_appended_to_every_row() {
        let rows: Vec<Vec<f32>> = (0..4).map(|_| vec![0f32; N_COLUMNS]).collect();
        let abs: Vec<i64> = (0..4).collect();
        let out = build_context(&rows, &abs);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|r| r.len() == N_LINE_MODEL_FEATURES));
        assert!(out.iter().all(|r| r.iter().all(|v| v.is_finite())));
    }
}
