//! Gap summaries and the structure flags — the last 12 deterministic columns.
//!
//! Port of `bibliography_positional_features.py::_gap_summaries` (7 values) and the
//! five `structure:` flags assembled in `bibliography_nextgen_table.py`.
//!
//! Two things make this more than a transcription:
//!
//! * **Character offsets, not bytes.** Every gap value is a position or a length
//!   divided by `len(normalized)`, which in Python is a count of code points. The
//!   count features could stay in byte space because they only ever compare spans;
//!   these divide by the total, so the span endpoints are converted to character
//!   indices first.
//! * **numpy's summation order.** `lengths.sum()` and `lengths.mean()` use pairwise
//!   summation, which is not left-to-right addition once an array reaches 8
//!   elements. `pairwise_sum` below reproduces it rather than assuming the
//!   difference washes out in the f32 cast.

use crate::features::{self, Span, F_PROSE_LEAD, F_PUNCTUATION, F_TABLE_ROW};
use crate::patterns::PATTERNS;

pub const GAP_SUMMARY_NAMES: [&str; 7] = [
    "unmatched_fraction",
    "unmatched_prefix_fraction",
    "unmatched_suffix_fraction",
    "longest_unmatched_fraction",
    "longest_unmatched_center",
    "unmatched_run_count",
    "mean_unmatched_run_fraction",
];

pub const N_GAPS: usize = GAP_SUMMARY_NAMES.len();

pub const STRUCTURE_NAMES: [&str; 5] = [
    "markdown_heading",
    "image_marker",
    "table_row",
    "rule_line",
    "bib_heading_lexicon",
];

pub const N_STRUCTURE: usize = STRUCTURE_NAMES.len();

/// numpy's `pairwise_sum_DOUBLE`, which `ndarray.sum()` and `.mean()` use.
///
/// Below 8 elements it is an ordinary loop; from 8 to 128 it accumulates into eight
/// partial sums and combines them as a balanced tree; above that it splits. That is
/// a different association order from left-to-right addition, so a naive sum can
/// differ in the last bits — usually invisible after the f32 cast, but "usually" is
/// not the bar this port is held to.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    const BLOCKSIZE: usize = 128;
    let n = a.len();
    if n < 8 {
        return a.iter().sum();
    }
    if n <= BLOCKSIZE {
        let mut r = [0f64; 8];
        r.copy_from_slice(&a[..8]);
        let limit = n - (n % 8);
        let mut i = 8;
        while i < limit {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for v in &a[limit..] {
            res += *v;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// `_merge_intervals` — sort, drop empties, coalesce touching or overlapping runs.
fn merge_intervals(intervals: &[Span]) -> Vec<Span> {
    let mut ordered: Vec<Span> = intervals.iter().copied().filter(|(a, b)| a < b).collect();
    ordered.sort_unstable();
    let mut merged: Vec<Span> = Vec::with_capacity(ordered.len());
    for (start, end) in ordered {
        match merged.last_mut() {
            // Python merges when `start <= last.end`, so abutting runs coalesce.
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// `_complement` — the stretches of the line no semantic detector claimed.
fn complement(intervals: &[Span], length: usize) -> Vec<Span> {
    let mut result = Vec::new();
    let mut cursor = 0usize;
    for (start, end) in merge_intervals(intervals) {
        debug_assert!(end <= length, "feature span lies outside normalized text");
        if cursor < start {
            result.push((cursor, start));
        }
        cursor = cursor.max(end);
    }
    if cursor < length {
        result.push((cursor, length));
    }
    result
}

/// `_gap_summaries`.
fn gap_summaries(intervals: &[Span], length: usize) -> [f32; N_GAPS] {
    if length == 0 || intervals.is_empty() {
        return [0.0; N_GAPS];
    }
    let lengths: Vec<f64> = intervals.iter().map(|(a, b)| (b - a) as f64).collect();
    // `np.argmax` returns the first maximum; `max_by` on ties returns the last, so
    // the comparison is written to keep the earlier index.
    let mut longest_index = 0usize;
    for (i, v) in lengths.iter().enumerate() {
        if *v > lengths[longest_index] {
            longest_index = i;
        }
    }
    let (longest_start, longest_end) = intervals[longest_index];
    let len_f = length as f64;
    let first = intervals[0];
    let last = intervals[intervals.len() - 1];
    let total = pairwise_sum(&lengths);
    let values: [f64; N_GAPS] = [
        total / len_f,
        if first.0 == 0 {
            (first.1 - first.0) as f64 / len_f
        } else {
            0.0
        },
        if last.1 == length {
            (last.1 - last.0) as f64 / len_f
        } else {
            0.0
        },
        lengths[longest_index] / len_f,
        (longest_start + longest_end) as f64 / (2.0 * len_f),
        intervals.len() as f64,
        (total / lengths.len() as f64) / len_f,
    ];
    let mut out = [0f32; N_GAPS];
    for (i, v) in values.iter().enumerate() {
        out[i] = *v as f32;
    }
    out
}

/// `SEMANTIC_UNION_EXCLUDED` — punctuation, prose leads and the table-row envelope
/// do not count as "explained" text when computing what is left unmatched.
#[inline]
fn excluded_from_semantic_union(feature: usize) -> bool {
    feature == F_PUNCTUATION || feature == F_PROSE_LEAD || feature == F_TABLE_ROW
}

/// Map byte offsets to character offsets for one normalized line.
///
/// Every span endpoint is on a character boundary, so a running count over
/// `char_indices` is enough; the vector is indexed by byte offset.
fn byte_to_char_map(text: &str) -> Vec<u32> {
    let mut map = vec![0u32; text.len() + 1];
    let mut chars = 0u32;
    for (byte, ch) in text.char_indices() {
        map[byte] = chars;
        chars += 1;
        // Interior bytes of a multi-byte character are never addressed, but filling
        // them keeps the lookup total rather than conditional.
        for slot in map.iter_mut().skip(byte + 1).take(ch.len_utf8() - 1) {
            *slot = chars;
        }
    }
    map[text.len()] = chars;
    map
}

/// The 7 gap values for one raw line.
pub fn line_gaps(text: &str) -> [f32; N_GAPS] {
    gaps_from(&features::analyze(text))
}

/// The same, from an already-computed [`features::Line`].
pub fn gaps_from(line: &features::Line) -> [f32; N_GAPS] {
    let normalized = line.normalized.as_str();
    let spans = &line.spans;
    let map = byte_to_char_map(normalized);
    let semantic: Vec<Span> = spans
        .iter()
        .enumerate()
        .filter(|(i, _)| !excluded_from_semantic_union(*i))
        .flat_map(|(_, v)| v.iter().copied())
        .map(|(a, b)| (map[a] as usize, map[b] as usize))
        .collect();
    let length = map[normalized.len()] as usize;
    gap_summaries(&complement(&semantic, length), length)
}

/// The 5 structure flags for one raw line.
///
/// Note these run on the **raw** line, not the NFKC-normalized one — the table
/// builder passes `text` straight to the markdown/image/rule patterns — while
/// `table_row` is read back out of the count features, which are normalized.
pub fn line_structure(text: &str, bib_heading_lexicon: bool) -> [f32; N_STRUCTURE] {
    structure_from(text, features::line_counts(text)[F_TABLE_ROW] > 0, bib_heading_lexicon)
}

/// The same, when the caller already knows whether the line is a table row.
pub fn structure_from(text: &str, table_row: bool, bib_heading_lexicon: bool) -> [f32; N_STRUCTURE] {
    let markdown = matches!(PATTERNS.get("TABLE_MARKDOWN_HEADING").find(text), Ok(Some(m)) if m.start() == 0);
    let image = matches!(PATTERNS.get("TABLE_IMAGE_MARKER").find(text), Ok(Some(m)) if m.start() == 0);
    let rule = match PATTERNS.get("TABLE_RULE_LINE").find(text) {
        Ok(Some(m)) => m.start() == 0 && m.end() == text.len(),
        _ => false,
    };
    [
        markdown as u8 as f32,
        image as u8 as f32,
        table_row as u8 as f32,
        rule as u8 as f32,
        // Gated on `markdown` in the table builder: a lexicon hit only counts when
        // the line is actually a markdown heading.
        (markdown && bib_heading_lexicon) as u8 as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_matches_naive_for_short_inputs() {
        let a: Vec<f64> = (1..8).map(|x| x as f64).collect();
        assert_eq!(pairwise_sum(&a), a.iter().sum::<f64>());
    }

    #[test]
    fn pairwise_uses_tree_association_past_eight() {
        // Values chosen so left-to-right and tree association genuinely differ.
        let mut a = vec![1e16, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        a.push(1.0);
        let naive: f64 = a.iter().sum();
        let pw = pairwise_sum(&a);
        assert_ne!(naive.to_bits(), pw.to_bits());
    }

    #[test]
    fn merge_coalesces_abutting_runs() {
        assert_eq!(merge_intervals(&[(0, 3), (3, 6), (8, 9)]), vec![(0, 6), (8, 9)]);
        // Empty spans are dropped before merging.
        assert_eq!(merge_intervals(&[(2, 2), (0, 1)]), vec![(0, 1)]);
    }

    #[test]
    fn complement_covers_the_unclaimed_tail() {
        assert_eq!(complement(&[(0, 3)], 10), vec![(3, 10)]);
        assert_eq!(complement(&[], 4), vec![(0, 4)]);
        assert_eq!(complement(&[(0, 4)], 4), Vec::<Span>::new());
    }

    #[test]
    fn byte_to_char_map_handles_multibyte() {
        let s = "αβγ";
        let m = byte_to_char_map(s);
        assert_eq!(m[0], 0);
        assert_eq!(m[2], 1);
        assert_eq!(m[4], 2);
        assert_eq!(m[6], 3);
    }

    #[test]
    fn blank_line_gaps_are_zero() {
        assert_eq!(line_gaps(""), [0.0; N_GAPS]);
    }
}
