//! The connector bundle — the last five probability columns.
//!
//! Port of `bibliography_role_features.connector_feature_row` (177 features) and the
//! driver loop in `_connector_probabilities`.
//!
//! Three properties of the original that are easy to lose:
//!
//! * **Non-candidates are not zero.** Lines outside the candidate window default to
//!   `(0, 0, 0, 1)` — `other = 1.0`.
//! * **`continuation_specialist` is not a fifth value.** It is `connector[:, 1]`
//!   copied, because no specialist model is configured
//!   (`continuation_specialist_policy: frozen_connector_continuation_fallback`).
//!   Verified against the deployed table: those two columns are bit-identical.
//! * **Every window walk stops at a physical gap**, a jump of more than
//!   `MAX_PHYSICAL_GAP` in the absolute line index. That is a page break, and
//!   neighbours across one are not neighbours.

use crate::features::{self, Line, N_FEATURES};
use crate::gaps::{N_GAPS};
use crate::shape::{line_shape, N_SHAPE};
use crate::tcn::MAX_PHYSICAL_GAP;

/// `WINDOW_RADII`.
pub const WINDOW_RADII: [usize; 5] = [1, 3, 5, 10, 30];
/// `PAIR_NAMES`.
pub const N_PAIR: usize = 9;
/// `CONNECTOR_PROBABILITY_COLUMNS`: connector, continuation, filler, other.
pub const N_CONNECTOR_COLUMNS: usize = 4;
pub const ENTRY_THRESHOLD: f32 = 0.25;
/// `candidate_window_mask(..., radius=30)`.
pub const CANDIDATE_RADIUS: i32 = 30;

/// 70 count + 34 shape + 7 gap + 4 anchor + 30 context + 18 pair + 8 joined
/// + 3 heading + 1 heading max + 2 = 177.
pub const N_CONNECTOR_FEATURES: usize =
    2 * N_FEATURES + N_SHAPE + N_GAPS + 4 + 2 * WINDOW_RADII.len() * 3 + 2 * N_PAIR + 8 + 3 + 1 + 2;

/// Per-line inputs the connector stage reads. Assembled once per document.
pub struct DocumentContext<'a> {
    pub texts: &'a [String],
    pub lines: &'a [Line],
    pub abs_indices: &'a [i64],
    pub entry: &'a [f32],
    /// `heading[:, 1:]` — the three typed columns, without the any-heading column.
    pub heading: &'a [[f32; 3]],
}

#[inline]
fn gap_between(abs: &[i64], low: usize, high: usize) -> i64 {
    abs[high] - abs[low]
}

/// `candidate_seed_distances` then `<= radius`.
///
/// A seed is a line whose entry probability clears the threshold or which the
/// heading stage nominated. Distance is measured forwards and backwards without
/// crossing a physical gap.
pub fn candidate_window_mask(
    entry: &[f32],
    heading_candidate: &[bool],
    abs: &[i64],
    radius: i32,
) -> Vec<bool> {
    let n = entry.len();
    let sentinel = i32::MAX;
    let mut distances = vec![sentinel; n];
    let seeds: Vec<bool> = (0..n)
        .map(|i| entry[i] >= ENTRY_THRESHOLD || heading_candidate[i])
        .collect();

    let mut previous: Option<usize> = None;
    for index in 0..n {
        if index > 0 && gap_between(abs, index - 1, index) > MAX_PHYSICAL_GAP {
            previous = None;
        }
        if seeds[index] {
            previous = Some(index);
        }
        if let Some(p) = previous {
            distances[index] = (index - p) as i32;
        }
    }
    let mut following: Option<usize> = None;
    for index in (0..n).rev() {
        if index + 1 < n && gap_between(abs, index, index + 1) > MAX_PHYSICAL_GAP {
            following = None;
        }
        if seeds[index] {
            following = Some(index);
        }
        if let Some(f) = following {
            distances[index] = distances[index].min((f - index) as i32);
        }
    }
    distances.iter().map(|d| *d <= radius).collect()
}

/// `_nearest_anchor` — distance to the closest line within 30 whose entry
/// probability clears the threshold, or `(31, 0.0)` if there is none.
fn nearest_anchor(entry: &[f32], abs: &[i64], index: usize, direction: i32) -> (f32, f32) {
    for distance in 1..=30i32 {
        let candidate = index as i32 + direction * distance;
        if candidate < 0 || candidate as usize >= entry.len() {
            break;
        }
        let candidate = candidate as usize;
        let (low, high) = if index <= candidate {
            (index, candidate)
        } else {
            (candidate, index)
        };
        // Any single step across the span larger than the gap ends the search — the
        // Python checks `diff(...) > MAX_PHYSICAL_GAP` over the whole slice, not just
        // the endpoints.
        if (low..high).any(|i| gap_between(abs, i, i + 1) > MAX_PHYSICAL_GAP) {
            break;
        }
        if entry[candidate] >= ENTRY_THRESHOLD {
            return (distance as f32, entry[candidate]);
        }
    }
    (31.0, 0.0)
}

/// `_window_values` — up to `radius` neighbours in one direction, stopping at a
/// physical gap. Order matters only for the mean's summation order.
fn window_values(entry: &[f32], abs: &[i64], index: usize, radius: usize, direction: i32) -> Vec<f32> {
    let mut selected = Vec::with_capacity(radius);
    let mut cursor = index as i32;
    for _ in 0..radius {
        let candidate = cursor + direction;
        if candidate < 0 || candidate as usize >= entry.len() {
            break;
        }
        let (low, high) = if cursor <= candidate {
            (cursor as usize, candidate as usize)
        } else {
            (candidate as usize, cursor as usize)
        };
        if gap_between(abs, low, high) > MAX_PHYSICAL_GAP {
            break;
        }
        selected.push(entry[candidate as usize]);
        cursor = candidate;
    }
    if direction < 0 {
        selected.reverse();
    }
    selected
}

/// `_pair_features` — nine shape comparisons between two adjacent lines.
fn pair_features(left: &str, right: &str) -> [f32; N_PAIR] {
    let l = line_shape(left);
    let r = line_shape(right);
    let left_length = (l[0] as f64).max(1.0);
    let right_length = (r[0] as f64).max(1.0);
    let joined = format!(
        "{} {}",
        crate::shape::py_rstrip_pub(&features::normalize(left)),
        crate::shape::py_lstrip_pub(&features::normalize(right))
    );
    let (mut open_p, mut close_p, mut open_b, mut close_b, mut quotes) = (0i64, 0i64, 0i64, 0i64, 0usize);
    for ch in joined.chars() {
        match ch {
            '(' => open_p += 1,
            ')' => close_p += 1,
            '[' => open_b += 1,
            ']' => close_b += 1,
            _ => {}
        }
        if matches!(
            ch,
            '\'' | '"' | '«' | '»' | '\u{201c}' | '\u{201d}' | '\u{2018}' | '\u{2019}'
        ) {
            quotes += 1;
        }
    }
    // Shape indices: 0 char_length, 6 leading_whitespace, 12 greek fraction,
    // 13 latin fraction, 18 starts_lowercase, 23 ends_opening_terminal.
    [
        (right_length / left_length) as f32,
        (r[6] - l[6]).abs(),
        (r[12] - l[12]).abs(),
        (r[13] - l[13]).abs(),
        l[23],
        r[18],
        (open_p - close_p).abs() as f32,
        (open_b - close_b).abs() as f32,
        (quotes % 2) as f32,
    ]
}

/// Which neighbour indices are reachable without crossing a physical gap.
fn neighbours(ctx: &DocumentContext, index: usize) -> (Option<usize>, Option<usize>) {
    let n = ctx.texts.len();
    let previous = if index > 0 && gap_between(ctx.abs_indices, index - 1, index) <= MAX_PHYSICAL_GAP
    {
        Some(index - 1)
    } else {
        None
    };
    let following = if index + 1 < n
        && gap_between(ctx.abs_indices, index, index + 1) <= MAX_PHYSICAL_GAP
    {
        Some(index + 1)
    } else {
        None
    };
    (previous, following)
}

/// The joined text of a line and one neighbour, as the driver builds it.
pub fn joined_text(texts: &[String], index: usize, neighbour: usize, left_first: bool) -> String {
    if left_first {
        format!(
            "{} {}",
            crate::shape::py_rstrip_pub(&texts[neighbour]),
            crate::shape::py_lstrip_pub(&texts[index])
        )
    } else {
        format!(
            "{} {}",
            crate::shape::py_rstrip_pub(&texts[index]),
            crate::shape::py_lstrip_pub(&texts[neighbour])
        )
    }
}

/// `connector_feature_row`.
///
/// `joined_score` supplies the P0D probability of each joined neighbour text; the
/// driver computes those in one deduplicated batch, so they are passed in rather
/// than recomputed here.
pub fn connector_feature_row(
    ctx: &DocumentContext,
    index: usize,
    candidate_mask: &[bool],
    joined_previous: Option<(f32, &Line)>,
    joined_next: Option<(f32, &Line)>,
) -> Vec<f32> {
    let n = ctx.texts.len();
    let counts = &ctx.lines[index].counts;
    let mut row: Vec<f32> = Vec::with_capacity(N_CONNECTOR_FEATURES);

    row.extend(counts.iter().map(|c| if *c > 0 { 1.0f32 } else { 0.0 }));
    row.extend(counts.iter().map(|c| (*c as f32).ln_1p()));
    row.extend(line_shape(&ctx.texts[index]));
    let own_gaps = crate::gaps::gaps_from(&ctx.lines[index]);
    row.extend(own_gaps);

    let (above_distance, above_probability) = nearest_anchor(ctx.entry, ctx.abs_indices, index, -1);
    let (below_distance, below_probability) = nearest_anchor(ctx.entry, ctx.abs_indices, index, 1);
    row.extend([above_distance, above_probability, below_distance, below_probability]);

    for direction in [-1i32, 1] {
        for radius in WINDOW_RADII {
            let values = window_values(ctx.entry, ctx.abs_indices, index, radius, direction);
            let max = values.iter().copied().fold(0.0f32, f32::max);
            let mean = if values.is_empty() {
                0.0f32
            } else {
                (crate::gaps::pairwise_sum(&values.iter().map(|v| *v as f64).collect::<Vec<_>>())
                    / values.len() as f64) as f32
            };
            let over = values.iter().filter(|v| **v >= ENTRY_THRESHOLD).count() as f32;
            row.extend([max, mean, over]);
        }
    }

    let (previous, following) = neighbours(ctx, index);
    match previous {
        Some(p) => row.extend(pair_features(&ctx.texts[p], &ctx.texts[index])),
        None => row.extend([0.0f32; N_PAIR]),
    }
    match following {
        Some(f) => row.extend(pair_features(&ctx.texts[index], &ctx.texts[f])),
        None => row.extend([0.0f32; N_PAIR]),
    }

    let base_score = ctx.entry[index];
    let nonzero_current = counts.iter().filter(|c| **c > 0).count() as i64;
    for (neighbour, joined) in [(previous, joined_previous), (following, joined_next)] {
        match (neighbour, joined) {
            (Some(nb), Some((joined_score, joined_line))) => {
                let neighbour_score = ctx.entry[nb];
                let nonzero_joined = joined_line.counts.iter().filter(|c| **c > 0).count() as i64;
                let nonzero_neighbour =
                    ctx.lines[nb].counts.iter().filter(|c| **c > 0).count() as i64;
                let distinct_gain =
                    (nonzero_joined - nonzero_current.max(nonzero_neighbour)).max(0) as f32;
                let neighbour_gaps = crate::gaps::gaps_from(&ctx.lines[nb]);
                let joined_gaps = crate::gaps::gaps_from(joined_line);
                let unmatched_gain =
                    (own_gaps[0].min(neighbour_gaps[0]) - joined_gaps[0]).max(0.0);
                row.extend([
                    joined_score,
                    joined_score - base_score.max(neighbour_score),
                    distinct_gain,
                    unmatched_gain,
                ]);
            }
            _ => row.extend([0.0f32; 4]),
        }
    }

    let heading = ctx.heading[index];
    row.extend(heading);
    row.push(heading.iter().copied().fold(0.0f32, f32::max));

    let inside_gap = (above_distance <= 30.0 && below_distance <= 30.0) as u8 as f32;
    let edge_distance = if candidate_mask[index] {
        let mut left_edge = index;
        while left_edge > 0 && candidate_mask[left_edge - 1] {
            left_edge -= 1;
        }
        let mut right_edge = index;
        while right_edge + 1 < n && candidate_mask[right_edge + 1] {
            right_edge += 1;
        }
        (index - left_edge).min(right_edge - index) as f32
    } else {
        0.0
    };
    row.extend([inside_gap, edge_distance]);

    debug_assert_eq!(row.len(), N_CONNECTOR_FEATURES);
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_width_matches_the_contract() {
        assert_eq!(N_CONNECTOR_FEATURES, 177);
    }

    #[test]
    fn candidate_mask_expands_by_radius_and_stops_at_gaps() {
        let entry = vec![0.0f32, 0.9, 0.0, 0.0];
        let heading = vec![false; 4];
        let abs: Vec<i64> = vec![0, 1, 2, 3];
        assert_eq!(
            candidate_window_mask(&entry, &heading, &abs, 1),
            vec![true, true, true, false]
        );
        // A page break isolates the far side even within the radius.
        let abs_gap: Vec<i64> = vec![0, 1, 200, 201];
        assert_eq!(
            candidate_window_mask(&entry, &heading, &abs_gap, 1),
            vec![true, true, false, false]
        );
    }

    #[test]
    fn nearest_anchor_reports_the_sentinel_when_none_is_reachable() {
        let entry = vec![0.0f32; 5];
        let abs: Vec<i64> = (0..5).collect();
        assert_eq!(nearest_anchor(&entry, &abs, 2, 1), (31.0, 0.0));
    }

    #[test]
    fn window_values_stop_at_a_physical_gap() {
        let entry = vec![0.1f32, 0.2, 0.3, 0.4];
        let abs: Vec<i64> = vec![0, 1, 500, 501];
        assert_eq!(window_values(&entry, &abs, 1, 3, 1), Vec::<f32>::new());
        assert_eq!(window_values(&entry, &abs, 0, 3, 1), vec![0.2f32]);
    }
}
