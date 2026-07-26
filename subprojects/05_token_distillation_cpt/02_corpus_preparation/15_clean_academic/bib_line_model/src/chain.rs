//! The upstream probability stages — the ten columns the line model reads first.
//!
//! Fold aggregation follows `_batched_predict`: sum `predict_proba[:, 1]` over the
//! five folds in **float64**, divide by the fold count, then cast to **float32**.
//! Doing the division in f32, or averaging logits instead of probabilities, would
//! both be plausible and both wrong.

use crate::artifacts::Model;
use crate::features::N_FEATURES;
use crate::predict::model_proba;

/// The P0D entry model's input: `concat((counts > 0), log1p(counts))`, 70 columns,
/// **unscaled**.
///
/// This is exactly columns 10..80 of the v3 row — the presence and log1p blocks that
/// are already verified bit-exact against the deployed table — so the entry stage
/// needs no feature work of its own.
pub fn entry_matrix(counts: &[u32; N_FEATURES]) -> Vec<f64> {
    let mut row = Vec::with_capacity(2 * N_FEATURES);
    row.extend(counts.iter().map(|c| if *c > 0 { 1.0f64 } else { 0.0 }));
    // Python builds this block as float32 (`np.log1p(counts.astype(np.float32))`)
    // and the model then sees float32 values, so the cast happens here too rather
    // than computing log1p in f64.
    row.extend(counts.iter().map(|c| (*c as f32).ln_1p() as f64));
    row
}

/// `probability:entry` — the five-fold mean.
pub fn entry_probability(folds: &[Model], counts: &[u32; N_FEATURES]) -> f32 {
    let features = entry_matrix(counts);
    let mut total = 0.0f64;
    for model in folds {
        total += model_proba(model, &features);
    }
    (total / folds.len() as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_matrix_is_the_presence_log1p_block() {
        let mut counts = [0u32; N_FEATURES];
        counts[0] = 3;
        let row = entry_matrix(&counts);
        assert_eq!(row.len(), 70);
        assert_eq!(row[0], 1.0);
        assert_eq!(row[1], 0.0);
        assert_eq!(row[N_FEATURES], (3.0f32).ln_1p() as f64);
    }
}

// ---------------------------------------------------------------------------
// heading bundle
// ---------------------------------------------------------------------------

use crate::artifacts::HeadingFold;
use crate::patterns::PATTERNS;
use crate::predict::Tfidf;
use crate::shape::{line_shape, py_strip, N_SHAPE};
use crate::unicode as u;

/// `HEADING_PROBABILITY_COLUMNS` — any-heading, then the three typed columns.
pub const N_HEADING_COLUMNS: usize = 4;
/// `HEADING_NUMERIC_NAMES` — the 34 shape values plus 9 context scalars.
pub const N_HEADING_NUMERIC: usize = N_SHAPE + 9;

/// `broad_heading_candidate` — the high-recall gate. Only candidates are scored;
/// every other line keeps a probability of exactly zero.
pub fn broad_heading_candidate(text: &str, previous_blank: bool, next_blank: bool) -> bool {
    let normalized = crate::features::normalize(text);
    let stripped = py_strip(&normalized);
    if stripped.is_empty() {
        return false;
    }
    if crate::structure::is_heading_or_subheading(text) {
        return true;
    }
    let tokens: Vec<&str> = PATTERNS
        .get("ROLE_WORD")
        .find_iter(stripped)
        .flatten()
        .map(|m| m.as_str())
        .collect();
    // Both limits are in characters, not bytes.
    if stripped.chars().count() > 200 || tokens.len() > 24 {
        return false;
    }
    let letters: Vec<char> = stripped.chars().filter(|c| u::py_isalpha(*c)).collect();
    let upper_fraction = if letters.is_empty() {
        0.0
    } else {
        letters.iter().filter(|c| u::py_isupper(**c)).count() as f64 / letters.len() as f64
    };
    let lexical: Vec<&&str> = tokens
        .iter()
        .filter(|t| t.chars().any(u::py_isalpha))
        .collect();
    let title_like = !lexical.is_empty() && {
        let leading_upper = lexical
            .iter()
            .filter(|t| t.chars().next().map_or(false, u::py_isupper))
            .count();
        leading_upper as f64 / lexical.len() as f64 >= 0.6
    };
    let structural = matches!(
        PATTERNS.get("ROLE_NUMBERED_HEADING").find(stripped),
        Ok(Some(m)) if m.start() == 0
    ) || stripped.starts_with('#');
    let isolated = previous_blank || next_blank;
    let non_sentence = stripped
        .chars()
        .next_back()
        .map_or(false, |c| !crate::shape::is_sentence_terminal(c));
    structural || upper_fraction >= 0.75 || (isolated && non_sentence && title_like)
}

/// `heading_numeric_features` — 34 shape values plus 9 scalars summarising the
/// entry probabilities in a 30-line window either side, clipped to the document.
pub fn heading_numeric_features(
    text: &str,
    previous_blank: bool,
    next_blank: bool,
    position_fraction: f64,
    above: &[f32],
    below: &[f32],
) -> [f32; N_HEADING_NUMERIC] {
    let mut out = [0f32; N_HEADING_NUMERIC];
    out[..N_SHAPE].copy_from_slice(&line_shape(text));
    // `max(initial=0.0)` — an empty window yields 0.0, not -inf.
    let max_of = |v: &[f32]| v.iter().copied().fold(0.0f32, f32::max);
    // numpy's `.mean()` on an empty array would warn and yield NaN, so Python
    // guards it with `if len(...)`.
    let mean_of = |v: &[f32]| {
        if v.is_empty() {
            0.0f32
        } else {
            (crate::gaps::pairwise_sum(&v.iter().map(|x| *x as f64).collect::<Vec<_>>())
                / v.len() as f64) as f32
        }
    };
    let over = |v: &[f32]| v.iter().filter(|x| **x >= 0.25).count() as f32;
    out[N_SHAPE] = previous_blank as u8 as f32;
    out[N_SHAPE + 1] = next_blank as u8 as f32;
    out[N_SHAPE + 2] = position_fraction as f32;
    out[N_SHAPE + 3] = max_of(above);
    out[N_SHAPE + 4] = max_of(below);
    out[N_SHAPE + 5] = mean_of(above);
    out[N_SHAPE + 6] = mean_of(below);
    out[N_SHAPE + 7] = over(above);
    out[N_SHAPE + 8] = over(below);
    out
}

/// Score one heading candidate with one fold.
///
/// `HeadingTransform.apply` hstacks `[char_tfidf | word_tfidf | scaled_numeric]` as
/// float32, so the values the linear models see are float32 even though the
/// coefficients are float64 and the dot product therefore accumulates in float64.
fn heading_fold_predict(
    fold: &HeadingFold,
    char_tfidf: &Tfidf,
    word_tfidf: &Tfidf,
    text: &str,
    numeric: &[f32; N_HEADING_NUMERIC],
) -> [f64; N_HEADING_COLUMNS] {
    let char_offset = 0usize;
    let word_offset = fold.char_tfidf.n_features;
    let numeric_offset = word_offset + fold.word_tfidf.n_features;

    let mut sparse: Vec<(usize, f64)> = Vec::new();
    for (i, v) in char_tfidf.transform(text) {
        sparse.push((char_offset + i, v as f32 as f64));
    }
    for (i, v) in word_tfidf.transform(text) {
        sparse.push((word_offset + i, v as f32 as f64));
    }
    let mut scaled: Vec<f64> = numeric.iter().map(|v| *v as f64).collect();
    fold.numeric_scaler.apply(&mut scaled);
    for (i, v) in scaled.iter().enumerate() {
        // The hstack is float32; keep the same rounding before the dot product.
        let value = *v as f32;
        if value != 0.0 {
            sparse.push((numeric_offset + i, value as f64));
        }
    }

    let any = crate::predict::linear_sparse_scores(&fold.any_model, &sparse);
    let typed = crate::predict::linear_sparse_scores(&fold.type_model, &sparse);
    // Binary logistic: P(class 1). Typed: softmax over the three heading types,
    // then multiplied by the any-heading probability (they are conditional).
    let any_probability = *any.last().unwrap_or(&0.0);
    let mut out = [0f64; N_HEADING_COLUMNS];
    out[0] = any_probability;
    for (k, p) in typed.iter().enumerate().take(N_HEADING_COLUMNS - 1) {
        out[k + 1] = p * any_probability;
    }
    out
}

/// The five-fold mean of the four heading columns for one candidate line.
pub fn heading_probabilities(
    folds: &[HeadingFold],
    char_tfidf: &[Tfidf],
    word_tfidf: &[Tfidf],
    text: &str,
    numeric: &[f32; N_HEADING_NUMERIC],
) -> [f32; N_HEADING_COLUMNS] {
    let mut total = [0f64; N_HEADING_COLUMNS];
    for (k, fold) in folds.iter().enumerate() {
        let local = heading_fold_predict(fold, &char_tfidf[k], &word_tfidf[k], text, numeric);
        for (acc, v) in total.iter_mut().zip(local.iter()) {
            *acc += v;
        }
    }
    let mut out = [0f32; N_HEADING_COLUMNS];
    for (o, t) in out.iter_mut().zip(total.iter()) {
        *o = (*t / folds.len() as f64) as f32;
    }
    out
}

#[cfg(test)]
mod heading_tests {
    use super::*;

    #[test]
    fn numeric_block_is_shape_plus_nine() {
        assert_eq!(N_HEADING_NUMERIC, 43);
        assert_eq!(crate::shape::LINE_SHAPE_NAMES.len() + 9, N_HEADING_NUMERIC);
    }

    #[test]
    fn empty_window_summaries_are_zero_not_nan() {
        let f = heading_numeric_features("## ΒΙΒΛΙΟΓΡΑΦΙΑ", true, true, 0.5, &[], &[]);
        assert!(f.iter().all(|v| v.is_finite()));
        assert_eq!(f[N_SHAPE + 3], 0.0);
        assert_eq!(f[N_SHAPE + 5], 0.0);
    }

    #[test]
    fn headings_are_candidates_and_prose_is_not() {
        assert!(broad_heading_candidate("## ΒΙΒΛΙΟΓΡΑΦΙΑ", true, true));
        assert!(!broad_heading_candidate(
            "Σύμφωνα με τα διαθέσιμα στοιχεία όσον αφορά στην εικοσαετία 1990-2009, το μερίδιο των πετρελαιοειδών αυξήθηκε σημαντικά και συνεχίζει.",
            false, false
        ));
        assert!(!broad_heading_candidate("   ", false, false));
    }
}
