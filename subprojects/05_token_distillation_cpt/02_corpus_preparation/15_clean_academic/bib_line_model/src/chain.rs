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
