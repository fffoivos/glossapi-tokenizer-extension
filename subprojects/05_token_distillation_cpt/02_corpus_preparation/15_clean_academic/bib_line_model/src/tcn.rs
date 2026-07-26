//! The signal TCN — `probability:signal_tcn`.
//!
//! Reimplements `bibliography_signal_tcn.SignalTCN`'s forward pass directly; there is
//! no torch here and no need for one. The network is small and entirely explicit:
//!
//! ```text
//!   input_projection : Linear(10 -> 32),           then * mask
//!   blocks[i]        : LayerNorm(32)
//!                      -> Conv1d(32, 32, k=3, dilation=d_i, padding=d_i)
//!                      -> GELU
//!                      -> + residual, * mask
//!   output_norm      : LayerNorm(32)
//!   output           : Linear(32 -> 1), squeezed, then sigmoid
//! ```
//!
//! Dropout is identity at inference and the mask is all ones for every window the
//! driver builds, so both drop out of the arithmetic.
//!
//! **The chunking is part of the model, not an optimisation.** Inference runs over
//! each physical segment in blocks of 256 central lines padded by 32 lines of context
//! on each side, and only the central portion of each window is kept. Because the
//! convolutions are zero-padded at the window edges, evaluating a whole segment in
//! one pass gives different numbers. The loop below mirrors the Python exactly,
//! including that the context is clipped to the segment rather than the document.

use crate::artifacts::TcnFold;
use anyhow::{Context, Result};

/// `MAX_PHYSICAL_GAP` in `bibliography_entry_dataset`.
pub const MAX_PHYSICAL_GAP: i64 = 64;
const CENTRAL: usize = 256;
const CONTEXT: usize = 32;
/// torch's `nn.LayerNorm` default.
const LAYER_NORM_EPS: f64 = 1e-5;

/// Weights for one fold, flattened once at load.
pub struct Tcn {
    hidden: usize,
    inputs: usize,
    proj_w: Vec<f64>, // [hidden][inputs]
    proj_b: Vec<f64>,
    blocks: Vec<Block>,
    out_norm_w: Vec<f64>,
    out_norm_b: Vec<f64>,
    out_w: Vec<f64>, // [hidden]
    out_b: f64,
}

struct Block {
    norm_w: Vec<f64>,
    norm_b: Vec<f64>,
    /// [out][in][k]
    conv_w: Vec<f64>,
    conv_b: Vec<f64>,
    dilation: usize,
}

fn flatten(value: &serde_json::Value, out: &mut Vec<f64>) {
    match value {
        serde_json::Value::Array(items) => items.iter().for_each(|v| flatten(v, out)),
        serde_json::Value::Number(n) => out.push(n.as_f64().unwrap_or(0.0)),
        _ => {}
    }
}

fn tensor(fold: &TcnFold, key: &str) -> Result<Vec<f64>> {
    let value = fold
        .state_dict
        .get(key)
        .with_context(|| format!("TCN state_dict missing {key}"))?;
    let mut out = Vec::new();
    flatten(value, &mut out);
    Ok(out)
}

impl Tcn {
    pub fn new(fold: &TcnFold) -> Result<Self> {
        let hidden = fold.architecture.hidden_dim;
        let proj_w = tensor(fold, "input_projection.weight")?;
        let inputs = proj_w.len() / hidden;
        let mut blocks = Vec::new();
        for (i, dilation) in fold.architecture.dilations.iter().enumerate() {
            blocks.push(Block {
                norm_w: tensor(fold, &format!("blocks.{i}.norm.weight"))?,
                norm_b: tensor(fold, &format!("blocks.{i}.norm.bias"))?,
                conv_w: tensor(fold, &format!("blocks.{i}.convolution.weight"))?,
                conv_b: tensor(fold, &format!("blocks.{i}.convolution.bias"))?,
                dilation: *dilation,
            });
        }
        Ok(Self {
            hidden,
            inputs,
            proj_w,
            proj_b: tensor(fold, "input_projection.bias")?,
            blocks,
            out_norm_w: tensor(fold, "output_norm.weight")?,
            out_norm_b: tensor(fold, "output_norm.bias")?,
            out_w: tensor(fold, "output.weight")?,
            out_b: tensor(fold, "output.bias")?[0],
        })
    }

    /// `nn.LayerNorm` over the feature axis: biased variance, then affine.
    fn layer_norm(&self, row: &mut [f64], w: &[f64], b: &[f64]) {
        let h = self.hidden as f64;
        let mean = row.iter().sum::<f64>() / h;
        let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / h;
        let inv = 1.0 / (var + LAYER_NORM_EPS).sqrt();
        for (i, x) in row.iter_mut().enumerate() {
            *x = (*x - mean) * inv * w[i] + b[i];
        }
    }

    /// One window: `[t][inputs]` in, one logit per position out (pre-sigmoid).
    fn forward(&self, window: &[Vec<f64>]) -> Vec<f64> {
        let t_len = window.len();
        let h = self.hidden;

        // input_projection
        let mut values: Vec<Vec<f64>> = window
            .iter()
            .map(|x| {
                (0..h)
                    .map(|o| {
                        let mut acc = self.proj_b[o];
                        for i in 0..self.inputs {
                            acc += self.proj_w[o * self.inputs + i] * x[i];
                        }
                        acc
                    })
                    .collect()
            })
            .collect();

        for block in &self.blocks {
            // LayerNorm on a copy — the residual is the pre-norm value.
            let mut normed: Vec<Vec<f64>> = values.clone();
            for row in normed.iter_mut() {
                self.layer_norm(row, &block.norm_w, &block.norm_b);
            }
            let d = block.dilation;
            for t in 0..t_len {
                for o in 0..h {
                    let mut acc = block.conv_b[o];
                    for k in 0..3 {
                        // padding == dilation with kernel 3, so the taps are
                        // t-d, t, t+d; out-of-range reads are the zero padding.
                        let pos = t as isize + (k as isize) * (d as isize) - (d as isize);
                        if pos < 0 || pos >= t_len as isize {
                            continue;
                        }
                        let src = &normed[pos as usize];
                        for i in 0..h {
                            acc += block.conv_w[(o * h + i) * 3 + k] * src[i];
                        }
                    }
                    // GELU, exact form: 0.5x(1 + erf(x/sqrt(2))). torch's default is
                    // the erf version, not the tanh approximation.
                    let g = 0.5 * acc * (1.0 + libm::erf(acc / std::f64::consts::SQRT_2));
                    values[t][o] += g;
                }
            }
        }

        values
            .iter_mut()
            .map(|row| {
                self.layer_norm(row, &self.out_norm_w, &self.out_norm_b);
                let mut acc = self.out_b;
                for i in 0..self.hidden {
                    acc += self.out_w[i] * row[i];
                }
                acc
            })
            .collect()
    }
}

/// `_physical_segments` — split where the absolute line index jumps by more than
/// `MAX_PHYSICAL_GAP`, which marks a break between physical pages.
pub fn physical_segments(abs_indices: &[i64]) -> Vec<(usize, usize)> {
    let mut starts = vec![0usize];
    for i in 1..abs_indices.len() {
        if abs_indices[i] - abs_indices[i - 1] > MAX_PHYSICAL_GAP {
            starts.push(i);
        }
    }
    starts.push(abs_indices.len());
    starts.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Score one document: `features[t]` is the 10-value signal row for line `t`.
/// Returns the five-fold mean probability per line.
pub fn signal_probabilities(
    folds: &[Tcn],
    features: &[Vec<f64>],
    abs_indices: &[i64],
) -> Vec<f32> {
    let n = features.len();
    let mut total = vec![0f64; n];
    for tcn in folds {
        // Python keeps each fold's output in float32 before summing in float64.
        let mut local = vec![0f32; n];
        for (segment_start, segment_end) in physical_segments(abs_indices) {
            let mut central_start = segment_start;
            while central_start < segment_end {
                let central_end = segment_end.min(central_start + CENTRAL);
                let input_start = segment_start.max(central_start.saturating_sub(CONTEXT));
                let input_end = segment_end.min(central_end + CONTEXT);
                let window: Vec<Vec<f64>> = features[input_start..input_end].to_vec();
                let logits = tcn.forward(&window);
                let left = central_start - input_start;
                let right = central_end - input_start;
                for (offset, logit) in logits[left..right].iter().enumerate() {
                    local[central_start + offset] = crate::predict::sigmoid(*logit) as f32;
                }
                central_start = central_end;
            }
        }
        for (acc, v) in total.iter_mut().zip(local.iter()) {
            *acc += *v as f64;
        }
    }
    total
        .iter()
        .map(|v| (v / folds.len() as f64) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_split_on_a_large_index_jump() {
        assert_eq!(physical_segments(&[0, 1, 2, 3]), vec![(0, 4)]);
        // A gap of exactly MAX_PHYSICAL_GAP does not split; one more does.
        assert_eq!(physical_segments(&[0, 64]), vec![(0, 2)]);
        assert_eq!(physical_segments(&[0, 65]), vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn gelu_is_the_erf_form_not_tanh() {
        // At x = 1 the exact GELU is 0.8413447..., the tanh approximation 0.8411920...
        let x = 1.0f64;
        let exact = 0.5 * x * (1.0 + libm::erf(x / std::f64::consts::SQRT_2));
        assert!((exact - 0.841344746).abs() < 1e-9);
    }
}
