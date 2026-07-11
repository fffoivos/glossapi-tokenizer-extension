//! Frozen table-of-contents line head from eval/toc_line_lr_model.json.
//!
//! The head reuses the bibliography model's 22 base features, appends five cheap ToC signals, applies
//! the front gate used in training/evaluation, and decodes spans with the conservative operating point
//! from eval/struct_smooth_params.json.

use once_cell::sync::Lazy;
use regex::Regex;

use crate::reference_module::Span;
use crate::span_line_model::{base_features, FEATS as BASE_FEATS};

const FEATS: usize = 27;
pub const MODEL_ID: &str = "toc_line_lr_v1";
pub const DECODER_ID: &str = "front30pct300_hysteresis_hi0.5_lo0.3_gap8_lmin2";
const MU: [f64; FEATS] = [
    0.24535, 0.09646, 0.01305, 0.02029, 0.06425, 0.1103, 0.03258, 0.05216, 0.34169, 0.00142,
    0.23913, 0.01421, 0.07026, 0.24538, 0.2454, 0.24539, 0.20549, 0.11032, 0.96995, 0.09293,
    0.01179, 0.54782, 0.04335, 0.03303, 0.11358, 0.24367, 0.00066,
];
const SD: [f64; FEATS] = [
    0.43029, 0.29522, 0.11351, 0.141, 0.24519, 0.31326, 0.17754, 0.22234, 0.44758, 0.03761,
    0.42655, 0.11834, 0.25559, 0.3468, 0.32925, 0.32231, 0.32298, 0.22096, 0.14168, 0.29033,
    0.08553, 0.3309, 0.20365, 0.17872, 0.3173, 0.42929, 0.02575,
];
const W: [f64; FEATS] = [
    -0.10431, -0.03828, -0.036, 0.11491, -0.00385, -0.0403, -0.00324, -0.00846, -0.26224,
    0.0045, 0.03155, -0.02438, -0.00217, -0.1584, -0.16332, -0.16648, -0.23395, -0.07033,
    0.0186, -0.00575, 0.00387, -1.17196, 0.31619, 0.44248, 0.32882, 0.66668, 0.1332,
];
const BIAS: f64 = -4.80116;
const THETA_HI: f64 = 0.5;
const THETA_LO: f64 = 0.3;
const GAP: usize = 8;
const LMIN: usize = 2;

static ATX: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s{0,3}#{1,6}\s").unwrap());
static TOC_LEADER: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\.{4,}|_{4,}|·{4,}|(?:\.\s){4,}").unwrap());
static TOC_SECNUM: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*\|?\s*\d+(?:\.\d+){1,4}\.?\s").unwrap());
static TOC_PAGETAIL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[.\s…·]\s*\d{1,4}\s*\|?\s*$").unwrap());
static MD_ROW: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*\|.*\|").unwrap());
const TOC_HDR_STEMS: &[&str] = &[
    "περιεχόμενα",
    "περιεχομενα",
    "πίνακας περιεχ",
    "πινακας περιεχ",
    "contents",
    "table of contents",
];

fn own_features(line: &str) -> [f64; 5] {
    let low = line.to_lowercase();
    let headerish = ATX.is_match(line) || line.trim().chars().count() < 40;
    [
        if TOC_LEADER.is_match(line) { 1.0 } else { 0.0 },
        if TOC_SECNUM.is_match(line) { 1.0 } else { 0.0 },
        if TOC_PAGETAIL.is_match(line) { 1.0 } else { 0.0 },
        if MD_ROW.is_match(line) { 1.0 } else { 0.0 },
        if headerish && TOC_HDR_STEMS.iter().any(|stem| low.contains(stem)) {
            1.0
        } else {
            0.0
        },
    ]
}

/// Per-present-line ToC probability after the frozen front gate.
pub fn toc_scores(present: &[&str], abs_idx: &[usize], n_total: usize) -> Vec<f64> {
    let base = base_features(present, abs_idx, n_total);
    let cut = 300usize.min(((n_total as f64) * 0.30).floor() as usize);
    present
        .iter()
        .enumerate()
        .map(|(i, line)| {
            if abs_idx[i] >= cut {
                return 0.0;
            }
            let own = own_features(line);
            let mut score = BIAS;
            for j in 0..BASE_FEATS {
                score += W[j] * ((base[i][j] - MU[j]) / SD[j]);
            }
            for (j, value) in own.iter().enumerate() {
                let model_index = BASE_FEATS + j;
                score += W[model_index] * ((*value - MU[model_index]) / SD[model_index]);
            }
            1.0 / (1.0 + (-score).exp())
        })
        .collect()
}

fn hysteresis(probabilities: &[f64]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut in_run = false;
    let mut start = 0usize;
    for (i, probability) in probabilities.iter().enumerate() {
        if !in_run {
            if *probability >= THETA_HI {
                in_run = true;
                start = i;
            }
        } else if *probability < THETA_LO {
            runs.push((start, i - 1));
            in_run = false;
        }
    }
    if in_run {
        runs.push((start, probabilities.len() - 1));
    }
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in runs {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 + GAP + 1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
        .into_iter()
        .filter(|(start, end)| end - start + 1 >= LMIN)
        .collect()
}

/// Whole-document ToC span detection with character offsets in Unicode scalar-value units.
pub fn toc_detect(doc_id: &str, _source: &str, text: &str) -> Vec<Span> {
    let raw: Vec<&str> = text.split('\n').collect();
    let n_total = raw.len();
    let mut present: Vec<&str> = Vec::new();
    let mut abs_idx: Vec<usize> = Vec::new();
    let mut char_start: Vec<usize> = Vec::new();
    let mut cursor = 0usize;
    for (i, line) in raw.iter().enumerate() {
        if !line.trim().is_empty() {
            present.push(line);
            abs_idx.push(i);
            char_start.push(cursor);
        }
        cursor += line.chars().count() + 1;
    }
    if present.is_empty() {
        return Vec::new();
    }
    let probabilities = toc_scores(&present, &abs_idx, n_total);
    hysteresis(&probabilities)
        .into_iter()
        .map(|(start, end)| Span {
            doc_id: doc_id.to_string(),
            kind: "toc_span".to_string(),
            char_start: char_start[start],
            char_end: char_start[end] + present[end].chars().count(),
            line_start: abs_idx[start],
            line_end: abs_idx[end],
            trigger: present[start].chars().take(40).collect(),
            gated_by: format!("toc_lr p={:.3}", probabilities[start]),
            row_id: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toc_signals_match_expected_shapes() {
        assert_eq!(own_features("1.2 Εισαγωγή ........ 7"), [1.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(own_features("## ΠΕΡΙΕΧΟΜΕΝΑ"), [0.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(own_features("Αυτό είναι κανονικό περιεχόμενο μιας παραγράφου."), [0.0; 5]);
    }

    #[test]
    fn front_gate_zeros_late_lines() {
        let present = vec!["## ΠΕΡΙΕΧΟΜΕΝΑ", "1. Εισαγωγή .... 1"];
        let probabilities = toc_scores(&present, &[50, 51], 100);
        assert_eq!(probabilities, vec![0.0, 0.0]);
    }

    #[test]
    fn emits_toc_span_for_strong_front_block() {
        let text = "## ΠΕΡΙΕΧΟΜΕΝΑ\n1.1 Εισαγωγή ........ 1\n1.2 Μέθοδος ........ 4\n\n## ΚΕΦΑΛΑΙΟ\n".to_string()
            + &"Κανονικό κείμενο του κεφαλαίου.\n".repeat(30);
        let spans = toc_detect("doc", "test", &text);
        assert!(!spans.is_empty());
        assert!(spans.iter().all(|span| span.kind == "toc_span"));
        assert_eq!(spans[0].line_start, 0);
    }
}
