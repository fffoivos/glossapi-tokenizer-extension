//! span_line_model — the deployed line-level bibliography-SPAN classifier (precision-first).
//!
//! Port of the interpretable Python pipeline (eval/span_signals.py + eval/line_lr.py +
//! eval/decode_spans.py): per-PRESENT-line (non-blank) features → standardized logistic-regression
//! dot-product → per-line probability → hysteresis run-length decoder → multi-span (start_line,end_line).
//!
//! Constants below are the exported eval/span_line_lr_model.json + eval/span_smooth_params.json — KEEP
//! IN SYNC if the model is refit. Regexes mirror span_signals.py EXACTLY (NOT the β-gate signals —
//! latin_fraction uses Greek U+0386..=03CE only, NAME includes polytonic U+1F00..=1FFF), verified to
//! per-line parity by tests/parity (see eval/rust_parity.py).
use once_cell::sync::Lazy;
use regex::Regex;

// ---- model (synced from eval/span_line_lr_model.json) ----
pub const FEATS: usize = 22;
const MU: [f64; FEATS] = [
    0.35766, 0.15613, 0.01851, 0.02984, 0.10199, 0.17111, 0.04709, 0.08027, 0.40398, 0.00202,
    0.17659, 0.02554, 0.10184, 0.35768, 0.35771, 0.35783, 0.3137, 0.17117, 0.95755, 0.11916,
    0.02128, 0.68316,
];
const SD: [f64; FEATS] = [
    0.47931, 0.36297, 0.13478, 0.17016, 0.30263, 0.3766, 0.21184, 0.27171, 0.46496, 0.04493,
    0.38132, 0.15775, 0.30244, 0.39577, 0.37694, 0.3688, 0.3766, 0.27155, 0.16742, 0.32398,
    0.11391, 0.27349,
];
const W: [f64; FEATS] = [
    0.06301, 0.27481, 0.15979, 0.12066, 0.18983, 0.03823, 0.09949, -0.02856, 0.54876, -0.13274,
    0.14505, 0.26076, -0.12174, 0.36788, 0.41554, 0.5211, 0.73694, 0.27771, -0.31073, 0.28922,
    0.12992, 0.54016,
];
const BIAS: f64 = -2.86499;
// hysteresis (eval/span_smooth_params.json)
const THETA_HI: f64 = 0.8;
const THETA_LO: f64 = 0.6;
const GAP: usize = 2;
const LMIN: usize = 2;

// ---- regexes (EXACTLY span_signals.py) ----
static NAME: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[Α-ΩA-ZΆ-Ώ\u{1F00}-\u{1FFF}][α-ωa-zά-ώϊϋ\u{1F00}-\u{1FFF}]{2,}").unwrap());
static AUTHOR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]+\s*,\s*[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ.]+").unwrap());
static GREEK_AUTHOR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[Α-ΩΆ-Ώ\u{1F00}-\u{1FFF}][α-ωά-ώϊϋ\u{1F00}-\u{1FFF}]+\s*,?\s*[Α-ΩΆ-Ώ\u{1F00}-\u{1FFF}]\.").unwrap());
static GR_EDMARK: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(μτφρ|επιμ|εκδ|εκδόσεις|εκδ́σεις|τόμ|τχ|σσ|σελ|αρ\. φ|ό\.π)\b\.?|\b(Αθήνα|Θεσσαλονίκη|Αθήναι|Λευκωσία)\s*[:·,]").unwrap()
});
static YEAR: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(1[5-9]\d{2}|20\d{2})\b").unwrap());
static YEAR_PAREN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\((1[5-9]\d{2}|20\d{2})\)").unwrap());
static PAGE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\d+\s*[-–]\s*\d+\b|σσ?\.\s*\d|pp?\.\s*\d").unwrap());
static PLACE_PUB: Lazy<Regex> = Lazy::new(|| Regex::new(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώ]+\s*:\s*[Α-ΩA-ZΆ-Ώ]").unwrap());
static URL: Lazy<Regex> = Lazy::new(|| Regex::new(r"https?://|doi\.|www\.|\.gr/|\.com/").unwrap());
static ATX: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s{0,3}#{1,6}\s").unwrap());
static NUM_PREFIX: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*(\[?\d{1,3}\]?[.)\]]|\d{1,3}\s)").unwrap());
static LEADNUM: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*\[?(\d{1,4})\]?[.)]").unwrap());

const LOC: &[&str] = &["σσ.", "σ.", "τ.", "τόμ.", "εκδ.", "εκδόσεις", "επιμ.", "μτφρ.", "vol.", "pp.", "eds.", "ed.", "et al"];
const H_BIB: &[&str] = &["βιβλιογραφ", "αναφορ", "πηγ", "references", "bibliograph", "δικτυογραφ", "ελληνογλωσσ", "ξενογλωσσ", "ιστοσελιδ", "works cited", "further reading"];
const H_CV: &[&str] = &["δημοσιευσ", "ανακοινωσ", "βιογραφικο", "σπουδες", "εκπαιδευσ", "εμπειρια", "σεμιναρ", "συνεδρι", "καταλογος ανακοιν", "curriculum"];
const H_APP: &[&str] = &["δραστηριοτ", "λεξεις-κλειδ", "λεξεις κλειδ", "απαντησ", "προαπαιτ", "ασκησ", "περιληψ", "συνοψ", "κριτηρι", "παραρτημα", "appendix"];

fn py_latin_fraction(s: &str) -> f64 {
    let (mut lat, mut grk) = (0u32, 0u32);
    for c in s.chars() {
        if ('A'..='Z').contains(&c) || ('a'..='z').contains(&c) {
            lat += 1;
        } else if ('\u{0386}'..='\u{03CE}').contains(&c) {
            grk += 1; // matches Python _GRK = [Α-Ωα-ωΆ-ώϊϋΐΰ] (NO polytonic)
        }
    }
    let d = lat + grk;
    if d == 0 { 0.0 } else { lat as f64 / d as f64 }
}

fn header_family(low: &str) -> &'static str {
    if H_CV.iter().any(|x| low.contains(x)) { "cv" }
    else if H_APP.iter().any(|x| low.contains(x)) { "app" }
    else if H_BIB.iter().any(|x| low.contains(x)) { "bib" }
    else { "" }
}

fn is_entry(l: &str, low: &str) -> bool {
    if !NAME.is_match(l) { return false; }
    YEAR.is_match(l) || PAGE.is_match(l) || AUTHOR.is_match(l) || PLACE_PUB.is_match(l)
        || GREEK_AUTHOR.is_match(l) || GR_EDMARK.is_match(l) || LOC.iter().any(|x| low.contains(x))
}

fn mean(v: &[f64], i: usize, k: usize) -> f64 {
    let a = i.saturating_sub(k);
    let b = (i + k + 1).min(v.len());
    if b > a { v[a..b].iter().sum::<f64>() / (b - a) as f64 } else { 0.0 }
}

/// Per-present-line probability. `present`: non-blank line texts in order; `abs_idx`: their absolute
/// line indices in the full doc; `n_total`: total doc lines (for `pos`). Mirrors line_lr.doc_features.
pub fn span_scores(present: &[&str], abs_idx: &[usize], n_total: usize) -> Vec<f64> {
    let n = present.len();
    let low: Vec<String> = present.iter().map(|s| s.to_lowercase()).collect();
    let ent: Vec<f64> = (0..n).map(|i| if is_entry(present[i], &low[i]) { 1.0 } else { 0.0 }).collect();
    let yr: Vec<f64> = present.iter().map(|s| if YEAR.is_match(s) { 1.0 } else { 0.0 }).collect();
    let pg: Vec<f64> = present.iter().map(|s| if PAGE.is_match(s) { 1.0 } else { 0.0 }).collect();
    let bibhdr: Vec<bool> = (0..n).map(|i| ATX.is_match(present[i]) && H_BIB.iter().any(|x| low[i].contains(x))).collect();
    // dist to nearest bib header above (present-line units); under-bib-block (most-recent ATX family)
    let mut dist = vec![1.0f64; n];
    let mut under = vec![0.0f64; n];
    let mut last: Option<usize> = None;
    let mut sect = "";
    for i in 0..n {
        if bibhdr[i] { last = Some(i); }
        dist[i] = match last { Some(l) => ((i - l).min(50) as f64) / 50.0, None => 1.0 };
        if ATX.is_match(present[i]) {
            let f = header_family(&low[i]);
            sect = if f.is_empty() { "other" } else { f };
        }
        under[i] = if sect == "bib" { 1.0 } else { 0.0 };
    }
    let nt = n_total.max(1) as f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let l = present[i];
        let lo = &low[i];
        let trimmed_len = l.trim().chars().count();
        let lead = LEADNUM.captures(l).and_then(|c| c.get(1)).and_then(|m| m.as_str().parse::<f64>().ok()).unwrap_or(0.0);
        let f = [
            ent[i],
            if AUTHOR.is_match(l) { 1.0 } else { 0.0 },
            if GREEK_AUTHOR.is_match(l) { 1.0 } else { 0.0 },
            if GR_EDMARK.is_match(l) { 1.0 } else { 0.0 },
            if YEAR_PAREN.is_match(l) { 1.0 } else { 0.0 },
            pg[i],
            if PLACE_PUB.is_match(l) { 1.0 } else { 0.0 },
            if LOC.iter().any(|x| lo.contains(x)) { 1.0 } else { 0.0 },
            py_latin_fraction(l),
            if bibhdr[i] { 1.0 } else { 0.0 },
            if trimmed_len < 40 { 1.0 } else { 0.0 },
            if URL.is_match(l) { 1.0 } else { 0.0 },
            if NUM_PREFIX.is_match(l) { 1.0 } else { 0.0 },
            mean(&ent, i, 3),
            mean(&ent, i, 8),
            mean(&ent, i, 12),
            mean(&yr, i, 8),
            mean(&pg, i, 8),
            dist[i],
            under[i],
            lead.min(300.0) / 300.0,
            abs_idx[i] as f64 / nt,
        ];
        let mut s = BIAS;
        for j in 0..FEATS {
            s += W[j] * ((f[j] - MU[j]) / SD[j]);
        }
        out.push(1.0 / (1.0 + (-s).exp()));
    }
    out
}

/// Hysteresis run-length over per-present-line probabilities → (start_idx,end_idx) present-line runs.
pub fn hysteresis(p: &[f64]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut inrun = false;
    let mut start = 0usize;
    for i in 0..p.len() {
        if !inrun {
            if p[i] >= THETA_HI { inrun = true; start = i; }
        } else if p[i] < THETA_LO {
            runs.push((start, i - 1)); inrun = false;
        }
    }
    if inrun { runs.push((start, p.len() - 1)); }
    // merge runs within GAP present lines
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in runs {
        if let Some(last) = merged.last_mut() {
            if s <= last.1 + GAP + 1 { last.1 = last.1.max(e); continue; }
        }
        merged.push((s, e));
    }
    merged.into_iter().filter(|(s, e)| e - s + 1 >= LMIN).collect()
}

use crate::reference_module::Span;

/// Whole-document bibliography-SPAN detection: split into lines, score present (non-blank) lines, decode
/// contiguous spans by hysteresis, map back to absolute line ranges. Emits `bib_span` records with the
/// opening probability in `trigger`/`gated_by` (auditable). `pos` uses the absolute line index over the
/// full document — the faithful reproduction of the trained model (eval used windows with TRUE indices).
pub fn span_detect(doc_id: &str, _source: &str, text: &str) -> Vec<Span> {
    let raw: Vec<&str> = text.split('\n').collect();
    let n_total = raw.len();
    let mut present: Vec<&str> = Vec::new();
    let mut abs_idx: Vec<usize> = Vec::new();
    let mut char_start: Vec<usize> = Vec::new();
    let mut cs = 0usize;
    for (i, l) in raw.iter().enumerate() {
        if !l.trim().is_empty() {
            present.push(l);
            abs_idx.push(i);
            char_start.push(cs);
        }
        cs += l.chars().count() + 1; // + newline
    }
    if present.is_empty() {
        return Vec::new();
    }
    let p = span_scores(&present, &abs_idx, n_total);
    hysteresis(&p)
        .into_iter()
        .map(|(s, e)| Span {
            doc_id: doc_id.to_string(),
            kind: "bib_span".to_string(),
            char_start: char_start[s],
            char_end: char_start[e] + present[e].chars().count(),
            line_start: abs_idx[s],
            line_end: abs_idx[e],
            trigger: present[s].chars().take(40).collect(),
            gated_by: format!("span_lr p={:.3}", p[s]),
            row_id: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn latin_fraction_excludes_polytonic_like_python() {
        // 'ἱστορίαι' is polytonic (U+1F00+) -> NOT counted Greek by the Python def -> latin_frac on a
        // pure-polytonic word is 0 (no Latin, no monotonic-Greek letters counted).
        assert_eq!(py_latin_fraction("ἱστορίαι"), 0.0);
        assert!((py_latin_fraction("Smith") - 1.0).abs() < 1e-9);
        assert!((py_latin_fraction("Παπαδ") - 0.0).abs() < 1e-9);
    }
    #[test]
    fn is_entry_matches_greek_and_latin() {
        assert!(is_entry("Παπαδόπουλος, Α. (2011). Τίτλος. Αθήνα: Εκδόσεις.", &"παπαδόπουλος, α. (2011). τίτλος. αθήνα: εκδόσεις.".to_string()));
        assert!(!is_entry("This is a plain sentence of running prose.", &"this is a plain sentence of running prose.".to_string()));
    }
    #[test]
    fn hysteresis_merges_and_trims() {
        // runs 0-1 and 4-5 are 2 lines apart (<= GAP) -> merge to 0-5; the lone 0.85 at idx 10 is 4
        // lines past (> GAP) and length 1 (< LMIN) -> dropped.
        let p = vec![0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.85];
        assert_eq!(hysteresis(&p), vec![(0, 5)]);
    }
}
