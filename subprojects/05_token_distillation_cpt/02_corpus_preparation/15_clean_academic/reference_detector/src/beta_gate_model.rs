//! beta_gate_model — the deployed β-gate: a 7-feature logistic regression over interpretable,
//! bibliographically-justified features, fit on 1985 Opus-labelled sections (see ../eval/).
//!
//! Held-out (document-grouped, leak-free) test: precision ≈ 0.85 / recall ≈ 0.91 / F1 ≈ 0.88 —
//! the position-aware break over the section-internal model (+0.031 F1, bootstrap CI excludes 0).
//! Decision: `sigmoid(w · standardize(x) + b) ≥ 0.5` ⇒ bibliography (drop candidate). The model is a
//! plain dot-product so it stays deterministic + auditable. Weights/μ/σ are the exported
//! `eval/beta_gate_model.json` — KEEP IN SYNC if the model is refit.
//!
//! Features (in order): log_entry, author, year_paren, header_cv, front_cluster, is_last, dist_end.
//!   "a long run of name+year entries, at the document END, not surrounded by front-matter, no CV header."

use crate::reference_signals as sig;

const MU: [f64; 7] = [1.5017, 0.3751, 0.2598, 0.0388, 0.2345, 0.1985, 0.3707];
const SD: [f64; 7] = [1.1775, 0.4112, 0.4802, 0.1931, 0.2517, 0.3989, 0.3439];
const W: [f64; 7] = [0.9165, 1.0616, 0.6030, -0.3989, -0.5411, 0.5197, -0.6415];
const BIAS: f64 = 0.3176;
const THRESHOLD: f64 = 0.5;

fn is_entry_line(line: &str) -> bool {
    if !sig::NAME_TOKEN.is_match(line) {
        return false;
    }
    sig::YEAR.is_match(line)
        || sig::PAGE_ENTRY.is_match(line)
        || sig::AUTHOR_GR.is_match(line)
        || sig::PLACE_COLON.is_match(line)
        || sig::ENTRY_LOC.iter().any(|x| line.contains(x))
}

/// Compute the 7 features for one β section. `pos` is this section's positional fraction in the doc;
/// `beta_positions` are the positional fractions of all β-classed sections in the same document.
pub fn features(header: &str, section: &str, pos: f32, beta_positions: &[f32]) -> [f64; 7] {
    let lines: Vec<&str> = section.lines().filter(|l| !l.trim().is_empty()).collect();
    let nl = lines.len().max(1) as f64;
    let entry_lines = lines.iter().filter(|l| is_entry_line(l)).count() as f64;
    let author = lines.iter().filter(|l| sig::AUTHOR_GR.is_match(l)).count() as f64 / nl;
    let year_paren = sig::YEAR_PAREN.find_iter(section).count() as f64 / nl;
    let folded_h = sig::fold_header(header);
    let header_cv = if sig::CV_STEMS.iter().any(|s| folded_h.contains(s)) { 1.0 } else { 0.0 };
    let nb = beta_positions.len().max(1) as f64;
    let front_cluster = beta_positions.iter().filter(|&&x| x < 0.3).count() as f64 / nb;
    let maxpos = beta_positions.iter().cloned().fold(pos, f32::max);
    let is_last = if pos >= maxpos - 1e-6 { 1.0 } else { 0.0 };
    let dist_end = (1.0 - pos) as f64;
    [(1.0 + entry_lines).ln(), author, year_paren, header_cv, front_cluster, is_last, dist_end]
}

/// Returns (bibliography probability, is_bibliography).
pub fn score(f: &[f64; 7]) -> (f64, bool) {
    let mut s = BIAS;
    for i in 0..7 {
        s += W[i] * ((f[i] - MU[i]) / SD[i]);
    }
    let p = 1.0 / (1.0 + (-s).exp());
    (p, p >= THRESHOLD)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn end_bibliography_scores_bib() {
        let sec = "Παπαδόπουλος, Γεώργιος. Ιστορία. Αθήνα: Εκδόσεις Α, 1999, 12-45.\nSmith, John. A title. London: Press, 2008, 3-9.\nΒερέμης, Θάνος. Νεότερη Ελλάδα. Αθήνα: Εκδόσεις, 2001.";
        let f = features("ΒΙΒΛΙΟΓΡΑΦΙΑ", sec, 0.95, &[0.1, 0.5, 0.95]);
        assert!(score(&f).1, "end-of-doc entry list should score bibliography");
    }
    #[test]
    fn front_cv_scores_kept() {
        let sec = "2009 Ειδικευόμενος, Παρίσι\n2010 Fellow, Cardiff\n2011 Course, London";
        let f = features("Μεταπτυχιακή εκπαίδευση", sec, 0.05, &[0.05, 0.08, 0.12]);
        assert!(!score(&f).1, "front-matter CV cluster should score kept-non-bib");
    }
}
