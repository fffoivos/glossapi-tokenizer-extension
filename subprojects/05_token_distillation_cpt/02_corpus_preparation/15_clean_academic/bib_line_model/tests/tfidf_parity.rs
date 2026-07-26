//! Does the ported TF-IDF actually agree with the fitted scikit-learn vectorizers?
//!
//! The analyzers in `predict.rs` are a reimplementation of sklearn's `char_wb` and
//! word n-gram rules from a reading of the source, which made them the least-verified
//! part of the port: the boundary rules are fiddly (each whitespace-split word is
//! padded with one space either side, and a word shorter than n contributes exactly
//! one padded n-gram rather than several), and a subtle disagreement would shift
//! every heading probability without raising an error anywhere.
//!
//! `fixtures/tfidf_cases.json` therefore holds the sparse rows the *fitted*
//! vectorizers actually produced for 4,000 real corpus lines, sampled at a fixed
//! stride across all 150 cohort-2 documents. This test reproduces them.

use bib_line_model::artifacts::HeadingFold;
use bib_line_model::predict::Tfidf;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    text: String,
    char_indices: Vec<usize>,
    char_values: Vec<f64>,
    word_indices: Vec<usize>,
    word_values: Vec<f64>,
}

#[derive(Deserialize)]
struct Cases {
    n_cases: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct HeadingBundleFile {
    folds: Vec<HeadingFold>,
}

fn load<T: for<'de> Deserialize<'de>>(rel: &str) -> T {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{path}: {e}"))
}

#[test]
fn tfidf_matches_fitted_sklearn() {
    let bundle: HeadingBundleFile = load("artifacts/heading_bundle.json");
    let fold = &bundle.folds[0];
    let char_tfidf = Tfidf::new(&fold.char_tfidf).expect("char vectorizer");
    let word_tfidf = Tfidf::new(&fold.word_tfidf).expect("word vectorizer");
    let cases: Cases = load("fixtures/tfidf_cases.json");

    // Values are compared with a relative tolerance rather than bit-exactly: the
    // vectorizers were fitted with dtype=float32 and their output is float32, while
    // the port accumulates in f64. Support — which indices are non-zero — is the part
    // that must match exactly, because that is what the analyzer determines.
    const TOL: f64 = 2e-6;
    let mut support_mismatch = 0usize;
    let mut value_mismatch = 0usize;
    let mut worst = 0f64;
    let mut first: Option<String> = None;

    for case in &cases.cases {
        for (label, tfidf, want_idx, want_val) in [
            ("char", &char_tfidf, &case.char_indices, &case.char_values),
            ("word", &word_tfidf, &case.word_indices, &case.word_values),
        ] {
            let got = tfidf.transform(&case.text);
            let got_idx: Vec<usize> = got.iter().map(|(i, _)| *i).collect();
            let mut want: Vec<(usize, f64)> = want_idx
                .iter()
                .copied()
                .zip(want_val.iter().copied())
                .collect();
            want.sort_by_key(|(i, _)| *i);
            let want_sorted: Vec<usize> = want.iter().map(|(i, _)| *i).collect();
            if got_idx != want_sorted {
                support_mismatch += 1;
                if first.is_none() {
                    let snippet: String = case.text.chars().take(60).collect();
                    first = Some(format!(
                        "{label}: support differs on {snippet:?} — {} vs {} terms",
                        got_idx.len(),
                        want_sorted.len()
                    ));
                }
                continue;
            }
            for ((_, g), (_, w)) in got.iter().zip(want.iter()) {
                let denom = w.abs().max(1e-12);
                let rel = (g - w).abs() / denom;
                worst = worst.max(rel);
                if rel > TOL {
                    value_mismatch += 1;
                    if first.is_none() {
                        let snippet: String = case.text.chars().take(60).collect();
                        first = Some(format!("{label}: {g} vs {w} (rel {rel:.2e}) on {snippet:?}"));
                    }
                }
            }
        }
    }

    eprintln!(
        "\ntfidf parity over {} cases: {} support mismatches, {} value mismatches, worst rel {:.2e}",
        cases.n_cases, support_mismatch, value_mismatch, worst
    );
    if let Some(detail) = &first {
        eprintln!("  first: {detail}");
    }
    assert_eq!(support_mismatch, 0, "analyzer disagrees with sklearn");
    assert_eq!(value_mismatch, 0, "tf-idf weights disagree with sklearn");
}
