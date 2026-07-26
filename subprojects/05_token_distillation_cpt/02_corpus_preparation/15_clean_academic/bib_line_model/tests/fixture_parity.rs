//! Feature-parity scoreboard against the Python extractor.
//!
//! `fixtures/lines_cohort2.json` is generated on Clariden by
//! `fixtures/generate_fixtures.py` from the deployed Python feature stack, so every
//! expected value in it is ground truth rather than a guess. It is the 40 hand-picked
//! shapes followed by 20,000 lines sampled at a fixed stride across all 150 cohort-2
//! documents. The hand-picked set alone was too weak a gate: it read 35/35 while two
//! real divergences were still live, both of which the corpus sample caught.
//!
//! The test reports per-feature agreement and, under `PARITY_STRICT=1`, asserts it.
//!
//! Shape values are compared bit-exactly. Python builds the vector in float64 and
//! casts with `np.asarray(values, dtype=np.float32)`; the port computes in f64 and
//! casts the same way, so equality is the right bar — a tolerance would hide exactly
//! the arithmetic-order mistakes worth catching.

use bib_line_model::features::{line_counts, FEATURE_NAMES};
use bib_line_model::shape::{line_shape, LINE_SHAPE_NAMES};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Case {
    text: String,
    counts: Vec<i64>,
    shape: Vec<f32>,
}

#[derive(Deserialize)]
struct Fixture {
    feature_names: Vec<String>,
    shape_names: Vec<String>,
    n_cases: usize,
    cases: Vec<Case>,
}

fn snippet(text: &str) -> String {
    let mut s: String = text.chars().take(56).collect();
    if text.chars().count() > 56 {
        s.push('…');
    }
    s
}

fn report(
    kind: &str,
    names: &[&str],
    exact: &BTreeMap<&str, usize>,
    diffs: &BTreeMap<&str, String>,
    n: usize,
) -> usize {
    let done = names
        .iter()
        .filter(|f| exact.get(**f).copied().unwrap_or(0) == n)
        .count();
    eprintln!("\n{kind} parity: {done}/{} exact on all {n} cases", names.len());
    for name in names {
        let hit = exact.get(*name).copied().unwrap_or(0);
        if hit != n {
            eprintln!(
                "  {name:<32} {hit:>6}/{n}   {}",
                diffs.get(*name).map(String::as_str).unwrap_or("")
            );
        }
    }
    done
}

#[test]
fn features_match_python() {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/lines_cohort2.json"
    ))
    .expect(
        "fixtures/lines_cohort2.json — regenerate on Clariden with fixtures/generate_fixtures.py",
    );
    let fixture: Fixture = serde_json::from_str(&raw).expect("fixture parses");

    assert_eq!(
        fixture.feature_names.len(),
        FEATURE_NAMES.len(),
        "count schema diverged"
    );
    for (i, name) in fixture.feature_names.iter().enumerate() {
        assert_eq!(name, FEATURE_NAMES[i], "count feature {i} order differs");
    }
    assert_eq!(
        fixture.shape_names.len(),
        LINE_SHAPE_NAMES.len(),
        "shape schema diverged"
    );
    for (i, name) in fixture.shape_names.iter().enumerate() {
        assert_eq!(name, LINE_SHAPE_NAMES[i], "shape value {i} order differs");
    }

    let mut c_exact: BTreeMap<&str, usize> = BTreeMap::new();
    let mut c_diff: BTreeMap<&str, String> = BTreeMap::new();
    let mut s_exact: BTreeMap<&str, usize> = BTreeMap::new();
    let mut s_diff: BTreeMap<&str, String> = BTreeMap::new();

    for case in &fixture.cases {
        let got = line_counts(&case.text);
        for (i, name) in FEATURE_NAMES.iter().enumerate() {
            if got[i] as i64 == case.counts[i] {
                *c_exact.entry(*name).or_insert(0) += 1;
            } else {
                c_diff.entry(*name).or_insert_with(|| {
                    format!(
                        "want {}, got {} on {:?}",
                        case.counts[i],
                        got[i],
                        snippet(&case.text)
                    )
                });
            }
        }
        let shape = line_shape(&case.text);
        for (i, name) in LINE_SHAPE_NAMES.iter().enumerate() {
            if shape[i].to_bits() == case.shape[i].to_bits() {
                *s_exact.entry(*name).or_insert(0) += 1;
            } else {
                s_diff.entry(*name).or_insert_with(|| {
                    format!(
                        "want {}, got {} on {:?}",
                        case.shape[i],
                        shape[i],
                        snippet(&case.text)
                    )
                });
            }
        }
    }

    let n = fixture.n_cases;
    let counts_done = report("count", &FEATURE_NAMES, &c_exact, &c_diff, n);
    let shape_done = report("shape", &LINE_SHAPE_NAMES, &s_exact, &s_diff, n);

    if std::env::var("PARITY_STRICT").is_ok() {
        assert_eq!(counts_done, FEATURE_NAMES.len(), "count features diverge");
        assert_eq!(shape_done, LINE_SHAPE_NAMES.len(), "shape values diverge");
    }
}
