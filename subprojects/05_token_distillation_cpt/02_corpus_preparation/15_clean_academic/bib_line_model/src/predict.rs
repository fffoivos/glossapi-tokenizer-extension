//! Model evaluation — the numerics that must agree with scikit-learn.
//!
//! Three families, each matched to its sklearn source:
//!
//! * `HistGradientBoostingClassifier` — prediction walks the raw feature against
//!   `num_threshold`; binning is training-only. Missing values follow
//!   `missing_go_to_left`. Raw score = baseline + sum of leaf values, then sigmoid.
//! * `LogisticRegression` — dot product + intercept, then sigmoid (binary) or
//!   softmax (multiclass).
//! * `TfidfVectorizer` — analyzer -> term counts -> sublinear tf -> idf -> L2.
//!
//! The comments cite the sklearn behaviour each block reproduces, because a silent
//! divergence here is the failure mode that would be hardest to find later.

use crate::artifacts::{HistGbModel, LinearModel, Model, TfidfModel, Tree};
use fancy_regex::Regex as FancyRegex;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

/// Python's word class, `str.isalnum()` plus underscore, which is exactly category
/// L* or N* plus `_`. See `fixtures/dump_patterns.py` for the same rewrite applied
/// to the feature regexes and for the assertion that the identity holds.
const PY_WORD: &str = r"\p{L}\p{N}_";

/// Rewrite a fitted `token_pattern` from Python's regex dialect into one that means
/// the same thing to the Rust engine.
///
/// The heading bundle's pattern is `(?u)\b[^\W_]+(?:[’'\-][^\W_]+)*\b`, and every
/// piece of it is a place the two engines disagree:
///
/// * `\W` — Rust's word class is Alphabetic|M|Nd|Pc|Join_Control and so accepts
///   combining marks, Python's does not. `[^\W_]` reduces exactly to `[\p{L}\p{N}]`.
/// * `\b` — defined in terms of that same word class, so it inherits the difference.
///   Expressed here as the explicit lookaround pair, which is why the tokenizer needs
///   fancy-regex rather than the `regex` crate.
/// * `(?u)` — a no-op for Rust, which is Unicode-aware by default.
///
/// Left unrewritten this cost exactly one token on one line in 4,000 — small, but the
/// kind of difference that is invisible until it moves a probability across 0.9.
fn python_token_pattern(pattern: &str) -> String {
    let boundary = format!(
        "(?:(?<=[{PY_WORD}])(?![{PY_WORD}])|(?<![{PY_WORD}])(?=[{PY_WORD}]))"
    );
    pattern
        .replace("(?u)", "")
        .replace(r"[^\W_]", r"[\p{L}\p{N}]")
        .replace(r"\b", &boundary)
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ---------------------------------------------------------------------------
// tree ensembles
// ---------------------------------------------------------------------------

/// Walk one tree. Mirrors sklearn's `_predict_one_from_raw_data`:
/// go left when `value <= num_threshold`, and on NaN follow `missing_go_to_left`.
#[inline]
fn tree_value(tree: &Tree, features: &[f64]) -> f64 {
    let mut node = 0usize;
    loop {
        if tree.is_leaf[node] != 0 {
            return tree.value[node];
        }
        let idx = tree.feature_idx[node] as usize;
        let x = features.get(idx).copied().unwrap_or(f64::NAN);
        let go_left = if x.is_nan() {
            tree.missing_go_to_left[node] != 0
        } else {
            x <= tree.num_threshold[node]
        };
        node = if go_left {
            tree.left[node] as usize
        } else {
            tree.right[node] as usize
        };
    }
}

/// Binary HistGB: P(class 1) = sigmoid(baseline + sum(leaves)).
pub fn hist_gb_proba(model: &HistGbModel, features: &[f64]) -> f64 {
    let mut raw = model.baseline_prediction.first().copied().unwrap_or(0.0);
    for tree in &model.trees {
        raw += tree_value(tree, features);
    }
    sigmoid(raw)
}

// ---------------------------------------------------------------------------
// linear models
// ---------------------------------------------------------------------------

/// Binary logistic: P(class 1). For a multiclass model this returns the softmax
/// over all outputs instead — see `linear_scores`.
pub fn linear_proba(model: &LinearModel, features: &[f64]) -> f64 {
    let scores = linear_scores(model, features);
    if scores.len() == 1 {
        scores[0]
    } else {
        // by convention the caller wants P(positive) = last class for binary-as-multiclass
        *scores.last().unwrap_or(&0.0)
    }
}

/// Decision function then link. Returns one probability per output.
pub fn linear_scores(model: &LinearModel, features: &[f64]) -> Vec<f64> {
    let mut raw: Vec<f64> = model
        .coef
        .iter()
        .enumerate()
        .map(|(k, w)| {
            let b = model.intercept.get(k).copied().unwrap_or(0.0);
            let mut acc = b;
            for (i, wi) in w.iter().enumerate() {
                if *wi != 0.0 {
                    acc += wi * features.get(i).copied().unwrap_or(0.0);
                }
            }
            acc
        })
        .collect();
    if model.link == "softmax" {
        let max = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for v in raw.iter_mut() {
            *v = (*v - max).exp();
            total += *v;
        }
        if total > 0.0 {
            for v in raw.iter_mut() {
                *v /= total;
            }
        }
        raw
    } else {
        raw.iter().map(|x| sigmoid(*x)).collect()
    }
}

/// Sparse variant: the TF-IDF blocks are overwhelmingly zero, so only touch
/// the non-zero entries. `offset` shifts into the concatenated feature space.
pub fn linear_accumulate_sparse(
    model: &LinearModel,
    nonzero: &[(usize, f64)],
    offset: usize,
    out: &mut [f64],
) {
    for (k, w) in model.coef.iter().enumerate() {
        let mut acc = out[k];
        for (idx, value) in nonzero {
            if let Some(wi) = w.get(offset + idx) {
                acc += wi * value;
            }
        }
        out[k] = acc;
    }
}

/// Score a linear model against a sparse feature vector.
///
/// The heading feature space is ~30k columns of which a line touches a few hundred,
/// so this walks the non-zeros rather than the coefficient rows. Returns one
/// probability per output: sigmoid for a binary model, softmax across outputs for a
/// multiclass one.
pub fn linear_sparse_scores(model: &Model, nonzero: &[(usize, f64)]) -> Vec<f64> {
    let m = match model {
        Model::Linear(m) => m,
        Model::HistGb(_) => panic!("linear_sparse_scores called on a tree model"),
    };
    let mut raw: Vec<f64> = m.intercept.clone();
    for (k, w) in m.coef.iter().enumerate() {
        let mut acc = raw[k];
        for (idx, value) in nonzero {
            if let Some(wi) = w.get(*idx) {
                acc += wi * value;
            }
        }
        raw[k] = acc;
    }
    if m.link == "softmax" {
        let max = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for v in raw.iter_mut() {
            *v = (*v - max).exp();
            total += *v;
        }
        if total > 0.0 {
            for v in raw.iter_mut() {
                *v /= total;
            }
        }
        raw
    } else {
        raw.iter().map(|x| sigmoid(*x)).collect()
    }
}

pub fn model_proba(model: &Model, features: &[f64]) -> f64 {
    match model {
        Model::HistGb(m) => hist_gb_proba(m, features),
        Model::Linear(m) => linear_proba(m, features),
    }
}

// ---------------------------------------------------------------------------
// TF-IDF
// ---------------------------------------------------------------------------

/// sklearn `CountVectorizer._white_spaces = re.compile(r"\s\s+")` — runs of two
/// or more whitespace characters collapse to one space before analysis.
static WHITE_SPACES: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s\s+").unwrap());

/// `char_wb` n-grams, reproducing sklearn's `_char_wb_ngrams` statement for
/// statement:
///
/// ```python
/// for w in text_document.split():
///     w = " " + w + " "
///     w_len = len(w)
///     for n in range(min_n, max_n + 1):
///         offset = 0
///         ngrams.append(w[offset : offset + n])
///         while offset + n < w_len:
///             offset += 1
///             ngrams.append(w[offset : offset + n])
///         if offset == 0:   # count a short word (w_len < n) only once
///             break
/// ```
///
/// The `if offset == 0: break` is the subtle part, and getting it wrong is not
/// harmless. It fires when the while loop never ran — that is, when `n >= w_len`,
/// not when `n > w_len`. Treating it as the strict inequality leaves the n-loop
/// running for larger `n`, and since `w[0:n]` saturates at the whole padded word,
/// every extra iteration re-emits a string that is already in the output. The
/// *support* is unchanged, so nothing looks wrong; only the term counts inflate,
/// and with sublinear tf and L2 normalisation that quietly shifts every weight in
/// the row. `tests/tfidf_parity.rs` caught it against the fitted vectorizer.
fn char_wb_ngrams(text: &str, min_n: usize, max_n: usize) -> Vec<String> {
    let collapsed = WHITE_SPACES.replace_all(text, " ");
    let mut out = Vec::new();
    for word in collapsed.split_whitespace() {
        let padded: Vec<char> = std::iter::once(' ')
            .chain(word.chars())
            .chain(std::iter::once(' '))
            .collect();
        let w_len = padded.len();
        for n in min_n..=max_n {
            let mut offset = 0usize;
            let take = n.min(w_len); // Python slicing clamps; Rust would panic.
            out.push(padded[offset..take].iter().collect::<String>());
            while offset + n < w_len {
                offset += 1;
                out.push(
                    padded[offset..(offset + n).min(w_len)]
                        .iter()
                        .collect::<String>(),
                );
            }
            if offset == 0 {
                break;
            }
        }
    }
    out
}

/// Word n-grams: tokenize with the fitted `token_pattern`, then join adjacent
/// tokens with a single space for n > 1 (sklearn `_word_ngrams`).
fn word_ngrams(text: &str, tokenizer: &FancyRegex, min_n: usize, max_n: usize) -> Vec<String> {
    let tokens: Vec<&str> = tokenizer.find_iter(text).flatten().map(|m| m.as_str()).collect();
    let mut out: Vec<String> = Vec::new();
    if min_n <= 1 && !tokens.is_empty() {
        out.extend(tokens.iter().map(|t| t.to_string()));
    }
    let start = min_n.max(2);
    for n in start..=max_n {
        if tokens.len() < n {
            break;
        }
        for window in tokens.windows(n) {
            out.push(window.join(" "));
        }
    }
    out
}

pub struct Tfidf {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f64>,
    min_n: usize,
    max_n: usize,
    lowercase: bool,
    sublinear_tf: bool,
    l2: bool,
    char_wb: bool,
    tokenizer: Option<FancyRegex>,
}

impl Tfidf {
    pub fn new(model: &TfidfModel) -> anyhow::Result<Self> {
        let char_wb = model.analyzer == "char_wb";
        let tokenizer = if char_wb {
            None
        } else {
            // sklearn's default is r"(?u)\b\w\w+\b"; the heading bundle overrides it.
            let raw = model
                .token_pattern
                .clone()
                .unwrap_or_else(|| r"\b\w\w+\b".to_string());
            Some(FancyRegex::new(&python_token_pattern(&raw))?)
        };
        Ok(Self {
            vocabulary: model.vocabulary.clone(),
            idf: model.idf.clone(),
            min_n: *model.ngram_range.first().unwrap_or(&1),
            max_n: *model.ngram_range.get(1).unwrap_or(&1),
            lowercase: model.lowercase,
            sublinear_tf: model.sublinear_tf,
            l2: model.norm.as_deref() == Some("l2"),
            char_wb,
            tokenizer,
        })
    }

    /// Transform one document into `(vocabulary_index, weight)` pairs.
    pub fn transform(&self, text: &str) -> Vec<(usize, f64)> {
        let lowered;
        let source = if self.lowercase {
            lowered = text.to_lowercase();
            lowered.as_str()
        } else {
            text
        };
        let grams = if self.char_wb {
            char_wb_ngrams(source, self.min_n, self.max_n)
        } else {
            match &self.tokenizer {
                Some(rx) => word_ngrams(source, rx, self.min_n, self.max_n),
                None => Vec::new(),
            }
        };
        let mut counts: HashMap<usize, f64> = HashMap::new();
        for gram in grams {
            if let Some(&index) = self.vocabulary.get(&gram) {
                *counts.entry(index).or_insert(0.0) += 1.0;
            }
        }
        let mut out: Vec<(usize, f64)> = counts
            .into_iter()
            .map(|(index, tf)| {
                // sublinear_tf: tf -> 1 + ln(tf)   (sklearn TfidfTransformer)
                let tf = if self.sublinear_tf { 1.0 + tf.ln() } else { tf };
                (index, tf * self.idf.get(index).copied().unwrap_or(0.0))
            })
            .collect();
        if self.l2 {
            let norm: f64 = out.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for (_, v) in out.iter_mut() {
                    *v /= norm;
                }
            }
        }
        out.sort_unstable_by_key(|(i, _)| *i);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_wb_matches_sklearn_padding() {
        // sklearn: " ab ", n=3 over the padded word " ab " -> [" ab", "ab "]
        let g = char_wb_ngrams("ab", 3, 3);
        assert_eq!(g, vec![" ab".to_string(), "ab ".to_string()]);
    }

    #[test]
    fn char_wb_short_word_emitted_once() {
        // padded "ab" is 4 chars; for n=5 sklearn emits the clamped slice once
        let g = char_wb_ngrams("ab", 5, 5);
        assert_eq!(g, vec![" ab ".to_string()]);
    }

    #[test]
    fn char_wb_stops_when_n_reaches_the_padded_length() {
        // " ab " is 4 chars. sklearn's `if offset == 0: break` fires at n == 4, so
        // n == 5 never runs and " ab " appears exactly once, not twice. Emitting it
        // twice leaves the support identical and only inflates the term count, which
        // is why this needed a test against the fitted vectorizer rather than a
        // reading of the source.
        let g = char_wb_ngrams("ab", 2, 5);
        assert_eq!(
            g,
            vec![" a", "ab", "b ", " ab", "ab ", " ab "]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn char_wb_collapses_runs_of_whitespace() {
        let g = char_wb_ngrams("a\n\n b", 2, 2);
        // two words -> " a", "a ", " b", "b "
        assert_eq!(g.len(), 4);
    }

    #[test]
    fn word_ngrams_unigram_and_bigram() {
        let rx = FancyRegex::new(&python_token_pattern(
            r"(?u)\b[^\W_]+(?:[’'\-][^\W_]+)*\b",
        ))
        .unwrap();
        let g = word_ngrams("alpha beta gamma", &rx, 1, 2);
        assert_eq!(
            g,
            vec!["alpha", "beta", "gamma", "alpha beta", "beta gamma"]
        );
    }

    #[test]
    fn sigmoid_is_symmetric_and_stable() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(-800.0) >= 0.0 && sigmoid(800.0) <= 1.0);
        assert!((sigmoid(2.0) + sigmoid(-2.0) - 1.0).abs() < 1e-12);
    }
}
