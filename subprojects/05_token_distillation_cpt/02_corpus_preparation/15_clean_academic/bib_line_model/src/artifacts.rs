//! Deserialization of the exported model stack.
//!
//! Mirrors `eval/sequence_models/export_line_model_artifacts.py` field-for-field.
//! Nothing here evaluates anything — see `trees`, `linear`, `tfidf`.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// One boosting tree, flattened from sklearn's structured `nodes` array.
#[derive(Debug, Deserialize)]
pub struct Tree {
    pub is_leaf: Vec<u8>,
    pub feature_idx: Vec<i32>,
    pub num_threshold: Vec<f64>,
    pub left: Vec<i32>,
    pub right: Vec<i32>,
    pub value: Vec<f64>,
    pub missing_go_to_left: Vec<u8>,
}

#[derive(Debug, Deserialize)]
pub struct HistGbModel {
    pub n_trees: usize,
    pub n_features: usize,
    pub baseline_prediction: Vec<f64>,
    pub trees: Vec<Tree>,
}

#[derive(Debug, Deserialize)]
pub struct LinearModel {
    /// `[n_outputs][n_features]`
    pub coef: Vec<Vec<f64>>,
    pub intercept: Vec<f64>,
    pub n_features: usize,
    /// "logistic" for binary, "softmax" for multiclass
    pub link: String,
}

/// Tagged by the `kind` field the exporter writes.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
pub enum Model {
    #[serde(rename = "hist_gradient_boosting")]
    HistGb(HistGbModel),
    #[serde(rename = "linear")]
    Linear(LinearModel),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
pub enum Scaler {
    #[serde(rename = "identity")]
    Identity,
    #[serde(rename = "standard_scaler")]
    Standard {
        with_mean: bool,
        mean: Option<Vec<f64>>,
        scale: Vec<f64>,
    },
}

impl Scaler {
    pub fn apply(&self, row: &mut [f64]) {
        if let Scaler::Standard { with_mean, mean, scale } = self {
            for (i, x) in row.iter_mut().enumerate() {
                if *with_mean {
                    if let Some(m) = mean.as_ref().and_then(|m| m.get(i)) {
                        *x -= m;
                    }
                }
                if let Some(s) = scale.get(i) {
                    if *s != 0.0 {
                        *x /= s;
                    }
                }
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct TfidfModel {
    pub analyzer: String,
    pub ngram_range: Vec<usize>,
    pub lowercase: bool,
    pub sublinear_tf: bool,
    pub norm: Option<String>,
    pub n_features: usize,
    pub vocabulary: HashMap<String, usize>,
    pub idf: Vec<f64>,
    pub token_pattern: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct HeadingFold {
    pub char_tfidf: TfidfModel,
    pub word_tfidf: TfidfModel,
    pub numeric_scaler: Scaler,
    pub any_model: Model,
    pub type_model: Model,
}

#[derive(Debug, Deserialize)]
pub struct ConnectorFold {
    pub arm: String,
    pub mean: Option<Vec<f64>>,
    pub scale: Option<Vec<f64>>,
    pub connector_model: Model,
    pub subtype_model: Model,
    pub other_model: Model,
}

#[derive(Debug, Deserialize)]
pub struct LineFold {
    pub scaler: Scaler,
    pub model: Model,
}

#[derive(Debug, Deserialize)]
pub struct Folds<T> {
    pub folds: Vec<T>,
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub schema_version: String,
    pub line_threshold: f64,
    pub feature_schema: String,
}

/// Everything the detector needs, loaded once and shared across worker threads.
pub struct Artifacts {
    pub manifest: Manifest,
    pub entry: Vec<Model>,
    pub line: Vec<LineFold>,
    pub heading: Vec<HeadingFold>,
    pub connector: Vec<ConnectorFold>,
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

impl Artifacts {
    pub fn load(root: &Path) -> Result<Self> {
        let manifest: Manifest = read_json(&root.join("manifest.json"))?;
        anyhow::ensure!(
            manifest.schema_version == "bibliography-line-model-export-v1",
            "unsupported artifact schema: {}",
            manifest.schema_version
        );
        let entry: Folds<Model> = read_json(&root.join("entry_p0d.json"))?;
        let line: Folds<LineFold> = read_json(&root.join("line_model.json"))?;
        let heading: Folds<HeadingFold> = read_json(&root.join("heading_bundle.json"))?;
        let connector: Folds<ConnectorFold> = read_json(&root.join("connector_bundle.json"))?;
        anyhow::ensure!(!entry.folds.is_empty(), "entry model has no folds");
        anyhow::ensure!(!line.folds.is_empty(), "line model has no folds");
        Ok(Self {
            manifest,
            entry: entry.folds,
            line: line.folds,
            heading: heading.folds,
            connector: connector.folds,
        })
    }
}
