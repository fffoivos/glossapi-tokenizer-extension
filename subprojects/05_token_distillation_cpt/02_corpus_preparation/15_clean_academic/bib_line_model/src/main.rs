//! bib_line_detect — thin CLI runner around the line classifier.
//!
//!   bib_line_detect --artifacts <dir> --input <jsonl> --out-spans <jsonl>
//!
//! Input is one JSON object per line with a `text` field and an id field; output
//! is one bibliography span per line, matching the `reference_detect` convention
//! so downstream consumers do not have to learn a second format.
//!
//! Python is only an I/O driver around this binary.

use anyhow::{Context, Result};
use bib_line_model::Artifacts;
use std::path::PathBuf;

struct Args {
    artifacts: PathBuf,
    input: String,
    out_spans: String,
    source: String,
    text_field: String,
    id_field: String,
    threshold: Option<f64>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        artifacts: PathBuf::from("artifacts"),
        input: "-".into(),
        out_spans: "spans.jsonl".into(),
        source: "unknown".into(),
        text_field: "text".into(),
        id_field: "source_doc_id".into(),
        threshold: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(key) = it.next() {
        let mut value = || it.next().context("missing value");
        match key.as_str() {
            "--artifacts" => a.artifacts = PathBuf::from(value()?),
            "--input" => a.input = value()?,
            "--out-spans" => a.out_spans = value()?,
            "--source" => a.source = value()?,
            "--text-field" => a.text_field = value()?,
            "--id-field" => a.id_field = value()?,
            "--threshold" => a.threshold = Some(value()?.parse()?),
            "-h" | "--help" => {
                eprintln!("{}", include_str!("usage.txt"));
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown flag: {other}"),
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let artifacts = Artifacts::load(&args.artifacts)
        .with_context(|| format!("loading artifacts from {}", args.artifacts.display()))?;
    let threshold = args.threshold.unwrap_or(artifacts.manifest.line_threshold);
    eprintln!(
        "bib_line_detect: schema={} features={} threshold={} entry_folds={} line_folds={}",
        artifacts.manifest.schema_version,
        artifacts.manifest.feature_schema,
        threshold,
        artifacts.entry.len(),
        artifacts.line.len(),
    );
    anyhow::bail!(
        "feature extraction not yet ported — artifacts load cleanly, \
         but the per-line feature stack is still to come"
    );
}
