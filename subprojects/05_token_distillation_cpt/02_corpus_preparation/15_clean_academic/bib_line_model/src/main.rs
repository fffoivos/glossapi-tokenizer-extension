//! bib_line_detect — thin CLI runner around the line classifier.
//!
//! Two modes:
//!
//!   bib_line_detect emit-table --input <jsonl> --out <npy>
//!       Write the deterministic columns (10..126 of the v3 contract) for every line
//!       of every document, in document order. This exists to be diffed against the
//!       deployed `features.npy` — the strongest available parity gate, because it
//!       compares against what the reference pipeline actually produced at scale
//!       rather than against fixtures regenerated from the same code.
//!
//!   bib_line_detect detect --artifacts <dir> --input <jsonl> --out-spans <jsonl>
//!       The real runner. Pending the model layers.
//!
//! Input is one JSON object per line with a `lines` array (each `{text, abs_idx}`),
//! matching the cohort documents.

use anyhow::{bail, Context, Result};

// See the Cargo.toml note: with the system allocator this binary got *slower* past
// 64 threads on a 288-core node. The work is embarrassingly parallel, so the
// ceiling was allocator contention rather than the algorithm.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use bib_line_model::table::{deterministic_row, N_COLUMNS, N_PROBABILITY};
use rayon::prelude::*;
use std::io::{BufRead, BufWriter, Write};
use std::path::PathBuf;

const DETERMINISTIC_COLUMNS: usize = N_COLUMNS - N_PROBABILITY;

struct Args {
    mode: String,
    input: PathBuf,
    out: PathBuf,
    artifacts: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let mode = it.next().unwrap_or_else(|| "help".into());
    let mut a = Args {
        mode,
        input: PathBuf::from("-"),
        out: PathBuf::from("out.npy"),
        artifacts: None,
    };
    while let Some(key) = it.next() {
        let mut value = || it.next().context("missing value");
        match key.as_str() {
            "--input" => a.input = PathBuf::from(value()?),
            "--out" => a.out = PathBuf::from(value()?),
            "--artifacts" => a.artifacts = Some(PathBuf::from(value()?)),
            "-h" | "--help" => {
                eprintln!("{}", include_str!("usage.txt"));
                std::process::exit(0);
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    Ok(a)
}

/// Minimal `.npy` v1.0 writer for a C-order float32 matrix.
fn write_npy_f32(path: &PathBuf, rows: usize, cols: usize, data: &[f32]) -> Result<()> {
    anyhow::ensure!(data.len() == rows * cols, "data does not match shape");
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}"
    );
    // The header (magic + version + length prefix included) must be 64-byte aligned.
    let prefix = 10;
    let mut total = prefix + header.len() + 1;
    while total % 64 != 0 {
        header.push(' ');
        total += 1;
    }
    header.push('\n');
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(b"\x93NUMPY")?;
    w.write_all(&[1u8, 0u8])?;
    w.write_all(&(header.len() as u16).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    for v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

fn read_lines(input: &PathBuf) -> Result<Vec<String>> {
    let file = std::fs::File::open(input)
        .with_context(|| format!("opening {}", input.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut texts = Vec::new();
    for raw in reader.lines() {
        let raw = raw?;
        if raw.trim().is_empty() {
            continue;
        }
        let doc: serde_json::Value = serde_json::from_str(&raw)?;
        // Two document shapes are in circulation: the sealed-cohort form with an
        // explicit `lines` array, and the corpus form carrying whole-document `text`
        // plus a [line_start, line_end) span. For the latter the pipeline's line
        // inventory is `text.split("\n")` — verified to reproduce the reference
        // count exactly (210,704 over the 150 cohort-2 documents), including the
        // empty trailing field a terminating newline produces.
        match doc.get("lines").and_then(|v| v.as_array()) {
            Some(lines) => texts.extend(lines.iter().map(|line| {
                line.get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string()
            })),
            None => {
                let text = doc
                    .get("text")
                    .and_then(|v| v.as_str())
                    .context("document has neither a `lines` array nor a `text` field")?;
                texts.extend(text.split('\n').map(str::to_string));
            }
        }
    }
    Ok(texts)
}

fn emit_table(args: &Args) -> Result<()> {
    let texts = read_lines(&args.input)?;
    eprintln!(
        "bib_line_detect: {} lines, {} rayon threads",
        texts.len(),
        rayon::current_num_threads()
    );
    let start = std::time::Instant::now();
    let rows: Vec<Vec<f32>> = texts.par_iter().map(|t| deterministic_row(t)).collect();
    let elapsed = start.elapsed();
    eprintln!(
        "  featurized in {:.1}s ({:.0} lines/s)",
        elapsed.as_secs_f64(),
        texts.len() as f64 / elapsed.as_secs_f64()
    );
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    write_npy_f32(&args.out, texts.len(), DETERMINISTIC_COLUMNS, &flat)?;
    eprintln!("  wrote {} ({} cols)", args.out.display(), DETERMINISTIC_COLUMNS);
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    match args.mode.as_str() {
        "emit-table" => emit_table(&args),
        "detect" => bail!("detect: the model layers are not ported yet"),
        other => {
            eprintln!("{}", include_str!("usage.txt"));
            bail!("unknown mode: {other}")
        }
    }
}
