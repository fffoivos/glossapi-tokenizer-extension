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
use bib_line_model::{chain, features, Artifacts};
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

/// Documents, each as its list of lines. The heading and connector stages need
/// document boundaries: blank-neighbour flags, the position fraction and the
/// entry-probability windows are all clipped to the document.
fn read_documents(input: &PathBuf) -> Result<Vec<Vec<String>>> {
    let file = std::fs::File::open(input)
        .with_context(|| format!("opening {}", input.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut docs = Vec::new();
    for raw in reader.lines() {
        let raw = raw?;
        if raw.trim().is_empty() {
            continue;
        }
        let doc: serde_json::Value = serde_json::from_str(&raw)?;
        match doc.get("lines").and_then(|v| v.as_array()) {
            Some(lines) => docs.push(
                lines
                    .iter()
                    .map(|l| {
                        l.get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string()
                    })
                    .collect(),
            ),
            None => {
                let text = doc
                    .get("text")
                    .and_then(|v| v.as_str())
                    .context("document has neither a `lines` array nor a `text` field")?;
                docs.push(text.split('\n').map(str::to_string).collect());
            }
        }
    }
    Ok(docs)
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

/// Emit `probability:entry` for every line, for diffing against column 0 of the
/// deployed features.npy.
fn emit_entry(args: &Args) -> Result<()> {
    let root = args
        .artifacts
        .as_ref()
        .context("emit-entry needs --artifacts")?;
    let artifacts = Artifacts::load(root)
        .with_context(|| format!("loading artifacts from {}", root.display()))?;
    let texts = read_lines(&args.input)?;
    eprintln!(
        "bib_line_detect: {} lines, {} entry folds, {} rayon threads",
        texts.len(),
        artifacts.entry.len(),
        rayon::current_num_threads()
    );
    let start = std::time::Instant::now();
    let values: Vec<f32> = texts
        .par_iter()
        .map(|t| chain::entry_probability(&artifacts.entry, &features::line_counts(t)))
        .collect();
    eprintln!("  scored in {:.1}s", start.elapsed().as_secs_f64());
    write_npy_f32(&args.out, texts.len(), 1, &values)?;
    eprintln!("  wrote {}", args.out.display());
    Ok(())
}

/// Emit the four heading columns for every line, for diffing against columns
/// 6..9 of the deployed features.npy (`bib_header`, `bib_subheader`,
/// `non_bib_header`; column 0 of this block is the any-heading probability the
/// connector stage consumes).
fn emit_heading(args: &Args) -> Result<()> {
    use bib_line_model::chain::{
        broad_heading_candidate, heading_numeric_features, heading_probabilities,
        N_HEADING_COLUMNS,
    };
    use bib_line_model::predict::Tfidf;

    let root = args
        .artifacts
        .as_ref()
        .context("emit-heading needs --artifacts")?;
    let artifacts = Artifacts::load(root)
        .with_context(|| format!("loading artifacts from {}", root.display()))?;
    let char_tfidf: Vec<Tfidf> = artifacts
        .heading
        .iter()
        .map(|f| Tfidf::new(&f.char_tfidf))
        .collect::<Result<_>>()?;
    let word_tfidf: Vec<Tfidf> = artifacts
        .heading
        .iter()
        .map(|f| Tfidf::new(&f.word_tfidf))
        .collect::<Result<_>>()?;

    let docs = read_documents(&args.input)?;
    let total: usize = docs.iter().map(|d| d.len()).sum();
    eprintln!(
        "bib_line_detect: {} documents, {} lines, {} heading folds, {} rayon threads",
        docs.len(),
        total,
        artifacts.heading.len(),
        rayon::current_num_threads()
    );
    let start = std::time::Instant::now();

    let per_doc: Vec<Vec<f32>> = docs
        .par_iter()
        .map(|lines| {
            // The entry probabilities feeding the context window are the same
            // five-fold means already verified against column 0.
            let entry: Vec<f32> = lines
                .iter()
                .map(|t| chain::entry_probability(&artifacts.entry, &features::line_counts(t)))
                .collect();
            let n = lines.len();
            let mut out = vec![0f32; n * N_HEADING_COLUMNS];
            for (offset, text) in lines.iter().enumerate() {
                let previous_blank = offset > 0
                    && bib_line_model::shape::py_strip(&lines[offset - 1]).is_empty();
                let next_blank = offset + 1 < n
                    && bib_line_model::shape::py_strip(&lines[offset + 1]).is_empty();
                if !broad_heading_candidate(text, previous_blank, next_blank) {
                    continue;
                }
                // Windows are clipped to the document, matching the Python slices
                // `entry[max(start, absolute-30):absolute]` and
                // `entry[absolute+1:min(end, absolute+31)]`.
                let above = &entry[offset.saturating_sub(30)..offset];
                let below = &entry[(offset + 1).min(n)..(offset + 31).min(n)];
                let numeric = heading_numeric_features(
                    text,
                    previous_blank,
                    next_blank,
                    offset as f64 / (n.saturating_sub(1)).max(1) as f64,
                    above,
                    below,
                );
                let p = heading_probabilities(
                    &artifacts.heading,
                    &char_tfidf,
                    &word_tfidf,
                    text,
                    &numeric,
                );
                out[offset * N_HEADING_COLUMNS..(offset + 1) * N_HEADING_COLUMNS]
                    .copy_from_slice(&p);
            }
            out
        })
        .collect();

    eprintln!("  scored in {:.1}s", start.elapsed().as_secs_f64());
    let flat: Vec<f32> = per_doc.into_iter().flatten().collect();
    write_npy_f32(&args.out, total, N_HEADING_COLUMNS, &flat)?;
    eprintln!("  wrote {}", args.out.display());
    Ok(())
}

/// Emit the 8 negative-role flags plus the header-kind flag per line, for diffing
/// against the reference `roles.npy` / `header_kinds.npy`.
fn emit_roles(args: &Args) -> Result<()> {
    use bib_line_model::roles::{negative_role, N_ROLES};
    let texts = read_lines(&args.input)?;
    eprintln!("bib_line_detect: {} lines", texts.len());
    let start = std::time::Instant::now();
    let rows: Vec<Vec<f32>> = texts
        .par_iter()
        .map(|t| {
            let line = features::analyze(t);
            let mut row = vec![0f32; N_ROLES + 1];
            if let Some(role) = negative_role(t, &line) {
                row[role] = 1.0;
            }
            row[N_ROLES] = bib_line_model::structure::is_heading_or_subheading(t) as u8 as f32;
            row
        })
        .collect();
    eprintln!("  scored in {:.1}s", start.elapsed().as_secs_f64());
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    write_npy_f32(&args.out, texts.len(), N_ROLES + 1, &flat)?;
    eprintln!("  wrote {}", args.out.display());
    Ok(())
}

/// Emit `probability:signal_tcn`, for diffing against column 1.
fn emit_signal(args: &Args) -> Result<()> {
    use bib_line_model::roles::{negative_role, N_ROLES};
    use bib_line_model::tcn::{signal_probabilities, Tcn};

    let root = args.artifacts.as_ref().context("emit-signal needs --artifacts")?;
    let artifacts = Artifacts::load(root)?;
    let folds: Vec<Tcn> = artifacts
        .signal_tcn
        .iter()
        .map(Tcn::new)
        .collect::<Result<_>>()?;
    let docs = read_documents(&args.input)?;
    let total: usize = docs.iter().map(|d| d.len()).sum();
    eprintln!(
        "bib_line_detect: {} documents, {} lines, {} TCN folds, {} rayon threads",
        docs.len(),
        total,
        folds.len(),
        rayon::current_num_threads()
    );
    let start = std::time::Instant::now();
    let per_doc: Vec<Vec<f32>> = docs
        .par_iter()
        .map(|lines| {
            // build_signal_features: [entry, 8 role flags, header_kind > 0]
            let rows: Vec<Vec<f64>> = lines
                .iter()
                .map(|t| {
                    let line = features::analyze(t);
                    let mut row = vec![0f64; N_ROLES + 2];
                    row[0] = chain::entry_probability(&artifacts.entry, &line.counts) as f64;
                    if let Some(role) = negative_role(t, &line) {
                        row[1 + role] = 1.0;
                    }
                    row[N_ROLES + 1] =
                        bib_line_model::structure::is_heading_or_subheading(t) as u8 as f64;
                    row
                })
                .collect();
            // cohort documents are contiguous, but segment generally.
            let abs: Vec<i64> = (0..lines.len() as i64).collect();
            signal_probabilities(&folds, &rows, &abs)
        })
        .collect();
    eprintln!("  scored in {:.1}s", start.elapsed().as_secs_f64());
    let flat: Vec<f32> = per_doc.into_iter().flatten().collect();
    write_npy_f32(&args.out, total, 1, &flat)?;
    eprintln!("  wrote {}", args.out.display());
    Ok(())
}

/// Emit the 177-feature connector matrix for every candidate, in document order
/// then ascending line index — the order `_connector_probabilities` appends in.
fn emit_connector(args: &Args) -> Result<()> {
    use bib_line_model::chain::{
        broad_heading_candidate, heading_numeric_features, heading_probabilities,
    };
    use bib_line_model::connector::{
        candidate_window_mask, connector_feature_row, joined_text, DocumentContext,
        CANDIDATE_RADIUS, N_CONNECTOR_FEATURES,
    };
    use bib_line_model::predict::Tfidf;

    let root = args.artifacts.as_ref().context("emit-connector needs --artifacts")?;
    let artifacts = Artifacts::load(root)?;
    let char_tfidf: Vec<Tfidf> = artifacts
        .heading
        .iter()
        .map(|f| Tfidf::new(&f.char_tfidf))
        .collect::<Result<_>>()?;
    let word_tfidf: Vec<Tfidf> = artifacts
        .heading
        .iter()
        .map(|f| Tfidf::new(&f.word_tfidf))
        .collect::<Result<_>>()?;

    let docs = read_documents(&args.input)?;
    eprintln!(
        "bib_line_detect: {} documents, {} lines, {} rayon threads",
        docs.len(),
        docs.iter().map(|d| d.len()).sum::<usize>(),
        rayon::current_num_threads()
    );
    let start = std::time::Instant::now();

    // Absolute indices are assigned across the whole corpus, so each document's
    // base offset is the running line total.
    let mut bases = Vec::with_capacity(docs.len());
    let mut running = 0i64;
    for doc in &docs {
        bases.push(running);
        running += doc.len() as i64;
    }

    let per_doc: Vec<(Vec<i64>, Vec<f32>)> = docs
        .par_iter()
        .zip(bases.par_iter())
        .map(|(texts, base)| {
            let n = texts.len();
            let lines: Vec<features::Line> = texts.iter().map(|t| features::analyze(t)).collect();
            let entry: Vec<f32> = lines
                .iter()
                .map(|l| chain::entry_probability(&artifacts.entry, &l.counts))
                .collect();
            let abs: Vec<i64> = (0..n as i64).map(|i| base + i).collect();

            let mut heading_candidate = vec![false; n];
            let mut heading = vec![[0f32; 3]; n];
            for (offset, text) in texts.iter().enumerate() {
                let previous_blank =
                    offset > 0 && bib_line_model::shape::py_strip(&texts[offset - 1]).is_empty();
                let next_blank = offset + 1 < n
                    && bib_line_model::shape::py_strip(&texts[offset + 1]).is_empty();
                if !broad_heading_candidate(text, previous_blank, next_blank) {
                    continue;
                }
                heading_candidate[offset] = true;
                let above = &entry[offset.saturating_sub(30)..offset];
                let below = &entry[(offset + 1).min(n)..(offset + 31).min(n)];
                let numeric = heading_numeric_features(
                    text,
                    previous_blank,
                    next_blank,
                    offset as f64 / (n.saturating_sub(1)).max(1) as f64,
                    above,
                    below,
                );
                let p = heading_probabilities(
                    &artifacts.heading,
                    &char_tfidf,
                    &word_tfidf,
                    text,
                    &numeric,
                );
                // heading[:, 1:] — the three typed columns.
                heading[offset] = [p[1], p[2], p[3]];
            }

            let mask = candidate_window_mask(&entry, &heading_candidate, &abs, CANDIDATE_RADIUS);
            let ctx = DocumentContext {
                texts,
                lines: &lines,
                abs_indices: &abs,
                entry: &entry,
                heading: &heading,
            };
            let mut index_out = Vec::new();
            let mut rows_out = Vec::new();
            for i in 0..n {
                if !mask[i] {
                    continue;
                }
                // The driver deduplicates joined lines by their count vector before
                // scoring; P0D is a pure function of those counts, so scoring each
                // directly gives the same numbers.
                let joined = |neighbour: Option<usize>, left_first: bool| {
                    neighbour.map(|nb| {
                        let line = features::analyze(&joined_text(texts, i, nb, left_first));
                        let score = chain::entry_probability(&artifacts.entry, &line.counts);
                        (score, line)
                    })
                };
                let prev = if i > 0 && abs[i] - abs[i - 1] <= bib_line_model::tcn::MAX_PHYSICAL_GAP
                {
                    Some(i - 1)
                } else {
                    None
                };
                let next = if i + 1 < n
                    && abs[i + 1] - abs[i] <= bib_line_model::tcn::MAX_PHYSICAL_GAP
                {
                    Some(i + 1)
                } else {
                    None
                };
                let jp = joined(prev, true);
                let jn = joined(next, false);
                let row = connector_feature_row(
                    &ctx,
                    i,
                    &mask,
                    jp.as_ref().map(|(s, l)| (*s, l)),
                    jn.as_ref().map(|(s, l)| (*s, l)),
                );
                index_out.push(base + i as i64);
                rows_out.extend(row);
            }
            (index_out, rows_out)
        })
        .collect();

    eprintln!("  built in {:.1}s", start.elapsed().as_secs_f64());
    let mut index = Vec::new();
    let mut flat = Vec::new();
    for (i, r) in per_doc {
        index.extend(i);
        flat.extend(r);
    }
    eprintln!("  {} candidates", index.len());
    write_npy_f32(&args.out, index.len(), N_CONNECTOR_FEATURES, &flat)?;
    // The index goes alongside so the comparison can align on absolute line.
    let index_f: Vec<f32> = index.iter().map(|v| *v as f32).collect();
    write_npy_f32(&args.out.with_extension("index.npy"), index.len(), 1, &index_f)?;
    eprintln!("  wrote {}", args.out.display());
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    match args.mode.as_str() {
        "emit-table" => emit_table(&args),
        "emit-entry" => emit_entry(&args),
        "emit-heading" => emit_heading(&args),
        "emit-roles" => emit_roles(&args),
        "emit-signal" => emit_signal(&args),
        "emit-connector" => emit_connector(&args),
        "detect" => bail!("detect: the model layers are not ported yet"),
        other => {
            eprintln!("{}", include_str!("usage.txt"));
            bail!("unknown mode: {other}")
        }
    }
}
