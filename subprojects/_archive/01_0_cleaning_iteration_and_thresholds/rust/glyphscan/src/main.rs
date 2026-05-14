use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::error::Error;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{Array, ArrayRef, LargeStringArray, StringArray};
use arrow::datatypes::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use rayon::prelude::*;
use regex::Regex;
use serde::Serialize;

type DynError = Box<dyn Error + Send + Sync>;

const POSTSCRIPT_LITERALS: &[&str] = &[
    "/hyphenminus",
    "/space",
    "/period",
    "/comma",
    "/colon",
    "/semicolon",
    "/slash",
    "/backslash",
    "/parenleft",
    "/parenright",
    "/bracketleft",
    "/bracketright",
    "/braceleft",
    "/braceright",
    "/quotesingle",
    "/quotedbl",
    "/exclam",
    "/question",
    "/asterisk",
    "/plus",
    "/minus",
    "/equal",
    "/less",
    "/greater",
    "/ampersand",
    "/percent",
    "/at",
    "/dollar",
    "/numbersign",
    "/underscore",
    "/asciitilde",
    "/asciicircum",
    "/endash",
    "/emdash",
    "/hyphen",
    "/bullet",
    "/copyright",
    "/registered",
    "/trademark",
    "/degree",
    "/plusminus",
    "/multiply",
    "/divide",
    "/section",
    "/paragraph",
    "/dagger",
    "/daggerdbl",
    "/ellipsis",
    "/elipsis",
    "/glyph",
];

#[derive(Clone, Debug)]
struct Config {
    input_root: PathBuf,
    compare_root: Option<PathBuf>,
    output_dir: PathBuf,
    text_column: String,
    source_column: String,
    doc_id_column: String,
    batch_size: usize,
    workers: usize,
    sample_per_bin: usize,
    max_line_records: usize,
    include_text_in_docs: bool,
}

#[derive(Clone, Debug)]
struct MatchHit {
    start: usize,
    end: usize,
    text: String,
    family: String,
    key: String,
    left: String,
    right: String,
    glued_left: bool,
    glued_right: bool,
}

#[derive(Default, Clone, Debug, Serialize)]
struct Totals {
    files: u64,
    rows: u64,
    docs_with_matches: u64,
    lines: u64,
    matched_lines: u64,
    matches: u64,
}

#[derive(Clone, Debug, Serialize)]
struct DocumentRecord {
    source_file: String,
    row_index: u64,
    source_dataset: String,
    source_doc_id: String,
    doc_lines: u64,
    matched_lines: u64,
    total_matches: u64,
    max_line_matches: u64,
    max_line_match_non_ws_coverage: f64,
    max_line_count_per_non_ws: f64,
    context_counts_json: String,
}

#[derive(Clone, Debug, Serialize)]
struct LineRecord {
    source_file: String,
    row_index: u64,
    source_dataset: String,
    source_doc_id: String,
    line_no: u64,
    count_bin: String,
    coverage_bin: String,
    context: String,
    line_matches: u64,
    match_non_ws_coverage: f64,
    count_per_non_ws: f64,
    non_ws_chars: u64,
    matched_non_ws_chars: u64,
    glued_match_count: u64,
    families_json: String,
    matches_json: String,
    pre_line: String,
    compare_line: Option<String>,
    compare_match_count: Option<u64>,
    compare_match_non_ws_coverage: Option<f64>,
}

#[derive(Clone, Debug)]
struct SampleEntry {
    rank: u64,
    record: LineRecord,
}

impl PartialEq for SampleEntry {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank
    }
}

impl Eq for SampleEntry {}

impl Ord for SampleEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank.cmp(&other.rank)
    }
}

impl PartialOrd for SampleEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct FileResult {
    totals: Totals,
    family_counts: HashMap<String, u64>,
    count_bin_counts: HashMap<String, u64>,
    coverage_bin_counts: HashMap<String, u64>,
    matrix_counts: HashMap<String, u64>,
    context_counts: HashMap<String, u64>,
    match_key_counts: HashMap<String, u64>,
    documents: Vec<DocumentRecord>,
    lines: Vec<LineRecord>,
    samples: HashMap<String, BinaryHeap<SampleEntry>>,
}

#[derive(Serialize)]
struct Summary<'a> {
    generated_at_unix: u64,
    input_root: &'a str,
    compare_root: Option<&'a str>,
    text_column: &'a str,
    source_column: &'a str,
    doc_id_column: &'a str,
    batch_size: usize,
    workers: usize,
    totals: Totals,
    family_counts: HashMap<String, u64>,
    count_bin_counts: HashMap<String, u64>,
    coverage_bin_counts: HashMap<String, u64>,
    matrix_counts: HashMap<String, u64>,
    context_counts: HashMap<String, u64>,
    match_key_counts: HashMap<String, u64>,
}

fn main() -> Result<(), DynError> {
    let config = parse_args()?;
    fs::create_dir_all(&config.output_dir)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .build_global()?;

    let matcher = build_matcher()?;
    let mut files = Vec::new();
    collect_parquets(&config.input_root, &mut files)?;
    files.sort();

    eprintln!(
        "glyphscan: scanning {} parquet files with {} workers",
        files.len(),
        config.workers
    );

    let results: Vec<FileResult> = files
        .par_iter()
        .map(|path| process_file(path, &config, &matcher))
        .collect::<Result<Vec<_>, _>>()?;

    let mut merged = FileResult::default();
    for result in results {
        merge_result(&mut merged, result, config.sample_per_bin, config.max_line_records);
    }
    merged.totals.files = files.len() as u64;

    write_outputs(&config, &merged)?;
    eprintln!(
        "glyphscan: done rows={} matched_lines={} matches={}",
        merged.totals.rows, merged.totals.matched_lines, merged.totals.matches
    );
    Ok(())
}

fn parse_args() -> Result<Config, DynError> {
    let mut args = std::env::args().skip(1);
    let mut config = Config {
        input_root: PathBuf::new(),
        compare_root: None,
        output_dir: PathBuf::new(),
        text_column: "text".to_string(),
        source_column: "source_dataset".to_string(),
        doc_id_column: "source_doc_id".to_string(),
        batch_size: 512,
        workers: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        sample_per_bin: 24,
        max_line_records: 1_000_000,
        include_text_in_docs: false,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input-root" => config.input_root = PathBuf::from(next_arg(&mut args, &arg)?),
            "--compare-root" => config.compare_root = Some(PathBuf::from(next_arg(&mut args, &arg)?)),
            "--output-dir" => config.output_dir = PathBuf::from(next_arg(&mut args, &arg)?),
            "--text-column" => config.text_column = next_arg(&mut args, &arg)?,
            "--source-column" => config.source_column = next_arg(&mut args, &arg)?,
            "--doc-id-column" => config.doc_id_column = next_arg(&mut args, &arg)?,
            "--batch-size" => config.batch_size = next_arg(&mut args, &arg)?.parse()?,
            "--workers" => config.workers = next_arg(&mut args, &arg)?.parse()?,
            "--sample-per-bin" => config.sample_per_bin = next_arg(&mut args, &arg)?.parse()?,
            "--max-line-records" => config.max_line_records = next_arg(&mut args, &arg)?.parse()?,
            "--include-text-in-docs" => config.include_text_in_docs = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }

    if config.input_root.as_os_str().is_empty() {
        return Err("--input-root is required".into());
    }
    if config.output_dir.as_os_str().is_empty() {
        return Err("--output-dir is required".into());
    }
    Ok(config)
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, DynError> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "glyphscan --input-root DIR --output-dir DIR [--compare-root DIR] [--workers N]\n\
         Scans parquet text rows for context-blind GLYPH/PostScript PDF residue.\n\
         Outputs summary.json, documents.csv, lines.jsonl, samples_by_bin.jsonl,\n\
         family_summary.csv, bin_summary.csv."
    );
}

fn build_matcher() -> Result<Regex, DynError> {
    let mut parts = vec![
        r"GLYPH&lt;[^&]{1,240}?&gt;".to_string(),
        r"GLYPH<[^>]{1,240}>".to_string(),
        r"GLYPH\([^)]{1,240}\)".to_string(),
        r"glyph\[[^\]]{1,240}\]".to_string(),
        r"glyph(?:&lt;|<)c=\d+,font=/[^>&]{1,240}(?:&gt;|>)".to_string(),
        r"(?:&lt;|<)c=\d+,font=/[^>&]{1,240}(?:&gt;|>)glyph".to_string(),
        r"/[A-Z]{6}\+[A-Z][A-Za-z0-9-]+".to_string(),
        r"/uni[0-9A-Fa-f]{4,6}".to_string(),
        r"/g(?:id)?\d+".to_string(),
        r"CID\+".to_string(),
    ];
    for literal in POSTSCRIPT_LITERALS {
        parts.push(regex::escape(literal));
    }
    parts.push(r"(?:GLYPH)+".to_string());
    Regex::new(&format!(r"(?i){}", parts.join("|"))).map_err(|e| e.into())
}

fn collect_parquets(root: &Path, out: &mut Vec<PathBuf>) -> Result<(), DynError> {
    if root.is_file() {
        if root.extension().and_then(|s| s.to_str()) == Some("parquet") {
            out.push(root.to_path_buf());
        }
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_parquets(&path, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
            out.push(path);
        }
    }
    Ok(())
}

fn process_file(path: &Path, config: &Config, matcher: &Regex) -> Result<FileResult, DynError> {
    let compare_path = config
        .compare_root
        .as_ref()
        .map(|root| root.join(path.file_name().unwrap_or_default()));

    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let projection = projection_for(&builder.schema(), builder.parquet_schema(), config)?;
    let mut reader = builder
        .with_batch_size(config.batch_size)
        .with_projection(projection)
        .build()?;

    let mut compare_reader = match compare_path.as_ref().filter(|p| p.exists()) {
        Some(path) => {
            let file = File::open(path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let projection = projection_for(&builder.schema(), builder.parquet_schema(), config)?;
            Some(
                builder
                    .with_batch_size(config.batch_size)
                    .with_projection(projection)
                    .build()?,
            )
        }
        None => None,
    };

    let mut result = FileResult::default();
    let mut row_base = 0u64;
    let source_file = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    while let Some(batch) = reader.next() {
        let batch = batch?;
        let compare_batch = match compare_reader.as_mut() {
            Some(reader) => reader.next().transpose()?,
            None => None,
        };

        let text_idx = field_index(batch.schema(), &config.text_column)?;
        let source_idx = field_index(batch.schema(), &config.source_column).ok();
        let doc_idx = field_index(batch.schema(), &config.doc_id_column).ok();

        let text_col = batch.column(text_idx).clone();
        let source_col = source_idx.map(|idx| batch.column(idx).clone());
        let doc_col = doc_idx.map(|idx| batch.column(idx).clone());

        let compare_text_col = match compare_batch.as_ref() {
            Some(batch) => field_index(batch.schema(), &config.text_column)
                .ok()
                .map(|idx| batch.column(idx).clone()),
            None => None,
        };

        for row in 0..batch.num_rows() {
            result.totals.rows += 1;
            let text = string_value(&text_col, row).unwrap_or("");
            let source_dataset = source_col
                .as_ref()
                .and_then(|col| string_value(col, row))
                .unwrap_or_else(|| path.file_stem().and_then(|s| s.to_str()).unwrap_or(""))
                .to_string();
            let source_doc_id = doc_col
                .as_ref()
                .and_then(|col| string_value(col, row))
                .unwrap_or("")
                .to_string();
            let compare_text = compare_text_col.as_ref().and_then(|col| string_value(col, row));

            scan_document(
                text,
                compare_text,
                matcher,
                &source_file,
                row_base + row as u64,
                &source_dataset,
                &source_doc_id,
                config,
                &mut result,
            )?;
        }
        row_base += batch.num_rows() as u64;
    }

    Ok(result)
}

fn projection_for(
    schema: &SchemaRef,
    parquet_schema: &parquet::schema::types::SchemaDescriptor,
    config: &Config,
) -> Result<ProjectionMask, DynError> {
    let mut indices = Vec::new();
    for name in [&config.text_column, &config.source_column, &config.doc_id_column] {
        if let Ok(idx) = schema.index_of(name) {
            indices.push(idx);
        }
    }
    if !indices.iter().any(|idx| schema.field(*idx).name() == &config.text_column) {
        return Err(format!("missing text column {}", config.text_column).into());
    }
    Ok(ProjectionMask::roots(parquet_schema, indices))
}

fn field_index(schema: SchemaRef, name: &str) -> Result<usize, DynError> {
    schema.index_of(name).map_err(|e| e.into())
}

fn string_value(array: &ArrayRef, row: usize) -> Option<&str> {
    if array.is_null(row) {
        return None;
    }
    if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
        return Some(arr.value(row));
    }
    if let Some(arr) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Some(arr.value(row));
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn scan_document(
    text: &str,
    compare_text: Option<&str>,
    matcher: &Regex,
    source_file: &str,
    row_index: u64,
    source_dataset: &str,
    source_doc_id: &str,
    config: &Config,
    result: &mut FileResult,
) -> Result<(), DynError> {
    let compare_lines: Option<Vec<&str>> = compare_text.map(|text| text.split('\n').collect());
    let mut doc_lines = 0u64;
    let mut doc_matched_lines = 0u64;
    let mut doc_matches = 0u64;
    let mut max_line_matches = 0u64;
    let mut max_line_coverage = 0.0;
    let mut max_line_count_per_non_ws = 0.0;
    let mut doc_context_counts: HashMap<String, u64> = HashMap::new();
    let mut in_code_fence = false;
    let mut in_latex_block = false;

    for (idx, line) in text.split('\n').enumerate() {
        doc_lines += 1;
        result.totals.lines += 1;
        let line_no = idx as u64 + 1;
        let context = classify_context(line, in_code_fence, in_latex_block);
        let compare_line = compare_lines
            .as_ref()
            .and_then(|lines| lines.get(idx))
            .copied();

        let hits = find_hits(matcher, line);
        if !hits.is_empty() {
            let metrics = line_metrics(line, &hits);
            let compare_metrics = compare_line.map(|line| {
                let hits = find_hits(matcher, line);
                let metrics = line_metrics(line, &hits);
                (hits.len() as u64, metrics.0)
            });
            let count_bin = count_bin(hits.len() as u64).to_string();
            let coverage_bin = coverage_bin(metrics.0).to_string();
            let matrix_key = format!("{count_bin}|{coverage_bin}");
            let mut family_counts: HashMap<String, u64> = HashMap::new();
            let mut match_key_counts: HashMap<String, u64> = HashMap::new();
            let mut glued = 0u64;
            for hit in &hits {
                *family_counts.entry(hit.family.clone()).or_default() += 1;
                *result.family_counts.entry(hit.family.clone()).or_default() += 1;
                *match_key_counts.entry(hit.key.clone()).or_default() += 1;
                *result.match_key_counts.entry(hit.key.clone()).or_default() += 1;
                if hit.glued_left || hit.glued_right {
                    glued += 1;
                }
            }

            result.totals.matched_lines += 1;
            result.totals.matches += hits.len() as u64;
            *result.count_bin_counts.entry(count_bin.clone()).or_default() += 1;
            *result.coverage_bin_counts.entry(coverage_bin.clone()).or_default() += 1;
            *result.matrix_counts.entry(matrix_key.clone()).or_default() += 1;
            *result.context_counts.entry(context.clone()).or_default() += 1;
            *doc_context_counts.entry(context.clone()).or_default() += 1;

            doc_matched_lines += 1;
            doc_matches += hits.len() as u64;
            max_line_matches = max_line_matches.max(hits.len() as u64);
            if metrics.0 > max_line_coverage {
                max_line_coverage = metrics.0;
            }
            if metrics.1 > max_line_count_per_non_ws {
                max_line_count_per_non_ws = metrics.1;
            }

            let record = LineRecord {
                source_file: source_file.to_string(),
                row_index,
                source_dataset: source_dataset.to_string(),
                source_doc_id: source_doc_id.to_string(),
                line_no,
                count_bin,
                coverage_bin,
                context,
                line_matches: hits.len() as u64,
                match_non_ws_coverage: metrics.0,
                count_per_non_ws: metrics.1,
                non_ws_chars: metrics.2,
                matched_non_ws_chars: metrics.3,
                glued_match_count: glued,
                families_json: serde_json::to_string(&family_counts)?,
                matches_json: serde_json::to_string(&hits_for_json(&hits))?,
                pre_line: line.to_string(),
                compare_line: compare_line.map(ToOwned::to_owned),
                compare_match_count: compare_metrics.map(|m| m.0),
                compare_match_non_ws_coverage: compare_metrics.map(|m| m.1),
            };

            if result.lines.len() < config.max_line_records {
                result.lines.push(record.clone());
            }
            add_sample(&mut result.samples, matrix_key, record, config.sample_per_bin);
        }

        update_context_state(line, &mut in_code_fence, &mut in_latex_block);
    }

    if doc_matches > 0 {
        result.totals.docs_with_matches += 1;
        result.documents.push(DocumentRecord {
            source_file: source_file.to_string(),
            row_index,
            source_dataset: source_dataset.to_string(),
            source_doc_id: source_doc_id.to_string(),
            doc_lines,
            matched_lines: doc_matched_lines,
            total_matches: doc_matches,
            max_line_matches,
            max_line_match_non_ws_coverage: max_line_coverage,
            max_line_count_per_non_ws,
            context_counts_json: serde_json::to_string(&doc_context_counts)?,
        });
    }
    Ok(())
}

fn find_hits(matcher: &Regex, line: &str) -> Vec<MatchHit> {
    matcher
        .find_iter(line)
        .map(|m| {
            let text = m.as_str().to_string();
            let (left, glued_left) = neighbor(line, m.start(), true);
            let (right, glued_right) = neighbor(line, m.end(), false);
            let (family, key) = classify_match(&text, glued_left, glued_right);
            MatchHit {
                start: m.start(),
                end: m.end(),
                text,
                family,
                key,
                left,
                right,
                glued_left,
                glued_right,
            }
        })
        .collect()
}

fn neighbor(line: &str, byte_idx: usize, left: bool) -> (String, bool) {
    let ch = if left {
        line[..byte_idx].chars().next_back()
    } else {
        line[byte_idx..].chars().next()
    };
    let text = ch.map(|c| c.to_string()).unwrap_or_default();
    let glued = ch.map(|c| c.is_alphanumeric()).unwrap_or(false);
    (text, glued)
}

fn classify_match(text: &str, glued_left: bool, glued_right: bool) -> (String, String) {
    let lower = text.to_ascii_lowercase();
    if lower.starts_with("glyph<")
        || lower.starts_with("glyph&lt;")
        || lower.starts_with("glyph(")
        || lower.starts_with("glyph[")
        || lower.contains("font=/")
    {
        ("glyph_structured".to_string(), "glyph_structured".to_string())
    } else if lower.contains("glyph") {
        let family = if glued_left || glued_right {
            "glyph_stem_embedded"
        } else {
            "glyph_stem_token"
        };
        let key = if is_repeated_glyph_stem(&lower) {
            format!("glyph_repeat_{}", lower.len() / "glyph".len())
        } else {
            lower
        };
        (family.to_string(), key)
    } else if lower.starts_with("/uni") {
        ("postscript_uni".to_string(), lower)
    } else if is_gid_name(&lower) {
        ("postscript_gid".to_string(), lower)
    } else if lower == "cid+" {
        ("cid_prefix".to_string(), "cid+".to_string())
    } else if is_font_subset_name(text) {
        ("font_subset".to_string(), "font_subset".to_string())
    } else {
        let family = if glued_left || glued_right {
            "postscript_literal_embedded"
        } else {
            "postscript_literal_token"
        };
        (family.to_string(), lower)
    }
}

fn is_repeated_glyph_stem(lower: &str) -> bool {
    if lower.is_empty() || lower.len() % "glyph".len() != 0 {
        return false;
    }
    lower.as_bytes().chunks("glyph".len()).all(|chunk| chunk == b"glyph")
}

fn is_gid_name(lower: &str) -> bool {
    if let Some(rest) = lower.strip_prefix("/gid") {
        return !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit());
    }
    if let Some(rest) = lower.strip_prefix("/g") {
        return !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit());
    }
    false
}

fn is_font_subset_name(text: &str) -> bool {
    let Some(rest) = text.strip_prefix('/') else {
        return false;
    };
    let Some((prefix, suffix)) = rest.split_once('+') else {
        return false;
    };
    prefix.len() == 6
        && prefix.bytes().all(|b| b.is_ascii_uppercase())
        && suffix
            .bytes()
            .next()
            .map(|b| b.is_ascii_uppercase())
            .unwrap_or(false)
}

fn line_metrics(line: &str, hits: &[MatchHit]) -> (f64, f64, u64, u64) {
    let non_ws = line.chars().filter(|c| !c.is_whitespace()).count() as u64;
    if non_ws == 0 {
        return (0.0, 0.0, 0, 0);
    }
    let mut spans: Vec<(usize, usize)> = hits.iter().map(|h| (h.start, h.end)).collect();
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in spans {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    let matched_non_ws = merged
        .iter()
        .map(|(start, end)| line[*start..*end].chars().filter(|c| !c.is_whitespace()).count() as u64)
        .sum::<u64>();
    (
        matched_non_ws as f64 / non_ws as f64,
        hits.len() as f64 / non_ws as f64,
        non_ws,
        matched_non_ws,
    )
}

fn count_bin(count: u64) -> &'static str {
    match count {
        0 => "0",
        1..=3 => "1-3",
        4..=8 => "4-8",
        9..=16 => "9-16",
        17..=32 => "17-32",
        _ => "33+",
    }
}

fn coverage_bin(coverage: f64) -> &'static str {
    let pct = coverage * 100.0;
    match pct {
        x if x < 10.0 => "00-10%",
        x if x < 20.0 => "10-20%",
        x if x < 30.0 => "20-30%",
        x if x < 40.0 => "30-40%",
        x if x < 50.0 => "40-50%",
        x if x < 60.0 => "50-60%",
        x if x < 70.0 => "60-70%",
        x if x < 80.0 => "70-80%",
        x if x < 90.0 => "80-90%",
        _ => "90-100%",
    }
}

fn classify_context(line: &str, in_code_fence: bool, in_latex_block: bool) -> String {
    let trimmed = line.trim();
    if in_code_fence || trimmed.starts_with("```") || trimmed.starts_with("~~~") {
        return "code_fence".to_string();
    }
    if in_latex_block || trimmed == "$$" || trimmed.contains("$$") {
        return "latex_math".to_string();
    }
    if trimmed.starts_with("<!--") && trimmed.ends_with("-->") {
        return "html_comment".to_string();
    }
    if trimmed.starts_with('|') || trimmed.matches('|').count() >= 2 {
        return "markdown_table".to_string();
    }
    if trimmed.starts_with('#') {
        return "markdown_heading".to_string();
    }
    if trimmed.starts_with("- ") || trimmed.starts_with("* ") || trimmed.starts_with("+ ") {
        return "markdown_list".to_string();
    }
    if trimmed.contains("http://") || trimmed.contains("https://") || trimmed.contains("www.") {
        return "url_or_link".to_string();
    }
    "prose_or_other".to_string()
}

fn update_context_state(line: &str, in_code_fence: &mut bool, in_latex_block: &mut bool) {
    let trimmed = line.trim_start();
    if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
        *in_code_fence = !*in_code_fence;
    }
    if line.matches("$$").count() % 2 == 1 {
        *in_latex_block = !*in_latex_block;
    }
}

fn hits_for_json(hits: &[MatchHit]) -> Vec<serde_json::Value> {
    hits.iter()
        .map(|hit| {
            serde_json::json!({
                "start": hit.start,
                "end": hit.end,
                "text": hit.text,
                "family": hit.family,
                "key": hit.key,
                "left": hit.left,
                "right": hit.right,
                "glued_left": hit.glued_left,
                "glued_right": hit.glued_right,
            })
        })
        .collect()
}

fn add_sample(
    samples: &mut HashMap<String, BinaryHeap<SampleEntry>>,
    key: String,
    record: LineRecord,
    sample_per_bin: usize,
) {
    let rank = stable_rank(&record);
    let heap = samples.entry(key).or_default();
    heap.push(SampleEntry { rank, record });
    if heap.len() > sample_per_bin {
        heap.pop();
    }
}

fn stable_rank(record: &LineRecord) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    record.source_file.hash(&mut hasher);
    record.row_index.hash(&mut hasher);
    record.line_no.hash(&mut hasher);
    record.pre_line.hash(&mut hasher);
    hasher.finish()
}

fn merge_result(
    target: &mut FileResult,
    mut source: FileResult,
    sample_per_bin: usize,
    max_line_records: usize,
) {
    target.totals.rows += source.totals.rows;
    target.totals.docs_with_matches += source.totals.docs_with_matches;
    target.totals.lines += source.totals.lines;
    target.totals.matched_lines += source.totals.matched_lines;
    target.totals.matches += source.totals.matches;
    merge_counts(&mut target.family_counts, source.family_counts);
    merge_counts(&mut target.count_bin_counts, source.count_bin_counts);
    merge_counts(&mut target.coverage_bin_counts, source.coverage_bin_counts);
    merge_counts(&mut target.matrix_counts, source.matrix_counts);
    merge_counts(&mut target.context_counts, source.context_counts);
    merge_counts(&mut target.match_key_counts, source.match_key_counts);
    target.documents.append(&mut source.documents);
    if target.lines.len() < max_line_records {
        let remaining = max_line_records - target.lines.len();
        target.lines.extend(source.lines.into_iter().take(remaining));
    }
    for (key, mut heap) in source.samples {
        let target_heap = target.samples.entry(key).or_default();
        while let Some(entry) = heap.pop() {
            target_heap.push(entry);
            if target_heap.len() > sample_per_bin {
                target_heap.pop();
            }
        }
    }
}

fn merge_counts(target: &mut HashMap<String, u64>, source: HashMap<String, u64>) {
    for (key, value) in source {
        *target.entry(key).or_default() += value;
    }
}

fn write_outputs(config: &Config, result: &FileResult) -> Result<(), DynError> {
    let generated_at_unix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let input_root = config.input_root.to_string_lossy();
    let compare_root_string = config
        .compare_root
        .as_ref()
        .map(|p| p.to_string_lossy().to_string());
    let summary = Summary {
        generated_at_unix,
        input_root: &input_root,
        compare_root: compare_root_string.as_deref(),
        text_column: &config.text_column,
        source_column: &config.source_column,
        doc_id_column: &config.doc_id_column,
        batch_size: config.batch_size,
        workers: config.workers,
        totals: result.totals.clone(),
        family_counts: result.family_counts.clone(),
        count_bin_counts: result.count_bin_counts.clone(),
        coverage_bin_counts: result.coverage_bin_counts.clone(),
        matrix_counts: result.matrix_counts.clone(),
        context_counts: result.context_counts.clone(),
        match_key_counts: result.match_key_counts.clone(),
    };

    fs::write(
        config.output_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;

    let mut doc_writer = csv::Writer::from_path(config.output_dir.join("documents.csv"))?;
    for doc in &result.documents {
        doc_writer.serialize(doc)?;
    }
    doc_writer.flush()?;

    let mut line_writer = BufWriter::new(File::create(config.output_dir.join("lines.jsonl"))?);
    for line in &result.lines {
        serde_json::to_writer(&mut line_writer, line)?;
        line_writer.write_all(b"\n")?;
    }
    line_writer.flush()?;

    let mut sample_writer = BufWriter::new(File::create(config.output_dir.join("samples_by_bin.jsonl"))?);
    let mut sample_keys: Vec<_> = result.samples.keys().cloned().collect();
    sample_keys.sort();
    for key in sample_keys {
        if let Some(heap) = result.samples.get(&key) {
            let mut entries: Vec<_> = heap.iter().collect();
            entries.sort_by_key(|entry| Reverse(entry.rank));
            for entry in entries {
                serde_json::to_writer(
                    &mut sample_writer,
                    &serde_json::json!({"sample_bin": key, "record": entry.record}),
                )?;
                sample_writer.write_all(b"\n")?;
            }
        }
    }
    sample_writer.flush()?;

    write_count_csv(config.output_dir.join("family_summary.csv"), "family", &result.family_counts)?;
    write_count_csv(config.output_dir.join("count_bin_summary.csv"), "count_bin", &result.count_bin_counts)?;
    write_count_csv(
        config.output_dir.join("coverage_bin_summary.csv"),
        "coverage_bin",
        &result.coverage_bin_counts,
    )?;
    write_count_csv(config.output_dir.join("matrix_summary.csv"), "count_coverage_bin", &result.matrix_counts)?;
    write_count_csv(config.output_dir.join("context_summary.csv"), "context", &result.context_counts)?;
    write_count_csv(config.output_dir.join("match_key_summary.csv"), "match_key", &result.match_key_counts)?;
    Ok(())
}

fn write_count_csv(path: PathBuf, name_col: &str, counts: &HashMap<String, u64>) -> Result<(), DynError> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([name_col, "count"])?;
    let mut items: Vec<_> = counts.iter().collect();
    items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (name, count) in items {
        writer.write_record([name.as_str(), &count.to_string()])?;
    }
    writer.flush()?;
    Ok(())
}
