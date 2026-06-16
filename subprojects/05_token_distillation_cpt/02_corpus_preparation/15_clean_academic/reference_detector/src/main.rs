//! reference_detect — thin CLI runner around the detector core.
//!
//! Modes:
//!   --mode wholedoc   input JSONL[.zst], one doc/line {id, text}  (greek_phd, openarchives)
//!   --mode sections   input JSONL[.zst], one doc/line {filename, rows:[{header,section,predicted_section,positional_fraction,row_id}]}
//!
//! Output:
//!   --out-spans     PATH   newline-delimited Span JSON (the auditable reference sidecar)
//!   --out-counters  PATH   newline-delimited per-doc Counters JSON
//!
//! Calibration knobs default to "segment-and-emit-everything" (no baked deletion threshold).
//! Python wraps this binary as a pure I/O driver; all per-doc/per-line work happens here in Rust.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};

use rayon::prelude::*;
use serde_json::Value;

use reference_detector::{detect_doc, detect_sections, DetectConfig, DocResult, SectionRow};

struct Args {
    mode: String,
    input: String,
    out_spans: String,
    out_counters: String,
    source: String,
    text_fields: Vec<String>,
    id_fields: Vec<String>,
    cfg: DetectConfig,
}

fn parse_args() -> Args {
    let mut a = Args {
        mode: "wholedoc".into(),
        input: "-".into(),
        out_spans: "spans.jsonl".into(),
        out_counters: "counters.jsonl".into(),
        source: "unknown".into(),
        text_fields: vec!["text".into(), "document".into(), "content".into(), "markdown".into(), "md".into()],
        id_fields: vec!["id".into(), "source_doc_id".into(), "doc_id".into(), "filename".into(), "md_filename".into()],
        cfg: DetectConfig::default(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().expect("missing value");
        match k.as_str() {
            "--mode" => a.mode = v(),
            "--input" => a.input = v(),
            "--out-spans" => a.out_spans = v(),
            "--out-counters" => a.out_counters = v(),
            "--source" => a.source = v(),
            "--text-field" => a.text_fields = vec![v()],
            "--id-field" => a.id_fields = vec![v()],
            "--min-position-fraction" => a.cfg.min_position_fraction = v().parse().unwrap(),
            "--bib-min-year-density" => a.cfg.bib_min_year_density = v().parse().unwrap(),
            "--footnote-prose-min-chars" => a.cfg.footnote_prose_min_chars = v().parse().unwrap(),
            "--footnote-cite-max-greek" => a.cfg.footnote_cite_max_greek = v().parse().unwrap(),
            "--cv-front-max-pos" => a.cfg.cv_front_max_pos = v().parse().unwrap(),
            "--emit-intext-spans" => a.cfg.emit_intext_spans = true,
            other => panic!("unknown arg: {other}"),
        }
    }
    a
}

fn open_reader(path: &str) -> Box<dyn BufRead> {
    if path == "-" {
        return Box::new(BufReader::new(io::stdin()));
    }
    let f = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    if path.ends_with(".zst") {
        let dec = zstd::stream::read::Decoder::new(f).expect("zstd decoder");
        Box::new(BufReader::new(dec))
    } else {
        Box::new(BufReader::new(f))
    }
}

fn pick_str<'a>(v: &'a Value, keys: &[String]) -> Option<&'a str> {
    for k in keys {
        if let Some(s) = v.get(k).and_then(|x| x.as_str()) {
            return Some(s);
        }
    }
    None
}

fn process_line(a: &Args, idx: usize, line: &str) -> Result<DocResult, String> {
    let v: Value = serde_json::from_str(line).map_err(|e| format!("json parse error: {e}"))?;
    if a.mode == "sections" {
        let filename = pick_str(&v, &a.id_fields).map(|s| s.to_string()).unwrap_or_else(|| format!("doc{idx}"));
        let rows_v = v
            .get("rows")
            .and_then(|x| x.as_array())
            .ok_or_else(|| "missing or non-array 'rows' field".to_string())?;
        let n = rows_v.len().max(1);
        let rows: Vec<SectionRow> = rows_v
            .iter()
            .enumerate()
            .map(|(i, r)| SectionRow {
                row_id: r.get("row_id").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                header: r.get("header").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                section: r.get("section").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                predicted_section: r.get("predicted_section").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                positional_fraction: r
                    .get("positional_fraction")
                    .and_then(|x| x.as_f64())
                    .map(|f| f as f32)
                    .unwrap_or((i as f32) / ((n - 1).max(1) as f32)),
            })
            .collect();
        Ok(detect_sections(&filename, &a.source, &rows, &a.cfg))
    } else {
        let id = pick_str(&v, &a.id_fields).map(|s| s.to_string()).unwrap_or_else(|| format!("doc{idx}"));
        let text = pick_str(&v, &a.text_fields)
            .ok_or_else(|| format!("missing text field (tried {:?})", a.text_fields))?;
        Ok(detect_doc(&id, &a.source, text, &a.cfg))
    }
}

fn main() {
    let a = parse_args();
    let mut reader = open_reader(&a.input);
    let mut spans_w = BufWriter::new(File::create(&a.out_spans).expect("out-spans"));
    let mut counters_w = BufWriter::new(File::create(&a.out_counters).expect("out-counters"));

    const BATCH: usize = 2000;
    let mut batch: Vec<(usize, String)> = Vec::with_capacity(BATCH);
    let mut idx = 0usize;
    let mut n_docs = 0usize;
    let mut n_err = 0usize;
    let mut buf = String::new();

    // Returns (n_ok, n_err). Malformed/schema-mismatched rows are NOT silently dropped: each emits an
    // auditable error record to the counters stream (`{"doc_idx","error"}`) and is counted, so a bad
    // row fails loudly (final summary + non-zero exit) instead of vanishing from the counters.
    let mut flush = |batch: &mut Vec<(usize, String)>, spans_w: &mut dyn Write, counters_w: &mut dyn Write| {
        let results: Vec<(usize, Result<DocResult, String>)> = batch
            .par_iter()
            .map(|(i, l)| (*i, process_line(&a, *i, l)))
            .collect();
        let mut ok = 0usize;
        let mut err = 0usize;
        for (i, r) in &results {
            match r {
                Ok(r) => {
                    ok += 1;
                    let cj = serde_json::to_string(&r.counters).unwrap();
                    counters_w.write_all(cj.as_bytes()).unwrap();
                    counters_w.write_all(b"\n").unwrap();
                    for s in &r.spans {
                        let sj = serde_json::to_string(s).unwrap();
                        spans_w.write_all(sj.as_bytes()).unwrap();
                        spans_w.write_all(b"\n").unwrap();
                    }
                }
                Err(e) => {
                    err += 1;
                    let er = serde_json::json!({"doc_idx": i, "error": e});
                    counters_w.write_all(er.to_string().as_bytes()).unwrap();
                    counters_w.write_all(b"\n").unwrap();
                }
            }
        }
        batch.clear();
        (ok, err)
    };

    loop {
        buf.clear();
        let read = reader.read_line(&mut buf).expect("read");
        if read == 0 {
            break;
        }
        let line = buf.trim_end().to_string();
        if line.is_empty() {
            continue;
        }
        batch.push((idx, line));
        idx += 1;
        if batch.len() >= BATCH {
            let (o, e) = flush(&mut batch, &mut spans_w, &mut counters_w);
            n_docs += o;
            n_err += e;
        }
    }
    let (o, e) = flush(&mut batch, &mut spans_w, &mut counters_w);
    n_docs += o;
    n_err += e;
    spans_w.flush().unwrap();
    counters_w.flush().unwrap();
    eprintln!(
        "reference_detect: mode={} source={} docs={} errors={} → {} / {}",
        a.mode, a.source, n_docs, n_err, a.out_spans, a.out_counters
    );
    if n_err > 0 {
        eprintln!("reference_detect: WARN {n_err} malformed/schema-mismatched rows emitted as error records to {} (filter on the \"error\" field to audit).", a.out_counters);
        // Fail hard only on TOTAL failure (e.g. wrong field names / format) so a clean schema mismatch
        // can't masquerade as success; sporadic bad rows stay auditable (records) without aborting the
        // downstream pipeline for an entire source.
        if n_docs == 0 {
            eprintln!("reference_detect: FATAL every row failed — likely wrong --text-field/--id-field or schema. Exiting non-zero.");
            std::process::exit(2);
        }
    }
}
