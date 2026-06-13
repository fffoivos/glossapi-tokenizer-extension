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

fn process_line(a: &Args, idx: usize, line: &str) -> Option<DocResult> {
    let v: Value = serde_json::from_str(line).ok()?;
    if a.mode == "sections" {
        let filename = pick_str(&v, &a.id_fields).map(|s| s.to_string()).unwrap_or_else(|| format!("doc{idx}"));
        let rows_v = v.get("rows")?.as_array()?;
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
        Some(detect_sections(&filename, &a.source, &rows, &a.cfg))
    } else {
        let id = pick_str(&v, &a.id_fields).map(|s| s.to_string()).unwrap_or_else(|| format!("doc{idx}"));
        let text = pick_str(&v, &a.text_fields)?;
        Some(detect_doc(&id, &a.source, text, &a.cfg))
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
    let mut buf = String::new();

    let mut flush = |batch: &mut Vec<(usize, String)>, spans_w: &mut dyn Write, counters_w: &mut dyn Write| {
        let results: Vec<DocResult> = batch
            .par_iter()
            .filter_map(|(i, l)| process_line(&a, *i, l))
            .collect();
        for r in &results {
            let cj = serde_json::to_string(&r.counters).unwrap();
            counters_w.write_all(cj.as_bytes()).unwrap();
            counters_w.write_all(b"\n").unwrap();
            for s in &r.spans {
                let sj = serde_json::to_string(s).unwrap();
                spans_w.write_all(sj.as_bytes()).unwrap();
                spans_w.write_all(b"\n").unwrap();
            }
        }
        batch.clear();
        results.len()
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
            n_docs += flush(&mut batch, &mut spans_w, &mut counters_w);
        }
    }
    n_docs += flush(&mut batch, &mut spans_w, &mut counters_w);
    spans_w.flush().unwrap();
    counters_w.flush().unwrap();
    eprintln!("reference_detect: mode={} source={} docs={} → {} / {}", a.mode, a.source, n_docs, a.out_spans, a.out_counters);
}
