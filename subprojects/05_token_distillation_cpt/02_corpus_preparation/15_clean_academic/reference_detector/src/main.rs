//! reference_detect — thin CLI runner around the detector core.
//!
//! Modes:
//!   --mode wholedoc          legacy rule-based whole-document detector
//!   --mode sections          section-labelled Kallipos/Pergamos detector
//!   --mode bib-spans|spans   promoted bibliography line head (`spans` is the compatibility alias)
//!   --mode toc-spans         table-of-contents line head
//!   --mode structure-spans   both frozen line heads
//!   --mode deterministic-structure  explainable rules + typed block decoder
//!   --mode score-lines       bibliography parity harness
//!   --mode toc-score-lines   ToC parity harness
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
use sha2::{Digest, Sha256};

use reference_detector::{
    detect_doc, detect_sections, span_line_model, structural_rules, toc_line_model, DetectConfig,
    DocResult, SectionRow, Span,
};

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

fn sha256_hex(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}

fn row_uid(source: &str, doc_id: &str) -> String {
    sha256_hex(format!("{source}\0{doc_id}").as_bytes())
}

fn merged_range_len(mut ranges: Vec<(usize, usize)>) -> usize {
    if ranges.is_empty() {
        return 0;
    }
    ranges.sort_unstable();
    let mut total = 0usize;
    let (mut start, mut end) = ranges[0];
    for (next_start, next_end) in ranges.into_iter().skip(1) {
        if next_start <= end {
            end = end.max(next_end);
        } else {
            total += end - start;
            (start, end) = (next_start, next_end);
        }
    }
    total + end - start
}

fn overlap_stats(spans: &[Span]) -> (usize, usize, usize) {
    let bib: Vec<_> = spans.iter().filter(|span| span.kind == "bib_span").collect();
    let toc: Vec<_> = spans.iter().filter(|span| span.kind == "toc_span").collect();
    let mut char_ranges = Vec::new();
    let mut line_ranges = Vec::new();
    let mut pairs = 0usize;
    for left in &bib {
        for right in &toc {
            let char_start = left.char_start.max(right.char_start);
            let char_end = left.char_end.min(right.char_end);
            if char_start < char_end {
                pairs += 1;
                char_ranges.push((char_start, char_end));
            }
            let line_start = left.line_start.max(right.line_start);
            let line_end = left.line_end.min(right.line_end);
            if line_start <= line_end {
                line_ranges.push((line_start, line_end + 1));
            }
        }
    }
    (
        pairs,
        merged_range_len(char_ranges),
        merged_range_len(line_ranges),
    )
}

/// Frozen line-head driver (rayon-batched). Malformed rows emit `{doc_idx,error}` records rather than
/// disappearing. Combined mode emits both span kinds and split counters.
fn run_span_mode(a: &Args, reader: &mut dyn BufRead, spans_w: &mut dyn Write, counters_w: &mut dyn Write) {
    const BATCH: usize = 2000;
    let mut batch: Vec<(usize, String)> = Vec::with_capacity(BATCH);
    let mut buf = String::new();
    let mut idx = 0usize;
    let mut n_ok = 0usize;
    let mut n_err = 0usize;

    let flush = |batch: &mut Vec<(usize, String)>, spans_w: &mut dyn Write, counters_w: &mut dyn Write| -> (usize, usize) {
        let out: Vec<(bool, String, Vec<String>)> = batch
            .par_iter()
            .map(|(i, l)| match serde_json::from_str::<Value>(l) {
                Err(e) => (true, serde_json::json!({"doc_idx": i, "error": format!("json parse error: {e}")}).to_string(), Vec::new()),
                Ok(v) => {
                    if a.mode == "deterministic-structure" {
                        let Some(id) = pick_str(&v, &a.id_fields).map(str::to_string) else {
                            return (true, serde_json::json!({"doc_idx": i, "error": format!("missing id field (tried {:?})", a.id_fields)}).to_string(), Vec::new());
                        };
                        let Some(text) = pick_str(&v, &a.text_fields) else {
                            return (true, serde_json::json!({"doc_idx": i, "error": format!("missing text field (tried {:?})", a.text_fields)}).to_string(), Vec::new());
                        };
                        let source = v.get("source")
                            .and_then(|value| value.as_str())
                            .filter(|value| !value.is_empty())
                            .unwrap_or(&a.source);
                        let original_sha256 = sha256_hex(text.as_bytes());
                        let original_chars = text.chars().count();
                        let uid = row_uid(source, &id);
                        let decision = structural_rules::structural_detect(
                            &id,
                            source,
                            text,
                            &structural_rules::StructuralConfig::default(),
                        );
                        let span_rows = decision.spans.iter().map(|span| {
                            serde_json::json!({
                                "schema_version": "academic-structure-deterministic-span-v1",
                                "doc_id": id,
                                "source": source,
                                "row_uid": uid,
                                "original_sha256": original_sha256,
                                "original_chars": original_chars,
                                "model_id": decision.model_id,
                                "decoder_id": decision.decoder_id,
                                "span": span,
                            }).to_string()
                        }).collect();
                        let mut decision_value = serde_json::to_value(&decision).unwrap();
                        let object = decision_value.as_object_mut().unwrap();
                        object.insert("row_uid".into(), Value::String(uid));
                        object.insert(
                            "original_sha256".into(),
                            Value::String(original_sha256),
                        );
                        object.insert("original_chars".into(), Value::from(original_chars));
                        (false, decision_value.to_string(), span_rows)
                    } else if a.mode == "score-lines" || a.mode == "bib-score-lines" || a.mode == "toc-score-lines" {
                        let present: Vec<String> = v.get("present").and_then(|x| x.as_array())
                            .map(|arr| arr.iter().filter_map(|s| s.as_str().map(str::to_string)).collect())
                            .unwrap_or_default();
                        let prefs: Vec<&str> = present.iter().map(String::as_str).collect();
                        let abs: Vec<usize> = v.get("abs_idx").and_then(|x| x.as_array())
                            .map(|arr| arr.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
                            .unwrap_or_else(|| (0..prefs.len()).collect());
                        let nt = v.get("n_total").and_then(|x| x.as_u64()).map(|u| u as usize).unwrap_or(prefs.len());
                        let p = if a.mode == "toc-score-lines" {
                            toc_line_model::toc_scores(&prefs, &abs, nt)
                        } else {
                            span_line_model::span_scores(&prefs, &abs, nt)
                        };
                        (false, serde_json::json!({"p": p}).to_string(), Vec::new())
                    } else {
                        let Some(id) = pick_str(&v, &a.id_fields).map(str::to_string) else {
                            return (true, serde_json::json!({"doc_idx": i, "error": format!("missing id field (tried {:?})", a.id_fields)}).to_string(), Vec::new());
                        };
                        match pick_str(&v, &a.text_fields) {
                            None => (true, serde_json::json!({"doc_idx": i, "error": format!("missing text field (tried {:?})", a.text_fields)}).to_string(), Vec::new()),
                            Some(text) => {
                                let mut spans = Vec::new();
                                if a.mode == "spans" || a.mode == "bib-spans" || a.mode == "structure-spans" {
                                    spans.extend(span_line_model::span_detect(&id, &a.source, text));
                                }
                                if a.mode == "toc-spans" || a.mode == "structure-spans" {
                                    spans.extend(toc_line_model::toc_detect(&id, &a.source, text));
                                }
                                spans.sort_by_key(|s| (s.char_start, s.char_end, s.kind.clone()));
                                let bib_spans = spans.iter().filter(|s| s.kind == "bib_span").count();
                                let toc_spans = spans.iter().filter(|s| s.kind == "toc_span").count();
                                let bib_lines: usize = spans.iter().filter(|s| s.kind == "bib_span")
                                    .map(|s| s.line_end - s.line_start + 1).sum();
                                let toc_lines: usize = spans.iter().filter(|s| s.kind == "toc_span")
                                    .map(|s| s.line_end - s.line_start + 1).sum();
                                let original_sha256 = sha256_hex(text.as_bytes());
                                let uid = row_uid(&a.source, &id);
                                let (overlap_pairs, overlap_chars, overlap_lines) = overlap_stats(&spans);
                                let sjs = spans.iter().map(|s| {
                                    let mut value = serde_json::to_value(s).unwrap();
                                    let object = value.as_object_mut().unwrap();
                                    object.insert("source".into(), Value::String(a.source.clone()));
                                    object.insert("row_uid".into(), Value::String(uid.clone()));
                                    object.insert("original_sha256".into(), Value::String(original_sha256.clone()));
                                    object.insert("original_chars".into(), Value::from(text.chars().count()));
                                    object.insert(
                                        "model_id".into(),
                                        Value::String(if s.kind == "toc_span" {
                                            format!("{}:{}", toc_line_model::MODEL_ID, toc_line_model::DECODER_ID)
                                        } else {
                                            format!("{}:{}", span_line_model::MODEL_ID, span_line_model::DECODER_ID)
                                        }),
                                    );
                                    value.to_string()
                                }).collect();
                                (false, serde_json::json!({
                                    "doc_id": id,
                                    "source": a.source,
                                    "row_uid": uid,
                                    "original_sha256": original_sha256,
                                    "original_chars": text.chars().count(),
                                    "bib_model_id": span_line_model::MODEL_ID,
                                    "bib_decoder_id": span_line_model::DECODER_ID,
                                    "toc_model_id": toc_line_model::MODEL_ID,
                                    "toc_decoder_id": toc_line_model::DECODER_ID,
                                    "bib_spans": bib_spans,
                                    "bib_lines": bib_lines,
                                    "toc_spans": toc_spans,
                                    "toc_lines": toc_lines,
                                    "overlap_pairs": overlap_pairs,
                                    "overlap_chars": overlap_chars,
                                    "overlap_lines": overlap_lines,
                                    "conflict_status": if overlap_pairs > 0 { "review_required" } else { "none" }
                                }).to_string(), sjs)
                            }
                        }
                    }
                }
            })
            .collect();
        let (mut ok, mut err) = (0usize, 0usize);
        for (is_err, cj, sjs) in &out {
            if *is_err { err += 1; } else { ok += 1; }
            counters_w.write_all(cj.as_bytes()).unwrap();
            counters_w.write_all(b"\n").unwrap();
            for sj in sjs {
                spans_w.write_all(sj.as_bytes()).unwrap();
                spans_w.write_all(b"\n").unwrap();
            }
        }
        batch.clear();
        (ok, err)
    };

    loop {
        buf.clear();
        if reader.read_line(&mut buf).expect("read") == 0 { break; }
        let line = buf.trim_end().to_string();
        if line.is_empty() { continue; }
        batch.push((idx, line));
        idx += 1;
        if batch.len() >= BATCH {
            let (o, e) = flush(&mut batch, spans_w, counters_w);
            n_ok += o; n_err += e;
        }
    }
    let (o, e) = flush(&mut batch, spans_w, counters_w);
    n_ok += o; n_err += e;
    spans_w.flush().unwrap();
    counters_w.flush().unwrap();
    eprintln!("reference_detect: mode={} source={} ok={} errors={} → {} / {}", a.mode, a.source, n_ok, n_err, a.out_spans, a.out_counters);
    if n_err > 0 {
        eprintln!("reference_detect: FATAL {n_err} malformed/schema-mismatched rows; production detection requires zero errors.");
        std::process::exit(2);
    }
}

fn process_line(a: &Args, _idx: usize, line: &str) -> Result<DocResult, String> {
    let v: Value = serde_json::from_str(line).map_err(|e| format!("json parse error: {e}"))?;
    if a.mode == "sections" {
        let filename = pick_str(&v, &a.id_fields)
            .map(str::to_string)
            .ok_or_else(|| format!("missing id field (tried {:?})", a.id_fields))?;
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
        let id = pick_str(&v, &a.id_fields)
            .map(str::to_string)
            .ok_or_else(|| format!("missing id field (tried {:?})", a.id_fields))?;
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

    // Frozen line-head paths stay separate from the legacy header→EOF machinery.
    if matches!(
        a.mode.as_str(),
        "spans"
            | "bib-spans"
            | "toc-spans"
            | "structure-spans"
            | "deterministic-structure"
            | "score-lines"
            | "bib-score-lines"
            | "toc-score-lines"
    ) {
        run_span_mode(&a, &mut *reader, &mut spans_w, &mut counters_w);
        return;
    }

    const BATCH: usize = 2000;
    let mut batch: Vec<(usize, String)> = Vec::with_capacity(BATCH);
    let mut idx = 0usize;
    let mut n_docs = 0usize;
    let mut n_err = 0usize;
    let mut buf = String::new();

    // Returns (n_ok, n_err). Malformed/schema-mismatched rows are NOT silently dropped: each emits an
    // auditable error record to the counters stream (`{"doc_idx","error"}`) and is counted, so a bad
    // row fails loudly (final summary + non-zero exit) instead of vanishing from the counters.
    let flush = |batch: &mut Vec<(usize, String)>, spans_w: &mut dyn Write, counters_w: &mut dyn Write| {
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
        eprintln!("reference_detect: FATAL {n_err} malformed/schema-mismatched rows; production detection requires zero errors.");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(kind: &str, char_start: usize, char_end: usize, line_start: usize, line_end: usize) -> Span {
        Span {
            doc_id: "d".into(),
            kind: kind.into(),
            char_start,
            char_end,
            line_start,
            line_end,
            trigger: "fixture".into(),
            gated_by: "fixture".into(),
            row_id: None,
        }
    }

    #[test]
    fn combined_head_overlap_is_reported_as_unique_mass() {
        let spans = vec![
            span("bib_span", 10, 30, 1, 3),
            span("bib_span", 25, 40, 3, 4),
            span("toc_span", 20, 35, 2, 3),
        ];
        let (pairs, chars, lines) = overlap_stats(&spans);
        assert_eq!(pairs, 2);
        assert_eq!(chars, 15, "overlapping conflict ranges must be unioned");
        assert_eq!(lines, 2);
    }
}
