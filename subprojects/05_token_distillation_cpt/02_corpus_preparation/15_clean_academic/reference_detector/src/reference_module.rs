//! reference_module — the Greek academic reference/citation detector.
//!
//! Contract (per the project rules + the investigation synthesis):
//!  - DETECT → SEGMENT → EMIT SPANS → COUNT per family. NEVER hard-delete, NEVER
//!    bake a deletion threshold. Every decision is a tag on an emitted span; the
//!    keep/drop policy is the caller's (a user knob downstream).
//!  - Counters are SPLIT per signal family (no weighted aggregate).
//!  - FAIL-CLOSED: when signals disagree, keep text (classify toward KEEP), so a
//!    boundary miss leaves junk in (auditable) rather than amputating body prose.
//!
//! Two entry points, one shared counter/span vocabulary:
//!  - `detect_doc`      — whole-document Markdown (greek_phd, openarchives).
//!  - `detect_sections` — section rows with a `predicted_section` label (kallipos, pergamos).

use serde::Serialize;

use crate::reference_signals as sig;

// ---------------------------------------------------------------------------
// Public config (calibration knobs — defaults = "segment-and-emit-everything",
// i.e. no deletion threshold baked in; the caller tightens these).
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct DetectConfig {
    /// A bib header must sit beyond this fraction of the doc to count as end-matter.
    pub min_position_fraction: f32,
    /// Year-density (years per bib-block line) at/above which the bib span is "confirmed".
    /// Default 0.0 → always confirmed (emit-everything). Span is emitted either way; only the tag differs.
    pub bib_min_year_density: f32,
    /// A footnote ≥ this many chars (with Greek dominance) is treated as prose/hybrid (KEPT).
    pub footnote_prose_min_chars: usize,
    /// A footnote whose Greek-letter fraction is below this is citation-only (drop candidate).
    pub footnote_cite_max_greek: f32,
    /// A front-positioned `β` section matching the CV header family below this position is kept.
    pub cv_front_max_pos: f32,
    /// Emit one span per in-text marker (large); default false → counters only.
    pub emit_intext_spans: bool,
}

impl Default for DetectConfig {
    fn default() -> Self {
        DetectConfig {
            min_position_fraction: 0.5,
            bib_min_year_density: 0.0,
            footnote_prose_min_chars: 200,
            footnote_cite_max_greek: 0.55,
            cv_front_max_pos: 0.4,
            emit_intext_spans: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Emitted records
// ---------------------------------------------------------------------------
#[derive(Serialize, Clone, Debug)]
pub struct Span {
    pub doc_id: String,
    /// endmatter_bib | footnote_citation | footnote_prose | beta_section |
    /// intext_paren_ay | intext_bracket | intext_paren_num
    pub kind: String,
    pub char_start: usize,
    pub char_end: usize,
    pub line_start: usize,
    pub line_end: usize,
    /// What triggered detection (header text, cue, latin-run, label, …).
    pub trigger: String,
    /// Gate reason / decision (e.g. bib | kept-non-bib:front_cv | year_density_below).
    pub gated_by: String,
    /// Optional row id for section sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_id: Option<String>,
}

#[derive(Serialize, Clone, Debug, Default)]
pub struct Counters {
    pub doc_id: String,
    pub source: String,
    // end-matter
    pub endmatter_header_found: bool,
    pub endmatter_header_multiplicity: u32,
    pub endmatter_bib_chars: usize,
    pub endmatter_year_density: f32,
    // footnote stream
    pub footnote_stream_lines: u32,
    pub footnote_citation_only: u32,
    pub footnote_prose_or_hybrid_kept_chars: usize,
    // separators (distinct codepoints, never summed)
    pub anoteleia_u0387_count: u32,
    pub anoteleia_u00b7_count: u32,
    // in-text families (independent)
    pub intext_paren_authoryear: u32,
    pub intext_bracket_numeric: u32,
    pub intext_paren_numeric: u32,
    pub intext_ocr_superscript_digit: u32,
    pub internal_xref_kept: u32,
    // section sources
    pub beta_sections_total: u32,
    pub beta_kept_as_non_bib: u32,
    pub beta_chars_flagged: usize,
    // routing
    pub dominant_regime: String,
}

#[derive(Default)]
pub struct DocResult {
    pub spans: Vec<Span>,
    pub counters: Counters,
}

// ---------------------------------------------------------------------------
// Line indexing with char offsets (Python-friendly char positions)
// ---------------------------------------------------------------------------
struct Line<'a> {
    idx: usize,
    char_start: usize,
    char_end: usize,
    text: &'a str,
}

fn index_lines(text: &str) -> (Vec<Line<'_>>, usize) {
    let mut lines = Vec::new();
    let mut pos = 0usize;
    for (i, raw) in text.split_inclusive('\n').enumerate() {
        let full = raw.chars().count();
        let trimmed = raw.strip_suffix('\n').unwrap_or(raw);
        let trimmed = trimmed.strip_suffix('\r').unwrap_or(trimmed);
        lines.push(Line {
            idx: i,
            char_start: pos,
            char_end: pos + trimmed.chars().count(),
            text: trimmed,
        });
        pos += full;
    }
    (lines, pos)
}

fn header_text(line: &str) -> &str {
    line.trim_start_matches(|c| c == '#' || c == ' ' || c == '\t')
}

fn is_allcaps_ish(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() || t.chars().count() > 80 {
        return false;
    }
    let mut upper = 0u32;
    let mut alpha = 0u32;
    for c in t.chars() {
        if c.is_alphabetic() {
            alpha += 1;
            if c.is_uppercase() {
                upper += 1;
            }
        }
    }
    alpha >= 3 && (upper as f32) >= 0.8 * (alpha as f32)
}

fn is_header_candidate(line: &str) -> bool {
    sig::ATX_HEADER.is_match(line) || is_allcaps_ish(line)
}

// ---------------------------------------------------------------------------
// Whole-document detection
// ---------------------------------------------------------------------------
pub fn detect_doc(doc_id: &str, source: &str, text: &str, cfg: &DetectConfig) -> DocResult {
    let (lines, total_chars) = index_lines(text);
    let total_lines = lines.len().max(1);
    let mut spans: Vec<Span> = Vec::new();
    let mut c = Counters {
        doc_id: doc_id.to_string(),
        source: source.to_string(),
        ..Default::default()
    };

    // --- (A) end-matter bibliography boundary -------------------------------
    // Take the FIRST qualifying header in the lower portion: a bibliography is often
    // sub-divided (Ελληνόγλωσση / Ξενόγλωσση / Δικτυογραφία), so the *earliest* bib-family
    // header in the end region begins the block; later ones are sub-lists subsumed by it.
    // Early false positives are already filtered by the TOC/colophon/lit-review/position guards.
    let mut first_qualifying: Option<usize> = None; // index into `lines`
    let mut multiplicity = 0u32;
    for (li, line) in lines.iter().enumerate() {
        if !is_header_candidate(line.text) {
            continue;
        }
        let folded = sig::fold_header(header_text(line.text));
        if !sig::contains_any(&folded, sig::BIB_HEADER_STEMS) {
            continue;
        }
        if sig::contains_any(&folded, sig::LITREVIEW_DENY)
            || sig::contains_any(&folded, sig::COLOPHON_DENY)
        {
            continue;
        }
        // reject TOC-context: the header line or its next 2 lines look like a TOC row
        let toc_window = (li..(li + 3).min(lines.len()))
            .any(|k| sig::TOC_CONTEXT.is_match(lines[k].text));
        if toc_window {
            continue;
        }
        // position guard: must be in the lower portion of the doc
        let pos_frac = line.char_start as f32 / total_chars.max(1) as f32;
        if pos_frac < cfg.min_position_fraction {
            continue;
        }
        multiplicity += 1;
        if first_qualifying.is_none() {
            first_qualifying = Some(li);
        }
    }

    let mut bib_start_char = total_chars; // default: no bib → body is whole doc
    if let Some(li) = first_qualifying {
        let header_line = &lines[li];
        bib_start_char = header_line.char_start;
        let block_lines: Vec<&Line> = lines.iter().filter(|l| l.idx >= li).collect();
        let block_text: String = block_lines.iter().map(|l| l.text).collect::<Vec<_>>().join("\n");
        let year_count = sig::YEAR.find_iter(&block_text).count() as f32;
        let year_density = year_count / (block_lines.len().max(1) as f32);
        let confirmed = year_density >= cfg.bib_min_year_density;
        c.endmatter_header_found = true;
        c.endmatter_header_multiplicity = multiplicity;
        c.endmatter_bib_chars = total_chars.saturating_sub(bib_start_char);
        c.endmatter_year_density = year_density;
        spans.push(Span {
            doc_id: doc_id.to_string(),
            kind: "endmatter_bib".to_string(),
            char_start: bib_start_char,
            char_end: total_chars,
            line_start: li,
            line_end: total_lines - 1,
            trigger: header_text(header_line.text).chars().take(60).collect(),
            gated_by: if confirmed { "year_density_ok".into() } else { "year_density_below".into() },
            row_id: None,
        });
    }

    // --- (B) footnote stream (body only, i.e. before bib_start) -------------
    let mut i = 0usize;
    while i < lines.len() {
        let line = &lines[i];
        if line.char_start >= bib_start_char {
            break;
        }
        let is_fn = sig::FOOTNOTE_LINE.is_match(line.text) && footnote_citation_signal(line.text);
        if !is_fn {
            i += 1;
            continue;
        }
        // group continuation lines until next footnote/blank/HR/header
        let start = i;
        let mut j = i + 1;
        while j < lines.len() {
            let lt = lines[j].text;
            if lt.trim().is_empty()
                || lt.trim_start().starts_with("---")
                || sig::FOOTNOTE_LINE.is_match(lt)
                || sig::ATX_HEADER.is_match(lt)
                || lines[j].char_start >= bib_start_char
            {
                break;
            }
            j += 1;
        }
        let span_text: String = lines[start..j].iter().map(|l| l.text).collect::<Vec<_>>().join("\n");
        let chars = span_text.chars().count();
        let greek = sig::greek_letter_fraction(&span_text);
        let cue = sig::CUE.is_match(&sig::fold(&span_text));
        let has_year = sig::YEAR.is_match(&span_text);
        let latin_run = sig::LATIN_AUTHOR_RUN.is_match(&span_text);
        // FAIL-CLOSED: only call citation_only when clearly so; else keep as prose.
        // A long, Greek-dominant footnote is prose/hybrid (KEPT) per cfg.footnote_prose_min_chars —
        // this guard makes the knob live and overrides the citation heuristics below.
        let long_greek_prose =
            chars >= cfg.footnote_prose_min_chars && greek >= cfg.footnote_cite_max_greek;
        let citation_only = !long_greek_prose
            && (greek < cfg.footnote_cite_max_greek
                || (latin_run && has_year && !cue && greek < 0.75)
                || (chars < 120 && has_year && !cue));
        c.footnote_stream_lines += 1;
        let (kind, gated) = if citation_only {
            c.footnote_citation_only += 1;
            ("footnote_citation", "citation_only")
        } else {
            c.footnote_prose_or_hybrid_kept_chars += chars;
            ("footnote_prose", "prose_or_hybrid_keep")
        };
        spans.push(Span {
            doc_id: doc_id.to_string(),
            kind: kind.to_string(),
            char_start: lines[start].char_start,
            char_end: lines[j - 1].char_end,
            line_start: start,
            line_end: j - 1,
            trigger: line.text.chars().take(40).collect(),
            gated_by: gated.to_string(),
            row_id: None,
        });
        i = j;
    }

    // --- (C) in-text family counters (body only) ---------------------------
    let body: String = lines
        .iter()
        .filter(|l| l.char_start < bib_start_char)
        .map(|l| l.text)
        .collect::<Vec<_>>()
        .join("\n");
    count_intext(doc_id, &body, bib_start_char, &lines, cfg, &mut c, &mut spans);

    // separators over whole doc
    let (a87, a_b7) = sig::anoteleia_counts(text);
    c.anoteleia_u0387_count = a87;
    c.anoteleia_u00b7_count = a_b7;

    c.dominant_regime = pick_regime(&c);
    DocResult { spans, counters: c }
}

fn footnote_citation_signal(line: &str) -> bool {
    let folded = sig::fold(line);
    let has_year = sig::YEAR.is_match(line);
    sig::CUE.is_match(&folded)
        || line.contains(sig::ANO_TELEIA)
        || line.contains(sig::MIDDLE_DOT)
        || sig::LATIN_AUTHOR_RUN.is_match(line)
        || (has_year && sig::PAGE_RANGE.is_match(line))
        || (has_year && sig::latin_letter_fraction(line) > 0.30)
}

fn count_intext(
    doc_id: &str,
    body: &str,
    bib_start_char: usize,
    lines: &[Line],
    cfg: &DetectConfig,
    c: &mut Counters,
    spans: &mut Vec<Span>,
) {
    c.intext_paren_authoryear = sig::INTEXT_PAREN_AY.find_iter(body).count() as u32;
    c.intext_bracket_numeric = sig::INTEXT_BRACKET.find_iter(body).count() as u32;
    c.intext_paren_numeric = sig::INTEXT_PAREN_NUM.find_iter(body).count() as u32;
    c.internal_xref_kept = sig::INTERNAL_XREF_KEEP.find_iter(&sig::fold(body)).count() as u32;
    // OCR'd superscript: a Greek letter glued to 1-2 digits then a separator (never stripped).
    let mut sup = 0u32;
    let chars: Vec<char> = body.chars().collect();
    let n = chars.len();
    let mut k = 0usize;
    while k + 2 < n {
        let a = chars[k];
        if (('\u{0370}'..='\u{03FF}').contains(&a)) && chars[k + 1].is_ascii_digit() {
            // up to 2 digits then space/punct
            let mut d = k + 1;
            while d < n && chars[d].is_ascii_digit() && d - k <= 2 {
                d += 1;
            }
            if d < n && matches!(chars[d], ' ' | ',' | '.' | ';' | '·' | '\u{0387}') {
                sup += 1;
            }
            k = d;
        } else {
            k += 1;
        }
    }
    c.intext_ocr_superscript_digit = sup;

    if cfg.emit_intext_spans {
        // best-effort: locate paren-AY spans by char offset within the body region
        let _ = (bib_start_char, lines); // body offsets align with doc offsets (body is a prefix)
        for m in sig::INTEXT_PAREN_AY.find_iter(body) {
            let cs = body[..m.start()].chars().count();
            let ce = body[..m.end()].chars().count();
            spans.push(Span {
                doc_id: doc_id.to_string(),
                kind: "intext_paren_ay".to_string(),
                char_start: cs,
                char_end: ce,
                line_start: 0,
                line_end: 0,
                trigger: m.as_str().chars().take(40).collect(),
                gated_by: "keep_and_count".to_string(),
                row_id: None,
            });
        }
    }
}

fn pick_regime(c: &Counters) -> String {
    let cands = [
        ("footnote_stream", c.footnote_stream_lines),
        ("intext_paren_authoryear", c.intext_paren_authoryear),
        ("intext_bracket_numeric", c.intext_bracket_numeric),
        ("intext_paren_numeric", c.intext_paren_numeric),
    ];
    let best = cands.iter().max_by_key(|(_, v)| *v).unwrap();
    if best.1 == 0 {
        if c.endmatter_header_found {
            "endmatter_only".to_string()
        } else {
            "none".to_string()
        }
    } else {
        best.0.to_string()
    }
}

// ---------------------------------------------------------------------------
// Section-level detection (kallipos / pergamos: predicted_section == "β")
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct SectionRow {
    pub row_id: String,
    pub header: String,
    pub section: String,
    pub predicted_section: String,
    pub positional_fraction: f32,
}

pub fn detect_sections(doc_id: &str, source: &str, rows: &[SectionRow], cfg: &DetectConfig) -> DocResult {
    let mut spans = Vec::new();
    let mut c = Counters {
        doc_id: doc_id.to_string(),
        source: source.to_string(),
        ..Default::default()
    };
    // doc-level neighbour signal for the β-gate model: positions of all β sections in this doc.
    let beta_positions: Vec<f32> = rows
        .iter()
        .filter(|r| r.predicted_section.trim() == "β")
        .map(|r| r.positional_fraction)
        .collect();
    let mut cum_chars = 0usize;
    for row in rows {
        let seclen = row.section.chars().count();
        // in-text counters across all body text
        c.intext_paren_authoryear += sig::INTEXT_PAREN_AY.find_iter(&row.section).count() as u32;
        c.intext_bracket_numeric += sig::INTEXT_BRACKET.find_iter(&row.section).count() as u32;
        c.intext_paren_numeric += sig::INTEXT_PAREN_NUM.find_iter(&row.section).count() as u32;
        c.internal_xref_kept += sig::INTERNAL_XREF_KEEP.find_iter(&sig::fold(&row.section)).count() as u32;
        let (a87, ab7) = sig::anoteleia_counts(&row.section);
        c.anoteleia_u0387_count += a87;
        c.anoteleia_u00b7_count += ab7;

        if row.predicted_section.trim() == "β" {
            c.beta_sections_total += 1;
            let folded_h = sig::fold_header(&row.header);
            let year_count = sig::YEAR.find_iter(&row.section).count() as u32;
            let line_count = row.section.lines().count().max(1) as f32;
            let year_density = year_count as f32 / line_count;
            let latin_frac = sig::latin_letter_fraction(&row.section);
            let dash_lines = row
                .section
                .lines()
                .filter(|l| {
                    let t = l.trim_start();
                    t.starts_with("- ") || t.starts_with("• ") || t.starts_with("* ")
                })
                .count() as f32;
            let dashlist_density = dash_lines / line_count;

            // Deployed β-gate: the 7-feature logistic regression (eval/beta_gate_model.json),
            // with hard keep-guards for colophon + textbook apparatus (C1: LR + hard guards).
            // `cfg.cv_front_max_pos` retained for API compat; the model now owns the position signal.
            let _ = cfg.cv_front_max_pos;
            let lr_f = crate::beta_gate_model::features(
                &row.header,
                &row.section,
                row.positional_fraction,
                &beta_positions,
            );
            let (prob, lr_bib) = crate::beta_gate_model::score(&lr_f);
            let (decision, reason): (&str, String) =
                if sig::contains_any(&folded_h, sig::KALLIPOS_FP_DENY) {
                    ("kept-non-bib", "guard:fp_deny".into())
                } else if sig::contains_any(&folded_h, sig::COLOPHON_DENY) {
                    ("kept-non-bib", "guard:colophon".into())
                } else if lr_bib {
                    ("bib", format!("lr:{:.2}", prob))
                } else {
                    ("kept-non-bib", format!("lr:{:.2}", prob))
                };

            if decision == "kept-non-bib" {
                c.beta_kept_as_non_bib += 1;
            } else {
                c.beta_chars_flagged += seclen;
            }
            spans.push(Span {
                doc_id: doc_id.to_string(),
                kind: "beta_section".to_string(),
                char_start: cum_chars,
                char_end: cum_chars + seclen,
                line_start: 0,
                line_end: 0,
                trigger: format!(
                    "{} | yc={} yd={:.2} latin={:.2} dash={:.2} pos={:.2}",
                    row.header.chars().take(40).collect::<String>(),
                    year_count,
                    year_density,
                    latin_frac,
                    dashlist_density,
                    row.positional_fraction
                ),
                gated_by: format!("{}:{}", decision, reason),
                row_id: Some(row.row_id.clone()),
            });
        }
        cum_chars += seclen + 1;
    }
    c.dominant_regime = pick_regime(&c);
    DocResult { spans, counters: c }
}

// ---------------------------------------------------------------------------
// Tests — anchored on the verified investigation findings
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_accents_and_homoglyphs() {
        assert!(sig::fold("ΒΙΒΛΙΟΓΡΑΦΙΑ").contains("βιβλιογραφ"));
        assert!(sig::fold_header("Bιβλιογραφία").contains("βιβλιογραφ")); // Latin capital B
        assert!(sig::fold("Πηγές").contains("πηγ"));
    }

    #[test]
    fn endmatter_boundary_takes_last_lower_header() {
        let doc = "intro line\n".to_string()
            + &"σώμα κειμένου με αρκετό περιεχόμενο εδώ.\n".repeat(40)
            + "## ΒΙΒΛΙΟΓΡΑΦΙΑ\n"
            + "Παπαδόπουλος, Α. (1999). Τίτλος. Αθήνα: Εκδόσεις.\n"
            + "Smith, J. (2008). A title. London: Press.\n";
        let r = detect_doc("d1", "greek_phd", &doc, &DetectConfig::default());
        assert!(r.counters.endmatter_header_found, "should find bib header");
        assert!(r.spans.iter().any(|s| s.kind == "endmatter_bib"));
        assert!(r.counters.endmatter_bib_chars > 0);
    }

    #[test]
    fn rejects_toc_row_as_boundary() {
        // A '## Βιβλιογραφία' that is a TOC row (followed by page numbers) must NOT segment.
        let doc = "## Βιβλιογραφία\n(σελ. 1029-1163) .......... 1029\n".to_string()
            + &"πραγματικό σώμα κειμένου εδώ.\n".repeat(50);
        let r = detect_doc("d2", "greek_phd", &doc, &DetectConfig::default());
        assert!(!r.counters.endmatter_header_found, "TOC row must be rejected");
    }

    #[test]
    fn footnote_stream_splits_citation_vs_prose() {
        let doc = "κύριο σώμα.\n".to_string()
            + "12 Smith, J. A title, London: Press, 2007, 23-59.\n"
            + "13 Όπως παρατηρεί ο συγγραφέας, η ελληνική ιστοριογραφία αναπτύχθηκε σημαντικά κατά τη διάρκεια του εικοστού αιώνα, με αρκετές μελέτες να εστιάζουν σε αυτό το ζήτημα διεξοδικά και αναλυτικά. Βλ. σχετικά τη συζήτηση παραπάνω.\n";
        let r = detect_doc("d3", "greek_phd", &doc, &DetectConfig::default());
        assert!(r.counters.footnote_stream_lines >= 2);
        assert!(r.counters.footnote_citation_only >= 1, "latin+year footnote = citation_only");
        assert!(r.counters.footnote_prose_or_hybrid_kept_chars > 0, "greek footnote kept as prose");
    }

    #[test]
    fn anoteleia_counted_by_codepoint() {
        let doc = "11 A, 2001 \u{0387} B, 2002 \u{0387} C, 2003\n"; // explicit GREEK ANO TELEIA
        let r = detect_doc("d4", "greek_phd", doc, &DetectConfig::default());
        assert!(r.counters.anoteleia_u0387_count >= 2);
        assert_eq!(r.counters.anoteleia_u00b7_count, 0);
    }

    #[test]
    fn beta_keeps_cv_and_colophon_drops_real_bib() {
        let rows = vec![
            SectionRow { row_id: "r1".into(), header: "Βιβλιογραφική αναφορά".into(),
                section: "Γεωργούλης, Ε. 2024. Τίτλος. Κάλλιπος.".into(),
                predicted_section: "β".into(), positional_fraction: 0.01 },
            SectionRow { row_id: "r2".into(), header: "Μεταπτυχιακή εκπαίδευση".into(),
                section: "2009 Ειδικευόμενος στο τμήμα ... Παρίσι\n2010 ...".into(),
                predicted_section: "β".into(), positional_fraction: 0.05 },
            SectionRow { row_id: "r3".into(), header: "ΒΙΒΛΙΟΓΡΑΦΙΑ".into(),
                section: "- Smith, J. 2008. A title. London: Press.\n- Jones, K. 2010. Another. NY.".into(),
                predicted_section: "β".into(), positional_fraction: 0.98 },
        ];
        let r = detect_sections("p1", "pergamos", &rows, &DetectConfig::default());
        assert_eq!(r.counters.beta_sections_total, 3);
        assert!(r.counters.beta_kept_as_non_bib >= 2, "colophon + CV kept");
        // r3 is a real end bibliography → flagged
        assert!(r.spans.iter().any(|s| s.gated_by.starts_with("bib")));
    }

    #[test]
    fn internal_xref_not_treated_as_citation() {
        let doc = "Όπως δείξαμε, βλ. Θεώρημα 1.28 για λεπτομέρειες.\n";
        let r = detect_doc("d5", "kallipos", doc, &DetectConfig::default());
        assert!(r.counters.internal_xref_kept >= 1);
    }
}
