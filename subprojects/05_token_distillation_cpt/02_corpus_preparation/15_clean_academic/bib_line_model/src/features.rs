//! Deterministic per-line features — the 35 counts the line model consumes.
//!
//! Port of `eval/sequence_models/bibliography_v2.py::_feature_spans`. The order of
//! `FEATURE_NAMES` is load-bearing: it is the column order of the feature matrix,
//! so it must match `bibliography_feature_explorer.FEATURE_SPECS` exactly.
//!
//! Two properties of the Python original the port has to keep:
//!
//! * **Span ownership.** Detectors do not count independently; a span claimed by a
//!   stronger detector is removed from the weaker ones (`_without_overlaps`). So a
//!   year inside an access-date is not also a standalone year. The arbitration is
//!   order-dependent — several steps read results a previous step already rewrote —
//!   so `feature_spans` below follows the Python statement order literally rather
//!   than being reorganised into something tidier.
//! * **Counts are span counts**, not match counts.
//!
//! Offsets here are **byte** offsets where Python's are character offsets. Every use
//! is either an overlap test, a comparison, or a slice at a boundary the regex engine
//! produced, and all three are order-isomorphic between the two spaces, so the
//! arbitration results are identical without paying for a char-index map per line.
//!
//! Progress is measured by `tests/fixture_parity.rs` against values generated from
//! the Python extractor.

use crate::patterns::PATTERNS;
use fancy_regex::Regex;

pub type Span = (usize, usize);

/// Column order of the count block. Must equal `FEATURE_SPECS` in
/// `bibliography_feature_explorer.py`.
pub const FEATURE_NAMES: [&str; 35] = [
    "year_count",
    "no_date_count",
    "numeric_date_count",
    "month_date_count",
    "access_date_count",
    "url_count",
    "doi_count",
    "isbn_count",
    "issn_count",
    "initial_count",
    "proper_name_word_count",
    "inverted_author_count",
    "name_initial_pair_count",
    "direct_author_count",
    "ampersand_count",
    "numbered_entry_count",
    "quoted_span_count",
    "editor_term_count",
    "thesis_term_count",
    "in_container_count",
    "edition_term_count",
    "dotted_word_count",
    "dotted_sequence_count",
    "volume_marker_count",
    "volume_shape_count",
    "journal_year_volume_count",
    "page_marker_count",
    "article_page_range_count",
    "page_range_count",
    "publisher_term_count",
    "place_name_count",
    "place_publisher_shape_count",
    "punctuation_count",
    "prose_lead_count",
    "table_row_count",
];

pub const N_FEATURES: usize = FEATURE_NAMES.len();

// Positional constants so the hot path never does a string lookup. Kept adjacent to
// FEATURE_NAMES; `indices_match_names` below asserts they agree.
pub const F_YEAR: usize = 0;
pub const F_NO_DATE: usize = 1;
pub const F_NUMERIC_DATE: usize = 2;
pub const F_MONTH_DATE: usize = 3;
pub const F_ACCESS_DATE: usize = 4;
pub const F_URL: usize = 5;
pub const F_DOI: usize = 6;
pub const F_ISBN: usize = 7;
pub const F_ISSN: usize = 8;
pub const F_INITIAL: usize = 9;
pub const F_PROPER_WORD: usize = 10;
pub const F_INVERTED_AUTHOR: usize = 11;
pub const F_NAME_INITIAL_PAIR: usize = 12;
pub const F_DIRECT_AUTHOR: usize = 13;
pub const F_AMPERSAND: usize = 14;
pub const F_NUMBERED_ENTRY: usize = 15;
pub const F_QUOTED: usize = 16;
pub const F_EDITOR_TERM: usize = 17;
pub const F_THESIS_TERM: usize = 18;
pub const F_IN_CONTAINER: usize = 19;
pub const F_EDITION_TERM: usize = 20;
pub const F_DOTTED_WORD: usize = 21;
pub const F_DOTTED_SEQUENCE: usize = 22;
pub const F_VOLUME_MARKER: usize = 23;
pub const F_VOLUME_SHAPE: usize = 24;
pub const F_JOURNAL_YEAR_VOLUME: usize = 25;
pub const F_PAGE_MARKER: usize = 26;
pub const F_ARTICLE_PAGE_RANGE: usize = 27;
pub const F_PAGE_RANGE: usize = 28;
pub const F_PUBLISHER_TERM: usize = 29;
pub const F_PLACE_NAME: usize = 30;
pub const F_PLACE_PUBLISHER_SHAPE: usize = 31;
pub const F_PUNCTUATION: usize = 32;
pub const F_PROSE_LEAD: usize = 33;
pub const F_TABLE_ROW: usize = 34;

/// `set('.,;:()[]«»“”"')` in `_feature_spans`.
const PUNCTUATION: [char; 13] = [
    '.', ',', ';', ':', '(', ')', '[', ']', '«', '»', '\u{201c}', '\u{201d}', '"',
];

// ---------------------------------------------------------------------------
// span helpers — direct ports
// ---------------------------------------------------------------------------

#[inline]
fn overlaps(left: Span, right: Span) -> bool {
    left.0 < right.1 && right.0 < left.1
}

/// `_without_overlaps` — keep only spans not already owned by a more specific
/// detector.
fn without_overlaps(spans: &[Span], blockers: &[Span]) -> Vec<Span> {
    spans
        .iter()
        .copied()
        .filter(|span| !blockers.iter().any(|b| overlaps(*span, *b)))
        .collect()
}

fn spans_of(re: &Regex, text: &str, offset: usize) -> Vec<Span> {
    let mut out = Vec::new();
    for m in re.find_iter(text).flatten() {
        out.push((offset + m.start(), offset + m.end()));
    }
    out
}

/// `_pattern_spans(..., group=name)` — the span of a named group, not the match.
fn spans_of_group(re: &Regex, text: &str, offset: usize, group: &str) -> Vec<Span> {
    let mut out = Vec::new();
    for caps in re.captures_iter(text).flatten() {
        if let Some(m) = caps.name(group) {
            out.push((offset + m.start(), offset + m.end()));
        }
    }
    out
}

/// `_dotted_sequences` — group adjacent residual dotted words into runs of >= 2.
/// A run breaks when there is non-whitespace text between two consecutive spans.
fn dotted_sequences(value: &str, dotted_words: &[Span]) -> Vec<Span> {
    let mut sequences = Vec::new();
    let mut run: Vec<Span> = Vec::new();
    for span in dotted_words.iter().copied() {
        if let Some(last) = run.last() {
            if !value[last.1..span.0].trim().is_empty() {
                if run.len() >= 2 {
                    sequences.push((run[0].0, run[run.len() - 1].1));
                }
                run.clear();
            }
        }
        run.push(span);
    }
    if run.len() >= 2 {
        sequences.push((run[0].0, run[run.len() - 1].1));
    }
    sequences
}

/// Python `str.isspace()`. Rust's `char::is_whitespace` is Unicode White_Space,
/// which omits the four ASCII separator controls Python also treats as space.
#[inline]
fn py_isspace(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '\u{1c}'..='\u{1f}')
}

/// Python `str.isdigit()`. Approximated by `is_numeric()`, which additionally
/// accepts Nl (e.g. Roman numeral codepoints); no such character appears in the
/// corpus, and the fixture set is the check on that.
#[inline]
fn py_isdigit(ch: char) -> bool {
    ch.is_ascii_digit() || ch.is_numeric()
}

#[inline]
fn py_isalnum(ch: char) -> bool {
    ch.is_alphanumeric()
}

/// `_analysis_bounds` — the author detectors run inside the pipes of a table row
/// rather than over the row markup.
fn analysis_bounds(value: &str) -> (usize, usize) {
    let mut start = value.len() - value.trim_start_matches(py_isspace).len();
    let mut end = value.trim_end_matches(py_isspace).len();
    if start < end {
        let first = value[start..end].chars().next();
        let last = value[start..end].chars().next_back();
        if first == Some('|') && last == Some('|') {
            start += 1; // '|' is one byte
            end -= 1;
            while start < end {
                let ch = value[start..end].chars().next().unwrap();
                if !py_isspace(ch) {
                    break;
                }
                start += ch.len_utf8();
            }
            while end > start {
                let ch = value[start..end].chars().next_back().unwrap();
                if !py_isspace(ch) {
                    break;
                }
                end -= ch.len_utf8();
            }
        }
    }
    (start, end)
}

/// `_numbered_entry_span` — a decorated leading number, found by strict linear scan.
/// Walks characters (not bytes) because the four-digit-year guard counts characters.
fn numbered_entry_span(text: &str) -> Option<Span> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte_at = |i: usize| if i < n { chars[i].0 } else { text.len() };

    let mut index = 0usize;
    while index < n && py_isspace(chars[index].1) {
        index += 1;
    }
    let start = index;
    while index < n && !py_isalnum(chars[index].1) {
        index += 1;
    }
    if index >= n || !py_isdigit(chars[index].1) {
        return None;
    }
    let digits_start = index;
    while index < n && py_isdigit(chars[index].1) {
        index += 1;
    }
    if index + 1 < n && matches!(chars[index].1, '.' | ',') && py_isdigit(chars[index + 1].1) {
        return None;
    }
    // A leading four-digit year is publication/date evidence, not a list index.
    if index - digits_start == 4 {
        if let Ok(v) = text[byte_at(digits_start)..byte_at(index)].parse::<u32>() {
            if (1500..=2099).contains(&v) {
                return None;
            }
        }
    }
    while index < n && !py_isalnum(chars[index].1) {
        index += 1;
    }
    Some((byte_at(start), byte_at(index)))
}

/// `_AUTHOR_PREFIX.fullmatch` — the pattern is `^…$` with only optional groups, so a
/// leftmost match from position 0 spans the whole string exactly when fullmatch does.
fn author_prefix_fullmatch(text: &str) -> bool {
    match PATTERNS.get("_AUTHOR_PREFIX").find(text) {
        Ok(Some(m)) => m.start() == 0 && m.end() == text.len(),
        _ => false,
    }
}

/// Python slicing clamps an inverted range to the empty string; Rust panics on it.
/// This matters for whitespace-only lines, where `_analysis_bounds` legitimately
/// returns `start > end` (lstrip consumes everything, rstrip leaves nothing).
#[inline]
fn py_slice(value: &str, start: usize, end: usize) -> &str {
    if start >= end {
        ""
    } else {
        &value[start..end]
    }
}

fn joined(spans: &[Vec<Span>], names: &[usize]) -> Vec<Span> {
    let mut out = Vec::new();
    for &name in names {
        out.extend_from_slice(&spans[name]);
    }
    out
}

// ---------------------------------------------------------------------------
// the port
// ---------------------------------------------------------------------------

/// `_feature_spans` — the arbitration, in Python's statement order.
pub fn feature_spans(value: &str) -> Vec<Vec<Span>> {
    let p = &*PATTERNS;
    let (start, end) = analysis_bounds(value);
    let analysis_value = py_slice(value, start, end);

    let mut spans: Vec<Vec<Span>> = vec![Vec::new(); N_FEATURES];

    // The broad pattern sweep. Note these all run over the *full* line, not the
    // analysis window — only the three author detectors use the window.
    for (idx, name) in [
        (F_YEAR, "_YEAR"),
        (F_NO_DATE, "_NO_DATE"),
        (F_NUMERIC_DATE, "_NUMERIC_DATE"),
        (F_MONTH_DATE, "_MONTH_DATE"),
        (F_ACCESS_DATE, "_ACCESS_DATE"),
        (F_URL, "_URL"),
        (F_DOI, "_DOI"),
        (F_ISBN, "_ISBN"),
        (F_ISSN, "_ISSN"),
        (F_INITIAL, "_INITIAL"),
        (F_PROPER_WORD, "_PROPER_WORD"),
        (F_AMPERSAND, "_AMPERSAND"),
        (F_QUOTED, "_QUOTED"),
        (F_EDITOR_TERM, "_EDITOR_TERMS"),
        (F_THESIS_TERM, "_THESIS_TERMS"),
        (F_EDITION_TERM, "_EDITION_TERMS"),
        (F_DOTTED_WORD, "_DOTTED_WORD"),
        (F_VOLUME_MARKER, "_VOLUME_MARKER"),
        (F_PAGE_MARKER, "_PAGE_MARKER"),
        (F_ARTICLE_PAGE_RANGE, "_ARTICLE_PAGE_RANGE"),
        (F_PUBLISHER_TERM, "_PUBLISHER_TERMS"),
        (F_PLACE_NAME, "_PLACE_NAMES"),
        (F_PLACE_PUBLISHER_SHAPE, "_PLACE_PUBLISHER_SHAPE"),
        (F_PROSE_LEAD, "_PROSE_LEAD"),
    ] {
        spans[idx] = spans_of(p.get(name), value, 0);
    }
    // volume_shape and journal_year_volume are rebuilt below with their guards;
    // in_container needs the group span rather than the match span.
    spans[F_PROSE_LEAD].truncate(1);
    spans[F_IN_CONTAINER] = spans_of_group(p.get("_IN_CONTAINER"), value, 0, "container_term");

    spans[F_INVERTED_AUTHOR] = spans_of(p.get("_INVERTED_AUTHOR"), analysis_value, start);
    spans[F_NAME_INITIAL_PAIR] = spans_of(p.get("_NAME_INITIAL_PAIR"), analysis_value, start);
    spans[F_DIRECT_AUTHOR] = spans_of(p.get("_DIRECT_AUTHOR"), analysis_value, start);

    let author_noise = joined(
        &spans,
        &[
            F_URL,
            F_DOI,
            F_ISBN,
            F_ISSN,
            F_NO_DATE,
            F_NUMERIC_DATE,
            F_MONTH_DATE,
            F_ARTICLE_PAGE_RANGE,
            F_QUOTED,
            F_EDITOR_TERM,
            F_THESIS_TERM,
            F_EDITION_TERM,
            F_VOLUME_MARKER,
            F_PAGE_MARKER,
        ],
    );
    spans[F_INVERTED_AUTHOR] = without_overlaps(&spans[F_INVERTED_AUTHOR], &author_noise);
    spans[F_DIRECT_AUTHOR] = without_overlaps(&spans[F_DIRECT_AUTHOR], &author_noise);
    for orientation in [F_INVERTED_AUTHOR, F_DIRECT_AUTHOR] {
        if let Some(first) = spans[orientation].first().copied() {
            if !author_prefix_fullmatch(py_slice(value, start, first.0)) {
                spans[orientation].clear();
            }
        }
    }

    // Comma boundaries make a direct list look locally like an inverted author.
    // Evaluate both orientations, keep the hypothesis explaining more authors;
    // coverage then earliest occurrence break ties.
    if !spans[F_INVERTED_AUTHOR].is_empty() && !spans[F_DIRECT_AUTHOR].is_empty() {
        let key = |v: &[Span]| -> (usize, usize, isize) {
            (
                v.len(),
                v.iter().map(|(a, b)| b - a).sum(),
                -(v[0].0 as isize),
            )
        };
        if key(&spans[F_DIRECT_AUTHOR]) > key(&spans[F_INVERTED_AUTHOR]) {
            spans[F_INVERTED_AUTHOR].clear();
        } else {
            spans[F_DIRECT_AUTHOR].clear();
        }
    }

    spans[F_NUMBERED_ENTRY] = match numbered_entry_span(analysis_value) {
        Some((a, b)) => vec![(start + a, start + b)],
        None => Vec::new(),
    };

    // Page ranges exclude identifiers and full dates. When both endpoints look like
    // years, keep the range as pages only if another publication year sits outside it.
    let page_blockers = joined(
        &spans,
        &[
            F_URL,
            F_DOI,
            F_ISBN,
            F_ISSN,
            F_NUMERIC_DATE,
            F_MONTH_DATE,
            F_ARTICLE_PAGE_RANGE,
        ],
    );
    let raw_years = spans[F_YEAR].clone();
    let mut page_ranges: Vec<Span> = Vec::new();
    for caps in p.get("_PAGE_RANGE").captures_iter(value).flatten() {
        let whole = caps.get(0).unwrap();
        let span = (whole.start(), whole.end());
        if page_blockers.iter().any(|b| overlaps(span, *b)) {
            continue;
        }
        let page_start: i64 = caps.name("page_start").unwrap().as_str().parse().unwrap_or(-1);
        let page_end: i64 = caps.name("page_end").unwrap().as_str().parse().unwrap_or(-1);
        let both_yearlike =
            (1500..=2099).contains(&page_start) && (1500..=2099).contains(&page_end);
        let outside_year = raw_years.iter().any(|y| !overlaps(span, *y));
        if both_yearlike && !outside_year {
            continue;
        }
        page_ranges.push(span);
    }
    spans[F_PAGE_RANGE] = page_ranges.clone();

    // A year followed by the first endpoint of a page range is not a year-volume pair.
    let mut journal_spans: Vec<Span> = Vec::new();
    for caps in p
        .get("_JOURNAL_YEAR_VOLUME")
        .captures_iter(value)
        .flatten()
    {
        let whole = caps.get(0).unwrap();
        let span = (whole.start(), whole.end());
        let volume = caps.name("journal_volume").unwrap();
        let volume_span = (volume.start(), volume.end());
        let overlaps_page_range = page_ranges.iter().any(|r| overlaps(span, *r));
        let volume_is_page_start = page_ranges.iter().any(|r| overlaps(volume_span, *r));
        if overlaps_page_range
            && (volume_is_page_start || caps.name("journal_issue").is_none())
        {
            continue;
        }
        journal_spans.push(span);
    }
    spans[F_JOURNAL_YEAR_VOLUME] = journal_spans;

    // Issue numbers cannot themselves be four-digit publication years.
    spans[F_VOLUME_SHAPE] = p
        .get("_VOLUME_SHAPE")
        .captures_iter(value)
        .flatten()
        .filter(|caps| {
            let issue: i64 = caps.name("issue").unwrap().as_str().parse().unwrap_or(-1);
            !(1500..=2099).contains(&issue)
        })
        .map(|caps| {
            let m = caps.get(0).unwrap();
            (m.start(), m.end())
        })
        .collect();

    // More-specific ownership rules — the broad detectors become fallback evidence
    // rather than repeated points for one textual event.
    spans[F_URL] = without_overlaps(&spans[F_URL], &spans[F_DOI].clone());
    spans[F_YEAR] = without_overlaps(
        &spans[F_YEAR],
        &joined(
            &spans,
            &[
                F_NUMERIC_DATE,
                F_MONTH_DATE,
                F_URL,
                F_DOI,
                F_ISBN,
                F_ISSN,
                F_JOURNAL_YEAR_VOLUME,
                F_ARTICLE_PAGE_RANGE,
                F_PAGE_RANGE,
            ],
        ),
    );
    spans[F_EDITOR_TERM] = without_overlaps(&spans[F_EDITOR_TERM], &spans[F_EDITION_TERM].clone());
    spans[F_VOLUME_SHAPE] = without_overlaps(
        &spans[F_VOLUME_SHAPE],
        &joined(&spans, &[F_JOURNAL_YEAR_VOLUME, F_VOLUME_MARKER]),
    );
    spans[F_PLACE_NAME] =
        without_overlaps(&spans[F_PLACE_NAME], &spans[F_PLACE_PUBLISHER_SHAPE].clone());
    spans[F_PUBLISHER_TERM] = without_overlaps(
        &spans[F_PUBLISHER_TERM],
        &joined(&spans, &[F_PLACE_PUBLISHER_SHAPE, F_URL, F_DOI]),
    );
    spans[F_AMPERSAND] =
        without_overlaps(&spans[F_AMPERSAND], &joined(&spans, &[F_URL, F_DOI]));
    spans[F_NUMBERED_ENTRY] = without_overlaps(
        &spans[F_NUMBERED_ENTRY],
        &joined(
            &spans,
            &[F_NUMERIC_DATE, F_MONTH_DATE, F_ARTICLE_PAGE_RANGE, F_PAGE_RANGE],
        ),
    );
    spans[F_INITIAL] = without_overlaps(
        &spans[F_INITIAL],
        &joined(
            &spans,
            &[
                F_NO_DATE,
                F_QUOTED,
                F_EDITOR_TERM,
                F_THESIS_TERM,
                F_EDITION_TERM,
                F_VOLUME_MARKER,
                F_PAGE_MARKER,
            ],
        ),
    );

    let dotted_blockers = joined(
        &spans,
        &[
            F_URL,
            F_DOI,
            F_NO_DATE,
            F_INITIAL,
            F_QUOTED,
            F_EDITOR_TERM,
            F_THESIS_TERM,
            F_EDITION_TERM,
            F_VOLUME_MARKER,
            F_PAGE_MARKER,
            F_PUBLISHER_TERM,
            F_PLACE_NAME,
            F_PLACE_PUBLISHER_SHAPE,
        ],
    );
    let residual_dotted = without_overlaps(&spans[F_DOTTED_WORD], &dotted_blockers);
    spans[F_DOTTED_SEQUENCE] = dotted_sequences(value, &residual_dotted);
    spans[F_DOTTED_WORD] = without_overlaps(&residual_dotted, &spans[F_DOTTED_SEQUENCE].clone());

    let proper_blockers = joined(
        &spans,
        &[
            F_URL,
            F_DOI,
            F_ISBN,
            F_ISSN,
            F_NUMERIC_DATE,
            F_MONTH_DATE,
            F_ACCESS_DATE,
            F_QUOTED,
            F_EDITOR_TERM,
            F_THESIS_TERM,
            F_IN_CONTAINER,
            F_EDITION_TERM,
            F_DOTTED_WORD,
            F_DOTTED_SEQUENCE,
            F_VOLUME_MARKER,
            F_PAGE_MARKER,
            F_PUBLISHER_TERM,
            F_PLACE_NAME,
            F_PLACE_PUBLISHER_SHAPE,
            F_PROSE_LEAD,
        ],
    );
    spans[F_PROPER_WORD] = without_overlaps(&spans[F_PROPER_WORD], &proper_blockers);

    let stripped = value.trim_matches(py_isspace);
    spans[F_TABLE_ROW] = if stripped.starts_with('|') && stripped.ends_with('|') {
        let stripped_start = value.len() - value.trim_start_matches(py_isspace).len();
        let stripped_end = value.trim_end_matches(py_isspace).len();
        vec![(stripped_start, stripped_end)]
    } else {
        Vec::new()
    };

    // Punctuation is the residue: characters no semantic detector claimed.
    let semantic: Vec<Span> = (0..N_FEATURES)
        .filter(|i| *i != F_PUNCTUATION)
        .flat_map(|i| spans[i].iter().copied())
        .collect();
    let candidates: Vec<Span> = value
        .char_indices()
        .filter(|(_, ch)| PUNCTUATION.contains(ch))
        .map(|(i, ch)| (i, i + ch.len_utf8()))
        .collect();
    spans[F_PUNCTUATION] = without_overlaps(&candidates, &semantic);

    spans
}

/// `token_count` — not one of the 35 columns, but part of `BibliographyFeatures`.
pub fn token_count(value: &str) -> usize {
    PATTERNS.get("_TOKEN").find_iter(value).flatten().count()
}

/// `unicodedata.normalize("NFKC", text)` — every entry point in `bibliography_v2`
/// normalizes before extracting, and all span offsets are relative to the result.
/// It is not cosmetic: it is what turns fullwidth forms and compatibility digits
/// into the ASCII the detectors are written against.
pub fn normalize(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfkc().collect()
}

/// Compute the count row for one line. Takes raw text and normalizes, matching
/// `extract_bibliography_features`.
pub fn line_counts(text: &str) -> [u32; N_FEATURES] {
    let value = normalize(text);
    let spans = feature_spans(&value);
    let mut counts = [0u32; N_FEATURES];
    for (i, s) in spans.iter().enumerate() {
        counts[i] = s.len() as u32;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_order_is_stable() {
        assert_eq!(FEATURE_NAMES.len(), 35);
        assert_eq!(FEATURE_NAMES[0], "year_count");
        assert_eq!(FEATURE_NAMES[N_FEATURES - 1], "table_row_count");
    }

    #[test]
    fn indices_match_names() {
        for (idx, name) in [
            (F_YEAR, "year_count"),
            (F_INITIAL, "initial_count"),
            (F_DIRECT_AUTHOR, "direct_author_count"),
            (F_DOTTED_SEQUENCE, "dotted_sequence_count"),
            (F_PUNCTUATION, "punctuation_count"),
            (F_TABLE_ROW, "table_row_count"),
        ] {
            assert_eq!(FEATURE_NAMES[idx], name);
        }
    }

    #[test]
    fn table_row_needs_both_pipes() {
        assert_eq!(line_counts("| a | b |")[F_TABLE_ROW], 1);
        // A leading pipe alone is not a row — Python checks both ends.
        assert_eq!(line_counts("| unterminated")[F_TABLE_ROW], 0);
    }

    #[test]
    fn year_inside_a_url_is_owned_by_the_url() {
        let c = line_counts("see https://x.gr/2013/2/a.pdf");
        assert_eq!(c[F_URL], 1);
        assert_eq!(c[F_YEAR], 0);
    }
}
