//! Explainable deterministic table-of-contents and bibliography companion.
//!
//! This module is intentionally additive.  It does not replace the frozen line
//! heads or `reference_module::detect_doc`, and it never deletes text.  It emits
//! local evidence plus conservative, confirmed block candidates.  A downstream
//! policy may inspect those candidates, but must make its own keep/drop decision.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::reference_signals as sig;

pub const MODEL_ID: &str = "deterministic_structural_rules_v1";
pub const DECODER_ID: &str = "confirmed_blocks_typed_gaps_v1";

// ---------------------------------------------------------------------------
// Public parity vocabulary
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StructureKind {
    Toc,
    Bibliography,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LineRole {
    Blank,
    Heading,
    Subheading,
    StrongEntryStart,
    WeakEntryStart,
    PossibleContinuation,
    HardOther,
    Other,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BibStyle {
    AuthorYear,
    Numeric,
    Publication,
    Web,
    Legal,
    Mixed,
    Unknown,
}

/// Stable reason codes are part of the Python/Rust parity surface.  Add new
/// variants instead of changing the meaning of an existing one.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    BlankLine,
    TocHeading,
    TocLeader,
    TocSectionPrefix,
    TocTableRow,
    TocPageArabic,
    TocPageRoman,
    TocTitleText,
    BibHeading,
    BibSubheading,
    BibAuthor,
    BibYear,
    BibNumberedEntry,
    BibPublication,
    BibPersistentId,
    BibUrl,
    BibLegalCitation,
    BibContinuationShape,
    HardProse,
    HardInlineCitation,
    HardStatisticalTable,
    HardLegalBody,
    HardNonStructuralHeading,
    HardCvSection,
    HardNotesSection,
    HardFootnoteStream,
    ConfirmedByHeading,
    ConfirmedHeaderlessDensity,
    ConfirmedPageProgression,
    ConfirmedStyleCoherence,
    TerminatedByHardBarrier,
    TerminatedByIncompatibleLine,
    ConflictFailClosed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PageTail {
    pub raw: String,
    /// Comparable page ordinal. Roman and Arabic pages are kept distinct by
    /// `is_roman`; a Roman-to-Arabic reset is valid ToC progression.
    pub ordinal: u32,
    pub is_roman: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StructureLineEvidence {
    /// Zero-based absolute line index in the input document.
    pub line_index: usize,
    /// Unicode scalar-value offsets; `char_end` is exclusive.
    pub char_start: usize,
    pub char_end: usize,
    pub toc_role: LineRole,
    pub bib_role: LineRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toc_page: Option<PageTail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bib_style: Option<BibStyle>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bib_entry_number: Option<u32>,
    pub hard_negative: bool,
    pub reason_codes: Vec<ReasonCode>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StructureSpan {
    pub kind: StructureKind,
    pub char_start: usize,
    pub char_end: usize,
    pub line_start: usize,
    pub line_end: usize,
    pub seed_line: usize,
    pub confirmed_by: Vec<usize>,
    /// Heading/entry anchors actually traversed by the decoder.
    pub supporting_lines: Vec<usize>,
    /// Weak, continuation, subheading, or blank lines included only because
    /// compatible anchors surrounded them (or a typed terminal tail allowed it).
    pub bridged_lines: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terminated_by: Option<usize>,
    pub reason_codes: Vec<ReasonCode>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StructureConflict {
    pub toc_line_start: usize,
    pub toc_line_end: usize,
    pub bib_line_start: usize,
    pub bib_line_end: usize,
    pub reason_code: ReasonCode,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StructuralDecision {
    pub doc_id: String,
    pub source: String,
    pub model_id: String,
    pub decoder_id: String,
    pub line_evidence: Vec<StructureLineEvidence>,
    /// Only confirmed, mutually non-conflicting candidates are emitted here.
    /// These are annotations, not deletion instructions.
    pub spans: Vec<StructureSpan>,
    /// Overlapping ToC/BIB candidates are withheld from `spans` and recorded
    /// here so downstream consumers fail closed.
    pub conflicts: Vec<StructureConflict>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StructuralConfig {
    pub toc_heading_window_present: usize,
    pub toc_heading_min_strong: usize,
    pub toc_headerless_window_present: usize,
    pub toc_headerless_min_strong: usize,
    /// Headerless ToCs must start before both limits. Confirmed explicit
    /// headings are exempt from this positional prior.
    pub toc_headerless_max_absolute_line: usize,
    pub toc_headerless_max_position_fraction: f32,
    pub bib_heading_window_present: usize,
    pub bib_heading_min_strong: usize,
    pub bib_headerless_window_present: usize,
    pub bib_headerless_min_strong: usize,
    /// Maximum number of typed weak/continuation present lines between anchors.
    pub max_soft_gap_present: usize,
    /// Token cap for the same typed gap. Whitespace tokens are used deliberately
    /// so the Rust and Python implementations can reproduce it cheaply.
    pub max_soft_gap_tokens: usize,
    pub max_blank_gap_lines: usize,
}

impl Default for StructuralConfig {
    fn default() -> Self {
        Self {
            toc_heading_window_present: 8,
            toc_heading_min_strong: 2,
            toc_headerless_window_present: 10,
            toc_headerless_min_strong: 4,
            toc_headerless_max_absolute_line: 300,
            toc_headerless_max_position_fraction: 0.30,
            bib_heading_window_present: 10,
            bib_heading_min_strong: 2,
            bib_headerless_window_present: 10,
            bib_headerless_min_strong: 4,
            max_soft_gap_present: 2,
            max_soft_gap_tokens: 80,
            max_blank_gap_lines: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Local signal inventory
// ---------------------------------------------------------------------------

static ATX: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s{0,3}#{1,6}\s+").unwrap());
static TOC_LEADER: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?:\.{3,}|_{3,}|…{2,}|·{3,}|(?:\.\s*){4,})").unwrap());
static TOC_SECTION_PREFIX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?iu)^\s*\|?\s*(?:\d+(?:\.\d+){0,5}|[ivxlcdm]+)[.)]?\s+\p{L}").unwrap()
});
static PAGE_TAIL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:^|[\s|.…·_-])(?P<page>\d{1,4}|[ivxlcdm]{1,8})\s*\|?\s*$").unwrap()
});
static BIB_AUTHOR: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?u)\b\p{Lu}[\p{L}'’\-]{2,},\s*(?:\p{Lu}\.?|\p{Lu}[\p{L}'’\-]{2,})").unwrap()
});
static BIB_VANCOUVER_AUTHOR: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?u)(?:^|\]\s*)\p{Lu}[\p{L}'’\-]{2,}\s+(?:\p{Lu}\.?\s*){1,3}[,.;]").unwrap()
});
static BIB_AUTHOR_YEAR_ENTRY: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?u)^\s*(?:\[?\d{1,4}\]?[.)]?\s+)?\p{Lu}[\p{L}'’\-]{1,40}\s+\((?:1[5-9]|20)\d{2}[a-zα-ω]?\)\s*[.,;:]",
    )
    .unwrap()
});
static BIB_NUMBER: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^\s*(?:\[(?P<bracket>\d{1,4})\]|(?P<plain>\d{1,4})[.)])\s+").unwrap()
});
static YEAR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(?:1[5-9]\d{2}|20\d{2})[a-zα-ω]?\b|\bχ\.?\s*χ\.?").unwrap());
static PUBLICATION: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)(?:\b(?:vol|no|issue|pp?|eds?|edition)\.?\s*\d*\b|\bεκδοσεισ\b|\b(?:εκδ|επιμ|μτφρ|τομ|τχ|σσ?|σελ)\.(?:\s*\d+)?\b|\b\d+\s*[-–]\s*\d+\b|\b(?:αθηνα|θεσσαλονικη|λευκωσια|london|new york)\s*[:·])",
    )
    .unwrap()
});
static PERSISTENT_ID: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(?:doi\s*:|doi\.org/|isbn(?:-1[03])?\s*:|issn\s*:)").unwrap());
static URL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(?:https?://|www\.|\.(?:gr|com|org|edu)/)").unwrap());
static NARRATIVE_AUTHOR_YEAR: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?iu)^\s*\p{L}[\p{L}'’\-]{1,40}\s+\((?:1[5-9]|20)\d{2}[a-zα-ω]?\)\s+\p{L}")
        .unwrap()
});
static STATISTICAL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:%|€|\$|\b(?:ποσοστο|μεσος ορος|συχνοτητα|ετος|year)\b)").unwrap()
});
static LEGAL_CITATION: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:\bν\.?\s*\d{2,5}/\d{4}\b|\bπ\.?δ\.?\s*\d{1,4}/\d{4}\b|\bφεκ\s+[α-ωa-z]?\s*\d{1,5}/\d{4}\b|\bcase\s+[ct]-?\d+)").unwrap()
});
static LEGAL_SECONDARY: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:\b(?:αρθρο|παρ\.|εδ\.|οδηγια|κανονισμος)\s*\d+|\b(?:ecli|eur-lex)\b)")
        .unwrap()
});

const TOC_HEADINGS: &[&str] = &[
    "περιεχομενα",
    "πινακας περιεχομενων",
    "αναλυτικα περιεχομενα",
    "contents",
    "table of contents",
    "toc",
];
const BIB_HEADINGS: &[&str] = &[
    "βιβλιογραφια",
    "βιβλιογραφικες αναφορες",
    "πηγες και βιβλιογραφια",
    "βιβλιογραφια και πηγες",
    "αναφορες",
    "references",
    "reference list",
    "bibliography",
    "bibliographical references",
    "bibliographic references",
    "works cited",
    "literature cited",
];
const BIB_SUBHEADINGS: &[&str] = &[
    "ελληνικη",
    "ελληνικη βιβλιογραφια",
    "ελληνογλωσση",
    "ελληνογλωσση βιβλιογραφια",
    "ξενη βιβλιογραφια",
    "ξενογλωσση",
    "ξενογλωσση βιβλιογραφια",
    "δικτυογραφια",
    "ιστοσελιδες",
    "ηλεκτρονικες πηγες",
    "νομοθεσια",
    "νομολογια",
    "πηγες",
    "greek references",
    "foreign references",
    "web references",
    "sources",
    "primary sources",
    "secondary sources",
    "further reading",
];
const CV_HEADINGS: &[&str] = &[
    "βιογραφικο σημειωμα",
    "βιογραφικο",
    "δημοσιευσεις",
    "επιλεγμενες δημοσιευσεις",
    "ανακοινωσεις",
    "curriculum vitae",
    "publications",
    "selected publications",
];
const NOTES_HEADINGS: &[&str] = &[
    "σημειωσεις",
    "υποσημειωσεις",
    "notes",
    "τελικες σημειωσεις",
    "end notes",
    "footnotes",
    "endnotes",
];
const INLINE_PROSE_CUES: &[&str] = &[
    "οπως υποστηριζει",
    "οπως αναφερει",
    "συμφωνα με",
    "στην παρουσα",
    "στην περιπτωση",
    "as argued by",
    "according to",
];
const LEGAL_BODY_CUES: &[&str] = &[
    "ο αιτων",
    "ο αναδοχος",
    "υποχρεουται να",
    "δικαιουται να",
    "η αιτηση",
    "κατα την εννοια",
];

#[derive(Clone)]
struct IndexedLine<'a> {
    line_index: usize,
    char_start: usize,
    char_end: usize,
    text: &'a str,
}

#[derive(Clone)]
struct Candidate {
    span: StructureSpan,
}

fn index_lines(text: &str) -> Vec<IndexedLine<'_>> {
    let raw: Vec<&str> = text.split('\n').collect();
    let mut cursor = 0usize;
    raw.iter()
        .enumerate()
        .map(|(line_index, line)| {
            let visible = line.strip_suffix('\r').unwrap_or(line);
            let start = cursor;
            let raw_chars = line.chars().count();
            cursor += raw_chars + usize::from(line_index + 1 < raw.len());
            IndexedLine {
                line_index,
                char_start: start,
                char_end: start + visible.chars().count(),
                text: visible,
            }
        })
        .collect()
}

fn canonical_heading(line: &str) -> String {
    let without_atx = ATX.replace(line, "");
    let folded = sig::fold(without_atx.trim());
    let trimmed = folded.trim_matches(|c: char| {
        c.is_whitespace() || matches!(c, ':' | ';' | '.' | '-' | '–' | '—' | '*' | '_')
    });
    let trimmed =
        trimmed.trim_start_matches(|c: char| c.is_ascii_digit() || matches!(c, '.' | ')' | ' '));
    trimmed.trim().to_string()
}

fn is_exact_heading(core: &str, headings: &[&str]) -> bool {
    headings.iter().any(|heading| core == sig::fold(heading))
}

fn is_allcaps_ish(line: &str) -> bool {
    let mut upper = 0usize;
    let mut alpha = 0usize;
    for c in line.chars() {
        if c.is_alphabetic() {
            alpha += 1;
            upper += usize::from(c.is_uppercase());
        }
    }
    alpha >= 3 && upper * 5 >= alpha * 4
}

fn heading_shape(line: &str) -> bool {
    ATX.is_match(line) || (line.trim().chars().count() <= 80 && is_allcaps_ish(line))
}

fn parse_page_tail(line: &str) -> Option<PageTail> {
    let captures = PAGE_TAIL.captures(line)?;
    let raw = captures.name("page")?.as_str();
    if raw.chars().all(|c| c.is_ascii_digit()) {
        let ordinal = raw.parse::<u32>().ok()?;
        return (ordinal > 0).then(|| PageTail {
            raw: raw.to_string(),
            ordinal,
            is_roman: false,
        });
    }
    let ordinal = roman_to_u32(raw)?;
    Some(PageTail {
        raw: raw.to_string(),
        ordinal,
        is_roman: true,
    })
}

fn roman_to_u32(raw: &str) -> Option<u32> {
    let upper = raw.to_ascii_uppercase();
    if upper.is_empty() {
        return None;
    }
    let value = |c| match c {
        'I' => Some(1u32),
        'V' => Some(5),
        'X' => Some(10),
        'L' => Some(50),
        'C' => Some(100),
        'D' => Some(500),
        'M' => Some(1000),
        _ => None,
    };
    let mut total = 0i32;
    let mut previous = 0u32;
    for c in upper.chars().rev() {
        let current = value(c)?;
        if current < previous {
            total -= current as i32;
        } else {
            total += current as i32;
            previous = current;
        }
    }
    let total = u32::try_from(total).ok()?;
    if !(1..=3999).contains(&total) || to_roman(total) != upper {
        return None;
    }
    Some(total)
}

fn to_roman(mut value: u32) -> String {
    const TABLE: &[(u32, &str)] = &[
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut out = String::new();
    for (unit, numeral) in TABLE {
        while value >= *unit {
            value -= *unit;
            out.push_str(numeral);
        }
    }
    out
}

fn table_row(line: &str) -> bool {
    line.trim_start().starts_with('|') && line.matches('|').count() >= 2
}

fn alpha_title(line: &str) -> bool {
    let letters = line.chars().filter(|c| c.is_alphabetic()).count();
    let words = line.split_whitespace().count();
    letters >= 3 && words <= 24
}

fn prose_shape(line: &str, folded: &str) -> bool {
    let chars = line.trim().chars().count();
    let words = line.split_whitespace().count();
    let cue = INLINE_PROSE_CUES.iter().any(|cue| folded.contains(cue));
    cue || (chars >= 150 && words >= 18 && !TOC_LEADER.is_match(line) && !table_row(line))
}

fn legal_body_shape(line: &str, folded: &str) -> bool {
    line.split_whitespace().count() >= 12 && LEGAL_BODY_CUES.iter().any(|cue| folded.contains(cue))
}

fn bib_entry_number(line: &str) -> Option<u32> {
    let captures = BIB_NUMBER.captures(line)?;
    captures
        .name("bracket")
        .or_else(|| captures.name("plain"))?
        .as_str()
        .parse()
        .ok()
}

fn analyze_line(
    line: &IndexedLine<'_>,
    deny_bib_scope: Option<ReasonCode>,
) -> StructureLineEvidence {
    let text = line.text;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return StructureLineEvidence {
            line_index: line.line_index,
            char_start: line.char_start,
            char_end: line.char_end,
            toc_role: LineRole::Blank,
            bib_role: LineRole::Blank,
            toc_page: None,
            bib_style: None,
            bib_entry_number: None,
            hard_negative: false,
            reason_codes: vec![ReasonCode::BlankLine],
        };
    }

    let folded = sig::fold(trimmed);
    let core = canonical_heading(trimmed);
    let toc_heading = is_exact_heading(&core, TOC_HEADINGS);
    let bib_heading = is_exact_heading(&core, BIB_HEADINGS);
    let bib_subheading = is_exact_heading(&core, BIB_SUBHEADINGS);
    let cv_heading = is_exact_heading(&core, CV_HEADINGS);
    let notes_heading = is_exact_heading(&core, NOTES_HEADINGS);
    let non_structural_heading =
        ATX.is_match(trimmed) && !toc_heading && !bib_heading && !bib_subheading;
    let leader = TOC_LEADER.is_match(trimmed);
    let section_prefix = TOC_SECTION_PREFIX.is_match(trimmed);
    let table = table_row(trimmed);
    let page = parse_page_tail(trimmed);
    let statistical = table && STATISTICAL.is_match(&folded) && !leader && !section_prefix;
    let prose = prose_shape(trimmed, &folded);
    let legal_body = legal_body_shape(trimmed, &folded);

    let mut reasons = Vec::new();
    if toc_heading {
        reasons.push(ReasonCode::TocHeading);
    }
    if leader {
        reasons.push(ReasonCode::TocLeader);
    }
    if section_prefix {
        reasons.push(ReasonCode::TocSectionPrefix);
    }
    if table {
        reasons.push(ReasonCode::TocTableRow);
    }
    if alpha_title(trimmed) {
        reasons.push(ReasonCode::TocTitleText);
    }
    if let Some(page) = &page {
        reasons.push(if page.is_roman {
            ReasonCode::TocPageRoman
        } else {
            ReasonCode::TocPageArabic
        });
    }
    if bib_heading {
        reasons.push(ReasonCode::BibHeading);
    }
    if bib_subheading {
        reasons.push(ReasonCode::BibSubheading);
    }
    if prose {
        reasons.push(ReasonCode::HardProse);
    }
    if statistical {
        reasons.push(ReasonCode::HardStatisticalTable);
    }
    if legal_body {
        reasons.push(ReasonCode::HardLegalBody);
    }
    if non_structural_heading {
        reasons.push(ReasonCode::HardNonStructuralHeading);
    }
    if cv_heading || deny_bib_scope == Some(ReasonCode::HardCvSection) {
        reasons.push(ReasonCode::HardCvSection);
    }
    if notes_heading || deny_bib_scope == Some(ReasonCode::HardNotesSection) {
        reasons.push(ReasonCode::HardNotesSection);
    }

    // An unrecognized ATX heading is a structural boundary even when its title
    // ends in a number or contains leader-like punctuation.
    let toc_hard = statistical || legal_body || non_structural_heading;
    let toc_role = if toc_heading {
        LineRole::Heading
    } else if toc_hard {
        LineRole::HardOther
    } else if page.is_some() && alpha_title(trimmed) && (leader || table || section_prefix) {
        LineRole::StrongEntryStart
    } else if page.is_some() && alpha_title(trimmed) {
        LineRole::WeakEntryStart
    } else if leader && alpha_title(trimmed) {
        LineRole::PossibleContinuation
    } else {
        LineRole::Other
    };

    let author = BIB_AUTHOR.is_match(trimmed)
        || BIB_VANCOUVER_AUTHOR.is_match(trimmed)
        || BIB_AUTHOR_YEAR_ENTRY.is_match(trimmed);
    let year = YEAR.is_match(trimmed);
    let number = bib_entry_number(trimmed);
    let publication = PUBLICATION.is_match(&folded);
    let persistent_id = PERSISTENT_ID.is_match(trimmed);
    let url = URL.is_match(trimmed);
    let legal_primary = LEGAL_CITATION.is_match(&folded);
    let legal_secondary = LEGAL_SECONDARY.is_match(&folded);
    let footnote_like = sig::FOOTNOTE_LINE.is_match(trimmed) && number.is_none();
    let inline_citation = (prose && year && author) || NARRATIVE_AUTHOR_YEAR.is_match(trimmed);

    if author {
        reasons.push(ReasonCode::BibAuthor);
    }
    if year {
        reasons.push(ReasonCode::BibYear);
    }
    if number.is_some() {
        reasons.push(ReasonCode::BibNumberedEntry);
    }
    if publication {
        reasons.push(ReasonCode::BibPublication);
    }
    if persistent_id {
        reasons.push(ReasonCode::BibPersistentId);
    }
    if url {
        reasons.push(ReasonCode::BibUrl);
    }
    if legal_primary || legal_secondary {
        reasons.push(ReasonCode::BibLegalCitation);
    }
    if inline_citation {
        reasons.push(ReasonCode::HardInlineCitation);
    }
    if footnote_like {
        reasons.push(ReasonCode::HardFootnoteStream);
    }

    let bib_hard = cv_heading
        || notes_heading
        || inline_citation
        || footnote_like
        || statistical
        || legal_body
        || non_structural_heading;
    let family_count = usize::from(author)
        + usize::from(year)
        + usize::from(number.is_some())
        + usize::from(publication)
        + usize::from(persistent_id || url)
        + usize::from(legal_primary)
        + usize::from(legal_secondary);
    let legal_subheading = bib_subheading && matches!(core.as_str(), "νομοθεσια" | "νομολογια");
    let style = if legal_primary || legal_secondary || legal_subheading {
        BibStyle::Legal
    } else if number.is_some() {
        BibStyle::Numeric
    } else if author && year {
        BibStyle::AuthorYear
    } else if persistent_id || url {
        BibStyle::Web
    } else if publication {
        BibStyle::Publication
    } else if family_count > 1 {
        BibStyle::Mixed
    } else {
        BibStyle::Unknown
    };

    let continuation = !author
        && number.is_none()
        && trimmed.chars().count() <= 220
        && (publication || persistent_id || url || legal_secondary);
    if continuation {
        reasons.push(ReasonCode::BibContinuationShape);
    }
    let bib_role = if bib_heading {
        LineRole::Heading
    } else if bib_subheading {
        LineRole::Subheading
    } else if bib_hard {
        LineRole::HardOther
    } else if family_count >= 2 && (author || number.is_some() || legal_primary) {
        LineRole::StrongEntryStart
    } else if continuation {
        LineRole::PossibleContinuation
    } else if family_count >= 1 && trimmed.chars().count() <= 240 {
        LineRole::WeakEntryStart
    } else {
        LineRole::Other
    };

    reasons.sort_by_key(|reason| *reason as usize);
    reasons.dedup();
    StructureLineEvidence {
        line_index: line.line_index,
        char_start: line.char_start,
        char_end: line.char_end,
        toc_role,
        bib_role,
        toc_page: page,
        bib_style: (style != BibStyle::Unknown).then_some(style),
        bib_entry_number: number,
        hard_negative: toc_role == LineRole::HardOther || bib_role == LineRole::HardOther,
        reason_codes: reasons,
    }
}

fn analyze_lines(lines: &[IndexedLine<'_>]) -> Vec<StructureLineEvidence> {
    let mut deny_bib_scope = None;
    let mut out = Vec::with_capacity(lines.len());
    for line in lines {
        let core = canonical_heading(line.text);
        let heading_candidate = ATX.is_match(line.text)
            || heading_shape(line.text)
            || is_exact_heading(&core, TOC_HEADINGS)
            || is_exact_heading(&core, BIB_HEADINGS)
            || is_exact_heading(&core, BIB_SUBHEADINGS)
            || is_exact_heading(&core, CV_HEADINGS)
            || is_exact_heading(&core, NOTES_HEADINGS);
        if heading_candidate {
            if is_exact_heading(&core, CV_HEADINGS) {
                deny_bib_scope = Some(ReasonCode::HardCvSection);
            } else if is_exact_heading(&core, NOTES_HEADINGS) {
                deny_bib_scope = Some(ReasonCode::HardNotesSection);
            } else if is_exact_heading(&core, BIB_HEADINGS) || is_exact_heading(&core, TOC_HEADINGS)
            {
                // An explicit structural heading safely closes a denied CV/notes scope.
                deny_bib_scope = None;
            }
        }
        out.push(analyze_line(line, deny_bib_scope));
    }
    out
}

fn role(evidence: &StructureLineEvidence, kind: StructureKind) -> LineRole {
    match kind {
        StructureKind::Toc => evidence.toc_role,
        StructureKind::Bibliography => evidence.bib_role,
    }
}

fn bib_scope_denied(evidence: &StructureLineEvidence) -> bool {
    evidence.reason_codes.iter().any(|reason| {
        matches!(
            reason,
            ReasonCode::HardCvSection | ReasonCode::HardNotesSection
        )
    })
}

fn toc_progresses(evidence: &[StructureLineEvidence], strong: &[usize]) -> bool {
    if strong.len() < 2 {
        return false;
    }
    let pages: Vec<&PageTail> = strong
        .iter()
        .filter_map(|line| evidence[*line].toc_page.as_ref())
        .collect();
    if pages.len() != strong.len() {
        return false;
    }
    let mut equal_run = 1usize;
    for pair in pages.windows(2) {
        if !pair[0].is_roman && pair[1].is_roman {
            return false;
        }
        if pair[0].is_roman != pair[1].is_roman {
            equal_run = 1;
            continue;
        }
        if pair[1].ordinal < pair[0].ordinal {
            return false;
        }
        if pair[1].ordinal == pair[0].ordinal {
            equal_run += 1;
            if equal_run > 4 {
                return false;
            }
        } else {
            equal_run = 1;
        }
    }
    true
}

fn bib_headerless_citation_specific(evidence: &StructureLineEvidence) -> bool {
    evidence.reason_codes.iter().any(|reason| {
        matches!(
            reason,
            ReasonCode::BibAuthor
                | ReasonCode::BibPublication
                | ReasonCode::BibPersistentId
                | ReasonCode::BibUrl
                | ReasonCode::BibLegalCitation
        )
    })
}

fn bib_style_compatible(evidence: &[StructureLineEvidence], strong: &[usize]) -> bool {
    if strong.len() < 2 {
        return false;
    }
    let styles: Vec<BibStyle> = strong
        .iter()
        .filter_map(|line| evidence[*line].bib_style)
        .collect();
    for candidate in [
        BibStyle::AuthorYear,
        BibStyle::Numeric,
        BibStyle::Publication,
        BibStyle::Web,
        BibStyle::Legal,
        BibStyle::Mixed,
    ] {
        if styles.iter().filter(|style| **style == candidate).count() >= 2 {
            if candidate == BibStyle::Numeric {
                let numbers: Vec<u32> = strong
                    .iter()
                    .filter_map(|line| evidence[*line].bib_entry_number)
                    .collect();
                return numbers.len() < 2 || numbers.windows(2).all(|pair| pair[1] > pair[0]);
            }
            return true;
        }
    }
    // Author-year and publication-style entries are compatible members of the
    // same conventional bibliography family.
    styles
        .iter()
        .filter(|style| **style == BibStyle::AuthorYear)
        .count()
        + styles
            .iter()
            .filter(|style| **style == BibStyle::Publication)
            .count()
        >= 2
}

fn window_after_seed(
    evidence: &[StructureLineEvidence],
    seed: usize,
    kind: StructureKind,
    max_present: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut strong = Vec::new();
    let mut compatible = Vec::new();
    let mut present = 0usize;
    for (line, item) in evidence.iter().enumerate().skip(seed + 1) {
        let line_role = role(item, kind);
        if line_role == LineRole::Blank {
            continue;
        }
        if present >= max_present {
            break;
        }
        present += 1;
        match line_role {
            LineRole::StrongEntryStart => {
                strong.push(line);
                compatible.push(line);
            }
            LineRole::WeakEntryStart | LineRole::PossibleContinuation | LineRole::Subheading => {
                compatible.push(line);
            }
            // A repeated ToC title is common page furniture in extracted PDFs.
            // It is neither page evidence nor a barrier: allow the seed window
            // to look through it, while still requiring real strong entries.
            LineRole::Heading if kind == StructureKind::Toc => compatible.push(line),
            LineRole::Heading | LineRole::HardOther | LineRole::Other => break,
            LineRole::Blank => unreachable!(),
        }
    }
    (strong, compatible)
}

fn headerless_window(
    evidence: &[StructureLineEvidence],
    start: usize,
    kind: StructureKind,
    max_present: usize,
) -> Vec<usize> {
    let mut strong = Vec::new();
    let mut present = 0usize;
    for item in evidence.iter().skip(start) {
        let line_role = role(item, kind);
        if line_role == LineRole::Blank {
            continue;
        }
        if present >= max_present {
            break;
        }
        present += 1;
        match line_role {
            LineRole::StrongEntryStart => strong.push(item.line_index),
            LineRole::WeakEntryStart | LineRole::PossibleContinuation => {}
            LineRole::Heading | LineRole::Subheading | LineRole::HardOther | LineRole::Other => {
                break
            }
            LineRole::Blank => unreachable!(),
        }
    }
    strong
}

fn is_anchor(line_role: LineRole) -> bool {
    matches!(line_role, LineRole::Heading | LineRole::StrongEntryStart)
}

fn is_soft(line_role: LineRole) -> bool {
    matches!(
        line_role,
        LineRole::Blank
            | LineRole::Subheading
            | LineRole::WeakEntryStart
            | LineRole::PossibleContinuation
    )
}

fn terminal_bib_continuation(item: &StructureLineEvidence) -> bool {
    item.bib_role == LineRole::PossibleContinuation
        && item.reason_codes.iter().any(|reason| {
            matches!(
                reason,
                ReasonCode::BibPublication | ReasonCode::BibPersistentId | ReasonCode::BibUrl
            )
        })
}

fn decode_seed(
    lines: &[IndexedLine<'_>],
    evidence: &[StructureLineEvidence],
    kind: StructureKind,
    seed: usize,
    confirmed_by: Vec<usize>,
    seed_reason: ReasonCode,
    cfg: &StructuralConfig,
) -> Option<Candidate> {
    let mut last_committed = seed;
    let mut supporting_lines = vec![seed];
    let mut bridged_lines = Vec::new();
    let mut pending: Vec<usize> = Vec::new();
    let mut pending_present = 0usize;
    let mut pending_tokens = 0usize;
    let mut blank_run = 0usize;
    let mut gap_overflow = false;
    let mut terminated_by = None;
    let mut termination_reason = None;

    for index in (seed + 1)..evidence.len() {
        let line_role = role(&evidence[index], kind);
        if line_role == LineRole::HardOther {
            terminated_by = Some(index);
            termination_reason = Some(ReasonCode::TerminatedByHardBarrier);
            break;
        }
        if line_role == LineRole::Other {
            terminated_by = Some(index);
            termination_reason = Some(ReasonCode::TerminatedByIncompatibleLine);
            break;
        }
        if is_anchor(line_role) {
            bridged_lines.extend(pending.iter().copied());
            supporting_lines.push(index);
            last_committed = index;
            pending.clear();
            pending_present = 0;
            pending_tokens = 0;
            blank_run = 0;
            continue;
        }
        if !is_soft(line_role) {
            terminated_by = Some(index);
            termination_reason = Some(ReasonCode::TerminatedByIncompatibleLine);
            break;
        }
        pending.push(index);
        if line_role == LineRole::Blank {
            blank_run += 1;
        } else {
            blank_run = 0;
            pending_present += 1;
            pending_tokens += lines[index].text.split_whitespace().count();
        }
        if pending.len() > cfg.max_soft_gap_present
            || blank_run > cfg.max_blank_gap_lines
            || pending_present > cfg.max_soft_gap_present
            || pending_tokens > cfg.max_soft_gap_tokens
        {
            gap_overflow = true;
            terminated_by = Some(index);
            termination_reason = Some(ReasonCode::TerminatedByIncompatibleLine);
            break;
        }
    }

    // A typed publisher/DOI/URL tail may close the final entry. Other pending
    // lines are kept outside the emitted span unless a following anchor confirms
    // them; this is the conservative `GAP_PENDING` behaviour.
    if kind == StructureKind::Bibliography
        && !gap_overflow
        && pending.iter().all(|line| {
            evidence[*line].bib_role == LineRole::Blank
                || terminal_bib_continuation(&evidence[*line])
        })
        && pending
            .iter()
            .any(|line| terminal_bib_continuation(&evidence[*line]))
    {
        if let Some(last) = pending.last() {
            bridged_lines.extend(pending.iter().copied());
            last_committed = *last;
        }
    }

    let confirmed_in_span = confirmed_by
        .iter()
        .filter(|line| **line <= last_committed)
        .count();
    let bridged_present = bridged_lines
        .iter()
        .filter(|line| evidence[**line].bib_role != LineRole::Blank)
        .count();
    let confirmed = match (kind, seed_reason) {
        (StructureKind::Toc, ReasonCode::ConfirmedByHeading) => {
            confirmed_in_span >= cfg.toc_heading_min_strong
        }
        (StructureKind::Toc, ReasonCode::ConfirmedHeaderlessDensity) => {
            confirmed_in_span >= cfg.toc_headerless_min_strong
        }
        (StructureKind::Bibliography, ReasonCode::ConfirmedByHeading) => {
            confirmed_in_span >= cfg.bib_heading_min_strong
                || (confirmed_in_span == 1 && bridged_present >= 2)
        }
        (StructureKind::Bibliography, ReasonCode::ConfirmedHeaderlessDensity) => {
            confirmed_in_span >= cfg.bib_headerless_min_strong
        }
        _ => false,
    };
    if !confirmed {
        return None;
    }

    let mut reason_codes = vec![seed_reason];
    reason_codes.push(match kind {
        StructureKind::Toc => ReasonCode::ConfirmedPageProgression,
        StructureKind::Bibliography => ReasonCode::ConfirmedStyleCoherence,
    });
    if let Some(reason) = termination_reason {
        reason_codes.push(reason);
    }
    Some(Candidate {
        span: StructureSpan {
            kind,
            char_start: lines[seed].char_start,
            char_end: lines[last_committed].char_end,
            line_start: seed,
            line_end: last_committed,
            seed_line: seed,
            confirmed_by,
            supporting_lines,
            bridged_lines,
            terminated_by,
            reason_codes,
        },
    })
}

fn seeds_for_kind(
    evidence: &[StructureLineEvidence],
    kind: StructureKind,
    cfg: &StructuralConfig,
) -> Vec<(usize, Vec<usize>, ReasonCode)> {
    let (heading_window, heading_min, headerless_window_size, headerless_min) = match kind {
        StructureKind::Toc => (
            cfg.toc_heading_window_present,
            cfg.toc_heading_min_strong,
            cfg.toc_headerless_window_present,
            cfg.toc_headerless_min_strong,
        ),
        StructureKind::Bibliography => (
            cfg.bib_heading_window_present,
            cfg.bib_heading_min_strong,
            cfg.bib_headerless_window_present,
            cfg.bib_headerless_min_strong,
        ),
    };
    let toc_headerless_cut = cfg
        .toc_headerless_max_absolute_line
        .min(((evidence.len() as f32) * cfg.toc_headerless_max_position_fraction).floor() as usize);
    let mut seeds = Vec::new();
    for (index, item) in evidence.iter().enumerate() {
        // CV/publication and notes scopes suppress bibliography decoding, but
        // do not mutate the intrinsic, context-free role of the current line.
        if kind == StructureKind::Bibliography && bib_scope_denied(item) {
            continue;
        }
        let line_role = role(item, kind);
        let typed_bib_subheading =
            kind == StructureKind::Bibliography && line_role == LineRole::Subheading;
        if line_role == LineRole::Heading || typed_bib_subheading {
            let (strong, compatible) = window_after_seed(evidence, index, kind, heading_window);
            let enough = if kind == StructureKind::Bibliography {
                strong.len() >= heading_min
                    || (strong.len() == 1 && compatible.len().saturating_sub(strong.len()) >= 2)
            } else {
                strong.len() >= heading_min
            };
            let coherent = match kind {
                StructureKind::Toc => toc_progresses(evidence, &strong),
                StructureKind::Bibliography => {
                    let base = bib_style_compatible(evidence, &strong)
                        || (strong.len() == 1 && compatible.len() >= 3);
                    base && (!typed_bib_subheading
                        || strong
                            .iter()
                            .filter(|line| bib_headerless_citation_specific(&evidence[**line]))
                            .count()
                            >= heading_min)
                }
            };
            if enough && coherent {
                seeds.push((index, strong, ReasonCode::ConfirmedByHeading));
            }
            continue;
        }
        if line_role != LineRole::StrongEntryStart {
            continue;
        }
        if kind == StructureKind::Toc && index >= toc_headerless_cut {
            continue;
        }
        let strong = headerless_window(evidence, index, kind, headerless_window_size);
        if strong.len() < headerless_min {
            continue;
        }
        let coherent = match kind {
            StructureKind::Toc => toc_progresses(evidence, &strong),
            StructureKind::Bibliography => {
                // Legal/reference apparatus needs an explicit heading; it is
                // intentionally excluded from headerless discovery.
                item.bib_style != Some(BibStyle::Legal)
                    && strong
                        .iter()
                        .all(|line| bib_headerless_citation_specific(&evidence[*line]))
                    && bib_style_compatible(evidence, &strong)
            }
        };
        if coherent {
            seeds.push((index, strong, ReasonCode::ConfirmedHeaderlessDensity));
        }
    }
    seeds
}

fn candidates_for_kind(
    lines: &[IndexedLine<'_>],
    evidence: &[StructureLineEvidence],
    kind: StructureKind,
    cfg: &StructuralConfig,
) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    let mut consumed_through = None;
    for (seed, confirmed_by, reason) in seeds_for_kind(evidence, kind, cfg) {
        if consumed_through.is_some_and(|end| seed <= end) {
            continue;
        }
        if let Some(candidate) = decode_seed(lines, evidence, kind, seed, confirmed_by, reason, cfg)
        {
            consumed_through = Some(candidate.span.line_end);
            candidates.push(candidate);
        }
    }
    candidates
}

fn overlaps(left: &StructureSpan, right: &StructureSpan) -> bool {
    left.line_start <= right.line_end && right.line_start <= left.line_end
}

/// Analyze a document and emit explainable, conservative ToC/BIB candidate
/// blocks. This function has no keep/drop input and performs no mutation.
pub fn structural_detect(
    doc_id: &str,
    source: &str,
    text: &str,
    cfg: &StructuralConfig,
) -> StructuralDecision {
    let lines = index_lines(text);
    let line_evidence = analyze_lines(&lines);
    let toc = candidates_for_kind(&lines, &line_evidence, StructureKind::Toc, cfg);
    let bib = candidates_for_kind(&lines, &line_evidence, StructureKind::Bibliography, cfg);
    let mut rejected_toc = vec![false; toc.len()];
    let mut rejected_bib = vec![false; bib.len()];
    let mut conflicts = Vec::new();
    for (toc_index, toc_candidate) in toc.iter().enumerate() {
        for (bib_index, bib_candidate) in bib.iter().enumerate() {
            if overlaps(&toc_candidate.span, &bib_candidate.span) {
                rejected_toc[toc_index] = true;
                rejected_bib[bib_index] = true;
                conflicts.push(StructureConflict {
                    toc_line_start: toc_candidate.span.line_start,
                    toc_line_end: toc_candidate.span.line_end,
                    bib_line_start: bib_candidate.span.line_start,
                    bib_line_end: bib_candidate.span.line_end,
                    reason_code: ReasonCode::ConflictFailClosed,
                });
            }
        }
    }
    let mut spans: Vec<StructureSpan> = toc
        .into_iter()
        .enumerate()
        .filter_map(|(index, candidate)| (!rejected_toc[index]).then_some(candidate.span))
        .chain(
            bib.into_iter()
                .enumerate()
                .filter_map(|(index, candidate)| (!rejected_bib[index]).then_some(candidate.span)),
        )
        .collect();
    spans.sort_by_key(|span| (span.line_start, span.line_end));
    StructuralDecision {
        doc_id: doc_id.to_string(),
        source: source.to_string(),
        model_id: MODEL_ID.to_string(),
        decoder_id: DECODER_ID.to_string(),
        line_evidence,
        spans,
        conflicts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detect(text: &str) -> StructuralDecision {
        structural_detect("doc", "fixture", text, &StructuralConfig::default())
    }

    #[test]
    fn single_line_evidence_handles_arabic_roman_and_table_toc_rows() {
        let result =
            detect("Πρόλογος ........ ix\n1.2 Μεθοδολογία ........ 417\n| 2. Αποτελέσματα | 512 |");
        assert_eq!(result.line_evidence[0].toc_role, LineRole::StrongEntryStart);
        assert_eq!(
            result.line_evidence[0].toc_page.as_ref().unwrap().ordinal,
            9
        );
        assert!(result.line_evidence[0].toc_page.as_ref().unwrap().is_roman);
        assert_eq!(result.line_evidence[1].toc_role, LineRole::StrongEntryStart);
        assert_eq!(
            result.line_evidence[1].toc_page.as_ref().unwrap().ordinal,
            417
        );
        assert_eq!(result.line_evidence[2].toc_role, LineRole::StrongEntryStart);
        assert!(result.line_evidence[2]
            .reason_codes
            .contains(&ReasonCode::TocTableRow));
    }

    #[test]
    fn rejects_noncanonical_roman_word_tail() {
        let result = detect("Αυτό είναι ένα κείμενο civil");
        assert!(result.line_evidence[0].toc_page.is_none());
        assert_eq!(result.line_evidence[0].toc_role, LineRole::Other);
    }

    #[test]
    fn roman_parser_accepts_subtractive_canonical_forms_only() {
        assert_eq!(roman_to_u32("IV"), Some(4));
        assert_eq!(roman_to_u32("ix"), Some(9));
        assert_eq!(roman_to_u32("XIV"), Some(14));
        assert_eq!(roman_to_u32("IIII"), None);
        assert_eq!(roman_to_u32("IIV"), None);
        assert_eq!(roman_to_u32("civil"), None);
    }

    #[test]
    fn toc_heading_requires_confirmation_and_never_runs_to_eof() {
        let result = detect("## ΠΕΡΙΕΧΟΜΕΝΑ\nΑυτό είναι κανονικό κείμενο του κεφαλαίου.");
        assert!(result.spans.is_empty());

        let result = detect(
            "## ΠΕΡΙΕΧΟΜΕΝΑ\n1. Εισαγωγή ........ 1\n2. Μέθοδος ........ 7\n## ΚΕΦΑΛΑΙΟ\nΚανονικό σώμα.",
        );
        let span = result
            .spans
            .iter()
            .find(|span| span.kind == StructureKind::Toc)
            .unwrap();
        assert_eq!((span.line_start, span.line_end), (0, 2));
        assert_eq!(span.terminated_by, Some(3));
    }

    #[test]
    fn exact_plain_title_case_headings_are_valid_anchors() {
        let toc = detect("Περιεχόμενα\nΕισαγωγή ........ 1\nΜέθοδος ........ 7");
        assert_eq!(toc.line_evidence[0].toc_role, LineRole::Heading);
        assert!(toc.spans.iter().any(|span| span.kind == StructureKind::Toc));

        let bib = detect(
            "Βιβλιογραφία\nΠαπαδόπουλος, Α. (2019). Τίτλος. Αθήνα: Εκδόσεις.\nSmith, J. (2020). A title. London: Press.",
        );
        assert_eq!(bib.line_evidence[0].bib_role, LineRole::Heading);
        assert!(bib
            .spans
            .iter()
            .any(|span| span.kind == StructureKind::Bibliography));
    }

    #[test]
    fn headerless_toc_requires_dense_monotonic_entries() {
        let result = detect(
            "1. Εισαγωγή ........ 1\n2. Θεωρία ........ 5\n3. Μέθοδος ........ 12\n4. Αποτελέσματα ........ 20\nΚανονικό κείμενο.",
        );
        let spans: Vec<_> = result
            .spans
            .iter()
            .filter(|span| span.kind == StructureKind::Toc)
            .collect();
        assert_eq!(spans.len(), 1);
        assert_eq!((spans[0].line_start, spans[0].line_end), (0, 3));

        let shuffled = detect(
            "1. Εισαγωγή ........ 20\n2. Θεωρία ........ 5\n3. Μέθοδος ........ 12\n4. Αποτελέσματα ........ 1",
        );
        assert!(shuffled
            .spans
            .iter()
            .all(|span| span.kind != StructureKind::Toc));
    }

    #[test]
    fn late_headerless_toc_is_rejected_but_late_explicit_heading_is_allowed() {
        let prefix = "Κανονικό σώμα.\n".repeat(40);
        let entries = "1. Εισαγωγή ........ 1\n2. Θεωρία ........ 5\n3. Μέθοδος ........ 12\n4. Αποτελέσματα ........ 20";
        let headerless = detect(&(prefix.clone() + entries));
        assert!(headerless
            .spans
            .iter()
            .all(|span| span.kind != StructureKind::Toc));

        let explicit = detect(
            &(prefix
                + "Περιεχόμενα\n"
                + "1. Εισαγωγή ........ 1\n2. Μέθοδος ........ 5\nΚανονικό σώμα."),
        );
        let span = explicit
            .spans
            .iter()
            .find(|span| span.kind == StructureKind::Toc)
            .unwrap();
        assert_eq!(span.line_start, 40);
        assert_eq!(span.line_end, 42);
    }

    #[test]
    fn bibliography_roles_require_independent_evidence_families() {
        let result = detect(
            "Παπαδόπουλος, Α. (2019). Τίτλος. Αθήνα: Εκδόσεις.\nΑθήνα: Εκδόσεις Πατάκη.\nΌπως υποστηρίζει ο Παπαδόπουλος, Α. (2019), η θεωρία αυτή είναι σημαντική και αναλύεται στην παρούσα ενότητα με λεπτομέρεια.",
        );
        assert_eq!(result.line_evidence[0].bib_role, LineRole::StrongEntryStart);
        assert_eq!(
            result.line_evidence[1].bib_role,
            LineRole::PossibleContinuation
        );
        assert_eq!(result.line_evidence[2].bib_role, LineRole::HardOther);
        assert!(result.line_evidence[2]
            .reason_codes
            .contains(&ReasonCode::HardInlineCitation));
    }

    #[test]
    fn bib_heading_is_only_an_anchor_and_block_stops_at_body() {
        let unconfirmed = detect("## ΒΙΒΛΙΟΓΡΑΦΙΑ\nΜία μόνη ασαφής γραμμή.");
        assert!(unconfirmed
            .spans
            .iter()
            .all(|span| span.kind != StructureKind::Bibliography));

        let distant = detect(
            &("Βιβλιογραφία\n".to_string()
                + &"\n".repeat(8)
                + "Παπαδόπουλος, Α. (2019). Τίτλος. Αθήνα: Εκδόσεις.\nSmith, J. (2020). A title. London: Press."),
        );
        assert!(distant
            .spans
            .iter()
            .all(|span| span.kind != StructureKind::Bibliography));

        let result = detect(
            "## ΒΙΒΛΙΟΓΡΑΦΙΑ\nΠαπαδόπουλος, Α. (2019). Τίτλος. Αθήνα: Εκδόσεις.\nSmith, J. (2020). A title. London: Press.\n## ΕΠΟΜΕΝΟ ΚΕΦΑΛΑΙΟ\nΚανονικό σώμα μετά τη βιβλιογραφία.",
        );
        let span = result
            .spans
            .iter()
            .find(|span| span.kind == StructureKind::Bibliography)
            .unwrap();
        assert_eq!((span.line_start, span.line_end), (0, 2));
        assert_eq!(span.terminated_by, Some(3));
    }

    #[test]
    fn bib_decoder_bridges_only_typed_continuations() {
        let result = detect(
            "## ΒΙΒΛΙΟΓΡΑΦΙΑ\nΠαπαδόπουλος, Α. (2019). Η γλώσσα.\nΑθήνα: Εκδόσεις Πατάκη.\nSmith, J. (2020). A title. London: Press.\nΑυτό είναι κανονικό σώμα κειμένου που δεν αποτελεί βιβλιογραφική συνέχεια και πρέπει να παραμείνει.\nBrown, K. (2021). Another title. London: Press.",
        );
        let span = result
            .spans
            .iter()
            .find(|span| span.kind == StructureKind::Bibliography)
            .unwrap();
        assert_eq!((span.line_start, span.line_end), (0, 3));
        assert_eq!(span.terminated_by, Some(4));
        assert_eq!(span.supporting_lines, vec![0, 1, 3]);
        assert_eq!(span.bridged_lines, vec![2]);
    }

    #[test]
    fn detects_multiple_formal_bibliography_blocks() {
        let result = detect(
            "## ΒΙΒΛΙΟΓΡΑΦΙΑ\nΠαπαδόπουλος, Α. (2019). Τίτλος. Αθήνα: Εκδόσεις.\nSmith, J. (2020). A title. London: Press.\n## ΚΕΦΑΛΑΙΟ\nΣώμα.\n## ΑΝΑΦΟΡΕΣ\nBrown, K. (2021). A title. London: Press.\nJones, P. (2022). A title. London: Press.\n## ΤΕΛΟΣ",
        );
        let bib_spans: Vec<_> = result
            .spans
            .iter()
            .filter(|span| span.kind == StructureKind::Bibliography)
            .collect();
        assert_eq!(bib_spans.len(), 2);
        assert_eq!((bib_spans[0].line_start, bib_spans[0].line_end), (0, 2));
        assert_eq!((bib_spans[1].line_start, bib_spans[1].line_end), (5, 7));
    }

    #[test]
    fn cv_and_notes_scopes_cannot_seed_headerless_bibliographies() {
        let result = detect(
            "## ΔΗΜΟΣΙΕΥΣΕΙΣ\nΠαπαδόπουλος, Α. (2019). Τίτλος. Αθήνα: Εκδόσεις.\nSmith, J. (2020). A title. London: Press.\nBrown, K. (2021). A title. London: Press.\nJones, P. (2022). A title. London: Press.",
        );
        assert!(result
            .spans
            .iter()
            .all(|span| span.kind != StructureKind::Bibliography));
        assert!(result.line_evidence[1]
            .reason_codes
            .contains(&ReasonCode::HardCvSection));

        let plain_cv = detect(
            "Δημοσιεύσεις\nΠαπαδόπουλος, Α. (2019). Τίτλος. Αθήνα: Εκδόσεις.\nSmith, J. (2020). A title. London: Press.\nBrown, K. (2021). A title. London: Press.\nJones, P. (2022). A title. London: Press.",
        );
        assert!(plain_cv
            .spans
            .iter()
            .all(|span| span.kind != StructureKind::Bibliography));

        let plain_notes = detect(
            "Σημειώσεις\nΠαπαδόπουλος, Α. (2019). Τίτλος. Αθήνα: Εκδόσεις.\nSmith, J. (2020). A title. London: Press.\nBrown, K. (2021). A title. London: Press.\nJones, P. (2022). A title. London: Press.",
        );
        assert!(plain_notes
            .spans
            .iter()
            .all(|span| span.kind != StructureKind::Bibliography));

        let cv_subsection = detect(
            "Δημοσιεύσεις\n## ΕΡΕΥΝΗΤΙΚΕΣ ΕΡΓΑΣΙΕΣ\nΠαπαδόπουλος, Α. (2019). Τίτλος. Αθήνα: Εκδόσεις.\nSmith, J. (2020). A title. London: Press.\nBrown, K. (2021). A title. London: Press.\nJones, P. (2022). A title. London: Press.",
        );
        assert!(cv_subsection
            .spans
            .iter()
            .all(|span| span.kind != StructureKind::Bibliography));
    }

    #[test]
    fn bare_number_footnote_stream_cannot_seed_headerless_bibliography() {
        let result = detect(
            "12 Smith, J. (2019). A title. London: Press.\n13 Brown, K. (2020). A title. London: Press.\n14 Jones, P. (2021). A title. London: Press.\n15 Miller, R. (2022). A title. London: Press.",
        );
        assert!(result
            .line_evidence
            .iter()
            .all(|line| line.reason_codes.contains(&ReasonCode::HardFootnoteStream)));
        assert!(result
            .spans
            .iter()
            .all(|span| span.kind != StructureKind::Bibliography));
    }

    #[test]
    fn isolated_reference_never_forms_a_block() {
        let result = detect("Παπαδόπουλος, Α. (2019). Τίτλος. Αθήνα: Εκδόσεις.");
        assert_eq!(result.line_evidence[0].bib_role, LineRole::StrongEntryStart);
        assert!(result.spans.is_empty());
    }

    #[test]
    fn json_contract_uses_stable_snake_case_names() {
        let result =
            detect("## ΠΕΡΙΕΧΟΜΕΝΑ\n1. Εισαγωγή ........ i\n2. Μέθοδος ........ ii\n## ΚΕΦΑΛΑΙΟ");
        let value = serde_json::to_value(result).unwrap();
        assert_eq!(value["model_id"], MODEL_ID);
        assert_eq!(value["decoder_id"], DECODER_ID);
        assert_eq!(value["spans"][0]["kind"], "toc");
        assert!(value["spans"][0]["supporting_lines"].is_array());
        assert!(value["spans"][0]["bridged_lines"].is_array());
        assert_eq!(value["line_evidence"][0]["toc_role"], "heading");
        assert!(value["line_evidence"][1]["reason_codes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason == "toc_page_roman"));
    }
}
