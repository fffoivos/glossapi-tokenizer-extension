//! reference_signals — the regex / label inventory + codepoint constants for the
//! Greek academic reference/citation detector.
//!
//! Grounded in the multi-agent investigation (see ../investigation/): the Greek
//! academic corpus carries citations in FOUR co-existing regimes (footnote stream,
//! ano-teleia–separated footnote lists, parenthetical author-year, bracket numeric)
//! plus an end-matter bibliography and, on the section-parquet sources, a `β` label.
//!
//! Design rules encoded here:
//!  - SEPARATE families, never a weighted aggregate.
//!  - Match the ano-teleia separator by CODEPOINT (U+0387 ano teleia AND U+00B7
//!    middle dot are equivalent roles but distinct codepoints — never NFC/NFKC-fold).
//!  - The `regex` crate has no look-around; every pattern below is look-around-free.
//!  - Header matching folds accents + lowercases + maps a small Latin→Greek homoglyph
//!    set (handles OCR'd `Bιβλιογραφία` with a Latin capital B).

use once_cell::sync::Lazy;
use regex::Regex;
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};

// ---------------------------------------------------------------------------
// Codepoint constants
// ---------------------------------------------------------------------------
pub const ANO_TELEIA: char = '\u{0387}'; // GREEK ANO TELEIA
pub const MIDDLE_DOT: char = '\u{00B7}'; // MIDDLE DOT (Latin-1) — equivalent role, distinct codepoint

// ---------------------------------------------------------------------------
// Accent folding (for header-stem matching)
// ---------------------------------------------------------------------------
fn fold_char(c: char) -> char {
    match c {
        'ά' | 'ὰ' | 'ᾶ' | 'ἀ' | 'ἁ' | 'ἄ' | 'ἂ' => 'α',
        'έ' | 'ὲ' | 'ἐ' | 'ἑ' | 'ἔ' | 'ἒ' => 'ε',
        'ή' | 'ὴ' | 'ῆ' | 'ἠ' | 'ἡ' | 'ἤ' | 'ἢ' => 'η',
        'ί' | 'ὶ' | 'ῖ' | 'ἰ' | 'ἱ' | 'ἴ' | 'ἲ' | 'ϊ' | 'ΐ' => 'ι',
        'ό' | 'ὸ' | 'ὀ' | 'ὁ' | 'ὄ' | 'ὂ' => 'ο',
        'ύ' | 'ὺ' | 'ῦ' | 'ὐ' | 'ὑ' | 'ὔ' | 'ὒ' | 'ϋ' | 'ΰ' => 'υ',
        'ώ' | 'ὼ' | 'ῶ' | 'ὠ' | 'ὡ' | 'ὤ' | 'ὢ' => 'ω',
        'ς' => 'σ',
        _ => c,
    }
}

/// Canonically decompose, strip Greek/Unicode combining marks, lowercase and
/// fold final sigma. This covers capital and adscript polytonic forms as well
/// as the explicitly listed common monotonic/precomposed forms.
pub fn fold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for decomposed in s.nfd().filter(|c| !is_combining_mark(*c)) {
        for lc in decomposed.to_lowercase() {
            out.push(fold_char(lc));
        }
    }
    out
}

/// Map a small set of Latin capital homoglyphs to their Greek look-alikes, then `fold`.
/// Applied ONLY to header-candidate lines (low risk, fixes OCR'd `Bιβλιογραφία`).
pub fn fold_header(s: &str) -> String {
    let mut pre = String::with_capacity(s.len());
    for c in s.chars() {
        let g = match c {
            'A' => 'Α', 'B' => 'Β', 'E' => 'Ε', 'H' => 'Η', 'I' => 'Ι', 'K' => 'Κ',
            'M' => 'Μ', 'N' => 'Ν', 'O' => 'Ο', 'P' => 'Ρ', 'T' => 'Τ', 'X' => 'Χ',
            'Y' => 'Υ', 'Z' => 'Ζ',
            _ => c,
        };
        pre.push(g);
    }
    fold(&pre)
}

// ---------------------------------------------------------------------------
// Header families (matched as folded substrings, NOT regex)
// ---------------------------------------------------------------------------

/// Stems that mark a reference / bibliography / sources section header.
pub const BIB_HEADER_STEMS: &[&str] = &[
    "βιβλιογραφ", "αναφορ", "πηγ", "ξενογλωσσ", "ελληνογλωσσ",
    "δικτυογραφ", "ιστοσελιδ", "παραπομπ", "references", "bibliograph", "works cited",
];

/// Folded substrings that DENY a header from being a bibliography boundary because it is
/// actually a body chapter (literature review) — `ανασκόπηση` / `επισκόπηση`.
pub const LITREVIEW_DENY: &[&str] = &["ανασκοπησ", "επισκοπησ"];

/// Folded substrings that DENY a `β`/header as the colophon citation page (front matter).
pub const COLOPHON_DENY: &[&str] = &["βιβλιογραφικη αναφορα", "καλλιπος"];

/// CV / front-matter section header family (Pergamos front-loaded `β` false positives).
pub const CV_HEADER_FAMILY: &[&str] = &[
    "βιογραφικο σημειωμα", "προσωπικα στοιχεια", "επαγγελματικη εμπειρια",
    "μεταπτυχιακη εκπαιδευση", "μετεκπαιδευση", "σπουδες", "εκπαιδευση",
    "ξενες γλωσσες", "βραβεια", "δημοσιευσεις", "curriculum vitae",
];

/// Kallipos non-bibliography `β` false-positive header family (textbook apparatus).
pub const KALLIPOS_FP_DENY: &[&str] = &[
    "προαπαιτουμενη γνωση", "λεξεις-κλειδια", "λεξεις κλειδια", "απαντηση", "λυση",
    "κριτηρι", "δραστηριοτητες", "συνοψη", "διαγνωση", "abstract", "περιληψη", "παραρτημα",
];

pub fn contains_any(folded: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| folded.contains(n))
}

// ---------------------------------------------------------------------------
// Line-level regexes (look-around-free)
// ---------------------------------------------------------------------------

/// ATX markdown header line, e.g. `## ΒΙΒΛΙΟΓΡΑΦΙΑ`.
pub static ATX_HEADER: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[ \t]*#{1,6}[ \t]+").unwrap());

/// Footnote line: leading 1–4 digit integer then whitespace (OCR'd footnote anchor).
pub static FOOTNOTE_LINE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[ \t]*\d{1,4}[ \t]+\S").unwrap());

/// TOC / dot-leader context — rejects a header that is really a table-of-contents row.
pub static TOC_CONTEXT: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\(σελ|\.{4,}|_{4,}|\d+\s*[-–]\s*\d+\s*\|?\s*$").unwrap());

/// Publication year (1500–2099).
pub static YEAR: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(1[5-9]\d{2}|20\d{2})\b").unwrap());

/// Page range, e.g. `123-145`.
pub static PAGE_RANGE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\d+\s*[-–]\s*\d+\b").unwrap());

/// Discursive footnote / citation cues (case-insensitive; applied to folded text).
pub static CUE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(βλ|ο\.?π|στο ιδιο|πρβλ|σημ|επιμ|ενδεικτικα|μτφρ|op\.? ?cit|loc\.? ?cit|ibid)\b")
        .unwrap()
});

/// In-text parenthetical author-year, e.g. `(Smith 2020)` / `(Παπαδόπουλος, 2011)`.
pub static INTEXT_PAREN_AY: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\([Α-ΩA-ZΆΈΉΊΌΎΏ][^()]{0,60}\s(1[5-9]|20)\d{2}\)").unwrap());

/// In-text bracket numeric (Vancouver), e.g. `[23]`.
pub static INTEXT_BRACKET: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[\d{1,3}\]").unwrap());

/// In-text parenthetical numeric (medical), e.g. `(4)` / `(4,5)`.
pub static INTEXT_PAREN_NUM: Lazy<Regex> = Lazy::new(|| Regex::new(r"\(\d{1,3}(,\d{1,3})*\)").unwrap());

/// Internal cross-reference that must be KEPT (not bibliographic), e.g. `βλ. Θεώρημα 1.28`.
pub static INTERNAL_XREF_KEEP: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)βλ\.?\s+(θεωρημα|παραδειγμα|σχημα|εξισωση|πινακα|ενοτητα|κεφαλαιο|εικονα)")
        .unwrap()
});

/// Latin author-title run: ≥2 capitalized Latin tokens joined by `, & .` — a citation signal.
pub static LATIN_AUTHOR_RUN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[A-Z][a-z]+([ ,.&]+[A-Z][a-z]+){1,}").unwrap());

// --- features for the deployed β-gate logistic regression (see beta_gate_model.rs) -----------
/// Capitalised Greek/Latin word (≥3 chars) — a name/title token.
pub static NAME_TOKEN: Lazy<Regex> = Lazy::new(|| Regex::new(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]{2,}").unwrap());
/// Author entry head: `Surname, Firstname/Initial` (Greek convention).
pub static AUTHOR_GR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]+\s*,\s*[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ.]+").unwrap());
/// Year in parentheses, e.g. `(2017)` — author-date signature.
pub static YEAR_PAREN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\((1[5-9]\d{2}|20\d{2})\)").unwrap());
/// Entry page/pages signal: `123-145`, `σ. 12`, `pp. 3`.
pub static PAGE_ENTRY: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b\d+\s*[-–]\s*\d+\b|σσ?\.\s*\d|pp?\.\s*\d").unwrap());
/// `Place: Publisher` imprint.
pub static PLACE_COLON: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώ]+\s*:\s*[Α-ΩA-ZΆ-Ώ]").unwrap());
/// CV / own-works header stems (folded) — negative signal.
pub const CV_STEMS: &[&str] = &["δημοσιευσ", "ανακοινωσ", "βιογραφικο", "σπουδες", "εκπαιδευσ", "εμπειρια", "σεμιναρ", "συνεδρι"];
/// Bibliographic locator abbreviations (substring).
pub const ENTRY_LOC: &[&str] = &["σσ.", "σ.", "τ.", "εκδ.", "επιμ.", "vol.", "pp.", "eds.", "et al"];

// ---------------------------------------------------------------------------
// Script-ratio helpers
// ---------------------------------------------------------------------------
fn is_greek(c: char) -> bool {
    matches!(c, '\u{0386}'..='\u{03CE}' | '\u{1F00}'..='\u{1FFF}')
}
fn is_latin(c: char) -> bool {
    c.is_ascii_alphabetic()
}

/// Fraction of letters that are Greek (0.0 if no letters).
pub fn greek_letter_fraction(s: &str) -> f32 {
    let (mut g, mut l) = (0u32, 0u32);
    for c in s.chars() {
        if is_greek(c) {
            g += 1;
        } else if is_latin(c) {
            l += 1;
        }
    }
    let tot = g + l;
    if tot == 0 {
        0.0
    } else {
        g as f32 / tot as f32
    }
}

/// Fraction of letters that are Latin (0.0 if no letters).
pub fn latin_letter_fraction(s: &str) -> f32 {
    let (mut g, mut l) = (0u32, 0u32);
    for c in s.chars() {
        if is_greek(c) {
            g += 1;
        } else if is_latin(c) {
            l += 1;
        }
    }
    let tot = g + l;
    if tot == 0 {
        0.0
    } else {
        l as f32 / tot as f32
    }
}

/// Count the two ano-teleia / middle-dot separators independently (returned as a pair).
pub fn anoteleia_counts(s: &str) -> (u32, u32) {
    let (mut a, mut b) = (0u32, 0u32);
    for c in s.chars() {
        if c == ANO_TELEIA {
            a += 1;
        } else if c == MIDDLE_DOT {
            b += 1;
        }
    }
    (a, b)
}
