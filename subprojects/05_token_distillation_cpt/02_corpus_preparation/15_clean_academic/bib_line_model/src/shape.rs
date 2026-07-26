//! `line_shape` — 34 language-agnostic shape values per line.
//!
//! Port of `bibliography_role_features.py::line_shape`. Unlike the count features,
//! these values are *measured in characters*, so this module counts `char`s rather
//! than bytes throughout; `char_length` is `len(str)` in Python, which is a count of
//! code points.
//!
//! Every character predicate goes through `crate::unicode`, which carries Python's
//! own tables — the general categories, the `str` methods, and the GREEK/LATIN split
//! that Python performs by inspecting each character's *name*.
//!
//! Python builds the vector as float64 and then casts with
//! `np.asarray(values, dtype=np.float32)`, so the output is f32 and every ratio is
//! computed in f64 first. That order is preserved here.

use crate::patterns::PATTERNS;
use crate::unicode as u;

pub const LINE_SHAPE_NAMES: [&str; 34] = [
    "char_length",
    "log1p_char_length",
    "token_count",
    "log1p_token_count",
    "mean_token_length",
    "maximum_token_length",
    "leading_whitespace",
    "trailing_whitespace",
    "letter_fraction",
    "digit_fraction",
    "uppercase_fraction_of_letters",
    "lowercase_fraction_of_letters",
    "greek_fraction_of_letters",
    "latin_fraction_of_letters",
    "punctuation_fraction",
    "symbol_fraction",
    "whitespace_fraction",
    "other_fraction",
    "starts_lowercase",
    "starts_uppercase",
    "starts_digit",
    "starts_bullet_or_number",
    "ends_sentence_terminal",
    "ends_opening_terminal",
    "parenthesis_balance",
    "bracket_balance",
    "quote_parity",
    "is_blank",
    "is_repeated_rule",
    "is_table_rule",
    "is_page_number",
    "is_bullet_only",
    "has_html_fragment",
    "has_replacement_character",
];

pub const N_SHAPE: usize = LINE_SHAPE_NAMES.len();

/// `_SENTENCE_TERMINAL = frozenset(".!?;·")`.
///
/// Written as escapes because the last member is **U+0387 GREEK ANO TELEIA**, not
/// the visually identical U+00B7 MIDDLE DOT. That distinction is load-bearing in an
/// unobvious direction: `line_shape` NFKC-normalizes first, and NFKC maps U+0387 to
/// U+00B7, so this member can never match. Reproducing the dead entry is the point —
/// a "helpful" correction to U+00B7 would make `ends_sentence_terminal` fire on
/// Greek text where the reference implementation does not.
const SENTENCE_TERMINAL: [char; 5] = ['\u{2e}', '\u{21}', '\u{3f}', '\u{3b}', '\u{387}'];
/// `_OPENING_TERMINAL = frozenset(",:;-/–—([{'\"«“")`
const OPENING_TERMINAL: [char; 14] = [
    ',', ':', ';', '-', '/', '–', '—', '(', '[', '{', '\'', '"', '«', '\u{201c}',
];
/// The characters counted for `quote_parity`: `"'\"«»“”‘’"`.
const QUOTES: [char; 8] = [
    '\'', '"', '«', '»', '\u{201c}', '\u{201d}', '\u{2018}', '\u{2019}',
];

/// Whether a character is in `_SENTENCE_TERMINAL`; see the constant's note on why
/// the Greek member is U+0387 and can never match post-NFKC.
#[inline]
pub fn is_sentence_terminal(ch: char) -> bool {
    SENTENCE_TERMINAL.contains(&ch)
}

/// `_ratio` — zero when the denominator is zero, rather than NaN.
#[inline]
fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator != 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Python `str.strip()` — strips by `str.isspace()`, not by Rust's White_Space.
#[inline]
pub fn py_strip(text: &str) -> &str {
    text.trim_matches(|ch| u::py_isspace(ch))
}

#[inline]
fn py_lstrip(text: &str) -> &str {
    text.trim_start_matches(|ch| u::py_isspace(ch))
}

#[inline]
fn py_rstrip(text: &str) -> &str {
    text.trim_end_matches(|ch| u::py_isspace(ch))
}

#[inline]
fn anchored_match(name: &str, text: &str) -> bool {
    // Every pattern used this way is `^`-anchored, so Python's `.match()` (which
    // anchors at position 0 regardless) and a leftmost search agree.
    matches!(PATTERNS.get(name).find(text), Ok(Some(m)) if m.start() == 0)
}

#[inline]
fn searches(name: &str, text: &str) -> bool {
    PATTERNS.get(name).is_match(text).unwrap_or(false)
}

/// `line_shape(text)` — takes raw text and NFKC-normalizes, as Python does.
pub fn line_shape(text: &str) -> [f32; N_SHAPE] {
    line_shape_normalized(&crate::features::normalize(text))
}

/// The same, when the caller already holds the NFKC-normalized line.
pub fn line_shape_normalized(normalized: &str) -> [f32; N_SHAPE] {
    let stripped = py_strip(normalized);

    let token_lengths: Vec<usize> = PATTERNS
        .get("_TOKEN")
        .find_iter(normalized)
        .flatten()
        .map(|m| m.as_str().chars().count())
        .collect();

    let n = normalized.chars().count();
    let nf = n as f64;

    let (mut letters, mut greek, mut latin) = (0usize, 0usize, 0usize);
    let (mut digits, mut upper, mut lower) = (0usize, 0usize, 0usize);
    let (mut punctuation, mut symbols, mut spaces) = (0usize, 0usize, 0usize);
    let mut quote_count = 0usize;
    let mut open_paren = 0i64;
    let mut close_paren = 0i64;
    let mut open_bracket = 0i64;
    let mut close_bracket = 0i64;
    let mut has_replacement = false;

    for ch in normalized.chars() {
        // `_letters_by_script` gates the script test on category L, so a Greek
        // *symbol* or punctuation mark is not counted as a Greek letter.
        if u::is_cat_l(ch) {
            letters += 1;
            greek += u::is_greek_letter(ch) as usize;
            latin += u::is_latin_letter(ch) as usize;
        }
        digits += u::is_cat_n(ch) as usize;
        punctuation += u::is_cat_p(ch) as usize;
        symbols += u::is_cat_s(ch) as usize;
        upper += u::py_isupper(ch) as usize;
        lower += u::py_islower(ch) as usize;
        spaces += u::py_isspace(ch) as usize;
        quote_count += QUOTES.contains(&ch) as usize;
        match ch {
            '(' => open_paren += 1,
            ')' => close_paren += 1,
            '[' => open_bracket += 1,
            ']' => close_bracket += 1,
            '\u{fffd}' => has_replacement = true,
            _ => {}
        }
    }
    let accounted = letters + digits + punctuation + symbols + spaces;

    let first = stripped.chars().next();
    let last = stripped.chars().next_back();
    let starts_list = anchored_match("ROLE_NUMBERED_HEADING", stripped)
        || anchored_match("ROLE_BULLET_PREFIX", normalized);

    let mean_token_length = if token_lengths.is_empty() {
        0.0
    } else {
        token_lengths.iter().sum::<usize>() as f64 / token_lengths.len() as f64
    };

    let values: [f64; N_SHAPE] = [
        nf,
        nf.ln_1p(),
        token_lengths.len() as f64,
        (token_lengths.len() as f64).ln_1p(),
        mean_token_length,
        token_lengths.iter().copied().max().unwrap_or(0) as f64,
        (n - py_lstrip(normalized).chars().count()) as f64,
        (n - py_rstrip(normalized).chars().count()) as f64,
        ratio(letters as f64, nf),
        ratio(digits as f64, nf),
        ratio(upper as f64, letters as f64),
        ratio(lower as f64, letters as f64),
        ratio(greek as f64, letters as f64),
        ratio(latin as f64, letters as f64),
        ratio(punctuation as f64, nf),
        ratio(symbols as f64, nf),
        ratio(spaces as f64, nf),
        ratio(n.saturating_sub(accounted) as f64, nf),
        first.map_or(false, |c| u::py_islower(c)) as u8 as f64,
        first.map_or(false, |c| u::py_isupper(c)) as u8 as f64,
        first.map_or(false, |c| u::py_isdigit(c)) as u8 as f64,
        starts_list as u8 as f64,
        last.map_or(false, |c| SENTENCE_TERMINAL.contains(&c)) as u8 as f64,
        last.map_or(false, |c| OPENING_TERMINAL.contains(&c)) as u8 as f64,
        (open_paren - close_paren) as f64,
        (open_bracket - close_bracket) as f64,
        (quote_count % 2) as f64,
        stripped.is_empty() as u8 as f64,
        anchored_match("ROLE_REPEATED_RULE", normalized) as u8 as f64,
        anchored_match("ROLE_TABLE_RULE", normalized) as u8 as f64,
        anchored_match("ROLE_PAGE_NUMBER", normalized) as u8 as f64,
        anchored_match("ROLE_BULLET_ONLY", normalized) as u8 as f64,
        searches("ROLE_HTML_FRAGMENT", normalized) as u8 as f64,
        has_replacement as u8 as f64,
    ];

    let mut out = [0f32; N_SHAPE];
    for (i, v) in values.iter().enumerate() {
        out[i] = *v as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape_of(text: &str) -> std::collections::HashMap<&'static str, f32> {
        let v = line_shape(text);
        LINE_SHAPE_NAMES.iter().copied().zip(v).collect()
    }

    #[test]
    fn blank_line_has_no_nans() {
        let v = line_shape("");
        assert!(v.iter().all(|x| x.is_finite()));
        let s = shape_of("");
        assert_eq!(s["is_blank"], 1.0);
        // Every ratio divides by zero here; Python's `_ratio` yields 0.0, not NaN.
        assert_eq!(s["letter_fraction"], 0.0);
        assert_eq!(s["uppercase_fraction_of_letters"], 0.0);
    }

    #[test]
    fn script_fractions_split_greek_and_latin() {
        let s = shape_of("Αθήνα Athens");
        assert_eq!(s["greek_fraction_of_letters"], 5.0 / 11.0);
        assert_eq!(s["latin_fraction_of_letters"], 6.0 / 11.0);
    }

    #[test]
    fn char_length_counts_codepoints_not_bytes() {
        // Six Greek characters, twelve UTF-8 bytes.
        assert_eq!(shape_of("Αθήνας")["char_length"], 6.0);
    }

    #[test]
    fn balances_and_parity() {
        let s = shape_of("(a [b] «c»");
        assert_eq!(s["parenthesis_balance"], 1.0);
        assert_eq!(s["bracket_balance"], 0.0);
        assert_eq!(s["quote_parity"], 0.0);
    }
}
