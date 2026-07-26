//! Python's Unicode predicates, from Python's own tables.
//!
//! `line_shape` classifies every character of every line by general category, by
//! `str` predicate, and — for the Greek/Latin split — by whether its *character
//! name* contains "GREEK" or "LATIN". Rust's standard library answers the first two
//! almost the same way and cannot answer the third at all:
//!
//! * `char::is_alphabetic` is the Alphabetic property, not category L (it accepts
//!   U+0345, category Mn).
//! * `char::is_numeric` is Nd|Nl|No; `str.isdigit()` is Nd plus digit-valued No.
//! * there is no character-name table in std.
//!
//! `fixtures/dump_unicode_tables.py` run-length encodes each predicate straight out
//! of the reference interpreter, so these answers are Python's by construction and
//! carry the Unicode version they were taken from. Lookup is a binary search over
//! sorted, disjoint ranges.

use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashMap;

const TABLES_JSON: &str = include_str!("../unicode_tables.json");

#[derive(Deserialize)]
struct TableFile {
    schema_version: String,
    unicodedata_version: String,
    tables: HashMap<String, Vec<(u32, u32)>>,
    /// Codepoint -> `str.casefold()` result, only where it differs. Keys arrive as
    /// decimal strings because JSON object keys are strings.
    #[serde(default)]
    casefold: HashMap<String, String>,
}

/// One predicate as sorted disjoint `[lo, hi]` codepoint ranges.
pub struct RangeSet(Vec<(u32, u32)>);

impl RangeSet {
    #[inline]
    pub fn contains(&self, ch: char) -> bool {
        let cp = ch as u32;
        self.0
            .binary_search_by(|&(lo, hi)| {
                if cp < lo {
                    std::cmp::Ordering::Greater
                } else if cp > hi {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
    }
}

pub struct Tables {
    pub unicodedata_version: String,
    pub cat_l: RangeSet,
    pub cat_n: RangeSet,
    pub cat_p: RangeSet,
    pub cat_s: RangeSet,
    pub name_greek: RangeSet,
    pub name_latin: RangeSet,
    pub isupper: RangeSet,
    pub islower: RangeSet,
    pub isspace: RangeSet,
    pub isdigit: RangeSet,
    pub isalpha: RangeSet,
    pub isalnum: RangeSet,
    /// Non-zero canonical combining class -- the characters `_fold` discards.
    pub combining: RangeSet,
    pub casefold: HashMap<u32, String>,
}

impl Tables {
    fn load() -> Result<Self> {
        let mut file: TableFile =
            serde_json::from_str(TABLES_JSON).context("parsing unicode_tables.json")?;
        anyhow::ensure!(
            file.schema_version == "bib-unicode-tables-v2",
            "unexpected unicode_tables.json schema {}",
            file.schema_version
        );
        let mut take = |name: &str| -> Result<RangeSet> {
            let ranges = file
                .tables
                .remove(name)
                .with_context(|| format!("table {name} missing"))?;
            Ok(RangeSet(ranges))
        };
        let casefold = file
            .casefold
            .iter()
            .map(|(k, v)| Ok((k.parse::<u32>()?, v.clone())))
            .collect::<Result<HashMap<u32, String>>>()?;
        Ok(Self {
            cat_l: take("cat_L")?,
            cat_n: take("cat_N")?,
            cat_p: take("cat_P")?,
            cat_s: take("cat_S")?,
            name_greek: take("name_greek")?,
            name_latin: take("name_latin")?,
            isupper: take("isupper")?,
            islower: take("islower")?,
            isspace: take("isspace")?,
            isdigit: take("isdigit")?,
            isalpha: take("isalpha")?,
            isalnum: take("isalnum")?,
            combining: take("combining")?,
            casefold,
            unicodedata_version: file.unicodedata_version,
        })
    }
}

pub static TABLES: Lazy<Tables> =
    Lazy::new(|| Tables::load().expect("unicode_tables.json is embedded and must parse"));

// Thin named wrappers, so call sites read like the Python they came from.
#[inline]
pub fn is_cat_l(ch: char) -> bool {
    TABLES.cat_l.contains(ch)
}
#[inline]
pub fn is_cat_n(ch: char) -> bool {
    TABLES.cat_n.contains(ch)
}
#[inline]
pub fn is_cat_p(ch: char) -> bool {
    TABLES.cat_p.contains(ch)
}
#[inline]
pub fn is_cat_s(ch: char) -> bool {
    TABLES.cat_s.contains(ch)
}
#[inline]
pub fn is_greek_letter(ch: char) -> bool {
    TABLES.name_greek.contains(ch)
}
#[inline]
pub fn is_latin_letter(ch: char) -> bool {
    TABLES.name_latin.contains(ch)
}
#[inline]
pub fn py_isupper(ch: char) -> bool {
    TABLES.isupper.contains(ch)
}
#[inline]
pub fn py_islower(ch: char) -> bool {
    TABLES.islower.contains(ch)
}
#[inline]
pub fn py_isspace(ch: char) -> bool {
    TABLES.isspace.contains(ch)
}
#[inline]
pub fn py_isdigit(ch: char) -> bool {
    TABLES.isdigit.contains(ch)
}
#[inline]
pub fn py_isalpha(ch: char) -> bool {
    TABLES.isalpha.contains(ch)
}
#[inline]
pub fn py_isalnum(ch: char) -> bool {
    TABLES.isalnum.contains(ch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_load() {
        assert!(!TABLES.unicodedata_version.is_empty());
    }

    #[test]
    fn categories_match_python() {
        assert!(is_cat_l('α') && is_cat_l('Z'));
        assert!(!is_cat_l('1') && !is_cat_l(' '));
        assert!(is_cat_n('1') && is_cat_n('½'));
        assert!(is_cat_p('.') && is_cat_p('«'));
        assert!(is_cat_s('+') && !is_cat_s('.'));
    }

    #[test]
    fn script_split_is_by_character_name() {
        assert!(is_greek_letter('α') && is_greek_letter('Ω'));
        assert!(is_latin_letter('a') && is_latin_letter('É'));
        // Cyrillic is neither, and must not be silently counted as Latin.
        assert!(!is_greek_letter('д') && !is_latin_letter('д'));
    }

    #[test]
    fn digit_predicate_is_narrower_than_is_numeric() {
        // U+2168 ROMAN NUMERAL NINE is Nl: `char::is_numeric` accepts it,
        // Python's `str.isdigit()` does not. This is the distinction the table
        // exists to preserve.
        assert!('\u{2168}'.is_numeric());
        assert!(!py_isdigit('\u{2168}'));
        assert!(py_isdigit('7'));
    }

    #[test]
    fn isspace_covers_the_ascii_separators() {
        // Python's str.isspace() accepts \x1c-\x1f; Rust's is_whitespace does not.
        assert!(py_isspace('\u{1c}'));
        assert!(!'\u{1c}'.is_whitespace());
    }
}
