//! The `bibliography_v2` regex set, compiled from Python's own pattern text.
//!
//! `patterns.json` is produced by `fixtures/dump_patterns.py`, which reads the
//! *compiled* `re.Pattern` objects out of the deployed module. That indirection is
//! deliberate. The patterns interpolate character classes that Python enumerates at
//! import time from `unicodedata` over the European script ranges — `_UPPER` is
//! 1,300-odd codepoints written out longhand — so any hand-written Rust equivalent
//! would be a guess about category tables that could drift between runtimes. Reusing
//! the exact source text makes the regex layer equivalent by construction, and the
//! embedded `unicodedata_version` records which table it was equivalent to.

use anyhow::{Context, Result};
use fancy_regex::Regex;
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashMap;

const PATTERNS_JSON: &str = include_str!("../patterns.json");

#[derive(Deserialize)]
struct PatternEntry {
    /// The dialect-adjusted source (named groups and backrefs rewritten).
    fancy: String,
    /// The original Python source, kept for auditing only.
    #[allow(dead_code)]
    python: String,
}

/// The frozen heading lexicons, dumped alongside the patterns. `analyze_bib_line`
/// returns HEADING or SUBHEADING only for exact members of the first two.
#[derive(Deserialize, Default)]
pub struct Lexicons {
    pub bib_headings: Vec<String>,
    pub bib_subheadings: Vec<String>,
    pub extra_bib_subheadings: Vec<String>,
}

#[derive(Deserialize)]
struct PatternFile {
    schema_version: String,
    rules_id: String,
    unicodedata_version: String,
    patterns: HashMap<String, PatternEntry>,
    #[serde(default)]
    lexicons: Lexicons,
}

pub struct Patterns {
    pub rules_id: String,
    pub unicodedata_version: String,
    pub lexicons: Lexicons,
    compiled: HashMap<String, Regex>,
}

impl Patterns {
    fn load() -> Result<Self> {
        let file: PatternFile =
            serde_json::from_str(PATTERNS_JSON).context("parsing patterns.json")?;
        anyhow::ensure!(
            file.schema_version == "bib-v2-patterns-v2",
            "unexpected patterns.json schema {}",
            file.schema_version
        );
        let mut compiled = HashMap::with_capacity(file.patterns.len());
        for (name, entry) in &file.patterns {
            let re = Regex::new(&entry.fancy)
                .with_context(|| format!("compiling {name}"))?;
            compiled.insert(name.clone(), re);
        }
        Ok(Self {
            rules_id: file.rules_id,
            unicodedata_version: file.unicodedata_version,
            lexicons: file.lexicons,
            compiled,
        })
    }

    #[inline]
    pub fn get(&self, name: &str) -> &Regex {
        self.compiled
            .get(name)
            .unwrap_or_else(|| panic!("pattern {name} missing from patterns.json"))
    }
}

pub static PATTERNS: Lazy<Patterns> =
    Lazy::new(|| Patterns::load().expect("patterns.json is embedded and must compile"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pattern_compiles() {
        // Lazy would otherwise hide a compile failure until first use.
        assert_eq!(
            PATTERNS.rules_id,
            "deterministic_bibliography_explicit_features_v2"
        );
        assert!(!PATTERNS.unicodedata_version.is_empty());
        for name in [
            "_YEAR",
            "_NUMERIC_DATE",
            "_INVERTED_AUTHOR",
            "_DIRECT_AUTHOR",
            "_ARTICLE_PAGE_RANGE",
            "_PAGE_RANGE",
            "_PROPER_WORD",
        ] {
            let _ = PATTERNS.get(name);
        }
    }

    #[test]
    fn backreference_dialect_survived_the_rewrite() {
        // _NUMERIC_DATE requires the two separators to be the same character;
        // `1.2-2020` must not match while `1.2.2020` must.
        let re = PATTERNS.get("_NUMERIC_DATE");
        assert!(re.is_match("04.12.2020").unwrap());
        assert!(!re.is_match("04.12-2020").unwrap());
    }

    #[test]
    fn lookbehind_dialect_survived_the_rewrite() {
        // _YEAR is `(?<!\d)…(?!\d)`: a year inside a longer digit run must not fire.
        let re = PATTERNS.get("_YEAR");
        assert!(re.is_match("in 1972.").unwrap());
        assert!(!re.is_match("119725").unwrap());
    }
}
