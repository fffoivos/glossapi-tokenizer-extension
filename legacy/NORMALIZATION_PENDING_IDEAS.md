# Pending normalization ideas (parked for later review)

**Status**: not reviewed, not pre-committed. These came out of a mechanical
line-level audit of the existing cleaner (`rust/glossapi_rs_cleaner/src/cleaning_module.rs`,
`rust/glossapi_rs_common/src/lib.rs`, `src/glossapi/corpus/phase_clean.py`,
`src/glossapi/text_sanitize.py`, `src/glossapi/ocr/utils/cleaning.py`) and the
Apertus `tokenizer.json`. They are level-of-detail work items, not
pipeline-level redesigns. Review before promoting.

## Resolution (2026-04-20)

Items below mapped to statuses in
[`corpus_clean_normalization/NORMALIZATION_DESIGN_20260420.md`](corpus_clean_normalization/NORMALIZATION_DESIGN_20260420.md):

| # | Item | Status |
|---|---|---|
| 1 | Polytonic preservation accidental | **promoted** (design #13 — make explicit) |
| 2 | Dot-leader target `.....` vs `…` | **kept as-is** (design #4 — `.....`) |
| 3 | No separator/layout normalization beyond dots | **promoted** (design #2) |
| 4 | HTML entity decoding runs before artifact detection (URL corruption risk) | **deferred** (not a weird-token source) |
| 5 | Unicode whitespace variants not folded | **promoted** (design #8) |
| 6 | Markdown pipe-table rule rows not canonicalized | **promoted** (design #3, parser-based) |
| 7 | `<br>` / `<br/>` stripped, losing row-internal line boundaries | **deferred** |
| 8 | Table-removed / text-missing placeholder standardization | **deferred** |
| 9 | Ligatures | **promoted** (design #7) |
| 10 | Latin-lookalike Greek homoglyphs | **off-scope** (too complex, not dominant in evidence) |
| 11 | Zero-width / invisible characters | **already shipped** |
| 12 | PUA range | **already shipped** |
| 13 | Soft hyphens at line boundaries | **already shipped** (char-level) |
| 14 | End-of-line ASCII hyphen de-hyphenation | **off-scope** (not a weird-token source) |
| 15 | Repeated-char garbage | **deferred** (share from OCR module if needed) |
| 16 | Cleaner / rule-discovery pipeline disconnected | **resolved** (closure pilot = normalize patch in TODO) |
| 17 | No token-impact metric | **deferred** (retokenize is the test; per design-doc framing) |

## Ground-truth Apertus facts (verified empirically)

- `normalizer` is **null** — Apertus applies no Unicode normalization.
- Pre-tokenizer: GPT-2-style regex split → ByteLevel. Vocab 131,072.
- Single-token vocab hits: bare α, monotonic ά (U+03AC), final sigma ς,
  Greek semicolon ·, en/em-dash, ellipsis …, smart quotes " ", NBSP.
- **No single-token merges** for: oxia ά (U+1F71), psili+oxia ἄ (U+1F04),
  tonos+ypogegrammeni ᾴ (U+1FB4), combining acute U+0301, NFD-decomposed
  α+◌́.
- Implication: Apertus was trained on monotonic-NFC Greek. Any corpus-level
  transform that moves our text away from that form degrades alignment.

## Issues in the existing cleaner

1. **Polytonic preservation is accidental.** `greek` set in
   `cleaning_module.rs:42-48` excludes U+1F00–U+1FFF; `unusual_chars`
   (lines 82–89) also skips it. Polytonic chars survive only because they
   land in neither set. A future edit adding that range to `unusual` would
   silently delete all polytonic Greek. Make it explicit: add U+1F00–U+1FFF
   to `greek`.
2. **Dot-leader target is `.....` (5 dots), not `…`.**
   `normalize_layout_leader_runs` (`cleaning_module.rs:113`) collapses
   `\.{4,}` → literal 5 dots. Apertus has a single-token `…` (U+2026);
   collapsing to that saves bytes and tokens.
3. **No separator/layout normalization** beyond dots. Runs of
   `-`, `_`, `*`, `═`, em-dash, Markdown table rule variants (`---`,
   `:---`, `:---:`) untouched. PDF TOCs produce enormous variety.
4. **HTML-entity decoding runs before artifact detection**
   (`cleaning_module.rs:254`) and would corrupt inline URLs (`%20` →
   space). Consider protecting URL spans.
5. **Unicode whitespace variants not folded.** U+2007 figure space, U+2009
   thin space, U+202F narrow NBSP, U+FEFF BOM, U+200B ZWSP — pass through.
   Apertus doesn't have merges for the rare ones.

## Table / separator / structure normalization

6. **Markdown pipe-table rule rows** are not canonicalized. `|---|`,
   `|:---:|`, `| ─── |`, `|----|` all appear post-OCR and produce token
   variability. A canonical form per cell (`---`, `:---`, `---:`, `:---:`)
   would compress this.
7. **`<br>` / `<br/>` in table cells** gets dropped by `strip_tags_custom`,
   losing row-internal line boundaries. Map to sentinel before tag strip.
8. Table-removed / text-missing placeholders are used but no
   standardization pass ensures consistent surrounding whitespace.

## PDF-extraction noise / OCR artifacts

9. **Ligatures** (fi U+FB01, fl U+FB02, ffi, ffl) — Apertus has no
   dedicated tokens; decompose to ASCII pairs.
10. **Latin-lookalike Greek homoglyphs** inside Greek words (Α/A, Ο/O,
    Ρ/P, Μ/M, Ν/N, Τ/T, Κ/K, Χ/X, Η/H, Ε/E, Ζ/Z, Β/B, Ι/I). Detect via
    script majority at word level; policy is the open question.
11. **Zero-width / invisible characters** (ZWSP, ZWJ, ZWNJ, LRM, RLM, BOM,
    soft hyphen). Strip in Greek-only corpus.
12. **PUA range** (U+E000–U+F8FF) common PDF extraction garbage; not in
    `unusual` set, passes through.
13. **Soft hyphens at line boundaries** (U+00AD) — PDF column-wrap
    residue. Remove and rejoin.
14. **End-of-line ASCII hyphens** (`word-\nword`) — needs paragraph-level
    de-hyphenation; current cleaner is line-based.
15. **Repeated-char garbage** — the OCR module handles it at stream-decode;
    non-OCR extraction paths don't. Share the heuristic.

## Bigger-picture observations (relate to the iterative pipeline)

16. **Cleaner and rule-discovery pipeline are disconnected.** wave10_v5
    produced 1 cleaning + 1 normalization candidate, but nothing in
    `core_clean_text` consumes them. Phase 7 (library implementation) is
    not started. Until then, the discovery loop doesn't close.
17. **No token-impact metric.** Rule promotion in the discovery pipeline
    is measured by review agreement, not by bytes-per-token compression
    on a held-out sample. The thing that matters for tokenizer work isn't
    directly measured.

## Decision gate

For each item above, the call is one of:
- Promote into a concrete PR against the library cleaner.
- Feed into the rule-discovery pipeline as a new review category.
- Defer indefinitely.

The call requires agreement with the iterative-pipeline direction — not to
be made unilaterally.
