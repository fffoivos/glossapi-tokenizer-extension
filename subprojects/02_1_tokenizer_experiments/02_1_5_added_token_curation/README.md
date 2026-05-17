# 02_1_5 Added-token curation

Sub-subproject of `02_1_tokenizer_experiments`. **Stage 5 (terminal,
post-cutoff):**

```
[02_1_1 tokenizer training]
[02_1_2 cutoff variant builder]
[02_1_3 fertility evaluation]
[02_1_4 cutoff analysis]              → cutoff pick
       │
       ▼
[02_1_5 added-token curation]         → per-token keep/remove decision
       │
       ▼ (manifest)
[02_2_tokenizer_implementation]       → consumes the removal manifest
```

## Goal

After the cutoff is picked (`02_1_4`) but before the merge-rule
extension is implemented (`02_2`), curate the kept added units: of the
tokens the cutoff retains, which ones should the implementer **remove**
(or mask) because they are encoding / extraction artifacts that do not
represent content the model should learn or emit?

This stage is **policy-only** — it does not modify any tokenizer file.
It emits an implementation manifest for the merge-rule implementer to
consume.

## Inputs

- `02_1_4_cutoff_analysis/artifacts/classified_added_tokens.jsonl` —
  per-token char-language bucket
- `~/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/glossary/tokens_glossary.jsonl` —
  Gemini-pass per-token category / morphology / meaning

## Outputs (`artifacts/`)

- `removal_list.jsonl` — one row per token marked for removal
  (`id`, `decoded`, `category`, `lang_bucket`, `removal_class`,
  `meaning_snippet`)
- `keep_list.jsonl` — one row per kept added token
- `decision_summary.json` — counts per removal class + per-cutoff
  impact

## Policy

The full reasoning lives in [`CURATION_REPORT.md`](CURATION_REPORT.md).
Quick summary of the six removal classes (widened 2026-05-17 after
audit discussion):

1. **`latin1_utf8_mojibake`** — whole `category = "mojibake"`. Latin-1-
   as-UTF-8 mojibake (`ÉÉ`, `Ø`, `ØØ`, `ÉÉÉÉ`, `ØØØØ`, `Ô`). **6** at
   25k.
2. **`mixed_script_artifact`** — whole `category = "mixed_script_token"`.
   Greek-Latin lookalike font-sub mojibake (`τo`, `Tο`, `Tα`, `Oι`,
   `Ωστόσο`) + punct+Greek BPE-boundary fragments (`.Ε`, `,τι`,
   `/και`, …). **77** at 25k.
3. **`pdf_postscript_glyph`** — whole `category = "postscript_glyph"`.
   Slash-prefixed PDF font-glyph names (`/Α`, `/η`, `/pi`, …). **14**
   at 25k.
4. **`cleaner_linenewline_placeholder`** — whole `category =
   "code_identifier"`. Cleaner newline placeholders (`LINENEWLINE`,
   `NEWLINENEWLINE`). **2** at 25k.
5. **`cleaner_linenewline_bpe_fragment`** — `category =
   "latin_acronym"` AND decoded ∈ {`LIN`, `ENEW`, `LINENEW`}. BPE
   pieces of LINENEWLINE. **3** at 25k. (Genuine acronyms like `EURO`
   stay.)
6. **`cleaner_extraction_tag`** — `category = "latin_fragment"` AND
   decoded ∈ {`-missing`, `-decoded`}. Cleaner extraction-tag
   fragments. **2** at 25k. (Greek-surname transliteration fragments
   like `opoulou` stay.)

**Total: 104 tokens at the full 25,600 vocab; 39 tokens at the
recommended 11,264 cutoff.**

The KEEP decisions and reasoning (for `dingbat_or_symbol`, long
`escaped_character_run`, `url_or_path`, the genuine
`latin_acronym` / `latin_abbreviation` / `latin_word` rows that
the narrow LINENEW / extraction-tag filters do **not** strip, all
`punctuation_run`, and every Greek-payload category —
`greek_word`, `greek_fragment`, `greek_morpheme`, `proper_noun`,
`greek_acronym`) are spelled out in CURATION_REPORT.md §
"KEEP — and why".

## Outputs

- **`manifests/` — git-tracked, consumed by `02_2`:**
  - `removal_list.jsonl` — one row per token marked for removal
    (`id`, `decoded`, `category`, `lang_bucket`, `removal_class`,
    `meaning_snippet`)
  - `decision_summary.json` — counts per removal class + per-cutoff
    impact + the rule predicates that were applied
- **`artifacts/` — gitignored, regenerable:**
  - `keep_list.jsonl` — one row per kept added token (~25k rows;
    derivable from the glossary, so it does not need to be in git)

## Scripts

- [`scripts/emit_removal_list.py`](scripts/emit_removal_list.py) —
  applies the six removal rules to the glossary + classified set
  and writes the manifest + artifact files above. Idempotent and
  deterministic.

## What this is not

- It is not an implementation. The actual `tokenizer.json` /
  `model.merges` edit happens in `02_2_tokenizer_implementation`. See
  CURATION_REPORT.md § "Implementation handoff" for the two compatible
  implementation options (embedding-init masking vs build a pruned
  variant) and which to prefer at the recommended cutoff.
- It is not a cleaner fix. Every entry in the removal list points at a
  cleaner pattern that should be addressed in the next training arm so
  these tokens never become BPE candidates in the first place. See
  CURATION_REPORT.md § "Forward path — fix the cleaner".
