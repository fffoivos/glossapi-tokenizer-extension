# Current Status — 01_0 Cleaning Iteration

## Active Wave

**wave 2 — pipeline cleanup landing** (2026-04-25 → 2026-04-26)

See `CLEANER_PIPELINE_CLEANUP_PLAN_2026-04-25.md` (this folder) and
`eellak/glossAPI/rust/glossapi_rs_cleaner/CHANGES_2026_04_25.md` for
the full record. Highlights:

- **Pilot B is the production Phase-A default**
  (`PhaseAMode::ParserSurgicalVerified` →
  `md_format_surgical::format_surgical_checked`). Parser-backed,
  preview-preserving. LineBased is regression-test-only.
- **Unified Rule B** owns all PostScript-glyph + PDF-font residue;
  four engines collapsed to two (Rule B + R1∪R2). No bare-word
  matchers (`GLYPH`, `hyphenminus` no longer trigger).
- **Per-char ops in 2 groups** — Group 1 STRIP + Group 2 FOLD.
  Adobe Symbol PUA + µ→μ folded via `fold_codepoint`; soft-hyphen
  strip via `is_unicode_noise_char`.
- **European-content preference.** KEEP Latin-1 + Latin-Ext-A +
  Cyrillic + Cyrillic Supp entirely; STRIP Latin-Ext-B except
  Romanian {Ș, ș, Ț, ț}; STRIP IPA / Latin-Ext-Additional / Coptic.
  R1∪R2 residue range narrowed to U+0180..U+024F minus Romanian
  to match.
- **Per-rule counters in `CleanStats`** —
  `rule_a_match_count`, `rule_b_match_count`, `residue_line_drop_count`.
  Production driver (`clean_and_stats_rowsharded.py`) sources parquet
  counter columns from these directly. Matcher (`glossapi_rs_noise`)
  no longer invoked per-row in production cleaning. Matcher PyO3
  surface preserved for `Corpus.clean_token_category_debug` and the
  `export_token_category_debug{,_parquet}.py` scripts.
- **`Corpus.clean` and `clean_text` share `build_script_char_sets`.**
  Both auto-add `punctuation`, `numbers`, `common_symbols` regardless
  of `scripts_to_keep`. Fixes a silent bug where the directory
  pipeline stripped ASCII punct + digits when callers passed
  restricted scripts.
- **`clean_text` PyO3 gains `phase_a_mode` arg** (parity with
  `clean_text_with_stats`).
- **`cmark_gfm_oracle::is_available` cached** via OnceLock — used to
  spawn `cmark-gfm --help` per doc. cmark-gfm is OPTIONAL dev tool;
  production uses in-process `dual_verify`. Do NOT install on prod
  hosts.
- **`\n{3+}` → `\n\n` post-loop normalize.** Lossless under markdown
  preview; cleans up runs left by per-char strip emptying adjacent
  PUA-bracket-glyph lines.
- **Bug fixes:**
  - `noise_metrics` token-category export now emits CHAR offsets
    (was BYTE offsets — Greek prefixes silently shifted slice
    boundaries).
  - `perf_mixed_doc_throughput_floor` marked `#[ignore]`
    (release-only; was failing every `cargo test`).
  - Empty-content `table_remover` edge case.

Branches:
- `eellak/glossAPI` `cleanup/cleaner-pipeline-20260425` — 2 commits
  (cleaner architecture + cleaning_scripts triage).
- `fffoivos/glossapi-tokenizer-extension`
  `codex/cleaner-iteration-subproject-20260423` — adds
  `train_bpe_from_text_shards.py` + `inspect_bpe_vocab_denoising.py`
  to `subprojects/02_1_tokenizer_experiments/scripts/` (the
  iteration-loop counterpart on the tokenizer side).

Test status: 374 cleaner unit tests pass + 10 noise tests pass +
30+ Python smoke checks pass. 0 build warnings.

100-doc end-to-end validation on
`openarchives.gr.part-00000.parquet`: 100/100 cleaned, 18.7% chars
removed, 25 docs/sec via Pilot B + in-process `dual_verify`. Gzipped
shards validated byte-identical to
`squash(clean_text_with_stats(raw, …))` — no hidden alteration.

## Earlier Wave

**wave 1 — cleaner refactor + charset filter v1** (2026-04-22 → 2026-04-23)

## What's Settled (this wave)

- Rule A span-strip (40 PS-glyph literals, `LeftmostLongest` AC mode)
- Rule B coverage predicate (`count ≥ 10 AND coverage ≥ 0.09`)
- Normalize AFTER cleaning (not before) — inline-TMC lines get normalized too
- Escape-aware `(?:\\_){4,}` → `---` separator
- Four-way per-doc char accounting (content_kept / line_drop / normalization /
  per_char_filter) emitted as separate columns
- `analyze_charset` Rust function — single-pass Unicode-block counts + ratios
- `LINE_REMOVED_COMMENT` marker for line-drop distinct from per-char strip
- Upstream noise-crate `write_files` flag eliminates matcher I/O contention
  (unblocks 56-worker full-corpus runs)
- Upstream pre-existing scores (`greek_badness_score`,
  `mojibake_badness_score`, `needs_ocr`, etc.) preserved, never overwritten

## What Exists Now

- cleaner patches committed on eellak/glossAPI branch
  `codex/three-counter-pipeline-20260421` (pushed, not yet PR'd)
- 168 k partial-corpus validation run produced distribution reports
  (/home/foivos/data/glossapi_work_cleaned_v3/charset_run/report/)
- 69-doc + 75-doc + 56-doc sub-agent reviews of the charset thresholds
  (calibration artifacts, informed THRESHOLDS.yaml provisional values)
- 1 000-doc sample (500 random <20% deletion + 500 random ≥20% deletion)
  at /home/foivos/data/glossapi_work_cleaned_v3/charset_run/deletion_band_500x500/
  — waiting on user review before threshold lands

## In Flight / Blocked

- **deletion % cutoff** (for `pct_chars_removed_non_empty`) — user review of
  1000-doc sample pending; no cutoff yet
- **content_chars_kept minimum** — under discussion, no decision
- **Greek-to-Greek codepage mojibake detector** (bigram plausibility) — gap
  identified in sub-agent reviews; upstream `greek_badness_score` catches
  the cases we've seen, so this is backstopped but not fully solved
- **Duplicated-letter font mojibake detector** — open gap; one case found
- **Full-canonical-corpus cleaner run** — blocked on (a) upstream noise-crate
  PR merge for the matcher speedup, (b) explicit user go-ahead for paid
  compute (50 M doc, estimated hours with fixed speed)

## Known Errors from Prior Runs (to avoid repeating)

- **Wrong corpus slice.** We cleaned `unified_corpus/data/*.parquet` (168 k
  docs — partial, pre-dedup, only 34% of openarchives, missing HPLT +
  finewiki + finepdfs + OpenSubtitles + greek_legal_code). Canonical input
  is `hf_release_publish_working/data/*.parquet` filtered by
  dedup `kept_docs.parquet` (49.4 M docs).
- **Conflated score with reject.** Our filter calculated ratios AND
  dropped above threshold inside the same code path. Proper separation:
  emit score columns from the cleaner, apply rejection from THRESHOLDS
  downstream.
- **Didn't preserve upstream badness scores.** Early driver versions
  overwrote or ignored upstream columns. Fixed: 2026-04-23 pass adds new
  columns as additions, never touches existing ones.
- **POC tokenizer trained on wrong data.** The 4 tokenizers at
  glossapi_work_cleaned_v2/.../final_4_tokenizers/ used partial glossapi
  + old-cleaner HPLT + pre-charset-filter. Not reference outputs.

## Next Decisions Needed

1. User review of 1000-doc sample → deletion-% cutoff
2. PR the cleaner branch into eellak/glossAPI main (today: just a
   pushed branch; next: open PR)
3. Approval for full-canonical-corpus cleaner run once speedup is live
4. Whether to add bigram-based Greek-to-Greek mojibake detector in this
   wave or defer to a wave 2
