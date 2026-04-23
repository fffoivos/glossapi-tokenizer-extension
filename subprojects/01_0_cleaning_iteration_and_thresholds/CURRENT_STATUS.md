# Current Status — 01_0 Cleaning Iteration

## Active Wave

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
