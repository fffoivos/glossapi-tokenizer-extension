# 01.0 Cleaning Iteration and Thresholds

## Status (2026-05-11)

**DONE for the C3 shipping path.** The wave-2 broad cleaner (the
generation that produced C3's training input) is settled. A parallel
per-line-badness scorer branch (`cleaner/per-line-badness-20260504`)
is in flight as a future improvement; it is decoupled from C3
shipping and would only matter for a future tokenizer arm.

See [../../docs/C3_CONVERGENCE.md](../../docs/C3_CONVERGENCE.md) for
the active scope. Historical status from 2026-04-23 below preserved
for context.

## Scope

Drive the iterative loop between `02_1_tokenizer_experiments` (which
surfaces residual bad tokens) and upstream `corpus.clean` in
`eellak/glossAPI` (where the fix lives). Produces the cleaner version +
threshold config that downstream subprojects `01_1_corpus_dedup` and
`01_2_training_dataset_mix` will consume.

The loop runs:
```
  tokenizer shows bad tokens  →  classify the noise  →  patch corpus.clean
                                                        (or just the
                                                         thresholds here)
                              ←  inspect output       ←  retrain tokenizer
```

Iteration does NOT require dedup — dedup is the final-validation gate
only. This subproject's critical path is cleaner ↔ tokenizer only.

## Already Decided

- upstream pre-existing per-doc quality signals (`greek_badness_score`,
  `mojibake_badness_score`, `needs_ocr`, `ocr_success`, `filter`,
  `quality_method`, `greek_percentage`, `latin_percentage`) are NEVER
  overwritten by this subproject's outputs — they remain as
  independent quality axes
- new quality signals are ADDED as separate columns:
  - `charset_greek_ratio`, `charset_moji_ratio`, `charset_punct_ratio`
  - `mojibake_noise_ratio = charset_moji_ratio + charset_punct_ratio`
    (additive, no weighting — per 2026-04-23)
  - `content_chars_kept`, `chars_dropped_by_{line_drop, normalization,
    per_char_filter}`, `lines_dropped_by_cleaner`,
    `marker_chars_{passthrough, added}`
- rejection thresholds are NOT hard-coded in upstream corpus.clean;
  they live in `THRESHOLDS.yaml` here and are consumed by the
  driver. This keeps iteration fast without upstream release churn.
- for iteration samples (pre-decision review), filenames are prefixed
  with zero-padded metric value so `ls` orders by metric value (see
  `feedback_metric_prefix_in_sample_filenames.md`).
- all Gemini API calls gated on explicit user approval (paid compute)
- all full-corpus cleaner runs gated on explicit user approval

## Outputs of this Subproject

- pinned cleaner version tag against `eellak/glossAPI`
- `THRESHOLDS.yaml` — all rejection cutoffs for charset / deletion /
  length / upstream-score axes
- enriched corpus parquets — the canonical corpus.clean outputs plus
  the new score columns, consumed by `01_1_corpus_dedup` and
  `01_2_training_dataset_mix`
- `CURRENT_STATUS.md` — which iteration we are on, which cutoffs are
  locked vs under review, what loops are pending

## Pointers

- cleaner crate (upstream, pinned): `eellak/glossAPI/rust/glossapi_rs_cleaner`
- cleaner crate changes map (this wave):
  `eellak/glossAPI/rust/glossapi_rs_cleaner/CHANGES_2026_04_22.md`
- charset analysis Rust module:
  `eellak/glossAPI/rust/glossapi_rs_cleaner/src/charset_module.rs`
- noise matcher speedup (write_files flag):
  `eellak/glossAPI/rust/glossapi_rs_noise/src/lib.rs`
- sample-review artifacts of the current iteration:
  `reports/2026-04-23_wave/` (when produced)

## Status (2026-04-23)

Current iteration: **wave 1 — cleaner refactor + charset filter v1**.
Thresholds are suggestive only; `THRESHOLDS.yaml` holds the *current
best guess* pending user review of the 500 × 500 deletion-band sample.
Full-canonical-corpus run gated on upstream noise-crate speedup being
merged, and on user approval.

## Files

- `README.md` — this file
- `TODO.md` — open action items for this iteration
- `CURRENT_STATUS.md` — what's settled / in-flight / blocked
- `THRESHOLDS.yaml` — the filter config
- `CLAUDE.md` — briefing for a fresh Claude session working on this subproject
- `scripts/` — iteration driver, samplers, analyzers, Gemini review
  harnesses (ported from the upstream cleaner repo's `cleaning_scripts/`)
- `reports/` — per-wave calibration results
- `eval/tokenizer_bad_token_inventory.md` — what tokens the tokenizer
  currently reveals as noise (drives the next iteration)
