# 02.2.4 — Language-category promotion

> **In one line:** the spec directory for turning per-token evidence into "the canonical token set of language L"; the ambitious two-method design was never built, but one concrete cut of it — multi-language PMI promotion — was specified here, implemented next door, and became the artifact the whole downstream program consumed.
> **Period:** 2026-05-15 → 2026-05-15 (commits `719d3834`, `0bbd93de`). **Status:** partially executed — the PMI pass shipped (as [`../02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/`](../02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/)); the `categories/<L>.jsonl` artifact and the F-vs-W comparison harness were never built.
> **Came from / led to:** [`02_2_1`](../02_2_1_char_language_membership/) + [`02_2_2`](../02_2_2_vocab_lang_attribution/) (+ the unbuilt [`02_2_3`](../02_2_3_token_classification/)) → this → `03_1_greek_embedding_diagnostic`, [`02_1_4_cutoff_analysis`](../../02_1_tokenizer_experiments/02_1_4_cutoff_analysis/), [`02_1_7_intrinsic_eval_sweep`](../../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/)

## Why this existed

The Greek embedding diagnostic in `03_1` ran seventeen scripts over a single hand-made file, `base_greek_tokens.jsonl` (1,494 strict-Greek ids), and its README blocked the non-Greek version of the analysis on "the user's per-language attribution". This directory was supposed to supply that: for every language worth analysing, a defensible, provenance-carrying list of token ids, in a uniform schema the diagnostic could drop in unchanged. The hard part was never the plumbing — it was deciding **what makes a token belong to a language** when English has no character that is exclusively English.

## History

### 2026-05-15 (early) — the design: two rival methods, decided empirically

[`PLAN.md`](PLAN.md) and [`METHODOLOGY.md`](METHODOLOGY.md) both landed in commit `719d3834`, marked "**proposal — for review before implementation**".

`PLAN.md` framed promotion by **per-language regime**: Regime A (strong T0 — Greek ~1,500 char-evidenced tokens, German 103 `ß`-bearing), Regime B (empty T0 — English and most Latin locales, where everything rests on a rate test), Regime C (aggregate-only — Cyrillic, CJK). Its proposed test was a Beta-Binomial 95 % CI on the log-ratio against the strongest sister language, at `min_count = 100`, `δ = 0.5`, with a sanity floor of ≥ 90 % mass coverage per language.

`METHODOLOGY.md` refused to pick a method a priori. It defined **Method F** (distribution-filtering — a discrete set; variants F1 count-threshold, F2 PMI, F3 CI-on-log-ratio, F4 top-K) against **Method W** (distribution-weighting — a continuous per-token weight; W1 rate, W2 sublinear power, W3 log, W4 PMI), and specified a comparison harness (`scripts/compare_F_vs_W.py`) that would run each 03_1 diagnostic under each variant with 100-resample bootstrap SEs and ship **the winner per diagnostic, not a universal default**. Two diagnostics (hull occupancy, infiltrators) were noted as requiring a discrete set by construction.

### 2026-05-15 (later) — the one cut that was actually built

[`PMI_PROMOTION_SPEC.md`](PMI_PROMOTION_SPEC.md) landed in commit `0bbd93de` as the build-spec for "the **first concrete promotion pass**": Method W4 (PMI) used as an F2-style filter, with the char mask toggled on and off so the two could be compared.

- Knobs: `α = 0.5`, `δ = 1.0` (≥ 10× over-represented), `min_count = 100`, marginal over the 87 canonical keys with ≥ 1 B firings, log base 10.
- Three outputs per key: **Variant A (masked)** = rate test + char-admissibility, **Variant B (unmasked)** = rate test alone, **Variant Δ** = `B \ A`, the audit trail of what the mask removed.
- Reasoning for PMI over the alternatives, recorded in the parent checkpoint: pairwise log-ratio does not scale past two languages; max-pooling shrinks monotonically as scope grows, so distinctive German tokens get killed by Dutch/Scandinavian co-firing; PMI scores each language independently against the corpus marginal and therefore **scales with the number of in-scope languages** ([`../CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md`](../CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md) § 3).
- The spec placed the implementation and its outputs **in `02_2_2`**, not here: `02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/`. That decision is why this directory ends up holding only specs.

The same commit made a small but telling edit to `METHODOLOGY.md`: the tier definitions were changed from hardcoded `popcount == 55` to `popcount == N_LANG_BITS` read from the char-tool manifest — the char tool had outgrown 55 language bits during the session (commit `0bbd93de`).

### What the spec itself declared out of the pass

`PMI_PROMOTION_SPEC.md` § "What this pass does *not* produce" is explicit: no Beta-Binomial CIs (the F3 variant), no W1/W2/W3 weights, no cross-source consistency check between the two English samples, and no promotion for the 1,842 non-cap-hit keys. A follow-up sweep of `δ ∈ {0.5, 1.0, 1.5, 2.0}` × `min_count ∈ {10, 100, 1000}` was named as the natural next step. None of it ran.

## Outcome

- **Built:** the PMI pass, per this directory's spec, at [`../02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/`](../02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/) — 87 keys × 3 variants = 261 committed token-set files, plus `summary.tsv`, `overlap_matrix.tsv`, `uncovered_tokens.tsv`, and a `manifest.json` stamped `built_at 2026-05-15T15:31:54Z` with `alpha 0.5 / delta 1.0 / min_count 100 / marginal_floor 1e9`, `n_marginal_keys 87`, `marginal_total 87,843,958,530`.
- **Not built:** `artifacts/categories/<L>.jsonl`, `artifacts/groups.json`, `scripts/promote_categories.py`, `scripts/validate.py`, `scripts/compare_F_vs_W.py`, the W-variant weight tables, and the per-diagnostic F-vs-W decision. This directory contains three Markdown files and nothing else.
- **The legacy interface was never migrated the way the plan described.** `PLAN.md` proposed symlinking `base_greek_tokens.jsonl` to `categories/Greek.jsonl` and re-verifying the Greek-vs-¬Greek classifier at macro-F1 ≈ 0.99. Instead `03_1` wrote its own loaders that parse the PMI `__masked.txt` tables directly (`build_groups_perlang_v3.py` for 11 languages, `build_groups_88lang_v4.py` for the 75 with non-empty masked sets), so the uniform-schema goal was bypassed rather than met.
- **The outputs were nonetheless the program's most-reused artifact.** `summary.tsv` anchors the "comparable-language vocab footprint" table in [`../../02_1_tokenizer_experiments/02_1_4_cutoff_analysis/REPORT.md`](../../02_1_tokenizer_experiments/02_1_4_cutoff_analysis/REPORT.md) § 1 (English 19,009 PMI tokens at 47.4 % mass; FineWeb-HQ English 19,339; French 9,694 at 58.7 %), is cited by [`../../02_1_tokenizer_experiments/02_1_6_representation_policy_analysis/README.md`](../../02_1_tokenizer_experiments/02_1_6_representation_policy_analysis/README.md), and selects the evaluation languages in `02_1_7_intrinsic_eval_sweep/scripts/02_prep_eval_configs.py` (keys with `masked_count > 100`, capped at 55).
- **Left open:** the F-vs-W question this directory was written to answer. `METHODOLOGY.md` § "Open issues" still carries all six, including "whether the canonical artifact is F or W".

## Where things are

| Artifact | Path | Note |
| --- | --- | --- |
| The shipped promotion pass | [`../02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/`](../02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/) | `build.py`, 261 table files, `summary.tsv`, `manifest.json` |
| Its build spec | [`PMI_PROMOTION_SPEC.md`](PMI_PROMOTION_SPEC.md) | knobs, algorithm steps A–E, output layout, sanity assertions |
| Method rationale | [`METHODOLOGY.md`](METHODOLOGY.md) | F vs W, per-diagnostic comparison table, bootstrap protocol |
| Regime framing + consumer contract | [`PLAN.md`](PLAN.md) | Regimes A/B/C, `categories/<L>.jsonl` schema, 03_1 compatibility path |

## Working documents

All three documents are historical and none was superseded by an implementation report:

- **Plans (never executed as written):** [`PLAN.md`](PLAN.md) — the `categories/` artifact and Beta-Binomial CI test; [`METHODOLOGY.md`](METHODOLOGY.md) — the F-vs-W comparison harness.
- **Spec (executed, outputs live in `02_2_2`):** [`PMI_PROMOTION_SPEC.md`](PMI_PROMOTION_SPEC.md).
