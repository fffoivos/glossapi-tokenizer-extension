# 02.2 — Tokenizer implementation

> **In one line:** created to hold the Apertus-compatible Greek merge-rule extension, this directory never did that job — it became instead the home of the per-token language-attribution machinery, which ran to completion and produced the per-language token sets that the cutoff decision and the embedding diagnostic actually consumed.
> **Period:** 2026-04-10 → 2026-05-18. **Status:** the attribution track completed; the named implementation track was never started here and was carried out elsewhere ([`02_1_2_cutoff_variant_builder`](../02_1_tokenizer_experiments/02_1_2_cutoff_variant_builder/) built the compatible variants, `03_3_cscs_experiments_kickoff` assembled and verified the ship bundles).
> **Came from / led to:** [`02_apertus_tokenizer_spec`](../02_apertus_tokenizer_spec/) → this → [`02_1_4_cutoff_analysis`](../02_1_tokenizer_experiments/02_1_4_cutoff_analysis/), [`02_1_7_intrinsic_eval_sweep`](../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/), [`03_apertus_extension_and_embedding_adaptation`](../03_apertus_extension_and_embedding_adaptation/)

## Why this existed

Two different questions ended up in this directory.

The original one, from April, was mechanical: how do you add Greek merges to Apertus's inherited `tekken` BPE without breaking it? The answer was decided up front — patch `model.vocab` and `model.merges` rather than calling `add_tokens(...)`, preserve every old id, append only, emit a manifest of every added unit, keep the final vocab divisible by 128 — with four required regression checks (first 1,000 ids, special-token behaviour, regex-split and byte-level behaviour, and a non-Greek smoke test). Those constraints held for the whole program; the code that enforces them was written next door.

The second question arrived in May and took over: **how much vocabulary does Apertus already spend on each language?** Nobody could size a Greek extension without an answer, and no answer existed, because Apertus inherited Mistral-Nemo's tokenizer and Mistral never published its training-language list. Answering it required two independent evidence layers — what characters each language *can* produce (strict, from CLDR) and what tokens each language *does* produce (empirical, from counting) — plus an explicit, defeasible rule for joining them. That is what the four sub-subprojects are.

## History

### 2026-04-10 — the directory is created (`f21eed85`)

A 22-line README and a 7-item TODO: design a compatible Greek BPE config, implement tokenizer diffing against Apertus, implement merge-rule extension assembly, add a regression checklist and executable tests. **None of the four TODO items was ever completed in this directory**; no BPE or merge-assembly code exists here.

### 2026-05-13/14 — the attribution machinery moves in (`002bddc5`, `0b20b96d`)

Two sub-subprojects appear under the still-unstarted implementation scope. [`02_2_2_vocab_lang_attribution/`](02_2_2_vocab_lang_attribution/) lands as a finished 8-worker distributed run: ~1 B Apertus tokens per language across 1,933 canonical keys, 113.4 B firings, ~2.5 h, ~$100. [`02_2_1_char_language_membership/`](02_2_1_char_language_membership/) lands as a 54-triple strict-rule CLDR bitmask, hardened the same day to 55 triples with a consumer-safe sparse-table query helper and a token-level audit gate. The parent TODO gains an "In progress" note reporting the run at 87 % complete.

### 2026-05-15 (early) — the four-stage pipeline is named (`719d3834`)

The directories are renumbered `02_2_1` … `02_2_4` and the README is rewritten to declare a pipeline: two independent evidence layers feeding a tier classifier feeding a category promoter feeding the embedding diagnostic. The char tool is rebuilt as three parallel layers (22 script / 31 family / 55 language bits, schema v4) with twelve per-script research notes. [`02_2_3_token_classification/`](02_2_3_token_classification/) and [`02_2_4_language_category_promotion/`](02_2_4_language_category_promotion/) are created — both marked "proposal — for review before implementation", and stage 3 stays a proposal permanently.

The same commit brings the first serious reviews of the histogram (Greek, English, German) and, with them, the finding that reframed everything: **the char mask cannot map tokens to languages, only exclude them.** English has no character that is exclusively English, so English attribution rests entirely on firing rates. It also brings [`REVIEW_ISSUES_20260514.md`](02_2_2_vocab_lang_attribution/analysis/german_review/REVIEW_ISSUES_20260514.md), which shows that the German-vs-English comparisons were confounded — German and Greek came from FineWeb-2-HQ, English from Clean-Wikipedia.

### 2026-05-15 (later) — everything ships in one session (`0bbd93de`)

The corpus confound is fixed by *adding* rather than replacing: English is re-tokenised from FineWeb-HQ and appended as `eng_Latn_fineweb_hq`, taking the matrix to 1,934 rows and turning the confound into a measurable quantity (the two English samples differ by 2,285 tokens and 6.6 pp of mass at premise level).

The promotion method is then chosen and built. [`METHODOLOGY.md`](02_2_4_language_category_promotion/METHODOLOGY.md) had refused to pick between filtering and weighting a priori and specified a comparison harness; [`PMI_PROMOTION_SPEC.md`](02_2_4_language_category_promotion/PMI_PROMOTION_SPEC.md) cut through it with one concrete pass — multi-language PMI at `α = 0.5`, `δ = 1.0`, `min_count = 100`, over the 87 keys with ≥ 1 B firings, with the char mask toggled on (Variant A) and off (Variant B) so the mask's contribution stays auditable. PMI was chosen over pairwise log-ratio and max-pooling for one reason: it scores each language independently against the corpus marginal and therefore **scales with the number of in-scope languages**, where max-pooling shrinks monotonically as scope grows.

Running it exposed the char tool's coverage gap — 34 of the 87 keys had no mapping — and triggered five char-tool releases in the same session (v3.2 → v3.3 → v3.3.1 → v3.3.2 → v3.3.3), each verified back by the consumer. Two of the bugs were severe and silent: Greek, the project's anchor language, was producing an **empty** masked set because `ell_Grek` was missing from the published key map; Standard Arabic likewise via the ISO macrolanguage/individual code split. Both were closed, and a build-time self-test was added so that class of bug cannot recur silently. [`CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md`](CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md) was written at the end of this session as the single end-to-end record.

### 2026-05-18 — the last commit (`7deea009`)

[`TOP_LANGUAGES_BY_VOCAB_SHARE.md`](02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/TOP_LANGUAGES_BY_VOCAB_SHARE.md) ranks all 86 languages by masked-set size and again by unique-set size, written to feed the C3 cutoff decision. Nothing in this directory changed afterwards.

### Reversals and things that did not happen

- **The named scope was never executed here.** The 02.1 README states that the shipped tokenizer variant "goes through downstream confirmation in `02_2_tokenizer_implementation` (compatibility checks)". It did not: the compatible variants were derived in `02_1_2_cutoff_variant_builder`, and the two ship bundles (`apertus_greek_modern_only_148480`, `apertus_greek_extended_153600`) were assembled and verified loadable in `03_3_cscs_experiments_kickoff`.
- **Stage 3 was skipped.** The `token_dataset_attribution.parquet` artifact was never built; its tier vocabulary survived as shared shorthand and ran only inline, for German.
- **Stage 4 was bypassed rather than completed.** The `categories/<L>.jsonl` interface, the `base_greek_tokens.jsonl` migration and the F-vs-W comparison harness were never built; `03_1` wrote its own loaders against the PMI `__masked.txt` tables instead.
- **The consumer's own estimate was retracted.** v3.3 added seven scripts and was predicted to lift coverage ~2 pp to ~87.5 %; measured gain was **zero**, because Apertus byte-fragments all seven scripts. Recorded as an Apertus-vocab fact, not a defect.
- **A review was logged and largely not applied.** `REVIEW_ISSUES_20260514.md` closes with "No code changes applied yet — awaiting direction"; only its Issue 1 was addressed, by the English re-run.

## Outcome

- **A reproducible per-token language attribution over the whole Apertus vocab.** 1,934 × 131,072 firing histogram (113.4 B firings at the 1,933-key run, 114.37 B after the English re-run row was appended) joined to strict CLDR char masks (29 script / 47 family / 88 language bits) via PMI promotion. **113,184 of 131,072 tokens (86.35 %)** land in at least one language's masked set; the 17,888 uncovered are classified by reason, the largest bucket being 7,463 tokens that no language over-represents by 10× (`analysis/main_token_sets_pmi/uncovered_tokens.tsv`, recomputed).
- **Coverage improved release by release** as the char tool grew: 81.18 % (v3.1) → 85.54 % (v3.2) → 85.55 % (v3.3.1) → 86.35 % (final), with unmapped corpus keys going 34 → 7 → 5 ([`02_2_1_char_language_membership/v3_2_INTEGRATION_REPORT_20260515.md`](02_2_1_char_language_membership/v3_2_INTEGRATION_REPORT_20260515.md), `analysis/main_token_sets_pmi/manifest.json`).
- **The masks passed an independent correctness test.** Across 87 languages exactly one cross-script overlap exists, `cmn_Hani` ↔ `jpn_Jpan` (shared Han characters); every other cross-script pair is zero. Within-script overlap tracks linguistic distance — Moroccan↔Standard Arabic at Jaccard 0.918, Croatian↔Bosnian 0.550, Czech↔Slovak 0.320 ([`CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md`](CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md) § 5).
- **Headline per-language figures** (masked sets, `summary.tsv`): Greek 1,479 tokens covering 86.94 % of Greek mass; German 7,329 / 53.55 %; English (FineWeb-HQ) 19,339 / 56.17 %; English (wiki) 19,009 / 47.42 %. Greek ranks 42nd by set size but 8th by uniqueness.
- **Carried forward**, and this is what makes the subproject load-bearing rather than exploratory: `summary.tsv` supplies the comparable-language anchor table behind the C3 cutoff recommendation in [`../02_1_tokenizer_experiments/02_1_4_cutoff_analysis/REPORT.md`](../02_1_tokenizer_experiments/02_1_4_cutoff_analysis/REPORT.md); `char_language_bitmask.parquet` classifies every added token in that same analysis; the PMI masked sets became the language groups for the v3 (11-language) and v4 (75-language) embedding diagnostics in `03_1`; and `lang_metadata.json` supplies the power-law fit in `03_3_cscs_experiments_kickoff/POLYTONIC_VOCAB_BUDGET_CHECK.md`.
- **Left open at the end**: the merge-rule extension and its regression suite (moved elsewhere); the tier artifact and the category-promotion artifact; the `δ × min_count × marginal-scope` knob sweep that both `METHODOLOGY.md` and the checkpoint name as the natural next workstream; training-weighted PMI, which needs per-locale Apertus training shares nobody sourced; 4 permanently zero-sum corpus keys; and five of the six 2026-05-14 review issues.

## Sub-subprojects

| Dir | Role | Period | Status | Result |
| --- | --- | --- | --- | --- |
| [`02_2_1_char_language_membership/`](02_2_1_char_language_membership/) | strict CLDR char-admissibility masks at script / family / language resolution | 2026-05-14 → 05-15 | completed (v3.3.3, schema v5) | 29 / 47 / 88 bits; zero cross-script leakage; consumed by the PMI pass and by `02_1_4` |
| [`02_2_2_vocab_lang_attribution/`](02_2_2_vocab_lang_attribution/) | empirical per-token firing histograms + all the analyses built on them | 2026-05-13 → 05-18 | completed | 1,934 × 131,072 matrix, 113.4 B firings; PMI promotion is the canonical output |
| [`02_2_3_token_classification/`](02_2_3_token_classification/) | tiered per-(token, dataset) labels under an explicit dataset-language premise | 2026-05-15 | abandoned as an artifact | one `PLAN.md`; tiers ran inline for German only |
| [`02_2_4_language_category_promotion/`](02_2_4_language_category_promotion/) | per-language canonical token sets for the embedding diagnostic | 2026-05-15 | partially executed | PMI spec shipped and built (in `02_2_2`); `categories/` artifact and F-vs-W harness never built |

## Where things are

| Artifact | Path | Note |
| --- | --- | --- |
| End-to-end record | [`CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md`](CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md) | corpus, char tool, method, results, coverage audit, overlap analysis — the one document to read |
| Canonical token sets | `02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/tables/<key>__{masked,unmasked,delta}.txt` | 87 keys × 3 = 261 files, tracked |
| Per-language summary | `02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/summary.tsv` | 87 rows; the most-cited file in the program |
| Build entry point | `02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/build.py` | deterministic; `build.py` → `coverage_audit.py` → `overlap_analysis.py` |
| Promotion spec | [`02_2_4_language_category_promotion/PMI_PROMOTION_SPEC.md`](02_2_4_language_category_promotion/PMI_PROMOTION_SPEC.md) | knobs, algorithm, sanity assertions |
| Char-mask sources of truth | `02_2_1_char_language_membership/{languages,families,scripts}.yaml` | stable bit assignments |
| Untracked inputs | `02_2_1_char_language_membership/artifacts/`, `02_2_2_vocab_lang_attribution/outputs/` | gitignored repo-wide; the parquets and `histogram_matrix.npz` must be rebuilt before any consumer runs |

## Working documents

- **Status snapshots and handoffs (historical):** [`CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md`](CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md) — accurate on method and audit, but its § 4 coverage table and its "7 remaining unmapped keys" predate the v3.3.2 locale additions that the committed `summary.tsv` and `manifest.json` reflect (5 unmapped; `als_Latn` → `sq` and `lat_Latn` → `la` both resolved). [`TODO.md`](TODO.md) — the April implementation checklist plus a mid-run progress note; none of its four items was done here.
- **Per-sub-subproject documents** are indexed in each sub-subproject's own README: designs and per-script notes under `02_2_1`, the run report and eight analysis directories under `02_2_2`, one plan under `02_2_3`, three specs under `02_2_4`.
