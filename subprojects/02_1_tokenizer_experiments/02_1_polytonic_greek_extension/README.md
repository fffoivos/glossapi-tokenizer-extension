# 02.1.polytonic — Polytonic Greek Extension

> **In one line:** a parallel arm that gave Ancient/Polytonic Greek its own orthographic lane on top of the frozen modern tokenizer; its own sweep recommended **+5,120**, but a 2026-07-29 model-side probe froze **+512** — vocab **148,992** — which is the tokenizer every production CPT run uses.
> **Period:** 2026-05-17 (first strict source-filter run) → 2026-06-11 (`a19c136f`, bundle committed), with the production decision on 2026-07-29. **Status:** completed; the sweep's own recommendation was overturned downstream.
> **Came from / led to:** [`../02_1_7_intrinsic_eval_sweep/`](../02_1_7_intrinsic_eval_sweep/README.md) (supplies the 148,480 base) → this → [`../../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/polytonic_cutoff_probe/`](../../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/polytonic_cutoff_probe/README.md) → CPT (subprojects 05–09)

## Why this existed

The modern-Greek extension was built and evaluated on modern web and academic text; polytonic Greek was never in its objective. Direct inspection confirmed the gap is total: the modern-Greek attribution review found 1,507 Greek-codepoint vocab entries but `greek_script_poly_any = 0`, and the `grc_Grek` FineWeb-2 attribution run (28,850 docs, 128.4 M sampled tokens) fired many Greek fragments while containing **no** token types with distinctive polytonic codepoints — with ~35.9 % of its sampled mass falling into unknown/byte fragments. Rather than reopen the just-frozen modern cutoff, this arm appends a distinct polytonic block after it.

## History

### 2026-05-17 — source selection and strict filtering

Run `polytonic_strict_w050_c010_20260517T131514Z`. Curated ancient/liturgical sources (First1KGreek, Perseus/classical, GOARCH liturgical) were taken by provenance; the mixed collections (Wikisource Greek, Scholarios graeca-patristic) were filtered by **distinctive** polytonic orthography — grave/varia, breathings, perispomeni, ypogegrammeni, or a Greek Extended codepoint whose NFD contains one of those. **Plain tonos/oxia does not count**, which is the discriminating choice: it is what separates polytonic text from ordinary accented modern Greek. Thresholds: `distinctive_polytonic_word_ratio ≥ 0.50` and `distinctive_polytonic_char_ratio ≥ 0.10`, plus `greek_percentage ≥ 50`, `latin_percentage ≤ 10`.

Wikisource kept 3,435 of 5,394 rows; Scholarios 13,419 of 14,118. After dedup with Greek diacritics preserved (MinHash 0.85): 19,360 decisions → **18,726 kept**, 634 dropped, 218 near-duplicate drops.

A cross-check against FineWeb-2's `grc_Grek` found **0** full-document matches after normalization but 424 cross-source near-duplicate families, i.e. overlap at the work/excerpt level rather than the document level. The kept corpus is smaller in documents than FineWeb-2 `grc_Grek` (18,726 vs 28,539) but carries far more text mass (83.0 M vs 30.2 M Unicode words) because its documents are complete curated works rather than crawl pages.

### 2026-05-18 — the cutoff sweep

Plan [`ANCIENT_GREEK_AFTER_C3_PLAN.md`](ANCIENT_GREEK_AFTER_C3_PLAN.md) pinned the base (148,480, sha `358ae3f2…`), a ceiling of +5,120 (→ 153,600 = 256 × 600) and a 512-step aligned grid, so every candidate stays 256-aligned. Run `c3p_polytonic_20260518T_impl` trained the continuation and built 11 variants; the same TokEval and MorphScore guards used for the modern sweep were re-run as regression checks.

Results on the balanced polytonic validation slice ([`analysis/c3p_polytonic_20260518T_impl/report/FULL_REPORT.md`](analysis/c3p_polytonic_20260518T_impl/report/FULL_REPORT.md)):

| variant | Greek-word fertility | distinctive-polytonic fertility | added-vocab utilization |
|---|---:|---:|---:|
| +0 (C3 base) | 3.0021 | 2.9961 | — |
| +512 | 2.3734 | 2.2008 | 0.9961 |
| +5,120 | **1.9610** | **1.7851** | 0.9854 |

Guards held: Apertus-55 TFG moved 0.1161 → 0.1162 across the whole grid; MorphScore recall stayed ~0.694; polytonic ids fire on modern C3 validation at only **0.31 %** of tokens at +5,120. FineWeb-2 Ancient Greek improved 2.8087 → 2.0050. **Recommendation: keep +5,120** — the grid was still improving at the budget edge and utilization was still high.

A separate check in [`../../03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/POLYTONIC_VOCAB_BUDGET_CHECK.md`](../../03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/POLYTONIC_VOCAB_BUDGET_CHECK.md) (2026-05-20) validated +5,120 against a script-isolated sub-1B-language power-law fit (`vocab_fired≥100 ≈ 0.1341 × tokens^0.5688`, R² 0.783, n = 194): at the corpus's ~163 M post-extension tokens the fit predicts 4,000–6,300 distinctive tokens, and +5,120 sits inside that band.

### 2026-05-20 → 2026-05-27 — packaging

Both ship bundles were rebuilt because the training pipeline emitted `tokenizer_config.json` with `tokenizer_class: TokenizersBackend`, which `AutoTokenizer` cannot load; the `tokenizer.json` files themselves were structurally correct ([`../../03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/SHIP_TOKENIZER_RECONSTRUCTION.md`](../../03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/SHIP_TOKENIZER_RECONSTRUCTION.md)). End-to-end verification of the 153,600 bundle: polytonic NT 60 → 20 tokens (−66.7 %), Katharevousa −65.9 %, modern Greek web −31.8 %, English and Russian **unchanged** — the multilingual-preservation constraint holds. The public release then shipped ModernGreek-148k as canonical with ModernGreek-Polytonic-154k as an optional supporting artifact.

### 2026-06-11 — committed

`a19c136f` landed the implementation bundle. The run itself is dated 2026-05-18; the 2026-05-18 snapshot in [`../../SUBPROJECTS_OVERVIEW.md`](../../SUBPROJECTS_OVERVIEW.md) records these artifacts as "local / not yet versioned", so the sweep's headline numbers predate their own commit by three weeks.

### 2026-07-29 — the endpoint: production selects +512, not +5,120

Re-opened as a model-side probe on the cleaned v2 corpus, deciding between +512 and +1,024 with a precommitted rule: reject any candidate whose modern-Greek BPB regresses more than 0.5 %, and take +1,024 only if it beats +512 on ancient BPB by ≥ 1 %. Decision and receipts in [`../../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md`](../../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md):

| metric (311 ancient docs / 1,000 modern docs) | Modern-148480 | **+512** | +1,024 |
|---|---:|---:|---:|
| Ancient tokens vs baseline | — | **−7.62 %** | −10.62 % |
| Ancient single-token word rate | 20.13 % | **41.05 %** | 43.23 % |
| Ancient BPB after bounded adaptation | 0.8533 | **0.9124** | 0.9283 |
| Modern BPB ratio to baseline | 1.0000 | **1.00138** | 1.00231 |

Both passed the modern guard; +1,024 modelled ancient text 1.74 % *worse* than +512, so it did not earn its extra 512 rows. **+512 frozen → vocab 148,992 = 256 × 582, sha `bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b`.** The +5,120 / 153,600 bundle became a historical specialization artifact.

Two things the probe surfaced that are worth keeping:

- **A calibration bug, found and fixed.** The first probe failed both modern guards by ~26 % despite almost unchanged modern token counts: positive-only token distillation had made the new output rows overconfident and inflated the expanded softmax denominator. The corrected pass froze the model, alternated disjoint ancient/modern calibration blocks, updated only ids ≥ 148,480, and exact-checked every pre-existing row — modern regression fell from 26.29 % to 0.138 %. Both the failed and corrected runs are retained.
- **Three suspicious ids reviewed and kept** (148924, 148979, 148987): not mojibake but partial UTF-8 ByteLevel merge components that fire inside valid polytonic surfaces (`ἷς`, `ὓς`, `Ἦχος`). Removing one would break a live merge dependency, so they stay in the tokenizer with merge-chain initialization and are excluded only from standalone token-distillation targets.

## Outcome

- **Shipped into production**: +512 polytonic merges on top of the 17,408 modern ones → **148,992**, bound by every CPT run from subproject 05 onward as `fffoivos/apertus-tokenizer-extension@fcd33ec09fb7d86bc072b3a4b3e890efa6473b66`, subfolder `greek-modern-polytonic-tokenizer`.
- **Sweep recommendation overturned**: the intrinsic-fertility argument for +5,120 lost to a BPB-based, model-in-the-loop comparison at a 10× smaller budget. Compression alone did not predict downstream modelling quality.
- **Corpus artifact**: 18,726 kept rows / ~510 M chars, sha `2b89e098…`, kept out of git under the `subprojects/**/data/` rule.
- **Caveats recorded in the sweep**: the `poly_underaccented_test` slice came out empty under the strict filter, so under-accented curated ancient text is untested; the `c3p_poly_added_0000` variant was reserialized by the builder so its JSON SHA differs from the original C3 ship SHA; and the Apertus-55 proxy config loaded 54 languages, not 55, in that environment.
- **Never closed** (from the parent [`../TODO.md`](../TODO.md)): freeze the source-selection policy; review the kept/dropped decisions and representative choices; define held-out polytonic and modern-Greek control slices before extension training.

## Where things are

| What | Where |
|---|---|
| Production decision (+512) | [`../../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md`](../../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md) |
| Production candidate manifests + byte-fragment audits (+0 / +512 / +1,024) | `analysis/c3p_polytonic_20260518T_impl/production_cutoff_candidates/` — the 18 MB `tokenizer.json` files are deliberately not copied here |
| Suspicious-token review | `analysis/c3p_polytonic_20260518T_impl/production_cutoff_candidates/suspicious_token_review.json` |
| Model probe (failed uncalibrated + corrected calibrated) | `analysis/c3p_polytonic_20260518T_impl/production_cutoff_candidates/model_probe/` |
| Append-only + byte-fragment auditor | `scripts/audit_polytonic_cutoff_tokens.py` |
| Sweep report and plots | [`analysis/c3p_polytonic_20260518T_impl/report/FULL_REPORT.md`](analysis/c3p_polytonic_20260518T_impl/report/FULL_REPORT.md) |
| Full 512-step variant manifest | `analysis/c3p_polytonic_20260518T_impl/variants/variants_manifest.json` |
| FineWeb-2 overlap evidence | `analysis/fineweb2_comparison/`, `analysis/fineweb2_overlap_main_grc_Grek_20260517T152232Z/` |
| Storage boundary (what is and is not kept locally) | [`ARTIFACTS.md`](ARTIFACTS.md) |

## Working documents

- [`ANCIENT_GREEK_AFTER_C3_PLAN.md`](ANCIENT_GREEK_AFTER_C3_PLAN.md) — the execution plan; §3's 512-step grid is the one that produced the eventual production point. Historical.
- [`ARTIFACTS.md`](ARTIFACTS.md) — storage-boundary note. Historical.
- `scripts/` — 14 scripts covering source audit, dedup input prep, splits, variant building, the cutoff/byte-fragment audit, evaluation, TokEval and MorphScore guards, and report rendering.
