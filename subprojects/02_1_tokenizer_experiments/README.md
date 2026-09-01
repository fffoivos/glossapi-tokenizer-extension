# 02.1 — Tokenizer Experiments

> **In one line:** the research arm that decided *how many and which* Greek BPE units to append to Apertus — four exploratory arms collapsed to one (C3), a 25-point cutoff sweep picked **17,408 added units (vocab 148,480)**, and a parallel polytonic arm added the block that made production's **148,992** tokenizer.
> **Period:** 2026-04-10 (`f21eed85`) → 2026-06-11 (`a19c136f`), with a downstream epilogue on 2026-07-29; the decisive work all happened 2026-04-29 → 2026-05-18. **Status:** completed. The modern-Greek decision was frozen 2026-05-18 in [`02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md) and never reopened; only the polytonic budget was re-decided later.
> **Came from / led to:** [`../02_apertus_tokenizer_spec/`](../02_apertus_tokenizer_spec/README.md) → this → [`../02_2_tokenizer_implementation/`](../02_2_tokenizer_implementation/README.md) → [`../03_apertus_extension_and_embedding_adaptation/`](../03_apertus_extension_and_embedding_adaptation/README.md) → CPT (subprojects 05–09)

## Why this existed

Apertus-8B ships 131,072 BPE units of which only ~1,479 are Greek-attributable, so Greek text costs ~2.4 tokens per word against a Latin-script baseline of roughly one. Two questions had to be answered before any model-side work could start: should the Greek units come from a *freshly discovered* BPE vocabulary or from *continuing* Apertus's own merge table, and how many units should be appended. Both had to be settled with intrinsic evidence only — no CPT budget existed yet to settle them behaviourally. Everything in this directory is that evidence.

## History

### 2026-04-10 → 2026-04-14 — from two corpus views to a four-arm matrix

The subproject opened (`f21eed85`) with a much smaller question: compare `GlossAPI-only` against `GlossAPI + HPLT`, using a 40k–50k discovery vocab, and evaluate extension sizes around 5k/10k/15k/20k with `modern_greek_eval` as the primary decision set. Two days later the discovery vocab was locked at 50k and the mixed view fixed at **70/30** by training-token mass (`340dbf8f`). On 2026-04-14 the scope doubled (`5d2e7747`): four arms, not two corpus views —

| arm | method | corpus |
|---|---|---|
| `F1` | fresh discovery BPE | GlossAPI only |
| `F2` | fresh discovery BPE | GlossAPI + HPLT 70/30 |
| `C1` | continuous BPE from Apertus | GlossAPI only |
| `C2` | continuous BPE from Apertus | GlossAPI + HPLT 70/30 |

and the candidate grid moved to `{10240, 15360, 20480, 25600}`, with the continuous arms trained to `131,072 + 25,600 = 156,672`. That grid stayed frozen for the next month.

### 2026-04-29 — the continuous-BPE plan and the four-phase evaluation

`edb98d6b` added [`02_1_1_tokenizer_training/CONTINUOUS_BPE_EXTENSION_PLAN.md`](02_1_1_tokenizer_training/CONTINUOUS_BPE_EXTENSION_PLAN.md) and its TODO, plus the continuous trainer itself. The plan staged evaluation into four phases: raw four-arm intrinsic comparison → Apertus-compatible mergeback → fertility on the merged variants → downstream confirmation. At that point the plan recorded `F1` and `F2` as existing and `C1`/`C2` as **not yet built** (PLAN §2.3).

### 2026-05-11 — convergence on C3, and a held-out integrity failure

The four-arm comparison never ran to completion as designed. It was closed by fiat in favour of a fifth arm that had been trained in the meantime: **C3** (`C3_wave2_broad_glossapi_plus_hplt_50_50`) — continuous BPE from Apertus on GlossAPI + HPLT at **50/50** by token mass, on the wave-2 broad cleaner output, 25,600 added units, total 156,672. `F1`/`F2`/`C1`/`C2` were demoted to "analyzed baselines only" ([`../../docs/C3_CONVERGENCE.md`](../../docs/C3_CONVERGENCE.md), "Date converged: 2026-05-11").

The same day's cutoff sweep ([`../../docs/C3_CUTOFF_REPORT.md`](../../docs/C3_CUTOFF_REPORT.md)) surfaced a real defect: C3's train/val/test splits were partitioned by **row index**, not by document or text, so duplicate texts landed on both sides. Verification found 29,527 duplicate texts inside train, 30 train∩val and 36 train∩test collisions (~0.4–0.5 % of the held-outs). Rather than rebuild the splits, a clean evaluation path was constructed: `virgin_hplt` (10,000 HPLT docs anti-joined against the C3 training mix) plus `C3_val_clean` (7,624 docs) and `C3_test_clean` (7,246). All later fertility numbers use those three.

### 2026-05-14 → 2026-05-17 — the analytic cutoff answer: 11,264

`002bddc5` rewrote this README around C3 only and archived the four-arm framing. [`02_1_4_cutoff_analysis/REPORT.md`](02_1_4_cutoff_analysis/REPORT.md) (2026-05-17) then combined three independent streams — per-language PMI footprints from `02_2_2`, held-out fertility from `02_1_3`, and a glossary × char-mask composition label per added token — and recommended **11,264 added units (vocab 142,336, fertility 1.47)**. The binding constraint was a self-imposed design cap: Greek payload ≤ the ~13k "uniquely English" anchor. The report was explicit that this cap was a conservative choice, not a fact, and that the same evidence would defend anything up to ~17k. It also noted that the fertility elbow lay *outside* the cap: "under the cap, push as high as the constraint allows."

In parallel, [`02_1_5_added_token_curation/CURATION_REPORT.md`](02_1_5_added_token_curation/CURATION_REPORT.md) (2026-05-17) fixed a six-class removal policy for extraction/encoding artefacts — **104 removable tokens at the full 25,600 vocab (0.41 %)**, 39 inside the 11,264 anchor — and emitted it as a machine-readable manifest rather than editing any tokenizer.

[`02_1_6_representation_policy_analysis/`](02_1_6_representation_policy_analysis/README.md) tried to replace the "match language X" anchors with a derived policy. It did not succeed; its Gini-only route predicted +3–5k and its Phase 3–4 synthesis recommended +5,120, both roughly 3× below what measurement later chose. Its lasting contributions were the Mistral-provenance finding and the discovery of the swiss-ai TokEval suite.

### 2026-05-17 → 2026-05-18 — the empirical answer: 17,408, and the cap dropped

[`02_1_7_intrinsic_eval_sweep/`](02_1_7_intrinsic_eval_sweep/README.md) rebuilt the decision on Apertus's own evaluation surface (TokEval, commit `0c4a9c641e78c8243ac753976267fd50675197cb`): 33 tokenizers across a 0 → 25,600 grid at 1k step, four metric families, 178,658 merged rows. The 13k cap was dropped and the decision reduced to a knee criterion — **pick the first cutoff where the next 1k of vocab buys less than 1 % of fertility improvement**. That is **17,408** ([`02_1_7_intrinsic_eval_sweep/REPORT.md`](02_1_7_intrinsic_eval_sweep/REPORT.md), [`CHOSEN_CUTOFF.md`](02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md), decided 2026-05-18):

- Greek fertility on `C3_val` **1.345**, down 44.2 % from `apertus_base` 2.41; **82.4 %** of the theoretical maximum gain (asymptote 1.118), against 88.4 % for the full 25,600.
- Vocab **148,480 = 128 × 1160 = 256 × 580**, Apertus ids `0..131,071` verbatim.
- Embedding cost 272 MiB BF16 across untied embedding + LM head, ~1.7 % of the 8B model.
- MorphScore recall moved 0.689 → 0.694 and was explicitly demoted to "supporting color" — recall is flat (0.686–0.695) across the whole sweep. TFG on Apertus-55 rose by 0.06 % (basis points), a metric the report argues is structurally biased against script-isolated extensions.

Two reviewer rounds then rejected the obvious ways to apply the curation list. Deleting the 69 in-cutoff noise ids gave 148,411 — neither 128-aligned nor append-only, ids renumbered. The accepted construction **skips the 69 during merge selection and backfills with the next valid C3 merges**: 17,477 merges walked, 17,408 accepted, 0 cascade-skips, vocab back at 148,480. Every directly measured metric was flat or marginally better for the backfilled build than for the raw one, consistently across all four in-domain Greek slices.

After the decision, the canonical tokenizer was run over the exact C3 BPE training corpus (14,401,554 rows / 99.257 B chars / 24.892 B tokens): GlossAPI-nanochat **49.79 %** of token mass vs HPLT **50.21 %** — the 50/50 mix was balanced by token mass, not row count — with 0 zero-firing added tokens in the combined corpus and 27 in HPLT alone ([`02_1_7_intrinsic_eval_sweep/FIRING_COUNT_RUN_20260518.md`](02_1_7_intrinsic_eval_sweep/FIRING_COUNT_RUN_20260518.md)).

`7deea009` (2026-05-18) is the commit that reorganised everything above into the numbered `02_1_1`…`02_1_7` sub-subprojects and landed the polytonic arm; `9a6b0392` added the firing-count workflow the same day.

### 2026-05-18 → 2026-07-29 — the polytonic arm

[`02_1_polytonic_greek_extension/`](02_1_polytonic_greek_extension/README.md) ran as a separate lane on the premise that polytonic Greek deserves its own orthography rather than being covered incidentally by the modern extension — supported by the finding that Apertus/C3 contain **zero** tokens with distinctive polytonic marks. It selected and deduplicated an 18,726-row / 510 M-char corpus, continued BPE on top of the frozen 148,480 tokenizer to +5,120, and swept 512-token cutoffs. It recommended **+5,120 (vocab 153,600)**: balanced polytonic-validation Greek-word fertility 3.0021 → 1.9610, added-vocab utilization 0.9854, modern-Greek polytonic-id firing 0.31 %. The run happened 2026-05-18 but only landed in git on **2026-06-11** (`a19c136f`); the 2026-05-18 snapshot in [`../SUBPROJECTS_OVERVIEW.md`](../SUBPROJECTS_OVERVIEW.md) records it as "local / not yet versioned".

### What shipped, and the reversal at the end

Three downstream steps moved the answer this subproject had recorded:

- **2026-05-20** — [`../03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/SHIP_TOKENIZER_RECONSTRUCTION.md`](../03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/SHIP_TOKENIZER_RECONSTRUCTION.md) found that *both* variants on disk emitted a `tokenizer_config.json` with `tokenizer_class: TokenizersBackend`, which `AutoTokenizer` cannot load. The `tokenizer.json` files were structurally fine; the HF wrapper was rebuilt from Apertus's canonical config. That doc also declared the composite **153,600** bundle the active CPT base.
- **2026-05-25/27** — the public release nonetheless shipped **ModernGreek-148k** (148,480, sha `358ae3f2…`) as the canonical artifact with ModernGreek-Polytonic-154k as optional, and the 3.5B TokenDistil continuation used the 148k tokenizer ([`../../release/apertus-tokenizer-extension/README.md`](../../release/apertus-tokenizer-extension/README.md)).
- **2026-07-29 — production took neither.** A model-side probe re-opened the polytonic budget on the cleaned v2 corpus and froze **+512** merges: vocab **148,992 = 256 × 582**, `tokenizer.json` sha `bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b` — [`../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md`](../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md). +512 cut ancient token count 7.62 % and doubled the ancient single-token word rate to 41.05 %; +1,024 compressed harder (−10.62 %) but modelled ancient text 1.74 % worse after identical bounded adaptation (BPB 0.9283 vs 0.9124), so it did not earn its extra rows. Both passed the precommitted 0.5 % modern-BPB regression guard. The +5,120 / 153,600 bundle was demoted to a historical specialization artifact.
- Every CPT run from subproject 05 onwards binds that tokenizer as `fffoivos/apertus-tokenizer-extension@fcd33ec09fb7d86bc072b3a4b3e890efa6473b66`, subfolder `greek-modern-polytonic-tokenizer` ([`../06_dataset_scheduling_experiments/INITIALIZATION_AND_TRAINING_DECISIONS.md`](../06_dataset_scheduling_experiments/INITIALIZATION_AND_TRAINING_DECISIONS.md) §1; [`../07_full_8b_cpt/README.md`](../07_full_8b_cpt/README.md)). Its SHA matches the `c3p_poly_added_0512` row of this subproject's [`variants_manifest.json`](02_1_polytonic_greek_extension/analysis/c3p_polytonic_20260518T_impl/variants/variants_manifest.json), so **the production tokenizer is this arm's +512 grid point** — modern 17,408 + polytonic 512.

## Outcome

- **Arm decided**: C3 (continuous BPE from Apertus, GlossAPI + HPLT 50/50, wave-2 broad cleaner, 156,672 max vocab). F1/F2/C1/C2 closed as analyzed baselines; the four-arm comparison was never completed on its own terms.
- **Modern-Greek cutoff decided**: 17,408 added units, curated + backfilled, vocab 148,480, sha `358ae3f2…`. Greek fertility 2.41 → 1.345 on `C3_val`.
- **Curation made structural, not runtime**: 69 noise ids are absent from the vocab entirely, so no downstream stage needs a "skip these" branch.
- **Reversal recorded**: the analytic 11,264 anchor (with its ~13k English cap) was superseded by the empirical 17,408; the policy-archaeology and Gini routes predicted +3–5k and were wrong by ~3×.
- **Polytonic budget re-decided downstream**: the arm's own +5,120 recommendation lost to +512 in the 2026-07-29 model-side probe; production vocab is **148,992** (17,408 modern + 512 polytonic).
- **Carried forward**: the tokenizer contract into [`../02_2_tokenizer_implementation/`](../02_2_tokenizer_implementation/README.md); the ship bundles into [`../03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/`](../03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/README.md); the +512 polytonic block into every CPT run from subproject 05 onwards.
- **Left open** (all still unchecked in [`TODO.md`](TODO.md)): freeze the polytonic source-selection policy; review the polytonic kept/dropped decisions; define held-out polytonic and modern-Greek control slices; and revisit whether extra modern-Greek control slices are needed once adaptation results arrive. The "publish to HF" item was satisfied later by the release repo rather than here.

## Sub-subprojects

| Dir | Role | Period | Status | Result |
|---|---|---|---|---|
| [`02_1_1_tokenizer_training/`](02_1_1_tokenizer_training/README.md) | Stage 1 — continuous-BPE trainer + the original four-arm plan | 2026-04-29 → 2026-05-18 | completed | Produced C3 at 156,672 and the archived F1/F2 discovery arms; front-end contract checks per run |
| [`02_1_2_cutoff_variant_builder/`](02_1_2_cutoff_variant_builder/README.md) | Stage 2 — truncate a full arm into Apertus-compatible variants | 2026-05-11 → 2026-05-18 | completed | 25 C3 variants at 1k step, seconds each; front-end JSON copied byte-identical |
| [`02_1_3_fertility_evaluation/`](02_1_3_fertility_evaluation/README.md) | Stage 3 — intrinsic + fertility metrics per (variant, slice) | 2026-05-11 → 2026-05-18 | completed | 78-row C3 sweep in ~3 min; also built the three verified-clean held-outs after the splitter bug |
| [`02_1_4_cutoff_analysis/`](02_1_4_cutoff_analysis/README.md) | Stage 4 — combine PMI anchors + fertility + token composition | 2026-05-17 | superseded by `02_1_7` | Recommended 11,264 (vocab 142,336); found noise flat at ~0.13 % across the whole range |
| [`02_1_5_added_token_curation/`](02_1_5_added_token_curation/README.md) | Stage 5 — per-token keep/remove policy | 2026-05-17 | completed | 104 removals at 25,600 in six classes; manifest only, no tokenizer edits |
| [`02_1_6_representation_policy_analysis/`](02_1_6_representation_policy_analysis/README.md) | Policy archaeology on Apertus's implicit language budget | 2026-05-17 → 2026-05-18 | abandoned (archive mode) | Produced no budget; contributed the Mistral-provenance finding and the TokEval discovery that seeded `02_1_7` |
| [`02_1_7_intrinsic_eval_sweep/`](02_1_7_intrinsic_eval_sweep/README.md) | The cutoff decision that shipped | 2026-05-17 → 2026-05-18 | completed | **17,408 added, vocab 148,480, curated + backfilled**; plus the corpus-wide firing-count attribution |
| [`02_1_polytonic_greek_extension/`](02_1_polytonic_greek_extension/README.md) | Parallel Ancient/Polytonic arm stacked on the frozen C3 tokenizer | 2026-05-17 → 2026-07-29 | completed; sweep recommendation overturned | Sweep recommended +5,120 (153,600); the 2026-07-29 production probe froze +512 → **148,992**, the tokenizer every CPT run uses |

## Where things are

| What | Where |
|---|---|
| The decision contract (17,408 / 148,480) | [`02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md) |
| Evidence behind it | [`02_1_7_intrinsic_eval_sweep/REPORT.md`](02_1_7_intrinsic_eval_sweep/REPORT.md) + `02_1_7_intrinsic_eval_sweep/artifacts/plots/` |
| Backfilled-tokenizer builder | [`02_1_7_intrinsic_eval_sweep/scripts/01c_build_curated_backfilled.py`](02_1_7_intrinsic_eval_sweep/scripts/01c_build_curated_backfilled.py) |
| Removal manifest consumed at build time | [`02_1_5_added_token_curation/manifests/removal_list.jsonl`](02_1_5_added_token_curation/manifests/removal_list.jsonl) |
| Arm decision + held-out integrity finding | [`../../docs/C3_CONVERGENCE.md`](../../docs/C3_CONVERGENCE.md) |
| First (superseded) cutoff report | [`02_1_4_cutoff_analysis/REPORT.md`](02_1_4_cutoff_analysis/REPORT.md) |
| Polytonic run bundle and variant manifest | [`02_1_polytonic_greek_extension/analysis/c3p_polytonic_20260518T_impl/`](02_1_polytonic_greek_extension/analysis/c3p_polytonic_20260518T_impl/report/FULL_REPORT.md) |
| Published tokenizer (ModernGreek-148k) | [`../../release/apertus-tokenizer-extension/greek-extension-tokenizer/README.md`](../../release/apertus-tokenizer-extension/greek-extension-tokenizer/README.md) |
| Production polytonic-cutoff decision (148,992) | [`../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md`](../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md) |
| Production tokenizer binding in the CPT runs | [`../06_dataset_scheduling_experiments/INITIALIZATION_AND_TRAINING_DECISIONS.md`](../06_dataset_scheduling_experiments/INITIALIZATION_AND_TRAINING_DECISIONS.md) §1 |

Bulky artifacts — the tokenizer variants themselves, raw metric tables, parquet outputs, the vendored TokEval checkout — were deliberately kept out of git; the canonical tokenizer and its evidence bundle live on Hugging Face under `fffoivos/apertus-tokenizer-extension`.

## Working documents

- [`TODO.md`](TODO.md) — the C3 handoff checklist plus the polytonic to-dos. Historical; useful mainly for which boxes stayed unchecked.
- Per-stage plans and reports are listed in each sub-subproject's README. The largest historical documents are [`02_1_1_tokenizer_training/CONTINUOUS_BPE_EXTENSION_PLAN.md`](02_1_1_tokenizer_training/CONTINUOUS_BPE_EXTENSION_PLAN.md) (four-arm framing, carries its own archive banner), [`02_1_7_intrinsic_eval_sweep/PLAN.md`](02_1_7_intrinsic_eval_sweep/PLAN.md) and [`02_1_7_intrinsic_eval_sweep/FIRING_COUNT_PLAN.md`](02_1_7_intrinsic_eval_sweep/FIRING_COUNT_PLAN.md) (both marked implemented), and the deprecated policy-analysis tree at `02_1_6_representation_policy_analysis/_deprecated_20260518/`.
