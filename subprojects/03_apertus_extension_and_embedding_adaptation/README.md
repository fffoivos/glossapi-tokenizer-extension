# 03 — Apertus extension and embedding adaptation

> **In one line:** the model-side half of the Greek tokenizer-extension programme — diagnose how Apertus-8B-2509 encodes Greek, clean the CPT corpus, stand up Clariden, then race four embedding-initialisation arms (Vanilla / ReTok / Centroid / TD-layer11) to 2 B → 3.5 B → 5 B tokens; it ended with data rather than a rule-bound winner, and with **base Apertus (Vanilla) as the safe default**.
> **Period:** 2026-04-10 → 2026-08-21 (commits `f21eed85` … `7dce0efb`). **Status:** the bakeoff completed 2026-05-26; one late decision (the +512 polytonic production cutoff) closed 2026-07-29; otherwise the directory survived as shared Clariden tooling for later subprojects.
> **Came from / led to:** [`02_1_tokenizer_experiments`](../02_1_tokenizer_experiments/) + [`02_2_tokenizer_implementation`](../02_2_tokenizer_implementation/) (frozen 17,408-token cutoff) → **this** → [`04_cpt_training_regime_on_vanilla`](../04_cpt_training_regime_on_vanilla/) (Vanilla regime diagnostic) and [`05_token_distillation_cpt`](../05_token_distillation_cpt/) (TD CPT), then 06/07/08 which kept reusing this directory's launcher and eval scripts.

## Why this existed

Subproject 02 froze a Greek tokenizer extension (+17,408 modern tokens → vocab 148,480; +5,120 polytonic stacked → 153,600). That only produces a vocabulary; it says nothing about whether *adding rows to Apertus's embedding and LM-head matrices and continuing pretraining* actually buys Greek capability, and at what cost to the model's multilingual character. Apertus is untied (`tie_word_embeddings=false`), so both `E` and `U` need explicit initialisation for the new IDs. This subproject was where that question got turned into a controlled experiment on CSCS Clariden, and where the corpus, the auth, the HF↔Megatron plumbing, and the eval harness for every later CPT run were built.

## History

### 2026-04-10 → 2026-05-14 — placeholder, then the diagnostic

The directory was created as a 22-line stub README plus a TODO (`f21eed85`), holding the position "this comes after tokenizer and corpus work". The first real content was [`03_1_greek_embedding_diagnostic/`](03_1_greek_embedding_diagnostic/README.md) (`002bddc5`, 2026-05-14; extended to per-language v3/v4 in `7deea009`, 2026-05-18): a read-only geometric study of Apertus's existing `E`/`U` rows. It produced the two numbers the later init arms consumed — **E target norm 5.05, U target norm 3.80** for Greek-content tokens.

### 2026-05-18 → 2026-05-19 — the C3 × Apertus dedup audit

[`03_2_apertus_c3_dedup_audit/`](03_2_apertus_c3_dedup_audit/README.md) measured document-level overlap between Apertus's Greek pretraining sources and the Greek corpus pool. Four review rounds reframed the scope before it ran (see `REVIEW_INTEGRATION_20260518*.md`); a partial 4-worker attempt was archived, and the full 8-worker GCP run finished 2026-05-19. Result: **2,223,781 of 98,203,721 pool documents overlap (~2.27 %)**, published as a hard-drop overlay. The held-out contamination check was **skipped** and never recovered — GCloud access was lost 2026-05-20.

### 2026-05-20 — planning bridge, and two scope reversals

[`03_3_cscs_experiments_kickoff/`](03_3_cscs_experiments_kickoff/README.md) landed in one day (`ec5ee52b`): state audit, `cscs-key` auth workflow, replay-language selection, the polytonic budget check, and two rebuilt HF-loadable ship tokenizers. `01d7befa` then adopted **cpt_plan v0.7** as canonical, overriding the older v0.12 experiment plan on framework (Megatron-LM-Swiss-AI, not HF Trainer), replay split (70/30, not 85/15) and replay language count (24, not the 34 that `REPLAY_LANGUAGE_SELECTION.md` had argued). Same day, two reversals in opposite directions: v0.7 first declared the **composite 153,600** tokenizer the CPT base, and then the bakeoff scope decision dropped polytonic and pinned the **modern-only 148,480** bundle for the arms ([`init_bakeoff/BAKEOFF_PLAN.md`](03_4_implementation_experiments/init_bakeoff/BAKEOFF_PLAN.md) scope-update block).

### 2026-05-21 — recipe, references, risks, and the overnight execution

[`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) and the reviewer packet were written and then audited against 8 pinned repos + 15 papers ([`references/`](references/README.md), `fde4146d`). The audit produced [`RISKS.md`](RISKS.md) — 17 silent-failure risks — including **R17**: `saver_core` has no protocol slot for Apertus's xIELU α and QK-Norm tensors, so a raw HF→Megatron conversion silently resets them. The overnight Clariden session (archived at [`_archive/2026-05-21_overnight_session/`](_archive/README.md)) built the corpus, proved the loader roundtrip (job `2333864`: standard tensors bit-exact, **128 R17 deltas = 32 layers × 4 xIELU params**), and produced a partial V4 baseline that had to be redone because `global_mmlu` was missing from the task list.

### 2026-05-22 — three arms train

`bakeoff_1node_chain_20260522_005620` ran Vanilla / ReTok / Centroid to iter 476 (~2.0 B tokens each) off R17-patched TP=2 checkpoints, with a checkpoint-eval watcher submitting per-iteration evals. Centroid was visibly broken from iter 130 (BPB 1.13 vs ≤0.76 elsewhere).

### 2026-05-23 → 2026-05-24 — the fourth arm

Token Distillation, "bracketed" in v0.7 §13, came back as [`TOKEN_DISTILLATION_PLAN.md`](TOKEN_DISTILLATION_PLAN.md) and ran as a gated ladder in one day: CPU coverage prepass (**99.82 % of the 17,408 new tokens had ≥100 usable snippets** → gate `run_full_td_100`), a two-candidate layer pilot that picked **target_layer=11** over the paper-default last layer, full-token TD at 25 snippets (17,377 trained / 15 skipped), an R17 roundtrip gate (job `2357565`), and a 2 B chained training arm. At 2 B, TD beat ReTok and Centroid decisively but did **not** beat Vanilla; `PRODUCTION_DECISION_STATE.md` recorded Vanilla as the safe default.

### 2026-05-24 → 2026-05-26 — the continuations, and the reversal

Rather than stop at 2 B, Vanilla/ReTok/TD were continued to iter 834 (~3.5 B) and Vanilla/TD to iter 1192 (~5.0 B). **The 2 B headline flipped**: at 5 B, TD-layer11 led Vanilla on all three downstream aggregates (Greek no-MT +1.28 pp, English retention +1.04 pp, multilingual +0.40 pp) while Vanilla kept tokenizer-fair BPB (0.4602 vs 0.4872) — see [`BAKEOFF_FINAL_RESULTS_20260526.md`](03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md). Then, on 2026-05-26, a purpose-built **native-Greek suite reversed the reversal**: on vetted native Greek MCQ (GreekMMLU + ILSP Medical + ILSP ASEP), Vanilla-5B 0.4305 > TD-5B 0.4109, and **Apertus-Base 0.4817 beat every continued arm** ([`NATIVE_GREEK_SUITE_RESULTS_20260526.md`](03_4_implementation_experiments/init_bakeoff/eval/NATIVE_GREEK_SUITE_RESULTS_20260526.md)).

Two methodological rules were fixed in the same window: `47f42dc2`/`504e5d38` wrote the [loss-measurement policy](03_4_implementation_experiments/init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md) (raw Megatron `lm loss` is not cross-tokenizer fair; use heldout BPB — historical artifacts call it `BPC`), and the Greek aggregate was redefined to exclude the two MT-derived diagnostics.

### 2026-06-11 — synthesis and archive

`a19c136f` landed the whole 2026-05-26 result set at once and reorganised the directory: ten planning/status docs moved to [`_archive/synthesis_sources_20260526/`](_archive/README.md) and were replaced by [`CPT_MASTER_20260526.md`](CPT_MASTER_20260526.md), which carries a **14-entry plan-vs-results discrepancy log**. Its headline admission: **none of the v0.12 §10 Q8 pre-commit thresholds (X / M_progress / M_ext / M_van / T) were ever locked**, so the bakeoff produced data, not an adjudicated winner.

### 2026-07-29 — the polytonic question reopened, and settled at +512

The polytonic layer that had been dropped from the bakeoff came back as a much smaller, properly gated question in [`03_4_implementation_experiments/polytonic_cutoff_probe/`](03_4_implementation_experiments/polytonic_cutoff_probe/README.md): append **+512 or +1,024** polytonic merges to the modern 148,480 tokenizer, with a pre-committed rule (reject > 0.5 % modern-BPB regression; take +1,024 only if it beats +512 on ancient BPB by ≥ 1 %). The first model probe failed both modern guards by ~26 % — positive-only token distillation had made the new output rows overconfident and inflated the softmax denominator — which was fixed by a frozen-model, balanced ancient/modern calibration pass over the appended LM-head rows only (26.29 % → **0.138 %**). **+512 was selected**: ancient token count −7.62 %, ancient single-token word rate 20.13 % → 41.05 %, modern BPB ratio 1.00138. `+1,024` passed the guard but modelled ancient text 1.74 % worse. The frozen bundle is [`ship/apertus_greek_modern_polytonic_148992/`](03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_polytonic_148992/) (vocab 148,992 = 256 × 582, `tokenizer.json` sha256 `bbb08e71…`), and it — not the old 153,600 bundle — is what subprojects 07/08 tokenized with. Evidence: [`PRODUCTION_DECISION_20260729.md`](03_4_implementation_experiments/polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md).

### 2026-06-16 → 2026-08-21 — afterlife as shared tooling

No new *bakeoff* experiment ran here after May. What kept changing was infrastructure other subprojects import: the peer-model GreekMMLU baseline launcher (`42c57a4d`, 2026-06-16), the native-MCQ runner and benchmark registry (`01cba0ee` 2026-07-12, `ba80bb0c` 2026-08-05, `5b6dd260` 2026-08-07), the polytonic-probe scripts that build 05's 148,992 production init (`f0dc31a0`, 2026-07-31 — `05_token_distillation_cpt/06_25b_midtraining_probe/initialization/build_production_init.sbatch` calls `build_incremental_checkpoint.py` → TD layer-11 → `calibrate_new_output_rows.py`), and roughly a dozen patches to `bakeoff_training/bakeoff_train.sbatch`, which became the trainer for the curriculum sweeps, the full-8B run and the targeted experiments (last touched `7dce0efb`, 2026-08-21).

## Outcome

- **Four arms, three budgets, no adjudicated winner.** Final endpoints: Vanilla and TD-layer11 at 5.000 B (iter 1192), ReTok at 3.498 B (iter 834), Centroid at 1.996 B (iter 476). Centroid was eliminated (BPB 0.8994 vs ≤0.53); ReTok was dominated by TD at every shared iteration.
- **The winner depends on the metric axis.** At 5 B: TD leads downstream aggregates, Vanilla leads BPB, Vanilla leads native Greek MCQ. TD's Greek-aggregate lead is carried by one task — drop `xquad_el` (+7.57 pp) and Vanilla is narrowly ahead on the remaining four. Sources: `BAKEOFF_FINAL_RESULTS_20260526.md`, `NATIVE_GREEK_SUITE_RESULTS_20260526.md`.
- **Every arm lost Greek capability relative to base Apertus.** V4 base reference ≈ 0.525 Greek aggregate vs 0.41–0.42 for the best arms; native MCQ 0.4817 vs 0.4305 best continued. This became the central question handed to subproject 04.
- **The decision rule was never instantiated.** `CPT_MASTER_20260526.md` §5 lists 6 HIGH-severity gaps: unlocked thresholds (D1), unmeasured per-language regression slices (D2), no V4 variance baseline (D3), the BPB-vs-downstream divergence (D4), decontamination not done (D5).
- **Production launcher built, never fired from here.** `production_cpt/submit_vanilla_base_15b_chain.sh` is dry-run validated (14-job chain, Goldfish loss, 15 B tokens) but gated on V1 / V4 / V8 / R17.
- **Carried forward:** the Vanilla-default question → 04 (which confirmed the regime, not the init, caused the native-Greek degradation); the TD challenger → 05; the trainer, the native-Greek MCQ runner and the polytonic init scripts → 05/06/07/08.
- **Left open at the end:** held-out contamination on the C3 val/test split (unrecoverable after the GCloud loss), the `{10K, 15K, 20K, 25K}` cutoff sweep, per-task confidence intervals on the 5 B headline, and the BPB truncation-bias re-check (Vanilla truncated 29.2 % of heldout docs vs TD's 24.8 %).

## Sub-subprojects

| Dir | Role | Period | Status | Result |
|---|---|---|---|---|
| [`03_1_greek_embedding_diagnostic/`](03_1_greek_embedding_diagnostic/README.md) | Geometry of Greek in Apertus's `E`/`U` matrices | 2026-05-12 → 05-18 | completed | Norm targets 5.05 / 3.80; Greek is a coherent subspace; no Greek↔English etymology bridge |
| [`03_2_apertus_c3_dedup_audit/`](03_2_apertus_c3_dedup_audit/README.md) | Overlap between Apertus pretraining and the Greek corpus pool | 2026-05-18 → 05-19 | completed, one gap | 2.27 % overlap; hard-drop overlay published; held-out check skipped |
| [`03_3_cscs_experiments_kickoff/`](03_3_cscs_experiments_kickoff/README.md) | Planning bridge: v0.12 → v0.7, auth, ship tokenizers | 2026-05-20 → 05-21 (its `ship/` gained a third bundle 2026-07-29) | completed | Two loadable ship bundles then; Clariden auth verified; no Slurm job ran here |
| [`03_4_implementation_experiments/`](03_4_implementation_experiments/README.md) | Everything that actually ran on Clariden | 2026-05-20 → 2026-08-21 | bakeoff completed 05-26; polytonic cutoff frozen 07-29; scripts reused to 08-21 | The 4-arm bakeoff, its eval stack, and the +512 polytonic production tokenizer |
| [`references/`](references/README.md) | Pinned primary sources for the recipe audit | 2026-05-21 | frozen | 8 repos at pinned commits + 15 papers |
| [`_archive/`](_archive/README.md) | Superseded plans, session logs, pre-5 B review material | archived 2026-06-11 | historical | 4 groups, nothing load-bearing |

## Where things are

| What | Where |
|---|---|
| Canonical synthesis + discrepancy log | [`CPT_MASTER_20260526.md`](CPT_MASTER_20260526.md) |
| Final 4-arm result | [`.../trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md`](03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md) |
| Greek-headline correction | [`.../eval/NATIVE_GREEK_SUITE_RESULTS_20260526.md`](03_4_implementation_experiments/init_bakeoff/eval/NATIVE_GREEK_SUITE_RESULTS_20260526.md) |
| Loss-reading rule | [`.../eval/LOSS_MEASUREMENT_POLICY.md`](03_4_implementation_experiments/init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md) (repo-wide copy at [`docs/LOSS_MEASUREMENT_POLICY.md`](../../docs/LOSS_MEASUREMENT_POLICY.md)) |
| Training spec + fidelity deviations | [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md), `CPT_MASTER` §3 |
| Ship tokenizers (loadable) | [`03_3_cscs_experiments_kickoff/ship/`](03_3_cscs_experiments_kickoff/ship/) — `apertus_greek_modern_only_148480/` (bakeoff), `apertus_greek_extended_153600/` (historical polytonic specialization), **`apertus_greek_modern_polytonic_148992/`** (production, frozen 2026-07-29) |
| Polytonic cutoff decision + gate table | [`.../polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md`](03_4_implementation_experiments/polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md) |
| Model weights, extended tokenizer, benchmark summaries | Hugging Face [`fffoivos/apertus-tokenizer-extension`](https://huggingface.co/fffoivos/apertus-tokenizer-extension) — `experiment-checkpoints/`, `greek-extension-tokenizer/`, `benchmark-evals/`, `supporting-material/` |
| Dedup overlay dataset | Hugging Face `fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z` |
| Training corpus, checkpoints, eval outputs | Clariden (`a0140`, user `fffoivos`): `/iopsstor/scratch/.../cpt_corpus/`, `/capstor/scratch/.../runs/bakeoff/`, `/capstor/scratch/.../runs/eval/` — full map in `CPT_MASTER` §7.2 |
| Trainer still used by 05–08 | [`.../bakeoff_training/bakeoff_train.sbatch`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/) |

## Working documents

Historical; kept for traceability, not for current decisions.

- **Plans / specs (top level):** [`TOKEN_DISTILLATION_PLAN.md`](TOKEN_DISTILLATION_PLAN.md) (the 4th-arm spec, 2026-05-22), [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) (audited 2026-05-21; its "production default" section was written before the 5 B and native-suite results), [`RISKS.md`](RISKS.md) (17 silent-failure risks; R17 is the one that materialised).
- **Stale status snapshot:** [`TODO.md`](TODO.md) — frozen at 2026-05-23, before the 3.5 B/5 B continuations; its links to `PRODUCTION_DECISION_STATE.md` point at the pre-archive location and it contains absolute paths from the original machine.
- **Archive:** [`_archive/`](_archive/README.md) — `synthesis_sources_20260526/` (the 10 docs `CPT_MASTER` replaced), `v0.6_planning/`, `2026-05-21_overnight_session/` (4 operational logs, ~219 KB), `2026-05-24_2B_bakeoff_review/` (reviewer material whose "Vanilla wins" conclusion the continuations overturned).
- **References:** [`references/MANIFEST.md`](references/MANIFEST.md) — pinned commits, paper list, citation convention. Repos are gitignored and rebuilt by `clone_references.sh`.
- **Late arrivals:** the whole `polytonic_cutoff_probe/` directory, the `ship/apertus_greek_modern_polytonic_148992/` bundle and the 2026-07-29 banner on `SHIP_TOKENIZER_RECONSTRUCTION.md` were never committed during the work; they were recovered from the owner's working tree on 2026-09-01 (`2aec4a66`, "Recover uncommitted working-tree files"). Their content dates from 2026-07-29 even though their commit date does not.
