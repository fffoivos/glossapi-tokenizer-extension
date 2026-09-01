# init_bakeoff — the 4-arm embedding-initialisation experiment

> **In one line:** four ways of giving Apertus-8B 17,408 new Greek embedding rows (Vanilla = none, ReTok, Centroid, Token-Distillation layer 11), trained on an identical Greek-heavy corpus to 2 B → 3.5 B → 5 B tokens on Clariden, with the whole supporting stack — corpus build, Megatron patches, eval harness — built alongside.
> **Period:** 2026-05-20 (`af438d4d`, arms + plan) → 2026-05-26 (final results). Scripts kept being patched for other subprojects until 2026-08-21.
> **Status:** completed. Centroid eliminated, ReTok dominated, **TD vs Vanilla never adjudicated** because the decision thresholds were never pre-committed.
> **Came from / led to:** [`../../03_3_cscs_experiments_kickoff/`](../../03_3_cscs_experiments_kickoff/README.md) → this → [`../../../04_cpt_training_regime_on_vanilla/`](../../../04_cpt_training_regime_on_vanilla/) (Vanilla) and [`../../../05_token_distillation_cpt/`](../../../05_token_distillation_cpt/) (TD).

## Why this existed

Vocabulary extension costs 142.6 M extra parameters per matrix pair (17,408 × 4,096 × 2) and only pays off if the new rows learn something the base tokenizer cannot express. The bakeoff was designed as a clean A/B: same corpus, same seed, same schedule, same engine — the only differential is how the new rows start (and, for Vanilla, whether they exist at all). Plan of record: [`BAKEOFF_PLAN.md`](BAKEOFF_PLAN.md).

## History

### Setup — 2026-05-20 → 2026-05-21

The three closed-form arms were written and smoke-tested locally (`af438d4d`), then the corpus and eval tooling followed (`11b5ba00`). The 2026-05-20 scope decision dropped the polytonic layer and pinned the **modern-only 148,480** tokenizer, on the reasoning that polytonic rows would be undertrained at 2 B tokens anyway.

The 2026-05-21 recipe audit against pinned sources found the blocker that shaped everything after: `swiss-ai/Megatron-LM` has **no HF→Megatron loader for Apertus**. One was written ([`megatron_patches/`](megatron_patches/README.md)), and its first roundtrip (job `2333864`) proved the standard tensors convert bit-exactly but **128 tensors — 32 layers × 4 xIELU params — silently reset to defaults**. That is risk **R17**, and it is not theoretical: an unpatched roundtrip drops `arc_easy` from 0.8363 to 0.2614, i.e. to chance ([`eval/V4_BENCHMARK_COMPARISON.md`](eval/V4_BENCHMARK_COMPARISON.md)). All three arms were re-converted through `patch_apertus_extras.py` and verified at zero drift (jobs `2341182` / `2341239` / `2341241`).

The corpus build took most of the overnight session: prepare-pool → NFC normalise → token-fair mix → concat → Megatron preprocess, with the mix recipe drifting and being steered back to **70 % Greek / 24 % replay / 4 % code / 2 % math** twice. Code fell back to `codeparrot/codeparrot-clean-train` because BigCode's StarCoder sources were gated ([`corpus_build/README.md`](corpus_build/README.md)).

### The 2 B bakeoff — 2026-05-22

`bakeoff_1node_chain_20260522_005620` ran Vanilla / ReTok / Centroid on one node each (4 × GH200, `normal`, 12 h, chained for walltime) to iter 476 ≈ 2.0 B tokens. A checkpoint watcher converted each saved checkpoint to HF and fired packed eval sidecars. Centroid was broken from the first eval (BPB 1.1318 at iter 130 vs 0.5432 for Vanilla) and was never continued.

Result at iter 476 — **Vanilla wins 3 of 4 metrics**: Greek no-MT 0.4131 / TD 0.4048 / ReTok 0.3906 / Centroid 0.2566; BPB 0.4906 / 0.5311 / 0.5739 / 0.8994.

### The fourth arm — 2026-05-23 → 2026-05-24

Token Distillation ran its own gated ladder in [`token_distillation/`](token_distillation/README.md) and produced `td_full25_layer11`, which then trained a full 2 B arm. At its own iter 476 it beat ReTok and Centroid on every intrinsic but still lost the aggregate Greek comparison to Vanilla, so `PRODUCTION_DECISION_STATE.md` picked Vanilla.

### The continuations, and the double reversal — 2026-05-24 → 2026-05-26

- **3.5 B** (`continuation_3p5b_20260524T143012Z`, three arms, three chained segments each): at iter 834 TD leads Greek no-MT by +1.40 pp. First reversal.
- **5 B** (`continuation_5b_td_vs_vanilla_20260525T142522Z`, Vanilla + TD only, jobs `2382982`–`2382985`): at iter 1192 TD leads all three downstream aggregates; Vanilla keeps BPB with the gap narrowed 0.110 → 0.027 and crossover extrapolated to ~6.5–6.8 B tokens.
- **2026-05-26, the native-Greek suite** ([`eval/NATIVE_GREEK_SUITE_RESULTS_20260526.md`](eval/NATIVE_GREEK_SUITE_RESULTS_20260526.md)): on vetted native Greek MCQ, Vanilla-5B 0.4305 beats TD-5B 0.4109, and **Apertus-Base 0.4817 beats every continued checkpoint**. Second reversal — the Greek headline goes back to Vanilla, and the honest reading becomes "the CPT regime is costing Greek capability".

ReTok was stopped at 3.5 B (TD-dominated). The 5 B result was written up in [`eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md`](eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md), which explicitly flags that TD's Greek lead is carried by `xquad_el` (+7.57 pp) and disappears without it.

## Outcome

- **Ranking is solid, selection is not.** The 2 B budget was enough to rank the arms (Centroid broken, ReTok dominated) but not to choose between TD and Vanilla; the 3.5 B and 5 B extensions were load-bearing for the reversal and still did not settle it, because no thresholds were locked.
- **BPB and downstream disagree.** Same arms, opposite winners on different metric axes — a divergence the governing plan did not anticipate (`../../CPT_MASTER_20260526.md` §4.4, discrepancy D4).
- **New-token rows plateau early.** TD's probability mass on new-token targets is flat at ~0.342 from iter 476 to 1192, so TD's gains past 2 B come from base-vocab adaptation, not further new-row training.
- **Production never launched from here** — `production_cpt/` is dry-run validated and gated on V1 / V4 / V8.
- **Known measurement caveat left open:** Vanilla truncates 29.2 % of held-out docs at 4,096 context vs TD's 24.8 %, so part of the 0.027 BPB gap may be methodological.

## Sub-subprojects

| Dir | Role | Period | Status | Result |
|---|---|---|---|---|
| [`arms/`](arms/README.md) | The four init methods + Clariden build/convert pipeline | 2026-05-20 → 05-21 | completed | Three HF checkpoints built and converted; ReTok and Centroid produce near-orthogonal rows (mean cos ≈ 0.03) |
| [`corpus_build/`](corpus_build/README.md) | The 70/24/4/2 mix, NFC, Megatron preprocessing, token accounting | 2026-05-20 → 05-26 | completed | 5,754,172-row mix; 9.83 B base-tokenized tokens |
| [`megatron_patches/`](megatron_patches/README.md) | HF→Megatron Apertus loader, R17 patcher, roundtrip verifier | 2026-05-21 → 05-23 | completed | R17 found and fixed; zero-drift roundtrips for all arms |
| [`bakeoff_training/`](bakeoff_training/README.md) | Trainer, config, chained submitters, run logs | 2026-05-21 → 2026-08-21 | bakeoff done; trainer reused by 05–08 | 21 training jobs across 2 B / 3.5 B / 5 B |
| [`eval/`](eval/README.md) | V4 baselines, tokenizer-fair metrics, new-token diagnostics, native-Greek suite, trajectory analysis | 2026-05-21 → 2026-08-07 | completed | The evidence base for every claim above |
| [`token_distillation/`](token_distillation/README.md) | The 4th arm: coverage gate, layer pilot, full TD, preservation | 2026-05-23 | completed | `td_full25_layer11`, layer 11 chosen over the paper default |
| [`production_cpt/`](production_cpt/README.md) | 15–20 B Vanilla production launcher | 2026-05-24 | prepared, never launched | 14-job dry-run chain validated |
| `release_upload/` | One script, `upload_release_checkpoints_to_hf_from_clariden.sh` — pushes checkpoints to the HF release repo | 2026-05-25 | utility | no README of its own |

## Where things are

| What | Where |
|---|---|
| Plan of record | [`BAKEOFF_PLAN.md`](BAKEOFF_PLAN.md) — arms table, fidelity constraints, Slurm shape, pre-Clariden checklist |
| Canonical result | [`eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md`](eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md) |
| Greek-headline correction | [`eval/NATIVE_GREEK_SUITE_RESULTS_20260526.md`](eval/NATIVE_GREEK_SUITE_RESULTS_20260526.md) |
| How to read loss | [`eval/LOSS_MEASUREMENT_POLICY.md`](eval/LOSS_MEASUREMENT_POLICY.md) — raw `lm loss` is not cross-tokenizer fair; use heldout BPB; historical `BPC` = BPB |
| CPU-partition guard | [`check_cpu_only_slurm.sh`](check_cpu_only_slurm.sh) — run before any dataset/conversion submit; forbids GPU directives on `xfer` jobs |

## Working documents

- [`BAKEOFF_PLAN.md`](BAKEOFF_PLAN.md) is a 2026-05-20 plan, not a record: its §6 checklist has open boxes that were later closed, its §7 "forthcoming sbatch templates" were superseded by the single parameterised `bakeoff_train.sbatch`, and its arm table describes three arms because TD did not exist yet.
- The previous version of this README described the bakeoff as three arms at 2 B tokens and listed `bakeoff_training/` as "the missing piece"; that state is what the history above supersedes.
