# 08 — Targeted 8B CPT experiments (the hard HPLT→OpenArchives cross-scale study)

> **In one line:** two targeted 8B continuation experiments were designed and both shelved, then replaced by a matched 8B + 1.5B replication of the historical "hard HPLT→OpenArchives" curriculum — which trained to completion at both scales and returned three negative-or-qualified answers: the historical 59.96% GreekMMLU headline was **not** replicated (best 57.90%, a 2.06 pp miss against a pre-registered ±1.0 pp band), 1.5B does **not** mirror the 8B trajectory, and holding the peak LR constant made the curve fall rather than rise.
> **Period:** 2026-08-11 → 2026-08-23 (doc dates; first commit 2026-08-16 — see "A note on the git dates").
> **Status:** completed; final report in [`presentations/hard_h2g_full_panel_stable_lr_20260822/`](presentations/hard_h2g_full_panel_stable_lr_20260822/), review integration opened as `train-apertus-with-glossapi` PR #17 (`d364dc45`).
> **Came from / led to:** [`05_token_distillation_cpt`](../05_token_distillation_cpt/) (the β₂ curriculum arm that became the replication target) and [`07_full_8b_cpt`](../07_full_8b_cpt/) (inherited recipe, DP32 geometry, parent checkpoints) → this → [`09_full_8b_cpt_results_analysis`](../09_full_8b_cpt_results_analysis/), which names this subproject as its follow-up track. The same "is the peak an artifact of LR-decay timing" question is tested causally on a *different* run in [`10_early_cooldown_causal_experiment`](../10_early_cooldown_causal_experiment/).

## Why this existed

Subproject 05 had produced a curriculum sweep whose selected β₂ arm reached 59.94% on GreekMMLU with a *hard* switch from HPLT to GlossAPI/OpenArchives data. Two questions followed. Is that headline reproducible at all, given that the packed binaries, the raw Greek pool, the decontamination queries and the per-question predictions of the original run had all been deleted from CSCS scratch? And can a 1.5B model stand in for 8B when screening curricula, so that future data experiments stop costing 8B-sized allocations? This subproject answered both on a rebuilt benchmark-clean corpus, with everything else held matched across the two scales.

## History

### 2026-08-11 — two targeted experiments and a resource plan

[`CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md`](CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md) froze the shared constants (148,992 tokenizer, AdEMAMix 0.9/0.999/0.999 α=4, peak/floor LR 5.5e-5/5.5e-6, Goldfish k=h=50, 1,024-sequence global batch, TP2/DP32 on 16 nodes; DP64 prohibited for failing the trajectory-parity bound) and defined **A** — a stationary 79/20/1 academic + polytonic mixture over `openarchives.gr`, `greek_phd` and matched HPLT, ~6,092 updates — and **B** — continuing 07's update-9,536 checkpoint over every unseen non-HPLT sequence, 2,754 updates to a final update 12,290. Owner authorization for both is in [`configs/owner_authorization.json`](configs/owner_authorization.json).

### 2026-08-12 — B retired; the scale-predictivity proposal

The owner dropped B; pending jobs `3061757`/`3061758` were cancelled at `00:00:00`, before any allocation. Its builder was frozen instead ([`CONTINUATION_DATA_BUILDER_HANDOFF_20260812.md`](CONTINUATION_DATA_BUILDER_HANDOFF_20260812.md), `configs/continuation_data_builder_v1.json`) so a future mix experiment could reuse it without silently reinterpreting it. A's data preparation *did* run on `debug`: HPLT extraction `3058831` (12,156,522 candidate rows), decontamination `3059161` (215 GreekMMLU removals), audits `3059186`/`3060122`, leaving 12,152,435 HPLT rows; academic extraction retained 158,092 of 158,289 documents at exactly 10,000,949,141 tokens. B's schedule also completed (`3060140`: 9,123,187,023 unseen non-HPLT tokens, zero prefix overlap) and its 16-node restart-parity smoke was queued and re-queued twice for leaf placement — the surviving trace is [`evidence/`](evidence/). The replacement programme, [`SCALE_PREDICTIVITY_STUDY_20260812.md`](SCALE_PREDICTIVITY_STUDY_20260812.md) (0.5B/1.5B/8B matched trajectories plus a Stage-2 arm-selection test), was written the same day and never launch-authorized.

### 2026-08-14 — the hard-H2G plan and two adversarial reviews

[`HARD_H_TO_G_8B_1P5B_REPLICATION_PLAN_20260814.md`](HARD_H_TO_G_8B_1P5B_REPLICATION_PLAN_20260814.md) narrowed the programme to three goals: **A** replicate the 13.497B-token 8B hard-transition run; **B** run identical data and schedule at 1.5B and test trajectory mirroring; **C** extend OpenArchives by two ~1B-token intervals (3,218 → 3,456 → 3,694). The 0.5B arm, checkpoint averaging and the stationary-mix control were dropped.

Two 12-agent "ultracode" reviews then ran against it, each finding adversarially re-verified, several with a live `ssh clariden`:

- [R1](REVIEW_ULTRACODE_HARD_H_TO_G_PLAN_20260814.md) — 48 raw findings → 46 surviving (1 blocker, 19 major, 13 minor, 13 note), 78 plan claims verified correct. Its cluster-side discoveries reshaped the experiment: the historical arm actually ran **randomized** GPTDataset order per `run_metadata.json`, not the `physical_order` the plan's cited env chain implied; the `curriculum_v2` data stage was a 424K empty skeleton; the raw Greek pool, the TD snippet corpus and the historical per-question predictions were gone; the as-consumed Megatron TD init survived only as an HF payload.
- [R2](REVIEW_ULTRACODE_R2_HARD_H_TO_G_PLAN_20260814.md) — 46/46 prior findings fixed, no regressions; 29 fresh (2 blockers, 12 major, 10 minor, 5 note). The blockers were structural: goal C's "append unseen documents but change no pre-3,218 sequence identity" is unsatisfiable under a randomized GPTDataset (appending re-permutes from position 0), and GreekMMLU decontamination was designed as a join against Stage-A receipts *and* a queries file that were both deleted. Resolved by design change — a separate frozen Phase-3 blend consumed from cursor 0, and regenerated queries with a fresh scan over the rebuilt streams.

[`ULTRACODE_R2_REMEDIATION_20260814.md`](ULTRACODE_R2_REMEDIATION_20260814.md) tracks each disposition and the bundle churn: v8 superseded, v9 failed closed in job `3082594` on a validation-panel row-schema mismatch, v10 passed 85/85 tests (`3082694`) and froze all 13 panels at 59,749 rows / 59,742 unique text hashes (`3082837`).

### 2026-08-15 — the 1.5B TD initialization is blocked, then unblocked

The 1.5B Token-Distillation init failed its predeclared row-norm gate (job `3086780`). The diagnostic (`3087726`, 17 s on one debug node) is in [`1P5B_TD_ROW_NORM_DIAGNOSTIC_20260815.md`](1P5B_TD_ROW_NORM_DIAGNOSTIC_20260815.md): 0.0% of the 17,408 added rows fell inside the frozen 8B-derived bands, but the TD deltas against the *1.5B reference* init were tiny (+0.000253 input, +0.002702 output), and the 1.5B reference itself sits ~4.6%/9.2% below those bands — an architecture mismatch, not a TD blow-up. The gate was **not** relaxed after seeing the result; the arm stayed blocked. Its replacement, [`configs/1p5b_td_acceptance_policy_v2.json`](configs/1p5b_td_acceptance_policy_v2.json), is an architecture-local band referenced to the 1.5B ReTok init. Its in-repo status is still `proposal_pending_owner_approval`, yet the 1.5B trajectory demonstrably ran — the approval receipt was written on Clariden by `scripts/freeze_1p5b_td_policy_authorization.py`, not into this tree.

### 2026-08-16 → 08-21 — making it actually launch on Clariden

Everything above entered git in one squashed commit, `de6d9b79` "Prepare matched H-to-G campaigns without LR pilot" (2026-08-16), which also records dropping the separate 1.5B LR pilot. The next six days are almost entirely CSCS execution engineering while the two trajectories trained: cache-geometry and producer-authorization repairs and allocation-free qualification preflights on 08-16 (`49d04335`…`5caa614d`); evaluator hardening on 08-17 (see branch provenance); the published-v2 anonymization authority and verified suffix continuation on 08-18 (`23137366`, `b0ac17bb`…`f48d099e`); torchrun-teardown and nested-allocation fixes on 08-21 (`da21b9a2`, `f85599b0`, `9095539b`). Two decision documents came out of it:

- [`ALLOCATION_GEOMETRY_DECISION_20260818.md`](ALLOCATION_GEOMETRY_DECISION_20260818.md) picked the smallest exact-batch profiles that could still finish in 12 h — 1.5B on 2 nodes (TP1/DP8/mb4/acc32), 8B on 4 nodes (TP2/DP8/mb2/acc64) — and, correcting its own earlier arithmetic error, made promotion conditional on a *measured* budget gate `Q + 1.15·blocks·W + 1200 ≤ 43,200 s` rather than an extrapolation.
- `PHASE3_ASSET_RESOLUTION_LOG_20260819.md` (recovered 2026-09-01): the Phase-3 unseen selector stopped on 166 byte-identical foreign-replay duplicates (audit `3119120`; owner authorized keeping the first copy, excluding 123,659 tokens); a 54,135-token catalog mismatch was traced to `add_special_tokens=False` versus Megatron's per-document BOS; the three Phase-3 streams were then frozen (OpenArchives 54,135 docs / 2,403,602,497 tokens; foreign replay 1,908,795 / 1,459,450,175; Old Greek 623,182 / 552,904,950) and the blend cache built on the fourth attempt (`3119459`).

The last two edits the old README ever received are from 08-21: sentinel calibration moved to the finalization gate (`6dd4dd6e`) and audited cache overlays accepted by the cross-scale ledger gate (`78aae62c`). The same day the two TD initialization checkpoints were released to private HF repos — [`publication/`](publication/).

### 2026-08-22 — results, and a correction that changed the primary metric

`de52e1bc` reported the matched cross-scale trajectories; [`HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md`](HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md) is the record. 34 checkpoints (17 per scale) scored in one 4-node allocation `3148664` (`COMPLETED`, 03:05:13 ≈ 49.4 GPU-h), 549,406 checkpoint-question observations. **1.5B does not mirror 8B**: HPLT-boundary → OA-endpoint is −3.09 pp at 1.5B versus +1.27 pp at 8B; accuracy-level Pearson −0.6698; adjacent-change Pearson −0.0581; 29 of 31 subjects peak during HPLT at 1.5B while 22 of 31 peak during OpenArchives at 8B. The same document qualifies this: 1.5B's *choice NLL* improved while its accuracy fell.

The first report of that day was endpoint-only: [`presentations/greekmmlu_endpoint_20260822/`](presentations/greekmmlu_endpoint_20260822/) compares just the two update-3,694 checkpoints on the 16,159-question subset (8B 9,232/16,159 = 57.13%, 1.5B 6,429/16,159 = 39.79%). It was superseded within the day by the full 17-checkpoint trajectory report, and was among the files recovered on 2026-09-01.

It also records a self-correction. The streams had already been decontaminated at build time, so removing the 473 matching questions again from the evaluation panel was a second application of decontamination. The 16,159-question subset was demoted to a sensitivity analysis and the full public 16,632-question panel became the corrected primary (`776e1e31`, `825e60be`).

### 2026-08-22/23 — full-panel scoring and the no-decay branch

[`HARD_H2G_FULL_PANEL_SCORING_AND_NO_DECAY_BRANCH_PLAN_20260822.md`](HARD_H2G_FULL_PANEL_SCORING_AND_NO_DECAY_BRANCH_PLAN_20260822.md) was written and authorized the same day. Reading the train log first showed the published accuracy peak at update 2,618 sits *inside* early LR decay (LR still 5.5e-5 at 2,570; onset 2,571–2,580), which moved the branch point to `iter_0002499`, the last pre-decay save. Track A rescored all 17 existing 8B checkpoints on the full panel and re-ran the pinned legacy BF16 evaluator (code revision `cfdd0e7b`) on the three decision-bearing ones; Track B resumed from 2,499 at a constant 5.5e-5 to 3,218, producing checkpoints paired one-for-one with the decayed arm. The blow-by-blow, including every failure, is [`HARD_H2G_FULL_PANEL_AND_NO_DECAY_EXECUTION_NOTES_20260822.md`](HARD_H2G_FULL_PANEL_AND_NO_DECAY_EXECUTION_NOTES_20260822.md). The report package was completed on 08-23 (`75e4077b`) and the review linked (`d364dc45`).

### A note on the git dates

No commit dates exist before 2026-08-16; the 08-11 → 08-15 chronology comes from the documents' dated bodies and the CSCS job IDs they cite. At the other end, `2aec4a66` (2026-09-01, "Recover uncommitted working-tree files from local worktrees") committed `publication/`, `presentations/greekmmlu_endpoint_20260822/`, `PHASE3_ASSET_RESOLUTION_LOG_20260819.md`, most of `operational_workarounds/` and two `scripts/rebind_*.py` — work from 2026-08-19 → 08-22 that had simply never been committed.

### Branch provenance

Developed on the `agent/h2g-*` family, in order: `h2g-nopilot-20260816` → `h2g-safe-open-verifier-20260817` → `h2g-published-anonymization-adoption-20260818` (merged into the consolidation branch as `a4bb296f`) → `resize-qualified-profile` → `h2g-teardown-workaround-128` → `h2g-extension-gate-decoupling` → `h2g-greekmmlu-trajectories` (leaf, last commit `d364dc45`, 2026-08-23).

`h2g-safe-open-verifier-20260817` is the one branch whose 13 commits are **not** in the consolidation branch — they conflict with the later lineage. They were a one-day evaluator-plumbing pass: `85bbbd42` fix single-file safetensors verifier traversal; `4d019212` harden targeted checkpoint export parity evidence (float32 probability-parity patch + converter overlay); `14007043` allow verified historical sources in the bundle deploy; `6134dd73`/`7a5d2cca` make checkpoint export work under adopted `salloc` steps and resolve it from verified load views; `39fa0dc8`/`c0a86198` support adopted four-node panel evaluations and match the allocation to its nested Slurm groups; `5766a7c1`/`2380ce0a` fit document-panel and GreekMMLU evaluation inside Clariden's debug QoS; `944a4c4e` preserve document-panel receipt output bindings; `0f1ed8f7` bind GreekMMLU sentinel manifests to final paths; `1ca1f5a8`/`69da589c` add and fix the immutable GreekMMLU evaluator runtime extension. Equivalent capability reached the leaf branch by other routes; the branch is retained as history.

## Outcome

- **Goal A — replication: MISS, reported as a miss.** Best legacy-BF16 score 9,630/16,632 = **57.9004%** at update 2,618 against the β₂ arm's 9,973/16,632 = 59.9627% — **2.0623 pp** outside the ±1.0 pp band ratified *before* the result existed. The band was not widened afterwards.
- **Goal B — scale mirroring: NO.** 1.5B and 8B diverge in direction, not only level (−3.09 pp vs +1.27 pp across the HPLT→OA boundary; level correlation −0.6698). 1.5B is not a usable proxy for selecting this curriculum from GreekMMLU trajectory shape.
- **Goal C — extension: ran at both scales to update 3,694.** 8B HPLT validation loss rose +0.0180 during the OpenArchives phase and fell −0.0199 in the extension, reaching a new measured minimum; 1.5B recovered only −0.0019 of its +0.0260. Accuracy did not improve past the peak at either scale.
- **Exploratory no-decay branch: STOP at the pre-registered gate.** Constant 5.5e-5 from update 2,499 gave 58.2371% (2,618), 57.7922% (2,856), 56.7100% (3,094), 55.7359% (3,218) — falling across every interval and 1.3528 pp behind the paired decayed arm at the endpoint. The 3,219→3,694 continuation was **not** submitted.
- **Full-panel 8B trajectory (corrected primary):** peak 9,676/16,632 = 58.1770% at 2,618; 57.0887% at 3,218; 56.9865% at 3,694.
- **Never executed:** Experiment A training (its data prep completed and is receipted, but no A training run appears in any document or commit); Experiment B; the 0.5B cell and the Stage-2 arm-selection controls; and — all deferred by owner decision on 2026-08-22 — the 1.5B peak-LR sweep, the 1.5B full-panel rescore, update-0 evaluation, and a decay-from-best-stable-point branch.
- **Compute for the final phase:** `3151839` 55.64 GPU-h, `3152592` 183.98 GPU-h (against a ~112 GPU-h estimate; training itself took ~2:06 of 2:52), `3153569` 0.23 GPU-h (mis-shaped, relinquished), `3153706` 3.97 GPU-h, plus `3148664`'s ~49.4 GPU-h for the subset trajectory.
- **Carried upstream:** ten reusable failure modes filed against `fffoivos/apertus-cscs-efficiency` — issues #61 (comment), #88, #128, #136, #137, #146 (with PR #148), #147, #149, #150, #151, #152.

## Sub-subprojects

| Dir | Role | Period | Status | Result |
| --- | --- | --- | --- | --- |
| [`clariden/`](clariden/) | 153 Slurm entry points: data prep, gates, training, evaluation | 08-16 → 08-22 | complete | the executable surface of the study |
| [`scripts/`](scripts/) | 136 builders, freezers, auditors, preflights | 08-16 → 08-22 | complete | every receipt is produced here |
| [`evaluation/`](evaluation/) | GreekMMLU/panel scorers, checkpoint export + parity | 08-16 → 08-22 | complete | full-panel, subset and legacy-BF16 evaluators |
| [`configs/`](configs/) | frozen machine contracts | 08-16 → 08-22 | complete | `hard_h_to_g_replication_v1.json` is the authority |
| [`tests/`](tests/) | 11 contract/gate test modules | 08-16 → 08-22 | complete | regression net for the fail-closed gates |
| [`presentations/`](presentations/) | three report packages | 08-22 → 08-23 | complete | endpoint → trajectory → full-panel + stable-LR |
| [`publication/`](publication/) | private HF release of the two TD init checkpoints | 08-21 | complete | 8B and 1.5B 148,480-vocab inits published |
| [`operational_workarounds/`](operational_workarounds/) | experiment-local adapters for live allocations | 08-19 → 08-23 | complete | each has an upstream issue |
| [`evidence/`](evidence/) | two Experiment-B allocation-transition receipts | 08-12 | superseded | only surviving trace of the retired B smoke |
| [`patches/`](patches/) | one inherited-trainer patch | 08-16 | complete | benchmark-range offset for 07's segment script |
| [`runtime_compat/`](runtime_compat/) | `sitecustomize.py` shim | 08-16 | complete | pinned Megatron DCP semantics on torch 2.9.1 |

## Where things are

| What | Where |
| --- | --- |
| Final report (primary result) | [`presentations/hard_h2g_full_panel_stable_lr_20260822/`](presentations/hard_h2g_full_panel_stable_lr_20260822/) |
| Blow-by-blow execution record | [`HARD_H2G_FULL_PANEL_AND_NO_DECAY_EXECUTION_NOTES_20260822.md`](HARD_H2G_FULL_PANEL_AND_NO_DECAY_EXECUTION_NOTES_20260822.md) |
| Cross-scale result and contract summary | [`HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md`](HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md) |
| Machine authority / legacy-evaluator contract | [`configs/hard_h_to_g_replication_v1.json`](configs/hard_h_to_g_replication_v1.json), [`configs/legacy_public_greekmmlu_v1.json`](configs/legacy_public_greekmmlu_v1.json) |
| Trainer and its gate | [`clariden/train_hard_h_to_g_segment.sbatch`](clariden/train_hard_h_to_g_segment.sbatch), [`scripts/preflight_train_segment.py`](scripts/preflight_train_segment.py) |
| CSCS stage root (runs, receipts, checkpoints) | `/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14` |
| Published TD inits (private) | `fffoivos/apertus-8b-token-distillation-init-148480`, `fffoivos/apertus-1p5b-token-distillation-init-148480` |

## Working documents

Historical; none is a current instruction. **Plans:** `CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md` (A never trained, B retired), `SCALE_PREDICTIVITY_STUDY_20260812.md` (design-only, superseded), `HARD_H_TO_G_8B_1P5B_REPLICATION_PLAN_20260814.md` (the plan that executed), `HARD_H2G_FULL_PANEL_SCORING_AND_NO_DECAY_BRANCH_PLAN_20260822.md`. **Reviews and dispositions:** `REVIEW_ULTRACODE_HARD_H_TO_G_PLAN_20260814.md` (157 KB), `REVIEW_ULTRACODE_R2_HARD_H_TO_G_PLAN_20260814.md` (164 KB), `ULTRACODE_R2_REMEDIATION_20260814.md`. **Handoffs and diagnostics:** `CONTINUATION_DATA_BUILDER_HANDOFF_20260812.md`, `1P5B_TD_ROW_NORM_DIAGNOSTIC_20260815.md`, `ALLOCATION_GEOMETRY_DECISION_20260818.md`, `PHASE3_ASSET_RESOLUTION_LOG_20260819.md`, `HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md`. **Operational log:** `HARD_H2G_FULL_PANEL_AND_NO_DECAY_EXECUTION_NOTES_20260822.md` — append-only, and the single best source for what actually happened on Clariden.
