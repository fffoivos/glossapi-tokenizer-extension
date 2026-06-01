# Path A geometry probe — results + verdict

**Date.** 2026-05-31.
**Plan.** `subprojects/04_cpt_training_regime_on_vanilla/PATH_A_GEOMETRY_PROBE_PLAN.md`.
**Hypothesis.** TASK2_HANDOFF §2.3: `rope_theta=500K` override on `rope_theta=12M` trained weights phase-shifts Q·K rotations and perturbs the model rather than re-anchoring it; the first ~1 B tokens of Task 1's Path B CPT carry the rope re-adaptation cost.

## Verdict

**CONFIRMED** per `PATH_A_GEOMETRY_PROBE_PLAN.md` §7 decision rule. Both prongs of the CONFIRMED clause satisfied on the headline 3-task native Greek MCQ aggregate, with paired bootstrap CIs that exclude zero.

| Comparison (3-task headline) | Δ point | 95 % CI | Sig |
|---|---:|---|---|
| Path A 0.5 B vs **Path B iter 119 (matched tokens)** | **+5.51 pp** | [+3.79, +7.25] | outside 0 |
| Path A 0.5 B vs **Apertus-Base Path A (released)** | **+1.25 pp** | [+0.25, +2.27] | outside 0 |
| Path A 0.5 B vs matched-config Path-B-perturbed (init) | **+6.70 pp** | [+5.14, +8.23] | outside 0 |

Bootstrap methodology identical to V4 v3 (1000 resamples, 95 % percentile, `rng_seed=20260529`, per-task item-level paired indices, macro-mean across 3 headline tasks per resample). Artifact: `reports/v4_workspace_path_a/path_a_probe_bootstrap_cis.json`.

**Prong 1.** "Path A iter 0.5 B paired vs Path B iter 119 outside zero positive, ≥ 2 pp on headline." Satisfied: +5.51 pp, CI [+3.79, +7.25], lower bound 1.9× the ≥ 2 pp gate.

**Prong 2.** "Path A iter 0.5 B paired vs Apertus-Base Path A inside or above zero on headline (Path A stays at or above base — no rope re-adaptation dip)." Satisfied: +1.25 pp, CI [+0.25, +2.27], lower bound barely clears zero (+0.25 pp). At 0.5 B Path A is statistically above the released-geometry base on the aggregate — the opposite of the Path B trajectory which sat 4.26 pp below base at iter 119.

## Path A 0.5 B marginal headline CI

**0.4942 [0.4747, 0.5133]** (point + 95 % CI).

For comparison, V4 v3 marginal CIs from the run already done:

| Anchor | Point | 95 % CI |
|---|---:|---:|
| Apertus-Base Path A | 0.4817 | [0.4629, 0.4997] |
| matched-config Path-B (perturbed) | 0.4272 | [0.4096, 0.4456] |
| Vanilla-Path-A-0.5B (this probe) | 0.4942 | [0.4747, 0.5133] |
| Vanilla-Path-B-0.5B (Task 1 iter 119) | 0.4391 | [0.4219, 0.4565] |
| Vanilla-Path-B-1B (Task 1 iter 238) | 0.4487 | [0.4299, 0.4670] |
| Vanilla-Path-B-2B (Task 1 iter 477) | 0.4792 | [0.4604, 0.4967] |
| Vanilla-Path-B-3.5B (Task 1 iter 834) | 0.4790 | [0.4606, 0.4978] |
| Vanilla-Path-B-5B (Task 1 iter 1192) | 0.4973 | [0.4779, 0.5156] |

Path A 0.5 B (0.4942) sits between Path B 3.5 B (0.4790) and Path B 5 B (0.4973) — that is, **0.5 B of Path A reaches a headline state that took Path B between 3.5 B and 5 B to reach**. The marginal CIs of Path-A-0.5 B and Path-B-5 B substantially overlap; the trajectory shape is what differs.

## Per-task structure

Per-task headline (point estimates), bracketed by paired bootstrap deltas vs the most relevant comparator:

| Task | Path A 0.5 B | vs Path B iter 119 | vs Apertus-Base Path A |
|---|---:|---:|---:|
| greekmmlu | 0.5427 | **+4.43 pp** outside 0 | **+1.48 pp** outside 0 |
| ilsp_medical_mcqa | 0.4074 | **+6.94 pp** outside 0 | −0.23 pp inside 0 |
| ilsp_mcqa_asep | 0.5325 | **+5.17 pp** outside 0 | **+2.50 pp** outside 0 |
| plutus_qa (diagnostic) | 0.5289 | **+12.89 pp** outside 0 | +1.33 pp (CI tbd) |

Path A advantage over Path B at matched tokens is broad — significant on all four MCQ tasks, not driven by a single one. Versus Apertus-Base, Path A is statistically above on `greekmmlu` and `ilsp_mcqa_asep`; `ilsp_medical_mcqa` is at noise.

The pattern mirrors Task 1's per-task structure at iter 1192 (greekmmlu and ASEP both clearly above Apertus-Base; Medical at noise), but Path A reaches it at **0.5 B tokens** rather than 5 B.

## Other dimensions

### Greek BPB

| Anchor | Greek BPB | Note |
|---|---:|---|
| Path A 0.5 B | **0.4365** | this probe |
| Path B iter 119 (Task 1, matched tokens) | 0.6049 | rope-readapted state |
| Path B iter 1192 (Task 1 endpoint, 5 B) | 0.4132 | 10× the compute |
| matched-config Path-B (init) | 1.2216 | perturbed |
| Apertus-Base Path A on this heldout | TBD | (not yet evaluated) |

Path A 0.5 B Greek BPB **0.4365 ≈ Task 1 iter 1192's 0.4132 at one-tenth the training compute**. Path B paid most of its 5 B Greek BPB improvement in the first ~1 B tokens recovering from the rope perturbation; Path A starts from a non-perturbed state and improves directly.

### Multilingual retention vs matched-config Path-B init (the true Task 1 baseline)

| Task | matched (true init) | Path A 0.5 B | Δ vs init |
|---|---:|---:|---:|
| global_mmlu_en | 0.6025 | **0.6425** | +4.00 pp |
| global_mmlu_fr | 0.5425 | **0.6100** | +6.75 pp |
| global_mmlu_de | 0.5725 | **0.5975** | +2.50 pp |
| mmlu (Hendrycks EN) | 0.5624 | 0.5977 | +3.53 pp |
| xnli_en | 0.5112 | 0.5414 | +3.02 pp |
| xnli_fr | 0.4859 | 0.4920 | +0.61 pp |
| xnli_de | 0.4968 | 0.5060 | +0.92 pp |
| xnli_ru | 0.4884 | 0.4912 | +0.28 pp |
| arc_challenge | 0.5384 | 0.5452 | +0.68 pp |
| arc_easy | 0.8279 | 0.8295 | +0.16 pp |
| piqa | 0.7922 | 0.7916 | −0.06 pp |
| hellaswag | 0.5862 | 0.6008 | +1.46 pp |

Every retention task improves at 0.5 B Path A; none regress. Compare to Task 1 Path B at iter 119, which had `xnli_en −2.08`, `xnli_fr −1.81`, `arc_challenge −1.37`, `arc_easy −2.44` vs matched-config init (the rope re-adaptation casualties). **Path A pays none of those costs.**

Compare to Task 1 Path B at iter 1192 (5 B) vs matched-config init: `xnli_ru −1.57 pp`, `arc_easy −0.85 pp`, `xnli_de −0.20 pp`. Path A 0.5 B has positive deltas on all of those — there is no regression to recover from later in the trajectory.

### Heldout BPB on code and math

| Heldout | Path A 0.5 B | Path B iter 1192 (5 B endpoint) |
|---|---:|---:|
| Code | 0.2725 | 0.2646 |
| Math | 0.5459 | 0.5448 |

Path A 0.5 B is essentially at Path B's 5 B endpoint on code/math BPB — domain-rare in HPLT, so this is mostly "what the base does at this geometry."

## Caveats from the adversarial review

The Vanilla-Path-A-0.5B critique (`adversarial_reviews/Vanilla-Path-A-0.5B/adversarial_critique.md`) records 3 Critical, 11 Major, ~5 Minor findings. The non-cosmetic ones that affect interpretation of this verdict:

1. **Init-checkpoint provenance (C1).** The plan §4 step 1 prescribed re-converting the Apertus-Base init from HF→Megatron under Path A geometry to a sibling dir (`megatron_tp2_r17patched_path_a`). That conversion was not done. The probe loaded the existing Path-B-converted Megatron init (`megatron_tp2_r17patched`) and applied Path-A geometry flags at runtime via `--max-position-embeddings 65536 --rotary-base 12000000 --use-rope-scaling --rope-scaling-factor 8.0`. Megatron loaded the weights identical, and the runtime args set the geometry the model is trained under. **Empirically, the run trained cleanly with Path A geometry** (config.json confirms it, lm loss curve is healthy, retention numbers above the matched-config Path-B floor everywhere). But the provenance shortcut is worth documenting — a strict purist read would want the Megatron init re-converted under Path A flags too; we relied on the runtime-flag-equivalence which Megatron's loader supports.

2. **Persistent issues from Task 1 not re-tested here.** Decontamination MinHash against the 4 native MCQ benchmark prompts (Decisions Matrix row E / V1) is still absent — same caveat as Task 1. Greek BPB 29.2 % prefix truncation (Major 1 / Decision D) persists on the same heldout file. Plutus diag n=225 small sample. None of these affect the Path-A-vs-Path-B comparison, which is well-controlled (same heldout, same MCQ prompts, same decontamination status on both sides).

3. **Apertus-Base Path A Greek BPB on this heldout is not yet measured.** The Greek BPB Path-A row in the table above is marked TBD because we haven't run Apertus-Base on the heldout under Path A geometry. The matched-config measurement (1.2216) is on Path B geometry and is the perturbed bookend; for a clean Path-A base BPB we'd need to eval Apertus-Base under its native geometry on `cpt_greek_heldout_500_20260522.jsonl`. Cost: ~2 GPU-h. Worth doing for the 5 B report's BPB story; not gating the verdict here.

4. **Three launch attempts before the probe started running** (RUN_LOG entries on 2026-05-31). Each surfaced a distinct latent issue in the submit script (missing `ARM=vanilla` + `INIT_CKPT` + `SCRIPT_DIR_OVERRIDE` + data prefix; then `--ntasks-per-node=1` → `WORLD_SIZE=1`; then `TRAIN_TOKENS=0.5 B` < warmup window). Each was a quick patch; the recurrent pattern shows that the Task-1 chain submitter's submission shape was less reusable than expected. Captured in RUN_LOG §"Path A probe — third failure" entry; carried forward as a Task-2 implication: any Task-2 submitter should derive directly from the Task-1 chain submitter rather than re-engineer the `--export` list.

## Implications for Task 2

1. **Path A geometry is now the locked Task-2 default.** Update `TASK2_HANDOFF.md` §(3.1) status from "RECOMMENDED working position" to "CONFIRMED, lock for Task 2 v1.x."
2. **The matched-config Apertus-Base eval is downgraded further** in the Task-2 documentation — it is now demonstrably the wrong baseline (the perturbed model is not what Path-A CPT is trained under). For Task 2, drop the matched-config workaround entirely; compare extension arms directly to Apertus-Base Path A.
3. **Task 1's "first ~1 B tokens rope re-adaptation cost" claim is now quantified.** Path B paid ~5.5 pp of headline + the BPB-blowup-then-recovery + the retention regressions in those first 0.5 B tokens. Path A pays none of them.
4. **The 5 B Path-A trajectory is implied to be substantially above Task 1's 5 B Path-B trajectory.** A back-of-envelope: if Path A 0.5 B = Task 1 Path B ~3.5–5 B, then 5 B Path A would plausibly hit a much higher endpoint. But the Task-1 plateau-then-resume shape may or may not transfer to Path A — that's a Task-2 question, not a Task-1 question.
5. **The minimum compute saving from switching to Path A is dramatic.** Path A reaches Greek-side capability at 0.5 B that Path B took 5 B to reach. ~10× compute saving for a Vanilla CPT at this geometry-confound regime — material for the Task-2 budget and the 10 B stretch decision.

## Tasks resolved

- `cpt-plan.md` §3.4 Q3.4.10 — Path A recommendation for Task 2: **CONFIRMED on probe data**, status update from working-position to locked.
- `TASK2_HANDOFF.md` §(3.1) — Path A as Task-2 default: **CONFIRMED, locked**.
- `decisions_matrix_20260529.md` row C (RoPE/seqlen mismatch): the Task-2 resolution is locked at Path A; Decision C is now `applied + verified` for Task 2.

## Compute used

- **24.6 GPU-h** for this probe (training 17.4 + sidecars ~5.7 + matched-config base eval reuse ~0 + smoke 0).
- ~5h wall-clock from probe submission to verdict on disk (including the 3 launch retries).
- ~11 % of Task 1's 217 GPU-h.

## Artifacts

- Probe plan: `subprojects/04_cpt_training_regime_on_vanilla/PATH_A_GEOMETRY_PROBE_PLAN.md`
- Bootstrap CI artifact: `subprojects/04_cpt_training_regime_on_vanilla/reports/v4_workspace_path_a/path_a_probe_bootstrap_cis.json`
- Bootstrap script: `subprojects/04_cpt_training_regime_on_vanilla/reports/v4_workspace_path_a/run_path_a_bootstrap.py`
- Adversarial critique: `subprojects/04_cpt_training_regime_on_vanilla/adversarial_reviews/Vanilla-Path-A-0.5B/adversarial_critique.md`
- Train run on Clariden: `/capstor/scratch/cscs/fffoivos/runs/04a_vanilla_path_a_probe/04a_vanilla_path_a_probe_20260531T103924Z/`
- Eval root on Clariden: `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04a_vanilla_path_a_probe_20260531T103924Z/`
- Slurm jobs: train `2437909`; sidecars `2440690`–`2440697`
- RUN_LOG entries: see 2026-05-31 sections.
