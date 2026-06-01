# Reply to the planning agent — Path-A probe + post-handoff updates

**Date.** 2026-06-01.
**Predecessor.** `TASK2_HANDOFF.md` (2026-05-30, Task-1 conclusion).
**Audience.** Whoever drafts Task 2 v1.x. This document is a *supplement*
to the original handoff, not a replacement; the handoff still stands and
the items below either add to it or update specific sections.

---

## TL;DR

Seven concrete updates since the original handoff:

1. **Corrected hyperparameters are the central finding.** The most important
   Task-1 result is that the Apertus-faithful CPT regime is solid: LR 1.1e-5,
   1.2 B-token warmup then constant, AdEMAMix β3=0.99, Goldfish k=h=50, and
   the proven batch/parallelism + sidecar-eval pattern. Future Task-2 arms
   should vary geometry/extension/layer/cutoff around this fixed regime unless
   explicitly labeled as hyperparameter ablations.
2. **Path-A geometry probe CONFIRMED.** All three load-bearing paired CIs
   exclude zero. Path-A is now the **locked** Task-2 default — not just
   recommended. `TASK2_HANDOFF.md §(3.1)` status flipped from "RECOMMENDED
   working position" → "CONFIRMED, LOCKED" on 2026-05-31.
3. **Matched-config Apertus-Base eval is empirically downgraded.** The
   probe confirmed the eval is perturbed, not a clean baseline. Drop it
   from Task-2 primary baselines; keep at most as a diagnostic bookend.
4. **Three new operational errors** surfaced during the Path-A probe
   launch (ARM env var, --ntasks-per-node=1, warmup-decay assertion
   re-hit). Each was a quick patch; together they argue for a Task-2
   submitter template hardened against these latent traps.
5. **TD layer-11 evidence audit done.** Layer 11 was a heuristic
   (one-third-depth from the TD package README), not the paper's
   recommendation. Paper default is `target_layer = -1` (last layer).
   Plan §3.2 called for a two-candidate pilot; only layer 11 ran. This
   is in `cpt-plan.md` §3.2 already but the *audit summary* is new and
   has now been written into §3.2 inline.
6. **Retention-baseline correction in `reports/5B_REPORT.md`.** The
   §7 retention table previously compared iter 1192 to iter 119 (i.e.
   to a mid-warmup, post-rope-readaptation state — not the true
   pre-CPT init). Corrected on 2026-05-30. Magnitudes shifted, especially
   on German MMLU and English commonsense; the corrected reading is in
   the report now. This is a class of error worth flagging for any Task-2
   retention-tracking design.
7. **eellak/greek-apertus upload scope discussed.** Four tiers + license
   considerations. Not yet acted on; deferred to a separate work item.

The single artifact for the verdict is
`reports/path_a_probe_results_20260531.md`. The full chain is in
`RUN_LOG_20260528.md` entries from `## 2026-05-30` onward.

---

## 1. Path-A geometry probe — VERDICT: CONFIRMED

Per `PATH_A_GEOMETRY_PROBE_PLAN.md` §7 decision rule.

**Three load-bearing paired bootstrap CIs (V4 v3 methodology — 1000
resamples, 95 % percentile, `rng_seed=20260529`, per-task item-level
paired):**

| Comparison (3-task headline) | Δ | 95 % CI | Sig |
|---|---:|---|---|
| Path A 0.5 B vs Path B iter 119 (matched tokens) | **+5.51 pp** | [+3.79, +7.25] | outside 0 |
| Path A 0.5 B vs Apertus-Base Path A (released) | **+1.25 pp** | [+0.25, +2.27] | outside 0 |
| Path A 0.5 B vs matched-config Path-B (init) | **+6.70 pp** | [+5.14, +8.23] | outside 0 |

**Path A 0.5 B marginal headline:** 0.4942 [0.4747, 0.5133]. This sits
*between* Task 1 Path B's 3.5 B and 5 B endpoints (0.4790 / 0.4973), at
**~one-tenth the compute**.

**Greek BPB:** Path A 0.5 B = 0.4365, vs Path B iter 119 = 0.6049 and
Task 1 Path B 5 B endpoint = 0.4132. Path A almost reaches the 5 B Path-B
endpoint on Greek BPB at 0.5 B tokens.

**Per-task breakdown vs Apertus-Base Path A (the released base):**

| Task | n | Apertus-Base Path A | Path A 0.5 B | Δ | Paired CI |
|---|---:|---:|---:|---:|---:|
| **greekmmlu** | 16 632 | 0.5280 [0.5198, 0.5357] | **0.5427** [0.5354, 0.5505] | **+1.48 pp** | outside 0 |
| **ilsp_mcqa_asep** | 1 200 | 0.5075 | **0.5325** [0.5025, 0.5592] | **+2.50 pp** | outside 0 |
| ilsp_medical_mcqa | 432 | 0.4097 | 0.4074 [0.3634, 0.4537] | −0.23 pp | inside 0 |
| plutus_qa (diag) | 225 | 0.5156 | 0.5289 [0.4667, 0.5911] | +1.33 pp | n/a |

The headline number worth highlighting: **`greekmmlu` Path A 0.5 B =
0.5427 [0.5354, 0.5505]**, statistically above Apertus-Base Path A's
0.5280 [0.5198, 0.5357]. **The largest single Greek MCQ benchmark
(n=16,632, by an order of magnitude over the others) is above the
released base in 0.5 B tokens, mid-warmup.** The CI bands touch at
0.5354 / 0.5357 but the paired bootstrap (which accounts for item-level
correlation across the two evaluations) cleanly rejects zero. ASEP is
also clearly above base. Medical and Plutus are within noise. Aggregate
is +1.25 pp ([+0.25, +2.27]) because Medical's inside-noise delta
slightly drags down the macro-mean — but the two largest-n tasks both
clear the base by significant margins.

**Retention:** every Task-1 rope-readaptation regression at iter 119 vs
matched-config init (xnli_en −2.08, xnli_fr −1.81, arc_challenge −1.37,
arc_easy −2.44, all in Path-B at iter 119) — **Path-A pays none of these
costs**. Every retention task is *positive* vs matched-config init at 0.5 B
Path-A.

### Implications for the Task-2 plan

- **Path-A is now locked.** Use it as the only training-time geometry
  unless you have a specific reason to test Path-B (e.g., a bakeoff-
  comparable re-run, which is a separate question).
- **The matched-config workaround is retired.** For Task-2 baselines,
  compare extension arms to Apertus-Base Path A directly (no bookend
  needed; the matched-config diagnostic stays on disk as a sensitivity
  artifact).
- **Compute budget implication.** Path-A reaches the 5 B Path-B headline
  state in ~0.5 B tokens. For Task-2 (vocabulary extension), this means
  a 5 B Path-A run leaves ~4.5 B of token budget *past* the "Task-1-5B
  equivalent" mark for extension-specific learning. If extension's value
  proposition is mostly fertility (inference efficiency) rather than
  capability, Task-2 doesn't need to be expensive — 1–2 B Path-A might
  suffice for the capability question.

---

## 2. Matched-config Apertus-Base eval — empirically downgraded

**Status update.** The matched-config Apertus-Base eval (rope_theta=500K
override on rope_theta=12M trained weights) was hypothesized to be a
*perturbation* not a *re-anchor* (TASK2_HANDOFF §2.3). The Path-A probe
empirically confirmed this: training under Path A *not* under matched-
config-Path-B produces +6.70 pp better headline at the same data exposure
than the matched-config eval suggests Apertus-Base "should" reach under
Path B. The matched-config eval is therefore confirmed as a *perturbation*,
not a *re-anchor*, end of story.

**Task-2 implication:**

- Drop matched-config from the primary cross-arm comparison set.
- Keep matched-config-Path-B-perturbed in `v4_bootstrap_cis_native_mcq.json`
  as a diagnostic for *rope sensitivity* (useful artifact for any future
  geometry-sensitive analysis), but with the explicit
  `perturbation_note: "DIAGNOSTIC of rope-perturbation, NOT a clean baseline."`
- For "iter-N vs Apertus-Base" claims in Task 2, use **Apertus-Base
  Path A** under its native geometry (which Task 2 also trains on). No
  caveat needed in either direction.

---

## 3. Three new errors surfaced during the Path-A probe launch

Each was a quick patch but together they argue for a Task-2 submitter
template that's hardened against these latent traps.

**3.1. `ARM=vanilla` env var not set.** `bakeoff_train.sbatch:39` requires
`ARM ∈ {vanilla, retok, centroid}` to pick the tokenizer + data prefix
case-statement. My Path-A submitter didn't set it; first attempt failed
in 9 seconds. Fix: add to `--export` list. The Task-1 chain submitter sets
it via `--export=ALL,ARM=vanilla,…` on L204 of
`submit_training_5b_chain.sh`. Not propagated to new submitters by default.

**3.2. `--ntasks-per-node=1` instead of `=$GPUS_PER_NODE`.** Slurm
inferred `WORLD_SIZE=1`. Megatron's distributed init then asserted
`world_size % decoder_model_size==0` and crashed because `decoder_model_size=2`
(TP=2). Second attempt failed in 29 seconds. Fix: change to
`--ntasks-per-node=$GPUS_PER_NODE`. Task-1 chain submitter has this right
on L197.

**3.3. Warmup-decay assertion re-hit.** Megatron's `OptimizerParamScheduler`
asserts `lr_warmup_steps < lr_decay_steps`. With `TRAIN_TOKENS=500M` and
warmup window `LR_WARMUP_TOKENS=1.2B`, warmup steps (= 1.2B/seq/batch =
~293k samples) > decay steps (= 500M/seq/batch = ~122k samples). Same
issue Task-1 hit on the first segment (handoff §2.8); the fix in Task-1
was to extend segment 1 past warmup. Path-A probe re-hit the same
constraint because my submitter set `TRAIN_TOKENS=500M` directly. **Fix
for Task-2 was different but cleaner**: set `TRAIN_TOKENS=1.5B` (a notional
scheduler target satisfying the assertion) and use `EXIT_INTERVAL=119` for
early exit at 0.5B. This pattern is now in `train_config_04a_path_a.env`
and worth lifting into any Task-2 submitter template.

### New Task-2 recommendation: hardened submitter template

Add to `TASK2_HANDOFF §3` as a new item:

> Build the Task-2 submission template by *deriving directly* from
> `scripts/submit_training_5b_chain.sh` (Task-1 chain submitter), not by
> rewriting the `--export` list from `bakeoff_train.sbatch`'s requirements.
> Pre-flight check: assert `ARM` is set, `--ntasks-per-node == GPUS_PER_NODE`,
> and `TRAIN_TOKENS / (SEQ_LENGTH * GLOBAL_BATCH_SIZE) > LR_WARMUP_TOKENS / (SEQ_LENGTH * GLOBAL_BATCH_SIZE)`
> before submission. Three of the Path-A failures would have been caught
> at submit time, not training-init time.

---

## 4. TD layer-11 evidence audit

The audit summary (just done in this session):

**Layer 11 was a heuristic, not validated.** Specifically:

- **Origin**: `ceil(num_hidden_layers / 3) = 11` is the "one-third depth"
  suggestion from the TD package's README, not the paper.
- **Paper recommendation**: `target_layer = -1` (last layer). Paper §5.3
  explicitly says last-layer is "a principled choice, as it guarantees
  that no subtoken interactions that are only modeled in later layers
  are excluded from the objective."
- **Original Task-1 plan** (`TOKEN_DISTILLATION_PLAN.md §16, §6.1`): two-
  candidate pilot — Candidate A = layer -1 (paper default), Candidate B =
  layer 11 (package README). Plus optional Candidate C from a logit-lens /
  tuned-lens probe identifying Apertus-specific `L*`.
- **What actually ran**: only layer 11 (Candidate B). Candidate A and the
  logit-lens probe were never executed.
- **Outcome**: TD layer-11 was the bakeoff's only TD arm. Per Task-1's
  5 B headline, TD layer-11 trailed Vanilla on the native MCQ aggregate
  (TD-5B headline 0.4109 vs Vanilla-5B 0.4305). Under Path A, this
  comparison would need to be re-run to be meaningful (Path-B contaminated
  by rope re-adaptation).

### Implications for Task 2 v1.x

This is documented in `cpt-plan.md §3.2` ("Layer 11 was a hypothesis in
the bakeoff, not a settled fact") with a recommended sweep over layers
`{4, 8, 11, 16, 20}`. For Task 2 specifically:

- **Run the layer sweep first.** Tiny cost (5 layers × ~0.5 B tokens each,
  on Path A, on the chosen Greek+replay mix) = ~125 GPU-h total.
  Methodology: same as the Path-A probe — single-checkpoint training to
  iter 119, single sidecar fan-out, paired CI vs Vanilla-Path-A baseline.
- **Add the logit-lens probe as a cheap diagnostic.** Per
  `TOKEN_DISTILLATION_PLAN §6.1` — runs in minutes, not hours. Gives an
  Apertus-specific data point for which layer's residual stream "knows
  the whole word." If `L*` clusters around a specific layer, include that
  as a sixth sweep candidate.
- **Document the chosen layer with full provenance** in the Task-2 spec.

---

## 5. Retention-baseline correction in `5B_REPORT.md` (post-handoff)

This was caught and fixed *between* the handoff being written and the
Path-A probe launching, so it's worth surfacing here.

**Bug.** The original §7 retention table in `5B_REPORT.md` compared
iter 1192 to iter 119 (and called the iter 119 value "the start"). But
iter 119 is *0.5 B tokens INTO training* — already past most of the
rope re-adaptation, not the true pre-CPT state. Several "retention gain"
deltas were inflated because they included recovery from the rope dip.

**Fix.** Pulled matched-config Apertus-Base retention numbers (which IS
our actual iter-0 / pre-CPT state under Path B). Updated `5B_REPORT.md
§7` with a corrected table that uses matched-config as the baseline
column and computes Δ vs that.

**Magnitudes changed:**

| Task | Old Δ (vs iter 119) | Corrected Δ (vs true init) |
|---|---:|---:|
| global_mmlu_en | +3.50 pp | +4.75 pp |
| global_mmlu_fr | +2.75 pp | +4.50 pp |
| global_mmlu_de | +6.27 pp | **+2.25 pp** |
| arc_easy | +1.59 pp | **−0.85 pp** |
| arc_challenge | +1.20 pp | **−0.17 pp** |
| xnli_en | +5.82 pp | +3.74 pp |
| xnli_fr | +3.74 pp | +1.93 pp |
| xnli_de | +1.01 pp | **−0.20 pp** |
| xnli_ru | −1.53 pp | **−1.57 pp** |

**Notable shifts:**
- German MMLU's apparent "+6.27 pp gain" was mostly rope re-adaptation
  recovery; corrected gain is +2.25 pp.
- English commonsense (`arc_easy`, `arc_challenge`) flipped from positive
  to slightly negative when measured against true init — they were the
  rope-readaptation casualties that didn't fully recover.
- Russian xnli's regression is *bigger* against true init than against
  iter 119.

### Task-2 implication

Always use **the true pre-CPT init** as the retention baseline. The
operational form: at Task-2 launch, evaluate the model loaded at training
geometry but with zero training steps applied (= iter 0). This is the
matched-config evaluation under Path-A in Task-2 (which, since Task-2
trains on Path A, IS the same as evaluating Apertus-Base under its
released geometry — no perturbation, clean comparison). This avoids the
class of error that bit the original §7 table.

---

## 6. Concrete updates landed in `TASK2_HANDOFF.md` and `cpt-plan.md`

In addition to writing this reply, I've added/updated:

- `TASK2_HANDOFF.md §(3.1)`: status flipped 2026-05-31 from "RECOMMENDED
  working position" → "CONFIRMED, LOCKED", with the three CIs inline + an
  operational note about the init-checkpoint conversion provenance.
- `TASK2_HANDOFF.md §2`: added entries 2.14 (ARM env-var), 2.15
  (ntasks/WORLD_SIZE), 2.16 (warmup-decay assertion re-hit on Path-A
  submitter, not just Task-1 first segment).
- `TASK2_HANDOFF.md §3`: added recommendation 3.11 about hardened Task-2
  submitter template.
- `cpt-plan.md §3.2`: added a "Layer 11 evidence audit" subsection with
  the heuristic provenance + the un-run candidates + the recommended
  layer sweep + logit-lens probe for v1.x.

---

## 7. Open items still on Task-2's plate (unchanged from handoff)

These are NOT changed by the Path-A probe; they remain the planning
agent's load-bearing decisions:

- **Vocabulary extension method** (TD layer-X / ReTok / Centroid — Task-1
  ruled out Centroid; TD is the favored candidate but with the layer
  caveat above).
- **BPE cutoff** for the extended tokenizer (the C3 grid
  `{10240, 15360, 20480, 25600}`).
- **Decontamination MinHash** against the 4 native MCQ benchmark prompts
  (handoff item §3.3 + §2 V1 — a launch blocker for Task-2 production).
- **BPB heldout rebuild** to drop 4096-overflow docs (handoff item §3.4).
- **WSD cooldown** for the final 10–20 % of any 7–10 B Task-2 run
  (handoff item §3.9).
- **Pre-commit decision thresholds** (`cpt-plan.md §10 Q8`) — these were
  never locked for Task 1 because it was diagnostic; Task 2 is a
  production-style experiment with adjudication consequences, so lock
  them in the v1.x spec.

---

## 8. Suggested next step for the planning agent

Given the probe verdict + the new errors + the TD-layer-audit, the
natural Task-2 v1.x shape is:

First, carry forward the corrected hyperparameter regime unchanged as the
default experimental center. Vary TD layer, BPE cutoff, extension method,
stabilization duration, and final cooldown only as named Task-2 variables.

1. **TD layer sweep** under Path A (Vanilla + 5 TD layers × ~0.5 B
   tokens each = ~6 × 17 GPU-h = ~102 GPU-h) → picks the layer.
2. **Decontamination MinHash + BPB heldout rebuild** in parallel (zero
   GPU cost).
3. **Pick BPE cutoff** from `{10240, 15360, 20480, 25600}` on tokenizer-
   quality metrics (zero CPT-side cost).
4. **Full extension CPT** at the chosen (layer, cutoff) × Path A
   geometry, 5–10 B tokens depending on §10 Q8 pre-commit thresholds.

This is a much smaller Task-2 than the original cpt-plan §3 envisioned,
because Path A makes the regime question moot — the open dimension is
the extension method + cutoff. Total Task-2 GPU-h is plausibly in the
~200–400 GPU-h range, not the 1000+ that a full extension-method bakeoff
under the original plan would have consumed.

---

## 9. Artifacts

For the planning agent to read:

- `TASK2_HANDOFF.md` — the original 13-error retrospective + 10
  recommendations + 6 open items. **This is the primary handoff doc; this
  reply is a supplement.**
- `reports/path_a_probe_results_20260531.md` — formal Path-A verdict.
- `reports/v4_bootstrap_cis_native_mcq.json` (v3) — bootstrap CI artifact
  for Task-1 (5 B Path-B run).
- `reports/v4_workspace_path_a/path_a_probe_bootstrap_cis.json` —
  bootstrap CI artifact for Path-A probe.
- `reports/5B_REPORT.md` — corrected post-2026-05-30 retention table
  (uses matched-config init as the baseline now).
- `reports/decisions_matrix_20260529.md` — 24-row Task-1 decisions
  matrix.
- `reports/plot_mmlu_trajectory.png` — Greek + English MMLU with Path-A
  overlay (gold star).
- `reports/plot_retention_per_language.png` — per-language EN/FR/DE/RU
  retention with matched-config as baseline.
- `cpt-plan.md` — the Task-1+Task-2 plan with §3.2 TD layer-11 audit
  added.
- `adversarial_reviews/Vanilla-Path-A-0.5B/adversarial_critique.md` —
  independent adversarial read of the Path-A probe.
- `RUN_LOG_20260528.md` — full append-only audit trail of Tasks
  1 + 1a (Path-A probe).
- `PATH_A_GEOMETRY_PROBE_PLAN.md` — the plan for the Path-A probe.

---

**End of reply.** Hand off to v1.x of Task 2 with these as input.
