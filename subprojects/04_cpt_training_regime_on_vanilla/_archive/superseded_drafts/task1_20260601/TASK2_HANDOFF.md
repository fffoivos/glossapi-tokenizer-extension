# Task 1 → Task 2 handoff

**Audience.** The planning agent producing Task 2 v1.x (production extension
experiment) per `cpt-plan.md` §3.

**Purpose.** This document is the deliberate handoff. It states what Task 1
actually established, where the methodology was wrong and how we recovered,
and what Task 2 must inherit from Task 1's lessons (operationally and
methodologically). It is not the 5 B report (`reports/5B_REPORT.md`) — that
document reports results to the broader project; this document is a peer
note to the next planning pass.

**Date.** 2026-05-30. After iter 1192 (5 B) endpoint + V4 v3 + corrections.

---

## (1) What Task 1 established

**Regime hypothesis is SUPPORTED.** The Apertus-faithful regime
(LR 1.1e-5 with 1.2 B-token warmup then constant; AdEMAMix β3=0.99; Goldfish
k=h=50) produces a Vanilla CPT trajectory that is statistically distinct
from the bakeoff Vanilla on the same Path-B geometry, and that meaningfully
exceeds the released Apertus-Base on the headline 3-task native Greek MCQ
aggregate by the 5 B endpoint.

**Most important carry-forward result:** the corrected hyperparameter
regime worked. Future experiments should inherit this as the default center:
LR 1.1e-5, 1.2 B-token warmup then constant schedule, AdEMAMix β3=0.99,
Goldfish k=h=50, the proven batch/parallelism shape, and the existing
sidecar-eval pattern. Path A fixes the geometry confound; TD-layer choice is
the next experimental variable. Neither should obscure that the corrected
regime is the baseline assumption. Any future run that changes these
hyperparameters should be labeled as an explicit hyperparameter ablation, not
as a normal Task-2 arm.

| Comparison | Δ (3-task headline) | 95 % paired CI | Significance |
|---|---:|---:|---|
| iter 1192 vs **bakeoff Vanilla-5B** (Path-B both sides) | **+6.69 pp** | [+5.13, +8.30] | outside zero |
| iter 1192 vs Apertus-Base Path A (released geometry) | +1.56 pp | [+0.16, +2.84] | outside zero (barely) |
| iter 1192 vs matched-config Apertus-Base (init, Path B perturbed) | +7.01 pp | [+5.37, +8.57] | outside zero |
| iter 1192 vs iter 477 (full post-warmup gain) | +1.82 pp | [+0.60, +3.06] | outside zero |
| iter 1192 vs iter 834 (post-plateau slope) | +1.84 pp | [+0.80, +2.95] | outside zero |
| iter 834 vs iter 477 (the plateau) | −0.02 pp | [−1.23, +1.14] | inside zero |

The cleanest comparison is iter 1192 vs bakeoff Vanilla at the same token
count and same Path-B geometry: **+6.69 pp on the 3-task headline, CI
[+5.13, +8.30], outside zero, significant across all 4 individual benchmarks
in addition to the aggregate.** This is the load-bearing regime claim.

Trajectory shape: warmup → **+3.05 pp post-warmup burst** at iter 477 →
flat 1.5 B-token segment (iter 477 → iter 834) → **+1.84 pp endpoint lift**
at iter 1192. The flat segment is NOT plateau; it is statistically distinct
zero on aggregate, but masks **task-level cancellation** (greekmmlu +2.09 pp
vs iter 477, ilsp_mcqa_asep −1.92 pp vs iter 477, ilsp_medical_mcqa neutral
— both significant on paired CI).

Per-task trajectory across the run:

| Task | init (matched Path B) | 0.5 B | 1 B | 2 B | 3.5 B | 5 B | Δ 5 B vs init |
|---|---:|---:|---:|---:|---:|---:|---:|
| greekmmlu | 0.4879 | 0.4985 | 0.5026 | 0.5075 | 0.5284 | **0.5584** | **+7.05 pp** |
| ilsp_medical_mcqa | 0.3264 | 0.3380 | 0.3634 | 0.3958 | ~0.3958 | 0.4028 | +7.64 pp |
| ilsp_mcqa_asep | 0.4675 | 0.4808 | 0.4800 | 0.5342 | 0.5150 | 0.5308 | +6.33 pp |
| plutus_qa (diag) | 0.3467 | 0.4000 | 0.4133 | 0.4889 | 0.4889 | **0.4356** | +8.89 pp |
| **headline 3-task** | 0.4272 | 0.4391 | 0.4487 | 0.4792 | 0.4790 | **0.4973** | **+7.01 pp** |

GreekMMLU is the broad-knowledge driver of the headline; medical and ASEP
are domain-narrower and plateau or fluctuate. Plutus is the smallest n
(225) and the most volatile.

Multilingual retention vs matched-config (true) init, the EN / FR / DE / RU
slice the goal specifies:

| Task | init | iter 1192 | Δ vs init |
|---|---:|---:|---:|
| global_mmlu_en | 0.6025 | 0.6500 | **+4.75 pp** |
| global_mmlu_fr | 0.5425 | 0.5875 | **+4.50 pp** |
| global_mmlu_de | 0.5725 | 0.5950 | +2.25 pp |
| mmlu (Hendrycks EN) | 0.5624 | 0.5798 | +1.74 pp |
| xnli_en | 0.5112 | 0.5486 | +3.74 pp |
| xnli_fr | 0.4859 | 0.5052 | +1.93 pp |
| xnli_de | 0.4968 | 0.4948 | −0.20 pp |
| **xnli_ru** | **0.4884** | **0.4727** | **−1.57 pp** |
| arc_challenge (EN) | 0.5384 | 0.5367 | −0.17 pp |
| arc_easy (EN) | 0.8279 | 0.8194 | **−0.85 pp** |
| piqa (EN) | 0.7922 | 0.7889 | −0.33 pp |
| hellaswag (EN) | 0.5862 | 0.5906 | +0.44 pp |

The regime did NOT cannibalize non-Greek capability: multilingual MMLU
gains across EN / FR / DE; xnli_en + xnli_fr gain. Two genuine regressions:
**xnli_ru −1.57 pp** and **English commonsense (arc/piqa) all slightly
below init** (the rope re-adaptation casualties that did not fully
recover).

Greek BPB (tokenizer-fair, heldout 500 docs): monotone improvement across
the run — 0.605 (iter 119) → 0.413 (iter 1192).

Compute: **217.2 GPU-h total** for the entire 5 B run (training 175.25 +
sidecar evals 36.56 + matched-config base eval 2.74 + MCQ resubmits 2.01
+ smokes 0.64). Wall-clock from training start to iter 1192 done ≈ 45 h.

---

## (2) Errors we made + how we recovered

These are the methodology and operational mistakes that occurred during
Task 1. Each is stated honestly so that Task 2 can avoid them by design.

### (2.1) Slurm `--export` comma-bug silently truncated `BENCHMARKS`

`scripts/submit_checkpoint_sidecars.sh` L156 passed
`--export=ALL,…,BENCHMARKS="$NATIVE_BENCHMARKS",…` where
`$NATIVE_BENCHMARKS = "greekmmlu,ilsp_medical_mcqa,ilsp_mcqa_asep,plutus_qa"`.
Slurm splits `--export` on every top-level comma; the BENCHMARKS value
collapsed to just `greekmmlu`, and the other three benchmark names became
empty extra env entries. Result: iter 238 and iter 477 native MCQ jobs
ran **GreekMMLU only**. The aggregate JSON nonetheless reported a
3-task `headline_policy` block, producing a misleading single-task
"headline" that would have compared against iter-119's true 3-task mean.

**How discovered.** The Vanilla-1B adversarial reviewer caught the bug
by reading the submitter log + the per-iter `run_metadata.json` and
correlating it with the aggregate `n_tasks` field. iter 119 escaped only
because its MCQ was manually resubmitted with `BENCHMARKS=all` after an
unrelated earlier failure.

**Recovery.** Fix at script L141–160: pull `BENCHMARKS` out of the
`--export` list and prefix it as a shell env-var on the sbatch invocation
(`BENCHMARKS="$NATIVE_BENCHMARKS" sbatch --export=ALL,…`). The shell
env-var carries through unsplit when `--export=ALL` is set. Hash
transition `7eb4667e…` → `e865c65a…`; mirrored to Clariden. iter 238 +
iter 477 MCQ resubmitted as jobs 2422769 + 2422770; iter 834 + iter 1192
fired under the fixed script and verified clean.

**Task-2 implication.** Audit every `--export=…,KEY="$VAR",…` in the
eval pipeline. The `script_audit_20260529.md` C2 + C3 findings flag two
more shell-quoting risks of the same class (`run_eval.sbatch:127`,
`run_greek_nlp_benchmark_hf.sbatch:60`) — currently safe but worth
hardening before Task 2's longer extension runs increase the risk
surface.

### (2.2) Geometry contradiction in cpt-plan.md §2.1 and hyperparameters.json

The plan's "Model and architecture (inherited from Apertus, all
unchanged)" table listed `RoPE θ = 500,000` AND `Max position = 65,536`,
both cited to Apertus paper §2.3. This is internally inconsistent: §2.3
is the initial pretraining stage (rope=500K, max_pos=4096); the released
Apertus-Base ships at §2.5 long-context extension geometry (rope=12M,
max_pos=65536, llama3 scaling). Our training command in fact uses
rope=500K + max_pos=4096 (Path B, inherited from bakeoff). Same
contradiction lived in `hyperparameters.json[base_model.architecture]`.

**How discovered.** The Vanilla-0.5B adversarial reviewer at iter 119
flagged the live training command's geometry against the base checkpoint's
config.json. Persisted as Critical-1 across every subsequent review until
fixed at iter 477 + iter 834.

**Recovery.** Rewrote cpt-plan.md §2.1 with two subsections:
*"Model and architecture (base, as released)"* showing the actual base
config (Path A); a new *"Training-time positional geometry override
(Path B)"* subsection documenting our override + rationale + cost +
empirical evidence. Added §3.4 Q3.4.10 with the explicit Path-A
recommendation for Task 2. Updated `hyperparameters.json` to make
`base_model.architecture` show the true Path-A values and added a new
top-level `training_geometry` block with Path-B values, source,
rationale, cost-of-path-b, matched-config diagnostic record, and
`task2_geometry_recommendation = Path A` with full Path-A values.

**Task-2 implication.** The plan and the JSON should be **internally
consistent and externally consistent with the live training command** at
all times. For Task 2, lock the Path-A values at the top of the v1.x
spec; do NOT inherit Path-B from this run.

### (2.3) Matched-config Apertus-Base eval perturbs the base, doesn't re-anchor

We built a matched-config Apertus-Base eval by symlinking Apertus-Base
weights into a sibling HF directory and overriding only
`rope_theta=500K`, `max_position=4096`, `rope_scaling=null`. The intent
was a clean Path-B baseline for "Apertus-Base under our training geometry."
The empirical result: matched-config Greek BPB = 1.2216 (vs Vanilla-2B
0.43) and matched-config native MCQ headline = 0.4272 (vs Apertus-Base
Path A 0.4817, a 5.5 pp drop). The override forces inference under
positional encoding the model was never trained with; KV/Q phases shift
relative to training. The model is *perturbed*, not *re-anchored*.

**How discovered.** The Vanilla-2B adversarial reviewer at iter 477.
Concretely: Greek BPB at 1.22 against an Apertus-Base that was
characterized in earlier project diagnostics as "Greek well-trained"
(Phase B v4 NLL ~0.95 on modern Greek) is implausible. The reviewer
isolated the rope-override as the cause and recommended an English
sanity check.

**Recovery.** Documented the matched-config eval as **diagnostic-only**,
not a clean baseline. The bracket for cross-arm "vs Apertus-Base" claims
in the 5 B report now uses BOTH bookends: Path A (released geometry,
geometry-mixed) AND matched-config Path B (perturbed). Both bookends
agree on the direction of the regime claim — iter 1192 is above both,
significantly. The matched-config artifact lives at
`hyperparameters.json[training_geometry.matched_config_diagnostic]` with
`status_note: "DIAGNOSTIC of rope-perturbation, NOT a clean baseline."`

**Task-2 implication.** Do not attempt to bridge a geometry mismatch by
config-only override of trained weights. The only clean comparisons are
(a) train and eval on the same geometry, or (b) accept the geometry as a
documented caveat. Task 2's Path-A recommendation removes the need for
any matched-config workaround.

### (2.4) Linear-extrapolation projection at iter 477 → 5 B was wrong

After iter 477 we wrote in the RUN_LOG: "If the post-warmup slope holds
at half rate, iter 1192 (5 B) ≈ 0.515." That projection was based on
naive linear extrapolation of the iter 238 → 477 slope. iter 834 came in
flat (0.4790, paired CI vs iter 477 brackets zero), invalidating the
slope assumption. iter 1192 in fact landed at 0.4973 — below the
projection but with a real +1.84 pp resumed lift.

**How discovered.** The Vanilla-3.5B adversarial reviewer caught the
inconsistency between the prior projection and the iter-477-vs-iter-834
paired CI. Reviewer's bracket for iter 1192 was [0.467, 0.491]; the
actual 0.4973 is JUST above the optimistic end of that bracket.

**Recovery.** Retracted the 0.515 projection in the RUN_LOG. The
endpoint trajectory turned out to be hybrid: warmup-finished jump +
brief plateau + real but smaller lift at 1192. Not the bakeoff
"peak-then-drift" shape; not the optimistic linear-extrapolation shape
either.

**Task-2 implication.** Do not extrapolate from two post-warmup points
on a single CPT trajectory. Either commit explicit pre-result thresholds
(per plan §6) and read the result against them, or report the trajectory
shape descriptively without prediction.

### (2.5) Plateau hypothesis (Vanilla-3.5B reviewer) turned out wrong

The Vanilla-3.5B reviewer favored "one-shot warmup-finished jump now
stabilized" as the trajectory hypothesis, with iter 1192 bracket
[0.467, 0.491]. The endpoint 0.4973 is above this bracket — the slope
was not zero after all. The paired CI iter 477 → iter 834 brackets zero
on aggregate (correct), but the per-task structure (greekmmlu still
gaining, ASEP regressing) hinted that the model was still learning, just
not net-additively across all tasks. iter 1192 confirmed this: the slope
re-resumed.

**How discovered.** Empirically by iter 1192's headline.

**Recovery.** Vanilla-5B reviewer's revised trajectory hypothesis:
warmup → +3.05 pp burst → flat 1.5 B segment (task-level cancellation) →
+1.84 pp endpoint lift. Information value of 5 B → 10 B is non-zero.

**Task-2 implication.** When the aggregate plateaus but per-task signals
are still moving in opposite directions, **the slope is not zero — it
just looks zero in the aggregate**. Do not interpret aggregate plateaus
as terminal without checking per-task structure first.

### (2.6) Retention plot used iter 119 as baseline instead of matched-config

The first version of `plot_retention_per_language.png` used iter 119 as
the "starting point" for per-language MMLU and XNLI deltas. iter 119 is
0.5 B tokens INTO training, not the pre-CPT state — it has already
absorbed a significant chunk of rope re-adaptation cost. So the "+3.87 /
+5.40 / +6.27 pp on EN / FR / DE MMLU" deltas were partly retention and
partly geometry recovery.

**How discovered.** Fivos noticed by pushing on the methodology: "Are
you sure that you are using as the start line the evaluated base
apertus before training started?"

**Recovery.** Pulled the matched-config Apertus-Base retention numbers
and updated `plot_retention_per_language.py` to use them as the
horizontal reference + delta target. Trajectory now starts at tokens=0
(matched-config base). Corrected deltas: EN MMLU +4.75 pp (was claimed
+3.87 from iter 119); FR MMLU +4.50 (was +5.40); DE MMLU +2.25 (was
+6.27). German was the most overstated. xnli_ru regression is bigger
than the iter-119-baseline version suggested (−1.57 vs −0.92). 5 B
report retention table § 7 updated with the same correction +
documentation of the rope-readapted iter 119 column.

**Task-2 implication.** **The pre-CPT state of any CPT run is the model
weights evaluated under the run's training geometry** — not iter 0 of
training, not "Apertus-Base under the model's released geometry."
For Task 2, generate the pre-CPT evaluation at the same training-time
geometry as the run, in advance, and use it as the baseline for every
retention or capability delta. If the training geometry matches the
released checkpoint geometry (Path A, as recommended), the eval IS the
released-checkpoint eval — no matched-config workaround needed.

### (2.7) Eval pipeline ran 201 retention tasks but only 12 are canonical

The lm-eval `retention_only` task group ran ~201 task entries per
checkpoint (global_mmlu for 12+ languages, xnli for all available, etc.).
The plan's canonical subset is 12 tasks (4 languages × ~3 tasks each).
The inclusion / exclusion decisions lived in prose comments and post-hoc
JSON `headline_policy` fields, **not at the eval submission layer**.
Every report read required a human or agent to remember which tasks
were canonical and which weren't. This produced exactly the case-by-case
interpretation drift Fivos called out ("how can you say it's excluded
and then call its regression real signal?").

**How discovered.** Fivos's pushback after the Vanilla-5B reviewer cited
xnli_el ("MT-derived, excluded") and xnli_ru ("retention concern") in
the same sentence without distinguishing layer.

**Recovery.** Wrote `goal/canonical_eval_tasks.json` as the explicit
lockdown (12 canonical retention tasks + 3 Greek MCQ headline + 1
diagnostic + 4 explicitly excluded MT-derived Greek + 3 heldout BPB).
Patched the renderer scripts (`scripts/collect_5b_report_state.py` and
`scripts/render_5b_report_status.py`) to filter retention task entries
against the lockdown at collection time; non-canonical entries are
dropped, not just hidden. Per-checkpoint retention metric count went
from 201 raw upstream entries → 12 canonical (189 dropped). Edited
`reports/5B_REPORT.md` to drop mentions of the 4 excluded MT-derived
Greek tasks + non-canonical retention tasks (global_mmlu_{ar,bn,…},
winogrande, etc.); reframed xnli_ru as legitimate retention finding.

**Task-2 implication.** **Lock the canonical eval task list at the eval
submission layer, not at the renderer or reviewer layer.** Replace the
`retention_only` task-group invocation with an explicit task-list
argument populated from `canonical_eval_tasks.json` (or its Task-2
extension). Don't measure things we've decided not to report.

### (2.8) First training segment failed before any work

Job 2417278 ("04van5b_i119" target) failed in Megatron with the
`OptimizerParamScheduler` assertion
`lr_warmup_steps < lr_decay_steps`. We had set the warmup window to the
1.2 B-token Apertus long-context-continuation value (287 steps) and the
first segment's per-segment decay-step target to 119 — warmup_steps >
decay_steps over the segment, the assertion was triggered, the job
crashed before training began.

**Recovery.** Patched `scripts/submit_training_5b_chain.sh` so segment 1
targets iter 357 (later iter 300 for walltime safety), preserving the
required 119 / 238 save points inside the first segment via
`SAVE_INTERVAL=119`. Lesson: Megatron's per-segment scheduler must see
`lr_warmup_steps` AND `lr_decay_steps` over the *full run*, not the
*current segment*. Segment 1 must extend past the warmup end.

**Task-2 implication.** Verify segment-1 length > warmup-end-iter at
launch time. Document this constraint in the Task-2 sbatch chain header
to prevent re-discovering.

### (2.9) First iter-119 sidecar conversion job failed

The first conversion attempt (job 2419080) failed because the Clariden
mirror lacked `_train_config_common.env`. The convert sbatch sources
this file to identify the Megatron→HF mapping; without it, conversion
crashes. Local repo had the file; it just wasn't synced to Clariden.

**Recovery.** Synced the non-secret config; SHA matched (`35ab0f87…`);
archived the failed manifest as
`sidecar_jobs_attempt_failed_convert_2419080_20260528T192233Z.tsv`;
resubmitted chain `2419108`–`2419114` clean. Also hardened the verifier
(`scripts/verify_checkpoint_sidecars.py`) so subsequent handoffs require
`expected_outputs_ready` ∧ `slurm_jobs_completed` ∧
`checksum_manifest_ready` ∧ nonempty output dirs (decisions matrix row O).

**Task-2 implication.** Verify the Clariden mirror has every file the
sidecar pipeline requires before the first checkpoint lands. Don't trust
"the chain worked in the smoke" — smokes use a different config path.

### (2.10) codex outage mid-run; Claude Code subagent took over reviews

Codex was unavailable for the duration of the iter-238 → iter-1192
review chain. The `run_checkpoint_adversarial_review.sh` script is hard-
wired to `codex exec` and exits 127 without it. The home-side watcher
that fires per-checkpoint reviews would have looped failing every 10 min.

**Recovery.** Stopped the codex-bound watcher; took over the
adversarial-review step with Claude Code subagents in the main session.
Each per-checkpoint review is now an `Agent` call with a fresh
subagent context. Output paths and prompt template preserved verbatim.
`review_metadata.env` records `BACKEND="claude-code-subagent"` so future
codex re-runs (or supersedes) are distinguishable from the original
chain. Vanilla-1B / 2B / 3.5B / 5B critiques are all Claude-subagent
output, marked as such.

**Task-2 implication.** The adversarial-review runner should accept a
backend argument (codex / claude) and not be hard-wired to a single
provider. The review prompt and review schema are provider-agnostic;
only the invocation differs.

### (2.11) Stale memory about Clariden `xfer` partition

My memory file said xfer was in Apertus maintenance reservation through
2026-06-11 and that CPU-only jobs should route to `normal` with
`--cpus-per-task=64 --mem=400G`. The live Clariden check during run
setup found the opposite: `normal`, `debug`, `low` all expose `gpu:4`;
`xfer` was the only visible CPU-only partition. The watcher and
checksum sidecars ran on xfer for the full duration without issue.

**Recovery.** Documented in RUN_LOG. The xfer-maintenance memory was
either stale or over-broad. The empirical-live-Clariden-check pattern
caught it before we wasted a CPU-only build on a GPU partition.

**Task-2 implication.** Verify partition routing empirically at launch
time; don't trust memory about reservations. Clariden partition policy
shifts on short notice.

### (2.12) Plutus-in-headline JSON bug

The pre-Task-1 native MCQ aggregator emitted `Vanilla-X_native_mcq_headline.json`
with Plutus included in the headline tasks, while the plan's headline is
the 3-task no-Plutus aggregate. Was structurally fixed before our run by
splitting the aggregate into `headline` (3 tasks) + `diagnostics`
(Plutus) blocks + an explicit `headline_policy` block. The fix was
**operationally re-broken** at iter 238 + iter 477 by the `--export`
comma bug above (only GreekMMLU ran; headline_policy declared 3 tasks
but n_tasks=1). Fixed again with the comma-bug recovery (§2.1).

**Task-2 implication.** Aggregator schema is correct; do not
re-implement the `headline_policy` field. The risk vector is the
*submission layer*, not the aggregator.

### (2.13) `truncation` field name false alarm

I claimed in a RUN_LOG entry that iter 834's Greek BPB JSON showed
`trunc=null` and that the schema had changed. The Vanilla-3.5B reviewer
caught the error: the field is named `truncation` (not `trunc`), value
unchanged at 0.292 across iter 119 / 238 / 477 / 834. Withdraw.

**Task-2 implication.** Read the schema; don't assume field names.

### (2.14) Path-A probe submitter missed `ARM=vanilla` env var (added post-handoff)

`bakeoff_train.sbatch:39` asserts `ARM ∈ {vanilla, retok, centroid}` at the
top of the script — the case-statement at L232 uses ARM to pick the
tokenizer + data prefix. My first Path-A submitter (`submit_04a_path_a_probe.sh`)
didn't include `ARM=vanilla` in the `--export` list. Job 2437889 failed
in 9 s with `ARM is required (vanilla|retok|centroid)`.

The Task-1 chain submitter sets it on `submit_training_5b_chain.sh:204`
(`--export=ALL,ARM=vanilla,...`). I rewrote the `--export` list from
scratch instead of deriving from the chain submitter and missed this +
3 other required env vars (`INIT_CKPT`, `SCRIPT_DIR_OVERRIDE`,
`BASE_DATA_PREFIX`/`EXT_DATA_PREFIX` from `dataset_paths.env`).

**Task-2 implication.** Don't re-derive the `--export` list. Inherit
the Task-1 chain submitter's shape and parameterize the differences
(geometry, target tokens, etc.). See §3.11 below.

### (2.15) Path-A probe submitter set `--ntasks-per-node=1` instead of `=$GPUS_PER_NODE`

After fixing 2.14, job 2437893 failed in 29 s with Megatron's distributed
init assertion `world_size % (encoder + decoder)`: `WORLD_SIZE=1` because
`SLURM_NTASKS=1`, but `decoder_model_size=2` from `TP=2`. Megatron then
crashed cleaning up an unrelated path with an `UnboundLocalError`.

Task-1 chain submitter has `--ntasks-per-node=$GPUS_PER_NODE` (= 4) on
`submit_training_5b_chain.sh:197`. Same provenance issue as 2.14.

**Task-2 implication.** Pre-flight assert `ntasks-per-node == GPUS_PER_NODE`
in any new submitter. See §3.11 below.

### (2.16) Warmup-decay assertion re-hit on Path-A probe (Task-1 §2.8 fix not propagated)

After fixing 2.14 + 2.15, job 2437896 failed at 1m10s on the same
`OptimizerParamScheduler` assertion that took down Task-1 segment 1
(see §2.8): `lr_warmup_steps < lr_decay_steps`. The probe set
`TRAIN_TOKENS=500M` directly; warmup window `LR_WARMUP_TOKENS=1.2B`
encodes to ~293k warmup samples > ~122k decay samples (at 500M target).

Task-1 fixed this by extending segment 1 to iter 357; that was specific
to the chain submitter. For the Path-A probe I used a *different* fix
(cleaner): set `TRAIN_TOKENS=1.5B` (notional, satisfies assertion) plus
`EXIT_INTERVAL=119` (early-exit at 0.5B). Effective training cost is iter
119 only; scheduler shape is warmup-then-constant-over-1.5B-notional.

**Task-2 implication.** Encode the constraint
`TRAIN_TOKENS > LR_WARMUP_TOKENS` as a pre-flight assertion in any new
submitter. Document the "notional target + early exit" trick for
probe-style runs that want to stop before the warmup window. See §3.11.

### (2.17) `5B_REPORT.md` §7 retention table used wrong baseline (caught + fixed)

Original table in the synth-agent-written `5B_REPORT.md` compared iter
1192 to *iter 119* and labeled it "Δ vs starting". But iter 119 is 0.5 B
tokens INTO training — past most of the rope re-adaptation, not the
pre-CPT state. Several "retention gain" deltas were inflated because
they included recovery from the rope dip.

Caught on 2026-05-30 when re-reading the report; fixed inline by
pulling matched-config Apertus-Base retention (which IS our iter 0 under
Path B) and computing Δ vs that. Magnitudes shifted:

| Task | Was (Δ vs iter 119) | Corrected (Δ vs true init) |
|---|---:|---:|
| global_mmlu_de | +6.27 pp | +2.25 pp |
| arc_easy | +1.59 pp | −0.85 pp |
| arc_challenge | +1.20 pp | −0.17 pp |
| xnli_de | +1.01 pp | −0.20 pp |
| xnli_ru | −1.53 pp | −1.57 pp |

German MMLU's apparent +6.27 pp gain was mostly rope re-adaptation
recovery; English commonsense regressions surfaced once the correct
baseline was used.

**Task-2 implication.** The pre-CPT state of any CPT run is the model
weights evaluated under the run's *training* geometry. For Task 2 (which
trains on Path A and inherits the released base's Path-A geometry), the
true init IS just Apertus-Base evaluated under its native config — no
matched-config workaround needed. Use this as the retention baseline
column from launch; do not use iter-119-equivalent values.

---

## (3) Recommendations for Task 2

Concrete items for the planning agent to inherit. Ordered by importance.

### (3.0) Keep the corrected hyperparameter regime at the center

The core Task-1 discovery is that the corrected Apertus-faithful CPT regime is
solid. For Task 2, start every arm from the same regime unless the experiment
is explicitly a hyperparameter ablation:

- LR `1.1e-5`, with 1.2 B-token warmup then constant unless a WSD cooldown is
  deliberately added for a long final run.
- AdEMAMix β3 `0.99`, α `8`, β1/β2 `0.9/0.999`, α/β3 warmup aligned to the
  LR warmup.
- Goldfish `k=h=50` with the proven hash settings.
- Same global batch, TP/PP, microbatch, bf16/fp32-master-gradient settings,
  and sidecar checkpoint/eval cadence unless a run's purpose requires a named
  deviation.

Use `goal/hyperparameters.json` as the machine-readable starting point and
write a Task-2 overlay for only the intentional diffs: Path-A geometry,
extension method, TD layer, BPE cutoff, stabilization duration, and any final
cooldown. Do not let those variables drift the corrected regime by accident.

### (3.1) Switch to Path A geometry — STATUS: CONFIRMED, LOCKED (2026-05-31)

`cpt-plan.md` §3.4 Q3.4.10 + `hyperparameters.json[training_geometry.task2_geometry_recommendation]`
record this. Concrete values:

```
rope_theta = 12000000
max_position_embeddings = 65536
rope_scaling = {rope_type: llama3, factor: 8.0, original_max_position_embeddings: 8192, low_freq_factor: 1.0, high_freq_factor: 4.0}
sequence_length = 4096   # training only; geometry supports more
```

**Status update (2026-05-31).** Originally written as "working position";
now LOCKED based on the Path A 0.5 B geometry probe. Bootstrap CIs
(`reports/v4_workspace_path_a/path_a_probe_bootstrap_cis.json`) confirm:

- Path A 0.5 B vs Path B iter 119 at matched tokens: +5.51 pp headline,
  paired 95 % CI [+3.79, +7.25], outside zero. ≥ 2 pp prong satisfied.
- Path A 0.5 B vs Apertus-Base Path A (released): +1.25 pp headline,
  paired 95 % CI [+0.25, +2.27], outside zero. No rope re-adaptation
  dip — Path A statistically clears the released-geometry base at
  0.5 B, opposite of Path B which dipped 4.26 pp below it.
- Path A 0.5 B Greek BPB 0.4365 ≈ Task 1 Path B 5 B endpoint 0.4132,
  at one-tenth the compute.

Verdict + full results in `reports/path_a_probe_results_20260531.md`.

Rationale (now both prospective + confirmed): no rope re-adaptation cost
in the first ~1 B tokens; no matched-config workaround; clean
apples-to-apples comparison vs Apertus-Base on the released geometry.
Cost: there is no bakeoff-Path-A counterpart, but Task 2's primary
comparison is *extension vs Vanilla under the same regime + geometry*,
not *extension vs bakeoff arms*.

**Operational note for Task 2 launch.** The init checkpoint can be
loaded under either Path A or Path B Megatron geometry at runtime
(verified by the probe — the Path-B-converted init worked cleanly with
Path A runtime flags). For Task 2, do the explicit Path A re-conversion
once and store at `…/init_checkpoints/…/megatron_tp2_r17patched_path_a`
so provenance is clean from launch. Probe used the runtime-flag path
under time pressure; not a defect in the result but a cleanup for the
production run.

### (3.2) Lock the canonical eval task list at submission, not interpretation

Take `goal/canonical_eval_tasks.json` (Task-1) as the seed; refine for
Task 2 if needed (e.g. add Greek-specific tasks if Task 2 has new
extension-aware benchmarks). Replace the lm-eval `retention_only`
task-group invocation with an explicit task-list argument populated from
the canonical file. Do not run tasks we've decided not to report.

### (3.3) Decontamination MinHash on training pool vs Greek MCQ prompts

`cpt-plan.md` §6 V1. Plan-coherent deferral for Task 1 (diagnostic); for
Task 2 (production), wire as a launch-blocker. Specific: MinHash the
training pool (`hplt_b1_5b.jsonl` or the Task-2 mix equivalent) against
the prompt + answer text of each of the 4 native Greek MCQ benchmarks
(greekmmlu, ilsp_medical_mcqa, ilsp_mcqa_asep, plutus_qa). Drop or
flag overlapping documents. Land a `decontamination_audit.json` artifact
before training submission.

### (3.4) Rebuild the BPB heldout to drop overflow docs

The current `cpt_greek_heldout_500_20260522.jsonl` has ~29.2 % of docs
prefix-truncated at `max_context=4096`. Within-run trajectory deltas are
valid (same heldout, same truncation); absolute BPB and cross-arm
comparison vs models evaluated at longer context are biased. Rebuild the
heldout to drop docs that overflow 4096 tokens, OR add a
`non_truncated_subset_bpb` field. Track the new heldout file's SHA in
the lockdown.

### (3.5) Track per-task and aggregate

Aggregate native MCQ hides the per-task structure that Task 1 surfaced:
greekmmlu monotone, medical narrow-domain plateau, ASEP non-monotonic,
Plutus volatile. For Task 2, emit per-task CIs as standard output of
every checkpoint review; surface significant task-level deltas in the
status renderer; don't aggregate-only.

### (3.6) Pre-commit decision thresholds (per plan §10 Q8) before launch

Plan §10 Q8 specifies five thresholds (X / M_progress / M_ext / M_van /
T) — they were never locked for Task 1, which is fine because Task 1
was diagnostic. Task 2 is a production-extension experiment with
adjudication consequences; lock the thresholds in v1.x before launch.
This is also a methodological hedge against the
"linear-extrapolation-projection" failure mode from §2.4 — pre-committed
thresholds remove the temptation to predict.

### (3.7) Production-blockers per plan §6 to satisfy before Task 2 launch

- **V1**: decontamination, §3.3 above.
- **V4**: bootstrap CIs on every baseline (Apertus-Base + bakeoff arms + any prior Task-2 init checkpoints). Methodology proven by `v4_bootstrap_cis_native_mcq.json` v3; replicate.
- **V8**: Goldfish hash uniformity over the *extended* vocabulary. Required for Task 2 specifically (Task 1 had no extension). Generate the hash uniformity report against the extended tokenizer's new tokens before extension training.
- **R17**: Apertus extras patch for HF → Megatron roundtrip. Already required and applied for Task 1; reapply for Task 2.

### (3.8) Adversarial-review runner backend selection

`scripts/run_checkpoint_adversarial_review.sh` is hard-wired to
`codex exec`. Add a backend flag (`--backend codex|claude`) so Task 2
can run reviews under whichever model is available at the time. The
prompt and output schema are provider-agnostic.

### (3.9) Consider WSD cooldown for the final 10–20 %

Plan §2.1 left LR schedule shape open (Q2.4.4). We locked constant
post-warmup; no cooldown. Apertus's long-context continuation used WSD
with 1-sqrt cooldown in the final 10–20 %. For a Task-2 7–10 B
extension run, cooldown is well-motivated. Not urgent for Task 1
results-reading; worth adding to Task 2's hyperparameter table.

### (3.10) Stretch decision (10 B) gated on the 5 B report

Plan §1.4 + cpt-plan §6 left the 10 B stretch as a post-result decision.
Per the 5 B report's trajectory shape (warmup → burst → plateau →
endpoint lift), the slope is non-zero and a 10 B continuation would
plausibly resolve (a) whether the trajectory continues to lift or
flattens, (b) whether xnli_ru and English commonsense regressions
reverse or compound, and (c) whether Plutus regression is endpoint-
specific or a continuing slope. Cost: another ~175 GPU-h training +
~30 GPU-h evals = ~205 GPU-h. Not gating Task 2; a parallel
information-value-of-continuation question for the planning agent.

### (3.11) Hardened submitter template — derive from Task-1 chain, don't re-derive (added 2026-06-01)

Three Path-A probe failures (§2.14, §2.15, §2.16) all traced to the same
root cause: I rewrote the `--export` list from `bakeoff_train.sbatch`'s
declared requirements rather than deriving from
`scripts/submit_training_5b_chain.sh` (the Task-1 chain submitter that
already encodes everything correctly).

For Task 2:

1. **Inherit the Task-1 chain submitter's submission shape verbatim.**
   Parameterize only the diffs (geometry, target tokens, run-tag,
   single-shot vs chained). Do NOT rewrite the `--export` list from
   scratch. The chain submitter is the authoritative source.

2. **Add a pre-flight assertion block** to the Task-2 submitter that
   refuses to submit unless:
   - `ARM` is set to one of `{vanilla, retok, centroid, td_layer<N>, …}`
   - `--ntasks-per-node` equals `GPUS_PER_NODE`
   - `TRAIN_TOKENS >= LR_WARMUP_TOKENS` (the Megatron
     `OptimizerParamScheduler` constraint, §2.8 / §2.16)
   - `INIT_CKPT`, `SCRIPT_DIR_OVERRIDE`, `BASE_DATA_PREFIX`,
     `EXT_DATA_PREFIX` are all set and the paths exist on Clariden.
   Three of the four Path-A failures would have surfaced at submit
   time, not at training-init time.

3. **Document the "notional target + early-exit" pattern** for
   probe-style runs: set `TRAIN_TOKENS` to a value that satisfies the
   `>= warmup` constraint and use `EXIT_INTERVAL` for the actual stop
   point. The Path-A probe used `TRAIN_TOKENS=1.5B` + `EXIT_INTERVAL=119`
   to run for 0.5 B tokens under a 1.2 B warmup window. Encoded in
   `scripts/train_config_04a_path_a.env` with a comment.

4. **The env-var-driven `bakeoff_train.sbatch`** (post 2026-05-31 patch
   adding the rope geometry overrides) is now general for both Path A
   and Path B. Reuse it directly; don't fork.

---

## (4) Open items / what Task 1 cannot answer

These are questions Task 1 was not designed to answer and Task 2 should
either inherit as design constraints or explicitly address.

### (4.1) Path-A regime baseline for Vanilla

There is no Path-A bakeoff or Path-A Vanilla CPT to compare against. The
clean "Vanilla under Path-A regime" measurement is the natural Task-2
companion — running Vanilla on Path A in parallel with the extension
arms would give a clean Vanilla baseline at the same geometry the
extension arms use. ~175 GPU-h to 5 B.

### (4.2) Geometry-recovery vs Greek-learning decomposition

Task 1's iter 1192 over-init delta (+7.01 pp on the 3-task headline) is
the sum of (a) rope re-adaptation recovery and (b) actual Greek learning.
We cannot isolate the two without a controlled comparison. A
Vanilla-Path-A run would zero out (a) and isolate (b).

### (4.3) Krikri counterfactual

`cpt-plan.md` §2.4.6 + §1.4 note that Krikri reported +10.8 % Greek
improvement on Llama-3.1-8B-Base with vocabulary extension + 83 B CPT.
Krikri did not isolate the vocab-extension contribution. We have no
direct counterfactual; Task 2 should consider whether to include a
no-extension Vanilla arm at the Path-A geometry as the counterfactual
the Krikri paper lacks.

### (4.4) Domain-specific forgetting (Plutus)

Plutus regressed 5.33 pp at iter 1192. Paired vs Apertus-Base
significant; paired vs iter 834 borderline. n=225, high noise. Cannot
discriminate domain-specific forgetting from finite-sample noise on this
benchmark alone. Task 2 should either (a) include a higher-n Greek
finance benchmark or (b) accept Plutus as too small to read at this
scale.

### (4.5) Russian xnli regression: continuing slope or endpoint transient?

xnli_ru is the only Russian retention signal we have (xstorycloze_ru is
absent from the executed `retention_only` bundle). The regression
direction is consistent (iter 119 +0.0 pp vs init, iter 1192 −1.57 pp
vs init) but the trajectory is too coarse-grained to call. Task 2 — or
a 10 B Task-1 stretch — needed to discriminate.

### (4.6) Per-task task-level cancellation mechanism

iter 477 → iter 834 saw greekmmlu +2.09 pp + ilsp_mcqa_asep −1.92 pp
within the same paired bootstrap. The mechanism for the simultaneous
gain on one Greek MCQ task and loss on another is unexplained. Could be
(a) capacity reallocation, (b) data-mix-driven shift in subject
coverage, (c) noise within bootstrap. Task 2 should track per-task
across checkpoints and flag when cancellation magnitude rises.

---

## (5) References

- `cpt-plan.md` — the experimental plan.
- `goal/goal.md` — Task 1 brief.
- `goal/hyperparameters.json` — Task 1 authoritative settings + Task-2 Path-A recommendation block.
- `goal/canonical_eval_tasks.json` — eval task lockdown.
- `reports/5B_REPORT.md` — the endpoint report.
- `reports/decisions_matrix_20260529.md` — 24-row decisions matrix.
- `reports/v4_bootstrap_cis_native_mcq.json` — V4 v3 bootstrap CIs.
- `reports/script_audit_20260529.md` — comprehensive script audit.
- `reports/plutus_investigation_20260530.md` — Plutus drop investigation.
- `reports/gpu_hours_breakdown_20260530.md` — final compute accounting.
- `reports/plot_*.png` — visual storyline (trajectory, MMLU, lm loss, retention).
- `adversarial_reviews/Vanilla-{0.5B,1B,2B,3.5B,5B}/adversarial_critique.md` — per-checkpoint adversarial reads.
- `RUN_LOG_20260528.md` — full append-only narrative log of the run.

External:

- Apertus paper §2.3 (initial pretraining), §2.5 (long-context extension, Path A geometry).
- AdEMAMix paper (Pagliardini et al.) — Appendix C β3 short-run guidance.
- "Reuse, Don't Retrain" (arXiv 2407.07263) — CPT recipe priors.
- Krikri (arXiv 2505.13772) — closest Greek-CPT precedent.

---

**End of handoff.** Treat `cpt-plan.md` §3 + this document as the joint
input for Task 2 v1.x planning. Task 1 closed.
