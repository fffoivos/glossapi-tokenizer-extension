# Hard H2G: full-panel 8B scoring + no-decay stable-LR branch — plan

Date: 2026-08-22

Status: **AUTHORIZED FOR EXECUTION 2026-08-22** — handed to the execution
agent to execute as written; see the authorization record below for which
gates are satisfied and which remain open. Owner decisions adopted today: the
extension question is reframed as **exploratory** ("do the checkpoints after
the peak keep rising if we never decay?"); full-panel 16,632 scoring is the
corrected primary; the 1.5B peak-LR search is **deferred**.

Parent documents:
[HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md](HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md),
[hard_h_to_g_replication_v1.json](configs/hard_h_to_g_replication_v1.json),
[legacy_public_greekmmlu_v1.json](configs/legacy_public_greekmmlu_v1.json).

## 0. Receipts gathered before writing this plan (read-only, 2026-08-22)

All from the 8B run
`.../20260814T201715Z-r2-v14/runs/efficiency_bound_proven_8b_v112_recovery_codebinding_v3_staticready/segments/s3/attempts/attempt_000003`:

- Checkpoint saves exist at iterations 2380, **2499**, 2618, 2737, 2856, 2975,
  3094, 3213, 3218 (native cadence 119), each a torch-dist checkpoint with
  optimizer state (`iter_0002380` = 130 `.distcp` shards; spot-checked).
- `train.log` LR receipts: `5.500000E-05` at iters 2380, 2499, **2570**;
  `5.060310E-05` at 2580; `4.219479E-05` at 2618. Decay onset is between
  2571–2580, matching the contract derivation (decay = last
  659,179 samples ≈ 644 updates; onset ≈ 2575). **The published accuracy peak
  at 2618 sits inside early decay.**
- Throughput receipt: 8.77 s/iter, 7,480 tokens/s/GPU on 64 GH200 (16 nodes),
  4.194M tokens/update.

Consequence: the branch point is **`iter_0002499`** (last pre-decay save),
not 2380 — 119 updates cheaper. Fallback: 2380.

## Track A — scoring the existing 8B checkpoints (no training dependency; run first)

**A1. Full public panel, common evaluator.** Score all 17 existing 8B
checkpoints on the full 16,632-question `dascim/GreekMMLU@6a03aa06` panel with
the common FP32 HF trajectory evaluator. Reuse: the frozen evaluator bundle
(tree SHA `b7a9e144…`), runtime `h2g_greekmmlu_eval_runtime_20260817_v2`, and
the existing HF exports (no re-conversion). Acceptance guards inherited from
canonical issues: import actual callable symbols under the exact uenv/venv
tuple before allocation ([#88](https://github.com/fffoivos/apertus-cscs-efficiency/issues/88)),
child `srun` stdin from `/dev/null`
([#136](https://github.com/fffoivos/apertus-cscs-efficiency/issues/136)).
Estimate: ~26 GPU-h (previous pass measured ~1.45 GPU-h per checkpoint at
16,159 questions).

**A2. Legacy BF16 replication scoring.** Run the pinned legacy evaluator
(code revision `cfdd0e7b`, BF16, max input 3,072, batch 16) on the
decision-bearing 8B checkpoints **2618, 3218, 3694**. This produces the
like-for-like comparison against the selected β₂=0.999 target
(`9973/16632` best, `9969/16632` final).

**A3. Pre-registered acceptance band (ratify BEFORE reading A2 results).**
Proposal: the run **replicates** if its best legacy-evaluator score is within
**±1.0 pp** of 59.9627% (~2.6 binomial SE on 16,632; margin covers the
corpus rebuild not being document-identical). Outside the band → report the
raw number as a miss and investigate the rebuild delta; no post-hoc widening.

**A4. Report revision.** After A1+A2: full-panel curve becomes the primary 8B
trajectory; the 16,159 subset is relabeled a sensitivity analysis; recompute
trajectory statistics restricted to updates ≤3218 (the extension segment is
LR-confounded); keep the parity-scoped-export caveat. The cross-scale
(1.5B-vs-8B) headline stays on the subset until the deferred 1.5B full-panel
rescore runs.

## Track B — no-decay stable-LR branch (exploratory)

**Question.** Does GreekMMLU keep rising past the observed peak when the LR
never decays — i.e., was the 2618 peak an artifact of decay timing, or had
OpenArchives gains saturated?

**B0. Branch point.** Resume the 8B run from
`/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14/runs/efficiency_bound_proven_8b_v112_recovery_codebinding_v3_staticready/segments/s3/attempts/attempt_000003/payload/checkpoints/iter_0002499`
(other-update source roots: [checkpoint_sources.tsv](presentations/greekmmlu_trajectory_20260822/evidence/checkpoint_sources.tsv)). Pre-flight:
re-verify the LR at the resumed first step is `5.5e-5` and that the resumed
data cursor continues phase-2 order (resume from checkpoint preserves it).

**B1. Contract deltas vs the main run — LR only.** Override the optimizer
scheduler to **constant LR 5.5e-5, no decay, ever** (same override mechanism
the extension used, `extension_override_opt_param_scheduler_expected`).
Everything else identical: phase-2 data order to 3218, AdEMAMix α/β₃ nominal
schedules to 3218 then frozen (as the extension did), Goldfish, clip, batch,
save cadence (keeps the paired grid 2618/2856/3094/3218).

**B2. Segment 1: 2499 → 3218.** 719 updates ≈ 3.0B tokens ≈ 1.75 h wall on
16 nodes ≈ **112 GPU-h**. Because data and order match the main run exactly,
checkpoints 2618/2856/3094/3218 are **paired** with the decayed arm — the
comparison isolates the LR schedule. Score branch checkpoints on the full
panel (common evaluator) as they land.

**B3. Gate at 3218.** If the no-decay arm's own curve is still rising →
continue **3219 → 3694** on the phase-3 unseen blend from cursor 0, still
constant LR (475 updates ≈ 2.0B tokens ≈ 74 GPU-h); this also pairs against
the original terminal-LR extension on identical data. If flat or falling →
stop; the saturation answer is obtained.

**Readout rules (pre-registered).**
- The exploratory signal is the **slope of the no-decay arm** past 2618.
- Do **not** read the level gap vs the decayed arm as "no decay is worse":
  undecayed checkpoints normally trail decayed ones (decay bump).
- If the arm rises materially, the confirmation move is a later short decay
  branch from the best stable point — **out of scope here, own authorization**.

**Retention note.** capstor scratch is purge-managed; the source checkpoints
date from 2026-08-14/21. Launch Track B (or safeguard `iter_0002499`) well
inside the retention window.

## Process and authority

- Kickoff per `prepare-apertus-experiment`: write the intake contract, run
  `plan_experiment_readiness.py`, record separate clocks with
  `record_launch_event.py` → `launch_timeline.json` at the campaign run root.
- Data worker = Clariden login environment; the Mac coordinates only.
- Reuse the canonical `apertus-cscs-efficiency` runner unmodified; any gap is
  an issue first, `workaround_*` in this tree if needed.
- Foreground `apertus-watch` session for every unattended remote chain; no
  fire-and-forget.
- **Owner authorization gates:** (1) the Track A scoring allocation;
  (2) the Track B production training submission; (3) the B3 gate decision;
  (4) ratification of the A3 acceptance band.

### Authorization record — 2026-08-22

The owner instructed that this plan be executed as written. Accordingly:

- Gate (1) **SATISFIED**: Track A scoring allocation authorized
  (spend envelope ≈ 35 GPU-h including branch-checkpoint scoring).
- Gate (2) **SATISFIED**: Track B segment B2 (branch 2499→3218) submission
  authorized (spend envelope ≈ 112 GPU-h).
- Gate (4) **SATISFIED**: the A3 band (±1.0 pp around 59.9627%, legacy
  evaluator) is ratified as of this date, before any A2 result exists.
- Gate (3) **OPEN**: the B3 continuation (3219→3694, ≈ 74 GPU-h) still
  requires an explicit owner decision at the 3218 gate — present the
  no-decay arm's full-panel curve alongside the paired decayed-arm values
  and the ~0.4 pp per-checkpoint noise floor when asking.

Any submission outside these envelopes, or any deviation from the frozen
contracts named above, returns to the owner first.

## Deferred register (owner-confirmed 2026-08-22)

1. **1.5B peak-LR sweep** — deferred. When picked up: short-horizon sweep
   (~3–4 LRs × ~1B tokens, select on validation-loss slope), then judge proxy
   validity by pre-registered **decision agreement** with 8B (sign of the
   HPLT-boundary→OA-endpoint delta; majority subject peak-phase), not score
   or shape matching. "Replication" (goal 1) is untouched — it is 8B-only.
2. **1.5B full-panel rescore** (needed before revising the cross-scale
   headline; cheap; can ride along any later scoring allocation).
3. **Update-0 evaluation** for both scales (parent-model baseline).
4. **Decay-from-best-stable-point branch** (only if B shows a rising arm).

## Cost summary

| Item | Est. |
|---|---|
| A1 full-panel scoring, 17 ckpts | ~26 GPU-h |
| A2 legacy BF16, 3 ckpts | ~5 GPU-h |
| B2 branch 2499→3218 | ~112 GPU-h (~1.75 h wall, 16 nodes) |
| B3 gated extension 3219→3694 | ~74 GPU-h (~1.2 h wall) |
| Branch-checkpoint scoring | ~6–9 GPU-h |
