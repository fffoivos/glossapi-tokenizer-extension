# curriculum_sweeps_v2 — five 13.5 B sweeps

> **In one line:** a rebuilt two-phase 13.5 B dataset plus nine held-out probes, used to
> sweep replay, peak LR and three AdEMAMix parameters — one axis at a time, same geometry
> throughout — producing the recipe that was frozen on 2026-07-11.
> **Period:** 2026-06-11 → 2026-06-17 (analysis and audit through 2026-07-11).
> **Status:** completed. **Do not relaunch** — the `curriculum_v2` Megatron binaries were
> deleted, so every `sweep_*.sh` now fails its data preflight by design.
> **Came from / led to:** the 13.5 B pilot in [`..`](../README.md) → this →
> [`../../PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md`](../../PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md).

## Why this existed

The pilot proved the tokenizer recipe but left every mixture and optimizer knob at a guess,
and it had exposed a mechanism bug: a physically ordered binary does not produce an ordered
curriculum, because Megatron shuffles at consumption. So the harness re-implements the
two-phase curriculum the only way that works — phase 1 trains `blend(hplt, replay)` to
`PHASE1_EXIT_ITER`, phase 2 **resumes** on `blend(glossapi, replay)` with the optimizer,
scheduler and step count intact and only the train dataloader's `consumed_samples` reset to 0
(`train/runtime_patches/reset_data_index_guard.py`). It also builds the six *forgetting*
held-outs the pilot lacked, so retention could be read as loss rather than as benchmarks.

## History

**Build (2026-06-11).** Three dataset submissions failed before one stuck, each for an
infrastructure reason, all recorded in
[`../../EXECUTION_LOG_CURRICULUM_SWEEPS_V2.md`](../../EXECUTION_LOG_CURRICULUM_SWEEPS_V2.md):
Slurm rewrites `$0`, so `source "$(dirname "$0")/../paths.env"` resolved wrong; then the
build Python was x86_64 while `normal` nodes are aarch64 (`Exec format error`); then `xfer`
was congested with next-day start estimates, so the CPU builds were wrapped to run the
aarch64 env inside the PyTorch uenv on `normal` with no GPU GRES. Code replay was switched
from CodeParrot to **StarCoderData** so the code held-out matches an Apertus source family
(`0b07d3a6`). `PHASE1_EXIT_ITER` was pinned to **2261** (19 × 119) from the realized
ext-tokenizer Stage-B sizes — the base-tokenizer binaries would round to 2380, but the
vanilla control deliberately keeps 2261 for comparability.

**Replay sweep (2026-06-11 → 06-12).** Four arms launched 18:13 Z: vanilla control plus TD at
R ∈ {0.35, 0.25, 0.15}, 16 nodes each. A ~9-hour Clariden login/compute outage began at
22:18 Z and was followed by a maintenance reservation; the log carries nine hourly heartbeats
and then a long manual sidecar catch-up run from home because the `xfer` watchers were down.
All four arms completed at 17:29 Z on 06-12, checkpoint 3218, 0 skipped / 0 NaN.

**LR sweep (2026-06-12 → 06-13).** At 18:45 Z the owner picked the replay split, which
required new plumbing: the old wrapper reused one combined `replay_only` binary through a
single weight and could not realize a separate old-Greek slot, so the replay JSONL was split
by `metadata.source` into foreign (3,808,235 rows) and old-Greek (1,223,498 rows) and
re-tokenized. Four LR chains launched 22:20 and all finished by 07:45 on 06-13.

**Alpha (2026-06-13).** Three arms, 09:06 Z → final GreekMMLU at 18:33 Z.

**Beta3 (2026-06-13 → 06-15) — a plan abandoned mid-flight.** The design called for a 2× HPLT
(~27 B) rebuild so the arms would have a longer stretch in which to diverge; the prep chain
was submitted and then stalled with `ReqNodeNotAvail` because `xfer` was reserved for
maintenance. On 06-15 the owner abandoned the 27 B build, ~70 G of staged scratch was deleted,
and the sweep re-ran on the existing 13.5 B binaries — accepting, in writing, that a null
result would then be partly a horizon artifact
([`BETA3_SWEEP_PLAN_20260613.md`](BETA3_SWEEP_PLAN_20260613.md) §0). A β₃ = 0.999 arm was
launched, then cancelled 48 minutes later once it was verified field-for-field identical to
the completed α = 4 run, which became the 0.999 point.

**Beta2 (2026-06-16).** Same trick: the α = 4 run *is* the β₂ = 0.995 point. The decisive
design choice was to **pin the LR warmup at 400 iterations** rather than use the config's
coupled `2/(1−β₂)`, which at β₂ = 0.999 would have made warmup 2,000 iterations — 62 % of the
run — and confounded the comparison.

**Audit (2026-07-11).** `analysis/audit_sweep_configs.py` normalized every arm's raw
`run_metadata.json`, removed run-local fields and the swept axis, and showed an identical
SHA-256 per sweep (β₂ arms: `72992288d2117774…`), recorded in
`results/sweep_config_audit_20260711.json`. Historical launchers were pinned back to their
original β₂ 0.995 / warmup 400 so the new production defaults cannot corrupt reproduction.

## Results

Final-checkpoint GreekMMLU (16,632 questions) unless noted; forgetting read as old-data
held-out LM loss.

| Sweep | Arms | Reading | Chosen |
|---|---|---|---|
| Replay | R = 0.35 / 0.25 / 0.15 | at `curr-5.0B`: 0.5492 / **0.5512** / 0.5271; forgetting flat across all three (+0.045…+0.052 nats); old Greek *improves* (≈ −0.19) at every R | **R = 0.25** → 79/20/1, old-Greek 5 % → 1 % |
| Peak LR | 2.75e-5 / 5.5e-5 / 8.25e-5 / 1.1e-4 | 0.5721 / 0.5850 / 0.5874 / **0.5921**; foreign Δloss −0.0900 / −0.0579 / −0.0279 / **+0.0011** | **5.5e-5** — GreekMMLU alone points to 1.1e-4; the owner chose loss-first |
| α | 0 / 4 / 8 | 0.5663 / **0.5948** / 0.5782 | **4** |
| β₃ | 0.99 / 0.995 / 0.999 | 0.5720 / 0.5791 / **0.5948** | **0.999** |
| β₂ | 0.99 / 0.995 / 0.999 | 0.5861 / 0.5948 / **0.5994** (best new-Greek held-out loss 1.6941) | **0.999**, valid only with the 400-it warmup |

## Where things are

| Path | What it is |
|---|---|
| `results/` | Every decision table (`peak_lr_*`, `alpha_*`, `beta2_*`, `beta3_*`) plus `sweep_config_audit_20260711.json` |
| `train/sweep_{replay,peak_lr,alpha,beta3,beta2}.sh` | The five launchers, plus `submit_curriculum_two_phase.sh` and `submit_vanilla_control.sh` |
| `train/runtime_patches/reset_data_index_guard.py` | The phase-2 data-index reset that makes the two-phase resume a curriculum |
| `train/UPSTREAM_EDITS.md` | The three env-gated edits to the deployed trainer/watcher (9-set extra-valid, trainer wrapper, watcher hardening) |
| `dataset/` | Recipe generation, phase mixing, Stage A/B, StarCoderData staging, the 3 new-Greek and 6 forgetting held-out builders and their tokenization |
| `analysis/` | `collect_greekmmlu.py`, `collect_forgetting_loss.py`, `audit_sweep_configs.py` (+ its test) |
| [`RUNBOOK.md`](RUNBOOK.md) | The historical build/pin/smoke/read procedure, with the artifact-status caveats |
| [`BETA2_SWEEP_PLAN_20260616.md`](BETA2_SWEEP_PLAN_20260616.md) · [`BETA3_SWEEP_PLAN_20260613.md`](BETA3_SWEEP_PLAN_20260613.md) | The two written sweep designs, both marked COMPLETE |

## Caveats recorded at the time

Per-token held-out loss is **not** comparable between the vanilla (base tokenizer) and TD
(ext tokenizer) arms — compare within an arm, or use GreekMMLU. The replay decision was taken
on 2026-06-12 while the final-checkpoint GreekMMLU sidecars were still draining; the numbers
quoted in [`../../PRODUCTION_MIX_DECISION_20260612.md`](../../PRODUCTION_MIX_DECISION_20260612.md)
as "peak" are the iteration-1190 readings, and that document's cited figure file
(`reports/cpt_curriculum_forgetting_learning.html`) is not in the repo. Each arm is one seed.
