# 07 — 8B LR-floor reconstruction (13.5 B tokens)

> **In one line:** the completed HPLT→GlossAPI 13.5 B experiment was rebuilt as a new
> immutable run on the published v2 corpus and the 148,992 tokenizer, then branched at
> update 2,574 into three cooldown floors (10 % / 20 % / 30 % of peak LR); all three tails
> ran to completion and were receipted.
> **Period:** 2026-08-01 → 2026-08-02 (Slurm job records in this README).
> **Status:** completed — three valid tails, terminal receipts frozen. No comparison of the
> three arms is committed in this repository.
> **Came from / led to:** [`../06_25b_midtraining_probe`](../06_25b_midtraining_probe/README.md)
> (its dataset stage and TD init are reused here) → this →
> [`../../07_full_8b_cpt`](../../07_full_8b_cpt), which trains at the WSD-10 floor.

> **Recovered artifact.** This directory was never committed while the work was running; it
> was recovered from a local working tree on 2026-09-01 (`2aec4a66`). Every fact below comes
> from the recovered files themselves.

## Why this existed

The 13.5 B two-phase sweeps in [`../03_training_experiments`](../03_training_experiments/README.md)
could no longer be reproduced — their `curriculum_v2` Megatron binaries had been deleted —
and they had run on the older 148,480 modern-only tokenizer. This subproject rebuilds that
experiment as an immutable, receipt-bound run on the current production assets, and uses it
to answer one open scheduling question the earlier sweeps never touched: **how deep should
the WSD cooldown go?** Everything before update 2,574 is shared, so the three arms differ
only in `LR_FINAL`.

## Experimental contract

From [`configs/recipe_13b_lr_floor.json`](configs/recipe_13b_lr_floor.json)
(`recipe_id: apertus8b_13b_hplt_glossapi_lr_floor_v4`, `status: frozen`):

- 13.5 B nominal / **13,497,270,272 effective tokens**, 3,218 updates at 4,194,304
  tokens/update; sequence 4,096, global batch 1,024.
- Phase 1 (updates 0–2,253) = 74.0741 % HPLT / 22.2222 % foreign-code-math replay /
  3.7037 % old Greek; phase 2 (2,253–3,218) swaps HPLT for non-HPLT GlossAPI at the same
  shares — i.e. the pilot's 10 B-new-Greek + 35 %-of-new-Greek replay proportions.
- Order randomized within pool, seed `20260609`. Shared LR prefix through update 2,574;
  branch tails to a floor of **10 %, 20 % or 30 %** of the `5.5e-5` peak, identical warmup
  start `5.5e-6`, identical WSD shape and duration.
- Tokenizer `fffoivos/apertus-tokenizer-extension@fcd33ec`, subfolder
  `greek-modern-polytonic-tokenizer`, 148,992 rows, no padding; layer-11 TD initialization
  with a zero-drift HF↔Megatron round trip.
- All 12 frozen LM-loss heldouts evaluated every 25 updates; tail save interval 107, giving
  averaging checkpoints 2675/2782/2889/2996/3103/3210 plus a forced terminal save at 3,218.

It explicitly does **not** claim byte identity with the deleted `curriculum_v2` binaries.

## History

**Dataset, in three versions, without rebuilding anything large.** The 445 GB stage from
`06_25b_midtraining_probe` already held the Greek shards and all 12 heldout binaries. The
inherited 60/40 replay split was repartitioned at the exact 2,253/3,218 boundary (164
immutable tasks, reused by receipt hash). Its capacity gate found small deficits in Chinese,
Russian, Spanish and Japanese only, so **v3** downloaded one pinned FineWeb2-HQ Parquet
shard per source and built just eight supplement tasks — no retokenization of the 63.8 B-token
Greek corpus. `dataset/freeze_derived_schedule.py` then froze an immutable manifest and
schedule over the already-materialized payloads rather than copying them.

**v3 failed at GPU startup; v4 fixed the runtime.** The clean July-31 Megatron clone had lost
the (uncommitted) named-extra-validation hook the June experiments relied on. **No training
iteration ran.** v4 restores it as an explicit commit `f8d8a30ba22a807321ec5875abbd9692b9282940`
over upstream `c92402e39ef3c8e69ea378a59e79059dc14541f4`, versioned as
[`train/runtime_patches/megatron_extra_valid_c92402e.patch`](train/runtime_patches/megatron_extra_valid_c92402e.patch).
Smoke job `2972665` then passed: two finite updates, 12 named heldouts, 24 loss records, exit `0:0`.

**Production graph submitted 2026-08-01T17:21:46Z** — phase 1 `2972672`, freeze `2972673`,
shared prefix `2972674`, freeze `2972675`, then T10/T20/T30 tails `2972676`/`2972678`/`2972680`
with their freezers. Launch-graph SHA-256 `6c3e141b…`, dataset manifest `5aad0aa7…`,
training-assets receipt `f7d68d5f…`.

**Two recoveries, neither touching model bytes.**

1. *Phase boundary.* Phase 1 finished at 2,253 (`0:0`, 131-file / 138 G checkpoint tree
   `77b3ebf2…`), but freezer `2972673` was cancelled — an `xfer` node cannot execute the
   freezer from `/iopsstor` — and retry `2974833` then hit `xfer`'s Python 3.6 rejecting
   `from __future__ import annotations`. The launcher now stages a Python-3.6-compatible
   two-file freezer bundle on Capstor; `2974886` succeeded (receipt `46c8eed6…`) and the
   shared prefix resumed with optimizer/RNG/sample state preserved and the phase-2 data
   index reset to zero. Recovery receipt `launch_graph_recovery_20260802.json` (`d9aa896e…`).
2. *The floors did not take.* On the first tail launch all three arms logged **identical
   learning rates**: `--override-opt_param_scheduler` changed the scheduler's `min_lr`, but
   restoring the shared optimizer state put T10's `min_lr` back into every optimizer
   parameter group, and `OptimizerParamScheduler.get_lr` prefers the parameter-group value.
   Jobs `2972676`/`2972678`/`2972680` were cancelled after a few updates and their logs kept
   under each arm's `invalid_attempt_optimizer_group_min_lr_20260802/`, excluded from the
   valid trajectories. [`train/runtime_patches/lr_floor_resume.py`](train/runtime_patches/lr_floor_resume.py)
   reapplies the floor to both the scheduler and the parameter groups after restore.
   Re-launch receipt `launch_graph_lr_floor_fix_20260802.json` (`c7160862…`). The fix is
   visible in the first valid update, 2,575: same 2,636,800 samples and the same loss
   `1.547596` on all three arms, with distinct LRs T10 `5.445129e-5`, T20 `5.451225e-5`,
   T30 `5.457322e-5`.

## Outcome

- Three valid tails completed, each exactly 644 updates (2,575–3,218), 26 complete 12-source
  validation panels (312 records), **zero skipped and zero NaN iterations**, no fatal log
  signature: T10 `2975267` (01:55:19), T20 `2975269` (01:55:46), T30 `2975271` (01:55:40),
  all `0:0`, each required checkpoint 131 non-empty files / 147,634,775,562 bytes.
- Terminal receipts (SHA-256) — T10 `161cb00c…` / tree `f5297751…`; T20 `c8d1c1cf…` /
  `c1067de4…`; T30 `97397c41…` / `3dc5d8fd…`.
- **No arm comparison, evaluation table or floor decision is committed here.** The later
  full run ([`../../07_full_8b_cpt`](../../07_full_8b_cpt)) and the 0.5 B scheduling study
  ([`../../06_dataset_scheduling_experiments`](../../06_dataset_scheduling_experiments))
  both use the **WSD-10** floor, but neither cites this experiment; the 0.5 B design doc
  even calls a "WSD-10/20/30 floor experiment" a *later, separate study*
  (`../../06_dataset_scheduling_experiments/FACTORIAL_EXPERIMENT_DESIGN.md`). Whether this
  run informed that choice is not recorded.

## Where things are

| Path | What it is |
|---|---|
| [`configs/recipe_13b_lr_floor.json`](configs/recipe_13b_lr_floor.json) | The frozen machine contract (v4) — geometry, phases, receipts, floors |
| [`train/submit_three_lr_tails.sh`](train/submit_three_lr_tails.sh) | The launcher that emits the whole dependency graph; refuses any cluster but `clariden` |
| [`train/runtime_patches/lr_floor_resume.py`](train/runtime_patches/lr_floor_resume.py) | The fix for the resume-only `min_lr` bug; hashed into future frozen assets |
| [`train/runtime_patches/megatron_extra_valid_c92402e.patch`](train/runtime_patches/megatron_extra_valid_c92402e.patch) | The recovered named-extra-validation hook |
| [`dataset/freeze_derived_schedule.py`](dataset/freeze_derived_schedule.py) | Freezes the manifest/schedule over existing payloads (no 445 GB copy) |
| [`clariden/`](clariden/) | Replay repartition/supplement builds, dataset + checkpoint freezers, segment sbatch |
| [`tests/test_lr_floor_design.py`](tests/test_lr_floor_design.py) | Encodes the invariants: only the floor differs, shared prefix precedes divergence, equal-interval averaging checkpoints, the param-group fix, Python-3.6 freezer staging |
