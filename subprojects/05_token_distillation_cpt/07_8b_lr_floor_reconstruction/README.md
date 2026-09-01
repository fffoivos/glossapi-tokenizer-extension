# 8B LR-floor reconstruction (13.5B tokens)

This subproject reconstructs the completed 8B HPLT-to-GlossAPI experiment as a
new, immutable experiment. It does **not** claim byte identity with the deleted
`curriculum_v2` binaries. The reconstruction uses the published bibliography-
cleaned v2 corpus, the 148,992-token modern-plus-polytonic tokenizer, the
verified layer-11 Token-Distillation initialization, and the new heldout panel.

## Experimental contract

- model: `swiss-ai/Apertus-8B-2509`;
- tokenizer: `fffoivos/apertus-tokenizer-extension@fcd33ec`, subfolder
  `greek-modern-polytonic-tokenizer`, 148,992 rows and no padding;
- initialization: the passed 148,992-row layer-11 TD initialization and its
  zero-drift HF/Megatron round trip;
- total nominal budget: 13.5B tokens, 3,218 updates and 13,497,270,272 effective
  tokens at 4,194,304 tokens/update;
- phase 1: updates 0..2,253, 74.0741% HPLT / 22.2222% foreign-code-math
  replay / 3.7037% old Greek;
- phase 2: updates 2,253..3,218, 74.0741% non-HPLT GlossAPI / 22.2222%
  foreign-code-math replay / 3.7037% old Greek;
- full-run corpus targets: 10B new Greek (70% HPLT, 30% GlossAPI) plus replay
  equal to 35% of new Greek (24% multilingual, 4% code, 2% math, 5% original
  Greek), matching the completed 8B experiment;
- within-pool order: randomized with seed `20260609`;
- shared LR prefix: updates 0..2,574;
- tails: same WSD start/duration/shape, ending at 10%, 20%, or 30% of the
  `5.5e-5` peak LR;
- fixed warmup start: `5.5e-6` for all three trajectories, so only the ending
  LR changes;
- evaluation: all 12 frozen LM-loss heldouts every 25 updates.

The exact recipe is in `configs/recipe_13b_lr_floor.json`.

## Dataset derivation

The 445GB source stage already contains the Greek training shards and all 12
heldout binaries. The inherited replay split was 60/40 and was therefore
repartitioned at the exact 2,253/3,218 boundary in the completed v2 CPU build.
That immutable 164-task replay build is reused by exact input- and heldout-
receipt hashes. Its capacity gate identified small deficits in Chinese,
Russian, Spanish and Japanese only.

The v3 dataset pipeline downloaded one pinned FineWeb2-HQ Parquet shard for each of
those four already-vetted replay sources, verifies exact byte sizes and SHA-256
hashes at a pinned dataset revision, and builds just eight supplement tasks
(four sources times two phases). It does not rebuild the completed 164 tasks or
retokenize the 63.8B-token Greek corpus.

The first v3 GPU startup exposed that the clean July-31 Megatron clone no
longer contained the uncommitted named-extra-validation hook used by the June
experiments. No training iteration ran. The v4 runtime restores that hook as
the explicit clean commit `f8d8a30ba22a807321ec5875abbd9692b9282940`, based
on upstream `c92402e39ef3c8e69ea378a59e79059dc14541f4`. The exact patch is
versioned at `train/runtime_patches/megatron_extra_valid_c92402e.patch`.
The v4 freeze reuses both completed v2 base replay and v3 supplements by exact
receipt hashes; it rebuilds no dataset payloads.

End-to-end validation-hook smoke job `2972665` completed on one Clariden node:
two finite updates, all 12 named heldouts built, 24 source-conditioned loss
records emitted (12 at each update), and exit code `0:0`.

`dataset/freeze_derived_schedule.py` validates the atomic build manifests and
sizes, then selects only:

- phase-1 HPLT, foreign replay and old-Greek shards; and
- phase-2 non-HPLT, foreign replay and old-Greek shards.

It then combines the 164-task base replay with the eight-task overlay and
freezes exact weighted Megatron prefixes, capacity checks, every shard
manifest hash, every payload hash declared by those manifests, and the 12-set
heldout inventory. No 445GB copy is made: the derived dataset is an immutable
manifest and schedule over the already materialized binary payloads.

## Launch graph

`train/submit_three_lr_tails.sh` submits this dependency graph:

```text
phase 1: 0..2253
  -> freeze checkpoint 2253
  -> phase 2 shared stable prefix: 2253..2574
  -> freeze checkpoint 2574
       -> T10: 2574..3218 -> freeze final
       -> T20: 2574..3218 -> freeze final
       -> T30: 2574..3218 -> freeze final
```

The three tail jobs consume the same phase-2 data prefixes, seed, phase-relative
sample index, optimizer state and RNG state. `LR_FINAL` is the only experimental
setting changed between them.

The tail save interval is 107 updates. The six equally spaced averaging
checkpoints are 2675, 2782, 2889, 2996, 3103 and 3210. Megatron also forces a
terminal save at exit iteration 3218; that is the raw final checkpoint.

## CSCS execution

Dataset freezing is CPU-only and may run on Bristen because the MLP filesystems
are shared. Training is Clariden-only (16 GH200 nodes / 64 GPUs per job).

```bash
export LR13_CODE_ROOT=/iopsstor/scratch/cscs/fffoivos/experiments/lr-floor-13b-v4/code
export LR13_RUN_ID=20260801T171214Z-lr-floor-13b-v4

# On Bristen: re-freeze the unchanged data schedule against the explicit
# extra-validation Megatron commit. The v2/v3 payload receipts are reused.
sbatch --parsable --account=a0140 --partition=normal \
  --export="ALL,LR13_CODE_ROOT=$LR13_CODE_ROOT,LR13_RUN_ID=$LR13_RUN_ID" \
  "$LR13_CODE_ROOT/clariden/freeze_dataset.sbatch"

# On Clariden after dataset_manifest.json and training_assets_receipt.json exist.
DRY_RUN=1 "$LR13_CODE_ROOT/train/submit_three_lr_tails.sh"
DRY_RUN=0 CONFIRM_GPU_LAUNCH=APERTUS8B_LR_FLOOR_3WAY \
  "$LR13_CODE_ROOT/train/submit_three_lr_tails.sh"
```

The live launcher refuses Bristen and any cluster other than `clariden`.

## Live v4 production graph

The production graph was submitted at `2026-08-01T17:21:46.885231+00:00`.
Its immutable receipts are:

- dataset manifest SHA-256:
  `5aad0aa73664be39ae349816894e907cd069254db369ef4e10d1502b5a431ba4`;
- training-assets receipt SHA-256:
  `f7d68d5f90b68795902d5786b087dcecffa5c5215b34eae00bcde2241f5687d9`;
- launch-graph SHA-256:
  `6c3e141b3ded28dc8f298fc3d595964f62c80119f8b802e7ae2d4e7927b3b23e`.

The Slurm jobs are:

| Stage | Job ID |
| --- | ---: |
| phase 1, updates 0..2253 | `2972672` |
| freeze update 2253 | `2972673` |
| shared phase-2 prefix, updates 2253..2574 | `2972674` |
| freeze update 2574 | `2972675` |
| T10 tail / final freeze | `2972676` / `2972677` |
| T20 tail / final freeze | `2972678` / `2972679` |
| T30 tail / final freeze | `2972680` / `2972681` |

The live phase-1 preflight passed against Megatron commit
`f8d8a30ba22a807321ec5875abbd9692b9282940`. At iteration 25, all 12
source-conditioned heldouts emitted finite loss values; training then resumed
at iteration 26 with zero skipped and zero NaN iterations. The launch graph is
stored at:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/lr_floor_13b/20260801T171214Z-lr-floor-13b-v4/submissions/launch_graph.json
```

### Phase-boundary recovery

Phase-1 training job `2972672` completed at iteration 2,253 with exit code
`0:0`. Its terminal checkpoint contains 131 non-empty files (138G) and is
bound by checkpoint-tree SHA-256
`77b3ebf23c7f13540f32692d60b262ec1527259924fb00b47973f31408dba490`.

The original receipt job `2972673` was cancelled before its batch script could
run because an `xfer` node cannot execute the freezer from `/iopsstor`. The
first recovery attempt, `2974833`, then exposed the `xfer` system Python 3.6
incompatibility with `from __future__ import annotations`. Neither failed job
changed checkpoint or model bytes.

The launcher now stages the two-file freezer bundle under the run's
`/capstor/.../submissions/freeze_bundle` directory and keeps that utility
Python-3.6-compatible. Recovery freezer `2974886` completed with exit code
`0:0` and emitted the frozen iteration-2,253 receipt with SHA-256
`46c8eed6be60ac534b8a3d5e0b075de5c1b0a8f50e21c2c4556f9c021a67911d`.
Shared-prefix job `2972674` then resumed from iteration 2,253 with global
optimizer/RNG/sample state preserved and the phase-2 data index reset to zero.

The live recovered downstream graph is:

| Stage | Job ID |
| --- | ---: |
| shared phase-2 prefix | `2972674` |
| freeze update 2574 | `2974835` |
| T10 / final freeze | `2972676` / `2974836` |
| T20 / final freeze | `2972678` / `2974837` |
| T30 / final freeze | `2972680` / `2974838` |

Its recovery receipt is
`submissions/launch_graph_recovery_20260802.json`, SHA-256
`d9aa896ea607af9ca9cab674c0f2cfc311a9f0f895dd663fa913b3125317da42`.

### LR-floor branch recovery and completed tails

The shared-prefix freezer `2974835` completed at iteration 2,574 with exit
code `0:0`. The immutable branch receipt has SHA-256
`e7071a16517447e10aa8fd817358f0253f39b2b147f37bcbd5b5984913fe2319`;
its 131-file checkpoint tree has SHA-256
`d532734f54782a06221c42d941ac0231059c8a04d7e02d3d2e0b44e4f7f407c7`.

The first tail launch exposed a resume-only Megatron interaction: although
`--override-opt_param_scheduler` changed the scheduler's `min_lr`, loading the
shared optimizer state restored T10's `min_lr` inside every optimizer parameter
group. `OptimizerParamScheduler.get_lr` gives the parameter-group value
precedence, so the first T10/T20/T30 jobs logged identical learning rates. The
invalid jobs `2972676`, `2972678`, and `2972680` were cancelled after only a
few updates; their logs and TensorBoard files are preserved under each arm's
`invalid_attempt_optimizer_group_min_lr_20260802/` directory and are excluded
from the valid trajectories.

`train/runtime_patches/lr_floor_resume.py` now reapplies the selected floor to
both the scheduler and optimizer parameter groups after restore. Future frozen
assets hash this wrapper. The live v4 recovery used a three-file immutable
Capstor bundle while retaining the original assets preflight. Its launch
receipt is `submissions/launch_graph_lr_floor_fix_20260802.json`, SHA-256
`c716086269614f8d38d2eed81d7203f49374f9a85d3f908539463a707dc1517d`.
The first valid update, 2,575, consumed the same 2,636,800 samples and produced
the same loss (`1.547596`) on all arms while logging the expected distinct
learning rates: T10 `5.445129e-5`, T20 `5.451225e-5`, and T30
`5.457322e-5`.

The completed recovery graph is:

| Arm | Tail job | Final freezer | Tail elapsed |
| --- | ---: | ---: | ---: |
| T10 | `2975267` | `2975268` | `01:55:19` |
| T20 | `2975269` | `2975270` | `01:55:46` |
| T30 | `2975271` | `2975272` | `01:55:40` |

Every job above completed with exit code `0:0`. Each arm contains exactly 644
updates (2,575 through 3,218), 26 complete 12-source validation panels (312
records), zero skipped iterations, zero NaN iterations, and no fatal log
signature. Every required checkpoint at 2,675, 2,782, 2,889, 2,996, 3,103,
3,210, and 3,218 contains 131 non-empty files and 147,634,775,562 bytes.

The immutable terminal receipts are:

| Arm | Receipt SHA-256 | Checkpoint-tree SHA-256 |
| --- | --- | --- |
| T10 | `161cb00ca1077d69db00a3182b85b6b05e8475fddcee84712ee8a98e9f47814c` | `f529775155fdbf2043d9ecde229e0599a2b459a4fbbf78a80fff2e1e505cde74` |
| T20 | `c8d1c1cfe9362e8481547aed3dca9134c92b367d53f0d3b5abc6b18f36d96591` | `c1067de4d6db7ce3656760654d208d005f02e441b9a704e8dfe0b350939de0c0` |
| T30 | `97397c41e60d9e1c69acabd977d2979a1ffa70b08077767a10e10eabcb4665a0` | `3dc5d8fdc151d5b206c1600415006554b06888af3807d8adfe1e91c88088b6b0` |
