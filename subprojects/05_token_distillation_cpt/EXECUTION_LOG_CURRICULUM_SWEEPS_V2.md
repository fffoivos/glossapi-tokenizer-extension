# Curriculum Sweeps v2 Execution Log

## 2026-06-11T11:52:17Z

Started from the v2 handoff plan. Confirmed Clariden SSH works from `home` and
the current CSCS certificate is valid until `2026-06-11T17:51:05Z`.

Implementation fixes applied locally:

- Resized phase build targets to `hplt_only=8.5B`, `glossapi_only=3.7B`,
  `replay_only=5.0B`.
- Set provisional `PHASE1_EXIT_ITER=2261` (`19*119`) in the v2 env, submitter,
  and eval cadence. This must still be recomputed from realized Stage-B `.bin`
  sizes before live sweeps.
- Added a CPU Slurm wrapper for the reused new-Greek holdout builder:
  `curriculum_sweeps_v2/dataset/build_newgreek_vals.sbatch`.
- Made `submit_curriculum_two_phase.sh` live-safe (`DRY_RUN=0
  CONFIRM_LAUNCH=1` required), smoke-overridable (`TOTAL_ITER`, `SEG`,
  `SAVE_INTERVAL`, `NODES`, `GPUS_PER_NODE`, `TIME_LIMIT`), and dry-run
  side-effect-free locally.
- Patched the local trainer mirror for env-gated `EXTRA_VALID_SETS` and
  optional `TRAINER_WRAPPER`.
- Patched the local GreekMMLU watcher mirror so `NATIVE_BENCHMARKS` and
  `SUBMIT_*` flags are explicit across per-checkpoint submit and self-resubmit.

Validation so far:

- Local `bash -n` passed for changed v2 shell scripts/sbatch files, the trainer
  mirror, and the watcher mirror.
- Local `python3 -m py_compile` passed for v2 dataset, analysis, and runtime
  patch Python files.
- Curriculum dry-run produced 4 production segments:
  `0..952`, `952..1904`, `1904..2261`, `2261..3218`.
- Smoke dry-run with `TOTAL_ITER=2 PHASE1_EXIT_ITER=1 SAVE_INTERVAL=1 SEG=1
  NODES=1` produced 2 segments and exported `RESET_DATA_INDEX=1` only on phase 2.

Local commit: `cfdd0e7 Add curriculum sweeps v2 harness`.

Remote validation after sync:

- Remote `bash -n` passed for v2 sbatches/scripts plus the deployed trainer and
  watcher.
- Remote source compile via `compile(...)` passed for v2 Python files.
- Remote curriculum dry-runs matched local output: 4 production segments and a
  2-segment boundary smoke.
- Pilot `DRY_RUN=1 SUBMIT_WATCHERS=0 bash scripts/launch_all.sh` still emitted
  the original two 4-segment arms; pilot configs/scripts do not export
  `TRAINER_WRAPPER` or `EXTRA_VALID_SETS`, so the new hooks stay dormant.
- Free space: `/iopsstor` 713T available, `/capstor` 129T available. No active
  `fffoivos` jobs before launch.

Dataset submission attempt 1:

- Generated 3 phase recipes under remote
  `curriculum_sweeps_v2/dataset/recipes`.
- Submitted CodeParrot `2519199`, new-Greek `2519200`, forgetting `2519201`,
  mix `2519202`, Stage A `2519203`, Stage B `2519204`, new-Greek tokenization
  `2519205`, forgetting tokenization `2519206`.
- `2519199` and `2519200` failed immediately because Slurm rewrites `$0` to
  `/var/spool/.../slurm_script`, so `source "$(dirname "$0")/../paths.env"`
  resolved incorrectly. Cancelled dependent jobs `2519201`-`2519206`.

Fix applied at `2026-06-11T11:58:54Z`:

- All dataset sbatches now source `paths.env` through `V2_DIR` or
  `SLURM_SUBMIT_DIR`, with a `../paths.env` fallback.
- Local `bash -n` passed and patched sbatches were resynced to Clariden.
