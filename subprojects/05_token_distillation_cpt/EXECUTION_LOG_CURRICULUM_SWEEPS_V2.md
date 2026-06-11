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

Local commit: `25635e5 Fix curriculum dataset sbatch path sourcing`.

Dataset submission attempt 2:

- Submitted CodeParrot `2519217`, new-Greek `2519218`, forgetting `2519219`,
  mix `2519220`, Stage A `2519221`, Stage B `2519222`, new-Greek tokenization
  `2519223`, forgetting tokenization `2519224`.
- `2519217` and `2519218` failed immediately with `Exec format error` for
  `/iopsstor/scratch/cscs/fffoivos/python_envs/cpt_build_xfer_py312/bin/python`.
  The jobs were running on `normal` GH/aarch64 nodes while the configured build
  Python is x86_64. Cancelled dependent jobs `2519219`-`2519224`.

Fix applied at `2026-06-11T12:02:21Z`:

- Switched CPU dataset sbatches back to `xfer`, matching the existing x86
  corpus build environment.
- Reduced heavy CPU job memory requests from `400G` to `240G`, because xfer
  nodes report 250G real memory.
- Moved the v2 eval watcher wrapper to a small xfer allocation (`1 CPU`, `4G`);
  it still submits GPU sidecars through the deployed helper.
- Local `bash -n`, local CPU/GPU grep, remote `bash -n`, remote CPU/GPU grep,
  and remote `sbatch --test-only` checks passed for the v2 dataset sbatches.

Follow-up at `2026-06-11T12:05:58Z`:

- `xfer` is valid but congested; `sbatch --test-only` estimated starts around
  the next day.
- Confirmed `/iopsstor/scratch/cscs/fffoivos/python_envs/cpt_build_py312`
  resolves inside `uenv run pytorch/v2.9.1:v2 --view=default --` and imports
  `pyarrow`, `datasets`, `datatrove`, `tokenizers`, `transformers`, `numpy`,
  and `torch` on `aarch64`.
- Pivoted dataset sbatches back to `normal` but wrapped all build Python calls
  in `run_build_py`, which executes the aarch64 env inside the PyTorch uenv.
  The jobs still request no GPU GRES.
- Remote `check_build_py`, import smoke, `bash -n`, CPU/GPU grep, and
  `sbatch --test-only` all passed for this normal/uenv route.

Local commit: `43bf75c Wrap curriculum CPU builds in uenv`.

Dataset submission attempt 3:

- Submitted CodeParrot `2519262`, new-Greek `2519263`, forgetting `2519264`,
  mix `2519265`, Stage A `2519266`, Stage B `2519267`, new-Greek tokenization
  `2519268`, forgetting tokenization `2519269`.
- `2519263` completed in `00:02:25` and wrote:
  - `val_hplt.jsonl` (538,782 docs, ~0.50B token est)
  - `val_openarchives.jsonl` (11,057 docs, ~0.50B token est)
  - `val_greek_phd.jsonl` (3,933 docs, ~0.50B token est)
  - `val_holdout_ids.parquet` (549,839 source ids reported; later mix log loaded
    547,689 unique drop keys)
- `2519268_[0-2]` completed and wrote all 6 new-Greek validation Megatron
  binaries (`val_{hplt,openarchives,greek_phd}_{base,ext}_text_document.{bin,idx}`).
- `2519262` completed in `00:11:30` and wrote
  `val_forget_code.jsonl` (200,984 docs, 2.00B chars) plus
  `forget_holdout_ids.parquet`.
- `2519264` completed in `00:00:49` and merged 1,161,094 forgetting ids. Held-out
  sizes:
  - `english`: 310,544 docs, 1.52B chars (~0.38B token est)
  - `de`: 127,590 docs, 0.44B chars (~0.11B token est)
  - `ru`: 75,915 docs, 0.33B chars (~0.08B token est)
  - `zh`: 79,306 docs, 0.10B chars (~0.03B token est)
  - `code`: 200,984 docs, 2.00B chars (~0.50B token est)
  - `old_greek`: 366,755 docs, 2.00B chars (~0.50B token est)
  The de/ru/zh same-shard source parquets are smaller than the nominal 0.5B-token
  held-out target; this is a result caveat, not a job failure.
- As of `2026-06-11T12:26:15Z`, `2519265_[0-2]` is running and writing progress.
  Current logs show all three phase mixes loaded their drop-id parquets and are
  interleaving:
  - `hplt_only`: 150M/8.5B tokens, ETA about 4.7h from observed rate
  - `glossapi_only`: 150M/3.7B tokens, ETA about 2.2h
  - `replay_only`: 150M/5.0B tokens, ETA about 2.3h
