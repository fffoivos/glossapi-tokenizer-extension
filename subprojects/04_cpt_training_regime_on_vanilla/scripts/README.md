# 04 / scripts — the run harness

> **In one line:** the 25 files that built the dataset, drove the segmented 5 B training chain, fanned out per-checkpoint eval sidecars, verified handoffs and rendered status — plus the Path-A probe launcher added at the end.
> **Period:** 2026-05-28 → 2026-06-11 (`37888147`, then `a19c136f`). **Status:** complete and no longer run from here; parts were inherited by [`../../05_token_distillation_cpt`](../../05_token_distillation_cpt).
> **Parent:** [`../README.md`](../README.md)

## Why this existed

Two constraints shaped everything here. Clariden's `normal` partition caps jobs at 12 h, so a 5 B-token run cannot be one job — hence a dependency chain of checkpoint-bounded segments. And evaluation must never block training — hence sidecars: a CPU-only watcher notices a finished checkpoint, submits convert + six evals + a checksum job, and a verifier decides when that checkpoint is safe to report on. These scripts do not re-implement the trainer; they drive subproject 03's `bakeoff_train.sbatch` through `TRAIN_CONFIG_OVERRIDE`, which is why Task-1 and bakeoff runs stay mechanically comparable.

## History

- **2026-05-28.** Written and smoked in one day: the recipe generator derives HPLT-only Greek B1 from the 03 bulk recipe rather than hand-copying JSON; the dataset sbatch is CPU-only and guarded; the 04 training config carries Goldfish + `β3=0.99` + 287-step warmup + constant LR. `hplt_b1_dataset_smoke.sbatch` needed two patches before it ran on `xfer` (no `uenv` on `PATH`, then no `/usr/bin/uenv` mounted at all — resolved with a `uv` Python 3.12 environment).
- **2026-05-28, twice.** `submit_training_5b_chain.sh` was patched for segment 1 twice: first to iter 357 after Megatron asserted `lr_warmup_steps < lr_decay_steps` (the segment was shorter than the 287-step warmup), then to iter 300 once real timing showed ~131 s/iter would exceed the 12 h cap. `SAVE_INTERVAL=119` keeps the iter-119/238 report checkpoints inside the longer segment.
- **2026-05-28, evening.** `verify_checkpoint_sidecars.py` was hardened after a review found the handoff gate passed on manifest rows alone: it now requires `expected_outputs_ready` ∧ `slurm_jobs_completed` ∧ `checksum_manifest_ready` ∧ non-empty output dirs (RUN_LOG §"Handoff Gate Hardening", §"Nonempty Output Gate"; Decisions Matrix row O).
- **2026-05-29.** The run's worst bug was here: `submit_checkpoint_sidecars.sh` passed `BENCHMARKS="$NATIVE_BENCHMARKS"` inside `--export=ALL,…` and Slurm split it on commas, so two checkpoints were evaluated on GreekMMLU alone. Fixed by setting `BENCHMARKS=` as an env prefix on the `sbatch` call (hash `7eb4667e…` → `e865c65a…`, still visible at `submit_checkpoint_sidecars.sh:142-145` with the explanatory comment). The whole directory was then audited: [`../reports/script_audit_20260529.md`](../reports/script_audit_20260529.md), 3 critical / 7 major / 9 minor.
- **2026-05-31.** For the Path-A probe, 03's `bakeoff_train.sbatch` was made env-var-driven for rope parameters (defaults unchanged, so Path-B reproducibility is intact), and `train_config_04a_path_a.env` + `submit_04a_path_a_probe.sh` were added. The submitter needed three fixes on live submissions — a missing `ARM=vanilla`, `--ntasks-per-node=1` collapsing `WORLD_SIZE` to 1, and the warmup-decay assertion again (solved here more cleanly with a notional `TRAIN_TOKENS=1.5e9` plus `EXIT_INTERVAL=119`).
- **2026-06-11 (`a19c136f`).** `watch_and_submit_checkpoint_sidecars.sbatch` gained a `CHECKPOINTS_FILE` override so other runs — the comment names the 13.5 B 2-arm experiment — supply their own checkpoint cadence instead of the hard-coded Vanilla list.

## Outcome

- The chain, sidecar and verifier pattern worked end to end for five checkpoints and was carried into Task 2 along with the adversarial-review runner.
- One audit item was never applied: the watcher (`watch_and_submit_checkpoint_sidecars.sbatch:6`) and the checksum sidecar (`submit_checkpoint_sidecars.sh:259`) are still on `--partition=xfer`. Decisions Matrix row I called for `normal --cpus-per-task=64 --mem=400G` before any Task-2 launch, because the Clariden `xfer` maintenance reservation ran to 2026-06-11; the iter-1192 checksum job completed only because the reservation had not yet bitten (5B report §10.7).
- The scripts assume a Clariden mirror of the repo; several also assume subproject 03's `init_bakeoff` tree for conversion and eval sbatches, and one early failure was exactly a missing mirrored file (`_train_config_common.env`).

## Where things are

| Group | Files | Role |
|---|---|---|
| Dataset | `make_hplt_b1_recipe.py`, `hplt_b1_dataset_smoke.sbatch`, `hplt_b1_dataset_build.sbatch`, `validate_dataset_smoke.py`, `build_code_math_heldouts.{py,sbatch}` | CPU-only (`xfer`) build of the 70/24/4/2 HPLT+B1 mix and the code/math heldouts, with doc-id exclusion against the final training mix. |
| Training | `train_config_04_vanilla.env`, `submit_training_smoke.sh`, `submit_training_5b_chain.sh` | The locked regime config and the segmented 5 B chain; the config is sourced by 03's trainer via `TRAIN_CONFIG_OVERRIDE`. |
| Path-A probe | `train_config_04a_path_a.env`, `submit_04a_path_a_probe.sh` | One-shot 0.5 B run under Path-A geometry (`MAX_POSITION_EMBEDDINGS=65536`, `ROTARY_BASE=12000000`, `USE_ROPE_SCALING=1`, factor 8.0); otherwise identical to the Task-1 config. |
| Sidecars | `submit_checkpoint_sidecars.sh`, `watch_and_submit_checkpoint_sidecars.sbatch`, `verify_checkpoint_sidecars.py`, `watch_checkpoint_sidecar_verification.sh`, `write_checkpoint_checksum_manifest.py`, `submit_eval_smoke.sh` | Convert → native MCQ / greek-nlp / Greek+code+math BPB / retention → checksum, plus the watcher that fires them and the verifier that gates the handoff. |
| Matched-config baseline | `build_apertus_base_matched_config.sh`, `eval_apertus_base_matched_config.sbatch` | Builds and evaluates the `rope_theta=500K` / `max_pos=4096` copy of Apertus-Base. Read the header: the reports later downgraded this from baseline to perturbation diagnostic. |
| Status | `collect_5b_report_state.py`, `render_5b_report_status.py`, `update_5b_report_status.sh`, `monitor_5b_status.sh` | Collect state → render Markdown, filtered through `../goal/canonical_eval_tasks.json`; the monitor is read-only over SSH. |
| Reviews | `run_checkpoint_adversarial_review.sh`, `watch_and_run_adversarial_reviews.sh` | The home-side `codex exec` review runner and its watcher. Both were idle from `2026-05-29T00:24Z`, when codex went offline and the reviews moved to Claude Code subagents. |
