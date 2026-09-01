# 05/clariden — the CPU build stages

> **In one line:** nine sbatch stages and a three-mode dispatcher that would have built the 25B probe's Megatron binaries on Clariden CPU nodes; the chain was never run.
> **Period:** 2026-07-12. **Status:** unused; the equivalent live chain is `../../06_25b_midtraining_probe/clariden/submit_data_pipeline.sh`.
> **Came from / led to:** wraps [`../scripts`](../scripts); its outputs were to be consumed by [`../train`](../train).

## The stages

| Stage | Purpose |
|---|---|
| `05_acquire_replay.sbatch` | Restage every pinned replay source (and optionally a clean Megatron checkout). |
| `07_build_old_greek.sbatch` | Rebuild `greek_replay.parquet` from receipted NanoChat shards and the Apertus overlap overlay. Depends on `05`. |
| `10_freeze_inputs.sbatch` | The exact receipt pass over the Phase-04 private release, replay sources, tokenizer tree and Megatron source. Produces `input_receipt.json`. |
| `20_build_heldouts.sbatch` | Build the nine deterministic heldout sets and their exclusion lists. |
| `30_build_train_shards.sbatch` | Training-binary Slurm array (default `MAX_PARALLEL_TRAIN=12`). |
| `40_build_heldout_shards.sbatch` | Heldout-binary Slurm array (default `MAX_PARALLEL_HELDOUT=6`). |
| `50_finalize_bridge.sbatch` | Exact accounting, uniqueness and capacity finalisation; emits `bridge_manifest.json`, `training_mix_79_20_1.json`, `training_data.env`. |
| `60_freeze_training_assets.sbatch` | Freeze launch-time code, TD init checkpoint and roundtrip evidence into `training_assets_receipt.json`. |
| `70_freeze_resume_checkpoint.sbatch` | Freeze one completed training segment before a relaunch (run between GPU segments, not part of the build chain). |

## Gating

`submit.sh` takes `restage`, `freeze`, `after-freeze` or `status` and defaults to `DRY_RUN=1`. A real build needs `CONFIRM_BUILD=1`; restaging additionally needs `CONFIRM_RESTAGE=PINNED_REPLAY_V1`, and replacing an unreceipted skeleton needs a further explicit `RESTAGE_REPLACE=1`. `freeze` and `after-freeze` are split because the dependency graph after freezing can only be sized once `input_receipt.json` exists.

## Outcome

- The chain is a complete design and has no run record in this repository. `BRIDGE_RUN_ID=full-corpus-25b-v1` appears only in the parent README's example commands.
- Its shape survived: [`../../06_25b_midtraining_probe/clariden`](../../06_25b_midtraining_probe/clariden) reproduces the same freeze → heldouts → train/heldout arrays → finalise → freeze-assets order, calls back into [`../scripts/build_binary_shard.py`](../scripts) for the array steps, and adds the phase-manifest and HF-materialisation stages this version lacked.
