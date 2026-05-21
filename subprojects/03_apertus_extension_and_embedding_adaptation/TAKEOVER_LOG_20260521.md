# Takeover log - 2026-05-21

This file records Codex takeover actions after the CSCS overnight run. It is the local, commit-tracked copy of operational changes, checks, job starts/stops, and errors encountered during the handoff.

## Initial live state

- Clariden user/account: `fffoivos` / `a0140`.
- Active Slurm job at takeover: `2334880` (`prepare_greek_pool`) RUNNING on `nid006899`.
- Failed prior `prepare_greek_pool` attempts verified by `sacct`: `2334476`, `2334826`.
- R1 roundtrip verified by `sacct`: `2333864` COMPLETED `0:0`.
- V4-HF baseline job `2334245` had eval artifacts, but Slurm marked `FAILED 13:0` because of the documented `ls | head` SIGPIPE after eval completion.

## Local fixes made

- Added missing `global_mmlu` to `eval/run_eval.sbatch` retention tasks so corrected V4 runs cover Apertus Table 14.
- Marked `v4_baseline_20260521` as a partial baseline for its listed tasks, not the final threshold source.
- Corrected `mix_builder.py` determinism wording: all arms share the same JSONL text stream, but base-vs-extended token IDs differ.
- Corrected `corpus_build/recipes/bulk.json` metadata to match the actual `70/24/4/2` Greek/replay/code/math weights.
- Fixed `normalize_nfc.sh` to include the `cpt/` directory, so the runbook-produced selected parquet is normalized when normalization runs after `prepare_greek_pool`.
- Added `mix_builder_full.sbatch` and moved `preprocess_data.sbatch` from the unavailable `xfer` partition to `normal`, so the after-prepare dependency chain can run unattended.
- Clarified R17 q/k norm evidence and added q/k max-diff printing to future `r1_roundtrip.sbatch` reruns.
- Updated `CSCS_OVERNIGHT_STATE.md` and `SESSION_LOG_20260521.md` with takeover state and known corrections.

## Errors encountered

- GCP active-instance check failed from `home` because `gcloud` requires non-interactive reauthentication:

```text
ERROR: (gcloud.compute.instances.list) There was a problem refreshing your current auth tokens: Reauthentication failed. cannot prompt during non-interactive execution.
```

No GCP instance state was verified during takeover.
- Local `pytest` verification could not run because this checkout has no pytest installed on the default `python3`:

```text
/bin/bash: line 1: pytest: command not found
/usr/bin/python3: No module named pytest
```

Static shell/Python/JSON checks did pass.

## Process decisions

- Did not stop `prepare_greek_pool` job `2334880`: DuckDB temp grew during checks, so it was actively spilling rather than stuck.
- Parallel `rsync` to Clariden has twice closed one SSH connection while other copies succeeded; each failed copy was retried by itself and succeeded. Local/deployed checksums matched after retry.
- A later multi-file `rsync` briefly copied `normalize_nfc.sh` to the remote subproject root instead of `corpus_build/`; removed that accidental remote copy and resynced the script to the correct path.
- Submitted corrected V4 evals after syncing:
  - `2335100` = V4-HF corrected baseline, output `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_baseline_v4_corrected_20260521_121639`
  - `2335101` = V4-post-conversion corrected baseline, output `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_postconv_v4_corrected_20260521_121639`
- Verified both eval jobs entered RUNNING state on normal partition and the logs show `global_mmlu` in `Selected Tasks`.
- V4-post-conversion eval `2335101` failed after ~2.5 minutes with `OSError: [Errno 37] No locks available` from `datasets`/`filelock` during dataset preparation. Patched `run_eval.sbatch` to set per-job `HF_HOME`, `HF_DATASETS_CACHE`, `XDG_CACHE_HOME`, and `TMPDIR` under `/iopsstor/scratch/cscs/fffoivos/tmp/eval_cache_$SLURM_JOB_ID`.
- Resubmitted post-conversion eval as `2335196`, output `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_postconv_v4_corrected_retry_20260521_122535`.
- Verified `2335196` entered RUNNING state and is using `EVAL_CACHE_ROOT=/iopsstor/scratch/cscs/fffoivos/tmp/eval_cache_2335196`.
- Queued the after-prepare corpus chain:
  - `2335157` normalize_nfc, dependency `afterok:2334880`
  - `2335158` mix_builder_smoke, dependency `afterok:2335157`
  - `2335159` mix_builder_full, dependency `afterok:2335158`
  - `2335160` base-tokenizer preprocess, dependency `afterok:2335159`
  - `2335161` extended-tokenizer preprocess, dependency `afterok:2335159`
- Added queueable init-checkpoint scripts:
  - `arms/build_init_checkpoints.sbatch` for HF-format Vanilla/ReTok/Centroid build.
  - `arms/convert_init_checkpoints.sbatch` for HF -> Megatron `torch_dist` release conversion.
  - `arms/submit_init_pipeline.sh` to submit build then conversion with `afterok`.
- Init build first attempt `2335353` failed immediately because Slurm runs a spooled copy of the sbatch file, so `dirname "$BASH_SOURCE"` resolved under `/var/spool/slurmd` instead of the repo. Fixed both init sbatch files to use `SLURM_SUBMIT_DIR` / `SCRIPT_DIR_OVERRIDE`.
- Init build second attempt `2335371` failed because the training uenv `pytorch/v2.6.0:v1` has a Transformers release too old for `model_type=apertus`. Fixed init build/conversion to default to `INIT_UENV_IMAGE=pytorch/v2.9.1:v2`, while leaving 2B training on the recipe's `pytorch/v2.6.0:v1`.
- Submitted live init chain after fixes:
  - `2335382` = `build_init_ckpts`, output `/capstor/scratch/cscs/fffoivos/runs/init/build_init_ckpts-2335382.out`
  - `2335384` = `convert_init_ckpts`, dependency `afterok:2335382`
- `2335382` completed successfully in 2m43s; `2335384` completed successfully in 1m41s. All three init checkpoints now have Megatron release directories:
  - `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron/release`
  - `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/retok/megatron/release`
  - `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/centroid/megatron/release`
- Continue monitoring `2334880`, corrected evals `2335100` / `2335196`, and corpus dependency chain `2335157`-`2335161`. When preprocess passes, submit the three 2B arms with `INIT_CKPT_ROOT=/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480 bash submit_all_arms.sh`.
