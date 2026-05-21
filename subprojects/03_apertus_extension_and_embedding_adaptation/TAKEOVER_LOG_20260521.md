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
- Found the same Slurm spool-path risk in `preprocess_data.sbatch` and `bakeoff_train.sbatch`: both sourced `_train_config_common.env` via `dirname "$0"`, which would resolve under `/var/spool/slurmd` inside Slurm. Patched both to use `SCRIPT_DIR_OVERRIDE` / `SLURM_SUBMIT_DIR`, and patched `submit_all_arms.sh` to pass `SCRIPT_DIR_OVERRIDE`.
- Because Slurm stores sbatch scripts at submission time, canceled the old pending preprocess jobs `2335160` / `2335161` and requeued them after the patch:
  - `2335581` = base-tokenizer preprocess, dependency `afterok:2335159`
  - `2335583` = extended-tokenizer preprocess, dependency `afterok:2335159`
- `2334880` completed successfully in 1h13m42s. Final selected CPT parquet:
  - path: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet`
  - rows: `47,061,862`
  - chars: `227,837,744,625`
  - external Apertus-overlap drop: `2,224,446` rows removed from `49,474,947`
- `2335157` normalize_nfc failed immediately because `normalize_nfc.sh` called a non-existent directory-mode CLI (`--root`, `--pattern`, `--workers`) on `verify_and_normalize_nfc.py`. Patched the wrapper to build the parquet file list itself and run `verify_and_normalize_nfc.py normalize <file> --out <tmp>` via `xargs -P`, then atomically replace each file. Since `$SELECTED` exists, it normalizes selected CPT + replay/code/math only, not raw nanochat or cpt intermediates.
- Requeued corpus chain after the normalize fix:
  - `2335826` = normalize_nfc
  - `2335827` = mix_builder_smoke, dependency `afterok:2335826`
  - `2335828` = mix_builder_full, dependency `afterok:2335827`
  - `2335829` = base-tokenizer preprocess, dependency `afterok:2335828`
  - `2335830` = extended-tokenizer preprocess, dependency `afterok:2335828`
- Corrected V4-HF baseline `2335100` completed successfully: Slurm `COMPLETED`, exit `0:0`, elapsed `01:10:29`. Local small-artifact copy committed under `03_4_implementation_experiments/init_bakeoff/eval/v4_baseline_corrected_20260521/`; large per-task sample JSONLs remain on Clariden.
- Continue monitoring corrected post-conversion eval `2335196` and corpus dependency chain `2335826` -> `2335827` -> `2335828` -> `2335829`/`2335830`. When preprocess passes, submit the three 2B arms with `INIT_CKPT_ROOT=/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480 bash submit_all_arms.sh`.

## Live continuation - 2026-05-21 ~14:45 UTC

- `2335826` normalize_nfc completed successfully in `01:03:16` and atomically replaced the selected CPT parquet. Post-normalization file size is `129,318,720,576` bytes; row/char counts remain the prepare output (`47,061,862` rows, `227,837,744,625` chars). The size delta is from the PyArrow rewrite/compression layout, not an intentional corpus expansion.
- The first post-normalize smoke (`2335827`) failed because Hugging Face `interleave_datasets` does not accept plain Python generators. `mix_builder.py` now uses a custom deterministic weighted sampler instead of the HF interleaver.
- Follow-up smoke attempts exposed slow or premature source setup before writing progress. `mix_builder.py` now skips zero-weight sources before loading them and streams local parquet sources directly with PyArrow batches instead of routing local parquets through HF `load_dataset(..., streaming=True)`.
- The recipe's old source labels were stale relative to the selected nanochat parquet: it used `glossAPI/...` labels and `HPLT/Greek`, while the actual `source_dataset` values are the original dataset names plus `HPLT/ell_Grek`. `recipes/bulk.json` now matches the verified source labels.
- `bigcode/starcoderdata` and candidate fallbacks were gated or script-based in ways that do not work in this CSCS path, and no local code parquet is staged. For the live CPT build, the planned 4% code bucket is set to weight `0.00` and folded into English FineWeb-Edu replay. The active recipe is therefore `70%` Greek, `28%` replay, `0%` code, `2%` math.
- Current active corpus chain:
  - `2336484` = `mix_builder_smoke`, RUNNING, output `/capstor/scratch/cscs/fffoivos/runs/preprocess/mix_smoke_50M.jsonl`
  - `2336485` = `mix_builder_full`, dependency `afterok:2336484`
  - `2336486` = base-tokenizer preprocess, dependency `afterok:2336485`
  - `2336488` = extended-tokenizer preprocess, dependency `afterok:2336485`
- `2336484` was verified alive and writing: at 14:43 UTC it had reached `20,702,489` / `50,000,000` target tokens and the JSONL had grown to `121M`.

## Live continuation - 2026-05-21 ~15:00 UTC

- `2336484` reached the smoke token target and wrote a manifest, but Slurm marked it `FAILED 6:0` because Python aborted during native reader teardown after output was written. Its manifest also exposed the more important quality bug: row-weighted source sampling produced wildly wrong token shares (for example Greek academic/literary dominated the 50M-token smoke).
- Cancelled the dependent full/preprocess jobs (`2336485`, `2336486`, `2336488`) so the bad sampler could not cascade.
- Patched `mix_builder.py` again: source selection is now token-fair (`tokens_so_far / target_weight`), so the next source is whichever is furthest behind its target token budget. On clean success, the script flushes stdout/stderr and exits via `os._exit(0)` to avoid the Clariden native-reader teardown abort while preserving real Python exceptions.
- Fresh smoke `2336566` completed successfully (`COMPLETED 0:0`, elapsed `00:05:26`):
  - output `/capstor/scratch/cscs/fffoivos/runs/preprocess/mix_smoke_50M.jsonl`, `50,000,643` tokens, `33,509` rows, `285M`
  - manifest scheduler `token_fair_min_tokens_over_weight`
  - token shares now match the recipe closely (Greek HPLT `0.3462` vs target `0.3500`, literary `0.1846` vs `0.1820`, academic `0.0554` vs `0.0560`, math `0.0198` vs `0.0200`)
- Clariden normal partition has `MaxTime=12:00:00`; the corrected smoke rate makes one monolithic 7B-token full job too close to the wall-time limit. Added `concat_bulk_mix.sbatch` and made `mix_builder_full.sbatch` array-aware.
- Current sharded full-build chain:
  - `2336647` = `mix_builder_full` array, `0-6%3`, each shard target `1,000,000,000` tokens
  - `2336680` = `concat_bulk_mix`, dependency `afterok:2336647`
  - `2336681` = base-tokenizer preprocess, dependency `afterok:2336680`
  - `2336682` = extended-tokenizer preprocess, dependency `afterok:2336680`
- After the user challenged the full-build speed, verified with `srun --jobid=2337770 --overlap ... ps ...` that a shard is one hot `python3 -u mix_builder.py` process at ~one CPU core. The bottleneck is the deterministic token-counting/writer loop, not the Slurm core allocation. Raised the live array throttle from `3` to `7` with `scontrol update JobId=2336647 ArrayTaskThrottle=7`, starting the final shard `2336647_6` immediately. This is the fastest safe path for the current run without rewriting the builder mid-flight.

## Live continuation - 2026-05-21 ~16:10 UTC

- Rechecked the live corpus direction against the project docs under `03_apertus_extension_and_embedding_adaptation`: the no-code `70/28/0/2` recipe was an operational workaround, not aligned with the intended preservation of multilingual/code/math replay in the CPT mix.
- Verified an accessible code fallback by running a direct 500k-token smoke against `codeparrot/codeparrot-clean-train` on Clariden with `text_column=content`. The smoke completed and produced `505,891` tokens from `210` rows under `/iopsstor/scratch/cscs/fffoivos/tmp/codeparrot_mix_smoke_234930/`.
- Cancelled the no-code chain before canonical concat/preprocess could run:
  - `2336647_0..2` completed, but were from the superseded no-code recipe.
  - `2336647_3..6` were cancelled.
  - downstream no-code jobs `2336680`, `2336681`, and `2336682` were cancelled before output.
- Restored the live bulk recipe to `70%` Greek, `24%` non-Greek replay, `4%` code, `2%` math. The code source is explicitly documented as the `codeparrot/codeparrot-clean-train` fallback, because BigCode StarCoder/The Stack sources were gated or script-backed under the current auth/runtime.
- Relaunched the corrected code-included chain with a separate shard prefix so old no-code shard files cannot collide:
  - `2337839` = `mix_builder_full` array, `0-6%7`, each shard target `1,000,000,000` tokens, `SHARD_PREFIX=/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_code_part_`
  - `2337846` = `concat_bulk_mix`, dependency `afterok:2337839`, output `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl`
  - `2337847` = base-tokenizer preprocess, dependency `afterok:2337846`
  - `2337848` = extended-tokenizer preprocess, dependency `afterok:2337846`
- At launch, six mix shards started immediately and one was pending on resources. Corrected post-conversion eval `2335196` remained running.
- The first code-included launch failed quickly: seven concurrent shards each resolved many HF replay datasets, hitting Hugging Face's `1000 api requests per 5 minutes` limit. Cancelled the array and dependencies before any canonical concat/preprocess could run.
- Steering fix for the rate limit: use the replay/math parquets already staged under `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/{replay,math}` instead of remote HF streaming for replay/math. Pulled the one missing Persian fallback shard (`fas_Arab`) into `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/replay/fas_Arab_fw2/data/fas_Arab/train/000_00000.parquet`.
- A 1M-token local-replay smoke completed successfully at `/iopsstor/scratch/cscs/fffoivos/tmp/local_replay_mix_smoke_161637/smoke.jsonl`. It verified all local replay/math paths and the codeparrot code fallback load.
- Relaunched again with local replay/math and a new shard prefix:
  - `2337911` = `mix_builder_full` array, `0-6%7`, `SHARD_PREFIX=/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_code_local_part_`
  - `2337912` = `concat_bulk_mix`, dependency `afterok:2337911`
  - `2337913` = base-tokenizer preprocess, dependency `afterok:2337912`
  - `2337914` = extended-tokenizer preprocess, dependency `afterok:2337912`
