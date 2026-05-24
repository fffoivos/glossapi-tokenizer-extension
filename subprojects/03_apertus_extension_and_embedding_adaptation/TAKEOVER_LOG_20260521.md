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
- Corrected post-conversion eval `2335196` was healthy but under-requested walltime: it hit the 4h limit at `926/1190` generate requests (`TIMEOUT`, Slurm cancelled at `2026-05-21T18:26:17` local cluster time). Raised `run_eval.sbatch` default walltime to `08:00:00` and resubmitted as split jobs:
  - `2338020` = post-conversion retention-only eval, output `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_postconv_v4_retention_retry_20260521_163240`
  - `2338021` = post-conversion Greek-only eval, output `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_postconv_v4_greek_retry_20260521_163240`
- Caught a second corpus correctness issue before concat: independent 1B-token array jobs would repeat the same per-source prefixes across shards. Cancelled `2337911`/`2337912`/`2337913`/`2337914` before canonical output.
- Patched `mix_builder.py` with `--source-shard-index` / `--source-shard-count`, which partitions each filtered source by eligible row index before token-fair sampling. Patched `mix_builder_full.sbatch` to pass the Slurm array index/count. Two-way smoke at `/iopsstor/scratch/cscs/fffoivos/tmp/source_shard_smoke_164009/` produced zero `(source, doc_id)` overlap.
- Relaunched disjoint sharded corpus chain:
  - `2338121` = `mix_builder_full` array, `0-6%7`, `SOURCE_SHARD_COUNT=7`, `SHARD_PREFIX=/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_disjoint_part_`
  - `2338122` = concat, dependency `afterok:2338121`
  - `2338123` = base-tokenizer preprocess, dependency `afterok:2338122`
  - `2338124` = extended-tokenizer preprocess, dependency `afterok:2338122`
- The disjoint row-sharded array was slower than the duplicate-prefix array and the live Slurm job could not be extended after start (`scontrol update JobId=2338121 TimeLimit=12:00:00` returned permission denied). Cancelled `2338121`/`2338122`/`2338123`/`2338124` before timeout risk.
- Raised `mix_builder_full.sbatch` default walltime to `12:00:00` and relaunched:
  - `2338295` = 12h disjoint `mix_builder_full` array, `0-6%7`, `SHARD_PREFIX=/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_disjoint_12h_part_`
  - `2338301` = concat, dependency `afterok:2338295`
  - `2338303` = base-tokenizer preprocess, dependency `afterok:2338301`
  - `2338304` = extended-tokenizer preprocess, dependency `afterok:2338301`
- Post-conversion retention eval `2338020` completed successfully (`COMPLETED 0:0`, elapsed `00:25:55`). Results are in `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_postconv_v4_retention_retry_20260521_163240/results_2026-05-21T18-58-29.766809.json`.

## Live continuation - 2026-05-21 ~19:10 UTC

- The 12h disjoint array `2338295` completed all seven shards successfully and concat `2338301` completed (`00:00:27`). The raw 7B stream was structurally valid: `7,000,094,000` tokens, `5,769,200` rows, and seven source-disjoint manifests.
- Manifest validation exposed a steering issue before the 2B arms launched: effective top-level mix was `65.185%` Greek, `27.466%` replay, `4.899%` code, `2.450%` math instead of the intended `70/24/4/2`. Cause: `greek_literary` exhausted at `122,832,101` tokens against a `1.274B` target, and the source-level scheduler redistributed that shortfall globally.
- Cancelled the two downstream preprocess jobs from that leaky mix before using them:
  - `2338303` base-tokenizer preprocess: `CANCELLED`, elapsed about `00:10`
  - `2338304` extended-tokenizer preprocess: `CANCELLED`, elapsed about `00:10`
- Preserved the leaky-but-auditable canonical output by moving:
  - `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl`
  - to `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_global_redistribution_2338301.jsonl`
  - and likewise for its manifest.
- Patched `mix_builder.py` to use a bucket-preserving token-fair scheduler: choose the most-behind top-level bucket first, then the most-behind source within that bucket. If one Greek source exhausts, the remaining Greek sources absorb the Greek shortfall before replay/code/math can grow.
- Patched `concat_bulk_mix.sbatch` to aggregate and print `per_bucket` metrics in the concatenated manifest.
- Patched `preprocess_data.sbatch` so `OUTPUT_PREFIX=/.../bulk_mix_text_document` maps to Megatron's raw `--output-prefix /.../bulk_mix`; otherwise Megatron appends `_text_document` twice and produces `bulk_mix_text_document_text_document.{bin,idx}`.
- Verified the scheduler invariant locally with a toy case: when the Greek bucket is behind and `greek_literary` is inactive, the next source is remaining Greek rather than replay.
- Synced the patched scripts to the Clariden execution mirror and relaunched the corrected bucket-preserving chain:

| Job | What | Dependency / output |
|---|---|---|
| `2338878` | `mix_builder_full` array `0-6%7`, 1B tokens per shard, source-row disjoint, bucket-preserving scheduler | writes `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_bucketfix_part_*.jsonl` |
| `2338879` | concat | `afterok:2338878`, writes canonical `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl` |
| `2338880` | base-tokenizer preprocess | `afterok:2338879`, writes `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_megatron/bulk_mix_text_document.{bin,idx}` |
| `2338881` | extended-tokenizer preprocess | `afterok:2338879`, writes `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_ext_megatron/bulk_mix_text_document.{bin,idx}` |

- As of launch, all seven `2338878` array tasks were running. Greek eval `2338021` remained running and healthy.
- Greek post-conversion eval `2338021` completed successfully (`COMPLETED 0:0`, elapsed `03:39:23`). Results are in `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_postconv_v4_greek_retry_20260521_163240/results_2026-05-21T22-12-31.611713.json`.
- Copied compact retention and Greek post-conversion result JSONs plus run metadata locally under `03_4_implementation_experiments/init_bakeoff/eval/v4_postconv_retry_20260521/`. Large per-sample JSONLs remain on Clariden.

## Live continuation - 2026-05-22 smoke and launch hardening

The vanilla training smoke is now proven on the patched R17-preserving checkpoint:

- Final successful one-node smoke: `2341506`, output root `/capstor/scratch/cscs/fffoivos/runs/bakeoff/smoke_vanilla_r17patched_v291npfix_20260522_001421`.
- Slurm result: `COMPLETED 0:0`, elapsed `00:23:46`, allocation `1` node / `4` GPUs.
- Training reached iteration `10` and exited via `--exit-interval 10`.
- Checkpoints were written successfully at `iter_0000005` and `iter_0000010`; `latest_checkpointed_iteration.txt` contains `10`.
- Iterations `1..10` had `0` skipped iterations and `0` NaN iterations.
- Steady-state one-node throughput after warmup was about `8.0k` tokens/sec/GPU, with iteration time around `130s`.
- Consequence: a one-node 2B-token arm is roughly `476 * 130s = 17.2h`, so it does not fit the previous 12h expectation.

Smoke attempts before the successful run exposed and fixed three runtime issues:

- `2341432` and `2341452`: initial xIELU optimizer audit was too naive for Megatron's distributed optimizer. It now accounts for direct refs, model-param maps, main-param shards, and data-parallel coverage.
- `2341460`: `pytorch/v2.6.0:v1` trained through five iterations but failed on checkpoint save because Megatron imports `torch.distributed.checkpoint.filesystem.SerializationFormat`, absent in that image.
- `2341496`: after moving to `pytorch/v2.9.1:v2`, the training uenv did not preserve outer `PYTHONPATH`; `bakeoff_train.sbatch` now exports `PYTHONPATH` inside the `srun bash -c` payload.
- `2341498`: v2.9.1 then failed checkpoint validation because current NumPy lacks `np.product`; `pretrain_gpt_te_guard.py` now installs a compatibility alias `np.product = np.prod` when needed.

Additional hardening:

- Added `check_tokenizer_adapter.py`; base and extended tokenizer adapter checks passed under both `pytorch/v2.6.0:v1` and `pytorch/v2.9.1:v2`.
- Switched the training default uenv to `pytorch/v2.9.1:v2`, because v2.6 can train but cannot write Megatron `torch_dist` checkpoints.
- Made `submit_vanilla_smoke.sh` and `submit_all_arms.sh` pass `ACCOUNT`, `PARTITION`, `NODES`, `GPUS_PER_NODE`, and `TIME_LIMIT` as explicit `sbatch` options. This is required because the static `#SBATCH` lines are one-node defaults and do not by themselves honor config/env overrides.
- Added Slurm shape/uenv fields to `run_metadata.json` so one-node vs two-node throughput can be audited from the run artifact.

Current next gate:

- Submitted two-node vanilla smoke `2341603` with `NODES=2`, output root `/capstor/scratch/cscs/fffoivos/runs/bakeoff/smoke_vanilla_r17patched_v291npfix_2node_20260522_003921`.
- It was allocated `2` nodes / `8` tasks with `WORLD_SIZE=8`; this tests the likely full-bakeoff shape before submitting all three arms.
- `2341603` failed before iteration 1 with inter-node `NCCL Error 2` after checkpoint load, dataset setup, and successful xIELU optimizer audit on all eight ranks. This was not a checkpoint/data/xIELU failure.
- Root steering: the known-good Clariden distributed pattern uses one Slurm task per node running `torchrun --nproc_per_node=4`, not eight direct Slurm tasks. Patched the launch path so `NODES>1` defaults to `LAUNCH_MODE=torchrun`, while one-node jobs keep the already-proven direct Slurm process-per-GPU launch.
- Submitted torchrun two-node retry `2341792`, output root `/capstor/scratch/cscs/fffoivos/runs/bakeoff/smoke_vanilla_r17patched_v291torchrun_2node_20260522_004426`; it is pending on Slurm priority as of submission.
- `2341792` failed immediately (`FAILED 1:0`, elapsed `00:00:17`) because torchrun prepends Python by default and treated the literal `python3` in `training_command.sh` as the script path. Patched torchrun launch to add `--no-python`, so it executes the existing `python3 pretrain_gpt_te_guard.py ...` command exactly.
- `2341810` used torchrun with `--no-python` and reached the first training step, but still failed with `NCCL Error 2` before iteration 1. The failure was preceded by PyTorch warning that NCCL was guessing the device from global rank.
- Patched `pretrain_gpt_te_guard.py` to call `torch.cuda.set_device(int(LOCAL_RANK))` before Megatron distributed initialization.
- Submitted set-device torchrun two-node retry `2341817`, output root `/capstor/scratch/cscs/fffoivos/runs/bakeoff/smoke_vanilla_r17patched_v291torchrun_setdev_2node_20260522_005115`; it is pending on Slurm priority as of submission.
- `2341817` also failed before iteration 1 with `NCCL Error 2` (`FAILED 1:0`, elapsed `00:02:23`). Conclusion for this run: multi-node model execution is not yet viable without deeper NCCL/Clariden debugging.
- Steering change: use the proven one-node path for the bakeoff. One-node throughput estimates a full 2B-token arm at about `17.2h`, while `normal` has `MaxTime=12:00:00`, so the full bakeoff needs checkpointed continuation.
- Patched `bakeoff_train.sbatch` with `RESUME_TRAINING=0|1`. Initial jobs keep `--no-load-optim --no-load-rng`; resume jobs load from the arm's own `checkpoints/` directory and restore optimizer/RNG.
- Patched `submit_all_arms.sh` with `CHAIN_RESUME=1`, which submits a same-arm resume job with `afterany:<initial_job>` dependency.
- Committed the smoke/resume hardening locally as `e17fcab` (`Harden Apertus bakeoff smoke and resume launch`).
- Submitted full one-node chained bakeoff tag `bakeoff_1node_chain_20260522_005620`:
  - initial jobs: `2341822` vanilla, `2341824` retok, `2341826` centroid
  - chained resume jobs: `2341823` vanilla afterany `2341822`, `2341825` retok afterany `2341824`, `2341827` centroid afterany `2341826`
  - output roots: `/capstor/scratch/cscs/fffoivos/runs/bakeoff/bakeoff_1node_chain_20260522_005620_{vanilla,retok,centroid}`
- Initial arms reached real training:
  - `2341822` vanilla iteration 1 at `2026-05-22 03:00:50`, `0` skipped / `0` NaN, `7564.7` tokens/sec/GPU.
  - `2341824` retok iteration 1 at `2026-05-22 03:04:34`, `0` skipped / `0` NaN, `7606.1` tokens/sec/GPU.
  - `2341826` centroid iteration 1 at `2026-05-22 03:06:10`, `0` skipped / `0` NaN, `7637.3` tokens/sec/GPU.
- Health check at `2026-05-22 01:13 UTC`:
  - Slurm status: initial jobs still running, resume jobs still pending on dependency.
  - `2341822` vanilla reached iteration `6`, still `0` skipped / `0` NaN, steady around `7.9k` tokens/sec/GPU.
  - `2341824` retok reached iteration `5`, still `0` skipped / `0` NaN, steady around `8.0k` tokens/sec/GPU.
  - `2341826` centroid reached iteration `4`, still `0` skipped / `0` NaN, steady around `8.0k` tokens/sec/GPU.
  - No arm has reached the first checkpoint yet; this is expected because `SAVE_INTERVAL=65`.
  - Required GCP active-instance safety check from `home` could not complete because `gcloud` user auth needs reauthentication and cannot prompt in non-interactive execution.

## Continuation - 2026-05-22 eval bridge before first checkpoint

- Required GCP active-instance safety check still cannot complete from `home` because `gcloud` auth needs an interactive re-login:
  - `Reauthentication failed. cannot prompt during non-interactive execution.`
- Live Slurm status at `2026-05-22 01:30:34 UTC`:
  - `2341822` vanilla running on `nid006929`, elapsed `00:33:13`.
  - `2341824` retok running on `nid006659`, elapsed `00:29:29`.
  - `2341826` centroid running on `nid006982`, elapsed `00:27:52`.
  - resume jobs `2341823`, `2341825`, `2341827` pending on dependency.
- Latest training-log health:
  - vanilla reached iteration `14` at `2026-05-22 03:29:32` local log time, with `0` skipped / `0` NaN.
  - retok reached iteration `12` at `2026-05-22 03:28:45` local log time, with `0` skipped / `0` NaN.
  - centroid reached iteration `12` at `2026-05-22 03:30:12` local log time, with `0` skipped / `0` NaN.
  - no checkpoint yet; `latest_checkpointed_iteration.txt` is absent for all three arms, expected until `iter_0000065`.
- Found an operational eval gap: `run_bakeoff_arm_eval.sh` expects an HF-format model path, but the live bakeoff writes Megatron `torch_dist` checkpoints.
- Added eval bridge tooling:
  - `eval/convert_bakeoff_checkpoint_to_hf.sbatch` converts one Megatron `iter_XXXXXXX` checkpoint to an HF directory using Megatron `loader core -> saver swissai_hf`.
  - `eval/submit_bakeoff_checkpoint_eval.sh` submits conversion plus dependent `run_eval.sbatch`, with optional tokenizer-fair metrics and new-token diagnostics when `SUBMIT_INTRINSIC=1` and `EVAL_JSONL` is staged.
- Hardened intrinsic eval wrappers:
  - `run_tokenizer_fair_metrics.sbatch` and `run_new_token_diagnostics.sbatch` now default to `pytorch/v2.9.1:v2`.
  - removed job-start `pip install` attempts inside the read-only uenv.
- Documented practical eval cadence in `EVAL_RECIPE.md`:
  - `iter_65`: Greek-only downstream canary on all three arms to prove save -> convert -> eval.
  - `iter_130`, `260`, `390`, and `455/final`: full downstream eval; intrinsic metrics/diagnostics once a held-out JSONL is staged.
- Added held-out slice builder:
  - `eval/build_cpt_heldout_jsonl.py`
  - `eval/build_cpt_heldout_jsonl.sbatch`
  - source: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet`
  - exclusion: all Greek `doc_id`s already used in `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl`
  - output default: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl`
- Synced the held-out builder to the Clariden execution mirror and submitted job `2341867` (`build_cpt_heldout`, `3:00:00`, pending on priority at `2026-05-22 01:38:02 UTC`).
- Submitted bridge smoke on the completed vanilla smoke checkpoint:
  - conversion job `2341869`, dependent limited Greek eval job `2341870`.
  - Result: conversion failed before eval, and `2341870` became `DependencyNeverSatisfied`.
  - Cause: Megatron `loader core` loading a TP=2 `torch_dist` checkpoint calls `torch.distributed.get_world_size()`, but upstream `tools/checkpoint/convert.py` does not initialize a process group.
  - Fix added: `eval/run_megatron_convert_with_pg.py` initializes a single-rank `gloo` process group before running Megatron `convert.py`; `convert_bakeoff_checkpoint_to_hf.sbatch` now uses it.
- Held-out build job `2341867` failed after loading `3,890,581` Greek training doc IDs because duplicate candidate scores/doc IDs allowed `heapq` to compare row dictionaries.
  - Fix added: heap candidates now include a numeric row-position tiebreaker, so ties never compare dict payloads.
- Relaunched held-out build as `2341875`; it started on `nid007538` at `2026-05-22 01:45:09 UTC`.
- Relaunched bridge smoke as conversion `2341876` plus dependent eval `2341877`.
  - Result: `2341876` failed immediately because `SCRIPT_DIR` was not exported into the inner `uenv` shell; `2341877` became `DependencyNeverSatisfied`.
  - Fix added: `convert_bakeoff_checkpoint_to_hf.sbatch` now exports `SCRIPT_DIR` before entering `uenv`.
- Relaunched bridge smoke again as conversion `2341881` plus dependent eval `2341882`.
  - Result: the single-rank `gloo` process group initialized, but the wrapper failed to import Megatron checkpoint plugins (`loader_core` / `core`) because it did not add `tools/checkpoint` to `sys.path`.
  - Fix added: `run_megatron_convert_with_pg.py` now prepends the converter script directory to `sys.path` before running `convert.py`.
- Relaunched bridge smoke again as conversion `2341883` plus dependent eval `2341884`.
  - Result: plugin loading worked, but Megatron checkpoint loading then failed because the data-parallel group was not initialized (`data parallel group with context parallel combined is not initialized`).
  - Fix added: `run_megatron_convert_with_pg.py` now initializes minimal Megatron model-parallel groups (`TP=1`, `PP=1`, `CP=1`) after the single-rank torch process group is created. The loader still reads the checkpoint's TP ranks sequentially.
- Relaunched bridge smoke again as conversion `2341891` plus dependent eval `2341892`.
  - Result: initializing Megatron model-parallel with `TP=1` was too broad; it conflicted with checkpoint TP=2 model construction and produced a `4096` vs `2048` TE row-parallel weight shape mismatch.
  - Fix added: remove model-parallel initialization; instead set only Megatron's data-parallel rank override to `0`, leaving `loader_core` to set TP/PP sizing from the checkpoint args.
- Relaunched bridge smoke again as conversion `2341897` plus dependent eval `2341898`.
  - Result: DP-rank override was not sufficient; sharded-state construction also asks for DP world size and hit the same uninitialized DP group path.
  - Fix added: set Megatron's private `_MPU_DATA_PARALLEL_WORLD_SIZE=1` override next to the DP rank override.
- Relaunched bridge smoke again as conversion `2341960` plus dependent eval `2341961`.
  - Result: conversion reached actual checkpoint load, but Megatron's sharding-integrity validator rejected the single-process sequential TP access pattern for `embedding.word_embeddings.weight`.
  - Fix added: in the converter wrapper only, monkeypatch `validate_sharding_integrity` to a no-op. The actual tensor load remains active and should still fail on missing/malformed tensors.
- Held-out build `2341875` completed its scan but failed quota fill because training consumed all eligible rows in the literary and dictionary/misc source filters.
  - Fix added: default held-out quotas now use only source buckets with remaining training-disjoint docs: HPLT, dialogue/textbooks, academic, and legal/civic.

## Continuation - 2026-05-22 converter bridge proof

- Relaunched held-out build as `2341967`; it started on `nid007538` at `2026-05-22 02:10:01 UTC`.
  - It loaded `3,890,581` Greek training doc IDs from `bulk_mix.jsonl`.
  - By `batch=3,250` / `32,500,000` rows it had filled the revised quotas: HPLT `330`, dialogue/textbooks `60`, academic `60`, legal/civic `50`.
  - The builder intentionally scans the full selected parquet, because the selection rule is lowest hash score per source bucket after excluding training doc IDs.
- Continued bridge smoke attempts on the completed vanilla smoke checkpoint:
  - `2341968` failed after bypassing sharding-integrity validation with `Number of local shards (1) does not match number of local shards metadata ... (2)`.
  - Root cause: the one-rank converter process group made Megatron's torch-dist adapter wrap remote TP shard metadata back onto rank 0.
  - Fix added in `run_megatron_convert_with_pg.py`: for Megatron's ShardedTensor construction only, report `CONVERT_FAKE_SHARDING_WORLD_SIZE=$TENSOR_MODEL_PARALLEL_SIZE` so non-current TP shards stay remote in metadata.
  - `2341979` and `2341981` exposed the next validation layer: PyTorch rejected metadata-only remote TP rank 1 because the real process group contains only rank 0.
  - Fix added in `run_megatron_convert_with_pg.py`: bypass PyTorch's remote-device process-group-rank validation only for rank-only placements in the sequential TP converter shim.
- Successful bridge smoke:
  - Conversion job `2341983` loaded both TP shards from `/capstor/scratch/cscs/fffoivos/runs/bakeoff/smoke_vanilla_r17patched_v291npfix_20260522_001421/checkpoints/iter_0000010`.
  - It wrote HF safetensors to `/capstor/scratch/cscs/fffoivos/runs/eval/eval_bridge_smoke_20260522l_vanilla/iter_0000010_hf`.
  - The output contains four safetensor shards plus tokenizer/config files and `bakeoff_conversion_metadata.json`.
  - Dependent limited Greek eval job `2341984` started on `nid006107`, proving the Slurm dependency and HF output path are usable. Results were still running at the time of this log entry.
- Live bakeoff status at `2026-05-22 02:27 UTC`:
  - `2341822` vanilla, `2341824` retok, and `2341826` centroid are still running.
  - Visible latest training lines remain `0` skipped / `0` NaN.
  - No first checkpoint has landed yet; hold real iter-65 canary submission until `iter_0000065` exists for each arm and the limited eval smoke has finished or clearly advanced past model execution.
- Limited eval smoke `2341984` completed successfully at `2026-05-22 02:29:38 UTC`.
  - Results: `/capstor/scratch/cscs/fffoivos/runs/eval/eval_bridge_smoke_20260522l_vanilla/iter_0000010_greek_only/results_2026-05-22T04-29-35.954152.json`
  - This proves the checkpoint eval bridge end to end: Megatron TP=2 `torch_dist` checkpoint -> HF safetensors -> `run_eval.sbatch` -> results and sample logs.
- Added and launched the iter-65 watcher:
  - script: `eval/watch_and_submit_checkpoint_evals.sh`
  - commit: `b5523cb` (`Add bakeoff checkpoint eval watcher`)
  - remote PID: `153032`
  - log: `/capstor/scratch/cscs/fffoivos/runs/eval/watch_iter65_submit_20260522.log`
  - state dir: `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_watch_iter_0000065_greek_only`
  - It waits for `iter_0000065` per arm and submits exactly once per arm via stamp files.
- Held-out verifier found the first completed JSONL was not acceptable:
  - path checked: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl`
  - failure: duplicate `doc_id` `f2c3acc2ab1892fb0b44e791273780a2a8c0e03c76b954e640d4daad0c41c289`
  - Fix added in commit `7dd139d` (`Enforce heldout doc id uniqueness`): retain an overselected heap per source, then assemble the final rows with a global doc-id guard.
  - Relaunched corrected held-out build as `2341996` on `nid006289`.
  - At `2026-05-22 02:42:43 UTC`, `2341996` was running and had reached `batch=750`; no corrected final JSONL has been accepted yet.
- Corrected held-out build `2341996` completed at `2026-05-22 02:58:39 UTC`.
  - Manifest reports `heap_kept_by_source`: HPLT `6600`, dialogue/textbooks `1200`, academic `1200`, legal/civic `1000`.
  - Manifest reports `selected_by_source`: HPLT `330`, dialogue/textbooks `60`, academic `60`, legal/civic `50`.
  - Manifest reports `duplicate_selected_candidates: 13` and `missing: {}`.
  - Acceptance verifier passed:
    - `heldout_rows 500`
    - `unique_doc_ids 500`
    - source counts match manifest
    - `training_overlap 0` against Greek `doc_id`s in `bulk_mix.jsonl`
  - Accepted heldout paths:
    - `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl`
    - `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.manifest.json`
- Relaunched the iter-65 watcher with intrinsic jobs enabled now that the heldout is accepted.
  - Old watcher PID `153032`/`153033` stopped before any `.submitted` stamps existed.
  - New watcher PID: `246038`
  - Verified environment includes `SUBMIT_INTRINSIC=1` and `EVAL_JSONL=/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl`.
  - As of `2026-05-22 03:10:09 UTC`, it is still waiting for all three `iter_0000065` checkpoint directories.
- Iter-65 checkpoint submission:
  - vanilla checkpoint dir appeared first; watcher submitted conversion/eval/intrinsics as `2342037`/`2342038`/`2342039`/`2342040`.
  - retok and centroid appeared later; watcher submitted retok `2342044`/`2342045`/`2342046`/`2342047` and centroid `2342048`/`2342049`/`2342050`/`2342051`.
  - Retok and centroid conversions completed and their dependent eval/intrinsic jobs started.
  - Vanilla conversion `2342037` was bad: the watcher fired when the `iter_0000065` directory existed but before `latest_checkpointed_iteration.txt` had advanced. The converter tried to load an incomplete checkpoint and then hung after `could not load the checkpoint`.
  - Cancelled only the bad vanilla chain (`2342037`-`2342040`) and resubmitted vanilla after the tracker showed `65`: conversion/eval/intrinsics `2342054`/`2342055`/`2342056`/`2342057`.
  - Hardening added: `watch_and_submit_checkpoint_evals.sh` now requires both the checkpoint directory and `latest_checkpointed_iteration.txt == ITER` before submitting, so future checkpoint watchers do not fire on an incomplete save directory.

## Continuation - 2026-05-22 iter-65 eval babysitting

- Checked live Slurm state at `2026-05-22 03:35-03:40 UTC`.
  - Training jobs `2341822` vanilla, `2341824` retok, and `2341826` centroid were still running with resume jobs pending on dependency.
  - All three checkpoint trackers reported `65`; no later checkpoint had landed yet.
  - Latest observed training logs were past `iter_0000065` with `0` skipped / `0` NaN and no OOM/NCCL/runtime error lines.
- The iter-65 main Greek lm-eval jobs were running:
  - retok `2342045`
  - centroid `2342049`
  - replacement vanilla `2342055`
  - As of this log entry, their output dirs contained only `run_metadata.json`; logs showed real lm-eval progress through context construction and loglikelihood/generate requests, not a startup failure.
- Found and fixed the first intrinsic-eval bug:
  - Failed jobs: retok `2342046`/`2342047`, centroid `2342050`/`2342051`, replacement vanilla `2342056`/`2342057`.
  - Error: `python3: can't open file '/var/spool/slurmd/job.../compute_tokenizer_fair_metrics.py'` or `compute_new_token_diagnostics.py`.
  - Cause: Slurm executed copied sbatch scripts from its spool dir, so `SCRIPT_DIR="$(dirname "$0")"` pointed at `/var/spool/slurmd/...` instead of the eval source directory.
  - Fix: `run_tokenizer_fair_metrics.sbatch` and `run_new_token_diagnostics.sbatch` now resolve `SCRIPT_DIR_OVERRIDE` first, then `SLURM_SUBMIT_DIR` only if it contains the expected Python file, and fail early with a clear missing-script error otherwise. `submit_bakeoff_checkpoint_eval.sh` now exports `SCRIPT_DIR_OVERRIDE` for intrinsic jobs too.
  - Synced the fix to the active Clariden mirror and resubmitted intrinsic-only jobs against the already-converted iter-65 HF outputs as `2342065`-`2342070`.
- Found and fixed the second intrinsic-eval bug:
  - First retry `2342065` failed while loading the model: Transformers rejected `device_map="cuda"` because the CSCS uenv does not include `accelerate`.
  - Fix: `compute_tokenizer_fair_metrics.py` and `compute_new_token_diagnostics.py` no longer pass `device_map`; they load normally and then call `model.to(args.device)`.
  - Cancelled the stale retry jobs `2342065`-`2342070`, synced the fix, and resubmitted as:
    - vanilla metrics `2342072`, diagnostics `2342073`
    - retok metrics `2342074`, diagnostics `2342075`
    - centroid metrics `2342076`, diagnostics `2342077`
  - Health evidence: vanilla metrics loaded all four HF checkpoint shards and reached heldout document processing (`doc 350` observed); retok metrics and diagnostics also loaded model shards successfully.
- Found and fixed a vanilla-control diagnostics edge case:
  - Vanilla has `tokenizer_vocab_size=131072`, so the requested new-token range `[131072, 148480)` is intentionally empty for that arm.
  - First vanilla diagnostics retry `2342073` failed because `_embedding_diagnostics` attempted `quantile()` on an empty new-row slice.
  - Fix: `compute_new_token_diagnostics.py` now clips the requested new-token range to the checkpoint vocab, records `available_new_id_range`, `n_new_requested`, `n_new`, `vocab_size`, and `applicable`, and emits `null` new-row stats when the control arm has no new rows.
  - Resubmitted vanilla diagnostics as `2342081`.
- Iter-65 intrinsic artifacts now exist for all three arms:
  - vanilla:
    - `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_vanilla/iter_0000065_tokenizer_fair_metrics.json`
    - `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_vanilla/iter_0000065_new_token_diagnostics.json`
  - retok:
    - `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_retok/iter_0000065_tokenizer_fair_metrics.json`
    - `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_retok/iter_0000065_new_token_diagnostics.json`
  - centroid:
    - `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_centroid/iter_0000065_tokenizer_fair_metrics.json`
    - `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_centroid/iter_0000065_new_token_diagnostics.json`
- Headline intrinsic values at iter-65:
  - vanilla: vocab `131072`, chars/token `2.5572`, tokens/word `2.6930`, BPC `0.6094`, NLL/char `0.7209`, `n_new=0`, new-token diagnostics not applicable by design.
  - retok: vocab `148480`, chars/token `3.9732`, tokens/word `1.7352`, BPC `0.9750`, NLL/char `1.1532`, `n_new=17408`, E-norm new/existing ratio `1.0578`, new-E cosine mean `0.0842`, mean new-target rank `1659.8`, average new-token prob mass `0.3202`.
  - centroid: vocab `148480`, chars/token `3.9732`, tokens/word `1.7352`, BPC `1.2511`, NLL/char `1.4797`, `n_new=17408`, E-norm new/existing ratio `1.0578`, new-E cosine mean `0.0199`, mean new-target rank `6091.9`, average new-token prob mass `0.3299`.
- Re-established checkpoint watchers beyond iter-65:
  - iter `130`, task group `full`, PID `4011`, launcher log `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_watch_iter_0000130_full_launcher.log`
  - iter `195`, task group `greek_only`, PID `4013`, launcher log `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_watch_iter_0000195_greek_only_launcher.log`
  - iter `260`, task group `full`, PID `4015`, launcher log `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_watch_iter_0000260_full_launcher.log`
  - iter `325`, task group `full`, PID `4017`, launcher log `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_watch_iter_0000325_full_launcher.log`
  - iter `390`, task group `full`, PID `4020`, launcher log `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_watch_iter_0000390_full_launcher.log`
  - iter `455`, task group `full`, PID `4022`, launcher log `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_watch_iter_0000455_full_launcher.log`
  - Each watcher is detached with `SUBMIT_INTRINSIC=1`, `POLL_SECONDS=300`, and `TIMEOUT_SECONDS=129600`.
- Hardened `summarize_bakeoff.py` for the live artifact layout:
  - It now loads `results.json` from the eval output dir while falling back to parent-level `iter_XXXXXXX_tokenizer_fair_metrics.json` and `iter_XXXXXXX_new_token_diagnostics.json`.
  - It also falls back to timestamped `results_*.json`, which is what `lm-eval` actually wrote for the iter-65 Greek canary.
  - It can also summarize from an arm root by taking the latest `iter_*` intrinsic JSONs.
  - It now runs on Clariden login-node `python3` (`3.6.15`), not only inside the PyTorch uenv.
  - It now includes the Greek downstream task columns: `el_arc`, `el_belebele`, `el_xnli`, `el_xquad_f1`, `el_mmlu`, `el_base44`, and `el_piqa`.
  - Smoke command passed:
    - `python3 summarize_bakeoff.py /capstor/..._vanilla/iter_0000065_greek_only /capstor/..._retok/iter_0000065_greek_only /capstor/..._centroid/iter_0000065_greek_only`
- Iter-65 centroid Greek eval completed:
  - result: `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_centroid/iter_0000065_greek_only/results_2026-05-22T05-49-24.403516.json`
  - headline metrics: `arc_challenge_mt_el acc_norm=0.2526`, `belebele_ell_Grek acc_norm=0.3011`, `xnli_el acc=0.4108`, `xquad_el f1=0.0206`, `global_mmlu_full_el acc=0.2839`, `include_base_44_greek_few_shot_en acc=0.2663`, `global_piqa_completions_ell_grek acc_norm=0.5300`.
  - retok and vanilla Greek eval jobs were still running at this log point.
- Iter-65 Greek canary eval is now complete for all three arms.
  - vanilla result: `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_vanilla/iter_0000065_greek_only/results_2026-05-22T05-56-07.902628.json`
  - retok result: `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_retok/iter_0000065_greek_only/results_2026-05-22T05-52-51.585170.json`
  - centroid result: `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_centroid/iter_0000065_greek_only/results_2026-05-22T05-49-24.403516.json`
  - consolidated remote summary: `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_iter0000065_summary.md`
  - local committed copy: `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_iter0000065_summary.md`
  - summary headline:
    - vanilla: BPC `0.6094`, NLL/char `0.7209`, `el_arc=0.427`, `el_belebele=0.549`, `el_xnli=0.410`, `el_xquad_f1=0.357`, `el_mmlu=0.450`, `el_base44=0.449`, `el_piqa=0.630`.
    - retok: BPC `0.9750`, NLL/char `1.1532`, `el_arc=0.279`, `el_belebele=0.408`, `el_xnli=0.397`, `el_xquad_f1=0.211`, `el_mmlu=0.333`, `el_base44=0.315`, `el_piqa=0.530`.
    - centroid: BPC `1.2511`, NLL/char `1.4797`, `el_arc=0.253`, `el_belebele=0.301`, `el_xnli=0.411`, `el_xquad_f1=0.021`, `el_mmlu=0.284`, `el_base44=0.266`, `el_piqa=0.530`.
  - Interpretation: iter-65 is an early canary, not the bakeoff decision point. It does prove the full save -> convert -> HF eval -> intrinsic metrics path, and at this early checkpoint vanilla is still ahead on Greek downstream metrics while retok is ahead of centroid on most new-token integration signals.
- Bootstrap CI pass for iter-65 completed for all three arms:
  - Fixed `compute_bootstrap_cis.py` to strip timestamp suffixes from `samples_*.jsonl` task names and to prefer `acc_norm` where available; `xquad_el` uses `f1`.
  - Follow-up fix: the script now also emits aggregate CIs for `global_mmlu_full_el` and `include_base_44_greek_few_shot_en`, pooling their per-subtask sample files.
  - Remote outputs:
    - `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_vanilla/iter_0000065_greek_only/bootstrap_cis.json`
    - `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_retok/iter_0000065_greek_only/bootstrap_cis.json`
    - `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_centroid/iter_0000065_greek_only/bootstrap_cis.json`
  - Local committed copies live under `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/*_iter0000065_bootstrap_cis.json`.
  - Selected 95% CIs:
    - vanilla: `arc_challenge_mt_el acc_norm 0.4266 [0.3984, 0.4539]`, `belebele_ell_Grek acc_norm 0.5489 [0.5133, 0.5822]`, `xquad_el f1 0.3571 [0.3348, 0.3779]`.
    - retok: `arc_challenge_mt_el acc_norm 0.2790 [0.2534, 0.3046]`, `belebele_ell_Grek acc_norm 0.4078 [0.3756, 0.4411]`, `xquad_el f1 0.2109 [0.1910, 0.2309]`.
    - centroid: `arc_challenge_mt_el acc_norm 0.2526 [0.2261, 0.2773]`, `belebele_ell_Grek acc_norm 0.3011 [0.2722, 0.3311]`, `xquad_el f1 0.0206 [0.0149, 0.0276]`.
  - Aggregate 95% CIs:
    - vanilla: `global_mmlu_full_el acc 0.4497 [0.4412, 0.4578]`, `include_base_44_greek_few_shot_en acc 0.4493 [0.4058, 0.4909]`.
    - retok: `global_mmlu_full_el acc 0.3332 [0.3257, 0.3409]`, `include_base_44_greek_few_shot_en acc 0.3152 [0.2771, 0.3514]`.
    - centroid: `global_mmlu_full_el acc 0.2839 [0.2768, 0.2912]`, `include_base_44_greek_few_shot_en acc 0.2663 [0.2301, 0.3043]`.
- Live training health at `2026-05-22 03:59 UTC`:
  - vanilla `2341822` reached iteration `81/476`, `0` skipped / `0` NaN.
  - retok `2341824` reached iteration `80/476`, `0` skipped / `0` NaN.
  - centroid `2341826` reached iteration `80/476`, `0` skipped / `0` NaN.
  - Later checkpoint watchers were verified by log heartbeat rather than `pgrep`: `pgrep` can land on a different Clariden login node, while the watcher logs for iter `130`, `195`, and `455` ticked from `03:54:48Z` to `03:59:48Z`.
- Added `bakeoff_training/summarize_training_logs.py` to parse Megatron progress lines into JSON/CSV/Markdown training curves.
  - Remote snapshot outputs:
    - `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_training_summary.md`
    - `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_training_summary.json`
    - `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_training_curve.csv`
  - Local committed copies live under `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/`.
  - Snapshot at `2026-05-22T04:05:38Z`:
    - vanilla: iter `84`, tokens `0.352B`, lm loss `2.1833`, `7905` tok/s/gpu, `0` skipped / `0` NaN.
    - retok: iter `83`, tokens `0.348B`, lm loss `4.2539`, `7944` tok/s/gpu, `0` skipped / `0` NaN.
    - centroid: iter `82`, tokens `0.344B`, lm loss `4.9980`, `7991` tok/s/gpu, `0` skipped / `0` NaN.

## Continuation - 2026-05-22 pre-iter-130 checks

- Live Slurm/checkpoint state at `2026-05-22 04:15 UTC`:
  - training jobs `2341822` vanilla, `2341824` retok, and `2341826` centroid are still running.
  - resume jobs `2341823`, `2341825`, and `2341827` are still pending on dependency.
  - checkpoint trackers remain at `65` for all three arms; only `iter_0000065` exists, so iter-130 has not landed yet.
  - iter-130 watcher log is heartbeating and still waiting, with the latest observed tick at `2026-05-22T04:14:49Z`.
- Latest training health at `2026-05-22 04:15 UTC`:
  - vanilla `2341822`: iter `88/476`, tokens `0.369B`, lm loss `2.149925`, `7904.8` tok/s/gpu, `0` skipped / `0` NaN.
  - retok `2341824`: iter `87/476`, tokens `0.365B`, lm loss `4.167927`, `7943.9` tok/s/gpu, `0` skipped / `0` NaN.
  - centroid `2341826`: iter `87/476`, tokens `0.365B`, lm loss `5.029734`, `7988.6` tok/s/gpu, `0` skipped / `0` NaN.
- Verified live tokenizer/data pairing from the actual Slurm logs:
  - vanilla `2341822`:
    - tokenizer: `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509`
    - tokenizer padded vocab: `131072`
    - data prefix: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_megatron/bulk_mix_text_document`
    - dataset builder config confirms `blend=(['...bulk_mix_base_megatron/bulk_mix_text_document'], None)`, `random_seed=20260520`, `sequence_length=4096`, `reset_position_ids=True`, `reset_attention_mask=True`, `eod_mask_loss=True`, `goldfish_loss=False`.
  - retok `2341824`:
    - tokenizer: `/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480`
    - tokenizer padded vocab: `148480`
    - data prefix: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_ext_megatron/bulk_mix_text_document`
    - dataset builder config confirms `blend=(['...bulk_mix_ext_megatron/bulk_mix_text_document'], None)` with the same seed/seq/document-boundary flags as vanilla.
  - centroid `2341826`:
    - tokenizer: `/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480`
    - tokenizer padded vocab: `148480`
    - data prefix: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_ext_megatron/bulk_mix_text_document`
    - dataset builder config confirms `blend=(['...bulk_mix_ext_megatron/bulk_mix_text_document'], None)` with the same seed/seq/document-boundary flags as vanilla.
  - This closes the live-run part of the tokenizer/data pairing check: Vanilla is using base-tokenized data and base vocab, while ReTok/Centroid use extended-tokenized data and extended vocab.

## Continuation - 2026-05-22 R17 documentation correction and live health

- Rechecked the active R17 evidence because older README/comment text still described the obsolete "R17 reset is acceptable for bakeoff" state.
  - The live bakeoff logs show all three arms load patched checkpoints:
    - vanilla: `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched`
    - retok: `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/retok/megatron_tp2_r17patched`
    - centroid: `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/centroid/megatron_tp2_r17patched`
  - Verified roundtrip logs for jobs `2341182`, `2341239`, and `2341241` report `standard_max_abs_diff=0.0`, `r17_max_abs_diff=0.0`, `xielu_max_abs_diff=0.0`, `qk_norm_max_abs_diff=0.0`, no changed-over-tolerance keys, no shape mismatches, and zero smoke-logit drift.
  - Corrected `megatron_patches/README.md`, `loader_apertus_hf.py` comments, and `install.sh` so the documented invariant is now: raw saver_core conversion drops Apertus extras, accepted bakeoff/production checkpoints must be patched with `patch_apertus_extras.py` and verified with `verify_hf_roundtrip.py`.
  - Synced the documentation/comment correction to the Clariden mirror at `/iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/megatron_patches/`.
  - Static checks passed locally and on Clariden: `python3 -m py_compile loader_apertus_hf.py`, `bash -n install.sh`, and stale R17-acceptance phrase search.
- Refreshed the training-curve snapshot from Clariden and copied it locally under `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/`.
  - Snapshot generated at `2026-05-22T04:28:05Z`:
    - vanilla: iter `94`, tokens `0.394B`, lm loss `2.1166`, `7903` tok/s/gpu, `0` skipped / `0` NaN.
    - retok: iter `93`, tokens `0.390B`, lm loss `4.0791`, `7943` tok/s/gpu, `0` skipped / `0` NaN.
    - centroid: iter `93`, tokens `0.390B`, lm loss `5.0340`, `7979` tok/s/gpu, `0` skipped / `0` NaN.
- Additional live log probe at `2026-05-22 04:31 UTC`:
  - vanilla reached iter `95`, retok reached iter `94`, centroid reached iter `94`; all still report `0` skipped / `0` NaN.
  - checkpoint trackers remain at `65` for all arms; iter-130 has not landed yet.
  - resume jobs `2341823`, `2341825`, and `2341827` were verified pending on `afterany` dependency and will load each arm's own `checkpoints/` directory with optimizer/RNG restoration.

## Continuation - 2026-05-22 iter-130 checkpoint and eval submission

- Iter-130 checkpoints landed for all three arms:
  - vanilla: tracker advanced to `130` and `iter_0000130` completed by `2026-05-22T05:48Z`.
  - retok: tracker advanced to `130` and `iter_0000130` completed by `2026-05-22T05:52Z`.
  - centroid: tracker advanced to `130` and `iter_0000130` completed by `2026-05-22T05:52Z`.
- The iter-130 watcher submitted all requested arms without manual duplicate submission:
  - vanilla submitted at `2026-05-22T05:49:52Z`: conversion `2342286`, full lm-eval `2342287`, tokenizer-fair metrics `2342288`, diagnostics `2342289`.
  - retok submitted at `2026-05-22T05:54:54Z`: conversion `2342298`, full lm-eval `2342299`, tokenizer-fair metrics `2342300`, diagnostics `2342301`.
  - centroid submitted at `2026-05-22T05:54:57Z`: conversion `2342302`, full lm-eval `2342303`, tokenizer-fair metrics `2342304`, diagnostics `2342305`.
  - Watcher state dir: `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_watch_iter_0000130_full`.
- Vanilla conversion `2342286` completed successfully (`COMPLETED 0:0`, elapsed `00:01:13`) and wrote HF output to `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_vanilla/iter_0000130_hf`.
- Vanilla tokenizer-fair metrics `2342288` completed successfully (`COMPLETED 0:0`, elapsed `00:01:52`):
  - output: `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_vanilla/iter_0000130_tokenizer_fair_metrics.json`
  - headline: BPC `0.5432`, NLL/char `0.6426`, chars/token `2.5572`, tokens/word `2.6930`.
- Vanilla diagnostics `2342289` wrote its JSON but failed on a summary-print formatting bug for the base-vocab control arm:
  - cause: vanilla has no available new rows (`n_new=0`), so `new_mean`/`new_p50`/ratio are `null`; the compact summary tried to format them with `:.3f`.
  - fix: `compute_new_token_diagnostics.py` now formats optional floats as `n/a` for the control/no-new-token case, including D6/D7 summary lines.
  - synced the fix to the Clariden mirror and submitted vanilla diagnostics retry `2342322`.
- Refreshed training curve snapshot after iter-130:
  - generated at `2026-05-22T06:03:47Z`.
  - vanilla: iter `137`, tokens `0.575B`, lm loss `1.9931`, `7914` tok/s/gpu, `0` skipped / `0` NaN.
  - retok: iter `136`, tokens `0.570B`, lm loss `3.5637`, `7939` tok/s/gpu, `0` skipped / `0` NaN.
  - centroid: iter `136`, tokens `0.570B`, lm loss `4.6826`, `7982` tok/s/gpu, `0` skipped / `0` NaN.
  - Local copies under `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/` were updated.

## Continuation - 2026-05-22 iter-130 evaluation completion

- Iter-130 eval stack completed for all arms.
  - Slurm accounting shows the following jobs completed with `0:0`: conversions `2342286` vanilla, `2342298` retok, `2342302` centroid; full evals `2342287` vanilla, `2342299` retok, `2342303` centroid; tokenizer-fair jobs `2342288` vanilla, `2342300` retok, `2342304` centroid; diagnostics `2342301` retok, `2342305` centroid, and retry `2342322` vanilla.
  - The original vanilla diagnostics job `2342289` failed with `1:0` only after writing the JSON, because the compact print path did not handle `null` new-token stats for the base-vocab control. The retry after the formatter fix completed successfully.
  - Consolidated remote summary: `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_iter0000130_summary.md`.
  - Local compact artifacts copied under `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/`: per-arm lm-eval `results.json`, bootstrap CIs, run metadata, tokenizer-fair metrics, new-token diagnostics, plus `bakeoff_1node_chain_20260522_005620_iter0000130_digest.md`.
- Iter-130 headline digest:
  - checkpoint: `iter_0000130`, about `0.545B` training tokens at checkpoint time, `130 / 476` planned steps.
  - vanilla: BPC `0.5432`, NLL/char `0.6426`, `el_arc=0.4275`, `el_belebele=0.5556`, `el_xnli=0.4137`, `el_xquad_f1=0.3524`, `el_mmlu=0.4459`, `el_base44=0.4819`, `el_piqa=0.5900`, `hellaswag=0.7648`, `arc_c=0.5614`, `mmlu=0.5572`.
  - retok: BPC `0.7561`, NLL/char `0.8943`, `el_arc=0.3157`, `el_belebele=0.4678`, `el_xnli=0.3916`, `el_xquad_f1=0.2737`, `el_mmlu=0.3693`, `el_base44=0.3859`, `el_piqa=0.6200`, `hellaswag=0.7494`, `arc_c=0.5290`, `mmlu=0.5538`.
  - centroid: BPC `1.1318`, NLL/char `1.3387`, `el_arc=0.2483`, `el_belebele=0.3211`, `el_xnli=0.3679`, `el_xquad_f1=0.0253`, `el_mmlu=0.2807`, `el_base44=0.2862`, `el_piqa=0.5400`, `hellaswag=0.7613`, `arc_c=0.5614`, `mmlu=0.5580`.
  - Interpretation remains early-checkpoint only: vanilla is still strongest on downstream Greek and retention-style task metrics; retok is clearly ahead of centroid on Greek downstream and new-token integration; centroid is still weak on Greek use despite okay retention-style scores.
- Refreshed live training snapshot at `2026-05-22T07:22:01Z`:
  - vanilla: iter `171`, tokens `0.717B`, lm loss `1.9547`, `7901` tok/s/gpu, `0` skipped / `0` NaN.
  - retok: iter `170`, tokens `0.713B`, lm loss `3.3444`, `7935` tok/s/gpu, `0` skipped / `0` NaN.
  - centroid: iter `170`, tokens `0.713B`, lm loss `4.4857`, `7989` tok/s/gpu, `0` skipped / `0` NaN.
- Live watcher state at `2026-05-22T07:21:40Z`:
  - Training jobs `2341822`, `2341824`, and `2341826` are still running.
  - Resume jobs `2341823`, `2341825`, and `2341827` remain pending on dependency.
  - Checkpoint trackers remain at `130`; iter-195 watcher is heartbeating and waiting for all three `iter_0000195` checkpoint directories.
- GCP cost check was attempted again from `home` and is still blocked by non-interactive `gcloud` reauthentication (`gcloud auth login` required). No GCP instance state has been verified in this continuation turn.

## Continuation - 2026-05-22 iter-195 checkpoint and Greek-only eval

- Live state before iter-195:
  - GCP cost check was attempted again and is still blocked by non-interactive `gcloud` reauthentication (`gcloud auth login` required).
  - Training snapshot at `2026-05-22T08:01:15Z`: vanilla iter `189`, retok iter `188`, centroid iter `188`; all `0` skipped / `0` NaN.
- Iter-195 checkpoint gating behaved correctly:
  - At `2026-05-22T08:16:30Z`, all three `iter_0000195` checkpoint directories were visible, but only vanilla's tracker had advanced to `195`; retok and centroid still said `130`.
  - The watcher did not submit early. It logged `checkpoint dir exists but tracker says '130'` for vanilla at `08:14:52Z`, then waited.
  - At `2026-05-22T08:19:52Z`-`08:19:59Z`, all trackers were `195` and the watcher submitted all three arms.
- Iter-195 job ids:
  - vanilla: conversion `2343234`, Greek-only lm-eval `2343235`, tokenizer-fair metrics `2343236`, diagnostics `2343237`.
  - retok: conversion `2343238`, Greek-only lm-eval `2343239`, tokenizer-fair metrics `2343240`, diagnostics `2343241`.
  - centroid: conversion `2343242`, Greek-only lm-eval `2343243`, tokenizer-fair metrics `2343244`, diagnostics `2343245`.
  - All twelve jobs completed with Slurm exit `0:0`; the three Greek-only evals took `00:23:31` vanilla, `00:21:27` retok, and `00:21:28` centroid.
- Iter-195 compact artifacts:
  - Remote summary: `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_iter0000195_summary.md`.
  - Local copies under `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/`: per-arm `results.json`, `bootstrap_cis.json`, `run_metadata.json`, tokenizer-fair metrics, new-token diagnostics, plus `bakeoff_1node_chain_20260522_005620_iter0000195_digest.md`.
- Iter-195 headline digest:
  - vanilla: BPC `0.5293`, NLL/char `0.6262`, `el_arc=0.4164`, `el_belebele=0.5478`, `el_xnli=0.4028`, `el_xquad_f1=0.3219`, `el_mmlu=0.4381`, `el_base44=0.4493`, `el_piqa=0.6400`.
  - retok: BPC `0.6827`, NLL/char `0.8075`, `el_arc=0.3259`, `el_belebele=0.4467`, `el_xnli=0.3803`, `el_xquad_f1=0.3077`, `el_mmlu=0.3858`, `el_base44=0.3786`, `el_piqa=0.5800`.
  - centroid: BPC `1.0396`, NLL/char `1.2296`, `el_arc=0.2594`, `el_belebele=0.3244`, `el_xnli=0.3627`, `el_xquad_f1=0.0261`, `el_mmlu=0.2841`, `el_base44=0.3043`, `el_piqa=0.5300`.
  - New-token integration: retok `D1_top1=0.2526`, `D2_mass=0.3398`, `D4_top1_new=0.4829`, `D5_util=0.102`; centroid `D1_top1=0.0403`, `D2_mass=0.3361`, `D4_top1_new=0.2066`, `D5_util=0.036`.
  - Interpretation remains pre-decision: vanilla still leads Greek downstream and BPC; retok has narrowed the gap and is much healthier than centroid on new-token use; centroid remains weak on Greek use.
- Training snapshot after iter-195 eval completion:
  - generated at `2026-05-22T08:48:32Z`.
  - vanilla: iter `210`, tokens `0.881B`, lm loss `1.8885`, `7910` tok/s/gpu, `0` skipped / `0` NaN.
  - retok: iter `209`, tokens `0.877B`, lm loss `3.2570`, `7936` tok/s/gpu, `0` skipped / `0` NaN.
  - centroid: iter `210`, tokens `0.881B`, lm loss `4.3097`, `7984` tok/s/gpu, `0` skipped / `0` NaN.

## Continuation - 2026-05-22 iter-260 full eval

- Live efficiency check:
  - Direct `nvidia-smi` sampling inside the three active Slurm allocations showed all 12 training GPUs at `98-100%` utilization.
  - Per-arm HBM use was roughly `85-88 GiB / 97.9 GiB`; power draw was roughly `480-515 W` per GPU.
  - Logged steady throughput remained near `7.9k` tok/s/GPU and `0` skipped / `0` NaN. The run is not maximally optimized because it uses `mb=2` for memory safety after `mb=4` OOM, but the allocated GPUs are being used hard.
- GCP cost check was attempted again from `home` and remains blocked by non-interactive `gcloud` reauthentication (`gcloud auth login` required). No GCP instance state has been verified in this continuation turn.
- Iter-260 checkpoint gating:
  - At `2026-05-22T10:43:43Z`, training logs showed all arms at or beyond iter `260`, but checkpoint tracker files lagged async-save completion.
  - The watcher did not submit early. It submitted vanilla and centroid only after their trackers reached `260`; retok's tracker advanced shortly after that pass and was submitted on the next watcher tick.
- Iter-260 job ids:
  - vanilla: conversion `2344155`, full lm-eval `2344156`, tokenizer-fair metrics `2344157`, diagnostics `2344158`.
  - centroid: conversion `2344159`, full lm-eval `2344160`, tokenizer-fair metrics `2344161`, diagnostics `2344162`.
  - retok: conversion `2344174`, full lm-eval `2344175`, tokenizer-fair metrics `2344176`, diagnostics `2344177`.
  - All twelve jobs completed with Slurm exit `0:0`; full eval elapsed times were `00:50:55` vanilla, `00:50:38` retok, and `00:47:44` centroid.
- Iter-260 compact artifacts:
  - Remote summary: `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_iter0000260_summary.md`.
  - Local copies under `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/`: per-arm `results.json`, `bootstrap_cis.json`, `run_metadata.json`, tokenizer-fair metrics, new-token diagnostics, plus `bakeoff_1node_chain_20260522_005620_iter0000260_digest.md`.
- Iter-260 headline digest:
  - vanilla: BPC `0.5173`, NLL/char `0.6120`, `el_arc=0.4061`, `el_belebele=0.5067`, `el_xnli=0.4092`, `el_xquad_f1=0.3022`, `el_mmlu=0.4285`, `el_base44=0.4239`, `el_piqa=0.6200`, `hellaswag=0.761`, `arc_c=0.536`, `mmlu=0.540`.
  - retok: BPC `0.6370`, NLL/char `0.7535`, `el_arc=0.3439`, `el_belebele=0.4600`, `el_xnli=0.3735`, `el_xquad_f1=0.3261`, `el_mmlu=0.3829`, `el_base44=0.4112`, `el_piqa=0.5800`, `hellaswag=0.738`, `arc_c=0.509`, `mmlu=0.545`.
  - centroid: BPC `0.9875`, NLL/char `1.1680`, `el_arc=0.2551`, `el_belebele=0.3378`, `el_xnli=0.3398`, `el_xquad_f1=0.0239`, `el_mmlu=0.2834`, `el_base44=0.3098`, `el_piqa=0.5100`, `hellaswag=0.757`, `arc_c=0.546`, `mmlu=0.549`.
  - New-token integration: retok `D1_top1=0.2915`, `D2_mass=0.3388`, `D4_top1_new=0.5268`, `D5_util=0.150`; centroid `D1_top1=0.0606`, `D2_mass=0.3406`, `D4_top1_new=0.2625`, `D5_util=0.076`.
  - Interpretation remains pre-decision: vanilla still leads Greek BPC and most downstream Greek metrics; retok is narrowing and is clearly healthier than centroid on new-token use; centroid remains weak for Greek despite okay retention-style scores.

## Continuation - 2026-05-22 first 12h handoff and resume2 fix

- The initial 12h jobs completed cleanly before the Slurm walltime:
  - vanilla `2341822`: `COMPLETED 0:0`, saved latest tracker `316`.
  - retok `2341824`: `COMPLETED 0:0`, saved latest tracker `317`.
  - centroid `2341826`: `COMPLETED 0:0`, saved latest tracker `319`.
- The automatically chained resume jobs failed before training:
  - vanilla `2341823`: `FAILED 15:0`, elapsed `00:01:13`.
  - retok `2341825`: `FAILED 15:0`, elapsed `00:01:18`.
  - centroid `2341827`: `FAILED 15:0`, elapsed `00:01:17`.
  - Exact cause: during `load_checkpoint()`, Megatron's torch_dist loader called `ckpt_metadata.mcore_data[...]`, but the PyTorch `Metadata` object in the just-written checkpoints has no `mcore_data` attribute. The error was:
    - `AttributeError: 'Metadata' object has no attribute 'mcore_data'. Did you mean: 'storage_data'?`
  - No training progress was lost in these failed resume jobs; they died during checkpoint load.
- Runtime fix:
  - Updated `megatron_patches/runtime/pretrain_gpt_te_guard.py` with `install_torch_dist_metadata_fallback()`.
  - The fallback catches only the missing-`mcore_data` `AttributeError`, reads PyTorch tensor storage metadata, and derives same-topology `TensorReformulationMetadata` entries for flattened tensors.
  - It logs a `RuntimeWarning` with the rank, checkpoint path, and number of derived entries. It still raises if tensor metadata is missing.
  - Local syntax check passed with `python3 -m py_compile`, and the updated wrapper was synced to the Clariden mirror.
- Manual resume2 relaunch:
  - vanilla `2345082`, from tracker `316`.
  - retok `2345083`, from tracker `317`.
  - centroid `2345084`, from tracker `319`.
  - All three successfully loaded checkpoints with the fallback active.
  - Verified post-resume training:
    - vanilla logged iterations `317` and `318`, `0` skipped / `0` NaN.
    - retok logged iterations `318` and `319`, `0` skipped / `0` NaN.
    - centroid logged iteration `320`, `0` skipped / `0` NaN.
  - The first resumed iterations are slightly slower due to checkpoint/load warmup; subsequent iterations returned to the expected ~390-410 TFLOP/s/GPU band.

## Continuation - 2026-05-22 eval packing correction

- Efficiency finding:
  - Training allocations are healthy: direct `nvidia-smi` sampling inside `2345082`/`2345083`/`2345084` showed all 12 training GPUs at `98-100%` utilization, with roughly `86-88 GiB / 97.9 GiB` HBM in use and logs around `409-411` TFLOP/s/GPU.
  - Full eval allocations were wasteful: each one-GPU `lm-eval` job was granted a whole 4-GPU node on Clariden's normal partition, with only GPU0 holding the model and the other three GPUs idle.
- Added a packed full-eval path:
  - `eval/run_eval_packed_arms.sbatch`: runs multiple single-GPU lm-eval arms concurrently inside one 4-GPU node allocation.
  - `eval/submit_bakeoff_checkpoint_eval_packed.sh`: submits the per-arm Megatron->HF conversions, then one packed dependent lm-eval job.
  - `eval/watch_and_submit_checkpoint_evals_packed.sh`: waits until all requested arms have complete checkpoints and matching trackers, then submits the packed chain once.
  - `eval/EVAL_RECIPE.md` now documents the packed path for future full checkpoints.
- Hardening added:
  - `submit_bakeoff_checkpoint_eval.sh` and `submit_bakeoff_checkpoint_eval_packed.sh` now fail before submission if `SUBMIT_INTRINSIC=1` but `EVAL_JSONL` is missing or empty.
  - The correct heldout path is `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl` (`500` docs, `33M`). Earlier example paths under `/capstor/.../data/bakeoff_eval/` were stale.
- Future watcher swap:
  - Old iter-390 and iter-455 per-arm watcher dirs were stamped with `vanilla.submitted`, `retok.submitted`, and `centroid.submitted` to prevent duplicate per-arm full eval submissions.
  - New packed watchers are running with a warm shared full-eval cache:
    - iter `390`, PID `243550`, state `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_watch_iter_0000390_full_packed`.
    - iter `455`, PID `243580`, state `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_watch_iter_0000455_full_packed`.
    - cache: `/iopsstor/scratch/cscs/fffoivos/tmp/eval_cache_bakeoff_full_shared` -> `/iopsstor/scratch/cscs/fffoivos/tmp/eval_cache_2344156`.
- Current training after the iter-325 checkpoint overhead returned to normal cadence:
  - vanilla iter `333`, `7969.8` tok/s/GPU, `0` skipped / `0` NaN.
  - retok iter `334`, `7912.8` tok/s/GPU, `0` skipped / `0` NaN.
  - centroid iter `335`, `7871.2` tok/s/GPU, `0` skipped / `0` NaN.
- Iter-325 full eval intervention:
  - Cancelled the three inefficient per-arm full-eval jobs after they spent ~15-18 minutes in CPU/dataset setup with only one GPU used per full-node allocation:
    - retok `2345267`: cancelled.
    - centroid `2345271`: cancelled.
    - vanilla `2345301`: cancelled.
  - First packed smoke `2345427` failed quickly because `uenv run` did not preserve the target-install `PYTHONPATH`; exact error in each arm log: `No module named lm_eval`.
  - Fixed `run_eval_packed_arms.sbatch` to use the same `uenv start --view=default --ignore-tty` pattern as the known-good single-arm eval script, exporting `PYTHONPATH` inside the uenv.
  - Second packed smoke `2345439` launched all three arms but hit Hugging Face API rate-limit backoff because each arm used a separate fresh cache and tried to load the same benchmark datasets concurrently.
  - Fixed the packed script to share one cache root across arms by default (`SHARE_EVAL_CACHE=1`) and to allow the warm full-eval cache to be injected via `EVAL_CACHE_ROOT`.
  - Current packed iter-325 retry `2345516` is running on one node (`nid007297`) with all three arm processes alive and model memory loaded on GPUs `0`, `1`, and `2`; no rate-limit warning has appeared after switching to the shared warm cache.
  - Added `flock` guards to the packed watcher and packed submitter. This protects the future iter-390/455 checkpoints from duplicate submissions even if earlier detached watcher attempts on different login nodes wake up at the same time.
  - Packed iter-325 validation at `2026-05-22T14:07:08Z`:
    - `2345516` is in `Running loglikelihood requests` for all three arms.
    - GPU sample inside the packed allocation: GPU0/GPU1/GPU2 at `100%` utilization with ~`85.8/92.9/92.9 GiB` HBM used; GPU3 is idle spare.
    - Progress sample: vanilla `2287/338117`, retok `10857/338117`, centroid `4303/338117` loglikelihood requests.
    - Training remained healthy at the same time: vanilla iter `344`, retok iter `345`, centroid iter `346`, all `0` skipped / `0` NaN.

## Continuation - 2026-05-22 iter-325 full eval complete

- Packed iter-325 full eval:
  - job `2345516` completed with Slurm exit `0:0`, elapsed `00:43:39`.
  - result files:
    - vanilla `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_vanilla/iter_0000325_full/results_2026-05-22T16-37-21.515595.json`
    - retok `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_retok/iter_0000325_full/results_2026-05-22T16-34-05.837089.json`
    - centroid `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_centroid/iter_0000325_full/results_2026-05-22T16-35-57.335820.json`
  - remote summary: `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_iter0000325_summary.md`.
  - local compact copies and digest were added under `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/`.
- Iter-325 headline:
  - vanilla: BPC `0.5045`, NLL/char `0.5968`, `el_arc=0.4121`, `el_belebele=0.5144`, `el_xnli=0.4028`, `el_xquad_f1=0.3193`, `el_mmlu=0.4240`, `el_base44=0.4275`, `el_piqa=0.6300`.
  - retok: BPC `0.6070`, NLL/char `0.7179`, `el_arc=0.3626`, `el_belebele=0.4767`, `el_xnli=0.3699`, `el_xquad_f1=0.3134`, `el_mmlu=0.3891`, `el_base44=0.3877`, `el_piqa=0.5600`.
  - centroid: BPC `0.9525`, NLL/char `1.1266`, `el_arc=0.2491`, `el_belebele=0.3233`, `el_xnli=0.3482`, `el_xquad_f1=0.0257`, `el_mmlu=0.2827`, `el_base44=0.3007`, `el_piqa=0.5100`.
  - New-token integration: retok `D1_top1=0.3183`, `D2_mass=0.3455`, `D4_top1_new=0.5612`, `D5_util=0.161`; centroid `D1_top1=0.0757`, `D2_mass=0.3409`, `D4_top1_new=0.2939`, `D5_util=0.068`.
  - Interpretation remains pre-decision: vanilla still leads Greek BPC and most Greek downstream metrics, retok continues narrowing and has the strongest new-token use, centroid remains weak on the Greek objective.

## Continuation - 2026-05-22 final 2B training complete

- CSCS auth was refreshed and verified with a fresh `cscs-key` certificate; `ssh clariden` worked again.
- GCP safety check could not be completed from `home`: `gcloud compute instances list` failed with non-interactive reauthentication required. No GCP resources were changed.
- Training completion:
  - vanilla `2345082`: `COMPLETED 0:0`, reached iteration `476/476`, saved `iter_0000476`, final loss `1.737108`, `0` skipped / `0` NaN.
  - retok `2345083`: `COMPLETED 0:0`, reached iteration `476/476`, saved `iter_0000476`, final loss `2.703995`, `0` skipped / `0` NaN.
  - centroid `2345084`: `COMPLETED 0:0`, reached iteration `476/476`, saved `iter_0000476`, final loss `3.787457`, `0` skipped / `0` NaN.
  - All checkpoint trackers now report `476`.
  - Local refreshed training artifacts:
    - `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_training_curve.csv`
    - `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_training_summary.json`
    - `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_training_summary.md`
- Iter-390 full packed eval:
  - job `2346267` completed `0:0`; compact local copies and digest were added under `eval/live_summaries/`.
- Iter-455 full packed eval:
  - job `2346980` completed `0:0`, elapsed `00:52:28`.
  - compact local copies and digest were added under `eval/live_summaries/`.
  - headline at iter 455:
    - vanilla: BPC `0.4916`, NLL/char `0.5816`, `el_arc=0.4113`, `el_belebele=0.5100`, `el_xnli=0.3988`, `el_xquad_f1=0.3059`, `el_mmlu=0.4231`, `hellaswag=0.7592`, `mmlu=0.5315`.
    - retok: BPC `0.5768`, NLL/char `0.6822`, `el_arc=0.3669`, `el_belebele=0.4844`, `el_xnli=0.3707`, `el_xquad_f1=0.3164`, `el_mmlu=0.3967`, `hellaswag=0.7479`, `mmlu=0.5560`, `D5_util=0.3160`.
    - centroid: BPC `0.9045`, NLL/char `1.0698`, `el_arc=0.2594`, `el_belebele=0.3344`, `el_xnli=0.3574`, `el_xquad_f1=0.0271`, `el_mmlu=0.2804`, `hellaswag=0.7578`, `mmlu=0.5450`, `D5_util=0.0860`.
- Final iter-476 eval chain:
  - Submitted with separate cache root `/iopsstor/scratch/cscs/fffoivos/tmp/eval_cache_bakeoff_full_iter0000476_packed` to avoid colliding with the still-running iter-455 eval.
  - Conversion jobs `2347091`, `2347094`, `2347097` completed `0:0`.
  - Intrinsic jobs `2347092`-`2347099` completed `0:0`.
  - Packed final downstream eval `2347100` launched on `nid007218`, using GPU0/GPU1/GPU2 for vanilla/retok/centroid and GPU3 as spare.
  - Packed final downstream eval `2347100` completed `0:0`, elapsed `00:51:10`.
  - compact local copies and digest were added under `eval/live_summaries/`.
  - headline at final iter 476:
    - vanilla: BPC `0.4906`, NLL/char `0.5804`, `el_arc=0.4206`, `el_belebele=0.5133`, `el_xnli=0.4020`, `el_xquad_f1=0.3101`, `el_mmlu=0.4214`, `hellaswag=0.7594`, `mmlu=0.5340`.
    - retok: BPC `0.5739`, NLL/char `0.6788`, `el_arc=0.3720`, `el_belebele=0.4967`, `el_xnli=0.3751`, `el_xquad_f1=0.3092`, `el_mmlu=0.3991`, `hellaswag=0.7488`, `mmlu=0.5542`, `D5_util=0.3580`.
    - centroid: BPC `0.8994`, NLL/char `1.0638`, `el_arc=0.2560`, `el_belebele=0.3411`, `el_xnli=0.3538`, `el_xquad_f1=0.0258`, `el_mmlu=0.2794`, `hellaswag=0.7599`, `mmlu=0.5444`, `D5_util=0.0920`.

## Continuation - 2026-05-22 CPU-only Slurm allocation correction

- User correction accepted: the problem was not accounting terminology, it was that CPU-only work had been submitted to GPU-allocating Clariden partitions.
- `sacct` audit over 2026-05-21..2026-05-23 found `157.45` allocated GPU-hours on CPU/conversion-style jobs:
  - `mix_builder`: `131.04`
  - `prepare_greek_pool`: `11.00`
  - `build_cpt_heldout`: `4.44`
  - `normalize_nfc`: `4.23`
  - `preprocess_data`: `3.29`
  - `checkpoint_conversion`: `3.19`
  - `build_init_ckpts`: `0.21`
  - `concat`: `0.06`
- Root cause: `sinfo` shows Clariden `normal`, `debug`, and `low` all have `gpu:4`; only `xfer` is visible as non-GPU. Therefore even CPU sbatches without explicit `--gpus` burned GPU nodes when `#SBATCH --partition=normal` was used.
- Corrective patch:
  - Added `init_bakeoff/slurm_cpu_only_guard.sh`; CPU-only sbatches now exit before doing work if they are on a non-`xfer` partition or Slurm assigns GPU GRES, unless `ALLOW_GPU_NODE_FOR_CPU=1` is set intentionally.
  - Moved CPU-only sbatches to `#SBATCH --partition=xfer`: corpus prep/build/concat, Megatron preprocessing, heldout build, init-checkpoint build/conversion, and bakeoff checkpoint-to-HF conversion.
  - Removed explicit GPU requests from CPU-only checkpoint init/conversion scripts.
  - Added `DUCKDB_MEMORY_LIMIT`, `DUCKDB_TEMP_DIRECTORY`, and `DUCKDB_THREADS` support in `glossapi_corpus_cli.pipeline._duckdb_connect_streaming`, and set the prepare job to spill DuckDB temp files under scratch with a bounded memory cap for xfer.
  - Updated docs/runbooks so CPU-only work is no longer described as running on `normal`/`debug`.
- Current `squeue -u fffoivos` was empty when this correction was made; no live Slurm jobs were cancelled.

## Continuation - 2026-05-22 future dataset-build guard

- Added `03_4_implementation_experiments/init_bakeoff/check_cpu_only_slurm.sh`.
- This script is the pre-submit audit for future dataset building and CPU-only checkpoint conversion work. It verifies the known CPU-only sbatches use `#SBATCH --partition=xfer`, contain no GPU directives, and call `require_cpu_only_slurm` before doing work.
- Updated the init-bakeoff runbook to make this audit the first step before any future dataset build or CPU-only conversion submit.

## Continuation - 2026-05-24 production CPT launcher

- Selected path is now concrete: Vanilla Apertus-8B-2509 with the base tokenizer
  remains the production default after the 2B bakeoff and bounded TD challenger.
- Prepared the production launcher under
  `03_4_implementation_experiments/init_bakeoff/production_cpt/`.
  It reuses the proven bakeoff trainer but explicitly exports:
  - `ARM=vanilla`
  - `LOSS_OBJECTIVE=goldfish`
  - `BASE_DATA_PREFIX=/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document`
  - `TRAIN_TOKENS=15000000000`
  - `SAVE_INTERVAL=120`
  - `LR_WARMUP_TOKENS=300000000`
  - `ADEMA_BETA3_WARMUP_STEPS=101`
  - `ADEMA_ALPHA_WARMUP_STEPS=101`
- Kept the launcher on the proven one-node, four-GH200 path. The previous
  two-node smoke failed before iteration 1 with NCCL/OFI `NO_SPACE`, so the
  production script refuses `NODES != 1`.
- Made `_train_config_common.env` override-safe for production values while
  preserving bakeoff defaults, and made `bakeoff_train.sbatch` switch to
  Goldfish only when `LOSS_OBJECTIVE=goldfish`.
- Clariden dry-run validation:
  - `dryrun_default_vanilla_base_15b_nfc_20260524T121007`
  - generated a default 14-job chain;
  - dependencies defaulted to `afterok`;
  - no Slurm jobs were submitted;
  - `squeue -u fffoivos` was empty after validation.
- Local audit copies:
  `03_4_implementation_experiments/init_bakeoff/production_cpt/dryrun_default_vanilla_base_15b_nfc_20260524T121007/`.
- Anneal remains explicitly out of this launcher. The current anneal recipe is a
  design artifact and must be rebuilt on `xfer` from the selected post-dedup
  Greek parquet plus local staged replay/code/math sources before it can become
  a second production phase.

## Continuation - 2026-05-24 3.5B bakeoff continuation plan

- Added dry-run-first continuation submitter:
  `03_4_implementation_experiments/init_bakeoff/bakeoff_training/submit_3p5b_continuation_chain.sh`.
- Added eval sidecar submitter:
  `03_4_implementation_experiments/init_bakeoff/eval/submit_3p5b_eval_sidecars.sh`.
- Scope:
  - continue Vanilla, ReTok, and TD layer11 from iter `476` to iter `834`
    (`3.498B` total tokens);
  - split each arm into chained segments ending at iter `585`, `715`, and
    `834`;
  - keep the three arms parallel;
  - submit eval sidecars at each boundary without making later training depend
    on eval.
- Invariants forced by the submitter:
  - `LOSS_OBJECTIVE=ntp`;
  - base data prefix:
    `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_megatron/bulk_mix_text_document`;
  - extended data prefix:
    `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_ext_megatron/bulk_mix_text_document`;
  - one node, four GH200 GPUs, TP=2, same bakeoff trainer;
  - eval jobs use `--nice=1000` by default.
- Clariden dry-run validation:
  - `continuation_3p5b_dryrun_20260524T020000Z`;
  - generated `9` training commands and `27` eval-sidecar commands;
  - source iter-476 checkpoints and both original bakeoff Megatron data prefixes
    existed;
  - every training command exported `LOSS_OBJECTIVE=ntp` and the expected
    bakeoff data prefixes;
  - eval conversion commands depend on the checkpoint-producing training
    segment; later training segments do not depend on eval.
- Representative non-submitting Slurm checks passed:
  - training `sbatch --test-only`;
  - checkpoint conversion `sbatch --test-only`;
  - packed eval `sbatch --test-only`;
  - follow-up `squeue` check showed those test-only IDs were not queued.
- Local audit copies:
  `03_4_implementation_experiments/init_bakeoff/bakeoff_training/dryrun_3p5b_continuation_20260524T020000Z/`.
- No live GPU training or eval jobs were submitted in this step. Live launch
  still requires:
  `DRY_RUN=0 CONFIRM_3P5B_LAUNCH=1 RUN_TAG=<real-run-tag> bash submit_3p5b_continuation_chain.sh`.

### 2026-05-24 3.5B continuation launch-readiness audit

- Re-audited the committed continuation scripts before launch.
- Fresh Clariden dry-run tag:
  `continuation_3p5b_audit_20260524T134014Z`.
- Dry-run produced the expected shape:
  - `9` training sbatch commands;
  - `27` eval-sidecar sbatch commands;
  - `9` rows in `training_chain.tsv` plus header;
  - `9` rows in `eval_sidecar_jobs.tsv` plus header.
- Verified remote prerequisites:
  - Vanilla, ReTok, and TD source checkpoint roots all have
    `latest_checkpointed_iteration.txt = 476` and `iter_0000476/`;
  - original bakeoff base/extended Megatron `.bin` and `.idx` files exist;
  - held-out Greek intrinsic eval JSONL exists with `500` rows;
  - full HF tokenizer/model dirs exist for conversion; the extended training
    tokenizer path is tokenizer-only as expected for Megatron training.
- Verified checkpoint cadence against the live Megatron code:
  - iter `585` and `715` are regular `SAVE_INTERVAL=65` checkpoints;
  - iter `834` is saved by Megatron's end-of-training final-save path because
    it is not divisible by `65`.
- Representative non-submitting Slurm parse checks passed for:
  - training;
  - checkpoint conversion;
  - packed downstream eval;
  - tokenizer-fair BPC metrics;
  - new-token diagnostics.
- Follow-up `squeue` checks confirmed the `--test-only` IDs were not queued,
  and `squeue -u fffoivos` was empty.
- GCP cost-safety check could not report current state because local `gcloud`
  requires interactive reauthentication; no GCP resources were touched.
- No live GPU training or eval jobs were submitted in this audit.

### 2026-05-24T13:44Z 3.5B launch gate held

- Automatic goal continuation resumed after the launch-readiness audit, but no
  explicit live-run go/no-go was given.
- Rechecked the Clariden mirror:
  - local and remote SHA-256 hashes match for both 3.5B submitters and this
    takeover log;
  - `squeue -u fffoivos` was empty.
- Rechecked remote launch inputs:
  - Vanilla, ReTok, and TD source checkpoint roots still report latest
    iteration `476` and contain `iter_0000476/`;
  - original bakeoff base/extended Megatron `.bin` and `.idx` files still
    exist;
  - held-out Greek eval JSONL still has `500` rows.
- Launch remains intentionally gated on an explicit live-submit command:
  `DRY_RUN=0 CONFIRM_3P5B_LAUNCH=1 RUN_TAG=<real-run-tag> bash submit_3p5b_continuation_chain.sh`.

### 2026-05-24T14:30Z 3.5B continuation live launch

- User gave explicit live-run go-ahead.
- Pre-launch `squeue -u fffoivos` was empty.
- GCP cost-safety check still could not report state because `gcloud` requires
  interactive reauthentication; no GCP resources were touched.
- Launched from Clariden with:
  `DRY_RUN=0 CONFIRM_3P5B_LAUNCH=1 RUN_TAG=continuation_3p5b_20260524T143012Z bash submit_3p5b_continuation_chain.sh`.
- Training chain submitted successfully:
  - Vanilla: `2369298` -> `2369299` -> `2369300`;
  - ReTok: `2369301` -> `2369302` -> `2369303`;
  - TD layer11: `2369304` -> `2369305` -> `2369306`.
- The eval sidecar submitter hit Clariden's submitted-job limit
  (`QOSMaxSubmitJobPerUserLimit`) after submitting the first five iter-585
  sidecars:
  - `2369307` `tohf_vanilla_585`;
  - `2369308` `bpc_vanilla_585`;
  - `2369309` `tohf_retok_585`;
  - `2369310` `bpc_retok_585`;
  - `2369311` `diag_retok_585`.
- Added `submit_3p5b_eval_sidecars_incremental.py` to submit the remaining
  eval DAG as queue slots open, without making training depend on eval.
- Started incremental eval submitter on Clariden:
  - PID: `175635`;
  - state dir:
    `/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_sidecar_eval_incremental`;
  - first state: `submitted=5 missing=22 active_jobs=14`.
- Clariden did not keep the detached login-node Python loop alive. Replaced it
  with a home-side transient systemd watcher:
  - unit: `apertus-3p5b-eval-watch.service`;
  - local log:
    `03_4_implementation_experiments/init_bakeoff/eval/continuation_3p5b_20260524T143012Z_home_watcher/watcher.log`;
  - behavior: every two minutes, SSH to Clariden and run one incremental
    submit pass until all `27` sidecar tasks have job IDs.
- Startup health:
  - all three first-segment jobs reached `training ...`;
  - Vanilla/ReTok/TD all loaded checkpoint iteration `476`;
  - all three completed iteration `477` without OOM, skipped iterations, or
    NaNs;
  - early per-GPU throughput was roughly `7.5k-7.7k` tokens/s.

### 2026-05-24T14:58Z 3.5B early-stability check

- Foreground babysitting reached the intended early resumed-optimizer window:
  - Vanilla reached iteration `487`;
  - ReTok reached iteration `487`;
  - TD layer11 reached iteration `486`.
- Health:
  - all three arms still running;
  - `loss scale = 1.0`;
  - `number of skipped iterations = 0`;
  - `number of nan iterations = 0`;
  - per-GPU throughput stable at roughly `7.9k-8.1k` tokens/s after warmup.
- Observed isolated grad-norm spikes:
  - Vanilla: `17.784` at iter `481`, then back near `1`;
  - ReTok: `7.051` at iter `482`, then back near `1-2`;
  - TD: no comparable early spike; latest observed grad norm `2.450`.
- No intervention taken. Continue babysitting until the first segment reaches
  iter `585`, writes checkpoints, and the eval sidecars begin to drain.

### 2026-05-24T15:31Z 3.5B one-hour segment-1 check

- Queue:
  - `2369298` Vanilla, `2369301` ReTok, and `2369304` TD layer11 still
    running;
  - chained segment-2/segment-3 jobs still pending on their dependencies;
  - first five iter-585 eval sidecars still pending on checkpoint dependencies.
- Progress:
  - Vanilla reached iteration `502`;
  - ReTok reached iteration `502`;
  - TD layer11 reached iteration `501`.
- Health remains good:
  - `loss scale = 1.0`;
  - `number of skipped iterations = 0`;
  - `number of nan iterations = 0`;
  - per-GPU throughput remains stable at roughly `7.9k-8.1k` tokens/s.
- Latest observed losses:
  - Vanilla iter `502`: `1.705135`;
  - ReTok iter `502`: `2.684626`;
  - TD layer11 iter `501`: `2.474802`.
- Eval watcher still active, but submitted-job count remains at `14`, so it
  cannot add the remaining sidecars until jobs leave the queue.
