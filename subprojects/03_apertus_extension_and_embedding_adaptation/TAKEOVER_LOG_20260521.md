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
