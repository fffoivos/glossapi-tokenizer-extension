# Greek Apertus 13.5B Two-Arm CPT Run Log

Started: 2026-06-09

Scope: build the documented 13.5B two-arm CPT dataset, prepare the documented
vanilla and TD init checkpoints, launch both experiments, and monitor to
completion. The written docs are the control surface: dataset recipe from
`03_training_experiments/dataset_build/bulk_13b.json` and handoffs,
hyperparameters from `03_training_experiments/configs/common_cpt.env` and the
hyperparameter docs, checkpoint geometry from `HANDOFF.md` and
`apertus_pretrain_checkpoints_notes.md`.

## Operating Principle

- Do not change the dataset recipe, hyperparameters, tokenizer size, checkpoint
  geometry, or training objective while debugging infrastructure.
- Repairs made during launch are packaging/guardrail repairs only: missing
  helper files, missing tokenizer metadata, and output validation gates.
- Any job that exits 0 but fails an artifact gate is treated as failed evidence.

## 2026-06-09 Progress

- Read the handoff/runbook material and used it as the launch contract.
- Verified Clariden access and the documented build venv.
- Synced the required local scripts/configs to Clariden.
- Patched the active Megatron fork for named extra validation sets and patched
  `bakeoff_train.sbatch` to pass the per-arm held-out validation binaries.
- Submitted the CPU-only dataset chain:
  - holdout validation build: `2509366`
  - mix array: `2509367` (`0-7%4`)
  - Stage A clean/decontam: `2509368`
  - original Stage B anonymize/preprocess: `2509369`
  - original val tokenization: `2509370`
- Holdout build `2509366` completed successfully. Produced:
  - `val_hplt.jsonl`
  - `val_openarchives.jsonl`
  - `val_greek_phd.jsonl`
  - `val_holdout_ids.parquet` with HPLT/openarchives IDs held out from train.
- Initial validation tokenization `2509370` failed before work because
  `tools/preprocess_data.py` was missing from the active Megatron checkout.
- Restored `tools/preprocess_data.py` from the pinned Swiss-AI reference repo.
- Submitted validation-tokenization retry `2509414`.
- `2509414` produced valid base-tokenizer validation binaries, but the extended
  tokenizer outputs were zero-byte `.bin` files with no `.idx`. Logs showed
  `NoneType` token IDs during `--append-eod`.
- Root cause: configured extended-tokenizer directory
  `/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480`
  had only `tokenizer.json`; it lacked `tokenizer_config.json` special-token
  metadata. This does not alter the vocab/ID map, but it breaks EOD appending.
- Copied `tokenizer_config.json` from the verified TD HF artifact into the
  configured extended-tokenizer directory.
- Hardened `tokenize_vals.sbatch`:
  - added `TOKENIZE_BASE` / `TOKENIZE_EXT` switches for targeted reruns;
  - removes stale outputs before tokenizing;
  - requires non-empty `.bin` and `.idx` outputs.
- Hardened `stageB_anon_preprocess.sbatch` with non-empty output gates for
  both base and extended full-corpus tokenizations.
- Canceled original pending Stage B job `2509369` because Slurm had captured
  the old script body before hardening. Resubmitted hardened Stage B as `2509486`
  with dependency `afterok:2509368`.
- Submitted ext-only validation-tokenization retry `2509508`.
- Ext-only validation-tokenization retry `2509508` completed successfully.
  All three documented held-out validation sets now have non-empty base and
  extended Megatron binaries:
  - `val_hplt_{base,ext}_text_document.{bin,idx}`
  - `val_openarchives_{base,ext}_text_document.{bin,idx}`
  - `val_greek_phd_{base,ext}_text_document.{bin,idx}`

## Init Checkpoints

- Staged geometry-reverted HF views for both arms:
  - `rope_theta`: `12000000 -> 500000`
  - `max_position_embeddings`: `65536 -> 4096`
  - preserved documented llama3 rope scaling.
- Confirmed TD tokenizer parsed vocab map matches the configured extended
  tokenizer; then pointed the staged TD HF view at the configured tokenizer
  artifact for clean provenance.
- Initial init conversions failed because checkpoint conversion plugins were
  incomplete in the active Megatron checkout.
- Made `megatron_patches/install.sh` self-contained by installing:
  - `loader_apertus_hf.py`
  - `saver_core.py`
  - `saver_swissai_hf.py`
  - `schema_core.py`
  - `schema_base.py`
  - `utils.py`
- TD init conversion/verification completed successfully as job `2509418`.
  Verification JSON:
  `/capstor/scratch/cscs/fffoivos/runs/init_td_l11_r17_cpt2arm_retry_20260609T194143Z/verification.json`
- Vanilla init conversion/verification completed successfully as job `2509428`.
  Verification JSON:
  `/capstor/scratch/cscs/fffoivos/runs/init_vanilla_r17_cpt2arm_retry2_20260609T194520Z/verification.json`
- Both verification files report zero standard diff, zero R17 diff, zero xIELU
  diff, zero QK-norm diff, no shape mismatches, and matching logits.

## Current State To Monitor

- Mix array `2509367`: shards `0-3` running; shards `4-7` pending on array
  limit.
- Stage A `2509368`: pending on mix array completion.
- Hardened Stage B `2509486`: pending on Stage A.
- Ext-only validation tokenization `2509508`: completed; all three
  `val_*_ext_text_document.{bin,idx}` files are non-empty.
- Before training launch, run an explicit docs-alignment audit:
  - dataset recipe and produced mix match the written recipe;
  - base/ext tokenized dataset prefixes exist and are non-empty;
  - held-out validation prefixes exist for each arm;
  - arm configs point at the verified init checkpoints;
  - `common_cpt.env` hyperparameters match the documented regime;
  - no old sweep/env variables override the documented config.

## Docs-Alignment Audit Notes

- `common_cpt.env` on Clariden currently resolves to the documented regime:
  - `TRAIN_TOKENS=13500000000`
  - `TRAIN_ITERS=3218`
  - `GLOBAL_BATCH_TOKENS=4194304`
  - `LR_PEAK=5.5e-5`, `LR_FINAL=5.5e-6`
  - `LR_WARMUP_ITERS=400`, `LR_WARMUP_TOKENS=1677721600`
  - `LR_WSD_DECAY_SAMPLES=659179`
  - `ADEMA_BETA2=0.995`, `ADEMA_BETA3=0.999`, `ADEMA_ALPHA=4.0`
  - `ADEMA_BETA3_WARMUP_STEPS=3218`, `ADEMA_ALPHA_WARMUP_STEPS=3218`
  - `SEQ_LENGTH=4096`, `ROTARY_BASE=500000`, `USE_ROPE_SCALING=1`
  - `MAKE_VOCAB_SIZE_DIVISIBLE_BY=256`
  - `LOSS_OBJECTIVE=goldfish`, `GOLDFISH_K=50`, `GOLDFISH_H=50`
  - `DATA_SEED=20260609`
- Tokenizer invariants on Clariden:
  - base vocab = `131072`, divisible by 256;
  - extended vocab = `148480`, divisible by 256;
  - both now have `</s>` id `2`, `<s>` id `1`, and `<pad>` id `3`.
- `bulk_13b.json` reports recipe `bulk_13b`, version
  `v2.0_70-30_holdout`. Bucket sums from rounded weights are:
  - Greek `0.740741`
  - multilingual replay `0.17776318`
  - code `0.02963`
  - math `0.014815`
  - Greek replay `0.037037`
  - total `0.99998618` due rounded per-source weights.
- The active mix jobs print the intended bucket targets from the recipe:
  Greek `0.7407`, replay `0.1778`, code `0.0296`, math `0.0148`,
  Greek replay `0.0370`.
- The JSON recipe contains an older internal `seed=20260520`, but the actual
  mix jobs are launched with the documented `DATA_SEED=20260609` plus shard
  offset (`20260609 + shard_id`) and print that seed in the logs.

## 2026-06-09T20:01:09Z - Logging Discipline And Current State

- User requested that work be logged continuously under this subproject so it
  can be reviewed later. This file is now the active operations log for the
  dataset build, checkpoint preparation, prelaunch gating, launch, and
  monitoring steps.
- Current observed Slurm state:
  - mix array `2509367`: shards `0-3` running for about 25 minutes; shards
    `4-7` pending on the `%4` array limit.
  - Stage A clean/decontam `2509368`: pending on mix array completion.
  - hardened Stage B anonymize/preprocess `2509486`: pending on Stage A.
- Current validation-tokenization state:
  - ext-only retry `2509508` completed before this checkpoint;
  - base and extended held-out validation binaries were listed on Clariden and
    are present/non-empty for `val_hplt`, `val_openarchives`, and
    `val_greek_phd`.
- Next local action: finish and sync an explicit prelaunch artifact gate script
  so the training launch depends on concrete checks rather than informal visual
  inspection.

## 2026-06-09T20:04:14Z - Prelaunch Gate And Submitter Hardening

- Added and synced:
  - `03_training_experiments/scripts/gate_cpt2arm_artifacts.sh`
  - updated `03_training_experiments/scripts/submit_two_arm_full_run.sh`
- Gate script syntax passed locally and remotely.
- First gate run found a gate-script shell bug: section-header `printf` calls
  began with `---`, which Bash treated as an option on Clariden. Fixed by
  printing headers through `printf '%s\n'`.
- Second gate run exposed an audit-resolution issue: sequentially sourcing
  `arm1_vanilla.env` then `arm2_modern_greek.env` kept the first arm's
  defaulted `INIT_CKPT`. Fixed the gate to resolve each arm config in a clean
  child shell with checkpoint/data/tokenizer overrides unset.
- Hardened `submit_two_arm_full_run.sh` so live submission refuses ambient
  `INIT_CKPT`, `BASE_DATA_PREFIX`, `EXT_DATA_PREFIX`, `BASE_TOKENIZER_DIR`,
  `EXT_TOKENIZER_DIR`, and key HP variables unless `ALLOW_OVERRIDES=1`.
  Rationale: for this production run, the written config files should be the
  source of truth and stale shell state must not silently override them.
- Clariden dry-run with watchers disabled confirms distinct init checkpoints:
  - vanilla segment 1:
    `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/cpt_2arm_13b/vanilla_base131072/megatron_tp2_r17patched`
  - TD segment 1:
    `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/cpt_2arm_13b/modern_greek_td148480/megatron_tp2_r17patched`
- Corrected artifact gate baseline:
  - PASS: documented HP/config invariants.
  - PASS: base tokenizer `131072` and extended tokenizer `148480`, both
    divisible by 256, with matching bos/eos/pad IDs.
  - PASS: vanilla and TD init checkpoint rank files and `latest=release`.
  - PASS: all base/ext held-out validation binaries for HPLT, OpenArchives,
    and Greek PhD.
  - EXPECTED FAIL: `bulk_mix_final.jsonl` plus base/ext full-training
    `.bin/.idx` files are still absent because Stage A/B have not run yet.

## 2026-06-09T20:04:51Z - Mix Array Progress Snapshot

- Slurm state:
  - `2509367_0`-`2509367_3`: running for ~29 minutes.
  - `2509367_[4-7%4]`: pending on `JobArrayTaskLimit`.
  - `2509368`: pending on mix completion.
  - `2509486`: pending on Stage A completion.
- Mix progress from shard logs:
  - shard 0: `550,277,611` tokens, `32.6%`, ETA `54.3 min`.
  - shard 1: `550,178,573` tokens, `32.6%`, ETA `55.1 min`.
  - shard 2: `550,051,626` tokens, `32.6%`, ETA `56.9 min`.
  - shard 3: `600,048,029` tokens, `35.6%`, ETA `49.1 min`.
- Current part-file sizes:
  - `bulk_mix_part_00.jsonl`: `3.6G`
  - `bulk_mix_part_01.jsonl`: `3.6G`
  - `bulk_mix_part_02.jsonl`: `3.5G`
  - `bulk_mix_part_03.jsonl`: `3.9G`

## 2026-06-09T20:06:38Z - Stage A/B Runtime Verification

- Reviewed the submitted Stage A/B scripts against the written chain:
  `mix -> HPLT E001 clean -> GreekMMLU correct_only decontam -> anonymize ->
  base/ext tokenization`.
- Stage A currently runs without activating the `cpt_build_py312` venv. Checked
  the actual Stage A Python payloads:
  - `dataset_build/hplt_clean.py` is stdlib-only.
  - `02_corpus_preparation/30_decontaminate/scripts/decontaminate.py` states
    and uses stdlib-only logic.
  - `python3 -m py_compile` for both scripts passed under the exact Clariden
    `uenv start pytorch/v2.9.1:v2 --view=default` environment.
  - Conclusion: no Stage A resubmit is needed for Python dependencies.
- Plain `uenv` Python does not have `pyarrow`; this is acceptable for Stage A
  but reinforces why Stage B must activate `cpt_build_py312`.
- Stage B runtime probe under `uenv` plus
  `/iopsstor/scratch/cscs/fffoivos/python_envs/cpt_build_py312` passed:
  - imports: `datatrove`, `transformers`, `tokenizers`;
  - compile: anonymizer script and active Megatron `tools/preprocess_data.py`.
- Conclusion: the currently pending hardened Stage B `2509486` has the required
  runtime dependencies for anonymization and both tokenization passes.

## 2026-06-09T20:09:48Z - Mix Array Progress Snapshot

- Slurm state unchanged:
  - `2509367_0`-`2509367_3`: running for ~34 minutes.
  - `2509367_[4-7%4]`: pending on `JobArrayTaskLimit`.
  - `2509368` and `2509486`: dependency-pending.
- Mix progress:
  - shard 0: `650,281,104` tokens, `38.5%`, ETA `49.4 min`.
  - shard 1: `650,183,467` tokens, `38.5%`, ETA `50.0 min`.
  - shard 2: `650,054,796` tokens, `38.5%`, ETA `51.5 min`.
  - shard 3: `700,078,071` tokens, `41.5%`, ETA `44.7 min`.
- Current part-file sizes:
  - `bulk_mix_part_00.jsonl`: `4.3G`
  - `bulk_mix_part_01.jsonl`: `4.3G`
  - `bulk_mix_part_02.jsonl`: `4.1G`
  - `bulk_mix_part_03.jsonl`: `4.5G`

## 2026-06-09T20:10:11Z - Remote Decontam Script Check

- Verified that the Clariden copy of
  `02_corpus_preparation/30_decontaminate/scripts/decontaminate.py` contains
  the `normalize_doc_id` fix in scan, worker, and filter-output paths.
- This closes the known Stage A risk where contaminated IDs could be generated
  under one fallback convention and filtered under another.

## 2026-06-09T20:10:37Z - Slurm Dependency Check

- `scontrol show job 2509368`:
  - `JobName=cpt13b_A_clean_decontam`
  - `JobState=PENDING`
  - `Dependency=afterok:2509367_*`
  - CPU-only: 1 node, 64 CPUs, 200G RAM, no GPU TRES.
- `scontrol show job 2509486`:
  - `JobName=cpt13b_B_anon_preprocess`
  - `JobState=PENDING`
  - `Dependency=afterok:2509368`
  - CPU-only: 1 node, 64 CPUs, 240G RAM, no GPU TRES.
- `sacct` confirms old Stage B `2509369` is `CANCELLED`; the live downstream
  path uses hardened Stage B `2509486`.

## 2026-06-09T20:11:02Z - Stored Stage B Script Check

- Used `scontrol write batch_script 2509486` to inspect the stored Slurm batch
  script, not just the current repo file.
- Confirmed the queued Stage B contains the hardening that matters:
  - removes stale `$STAGE/decontam_indir/part_*.jsonl`;
  - asserts non-empty
    `bulk_mix_base_text_document.{bin,idx}`;
  - asserts non-empty
    `bulk_mix_ext_text_document.{bin,idx}`.

## 2026-06-09T20:11:32Z - Watcher Partition Check

- Checked `sinfo -p xfer,normal`.
- `xfer` is currently `up`, with one `idle` node and one mixed node.
- This means the documented watcher default `WATCHER_PARTITION=xfer` is viable
  right now. Recheck immediately before live training launch.

## 2026-06-09T20:12:15Z - Watcher/Sidecar Readiness Check

- Reviewed the two watcher paths that `launch_all.sh` will use:
  - vanilla:
    `subprojects/04_cpt_training_regime_on_vanilla/scripts/watch_and_submit_checkpoint_sidecars.sbatch`
  - TD:
    `subprojects/05_token_distillation_cpt/scripts/watch_and_submit_td_checkpoint_sidecars.sbatch`
- Syntax passed locally and on Clariden for both watcher scripts and both
  submit-sidecar scripts.
- Required code/math heldout files exist and are non-empty on Clariden:
  - `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_code_heldout_200_20260528.jsonl`
  - `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_math_heldout_200_20260528.jsonl`
- Notes:
  - TD watcher forwards `HF_TOKENIZER_DIR`; `launch_all.sh` passes the modern
    Greek tokenizer path.
  - Vanilla watcher relies on the base tokenizer default in
    `submit_checkpoint_sidecars.sh`, which matches the vanilla arm.

## 2026-06-09T20:12:49Z - Full Launch Dry-Run

- Ran on Clariden from
  `03_training_experiments`:
  `DRY_RUN=1 SUBMIT_WATCHERS=1 bash scripts/launch_all.sh`.
- Dry-run confirms:
  - total iters `3218`;
  - global-batch tokens `4194304`;
  - watcher partition `xfer`;
  - 14 benchmark checkpoints per arm (`238`-iter cadence plus final).
- Vanilla dry-run:
  - config `arm1_vanilla.env`;
  - init checkpoint
    `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/cpt_2arm_13b/vanilla_base131072/megatron_tp2_r17patched`;
  - watcher command exports `RUN_ROOT`, `TRAIN_RUN_DIR`,
    `CHECKPOINTS_FILE`, base `HF_TOKENIZER_DIR`, and `EVAL_ARM=vanilla`.
- TD dry-run:
  - config `arm2_modern_greek.env`;
  - init checkpoint
    `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/cpt_2arm_13b/modern_greek_td148480/megatron_tp2_r17patched`;
  - watcher command exports `RUN_ROOT`, `TRAIN_RUN_DIR`,
    `CHECKPOINTS_FILE`, extended `HF_TOKENIZER_DIR`, and `EVAL_ARM=td`.
- No launch performed. Live launch remains gated on Stage B outputs and the
  final artifact gate.

## 2026-06-09T20:15:43Z - Mix Array Progress Snapshot

- `collect_metrics.py` compiles locally and on Clariden; ready for post-launch
  log parsing once training starts.
- Slurm state unchanged:
  - `2509367_0`-`2509367_3`: running for ~40 minutes.
  - `2509367_[4-7%4]`: pending on `JobArrayTaskLimit`.
  - `2509368` and `2509486`: dependency-pending.
- Mix progress:
  - shard 0: `800,294,129` tokens, `47.4%`, ETA `42.2 min`.
  - shard 1: `800,186,619` tokens, `47.4%`, ETA `42.5 min`.
  - shard 2: `750,055,754` tokens, `44.4%`, ETA `46.6 min`.
  - shard 3: `850,083,135` tokens, `50.4%`, ETA `37.6 min`.
- Current part-file sizes:
  - `bulk_mix_part_00.jsonl`: `5.1G`
  - `bulk_mix_part_01.jsonl`: `5.0G`
  - `bulk_mix_part_02.jsonl`: `4.9G`
  - `bulk_mix_part_03.jsonl`: `5.4G`

## 2026-06-09T20:16:16Z - Recipe Source-Bucket Check

- Checked local and Clariden `dataset_build/bulk_13b.json`; both report:
  - `name=bulk_13b`
  - `version=v2.0_70-30_holdout`
- The submitted unseen-Greek bucket is:
  - `greek_hplt_70`, weight `0.5185187` of total;
  - `greek_openarchives_30`, weight `0.2222223` of total.
- Replay buckets:
  - multilingual replay total from tiered replay sources;
  - `code_codeparrot_clean`, weight `0.02963`;
  - `math_finemath`, weight `0.014815`;
  - `greek_replay_apertus_original`, weight `0.037037`.
- Active docs agree with this recipe:
  `10B unseen = 70% HPLT + 30% openarchives + replay -> 13.5B`.

## 2026-06-09T20:16:45Z - Storage Headroom Check

- `df -h` on Clariden:
  - `/iopsstor/scratch/cscs`: `673T` available.
  - `/capstor/scratch/cscs`: `137T` available.
- Current stage usage:
  - `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt_2arm_13b`: `45G`.
  - `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b`: `412K`.
- Conclusion: enough storage headroom for Stage A/B intermediates and both
  tokenized full-corpus binaries.

## 2026-06-09T20:21:41Z - Mix Array Progress Snapshot

- Slurm state unchanged:
  - `2509367_0`-`2509367_3`: running for ~46 minutes.
  - `2509367_[4-7%4]`: pending on `JobArrayTaskLimit`.
  - `2509368` and `2509486`: dependency-pending.
- Mix progress:
  - shard 0: `950,330,506` tokens, `56.3%`, ETA `35.0 min`.
  - shard 1: `900,204,834` tokens, `53.3%`, ETA `37.7 min`.
  - shard 2: `900,113,734` tokens, `53.3%`, ETA `39.2 min`.
  - shard 3: `1,000,085,132` tokens, `59.3%`, ETA `31.0 min`.
- Current part-file sizes:
  - `bulk_mix_part_00.jsonl`: `5.8G`
  - `bulk_mix_part_01.jsonl`: `5.8G`
  - `bulk_mix_part_02.jsonl`: `5.6G`
  - `bulk_mix_part_03.jsonl`: `6.2G`

## 2026-06-09T20:32:01Z - Mix Array Progress Snapshot

- Slurm state unchanged:
  - `2509367_0`-`2509367_3`: running for ~56 minutes.
  - `2509367_[4-7%4]`: pending on `JobArrayTaskLimit`.
  - `2509368` and `2509486`: dependency-pending.
- Mix progress:
  - shard 0: `1,150,386,065` tokens, `68.2%`, ETA `25.5 min`.
  - shard 1: `1,150,287,884` tokens, `68.2%`, ETA `25.6 min`.
  - shard 2: `1,100,170,556` tokens, `65.2%`, ETA `29.2 min`.
  - shard 3: `1,200,088,472` tokens, `71.1%`, ETA `22.1 min`.
- Current part-file sizes:
  - `bulk_mix_part_00.jsonl`: `7.2G`
  - `bulk_mix_part_01.jsonl`: `7.1G`
  - `bulk_mix_part_02.jsonl`: `6.9G`
  - `bulk_mix_part_03.jsonl`: `7.5G`

## 2026-06-09T20:47:24Z - Mix Array Progress Snapshot

- Slurm state unchanged:
  - `2509367_0`-`2509367_3`: running for ~1h12m.
  - `2509367_[4-7%4]`: pending on `JobArrayTaskLimit`.
  - `2509368` and `2509486`: dependency-pending.
- Mix progress:
  - shard 0: `1,500,621,167` tokens, `88.9%`, ETA `8.8 min`.
  - shard 1: `1,450,297,156` tokens, `85.9%`, ETA `11.3 min`.
  - shard 2: `1,400,292,728` tokens, `83.0%`, ETA `14.3 min`.
  - shard 3: `1,550,383,239` tokens, `91.9%`, ETA `6.1 min`.
- Current part-file sizes:
  - `bulk_mix_part_00.jsonl`: `9.2G`
  - `bulk_mix_part_01.jsonl`: `9.2G`
  - `bulk_mix_part_02.jsonl`: `8.7G`
  - `bulk_mix_part_03.jsonl`: `9.7G`

## 2026-06-09T20:57:47Z - Mix Array Wave Transition

- Slurm state:
  - `2509367_0`: completed; manifest written.
  - `2509367_1`: completed; manifest written.
  - `2509367_3`: completed; manifest written.
  - `2509367_2`: still running at `1,600,350,307` tokens, `94.8%`,
    ETA `4.3 min`.
  - `2509367_4`: started, running for ~6 minutes; last progress
    `100,007,122` tokens, `5.9%`, ETA `85.8 min`.
  - `2509367_[5-7%4]`: still pending on `JobArrayTaskLimit`.
  - `2509368` and `2509486`: dependency-pending.
- Finished shard manifests are present for parts 00, 01, and 03.
- Current part-file sizes:
  - `bulk_mix_part_00.jsonl`: `11G`
  - `bulk_mix_part_01.jsonl`: `11G`
  - `bulk_mix_part_02.jsonl`: `11G` and still growing.
  - `bulk_mix_part_03.jsonl`: `11G`
  - `bulk_mix_part_04.jsonl`: `635M` and growing.

## 2026-06-09T21:03:16Z - Mix Array Wave Two Started

- Slurm state:
  - `2509367_0`-`2509367_3`: completed; manifests present for parts 00-03.
  - `2509367_4`: running for ~12 minutes; last progress `200,013,015`
    tokens, `11.9%`, ETA `74.1 min`.
  - `2509367_5`: running for ~3 minutes; scheduler initialized and output
    file is growing.
  - `2509367_6`: running for ~2 minutes; scheduler initialized and output
    file is growing.
  - `2509367_7`: running for ~2 minutes; scheduler initialized and output
    file exists.
  - `2509368` and `2509486`: dependency-pending.
- Current part-file sizes:
  - parts 00-03: `11G` each, complete.
  - `bulk_mix_part_04.jsonl`: `1.4G`.
  - `bulk_mix_part_05.jsonl`: `118M`.
  - `bulk_mix_part_06.jsonl`: `12M`.
  - `bulk_mix_part_07.jsonl`: file exists; early startup.

## 2026-06-09T21:20:14Z - Review-Agent Findings Addressed

- Read `REVIEW_20260610_CPT_2ARM_PRELAUNCH.md`.
- B1 BLOCKER confirmed and fixed:
  - The launcher uses the full-repo trainer dir:
    `$SC/repo/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training`.
  - That tree was missing `../megatron_patches/runtime/pretrain_gpt_te_guard.py`.
  - Rsynced the `megatron_patches/` subtree from the standalone legacy 03 tree
    into the full-repo tree, without repointing `TRAIN_DIR`.
  - Verified the exact runtime path:
    `$TRAIN_DIR/../megatron_patches/runtime/pretrain_gpt_te_guard.py`.
  - Verified `python3 -m py_compile` on the wrapper.
- M1 MAJOR fixed:
  - Patched active Megatron fork
    `/iopsstor/scratch/cscs/fffoivos/code/training/Megatron-LM-Swiss-AI/megatron/training/training.py`
    so `evaluate_and_print_results()` prefixes TensorBoard/W&B validation
    scalar keys when the eval prefix ends in `[name]`.
  - Remote backup:
    `training.py.codex_m1_20260609T2116Z.bak`.
  - Verified active Megatron file compiles and contains `validation_name` /
    `tb_key` logic.
  - Patched `03_training_experiments/scripts/collect_metrics.py` to parse
    named validation lines from `.out` files into CSV rows with
    `metric_type=valid` and `validation_set=<name>`.
  - Local and remote synthetic-log tests produced one training row plus
    distinct `hplt`, `openarchives`, and `greek_phd` validation rows.
  - Updated and synced `dataset_build/EXTRA_VALID_README.md` so it no longer
    claims stock Megatron emits distinct per-set TensorBoard scalars.
- Additional hardening from review minors:
  - Extended `submit_two_arm_full_run.sh` export-hygiene guard to cover omitted
    WSD warmup/cooldown, AdEMAMix warmup, and eval cadence variables.
  - Wired `launch_all.sh` to run `scripts/gate_cpt2arm_artifacts.sh` for live
    launches by default (`RUN_ARTIFACT_GATE=1`).
  - Updated `LAUNCH_RUNBOOK.md` to make the gate an explicit required step.
  - Extended `gate_cpt2arm_artifacts.sh` to check the TE guard wrapper and the
    active Megatron extra-valid TensorBoard-key patch.
- Verification after fixes:
  - Remote syntax passed for `launch_all.sh`, `submit_two_arm_full_run.sh`, and
    `gate_cpt2arm_artifacts.sh`.
  - Remote dry-run with watchers enabled still resolves both arms correctly.
  - Upgraded artifact gate passes config/tokenizer/init/runtime/held-out-val
    checks and fails only on expected Stage-B outputs:
    `bulk_mix_final.jsonl` and `bulk_mix_{base,ext}_text_document.{bin,idx}`.
- Smoke status:
  - The review's 50-iter `EVAL_INTERVAL=5` smoke is still pending. It should be
    run before the full live launch once a suitable training data prefix exists
    after Stage B, unless we deliberately choose a smaller smoke prefix.

## 2026-06-09T21:20:41Z - Mix Array Progress Snapshot

- Wave two Slurm state:
  - `2509367_4`: running for ~27 minutes.
  - `2509367_5`: running for ~19 minutes.
  - `2509367_6`: running for ~17 minutes.
  - `2509367_7`: running for ~17 minutes.
  - `2509368` and `2509486`: dependency-pending.
- Mix progress:
  - shard 4: `500,230,441` tokens, `29.6%`, ETA `57.6 min`.
  - shard 5: `300,010,963` tokens, `17.8%`, ETA `75.8 min`.
  - shard 6: `300,020,728` tokens, `17.8%`, ETA `75.3 min`.
  - shard 7: `400,025,550` tokens, `23.7%`, ETA `54.3 min`.
- Current wave-two part-file sizes:
  - `bulk_mix_part_04.jsonl`: `3.4G`.
  - `bulk_mix_part_05.jsonl`: `2.0G`.
  - `bulk_mix_part_06.jsonl`: `2.0G`.
  - `bulk_mix_part_07.jsonl`: `2.5G`.

## 2026-06-09T21:27:58Z - User ETA Check

- Live status checked after user asked for ETA.
- Mix array:
  - parts 00-03 completed successfully.
  - `2509367_4`: `700,255,909` tokens, `41.5%`, ETA `48.1 min`.
  - `2509367_5`: `450,013,245` tokens, `26.7%`, ETA `67.1 min`.
  - `2509367_6`: `450,047,839` tokens, `26.7%`, ETA `67.3 min`.
  - `2509367_7`: `600,032,000` tokens, `35.6%`, ETA `45.1 min`.
  - Approximate total mix progress: about two-thirds complete.
- Stage A `2509368`: still dependency-pending.
- Stage B `2509486`: still dependency-pending.
- Local leftover delayed-poll process from the interrupted turn was killed;
  only Slurm jobs remain active.

## 2026-06-09T21:28:57Z - Mix Array Progress Snapshot

- Confirmed no leftover local delayed-poll process remains after the interrupted
  turn.
- Wave two Slurm state:
  - `2509367_4`: running for ~37 minutes.
  - `2509367_5`: running for ~29 minutes.
  - `2509367_6`: running for ~28 minutes.
  - `2509367_7`: running for ~28 minutes.
  - `2509368` and `2509486`: still dependency-pending.
- Mix progress:
  - shard 4: `750,260,581` tokens, `44.5%`, ETA `45.6 min`.
  - shard 5: `500,013,621` tokens, `29.6%`, ETA `64.5 min`.
  - shard 6: `500,048,242` tokens, `29.6%`, ETA `64.7 min`.
  - shard 7: `650,032,957` tokens, `38.5%`, ETA `42.9 min`.
- Stage A/B logs are still empty because neither job has started.

## 2026-06-09T21:33:45Z - CPU-Only Resource Correction

- Investigated user concern that dataset building was slow and whether more
  CPU parallelism was available.
- Findings:
  - The mix build is parallel only across array shards: `0-7%4` in the live
    submitted job, with one Python process per shard.
  - `mix_builder.py` streams rows in deterministic order and calls
    `tokenizer.encode(text)` once per row for token-budget accounting. That
    hot path is not a multi-process CPU fan-out; more cores inside one shard
    do not produce linear speedup without rewriting the mixer around batched
    tokenization / parallel source readers.
  - Slurm showed the live `normal`-partition mix shards allocated whole nodes
    with `AllocTRES=cpu=288,...,gres/gpu=4`, even though the workload itself is
    CPU-only. This is a resource-placement issue, not GPU use by the code.
  - `sinfo` showed `xfer` is currently available with two idle CPU-only nodes
    (`128` CPUs, no GRES).
- Actions:
  - Moved dependency-pending Stage A (`2509368`) and Stage B (`2509486`) to
    `Partition=xfer`; both remained dependency-pending and now request CPU-only
    nodes.
  - Patched and synced dataset-build Slurm wrappers to use `#SBATCH
    --partition=xfer` for future CPU-only corpus jobs.
  - Patched the documented future mix launch from `--array=0-7%4` to
    `--array=0-7%2` so future reruns match the two CPU-only `xfer` nodes.
- Current running mix shards cannot be moved in place. Cancelling/requeueing
  shards `4-7` onto `xfer` would require preserving the 8-way shard count and
  would discard their in-progress JSONL output.

## 2026-06-09T21:45:03Z - Mix Progress After Resource Correction

- Stage A (`2509368`) and Stage B (`2509486`) remain dependency-pending and
  pointed at `xfer`.
- Running mix wave remains on its originally allocated `normal` nodes:
  - `2509367_4`: `1,050,278,948` tokens, `62.2%`, ETA `31.0 min`.
  - `2509367_5`: `800,250,588` tokens, `47.4%`, ETA `48.4 min`.
  - `2509367_6`: `750,074,028` tokens, `44.4%`, ETA `51.4 min`.
  - `2509367_7`: `1,050,197,242` tokens, `62.2%`, ETA `26.2 min`.

## 2026-06-09T22:00:22Z - Mix Progress

- Stage A (`2509368`) and Stage B (`2509486`) remain dependency-pending on
  `xfer`.
- Running mix wave:
  - `2509367_4`: `1,400,318,633` tokens, `83.0%`, ETA `13.9 min`.
  - `2509367_5`: `1,050,436,071` tokens, `62.2%`, ETA `34.7 min`.
  - `2509367_6`: `1,050,403,947` tokens, `62.2%`, ETA `35.0 min`.
  - `2509367_7`: `1,400,217,008` tokens, `83.0%`, ETA `11.7 min`.

## 2026-06-09T22:15:41Z - Mix Shards 4 and 7 Complete

- Stage A (`2509368`) and Stage B (`2509486`) remain dependency-pending on
  `xfer`; this is expected until all mix array shards finish.
- Completed:
  - `2509367_4`: `DONE shard 4 Tue Jun 9 22:13:19 UTC 2026`.
  - `2509367_7`: `DONE shard 7 Tue Jun 9 22:10:09 UTC 2026`.
- Still running:
  - `2509367_5`: `1,350,529,555` tokens, `80.0%`, ETA `18.4 min`.
  - `2509367_6`: `1,300,657,880` tokens, `77.1%`, ETA `21.3 min`.

## 2026-06-09T22:41:01Z - Mix Complete; xfer Env Failure Found

- Mix array `2509367` completed all eight shards successfully.
  - `2509367_5`: `DONE shard 5 Tue Jun 9 22:31:30 UTC 2026`.
  - `2509367_6`: `DONE shard 6 Tue Jun 9 22:33:16 UTC 2026`.
- Stage A `2509368` started on `xfer`, concatenated the eight shards, and
  failed before cleaning/decontamination:
  - `bulk_mix.jsonl`: `10,940,329` rows.
  - Error: `uenv: command not found`.
- Stage B `2509486` became `DependencyNeverSatisfied`; it was cancelled before
  recovery submission.
- Diagnosis:
  - `xfer` is CPU-only (`128` CPUs, no GPU GRES), but it does not provide
    `uenv`; its system Python is `3.6.15` and lacks required packages.
  - `uv` and a standalone x86_64 Python 3.12 are available on `xfer`.
  - The previous build venv points at an ARM `/user-environment/.../python3.12`
    and cannot run on `xfer`.
- Recovery edits:
  - Added `dataset_build/build_xfer_env.sbatch`.
  - Patched CPU corpus wrappers to use `xfer` plus
    `$SC/python_envs/cpt_build_xfer_py312` instead of `uenv`.
  - Patched Stage A to use the standalone xfer Python 3.12 directly because it
    only needs stdlib.
  - Added stale-output cleanup before Stage A/B retries.
  - Updated `dataset_build/HANDOFF.md` with the xfer-native env dependency.
- Recovery jobs submitted:
  - `2510620`: `cpt13b_xfer_env`.
  - `2510621`: fresh `cpt13b_A_clean_decontam`.
  - `2510622`: fresh `cpt13b_B_anon_preprocess`, dependent on both `2510621`
    and `2510620`.

## 2026-06-09T22:53:14Z - xfer Env Complete; Stage A Running

- `2510620` (`cpt13b_xfer_env`) completed in `00:02:16` on `xfer`.
  Verification output:
  - `xfer env OK`.
  - `torch 2.9.1+cpu`.
  - `numpy 2.4.1`.
  - `transformers 5.10.2`.
- `2510621` (`cpt13b_A_clean_decontam`) is running on `xfer`.
  - Re-concatenated eight shards.
  - `bulk_mix.jsonl`: `10,940,329` rows.
  - Entered HPLT confident-only E001 clean.
  - Progress seen on stderr: `cleaned 1,000,000`, `cleaned 2,000,000`.
- `2510622` (`cpt13b_B_anon_preprocess`) remains dependency-pending.

## 2026-06-09T23:03:32Z - Stage A E001 Clean Complete

- `2510621` remains running on `xfer`.
- HPLT confident-only E001 clean completed:
  - `10,940,329` docs cleaned.
  - Output: `$STAGE/bulk_mix_clean.jsonl`.
- Stage A entered GreekMMLU `correct_only` decontamination and is loading
  queries from the documented 5B decontam query artifact.
- `2510622` remains dependency-pending.

## 2026-06-09T23:13:51Z - Stage A Decontam Filtering

- `2510621` remains running on `xfer`.
- GreekMMLU `correct_only` decontam scan completed:
  - Scan time: `488.9 s`.
  - Contaminated doc IDs: `289`.
  - Audit artifacts written under `$STAGE/decontam_output/`.
- Filter pass is writing:
  - `$STAGE/bulk_mix_decontam.jsonl`: `17G` at snapshot time.
  - `$STAGE/bulk_mix_dropped.jsonl`: exists and is being written.
  - Progress at snapshot: `2,100,000` rows, `2,099,950` clean, `50` dropped.
- `2510622` remains dependency-pending.

## 2026-06-09T23:24:17Z - Stage A Complete; Stage B Started

- Stage A `2510621` completed successfully in `00:37:26` on `xfer`.
- Final Stage A outputs:
  - `$STAGE/bulk_mix_decontam.jsonl`: `83G`.
  - `$STAGE/bulk_mix_dropped.jsonl`: `26M`.
  - Clean rows: `10,940,034`.
  - Dropped rows: `295`.
- GreekMMLU decontam summary:
  - `correct_only` contaminated items: `478`.
  - `correct_only` contaminated item-doc pairs: `1069`.
  - `any` contaminated items: `636`.
  - `all` contaminated items: `217`.
- Stage B `2510622` started on `xfer` after Stage A + xfer-env dependencies.
  Slurm allocation shows CPU-only resources: `cpu=64,mem=240G,node=1`.

## 2026-06-09T23:31:48Z - Stage B Env Dependency Miss; Retry Submitted

- Stage B `2510622` failed early in anonymization after splitting the
  decontaminated corpus into 64 JSONL shards.
- Error:
  - `ImportError: Please install orjson to use JsonlReader`.
- Fixes:
  - Added `orjson` to `build_xfer_env.sbatch`.
  - Added a Stage B xfer-env preflight before the expensive 83G split:
    imports `datatrove`, `orjson`, `torch`, `transformers`, and checks both
    tokenizers load with expected vocab sizes (`131072`, `148480`).
  - Synced patched scripts to Clariden.
- Retry submitted:
  - `2511102`: xfer-env refresh.
  - `2511103`: Stage B retry, dependent on `2511102`.

## 2026-06-09T23:34:50Z - Stage B Retry Preflight Passed

- `2511102` completed in `00:00:39`; installed `orjson==3.11.9`.
- `2511103` is running on `xfer`.
- Stage B preflight passed before splitting:
  - Base tokenizer vocab: `131072`.
  - Extended tokenizer vocab: `148480`.
  - `torch 2.9.1+cpu`.
  - `numpy 2.4.1`.
  - `transformers 5.10.2`.

## 2026-06-09T23:40:07Z - Stage B Formatter Bug; Retry Submitted

- Stage B `2511103` failed in DataTrove anonymization.
- Cause:
  - `PIIFormatter` subclassed DataTrove `BaseFormatter`, whose default
    `run()` passes `doc.text` (a string) to `format()`.
  - Our `PIIFormatter.format()` incorrectly expected a `Document` and accessed
    `doc.text`, producing `AttributeError: 'str' object has no attribute 'text'`.
- Fixes:
  - Patched `PIIFormatter` to override `run(data, rank, world_size)`, mutate
    each `Document`, and preserve `pii_count` / `pii_by_type` metadata.
  - Kept `format(text)` as string-only compatibility behavior.
  - Added a Stage B preflight smoke using a dummy `Document` to assert
    `test@example.com` becomes `<email-pii>` and metadata count is `1`.
- Synced formatter and Stage B wrapper to Clariden.
- New Stage B retry submitted:
  - `2511134`.

## 2026-06-09T23:48:21Z - Stage B Anonymization Running

- Stage B retry `2511134` is running on `xfer`.
- Expanded preflight passed again.
- DataTrove anonymization is active:
  - `decontam_indir`: `83G`, 64 split input shards.
  - `masked`: `18G`, 64 output files at snapshot time.
  - Logs show rank-level JsonlReader -> PIIFormatter -> JsonlWriter pipeline.

## 2026-06-09T23:58:48Z - Stage B Anonymization Complete; Tokenization Started

- Stage B `2511134` remains running on `xfer`.
- DataTrove anonymization completed successfully:
  - `10,940,034` docs processed.
  - `doc_len` before masking: `52,297,237,768`.
  - `doc_len` after masking: `52,293,571,999`.
  - Runtime: about `4 minutes`.
- `$STAGE/bulk_mix_final.jsonl` exists and was `68G` at the snapshot.
- Stage B moved into post-anonymization/tokenization; Megatron output directory
  is active.

## 2026-06-10T00:04:02Z - Stage B Megatron Import Dependency; Reuse-Final Retry

- Stage B `2511134` failed in Megatron preprocessing import.
- Error:
  - `ModuleNotFoundError: No module named 'einops'`.
  - Import path: `tools/preprocess_data.py` -> Megatron tokenizer ->
    multimodal tokenizer -> vision/RADIO -> `einops`.
- The anonymized final stream was produced before the failure:
  - `$STAGE/bulk_mix_final.jsonl`: `84G`.
- Fixes:
  - Added `einops` to `build_xfer_env.sbatch`.
  - Added Megatron preprocessing imports to the Stage B preflight so missing
    Megatron-side dependencies fail before anonymization.
  - Added `REUSE_FINAL=1` mode to Stage B so retries can reuse the already
    anonymized `bulk_mix_final.jsonl` and go straight to tokenization.
  - Fixed the literal `$(wc ...)` masked-docs print.
- Retry submitted:
  - `2511171`: xfer-env refresh.
  - `2511172`: Stage B retry with `REUSE_FINAL=1`, dependent on `2511171`.

## 2026-06-10T00:07:14Z - Stage B Reuse-Final Retry Running

- `2511171` completed in `00:00:26`; installed `einops==0.8.2`.
- `2511172` is running on `xfer`.
- Expanded preflight passed:
  - Tokenizers load with vocab sizes `131072` and `148480`.
  - Megatron preprocessing imports now pass.
- `REUSE_FINAL=1` path is active:
  - Stage B is reusing existing `$STAGE/bulk_mix_final.jsonl`.
  - Job moved past the previous missing-`einops` failure.
- Import warnings about missing Apex/Transformer Engine are expected for this
  CPU-only preprocessing environment.

## 2026-06-10T00:12:26Z - Stage B Base Tokenization In Progress

- Stage B `2511172` remains running on `xfer`.
- Reuse-final path printed `masked docs: 10940034`.
- Base-tokenizer Megatron preprocessing is active:
  - `$STAGE/megatron/bulk_mix_base_text_document.bin`: `8.5G` at snapshot.
  - Megatron dir total: `23G`.
  - Progress in stderr: about `1.27M / 10.94M` docs at about `5.2k docs/s`.

## 2026-06-10T00:49:31Z - Base Tokenization Complete; Extended Tokenization Started

- Stage B `2511172` remains running on `xfer`.
- Base-tokenizer Megatron preprocessing completed:
  - `$STAGE/megatron/bulk_mix_base_text_document.bin`: `73G`.
  - `$STAGE/megatron/bulk_mix_base_text_document.idx`: `209M`.
- Extended-tokenizer preprocessing has started:
  - `$STAGE/megatron/bulk_mix_ext_text_document.bin`: `5.8G` at snapshot.
  - Progress in stderr: about `1.26M / 10.94M` docs, roughly `6.4k docs/s`.

## 2026-06-10T01:04:52Z - Extended Tokenization In Progress

- Stage B `2511172` remains running on `xfer`.
- Extended-tokenizer Megatron preprocessing is active:
  - `$STAGE/megatron/bulk_mix_ext_text_document.bin`: `32G` at snapshot.
  - Megatron dir total: `119G`.
  - Progress in stderr: about `6.82M / 10.94M` docs, roughly `6.1k docs/s`.

## 2026-06-10T01:22:05Z - Stage B Complete; Artifact Gate Passed

- Stage B retry `2511172` completed successfully on `xfer`:
  - State: `COMPLETED`, elapsed `01:12:05`, exit `0:0`, `64` CPUs.
  - Reused existing anonymized final stream via `REUSE_FINAL=1`.
  - Final masked docs: `10,940,034`.
- Full training Megatron artifacts now exist:
  - Base train: `bulk_mix_base_text_document.bin` `73G`; `.idx` `209M`.
  - Extended train: `bulk_mix_ext_text_document.bin` `51G`; `.idx` `209M`.
- Ran `scripts/gate_cpt2arm_artifacts.sh` from the Clariden experiment tree.
- Gate result: `ARTIFACT GATE PASSED`.
- The slow build phase is now complete. The bottleneck was the serial
  per-tokenizer Megatron preprocessing pass over `bulk_mix_final.jsonl`; the
  earlier mix stage was parallel across Slurm array shards, but the Stage B
  train tokenization was one job per tokenizer, not internally parallelized
  beyond the preprocessing script's own single-process stream.

## 2026-06-10T01:27:00Z - 50-Iteration Launch Smoke Submitted

- Full dataset artifact gate passed immediately before smoke submission.
- Smoke shape:
  - Real launcher/configs/checkpoints/data prefixes.
  - Full-run `TRAIN_TOKENS=13500000000` preserved so WSD stays anchored to the
    production run.
  - `N_SEGMENTS=1`, `EXIT_INTERVAL=50`, `EVAL_INTERVAL=5`, `EVAL_ITERS=1`.
  - Benchmark watchers disabled for the smoke.
  - Run root: `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b_smoke`.
- First live `launch_all.sh` submitted vanilla job `2511279` then aborted before
  TD because `submit_two_arm_full_run.sh` ended with a dry-run-only `&& echo`
  expression, returning status `1` on successful live submissions under the
  parent `set -e`.
- Fixed `submit_two_arm_full_run.sh` to use an explicit `if` for the dry-run
  message, then submitted TD directly.
- Smoke jobs:
  - Vanilla: `2511279`.
  - TD: `2511282`.

## 2026-06-10T01:40:58Z - First Smoke Stopped; Extra-Valid Iterator Bug Found

- First smoke proved:
  - Both arms ran concurrently on `normal`/GH200.
  - Both init checkpoints loaded successfully.
  - Both train datasets and all three named held-out datasets were discovered.
  - Training reached iteration `5` without skips or NaNs.
- Missing: no per-set validation loss lines appeared at `EVAL_INTERVAL=5`.
- Stopped jobs to avoid wasting GPU time:
  - `2511279`: cancelled at `00:16:32`.
  - `2511282`: cancelled at `00:15:57`.
- Root cause:
  - `pretrain_gpt.py::build_extra_valid_iterators()` changed `config.split` to
    `0,1,0` but retained the main run's precomputed `split_matrix` for
    `100,0,0`.
  - The builder therefore returned the extra dataset as the train split while
    the code read `valid_ds`, yielding `None` iterators and no eval calls.
- Fix applied on Clariden Megatron fork:
  - Backup: `pretrain_gpt.py.codex_extra_valid_20260610T0143Z.bak`.
  - Build each held-out stream as a single-prefix dataset with
    `[eval_samples, 0, 0]` and use the returned `eval_ds` as the eval iterator.
  - `python3 -m py_compile pretrain_gpt.py` passed.
- Submitter fixes:
  - `submit_two_arm_full_run.sh` and `launch_all.sh` both had a final
    dry-run-only `&& echo` expression that returned status `1` after successful
    live submissions. Replaced with explicit `if` blocks.
- Post-fix short smoke submitted:
  - Shape: `N_SEGMENTS=1`, `EXIT_INTERVAL=6`, `EVAL_INTERVAL=5`,
    `EVAL_ITERS=1`, watchers disabled.
  - Vanilla: `2511608`.
  - TD: `2511609`.

## 2026-06-10T02:04:50Z - Second Smoke Stopped; TP Eval Participation Bug Found

- Jobs `2511608` and `2511609` reached iteration `5`; the eval branch fired
  (`RerunMode.DISABLED`) but no validation loss printed and the jobs did not
  progress to iteration `6`.
- Stopped jobs to avoid wasting GPU time:
  - `2511608`: cancelled at `00:20:39`.
  - `2511609`: cancelled at `00:20:33`.
- Root cause:
  - The patched extra-valid loop skipped sets whose iterator was `None`.
  - Under TP=2, only tensor-parallel source ranks own data iterators; sibling TP
    ranks must still enter `evaluate()` with `data_iterator=None` so
    `get_batch_on_this_tp_rank()` can receive broadcast batches and all model
    collectives stay aligned.
- Fix applied on Clariden Megatron fork:
  - Backup: `training.py.codex_extra_valid_tp_20260610T0207Z.bak`.
  - Removed the `None`-iterator skip in both interval eval and final eval.
  - Added comments explaining TP source-rank iterator ownership.
  - `python3 -m py_compile training.py` passed.
- Post-fix TP-aligned short smoke submitted:
  - Shape: `N_SEGMENTS=1`, `EXIT_INTERVAL=6`, `EVAL_INTERVAL=5`,
    `EVAL_ITERS=1`, watchers disabled.
  - Vanilla: `2511661`.
  - TD: `2511662`.

## 2026-06-10T02:29:00Z - Third Smoke Validated Extra-Valid Eval; Checkpoint Save Bug Found

- Jobs `2511661` and `2511662` reached iteration `5`, printed all three
  per-set held-out validation losses, completed iteration `6`, then failed
  while saving the exit checkpoint.
- Validation evidence:
  - Vanilla at iter 5:
    - `hplt` lm loss `1.469468`, PPL `4.346922`.
    - `openarchives` lm loss `1.335643`, PPL `3.802440`.
    - `greek_phd` lm loss `1.378093`, PPL `3.967328`.
  - TD at iter 5:
    - `hplt` lm loss `6.227324`, PPL `506.3983`.
    - `openarchives` lm loss `4.489066`, PPL `89.03826`.
    - `greek_phd` lm loss `4.540198`, PPL `93.70934`.
- Failure:
  - Both arms raised `TypeError: cannot pickle 'generator' object` during
    checkpoint save.
  - Root cause: the extra-valid iterator dict was attached to global `args` as
    `args._extra_valid_iterators`; checkpoint serialization deep-copies through
    `args`, so it attempted to pickle live generator objects.
- Fix applied on Clariden Megatron fork:
  - Backup: `training.py.codex_extra_valid_ckpt_20260610T0230Z.bak`.
  - Added `save_checkpoint_without_extra_valid_iterators()`, which temporarily
    removes `args._extra_valid_iterators` around checkpoint serialization and
    restores it immediately afterward.
  - Routed the post-training and timed checkpoint-save paths through the shim.
  - `python3 -m py_compile megatron/training/training.py` passed.

## 2026-06-10T02:47:00Z - End-To-End Short Smoke Passed

- Submitted save/eval smoke:
  - Shape: `N_SEGMENTS=1`, `EXIT_INTERVAL=6`, `SAVE_INTERVAL=6`,
    `EVAL_INTERVAL=5`, `EVAL_ITERS=1`, watchers disabled.
  - Vanilla: `2511696`.
  - TD: `2511697`.
- Final Slurm states:
  - `2511696`: `COMPLETED`, elapsed `00:18:24`, exit `0:0`.
  - `2511697`: `COMPLETED`, elapsed `00:18:36`, exit `0:0`.
- Both arms:
  - loaded init checkpoints;
  - reached iteration `6`;
  - printed all three per-set held-out validation losses at iteration `5`;
  - reported zero skipped iterations and zero NaN iterations;
  - saved checkpoint at iteration `6` successfully.
- Smoke validation evidence:
  - Vanilla at iter 5:
    - `hplt` lm loss `1.469567`, PPL `4.347351`.
    - `openarchives` lm loss `1.335728`, PPL `3.802764`.
    - `greek_phd` lm loss `1.378191`, PPL `3.967716`.
  - TD at iter 5:
    - `hplt` lm loss `6.236537`, PPL `511.0858`.
    - `openarchives` lm loss `4.494983`, PPL `89.56662`.
    - `greek_phd` lm loss `4.546360`, PPL `94.28857`.
- `collect_metrics.py` parsed the smoke successfully:
  - output CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b_smoke/metrics_smoke6_saveeval_20260610T0231Z.csv`;
  - `18` rows total;
  - valid sets detected for both arms: `greek_phd`, `hplt`, `openarchives`;
  - summary reported zero NaN and zero skipped iterations.
- Measured training/eval timing:
  - steady training iteration time: about `130s` vanilla, `132s` TD.
  - each three-set held-out eval event costs about `145s`.
  - Therefore `EVAL_INTERVAL=1` would materially slow the full run.
- Full-run eval-cadence decision:
  - changed `configs/common_cpt.env` default from `EVAL_INTERVAL=1` to
    `EVAL_INTERVAL=25`, keeping `EVAL_ITERS=1`.
  - updated `dataset_build/HANDOFF.md` and `dataset_build/EXTRA_VALID_README.md`
    so docs match the as-run config.
  - synced these files to the Clariden repo mirror.
  - verified remote config resolves to `EVAL_INTERVAL=25`, `EVAL_ITERS=1`,
    `SAVE_INTERVAL=119`.
- Ran `scripts/gate_cpt2arm_artifacts.sh` after the cadence change.
  - Result: `ARTIFACT GATE PASSED`.
  - The gate now explicitly checks `EVAL_INTERVAL=25` and `EVAL_ITERS=1` so
    future launches cannot silently revert to every-iteration held-out eval.

## 2026-06-10T02:52:00Z - Full Two-Arm Launch Submitted

- Launched from the Clariden mirror with:
  - `DRY_RUN=0 CONFIRM_LAUNCH=1 bash scripts/launch_all.sh`.
  - Stamp: `20260610T025159Z`.
  - Prelaunch artifact gate enabled and passed.
  - Watcher partition: `xfer`.
- Full-run config at launch:
  - `TRAIN_TOKENS=13500000000`.
  - `TRAIN_ITERS=3218`.
  - `SAVE_INTERVAL=119`.
  - `EVAL_INTERVAL=25`, `EVAL_ITERS=1`.
  - `N_SEGMENTS=14`, `12:00:00` per segment.
- Vanilla chain:
  - run tag: `cpt13b_vanilla_20260610T025159Z`.
  - run dir:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/cpt13b_vanilla_20260610T025159Z`.
  - segment jobs: `2511719` through `2511732`.
  - watcher job: `2511733`.
  - chain manifest:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/cpt13b_vanilla_20260610T025159Z_submit_state/chain.tsv`.
- TD chain:
  - run tag: `cpt13b_td_20260610T025159Z`.
  - run dir:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/cpt13b_td_20260610T025159Z`.
  - segment jobs: `2511734` through `2511747`.
  - watcher job: `2511748`.
  - chain manifest:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/cpt13b_td_20260610T025159Z_submit_state/chain.tsv`.
- Initial Slurm state:
  - vanilla segment 1 `2511719`: `RUNNING` on `normal`.
  - TD segment 1 `2511734`: `RUNNING` on `normal`.
  - watcher `2511733`: `RUNNING` on `xfer`.
  - watcher `2511748`: `RUNNING` on `xfer`.
  - remaining segments: dependency-pending.

## 2026-06-10T02:56:00Z - Full Run Early Health Check

- Active jobs still running:
  - vanilla segment 1 `2511719` on `normal`.
  - TD segment 1 `2511734` on `normal`.
  - watcher `2511733` on `xfer`.
  - watcher `2511748` on `xfer`.
- Both arms loaded init checkpoints successfully.
- Both arms reached iteration `1`:
  - Vanilla iter 1: lm loss `1.458773`, LR `5.623750e-06`,
    throughput `395.0` TFLOP/s/GPU, skipped `0`, NaN `0`.
  - TD iter 1: lm loss `5.689698`, LR `5.623750e-06`,
    throughput `392.3` TFLOP/s/GPU, skipped `0`, NaN `0`.
- These match the successful save/eval smoke behavior closely.

## 2026-06-10T03:03:00Z - Full Run Advancing Normally

- Active jobs still running:
  - vanilla segment 1 `2511719` on `normal`.
  - TD segment 1 `2511734` on `normal`.
  - watcher `2511733` on `xfer`.
  - watcher `2511748` on `xfer`.
- Both arms reached iteration `4` with zero skipped and zero NaN iterations.
- Latest live-log lines:
  - Vanilla iter 4: lm loss `1.472298`, LR `5.995000e-06`,
    throughput `413.1` TFLOP/s/GPU, ETA `4 days, 20:48:31`.
  - TD iter 4: lm loss `5.207827`, LR `5.995000e-06`,
    throughput `411.7` TFLOP/s/GPU, ETA `4 days, 22:09:13`.
- Ran `collect_metrics.py` on the live full-run logs:
  - output CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `8` rows total at this snapshot;
  - summary reported zero NaN and zero skipped iterations for both arms.
- Next important checkpoint: first production held-out eval at iteration `25`.

## 2026-06-10T03:18:00Z - Full Run Steady-State Heartbeat

- Active jobs still running:
  - vanilla segment 1 `2511719` on `normal`.
  - TD segment 1 `2511734` on `normal`.
  - watcher `2511733` on `xfer`.
  - watcher `2511748` on `xfer`.
- Both arms reached iteration `11` with zero skipped and zero NaN iterations.
- Latest live-log lines:
  - Vanilla iter 11: lm loss `1.433569`, LR `6.861250e-06`,
    throughput `412.6` TFLOP/s/GPU, ETA `4 days, 20:41:41`.
  - TD iter 11: lm loss `4.306079`, LR `6.861250e-06`,
    throughput `412.1` TFLOP/s/GPU, ETA `4 days, 21:47:45`.
- Ran `collect_metrics.py` on the live full-run logs:
  - updated the same metrics CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `22` rows total at this snapshot;
  - summary reported zero NaN and zero skipped iterations for both arms.
- Next important checkpoint remains first production held-out eval at iteration
  `25`.

## 2026-06-10T03:55:00Z - First Production Held-Out Eval Passed

- Active jobs still running:
  - vanilla segment 1 `2511719` on `normal`.
  - TD segment 1 `2511734` on `normal`.
  - watcher `2511733` on `xfer`.
  - watcher `2511748` on `xfer`.
- Both arms reached iteration `25`, printed all three production held-out
  validation losses at `EVAL_INTERVAL=25`, and then continued training.
- Vanilla:
  - iter 25 train loss `1.415766`, LR `8.593750e-06`, skipped `0`, NaN `0`.
  - validation at iter 25:
    - `hplt` lm loss `1.425550`, PPL `4.160147`.
    - `openarchives` lm loss `1.287299`, PPL `3.622989`.
    - `greek_phd` lm loss `1.331695`, PPL `3.787458`.
  - reached iter 27 by this snapshot, ETA about `4 days, 19:56:42`.
- TD:
  - iter 25 train loss `3.482515`, LR `8.593750e-06`, skipped `0`, NaN `0`.
  - validation at iter 25:
    - `hplt` lm loss `4.271772`, PPL `71.64851`.
    - `openarchives` lm loss `2.927790`, PPL `18.68630`.
    - `greek_phd` lm loss `2.931250`, PPL `18.75105`.
  - reached iter 26 by this snapshot, ETA about `4 days, 21:19:14`.
- Ran `collect_metrics.py` on the live full-run logs:
  - updated CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `59` rows total at this snapshot;
  - valid sets detected for both arms: `greek_phd`, `hplt`, `openarchives`;
  - summary reported zero NaN and zero skipped iterations for both arms.
- Next important checkpoint: first regular save at iteration `119`; first
  benchmark-watch checkpoint target is iteration `238`.

## 2026-06-10T03:58:00Z - Post-Eval Continuation Check

- Active jobs still running:
  - vanilla segment 1 `2511719` on `normal`.
  - TD segment 1 `2511734` on `normal`.
  - watcher `2511733` on `xfer`.
  - watcher `2511748` on `xfer`.
- Training continued after the iteration-25 held-out eval:
  - Vanilla reached iter 28: lm loss `1.412282`, LR `8.965000e-06`,
    throughput `413.0` TFLOP/s/GPU, skipped `0`, NaN `0`.
  - TD reached iter 27: lm loss `3.346380`, LR `8.841250e-06`,
    throughput `411.7` TFLOP/s/GPU, skipped `0`, NaN `0`.
- Ran `collect_metrics.py` again:
  - updated CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `61` rows total at this snapshot;
  - valid sets detected for both arms: `greek_phd`, `hplt`, `openarchives`;
  - summary reported zero NaN and zero skipped iterations for both arms.
- Checked non-empty `.err` files. Contents were benign startup/runtime warnings
  and the TE/xIELU guard audit:
  - PyTorch/NCCL deprecation/device warnings only.
  - `pretrain_gpt_te_guard` reported `missing=0` for xIELU optimizer audit on
    both arms.
  - No traceback, `TypeError`, or `RuntimeError` observed.
- Next operational milestone: iteration `50` held-out eval.

## 2026-06-10T04:08:00Z - Iteration-50 Approach Heartbeat

- Active jobs still running:
  - vanilla segment 1 `2511719` on `normal`.
  - TD segment 1 `2511734` on `normal`.
  - watcher `2511733` on `xfer`.
  - watcher `2511748` on `xfer`.
- Latest observed training progress:
  - Vanilla reached iter 33: lm loss `1.406988`, LR `9.583750e-06`,
    throughput `412.9` TFLOP/s/GPU, skipped `0`, NaN `0`.
  - TD reached iter 32: lm loss `3.239351`, LR `9.460000e-06`,
    throughput `411.7` TFLOP/s/GPU, skipped `0`, NaN `0`.
- Ran `collect_metrics.py` again:
  - updated CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `71` rows total at this snapshot;
  - valid sets detected for both arms: `greek_phd`, `hplt`, `openarchives`;
  - summary reported zero NaN and zero skipped iterations for both arms.
- Next operational milestone remains the iteration `50` held-out eval.

## 2026-06-10T04:29:00Z - Iteration-42 Heartbeat

- Active jobs still running:
  - vanilla segment 1 `2511719` on `normal`.
  - TD segment 1 `2511734` on `normal`.
  - watcher `2511733` on `xfer`.
  - watcher `2511748` on `xfer`.
- Latest observed training progress:
  - Vanilla reached iter 42: lm loss `1.412787`, LR `1.069750e-05`,
    throughput `413.0` TFLOP/s/GPU, skipped `0`, NaN `0`.
  - TD reached iter 42: lm loss `2.991221`, LR `1.069750e-05`,
    throughput `411.5` TFLOP/s/GPU, skipped `0`, NaN `0`.
- Ran `collect_metrics.py` again:
  - updated CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `90` rows total at this snapshot;
  - valid sets detected for both arms: `greek_phd`, `hplt`, `openarchives`;
  - summary reported zero NaN and zero skipped iterations for both arms.
- Next operational milestone remains the iteration `50` held-out eval.

## 2026-06-10T04:55:00Z - Second Production Held-Out Eval Passed

- Active jobs still running:
  - vanilla segment 1 `2511719` on `normal`.
  - TD segment 1 `2511734` on `normal`.
  - watcher `2511733` on `xfer`.
  - watcher `2511748` on `xfer`.
- Both arms reached iteration `50`, printed all three held-out validation losses,
  and resumed training afterward.
- Vanilla:
  - iter 50 train loss `1.402797`, LR `1.168750e-05`, skipped `0`, NaN `0`.
  - validation at iter 50:
    - `hplt` lm loss `1.408340`, PPL `4.089162`.
    - `openarchives` lm loss `1.260125`, PPL `3.525863`.
    - `greek_phd` lm loss `1.308717`, PPL `3.701420`.
  - reached iter 53 by this snapshot, train loss `1.375656`,
    throughput `413.3` TFLOP/s/GPU.
- TD:
  - iter 50 train loss `2.878681`, LR `1.168750e-05`, skipped `0`, NaN `0`.
  - validation at iter 50:
    - `hplt` lm loss `3.484933`, PPL `32.62025`.
    - `openarchives` lm loss `2.408433`, PPL `11.11653`.
    - `greek_phd` lm loss `2.450562`, PPL `11.59486`.
  - reached iter 52 by this snapshot, train loss `2.821623`,
    throughput `411.5` TFLOP/s/GPU.
- Ran `collect_metrics.py` again:
  - updated CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `117` rows total at this snapshot;
  - valid sets detected for both arms: `greek_phd`, `hplt`, `openarchives`;
  - summary reported zero NaN and zero skipped iterations for both arms.
- Next milestones:
  - iteration `75` held-out eval;
  - iteration `100` held-out eval;
  - first regular checkpoint save at iteration `119`.

## 2026-06-10T04:57:00Z - Resume Heartbeat After Iteration-50 Eval

- Active jobs still running:
  - vanilla segment 1 `2511719` on `normal`.
  - TD segment 1 `2511734` on `normal`.
  - watcher `2511733` on `xfer`.
  - watcher `2511748` on `xfer`.
- Latest observed training progress:
  - Vanilla reached iter 54: lm loss `1.392873`, LR `1.218250e-05`,
    throughput `413.3` TFLOP/s/GPU, skipped `0`, NaN `0`.
  - TD reached iter 53: lm loss `2.778572`, LR `1.205875e-05`,
    throughput `411.6` TFLOP/s/GPU, skipped `0`, NaN `0`.
- Ran `collect_metrics.py` again:
  - updated CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `119` rows total at this snapshot;
  - valid sets detected for both arms: `greek_phd`, `hplt`, `openarchives`;
  - summary reported zero NaN and zero skipped iterations for both arms.
- Next operational milestone remains the iteration `75` held-out eval.

## 2026-06-10T05:58:00Z - Third Production Held-Out Eval Passed

- Active jobs still running:
  - vanilla segment 1 `2511719` on `normal`.
  - TD segment 1 `2511734` on `normal`.
  - watcher `2511733` on `xfer`.
  - watcher `2511748` on `xfer`.
- Both arms reached iteration `75`, printed all three held-out validation losses,
  and resumed training afterward.
- Vanilla:
  - iter 75 train loss `1.386543`, LR `1.478125e-05`, skipped `0`, NaN `0`.
  - validation at iter 75:
    - `hplt` lm loss `1.398878`, PPL `4.050652`.
    - `openarchives` lm loss `1.243402`, PPL `3.467388`.
    - `greek_phd` lm loss `1.296437`, PPL `3.656246`.
  - reached iter 81 by this snapshot, train loss `1.394643`,
    throughput `413.2` TFLOP/s/GPU.
- TD:
  - iter 75 train loss `2.497794`, LR `1.478125e-05`, skipped `0`, NaN `0`.
  - validation at iter 75:
    - `hplt` lm loss `3.015740`, PPL `20.40418`.
    - `openarchives` lm loss `2.134591`, PPL `8.453585`.
    - `greek_phd` lm loss `2.195731`, PPL `8.986565`.
  - reached iter 80 by this snapshot, train loss `2.478377`,
    throughput `412.2` TFLOP/s/GPU.
- Ran `collect_metrics.py` again:
  - updated CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `179` rows total at this snapshot;
  - valid sets detected for both arms: `greek_phd`, `hplt`, `openarchives`;
  - summary reported zero NaN and zero skipped iterations for both arms.
- Next milestones:
  - iteration `100` held-out eval;
  - first regular checkpoint save at iteration `119`.

## 2026-06-10T06:59:00Z - Fourth Production Held-Out Eval Passed

- Active jobs still running:
  - vanilla segment 1 `2511719` on `normal`.
  - TD segment 1 `2511734` on `normal`.
  - watcher `2511733` on `xfer`.
  - watcher `2511748` on `xfer`.
- Both arms reached iteration `100`, printed all three held-out validation
  losses, and resumed training afterward.
- Vanilla:
  - iter 100 train loss `1.377998`, LR `1.787500e-05`, skipped `0`, NaN `0`.
  - validation at iter 100:
    - `hplt` lm loss `1.392096`, PPL `4.023274`.
    - `openarchives` lm loss `1.231256`, PPL `3.425528`.
    - `greek_phd` lm loss `1.288856`, PPL `3.628631`.
  - reached iter 108 by this snapshot, train loss `1.361896`,
    throughput `413.3` TFLOP/s/GPU.
- TD:
  - iter 100 train loss `2.422325`, LR `1.787500e-05`, skipped `0`, NaN `0`.
  - validation at iter 100:
    - `hplt` lm loss `2.826345`, PPL `16.88363`.
    - `openarchives` lm loss `2.019507`, PPL `7.534609`.
    - `greek_phd` lm loss `2.090080`, PPL `8.085564`.
  - reached iter 106 by this snapshot, train loss `2.364927`,
    throughput `411.4` TFLOP/s/GPU.
- Ran `collect_metrics.py` again:
  - updated CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `238` rows total at this snapshot;
  - valid sets detected for both arms: `greek_phd`, `hplt`, `openarchives`;
  - summary reported zero NaN and zero skipped iterations for both arms.
- Next milestone: first regular checkpoint save at iteration `119`.

## 2026-06-10T07:51:53Z - First Production Checkpoint Saved; Eval 125 Passed

- Both arms saved regular checkpoint `iter_0000119` successfully:
  - vanilla:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/cpt13b_vanilla_20260610T025159Z/checkpoints/iter_0000119/common.pt`.
  - TD:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/cpt13b_td_20260610T025159Z/checkpoints/iter_0000119/common.pt`.
  - both `latest_checkpointed_iteration.txt` files report `119`.
- Watcher jobs are still running on `xfer`; checkpoint sidecar `04gnlp_i119`
  is running on `normal`.
- Both arms reached iteration `125`, printed all three held-out validation
  losses, and resumed training afterward.
- Vanilla:
  - iter 125 train loss `1.372797`, LR `2.096875e-05`, skipped `0`, NaN `0`.
  - validation at iter 125:
    - `hplt` lm loss `1.384848`, PPL `3.994219`.
    - `openarchives` lm loss `1.220578`, PPL `3.389146`.
    - `greek_phd` lm loss `1.282693`, PPL `3.606339`.
  - reached iter 131 by this snapshot, train loss `1.379488`,
    throughput `412.8`-`413.3` TFLOP/s/GPU in recent lines.
- TD:
  - iter 125 train loss `2.299750`, LR `2.096875e-05`, skipped `0`, NaN `0`.
  - validation at iter 125:
    - `hplt` lm loss `2.701434`, PPL `14.90108`.
    - `openarchives` lm loss `1.942258`, PPL `6.974481`.
    - `greek_phd` lm loss `2.019880`, PPL `7.537419`.
  - reached iter 129 by this snapshot, train loss `2.302983`,
    throughput `411.6`-`412.1` TFLOP/s/GPU in recent lines.
- Ran `collect_metrics.py` again:
  - updated CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `290` rows total at this snapshot;
  - valid sets detected for both arms: `greek_phd`, `hplt`, `openarchives`;
  - summary reported zero NaN and zero skipped iterations for both arms.
- Next milestones:
  - sidecar completion for checkpoint `119`;
  - held-out eval at iteration `150`;
  - segment checkpoint/exit around iteration `238`.

## 2026-06-10T07:53:23Z - Monitoring Refresh After Checkpoint 119

- Queue state:
  - vanilla segment 1 `2511719` still running on `normal`.
  - TD segment 1 `2511734` still running on `normal`.
  - watcher jobs `2511733` and `2511748` still running on `xfer`.
  - checkpoint sidecar `2512232` (`04gnlp_i119`) running on `normal`.
  - later training segments remain dependency-pending, as expected.
- Rechecked recent train logs:
  - vanilla reached iter `131`, train loss `1.379488`, LR `2.171125e-05`,
    skipped `0`, NaN `0`.
  - TD reached iter `130`, train loss `2.303216`, LR `2.158750e-05`,
    skipped `0`, NaN `0`.
  - no traceback, runtime error, type error, CUDA OOM, skipped-iteration, or
    NaN evidence in the filtered recent log slices.
- Re-ran `collect_metrics.py`:
  - updated CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `291` rows total at this snapshot;
  - vanilla summary: iters `1-131`, last loss `1.3795`, min loss `1.3586`,
    LR `2.171e-05`, `413` TFLOP/s/GPU, NaN `0`, skipped `0`.
  - TD summary: iters `1-130`, last loss `2.3032`, min loss `2.2805`,
    LR `2.159e-05`, `412` TFLOP/s/GPU, NaN `0`, skipped `0`.
  - valid sets detected for both arms: `greek_phd`, `hplt`, `openarchives`.
- Next milestones remain:
  - sidecar completion for checkpoint `119`;
  - held-out eval at iteration `150`;
  - segment checkpoint/exit around iteration `238`.

## 2026-06-10T07:56:13Z - Vanilla Watcher Cadence Corrected

- Found an eval-watcher asymmetry while checking checkpoint-119 sidecars:
  - TD watcher `2511748` is using the 13.5B `CHECKPOINTS_FILE` cadence, whose
    first benchmark target is iter `238`.
  - vanilla watcher `2511733` was a stale deployed copy of the legacy
    5B watcher. It ignored `CHECKPOINTS_FILE`, submitted an extra iter-119
    sidecar, and would have followed the old `119,238,477,834,1192` cadence
    instead of the documented 13.5B cadence.
- Corrective action:
  - synced the already-fixed local
    `subprojects/04_cpt_training_regime_on_vanilla/scripts/watch_and_submit_checkpoint_sidecars.sbatch`
    to the Clariden repo mirror;
  - verified remote `bash -n` and confirmed the deployed script contains the
    `CHECKPOINTS_FILE`/`mapfile` override;
  - cancelled stale vanilla watcher `2511733`;
  - submitted replacement vanilla watcher `2512336` on `xfer`, using:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/cpt13b_vanilla_20260610T025159Z_orch/benchmark_checkpoints.tsv`.
- Training jobs were not touched.
- Already-submitted vanilla iter-119 sidecars were not cancelled; they are extra
  early signal, not part of the 13.5B cadence.
- Verified replacement watcher:
  - Slurm state: `2512336` running on `xfer`;
  - stored batch script for `2512336` contains the `CHECKPOINTS_FILE` override;
  - state dir currently has only the old `iter_119.submitted`/submit log plus
    watcher env files; no incorrect future checkpoint submissions yet.

## 2026-06-10T07:56:47Z - Post-Watcher-Fix Health Check

- Queue after watcher replacement:
  - vanilla training `2511719` running on `normal`.
  - TD training `2511734` running on `normal`.
  - corrected vanilla watcher `2512336` running on `xfer`.
  - TD watcher `2511748` running on `xfer`.
  - extra vanilla iter-119 Greek-NLP sidecar `2512232` running on `normal`.
- Sidecar `2512232`:
  - output path points to the correct 13B vanilla checkpoint HF conversion:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/eval_cpt13b_vanilla_20260610T025159Z/iter_0000119_hf`;
  - display label is stale (`Vanilla-0.5B`) from legacy sidecar naming;
  - machine-translation task completed; NER was at `93/100` in the latest tail.
- Re-ran `collect_metrics.py`:
  - updated CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `294` rows total at this snapshot;
  - vanilla summary: iters `1-133`, last loss `1.3688`, min loss `1.3586`,
    LR `2.196e-05`, `413` TFLOP/s/GPU, NaN `0`, skipped `0`.
  - TD summary: iters `1-131`, last loss `2.3132`, min loss `2.2805`,
    LR `2.171e-05`, `412` TFLOP/s/GPU, NaN `0`, skipped `0`.

## 2026-06-10T08:02:26Z - Retention Sidecar CLI Repair

- Full iter-119 sidecar bundle check found:
  - conversion `2512230`: completed.
  - native MCQ `2512231`: completed.
  - Greek-NLP `2512232`: running.
  - heldout Greek/code/math BPB `2512233`/`2512235`/`2512236`: completed.
  - checksum `2512237`: completed.
  - retention `2512234`: failed before model load with
    `ImportError: cannot import name 'HarnessCLI' from 'lm_eval._cli'`.
- Root cause:
  - the Clariden lm-eval target install at
    `/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval` had
    `lm_eval/__main__.py`, `lm_eval/_cli/harness.py`, and
    `lm_eval/_cli/run.py`, but the helper CLI modules existed only as pyc files
    under `_cli/__pycache__`.
  - Python did not import those sourceless pyc modules from `__pycache__`, so
    `python3 -m lm_eval` failed before argument parsing.
- Corrective action:
  - added and synced
    `03_training_experiments/scripts/repair_lm_eval_cli_install.py`;
  - ran it under `uenv start pytorch/v2.9.1:v2`, targeting the lm-eval install;
  - verified `python3 -m lm_eval --help` now works in the same uenv/PYTHONPATH
    context used by retention sidecars.
- Submitted a small retention smoke, not a full extra 119 retention run:
  - job `2512461`, name `04ret_i119_smoke`;
  - target HF checkpoint:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/eval_cpt13b_vanilla_20260610T025159Z/iter_0000119_hf`;
  - output dir:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/eval_cpt13b_vanilla_20260610T025159Z/iter_0000119/retention_cli_repair_smoke`;
  - `TASK_GROUP=retention_only`, `LIMIT=1`;
  - initial state: pending on `normal` with reason `Priority`.

## 2026-06-10T08:15:18Z - Retention Repair Narrowed; Training Still Healthy

- First retention smoke `2512461` failed after the initial CLI repair exposed a
  second target-install issue:
  - after `lm_eval._cli` was fixed, the package reached
    `lm_eval.config.evaluate_config`, then dependency imports;
  - broad pyc exposure briefly made the target-install NumPy shadow uenv NumPy,
    so the repair script was narrowed again.
- Final repair-script policy:
  - expose sourceless pyc modules only for
    `lm_eval`, `datasets`, `pyarrow`, `multiprocess`, `pandas`, `dateutil`,
    and `pytz`;
  - clean up accidental broad symlinks outside that allowlist;
  - keep NumPy resolved from the uenv.
- Verified under `uenv start pytorch/v2.9.1:v2` with
  `PYTHONPATH=/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval`:
  - NumPy resolves to
    `/user-environment/env/default/lib/python3.12/site-packages/numpy/__init__.py`,
    version `2.4.1`;
  - `datasets.load_dataset` imports;
  - `lm_eval.config.evaluate_config.EvaluatorConfig` imports;
  - `lm_eval.simple_evaluate` imports;
  - `python3 -m lm_eval --help` prints the CLI help.
- Submitted second small retention smoke:
  - job `2512520`, name `04ret_i119_smoke2`;
  - output dir:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/eval_cpt13b_vanilla_20260610T025159Z/iter_0000119/retention_cli_repair_smoke_v2`;
  - `TASK_GROUP=retention_only`, `LIMIT=1`;
  - initial state: pending on `normal` with reason `Priority`.
- Greek-NLP iter-119 sidecar `2512232` is still running; it completed POS and
  entered summarization in the latest observed tail.
- Re-ran `collect_metrics.py`:
  - updated CSV:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`;
  - `311` rows total at this snapshot;
  - vanilla summary: iters `1-141`, last loss `1.3725`, min loss `1.3586`,
    LR `2.295e-05`, `413` TFLOP/s/GPU, NaN `0`, skipped `0`.
  - TD summary: iters `1-140`, last loss `2.2510`, min loss `2.2505`,
    LR `2.282e-05`, `411` TFLOP/s/GPU, NaN `0`, skipped `0`.

## 2026-06-10T08:46:19Z - Build Parallelism, Retention Smoke 3, Eval-150

- Clarified completed data-build timing from Slurm accounting:
  - Stage A `cpt13b_A_clean_decontam` job `2510621`: completed on `xfer`
    in `00:37:26` with `64` CPUs.
  - Stage B `cpt13b_B_anon_preprocess` job `2511172`: completed on `xfer`
    in `01:12:05` with `64` CPUs.
  - The GreekMMLU decontamination scan ran in parallel with `n_workers=64`
    and finished the scan in `488.9s`; the later filter/rewrite pass took
    `667.4s` over `10,940,329` rows, dropping `295`.
  - Main bottlenecks were large JSONL rewrites plus tokenizer/preprocess passes,
    not a lack of requested CPUs on the completed jobs.
- Retention smoke status:
  - smoke `2512461` failed after exposing sourceless `lm_eval` helper modules.
  - smoke `2512520` progressed further, then exposed missing sourceless
    dependency imports around `accelerate` and downstream scientific packages.
  - kept NumPy resolved from the uenv and expanded the targeted pyc-exposure
    allowlist only for needed packages/modules:
    `lm_eval`, `datasets`, `pyarrow`, `multiprocess`, `pandas`, `dateutil`,
    `pytz`, `accelerate`, `sklearn`, `scipy`, `joblib`, `tabulate`, and
    top-level `threadpoolctl`.
  - patched and synced the retention launcher:
    `/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_eval.sbatch`.
  - verified the patched sbatch with `bash -n`; verified under the uenv that
    `scipy.sparse.csr_matrix`, `sklearn.metrics.roc_curve`, `joblib.Parallel`,
    `threadpoolctl`, and `lm_eval.models.huggingface.HFLM` import.
  - submitted smoke `2512673`, name `04ret_i119_smoke3`, using
    `TASK_GROUP=retention_only`, `LIMIT=1`, output dir
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/eval_cpt13b_vanilla_20260610T025159Z/iter_0000119/retention_cli_repair_smoke_v3`;
    initial state: running on `normal`.
- Eval-150 reached for both arms.
  - Vanilla train latest observed: iter `155`, loss `1.367011E+00`,
    LR `2.468125E-05`, `413.0` TFLOP/s/GPU, NaN `0`, skipped `0`.
  - Vanilla validation at iter `150`:
    `hplt=1.379175E+00`, `openarchives=1.212814E+00`,
    `greek_phd=1.277991E+00`.
  - TD train latest observed: iter `153`, loss `2.205161E+00`,
    LR `2.443375E-05`, `411.4` TFLOP/s/GPU, NaN `0`, skipped `0`.
  - TD validation at iter `150`:
    `hplt=2.622303E+00`, `openarchives=1.891490E+00`,
    `greek_phd=1.974616E+00`.
  - `collect_metrics.py` wrote `344` rows to
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/metrics_20260610T025159Z.csv`.

## 2026-06-10T08:58:47Z - Retention Smoke 4 Progress; Training Still Clean

- Retention smoke `2512673` (`04ret_i119_smoke3`) failed after model load at
  dataset fingerprinting:
  - error: `AttributeError: module 'xxhash' has no attribute 'xxh64'`.
  - cause: `xxhash` existed in the target install as a sourceless package under
    `__pycache__`, so Python saw an empty namespace package while the compiled
    `_xxhash` extension was present but not initialized.
- Narrow repair:
  - added `xxhash` to the allowlisted packages in
    `03_training_experiments/scripts/repair_lm_eval_cli_install.py`;
  - synced the repair script to Clariden;
  - verified under the retention sbatch uenv/PYTHONPATH that:
    `xxhash.xxh64(b"test")`, `datasets.fingerprint.Hasher.hash("test")`, and
    `lm_eval.models.huggingface.HFLM` all import/run.
- Submitted retention smoke `2512712` (`04ret_i119_smoke4`):
  - output dir:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/eval_cpt13b_vanilla_20260610T025159Z/iter_0000119/retention_cli_repair_smoke_v4`;
  - state at latest check: running on `normal`, elapsed `00:05:27`;
  - it has passed the previous failure point: selected tasks, initialized the
    HF model, loaded checkpoint shards, and is generating/mapping task datasets.
- Training status at latest check:
  - vanilla segment 1 running, latest observed iter `160`, loss
    `1.373948E+00`, LR `2.530000E-05`, `413.0` TFLOP/s/GPU, NaN `0`,
    skipped `0`.
  - TD segment 1 running, latest observed iter `158`, loss `2.241182E+00`,
    LR `2.505250E-05`, `411.5` TFLOP/s/GPU, NaN `0`, skipped `0`.
  - chained later segments remain dependency-pending as expected; both watcher
    jobs are running on `xfer`.

## 2026-06-10T09:20:52Z - Corrected Full-Run Scale From 1 Node/Arm to 16 Nodes/Arm

- Found the live training launch was under-scaled:
  - original vanilla segment `2511719`: `1` node, `4` GPUs.
  - original TD segment `2511734`: `1` node, `4` GPUs.
  - total live training GPU count was therefore `8` GPUs across both arms,
    which produced real ETA lines of about `4d15h` per arm.
- Correct full-run scale decision:
  - full 13.5B CPT should run at `16` nodes per arm, `4` GPUs/node:
    `64` GPUs/arm, `128` training GPUs total for the two parallel arms.
  - Keep TP/checkpoint geometry fixed: TP=`2`, PP=`1`, global batch=`1024`;
    the scale change increases data parallelism only.
- Permanent tooling/doc updates:
  - changed `03_training_experiments/configs/common_cpt.env` default
    `NODES` from `1` to `16`.
  - changed `scripts/submit_two_arm_full_run.sh` default `NODES` from `1`
    to `16`.
  - added `scripts/submit_scaled_resume_chain.sh` for same-run-dir resume from
    `OUTPUT_DIR/checkpoints` with `RESUME_TRAINING=1`.
  - updated `LAUNCH_RUNBOOK.md` to document `16` nodes/arm as the full-run
    default and `1` node as diagnostic only.
  - synced all four files to the Clariden repo mirror and verified script syntax
    with `bash -n`.
- Dry-run verification:
  - `submit_scaled_resume_chain.sh vanilla` and `td` both saw
    `latest_checkpointed_iteration.txt = 119` and produced 14-segment
    `16`-node resume chains in dry-run.
- Live correction:
  - cancelled obsolete original jobs:
    - vanilla chain `2511719`-`2511732`;
    - TD chain `2511734`-`2511747`;
    - nonessential retention smoke `2512794`.
  - submitted scaled vanilla resume chain:
    - first job `2512838`, final chain job `2512859`;
    - manifest:
      `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/cpt13b_vanilla_20260610T025159Z_scaled_resume_state/scaled_chain_20260610T092002Z.tsv`.
  - submitted scaled TD resume chain:
    - first job `2512860`, final chain job `2512873`;
    - manifest:
      `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/cpt13b_td_20260610T025159Z_scaled_resume_state/scaled_chain_20260610T092011Z.tsv`.
  - dependency audit passed: segment 2+ jobs have `afterok` dependencies on
    the previous scaled segment; first jobs have no dependency and are waiting
    for scheduling/resources.
- Important consequence:
  - the corrected run resumes from checkpoint `119`; the cancelled 1-node
    progress after iter `119` is intentionally discarded to avoid waiting
    several more hours for the next checkpoint at diagnostic scale.

## 2026-06-10T09:23:51Z - Held Scaled Launch Pending Dataset Order Decision

- User clarified intended ordering for the new-Greek stream: HPLT and
  openarchives should be cleanly separable, not merely present at a 70/30 token
  ratio.
- Current builder implementation was checked:
  - `mix_builder.py` uses a deterministic bucket-preserving token-fair
    interleaver;
  - `mix_13b.sbatch` emits 8 disjoint interleaved shards, then Stage A
    concatenates `bulk_mix_part_*.jsonl`;
  - therefore the already-built dataset satisfies the token ratio, but not a
    strict HPLT block followed by openarchives block ordering.
- Safety action:
  - held the first scaled 16-node jobs (`2512838`, `2512860`) before they
    started;
  - downstream scaled segments remain dependency-pending.
- Artifact inspection:
  - `bulk_mix.jsonl`, `bulk_mix_clean.jsonl`, and `bulk_mix_decontam.jsonl`
    preserve top-level `source` and `doc_id`;
  - `bulk_mix_final.jsonl` preserves provenance as `metadata.source` and `id`.
- Practical implication:
  - if strict ordering is required, we can stream-repartition the existing
    cleaned/decontaminated/anonymized final JSONL by source and retokenize both
    Megatron data prefixes;
  - the Megatron `.bin/.idx` files cannot be safely reordered in place;
  - if the strict order is part of the scientific contract, the checkpoint-119
    resumes should not be used, because those checkpoints already consumed the
    old interleaved stream.

## 2026-06-10T09:31:36Z - Replay-Fixed Ordering Plan Implemented In Tooling

- User clarified: replay must stay the same.
- Final ordering policy:
  - preserve every non-new-Greek row (multilingual/code/math/Greek-replay
    replay) at its original line position;
  - reorder only the slots occupied by the two new-Greek sources;
  - the filtered new-Greek subsequence becomes HPLT first, then openarchives.
- Added local tooling/docs for this policy:
  - `03_training_experiments/dataset_build/reorder_new_greek_slots.py`;
  - `03_training_experiments/dataset_build/stageC_order_replay_fixed_preprocess.sbatch`;
  - updated the arm configs to launch from
    `bulk_mix_ordered_replay_fixed_{base,ext}_text_document`;
  - updated the artifact gate to require the ordered JSONL, manifest, and
    ordered Megatron binaries.
- Verified locally:
  - `python3 -m py_compile reorder_new_greek_slots.py`;
  - `bash -n stageC_order_replay_fixed_preprocess.sbatch`;
  - `bash -n scripts/gate_cpt2arm_artifacts.sh`;
  - tiny fixture test confirmed replay rows stay at the same line numbers while
    new-Greek slots become HPLT...openarchives.
- Canceled stale jobs tied to the interleaved-stream checkpoint/run:
  - scaled resume chain jobs `2512838`, `2512840`, `2512842`, `2512844`,
    `2512846`, `2512848`, `2512851`, `2512853`-`2512861`, `2512862`-`2512873`;
  - old xfer watchers `2512336` and `2511748`.
- Next action:
  - sync Stage-C tooling/configs to Clariden;
  - submit Stage C on `xfer`;
  - run the artifact gate;
  - relaunch both arms fresh from init checkpoints, not from checkpoint `119`.

## 2026-06-10T09:36Z - Stage C Submitted On CPU-Only `xfer`

- Synced Stage-C tooling/config/docs/run log to the Clariden repo mirror.
- Remote login-safe checks passed:
  - `bash -n stageC_order_replay_fixed_preprocess.sbatch`;
  - `bash -n scripts/gate_cpt2arm_artifacts.sh`.
- Note: the xfer build Python is x86-only and cannot be run on ARM login nodes;
  the Python syntax check was done locally, and the real interpreter check will
  occur inside the `xfer` job.
- Submitted CPU-only Stage C:
  - job id: `2512949`;
  - partition: `xfer`;
  - script:
    `$SC/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt/03_training_experiments/dataset_build/stageC_order_replay_fixed_preprocess.sbatch`.
- Expected outputs:
  - `$SC/cpt_corpus/cpt_2arm_13b/bulk_mix_ordered_replay_fixed_final.jsonl`;
  - `$SC/cpt_corpus/cpt_2arm_13b/bulk_mix_ordered_replay_fixed_manifest.json`;
  - `$SC/cpt_corpus/cpt_2arm_13b/megatron/bulk_mix_ordered_replay_fixed_base_text_document.{bin,idx}`;
  - `$SC/cpt_corpus/cpt_2arm_13b/megatron/bulk_mix_ordered_replay_fixed_ext_text_document.{bin,idx}`.

## 2026-06-10T11:27:09Z - Ordered Dataset Gate Passed; Fresh Two-Arm Launch Submitted

- Stage C completed successfully:
  - job id `2512949`;
  - state `COMPLETED`, exit code `0:0`, elapsed `01:44:58`, partition `xfer`.
- Ordered dataset artifacts:
  - `bulk_mix_ordered_replay_fixed_final.jsonl` = `90134865928` bytes;
  - `bulk_mix_ordered_replay_fixed_manifest.json` = `2777` bytes;
  - base training binary = `77880489752` bytes, idx = `218800722` bytes;
  - ext training binary = `54046553500` bytes, idx = `218800722` bytes.
- Ordering gate passed:
  - `non_new_positions_preserved = true`;
  - `ordered_violation_count = 0`;
  - `last_hplt_new_greek_slot = 7852068`;
  - `first_openarchives_new_greek_slot = 7852069`.
- Full prelaunch artifact gate passed immediately before live launch:
  - config invariants, tokenizer invariants, init checkpoints, TE guard,
    extra-valid TB patch, held-out validation binaries, ordered manifest, and
    ordered base/ext training binaries all checked.
- Fresh launch stamp: `20260610T112618Z`.
- Vanilla fresh chain:
  - run tag `cpt13b_vanilla_20260610T112618Z`;
  - segment jobs `2513524`-`2513537`;
  - segment 1 `resume=0`, init checkpoint =
    `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/cpt_2arm_13b/vanilla_base131072/megatron_tp2_r17patched`;
  - watcher job `2513538`.
- TD fresh chain:
  - run tag `cpt13b_td_20260610T112618Z`;
  - segment jobs `2513539`-`2513553`;
  - segment 1 `resume=0`, init checkpoint =
    `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/cpt_2arm_13b/modern_greek_td148480/megatron_tp2_r17patched`;
  - watcher job `2513554`.
- Queue state at launch check:
  - first training jobs are pending (`2513524` reason `Resources`,
    `2513539` reason `Priority`);
  - watcher jobs are running on `xfer`.
- Next action: monitor until both segment-1 jobs start, verify logs show the
  ordered data prefixes and clean first iterations, then collect metrics.

## 2026-06-10T11:50:41Z - Dataset Prep Complete; First Training Launch Failed In NCCL/OFI

- Dataset preparation is complete. The ordered, replay-fixed artifacts finished
  in Stage C and passed the full artifact gate before launch.
- First fresh two-arm launch failed before any training iterations:
  - vanilla segment 1 job `2513524`: `FAILED`, exit `6:0`, elapsed `00:02:14`;
  - TD segment 1 job `2513539`: `FAILED`, exit `15:0`, elapsed `00:02:03`.
- The logs reached correct setup before the failure:
  - ordered base/ext data prefixes loaded;
  - all three held-out validation sets built;
  - init checkpoints loaded at iteration 0;
  - Goldfish, AdEMAMix, WSD, RoPE, and extra-valid settings were present.
- Failure signature is a distributed runtime/fabric error, not a dataset-build
  error:
  - `NCCL Error 2: unhandled system error`;
  - `NET/OFI Request ... completed with error. RC: 5. Error: 16 (NO_SPACE)`;
  - failing process group: `DATA_PARALLEL_GROUP_WITH_CP`.
- Canceled the stale dependent training jobs and xfer watcher jobs:
  - vanilla pending jobs `2513525`-`2513537`, watcher `2513538`;
  - TD pending jobs `2513540`-`2513553`, watcher `2513554`.
- Next action:
  - diagnose the NCCL/OFI `NO_SPACE` launch failure against Clariden runbooks
    and prior working configs;
  - relaunch from the init checkpoints only after a launch-shape fix or
    diagnostic plan is selected;
  - verify first iteration/loss lines before calling training underway.

## 2026-06-10T11:55:44Z - Runtime Diagnosis Patch Prepared Locally

- Root-cause evidence:
  - the failed 16-node jobs used direct multi-task Slurm launch
    (`LAUNCH_MODE=slurm`);
  - the established bakeoff config already warned that direct multi-node Slurm
    launch had hit inter-node NCCL errors before iteration 1;
  - an older 2-node Megatron TD smoke with `torchrun` also hit
    `NET/OFI ... NO_SPACE`, so launch mode alone is not sufficient;
  - CSCS Alps NCCL/uenv guidance requires forcing the libfabric plugin and
    setting the Alps `FI_CXI_*` transport knobs. Our trainer had only a subset.
- Local code changes prepared:
  - `03_training_experiments/configs/common_cpt.env`: auto-select
    `LAUNCH_MODE=torchrun` when `NODES>1`, else `slurm`.
  - `scripts/submit_two_arm_full_run.sh`: same auto-selection inside the child
    submitter; in `torchrun` mode request `--ntasks-per-node=1`.
  - `scripts/submit_scaled_resume_chain.sh`: same child submitter behavior for
    resume chains.
  - `init_bakeoff/bakeoff_training/bakeoff_train.sbatch`: add the
    CSCS-recommended NCCL/libfabric env block:
    `NCCL_NET="AWS Libfabric"`, `NCCL_NET_GDR_LEVEL=PHB`,
    `NCCL_CROSS_NIC=1`, `NCCL_PROTO=^LL128`,
    `FI_CXI_DEFAULT_CQ_SIZE=131072`, `FI_CXI_DEFAULT_TX_SIZE=16384`,
    `FI_CXI_DISABLE_HOST_REGISTER=1`, `FI_CXI_RX_MATCH_MODE=software`,
    `FI_MR_CACHE_MONITOR=userfaultfd`, plus the eager-message workaround
    `FI_CXI_RDZV_{GET_MIN,THRESHOLD,EAGER_SIZE}=0`.
  - `scripts/gate_cpt2arm_artifacts.sh`: require `NODES=16`,
    `LAUNCH_MODE=torchrun`, and the NCCL/libfabric block in the deployed
    trainer.
  - `HANDOFF.md` and `LAUNCH_RUNBOOK.md`: document the runtime invariant.
- Local verification:
  - `bash -n` passed for common config, both submitters, gate, and trainer;
  - local dry-run with path overrides printed `LAUNCH_MODE=torchrun` and
    `sbatch_ntasks_per_node=1` for both arms.
- Next action:
  - sync the patch to Clariden;
  - run remote syntax checks and artifact gate;
  - submit a short 16-node smoke from init checkpoint before relaunching the
    full two-arm chains.

## 2026-06-10T11:58:02Z - Runtime Patch Synced; Remote Gate Passed

- Synced to Clariden:
  - patched trainer `bakeoff_train.sbatch`;
  - common config, both submitters, artifact gate, launch runbook, handoff,
    and this run log.
- Corrected a transient rsync placement mistake:
  - removed stray duplicate files from the `03_training_experiments/` root;
  - verified the real copies live under `configs/` and `scripts/`.
- Remote syntax checks passed for:
  - `configs/common_cpt.env`;
  - `scripts/submit_two_arm_full_run.sh`;
  - `scripts/submit_scaled_resume_chain.sh`;
  - `scripts/gate_cpt2arm_artifacts.sh`;
  - `bakeoff_train.sbatch`.
- Remote artifact gate result: `ARTIFACT GATE PASSED`.
  - It now verifies `NODES=16`, `LAUNCH_MODE=torchrun`, the CSCS Alps
    NCCL/libfabric runtime block, ordered replay-fixed data, init checkpoints,
    TE guard, extra-valid patch, and all training/validation binaries.
- Next action: submit a short 16-node smoke from the init checkpoint and require
  real iteration/loss lines before the full relaunch.

## 2026-06-10T11:58:34Z - Submitted 16-Node Vanilla Runtime Smoke

- Submitted a short full-scale runtime smoke:
  - job id `2513687`;
  - job name `cpt13b_smoke16_van`;
  - partition `normal`;
  - shape `16` nodes, `64` GPUs total;
  - launch mode `torchrun`, `--ntasks-per-node=1`;
  - `EXIT_INTERVAL=2`, `DISABLE_SAVE=1`, `EVAL_INTERVAL=999999`;
  - init checkpoint:
    `$SC/init_checkpoints/cpt_2arm_13b/vanilla_base131072/megatron_tp2_r17patched`.
- Output directory:
  `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_20260610T115834Z`.
- Slurm state after submission:
  - `PENDING`;
  - reason changed from `Resources` to `Priority`;
  - `squeue --start` predicts `2026-06-10 14:26:46 CEST`.
- This is a runtime smoke only. It does not alter dataset/checkpoint artifacts
  and it will not save a training checkpoint.

## 2026-06-10T12:13:50Z - 16-Node Vanilla Runtime Smoke Failed

- Smoke job `2513687` failed:
  - state `FAILED`;
  - exit `15:0`;
  - elapsed `00:01:52`;
  - allocated `16` nodes / `64` GPUs.
- Log evidence:
  - trainer printed the patched runtime block:
    `NCCL_NET=AWS Libfabric`, `NCCL_NET_GDR_LEVEL=PHB`,
    `NCCL_CROSS_NIC=1`, `NCCL_PROTO=^LL128`,
    `FI_CXI_DEFAULT_CQ_SIZE=131072`, `FI_CXI_DEFAULT_TX_SIZE=16384`,
    `FI_CXI_RX_MATCH_MODE=software`, `FI_MR_CACHE_MONITOR=userfaultfd`;
  - distributed setup was `WORLD_SIZE=64`, `LAUNCH_MODE=torchrun`;
  - checkpoint loaded successfully at iteration 0;
  - failure occurred before any training iteration:
    `NET/OFI Request ... Error: 16 (NO_SPACE)` and
    `NCCL Error 2: unhandled system error`.
- Conclusion:
  - ordered dataset, tokenizer, init checkpoint, and hyperparameters are not
    implicated by this failure;
  - the active blocker is multi-node Megatron/NCCL/OFI runtime on Clariden.
- Next diagnostic:
  - submit a 2-node Megatron smoke with `NCCL_DEBUG=INFO` and
    `MPICH_GPU_SUPPORT_ENABLED=0` to determine whether any multi-node Megatron
    shape works with this trainer/uenv stack after the runtime patch.

## 2026-06-10T12:19:57Z - 2-Node Megatron/NCCL Diagnostic Also Failed

- Submitted 2-node debug smoke:
  - job id `2513773`;
  - job name `cpt13b_smoke2_van_dbg`;
  - partition `debug`;
  - nodes `nid006130,nid006148`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_nccldebug_20260610T121427Z`.
- Result:
  - state `FAILED`;
  - exit `1:0`;
  - elapsed `00:02:23`.
- Positive evidence:
  - `NCCL_DEBUG=INFO` confirmed the plugin is present and used:
    `NET/Plugin: Loaded net plugin AWS Libfabric (v11)` and
    `Using network AWS Libfabric` on both nodes;
  - checkpoint loaded successfully at iteration 0.
- Failure:
  - before iteration 1, same `NET/OFI Request ... Error: 16 (NO_SPACE)`;
  - `MPICH_GPU_SUPPORT_ENABLED=0` did not avoid it.
- Interpretation:
  - not a missing `aws-ofi-nccl` plugin problem;
  - not a 16-node-only scale problem;
  - the blocker is at least 2-node Megatron communication on this stack,
    likely in the initial data-parallel/distributed-optimizer collective.
- Next diagnostic:
  - try CSCS-documented point-to-point knob
    `NCCL_NCHANNELS_PER_NET_PEER=4` on the 2-node smoke.

## 2026-06-10T12:24:04Z - Peer-Channel Diagnostic Failed; Multi-Node Blocker Confirmed

- Submitted 2-node diagnostic with:
  - `NCCL_NCHANNELS_PER_NET_PEER=4`;
  - `NCCL_DEBUG=INFO`;
  - `MPICH_GPU_SUPPORT_ENABLED=0`;
  - otherwise same 2-node vanilla smoke shape.
- Job:
  - id `2513801`;
  - name `cpt13b_smoke2_peer4`;
  - partition `debug`;
  - nodes `nid006130,nid006139`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_ncclpeer4_20260610T122028Z`.
- Result:
  - state `FAILED`;
  - exit `1:0`;
  - elapsed `00:02:22`.
- Evidence:
  - AWS Libfabric plugin loaded and was used on both nodes;
  - checkpoint loaded successfully at iteration 0;
  - same pre-iteration `NET/OFI Request ... Error: 16 (NO_SPACE)` failure.
- Decision:
  - do not relaunch the full 16-node chains yet;
  - do not spend more jobs on undocumented NCCL knob guessing;
  - the next reasonable paths are:
    1. escalate the minimal reproducer to CSCS/SwissAI runtime support;
    2. inspect Megatron distributed-optimizer initialization specifically
       (diagnostic: optional no-distributed-optimizer/overlap smoke, not a
       production recipe);
    3. fall back to the proven 1-node-per-arm chain only if wall time is
       acceptable or no runtime fix is available.

## 2026-06-10T12:25:50Z - Prepared Distributed-Optimizer Isolation Switch

- Added diagnostic-only env switches in `bakeoff_train.sbatch`:
  - `USE_DISTRIBUTED_OPTIMIZER=0` omits `--use-distributed-optimizer`;
  - `USE_COMM_OVERLAP=0` omits `--overlap-grad-reduce` and
    `--overlap-param-gather`.
- Production defaults are unchanged:
  - both switches default to `1`;
  - normal full-run launches still use distributed optimizer and overlap.
- Purpose:
  - isolate whether the 2-node `NO_SPACE` failure is tied to the
    distributed-optimizer/overlap communication path, not to data/checkpoints.

## 2026-06-10T12:30:58Z - No-Distributed-Optimizer Diagnostic Failed

- Submitted 2-node diagnostic with:
  - `USE_DISTRIBUTED_OPTIMIZER=0`;
  - `USE_COMM_OVERLAP=0`;
  - `NCCL_DEBUG=INFO`;
  - `MPICH_GPU_SUPPORT_ENABLED=0`.
- Job:
  - id `2513824`;
  - name `cpt13b_smoke2_nodopt`;
  - partition `debug`;
  - nodes `nid006273,nid006302`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_no_distopt_20260610T122645Z`.
- Result:
  - state `FAILED`;
  - exit `15:0`;
  - elapsed `00:03:43`.
- Evidence:
  - trainer confirmed `distributed flags: use_distributed_optimizer=0 use_comm_overlap=0`;
  - AWS Libfabric plugin loaded and was used on both nodes;
  - checkpoint loaded successfully at iteration 0;
  - same pre-iteration `NET/OFI Request ... Error: 16 (NO_SPACE)` failure.
- Interpretation:
  - the immediate 2-node failure is broader than distributed optimizer or
    Megatron overlap flags.
- Next diagnostic:
  - compare uenv image: run the same no-save 2-node smoke under
    `pytorch/v2.6.0:v1`, because the original toy multi-node proof used v2.6
    while the CPT trainer uses v2.9.1:v2 for checkpoint-save support.

## 2026-06-10T12:35:34Z - v2.6 uenv Comparison Inconclusive

- Submitted 2-node diagnostic under `UENV_IMAGE=pytorch/v2.6.0:v1`:
  - job id `2513857`;
  - job name `cpt13b_smoke2_u260`;
  - partition `debug`;
  - nodes `nid006273,nid006302`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_uenv260_20260610T123131Z`.
- Result:
  - state `FAILED`;
  - exit `1:0`;
  - elapsed `00:00:37`.
- Failure occurred before the NCCL path:
  - `ImportError: cannot import name 'SerializationFormat' from torch.distributed.checkpoint.filesystem`.
- Interpretation:
  - v2.6 cannot currently answer the NCCL question without a Megatron/env
    compatibility patch;
  - this is the known reason the CPT trainer had been pinned to
    `pytorch/v2.9.1:v2` for checkpoint support.

## 2026-06-10T12:40:50Z - Pure PyTorch All-Reduce Succeeds; Failure Is Megatron-Specific

- Added reusable diagnostic scripts:
  - `03_training_experiments/scripts/torchrun_allreduce_smoke.py`;
  - `03_training_experiments/scripts/submit_torchrun_allreduce_smoke.sbatch`.
- Submitted 2-node pure PyTorch/NCCL all-reduce under `pytorch/v2.9.1:v2` with
  the same AWS Libfabric environment:
  - job id `2513892`;
  - job name `allreduce_u291`;
  - partition `debug`;
  - nodes `nid006910,nid006931`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_allreduce_u291_20260610T123750Z`.
- Result:
  - state `COMPLETED`;
  - exit `0:0`;
  - elapsed `00:00:23`;
  - log confirmed `Using network AWS Libfabric`;
  - `allreduce_smoke ... world_size=8 value=28.0 expected=28.0 torch=2.9.1 cuda=12.9`.
- Narrowed Megatron failure phase from the no-distopt run:
  - Megatron initializes TP=2 / PP=1;
  - model, optimizer, and LR scheduler build successfully;
  - init checkpoint loads successfully at iteration 0;
  - train + all three extra validation datasets build successfully;
  - log reaches `training ...` and `[before the start of training step]`;
  - first training step fails in `DATA_PARALLEL_GROUP_WITH_CP` with
    `NET/OFI ... NO_SPACE`.
- Interpretation:
  - Clariden + `pytorch/v2.9.1:v2` + AWS Libfabric can run basic 2-node NCCL;
  - the blocker is a Megatron/SuisseAI training-step communication pattern, not
    the general inter-node NCCL runtime.

## 2026-06-10T12:44:10Z - Pure PyTorch Megatron-Style Subgroups Also Succeed

- Added a second pure PyTorch diagnostic:
  - `03_training_experiments/scripts/torchrun_megatron_group_smoke.py`.
- Submitted 2-node NCCL subgroup smoke under the same `pytorch/v2.9.1:v2`
  uenv and AWS Libfabric environment:
  - job id `2513910`;
  - job name `megatron_groups_u291`;
  - partition `debug`;
  - nodes `nid007436,nid007440`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_megatron_groups_u291_20260610T124332Z`.
- Result:
  - state `COMPLETED`;
  - exit `0:0`;
  - elapsed `00:00:25`;
  - log confirmed `Using network AWS Libfabric`;
  - subgroup layout matched the relevant 2-node Megatron shape:
    TP groups `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`;
    DP-with-CP-like groups `[0,2,4,6]`, `[1,3,5,7]`;
  - tiny and 1M-float all-reduces passed for 8 iterations.
- Interpretation:
  - the failure is not caused merely by creating TP and DP-with-CP-shaped NCCL
    groups;
  - it is specific to the full Megatron/SuisseAI training-step communication
    path.

## 2026-06-10T12:50:00Z - CPU-Affinity Diagnostic Failed; Affinity Is Not Root Cause

- Submitted the 2-node no-distributed-optimizer/no-overlap Megatron smoke with
  wider CPU allocation:
  - job id `2513920`;
  - job name `cpt13b_smoke2_cpu288`;
  - partition `debug`;
  - nodes `nid006033,nid006045`;
  - `--cpus-per-task=288`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_nodopt_cpu288_20260610T124611Z`.
- Result:
  - state `FAILED`;
  - exit `1:0`;
  - elapsed `00:03:51`.
- Evidence:
  - CPU affinity was correct for all four local workers
    (`GPU 0 0-71`, `GPU 1 72-143`, `GPU 2 144-215`, `GPU 3 216-287`);
  - trainer confirmed `use_distributed_optimizer=0 use_comm_overlap=0`;
  - model/optimizer/LR scheduler setup, checkpoint load, train dataset, and
    extra validation datasets completed;
  - log reached `training ...` and `[before the start of training step]`;
  - the same `DATA_PARALLEL_GROUP_WITH_CP` / `NET/OFI ... NO_SPACE` failure
    occurred.
- Interpretation:
  - the one-task-per-node torchrun shape needs enough CPUs for four workers,
    but fixing affinity does not resolve the NCCL/OFI failure.

## 2026-06-10T12:56:00Z - Prepared Mock-Data Megatron Diagnostic Switch

- Added diagnostic-only `USE_MOCK_DATA` support to `bakeoff_train.sbatch`:
  - default remains `USE_MOCK_DATA=0`, so production launches still use the
    ordered CPT dataset and all held-out validation sets;
  - `USE_MOCK_DATA=1` adds Megatron `--mock-data`, omits the real `--data-path`,
    and disables extra validation for that diagnostic run;
  - `run_metadata.json` now records `mock_data`.
- Purpose:
  - isolate whether the failing full Megatron first training step requires the
    real indexed dataset/dataloaders, or whether the model/runtime collective
    fails even with synthetic data.

## 2026-06-10T13:00:00Z - Mock-Data Megatron Still Fails; Dataset Cleared

- Submitted 2-node mock-data Megatron diagnostic:
  - job id `2513951`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_20260610T125615Z`;
  - `USE_MOCK_DATA=1`, `ENABLE_EXTRA_VALID=0`,
    `USE_DISTRIBUTED_OPTIMIZER=0`, `USE_COMM_OVERLAP=0`.
- Result:
  - state `FAILED`, exit `1:0`, elapsed `00:03:43`.
- Evidence:
  - `mock_data=True`, `extra_valid_data_path=None`;
  - checkpoint loaded at iteration 0;
  - log reached `[before the start of training step]`;
  - same `DATA_PARALLEL_GROUP_WITH_CP` / `NET/OFI ... NO_SPACE` failure.
- Interpretation:
  - real CPT indexed data, dataloaders, and extra validation are not required
    to trigger the failure.

## 2026-06-10T13:04:00Z - Production-Comm Mock and Peer4 Mock Also Fail

- Production-communication mock:
  - job id `2513980`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_prodcomm_20260610T130129Z`;
  - `USE_DISTRIBUTED_OPTIMIZER=1`, `USE_COMM_OVERLAP=1`;
  - DDP config reported `bucket_size=40000000`;
  - failed with `NET/OFI ... NO_SPACE` before iteration 1.
- Same shape with CSCS-recommended peer-channel knob:
  - job id `2513992`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_prodcomm_peer4_20260610T130506Z`;
  - `NCCL_NCHANNELS_PER_NET_PEER=4`;
  - failed with the same `NO_SPACE`.
- Interpretation:
  - the earlier `NCCL_NCHANNELS_PER_NET_PEER=4` result was not explained by
    the old CPU allocation issue; it does not resolve the Megatron first-step
    failure.

## 2026-06-10T13:12:00Z - Large Pure-PyTorch Collectives Pass

- Extended `torchrun_megatron_group_smoke.py` to parameterize payload size,
  dtype, and collective type. Defaults remain the previous 1M float32
  all-reduce smoke.
- 40M bfloat16 all-reduce passed:
  - job id `2514008`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_megatron_groups_40m_bf16_20260610T130856Z`;
  - state `COMPLETED`, elapsed `00:00:25`.
- 40M bfloat16 reduce-scatter passed:
  - job id `2514023`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_megatron_groups_rsag_40m_bf16_20260610T131049Z`;
  - state `COMPLETED`, elapsed `00:00:25`.
  - Note: Slurm comma-splitting meant this run only carried
    `SMOKE_COLLECTIVES=reduce_scatter`; all-gather was run separately.
- 40M bfloat16 all-gather passed:
  - job id `2514024`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_megatron_groups_ag_40m_bf16_20260610T131138Z`;
  - state `COMPLETED`, elapsed `00:00:27`.
- Interpretation:
  - the failure is not generic 2-node NCCL with Megatron-shaped TP/DP groups;
  - it is not generic 40M bfloat16 DP all-reduce, reduce-scatter, or
    all-gather under AWS Libfabric.

## 2026-06-10T13:25:00Z - Goldfish, Bucket Size, and No-Overlap Cleared

- Made `LOSS_OBJECTIVE` overridable in `common_cpt.env` while preserving the
  default `goldfish` production setting.
- Mock-data NTP diagnostic:
  - job id `2514035`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_ntp_20260610T131312Z`;
  - `loss objective: ntp`;
  - failed with the same `NET/OFI ... NO_SPACE`;
  - first failing collective reported
    `OpType=COALESCED, NumelIn=268435456, NumelOut=67108864`.
- Added optional `DDP_BUCKET_SIZE` pass-through to `bakeoff_train.sbatch`;
  default remains Megatron's existing value.
- Mock-data `DDP_BUCKET_SIZE=5000000` diagnostic:
  - job id `2514047`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_bucket5m_20260610T131655Z`;
  - DDP config reported `bucket_size=5000000`;
  - failed with the same `NO_SPACE`.
- Mock-data distributed-optimizer/no-overlap diagnostic:
  - job id `2514051`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_distopt_nooverlap_20260610T132009Z`;
  - `USE_DISTRIBUTED_OPTIMIZER=1`, `USE_COMM_OVERLAP=0`;
  - failed in log with the same `NO_SPACE`; job was cancelled during watchdog
    cleanup to release debug GPUs;
  - first failing collective reported
    `OpType=COALESCED, NumelIn=4026810368, NumelOut=1006702592`.
- Interpretation:
  - Goldfish is not the cause;
  - simply shrinking DDP buckets to 5M is not enough;
  - disabling Megatron comm overlap while keeping distributed optimizer is not
    enough;
  - the remaining blocker is a full Megatron/SuisseAI DDP training-step
    collective pattern on the Clariden `pytorch/v2.9.1:v2` stack.

## 2026-06-10T13:40:00Z - Exact-Size Pure Collectives Pass; Direct Slurm Still Fails

- Ran pure PyTorch controls matching the NTP Megatron failure's reported
  coalesced collective size:
  - failing Megatron NTP collective had
    `NumelIn=268435456, NumelOut=67108864`.
- Exact-size reduce-scatter passed:
  - job id `2514102`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_megatron_groups_rs_67108864_bf16_20260610T133611Z`;
  - `SMOKE_PAYLOAD_ELEMENTS=67108864`, `SMOKE_DTYPE=bfloat16`,
    `SMOKE_COLLECTIVES=reduce_scatter`;
  - state `COMPLETED`, elapsed `00:00:26`.
- Exact-size all-gather passed:
  - job id `2514107`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/torchrun_megatron_groups_ag_67108864_bf16_20260610T133703Z`;
  - `SMOKE_PAYLOAD_ELEMENTS=67108864`, `SMOKE_DTYPE=bfloat16`,
    `SMOKE_COLLECTIVES=all_gather`;
  - state `COMPLETED`, elapsed `00:00:25`.
- Tested direct Slurm rank launch (not torchrun):
  - job id `2514116`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_slurm_20260610T133829Z`;
  - 2 nodes, `--ntasks-per-node=4`, `--cpus-per-task=72`,
    `LAUNCH_MODE=slurm`, `USE_MOCK_DATA=1`;
  - DDP config: distributed optimizer + overlap, bucket size 40M;
  - failed before iteration 1 with the same
    `DATA_PARALLEL_GROUP_WITH_CP` / `NET/OFI ... NO_SPACE`.
- Interpretation:
  - torchrun is not the root cause;
  - pure NCCL at the exact NTP failure size works;
  - the blocker remains full Megatron/SuisseAI first-step DDP collective
    behavior.

## 2026-06-10T13:50:00Z - Current-Recipe 1-Node Real-Data Smoke Passes

- Submitted a single-node, real-data vanilla smoke with the current CPT recipe:
  - job id `2514129`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke1_vanilla_realdata_20260610T134150Z`;
  - 1 node / 4 GPUs, `LAUNCH_MODE=slurm`;
  - real ordered base-tokenized data:
    `bulk_mix_ordered_replay_fixed_base_text_document`;
  - `USE_MOCK_DATA=0`, `LOSS_OBJECTIVE=goldfish`, default distributed optimizer
    + overlap, `DISABLE_SAVE=1`, `EXIT_INTERVAL=2`, `ENABLE_EXTRA_VALID=0`.
- Result:
  - state `COMPLETED`, exit `0:0`, elapsed `00:05:49`.
- Evidence:
  - checkpoint loaded successfully at iteration 0;
  - reached real training with no NaNs/skips;
  - iteration 1: `lm loss: 1.476209E+00`, grad norm `4.252`;
  - iteration 2: `lm loss: 1.472929E+00`, grad norm `3.955`;
  - exited cleanly at iteration 2.
- Interpretation:
  - the current dataset, checkpoint, trainer, Goldfish objective, optimizer, and
    single-node communication path are healthy;
  - the failure boundary is specifically inter-node full-Megatron DDP
    communication on Clariden.

## 2026-06-10T14:04:52Z - CXI Request-Buffer Diagnostic Did Not Fix Multi-Node Megatron

- Submitted 2-node mock-data Megatron smoke:
  - job id `2514355`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_cxi_reqbuf_20260610T140452Z`;
  - shape: `NODES=2`, `GPUS_PER_NODE=4`, `LAUNCH_MODE=torchrun`,
    `USE_MOCK_DATA=1`, `ENABLE_EXTRA_VALID=0`, `DISABLE_SAVE=1`,
    `EXIT_INTERVAL=2`;
  - diagnostic env:
    `FI_CXI_REQ_BUF_SIZE=33554432`,
    `FI_CXI_REQ_BUF_MIN_POSTED=8`,
    `FI_CXI_DEFAULT_RX_SIZE=16384`,
    `FI_LOG_LEVEL=warn`, `FI_LOG_PROV=cxi`.
- Result: `FAILED`, exit `1:0`, elapsed `00:02:34`.
- Interpretation:
  - increasing CXI request buffers to 32 MiB was not sufficient;
  - the run still failed in the same first-step NCCL system-error window;
  - this weakens the simple "missing request buffer env only" hypothesis.

## 2026-06-10T14:09:51Z - Hybrid/Default-RDZV Diagnostic Still Hits `NO_SPACE`

- Patched and rsynced the deployed `bakeoff_train.sbatch` so that:
  - current defaults stay unchanged for ordinary launches;
  - `FI_CXI_RDZV_*` can be unset with `FI_CXI_USE_DEFAULT_RDZV=1`;
  - optional CXI request-buffer/log vars are echoed in job output for audit.
- Submitted 2-node mock-data Megatron smoke:
  - job id `2514376`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_cxi_hybrid_default_rdzv_20260610T140951Z`;
  - effective printed env included
    `FI_CXI_RX_MATCH_MODE=hybrid` and
    `FI_CXI_RDZV_{GET_MIN,THRESHOLD,EAGER_SIZE}=<unset>`.
- Result: `FAILED`, exit `1:0`, elapsed `00:02:22`.
- Key evidence:
  - NCCL selected AWS Libfabric/CXI with `SENDRECV`;
  - failure remained `NET/OFI ... Error: 16 (NO_SPACE)`;
  - failing collective:
    `OpType=COALESCED, NumelIn=268435456, NumelOut=67108864`;
  - representative OFI request:
    `Request: { dev: 2, size: 4, state: CREATED, direction: RECV }`.
- Additional environment check:
  - `uenv run pytorch/v2.9.1:v2 --view=default` overrides
    `FI_CXI_RX_MATCH_MODE` and `FI_CXI_RDZV_*` to its own defaults:
    `hybrid`, RDZV zeros, and `FI_CXI_RDZV_PROTO=alt_read`;
  - therefore the review-agent claim that the trainer effectively ran with
    `FI_CXI_RX_MATCH_MODE=software` is not correct for the actual process
    inside `uenv`.

## 2026-06-10T14:14:43Z - Forced OFI RDMA Protocol Is Not Viable In This Uenv

- Submitted 2-node mock-data Megatron smoke:
  - job id `2514396`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_ofi_rdma_20260610T141443Z`;
  - diagnostic env included
    `OFI_NCCL_PROTOCOL=RDMA`,
    `FI_CXI_RX_MATCH_MODE=hybrid`,
    `FI_CXI_RDZV_PROTO=alt_read`,
    `NCCL_NCHANNELS_PER_NET_PEER=4`,
    `NCCL_NET_GDR_C2C=1`,
    `NCCL_NET_GDR_READ=1`.
- Result: `FAILED`, exit `15:0`, elapsed `00:02:04`.
- Key evidence:
  - NCCL confirmed `Using transport protocol RDMA (user set)`;
  - channels used `NET/AWS Libfabric/.../GDRDMA`;
  - failure occurred earlier than the SENDRECV `NO_SPACE` failure:
    `fi_writedata failed; RC: -38, Error: Function not implemented`;
  - RDMA is therefore not a usable workaround in this environment as-is.
- Current narrowed interpretation:
  - the viable OFI protocol path is still `SENDRECV`;
  - the failure is receive-side/resource related under Megatron's first
    coalesced data-parallel collective;
  - installed libfabric strings explicitly mention two relevant documented
    mitigations:
    increasing `FI_CXI_REQ_BUF_SIZE` for "request list full" and increasing
    `FI_CXI_OFLOW_BUF_SIZE` for "overflow no match";
  - next cheap test, once CSCS access is restored, should try larger request
    buffers plus overflow buffers under the uenv SENDRECV path before
    escalating the updated evidence to CSCS/SwissAI runtime support.

## 2026-06-10T14:20:53Z - CSCS Auth Expired During Diagnostics

- Local CSCS certificate expired at `2026-06-10T17:20:53+03:00`
  (`2026-06-10T14:20:53Z`).
- Direct `cscs-key --headless sign` produced device code `BKMB-YLWC`, but the
  token poll timed out.
- Direct `home` reachability to
  `https://auth.cscs.ch/auth/realms/cscs/.well-known/openid-configuration`
  also timed out.
- MacBook fallback was unavailable from `home` at this moment:
  `ssh macbook` returned `No route to host`.
- Work is temporarily blocked until CSCS auth is refreshed or the network path
  to CSCS auth returns.

## 2026-06-10T15:51:00Z - Socket/HSN Fallback Passes 2- and 4-Node Megatron Smokes

- CSCS auth was refreshed and `ssh clariden` works again.
- Patched and rsynced the deployed `bakeoff_train.sbatch` so that NCCL/CXI
  environment variables are re-applied inside the `uenv run ... bash -c`
  process. This matters because `uenv run pytorch/v2.9.1:v2 --view=default`
  rewrites some pre-uenv exports.
- Additional AWS Libfabric/CXI diagnostics after that patch still failed:
  - `2514693`: 2-node software-match-mode smoke,
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_vanilla_mockdata_uenv_inside_software_20260610T150431Z`,
    failed with `NET/OFI ... NO_SPACE`;
  - `2514729`: 2-node software mode with
    `FI_CXI_REQ_BUF_SIZE=33554432`, `FI_CXI_REQ_BUF_MIN_POSTED=8`,
    `FI_CXI_DEFAULT_RX_SIZE=16384`, failed with `NO_SPACE`;
  - `2514751`: 2-node software mode with more posted request buffers/RX queue,
    failed with `NO_SPACE`;
  - `2514764`: 2-node software mode with
    `NCCL_NCHANNELS_PER_NET_PEER=1`, failed with `NO_SPACE`;
  - `2514776`: direct Slurm rank launch with inside-uenv env reapplication,
    failed with `NO_SPACE`;
  - `2514815`: no distributed optimizer/no overlap retry, failed with
    `NO_SPACE`;
  - `2514875`: review-agent `NCCL_PROTO=Simple`/channel-limited probe, failed
    before training on an initial tiny default-process-group all-reduce with
    `NET/OFI ... Error: 16 (NO_SPACE)`.
- A plain runtime override to `TENSOR_MODEL_PARALLEL_SIZE=4` is not possible
  with the existing TP=2 init checkpoints:
  - `2514784` failed at checkpoint load with expected TP4 shard-shape mismatch.
- Socket-over-HSN fallback evidence:
  - `2514830`: 2-node mock-data Megatron smoke with
    `NCCL_NET=Socket`, `NCCL_SOCKET_IFNAME=hsn`, `EXIT_INTERVAL=1`,
    completed; iteration 1 elapsed `82322.3 ms`, `tokens/sec/gpu: 6368.7`;
  - `2514842`: 2-node real-data Megatron smoke with the ordered base dataset,
    completed; iteration 1 elapsed `81933.7 ms`, `lm loss: 1.476208E+00`;
  - `2514854`: 4-node mock-data Megatron smoke with
    `NCCL_NET=Socket`, `NCCL_SOCKET_IFNAME=hsn`, `EXIT_INTERVAL=1`,
    completed; iteration 1 elapsed `50116.3 ms`, `tokens/sec/gpu: 5230.7`.
- Interpretation:
  - the blocker is not Megatron, checkpoints, data, or multi-node torchrun in
    general; it is the AWS Libfabric/CXI path for this trainer/runtime shape;
  - Socket over HSN is now the only validated inter-node path for the current
    job shape;
  - the next required probe is a 16-node, one-iteration Socket/HSN smoke to get
    a full-scale iteration-time estimate before launching both full arms.
- Submitted full-node-count Socket scale smoke:
  - job id `2514876`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_mockdata_socket_hsn_20260610T155226Z`;
  - 16 nodes / 64 GPUs, `NCCL_NET=Socket`, `NCCL_SOCKET_IFNAME=hsn`,
    `USE_MOCK_DATA=1`, `ENABLE_EXTRA_VALID=0`, `DISABLE_SAVE=1`,
    `EXIT_INTERVAL=1`;
  - scheduler predicted start around `2026-06-10T18:05:02+02:00`; result
    pending at the time of this note.

## 2026-06-10T16:26:14Z - 16-Node Socket Smoke Passes, But Throughput Misses 12h Target

- Full-node-count Socket/HSN smoke completed:
  - job id `2514876`;
  - state `COMPLETED`, exit `0:0`, elapsed `00:02:10`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_mockdata_socket_hsn_20260610T155226Z`;
  - 16 nodes / 64 GPUs, `NCCL_NET=Socket`, `NCCL_SOCKET_IFNAME=hsn`,
    `USE_MOCK_DATA=1`, `ENABLE_EXTRA_VALID=0`, `DISABLE_SAVE=1`,
    `EXIT_INTERVAL=1`.
- Evidence:
  - NCCL used HSN interfaces via `NET/Socket`;
  - iteration 1 reached cleanly;
  - iteration 1 elapsed `30046.7 ms`;
  - `tokens/sec/gpu: 2181.1`, `throughput per GPU: 112.4 TFLOP/s/GPU`;
  - no skips or NaNs; exited at iteration 1.
- ETA math from this smoke:
  - `3218 * 30.0467s = 26.86h` raw training time per arm before validation and
    checkpoint overhead;
  - with both arms truly parallel, wall time would still be about this raw
    value, but it would require 32 nodes / 128 GPUs concurrently;
  - serial arms would be roughly double;
  - current default `EXIT_INTERVAL=238` would split Socket training into ~2h
    chunks, causing unnecessary queue churn.
- Interpretation:
  - Socket/HSN is a functional fallback for multi-node Megatron;
  - it does not satisfy the earlier "12h tops" expectation;
  - do not launch the full two-arm run as-is unless accepting a ~27h+ raw
    training wall estimate or after finding a faster transport/scale path.

## 2026-06-10T18:43:00Z - CXI Deep Dive Finds `NCCL_NET_FORCE_FLUSH=1` Root Cause

- Review-agent deep dive updated
  `CXI_NOSPACE_DEEP_DIVE_20260610.md` after additional tests.
- Earlier hypotheses were eliminated:
  - communicator-count/resource fan-out;
  - PyTorch allocator / cuMemMap / VMM registration warnings.
- Decisive 2-node test:
  - job id `2515069`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke2_van_noflush_164007`;
  - same CXI/AWS Libfabric training shape, but with
    `NCCL_NET_FORCE_FLUSH=0`;
  - state `COMPLETED`, exit `0:0`, elapsed `00:02:38`;
  - reached iteration 1 with `elapsed time per iteration (ms): 72458.7`,
    `tokens/sec/gpu: 7235.7`, no skips/NaNs.
- Interpretation:
  - the trainer's hardcoded `NCCL_NET_FORCE_FLUSH=1` is the likely root cause
    of the `size:4 RECV` `NO_SPACE` failure on the AWS OFI NCCL SENDRECV path;
  - Socket/HSN remains a functional fallback, but should not be the preferred
    launch path if CXI validates at scale.
- Changes made:
  - patched `bakeoff_train.sbatch` so `NCCL_NET_FORCE_FLUSH` defaults to `0`;
  - added `NCCL_NET_FORCE_FLUSH` to the runtime audit echo;
  - added an artifact-gate check that the trainer has force-flush disabled.
- Submitted 16-node CXI validation smoke:
  - job id `2515665`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_mockdata_cxi_noflush_20260610T183943Z`;
  - 16 nodes / 64 GPUs, AWS Libfabric/CXI, `NCCL_NET_FORCE_FLUSH=0`,
    `USE_MOCK_DATA=1`, `ENABLE_EXTRA_VALID=0`, `DISABLE_SAVE=1`,
    `EXIT_INTERVAL=1`;
  - pending at time of note. This is now the launch gate.

## 2026-06-10T18:54:00Z - 4-Node CXI No-Flush Smoke Passes; 16-Node Gate Still Pending

- Motivation:
  - user asked whether the same principle could be tested on a smaller batch
    while the 16-node launch-scale smoke waited for resources.
- Scheduler attempts:
  - 8-node `debug` was rejected because `debug` has `MaxNodes=4`;
  - 4-node `debug` at `01:30:00` and `00:30:00` was blocked by
    `QOSMaxNodeMinutesPerJob`;
  - 4-node `debug` at `00:10:00` without overriding the trainer's
    `#SBATCH --signal=SIGUSR2@600` failed immediately from the walltime signal,
    not from NCCL/CXI.
- Successful 4-node smoke:
  - job id `2515691`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke4_vanilla_mockdata_cxi_noflush_20260610T185137Z`;
  - 4 nodes / 16 GPUs, AWS Libfabric/CXI, `NCCL_NET_FORCE_FLUSH=0`,
    `USE_MOCK_DATA=1`, `ENABLE_EXTRA_VALID=0`, `DISABLE_SAVE=1`,
    `EXIT_INTERVAL=1`;
  - command-line Slurm override `--signal=SIGUSR2@60` gave the one-iteration
    debug job enough time despite the 10-minute wall limit.
- Result:
  - state `COMPLETED`, exit `0:0`, elapsed `00:01:46`;
  - runtime audit printed `NCCL_NET=AWS Libfabric` and
    `NCCL_NET_FORCE_FLUSH=0`;
  - `WORLD_SIZE=16`, `LAUNCH_MODE=torchrun`;
  - reached iteration 1 and exited at `EXIT_INTERVAL=1`;
  - iteration 1 elapsed `40043.2 ms`, `tokens/sec/gpu: 6546.5`,
    `throughput per GPU: 337.4 TFLOP/s/GPU`;
  - no `NET/OFI ... NO_SPACE` failure.
- Caveat:
  - stderr still contains known CXI CUDA `sync_memops` warnings and
    post-exit `destroy_process_group` / TCPStore shutdown warnings, but Slurm
    state is `COMPLETED` and the training iteration finished. These are not the
    previous pre-iteration `NO_SPACE` failure.
- Interpretation:
  - the review-agent root-cause candidate now generalizes from 2 nodes to 4
    nodes for the Megatron job shape;
  - this strengthens the case for disabling `NCCL_NET_FORCE_FLUSH`, but the
    pending 16-node job `2515665` remains the full-scale launch gate.

## 2026-06-10T19:40:00Z - 16-Node CXI No-Flush Passes; 16-Node ETA Measured

- Full-scale CXI transport gate:
  - job id `2515665`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_mockdata_cxi_noflush_20260610T183943Z`;
  - state `COMPLETED`, exit `0:0`, elapsed `00:01:48`;
  - 16 nodes / 64 GPUs, AWS Libfabric/CXI,
    `NCCL_NET_FORCE_FLUSH=0`, `WORLD_SIZE=64`, `LAUNCH_MODE=torchrun`;
  - iteration 1 reached cleanly at `15623.3 ms`, no skips/NaNs, no
    `NET/OFI ... NO_SPACE`.
- Real-data training timing:
  - job id `2515841`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_realdata_cxi_noflush_timing_20260610T191629Z`;
  - state `COMPLETED`, exit `0:0`, elapsed `00:02:56`;
  - real base-tokenized CPT data, extra validation off, saves off,
    `EXIT_INTERVAL=10`;
  - iteration 1 `15453.2 ms`;
  - iterations 2-10 median `8559.8 ms`, mean `8629.9 ms`,
    range `8553.0-8869.4 ms`.
- Extra-validation overhead:
  - job id `2515891`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_realdata_cxi_noflush_eval_20260610T192032Z`;
  - state `COMPLETED`, exit `0:0`, elapsed `00:02:57`;
  - real data, extra validation on, `EVAL_INTERVAL=1`, `EVAL_ITERS=1`,
    saves off, `EXIT_INTERVAL=1`;
  - per-set validation printed for `hplt`, `openarchives`, `greek_phd`;
  - iteration 1 line at `21:30:44`, exit line at `21:30:55`;
  - estimated three-set validation event overhead: about `11 s`.
- Checkpoint-save overhead:
  - job id `2515966`;
  - output:
    `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_realdata_cxi_noflush_save_20260610T193249Z`;
  - state `COMPLETED`, exit `0:0`, elapsed `00:02:05`;
  - real data, extra validation off, `SAVE_INTERVAL=1`, saves on,
    `EXIT_INTERVAL=1`;
  - checkpoint save timer `22174 ms`;
  - smoke checkpoint size: `136G`.
- ETA from measured components:
  - `TRAIN_ITERS=3218`;
  - steady-state training mean: `3218 * 8.6299s = 7.71h`;
  - held-out eval: `128` events at about `11s` each = `0.39h`;
  - checkpoint saves: `27` interval saves at about `22.17s` each = `0.17h`;
  - segment startup allowance: `4 * 95s = 0.11h`;
  - estimated allocated runtime per arm with 4 segments: about `8.3-8.5h`,
    excluding Slurm queue wait and sidecar benchmark jobs.
- Submitter update:
  - for `NODES>=16`, default `EXIT_INTERVAL=952` and `N_SEGMENTS=4`;
  - this keeps `EXIT_INTERVAL` divisible by `SAVE_INTERVAL=119` while avoiding
    14 short requeues now that CXI is validated.
- Detailed report:
  - `reports/CPT_16NODE_CXI_TIMING_20260610.md`.
