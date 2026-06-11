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

Babysitting snapshot `2026-06-11T12:56:18Z`:

- `2519265_[0-2]` still running; `2519266_[0-2]` and `2519267_[0-2]` pending
  on the mix dependency.
- `2519269_[0-5]` completed; all old-data forgetting validation sets have been
  tokenized for both base/ext tokenizers.
- Mix progress:
  - `hplt_only`: 1.10B/8.50B tokens (12.9%), rate ~508k tok/s, ETA ~243 min
  - `glossapi_only`: 1.25B/3.70B tokens (33.9%), rate ~578k tok/s, ETA ~70 min
  - `replay_only`: 1.15B/5.00B tokens (23.0%), rate ~542k tok/s, ETA ~118 min
- Current JSONL sizes: `hplt_only` 8.6G, `glossapi_only` 7.7G,
  `replay_only` 5.6G.

Babysitting snapshot `2026-06-11T13:07:12Z`:

- `2519265_[0-2]` still running; Stage A/B still dependency-held.
- `hplt_only`: 1.45B/8.50B tokens (17.1%), rate ~506k tok/s, ETA ~232 min.
- `glossapi_only`: 1.65B/3.70B tokens (44.7%), rate ~584k tok/s, ETA ~58 min.
- `replay_only`: 1.50B/5.00B tokens (30.0%), rate ~539k tok/s, ETA ~108 min.
- Current JSONL sizes: `hplt_only` 11.1G, `glossapi_only` 10.1G,
  `replay_only` 7.2G.

Intervention `2026-06-11T13:20Z`:

- Review feedback identified a material replay-data issue: the currently staged
  shard-0 pools for `english`, `de`, `ru`, and `zh` were fully consumed by the
  2B-char forgetting validation target, leaving zero replay training rows for
  those sources.
- Canceled active mix array `2519265` and dependency-held Stage A/B jobs
  `2519266`/`2519267` before Stage A/B consumed the compromised replay output.
- Verified staged Clariden replay inventory had exactly one local parquet for
  each affected source, but upstream HF metadata is much larger:
  `epfml/FineWeb2-HQ` has 570 `deu_Latn`, 1205 `rus_Cyrl`, and 975 `cmn_Hani`
  parquet files; `HuggingFaceFW/fineweb-edu` has 14 `sample/10BT` parquets.
- Initial thought was to retrieve a bounded prefix of extra upstream shards, but
  the user clarified the stricter rule: forgetting validation must come from
  Apertus-8B pretraining sources, and the absolute 0.5B-token target must fall
  back to a relative split when the eligible pool is too small.
- Located the prior authoritative inventory:
  `docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`. It pins Apertus-8B
  pretraining sources and the final Greek share estimate (3.111B / 13.545T =
  0.023%). Relevant here: English FineWeb-Edu and de/ru/zh FineWeb2-HQ are
  Apertus pretraining source families; CodeParrot is not (Apertus used
  StarCoder/Stack-family code).
- No extra-shard staging job was launched. Patched `build_forgetting_vals.py` to
  write `forgetting_val_manifest.json` and to use the absolute-then-relative
  split policy (`FORGET_CHAR_BUDGET=2B chars`, `MAX_VAL_FRACTION=0.25`).

Rebuild submission `2026-06-11T13:55Z`:

- Synced corrected v2 scripts/runbook plus
  `docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md` to Clariden.
- Regenerated remote phase recipes.
- Submitted affected rebuild chain:
  - forgetting validation rebuild: `2519710`
  - old-data validation tokenization: `2519711` (`afterok:2519710`, array 0-5)
  - phase mix rebuild: `2519712` (`afterok:2519710`, array 0-2)
  - Stage A GreekMMLU-only decontam: `2519713` (`afterok:2519712`, array 0-2)
  - Stage B anonymize/tokenize: `2519714` (`afterok:2519713`, array 0-2)

Forgetting rebuild result:

- `2519710` completed in `00:01:33`.
- New `forgetting_val_manifest.json` confirms split/provenance:
  - `english`: 25.0002% held out, `apertus_pretraining_source_family`
  - `de`: 25.0020% held out, `apertus_pretraining_source_family`
  - `ru`: 25.0002% held out, `apertus_pretraining_source_family`
  - `zh`: 25.0010% held out, `apertus_pretraining_source_family`
  - `old_greek`: 17.5306% held out, `apertus_overlap_overlay`,
    `strict_item_level_seen_by_apertus=true`
  - `code`: `proxy_not_strict` because CodeParrot is not Apertus
    StarCoder/Stack-family pretraining data.
- New `forget_holdout_ids.parquet` has 709,215 unique ids, down from the
  previous 1,161,094 full-pool holdout.
- Old-data validation tokenization `2519711_[0-5]` completed; all 12
  base/ext binaries exist for `english`, `de`, `ru`, `zh`, `code`, and
  `old_greek`.

Mix babysitting snapshot `2026-06-11T13:37Z`:

- `2519712_[0-2]` running.
- `replay_only` loaded the corrected 709,215 drop keys and reached 100M/5.0B
  tokens at ~569k tok/s.
- `hplt_only` reached 100M/8.5B tokens at ~474k tok/s.
- `glossapi_only` reached 50M/3.7B tokens at ~300k tok/s.

Code-source intervention `2026-06-11T14:02Z`:

- User accepted the `bigcode/starcoderdata` Hugging Face terms. Re-tested from
  Clariden with `hf_hub_download(..., dry_run=True)`:
  `python/train-00000-of-00059.parquet` and
  `javascript/train-00000-of-00065.parquet` now return OK.
- Canceled the CodeParrot-based chain before Stage A/B could consume it:
  - `2519712_[0-2]`: `CANCELLED by 1883` at about 16 min runtime.
  - `2519713_[0-2]` and `2519714_[0-2]`: canceled at `00:00:00`.
- Patched v2 to replace CodeParrot with an Apertus-family StarCoderData
  source:
  - added `dataset/stage_starcoderdata_subset.{py,sbatch}` to download a
    deterministic subset and rewrite shards with stable
    `doc_id=starcoderdata:<repo_path>:<upstream_id>`;
  - updated `make_phase_recipes.py` so the replay code bucket becomes
    `code_starcoderdata_subset`, local parquet
    `$SC/cpt_corpus/replay/starcoderdata_v2/*/*.parquet`, text column
    `content`, drop key `doc_id`;
  - updated `build_forgetting_vals.py` so `val_forget_code.jsonl` is carved
    from the same staged StarCoderData pool and subject to the same
    absolute-then-relative validation split policy.
- Local checks passed:
  - `python3 -m py_compile` for the new staging script plus the patched recipe
    and forgetting builders;
  - `bash -n` for staging/forgetting/mix sbatches;
  - local recipe generation confirms replay has 27 sources, source-weight sum
    `0.99999999`, and code source `code_starcoderdata_subset`.
- Synced patched v2 tree to Clariden, regenerated recipes, and removed stale
  CodeParrot-derived phase outputs/tokenized code validation artifacts.
- Submitted replacement chain:
  - StarCoderData staging: `2519897`
  - forgetting validation rebuild: `2519898` (`afterok:2519897`)
  - old-data validation tokenization: `2519899` (`afterok:2519898`, array 0-5)
  - phase mix rebuild: `2519900` (`afterok:2519898`, array 0-2)
  - Stage A GreekMMLU-only decontam: `2519901` (`afterok:2519900`, array 0-2)
  - Stage B anonymize/tokenize: `2519902` (`afterok:2519901`, array 0-2)
- `2519897` started on `nid007589` at about `2026-06-11T14:23:41Z` and
  completed in `00:05:04` with exit `0:0`.
- Verified staged StarCoderData pool:
  - 28 rewritten parquet shards under
    `$SC/cpt_corpus/replay/starcoderdata_v2`;
  - 6,802,565 rows;
  - every checked shard has `content` and stable `doc_id`;
  - manifest written to
    `$SC/cpt_corpus/replay/starcoderdata_v2/manifest.json`.
- `2519898` started automatically after `2519897`; first log shows
  English/de/ru/zh validation rebuilt with the 25% relative fallback policy.
- `2519898` completed successfully at `2026-06-11T14:31:20Z`.
  `$STAGE/forgetting_val_manifest.json` now reports:
  - `english`: 77,227 / 310,544 docs, 0.38B / 1.52B chars, 25.0002%,
    relative fallback, Apertus source family.
  - `de`: 27,193 / 127,590 docs, 0.11B / 0.44B chars, 25.0020%,
    relative fallback, Apertus source family.
  - `ru`: 18,194 / 75,915 docs, 0.08B / 0.33B chars, 25.0002%,
    relative fallback, Apertus source family.
  - `zh`: 18,862 / 79,306 docs, 0.03B / 0.10B chars, 25.0010%,
    relative fallback, Apertus source family.
  - `code`: 257,526 / 6,802,565 docs, 2.00B / 28.29B chars,
    7.0698%, absolute target, StarCoderData Apertus source family.
  - `old_greek`: 366,755 / 2,224,446 docs, 2.00B / 11.41B chars,
    17.5306%, absolute target, Apertus-overlap overlay.
- New `forget_holdout_ids.parquet`: 765,757 unique ids.
- `2519899` old-data validation tokenization began automatically after
  `2519898`; `english` completed first.
- `2519899_[0-5]` completed successfully:
  - `english` `00:01:29`
  - `de` `00:01:20`
  - `ru` `00:01:08`
  - `zh` `00:01:08`
  - `code` `00:02:39`
  - `old_greek` `00:02:57`
- Verified all 12 old-data validation binaries exist in `$MEGOUT`:
  `english`, `de`, `ru`, `zh`, `code`, `old_greek` × `base`/`ext`.
- `2519900_[0-2]` phase mix rebuild is running as of the latest snapshot;
  first observed rates:
  - `hplt_only`: 50.0M / 8.5B tokens, ~409k tok/s.
  - `glossapi_only`: 50.1M / 3.7B tokens, ~309k tok/s.
  - `replay_only`: 50.0M / 5.0B tokens, ~523k tok/s.

Mix babysitting snapshot `2026-06-11T15:00Z`:

- `2519900_[0-2]` still running; Stage A/B (`2519901`/`2519902`) still
  dependency-held.
- `hplt_only`: 600.0M / 8.5B tokens (7.1%), ~466k tok/s, ETA ~282 min.
- `glossapi_only`: 751.1M / 3.7B tokens (20.3%), ~554k tok/s, ETA ~89 min.
- `replay_only`: 700.0M / 5.0B tokens (14.0%), ~555k tok/s, ETA ~129 min.
- No stderr output observed from the mix builders.

Mix babysitting snapshot `2026-06-11T15:15Z`:

- `2519900_[0-2]` still running; Stage A/B still dependency-held.
- `hplt_only`: 1.05B / 8.5B tokens (12.4%), ~464k tok/s, ETA ~268 min.
- `glossapi_only`: 1.25B / 3.7B tokens (33.9%), ~565k tok/s, ETA ~72 min.
- `replay_only`: 1.15B / 5.0B tokens (23.0%), ~530k tok/s, ETA ~121 min.
- No stderr output observed from the mix builders.

Mix babysitting snapshot `2026-06-11T15:30Z`:

- `2519900_[0-2]` still running; Stage A/B still dependency-held.
- `hplt_only`: 1.45B / 8.5B tokens (17.1%), ~464k tok/s, ETA ~253 min.
- `glossapi_only`: 1.80B / 3.7B tokens (48.8%), ~570k tok/s, ETA ~55 min.
- `replay_only`: 1.60B / 5.0B tokens (32.0%), ~519k tok/s, ETA ~109 min.
- No stderr output observed from the mix builders.

Mix acceleration intervention `2026-06-11T15:40Z`:

- Confirmed the slow mix array was effectively single-core despite requesting
  64 CPUs: `sstat` showed about one CPU-hour per one wall-hour for the batch
  step.
- Patched `dataset/mix_phase_binaries.sbatch` to use
  `MIX_SHARDS=16`: each binary now launches 16 concurrent
  `mix_builder.py --source-shard-index/--source-shard-count` processes on its
  allocated CPU node, then concatenates shard JSONLs back to
  `$STAGE/{hplt_only,glossapi_only,replay_only}.jsonl` before Stage A.
- Canceled the under-parallelized chain:
  - `2519900_[0-2]` after about one hour of runtime.
  - dependency-held `2519901`/`2519902` before they ran.
- Submitted accelerated replacement chain:
  - sharded mix: `2520411` (`MIX_SHARDS=16`, array 0-2), running immediately
    on `nid006727`, `nid006731`, `nid006747`;
  - Stage A: `2520414` (`afterok:2520411_*`);
  - Stage B: `2520415` (`afterok:2520414_*`).
- First accelerated progress sample at `2026-06-11T15:47Z`:
  - HPLT shards: 100M / 531.25M tokens each; per-shard ETA ~19.5-23.5 min.
  - GlossAPI shards: 50M / 231.25M tokens each; per-shard ETA ~17.5-21 min.
  - Replay shards: mostly 100M / 312.5M tokens each; per-shard ETA ~9.7-13.5
    min, with one lagging shard at 50M and ~17.2 min ETA.
- New expected mix completion if rates hold: about `2026-06-11T16:10Z`.

Old-Greek overlay revalidation `2026-06-11T14:00Z`:

- Checked live Clariden files:
  - overlay:
    `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/dedup_audit/artifacts/dedup_20260519T010924Z/cpt_final_overlay/apertus_overlap_drop_docs.parquet`
  - replay:
    `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/greek_replay/greek_replay.parquet`
- Overlay rows/distinct pairs: 2,223,742.
- Greek replay rows: 2,224,446; distinct `(source_dataset, source_doc_id)`
  pairs: 2,223,742.
- Missing replay pairs from overlay: 0. Verdict: PASS for the current
  `old_greek` strict overlay provenance label.

Accelerated mix completion `2026-06-11T16:10Z`:

- Sharded replacement chain worked as intended: the slow single-process mix was
  canceled and the replacement array `2520411` finished all three phase JSONLs
  in under 30 minutes after start.
- Completion times:
  - `replay_only`: `2026-06-11T15:59:48Z`, output
    `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2/replay_only.jsonl`
    size 23G.
  - `glossapi_only`: `2026-06-11T16:03:05Z`, output
    `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2/glossapi_only.jsonl`
    size 21G.
  - `hplt_only`: `2026-06-11T16:09:26Z`, output
    `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2/hplt_only.jsonl`
    size 61G.
- Stage A decontamination `2520414_[0-2]` released immediately after the mix
  dependency cleared and is running on three CPU nodes. Stage B `2520415_[0-2]`
  remains dependency-held on Stage A.

Training launch guard `2026-06-11T16:24Z`:

- Verified the real two-phase training driver defaults to `NODES=16`,
  `GPUS_PER_NODE=4`, and `TIME_LIMIT=06:00:00`; sweep launchers submit
  independent arms without cross-arm dependencies, so replay/LR arms can run in
  parallel if Slurm capacity is available.
- Added a live-launch guard to `train/submit_curriculum_two_phase.sh`: any
  non-smoke live training run (`DRY_RUN=0`, `TOTAL_ITER>2`) now fails fast if
  `NODES < 16`. The planned boundary smoke remains allowed with `TOTAL_ITER=2`
  and `NODES=1`.

Stage A completion `2026-06-11T16:24Z`:

- `2520414_[0-2]` completed and released Stage B immediately.
- Completion times:
  - `replay_only`: `2026-06-11T16:12:28Z`, decontam output 23G, dropped 275K.
  - `glossapi_only`: `2026-06-11T16:15:08Z`, decontam output 21G, dropped 16M.
  - `hplt_only`: `2026-06-11T16:23:59Z`, decontam output 61G, dropped 12M.
- HPLT Stage A used `parallel n_workers=64` for GreekMMLU decontamination;
  the scan completed in 219.7 seconds before the final filtering pass.
- Stage B `2520415_[0-2]` started immediately for
  `hplt_only`, `glossapi_only`, and `replay_only`.

Stage B restart `2026-06-11T16:28Z`:

- Initial Stage B array `2520415_[0-2]` failed within 20 seconds because the
  `cpt_build_py312` build environment lacked `orjson`, which datatrove
  requires for `JsonlReader`.
- Installed `orjson==3.11.9` into the existing build venv and verified import
  through the same `run_build_py` wrapper used by the sbatch scripts.
- Resubmitted Stage B as `2520650_[0-2]`; all three tasks started.
- Clariden resource note: the scripts request CPU/memory only (`ReqTRES` has no
  GPU request), but the visible `normal/debug/low` partitions are GPU-node
  partitions, so Slurm reports full-node allocations with `gres/gpu=4`. The
  only no-GRES partition visible is `xfer` (2 nodes, currently busy). For speed,
  the active Stage B restart is running on `normal`; future CPU-only strictness
  would require an available CPU/no-GRES partition or a slower `xfer` queue.

Stage B completion and boundary pin `2026-06-11T17:30Z`:

- Stage B restart `2520650_[0-2]` completed successfully:
  - `glossapi_only`: `00:16:11`, done at `2026-06-11T16:43:57Z`.
  - `replay_only`: `00:24:06`, done at `2026-06-11T16:51:52Z`.
  - `hplt_only`: `00:58:28`, done at `2026-06-11T17:25:49Z`.
- Verified all six train binaries exist:
  `hplt_only`, `glossapi_only`, `replay_only` × `base`, `ext`.
- Exact `.bin` token counts (`bytes/4`):
  - ext: `hplt_only=8,515,361,723`, `glossapi_only=3,694,776,527`,
    raw boundary `3218 * hplt/(hplt+glossapi) = 2244.24`, rounded to
    `2261` (`19*119`).
  - base: `hplt_only=13,802,297,740`, `glossapi_only=5,300,413,963`,
    would round to `2380`; keep the vanilla control on the same `2261`
    curriculum schedule as TD for comparability.
- No config value change needed: `curriculum_common.env`,
  `submit_curriculum_two_phase.sh`, and `cadence_curriculum.tsv` already carry
  the pinned ext boundary `2261`. Updated comments/runbook to mark it pinned,
  not provisional.

Dataset integrity verification `2026-06-11T17:50Z`:

- Added `dataset/verify_curriculum_outputs.py` and ran it as Slurm job
  `2520880` (`COMPLETED`, `00:03:52`).
- Verification report:
  `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2/verify_curriculum_outputs.json`.
- Result: `ok=true`.
- Checked all expected train and extra-valid `.bin/.idx` files:
  `hplt_only`, `glossapi_only`, `replay_only` × `base/ext`; new-Greek held-outs
  `hplt/openarchives/greek_phd` × `base/ext`; forgetting held-outs
  `english/de/ru/zh/code/old_greek` × `base/ext`.
- Held-out ID exclusion on ID-preserving decontam JSONLs:
  - `hplt_only_decontam.jsonl`: 9,535,742 rows, 0 missing `doc_id`,
    0 new-holdout overlap, 0 forgetting-holdout overlap.
  - `glossapi_only_decontam.jsonl`: 77,136 rows, 0 missing `doc_id`,
    0 new-holdout overlap, 0 forgetting-holdout overlap.
  - `replay_only_decontam.jsonl`: 5,031,733 rows, 0 missing `doc_id`,
    0 new-holdout overlap, 0 forgetting-holdout overlap.
