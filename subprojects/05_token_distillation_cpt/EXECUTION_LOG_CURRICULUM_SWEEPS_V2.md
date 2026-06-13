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

Boundary smoke `2026-06-11T18:10Z`:

- First boundary-smoke attempt `curr_smoke_boundary_20260611T175038Z`:
  - segment 1 `2520901`: `COMPLETED`, `00:03:55`;
  - segment 2 `2520902`: `FAILED`, `00:00:41`.
- Failure cause: the original `reset_data_index_guard.py` imported
  `megatron.legacy.data.data_samplers` directly before Megatron's normal import
  order, producing a circular import:
  `ImportError: cannot import name 'build_pretraining_data_loader' from
  partially initialized module 'megatron.legacy.data.data_samplers'`.
- Patched the reset guard to install a lazy `MetaPathFinder` import hook for
  `megatron.legacy.data.data_samplers`; the hook wraps the real module only
  after import completes, avoiding the circular import while preserving the
  same phase-2 train-dataloader reset behavior.
- Local and remote Python compile checks passed for the patched reset guard.
- Second boundary-smoke attempt `curr_smoke_boundary_20260611T180000Z`:
  - segment 1 `2520929`: `COMPLETED`, `00:04:00`;
  - segment 2 `2520930`: `COMPLETED`, `00:04:52`.
- Acceptance evidence:
  - phase 1 logged finite iteration 1:
    loss `5.979677`, LR `5.623750e-06`, skipped `0`, NaN `0`, checkpoint
    `iter_0000001` saved;
  - phase 2 loaded `iter_0000001`, all ranks logged
    `train dataloader consumed_samples 1024 -> 0`, and the scheduler continued
    to LR `5.747500e-06` rather than resetting;
  - phase 2 logged finite iteration 2:
    loss `4.697852`, skipped `0`, NaN `0`, checkpoint `iter_0000002` saved;
  - all 9 extra validation names appeared in the smoke logs:
    `hplt`, `openarchives`, `greek_phd`, `english`, `de`, `ru`, `zh`, `code`,
    `old_greek`;
  - traceback/error grep was clean apart from Megatron's inert config lines
    `error_injection_rate=0` / `error_injection_type=transient_error`.
- Verdict: PASS. Proceed to live replay/control launches with the enforced
  `NODES>=16` production floor.

Replay/control production launch `2026-06-11T18:13Z`:

- Ran remote dry-run immediately before launch; confirmed every production
  segment emits `--nodes=16 --gpus-per-node=4 --gres=gpu:4`, four segments per
  chain, and `RESET_DATA_INDEX=1` only on each phase-2 segment.
- Queue was empty for `fffoivos` before launch.
- Submitted live with `DRY_RUN=0 CONFIRM_LAUNCH=1`.
- Vanilla control:
  - run tag: `curr_vanilla_r0.35_20260611T181233Z`
  - jobs: `2520960` -> `2520961` -> `2520962` -> `2520963`
  - each training segment requests 16 nodes / 64 GPUs
  - GreekMMLU watcher: `2520980`, `EVAL_ARM=vanilla`, running on `xfer`
    with `cpu=1,mem=4G,node=1`
- TD replay sweep:
  - R=0.35 run tag: `curr_td_replay0.35_20260611T181235Z`
    jobs `2520964` -> `2520966` -> `2520967` -> `2520968`
  - R=0.25 run tag: `curr_td_replay0.25_20260611T181235Z`
    jobs `2520969` -> `2520971` -> `2520972` -> `2520973`
  - R=0.15 run tag: `curr_td_replay0.15_20260611T181235Z`
    jobs `2520974` -> `2520975` -> `2520976` -> `2520977`
  - every training segment requests 16 nodes / 64 GPUs
  - GreekMMLU watchers: `2520981`, `2520982`, `2520983`, all running on
    `xfer` with `cpu=1,mem=4G,node=1`
- First segments `2520960`, `2520964`, `2520969`, and `2520974` are independent
  and pending with reason `Priority`; downstream segments are dependency-held.
  This is the intended parallelism: Clariden can start any or all first
  segments as capacity opens.
- Launch log:
  `/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/launch_replay_control_20260611T181233Z.log`.

Launch babysitting snapshot `2026-06-11T18:19Z`:

- Clariden local display timezone is `+0200`.
- Watchers `2520980`-`2520983` are running on `xfer`.
- First-segment training jobs remain pending, independent, and eligible to run
  in parallel. Current soft `squeue --start` estimates:
  - vanilla `2520960`: `2026-06-11T21:28:42+0200`
  - TD R=0.35 `2520964`: `2026-06-11T22:44:22+0200`
  - TD R=0.25 `2520969`: `2026-06-11T22:44:22+0200`
  - TD R=0.15 `2520974`: `2026-06-11T22:53:11+0200`

User-facing status check `2026-06-11T18:30Z`:

- Confirmed again that the launched production set is:
  vanilla control plus TD replay `R in {0.35, 0.25, 0.15}`.
- No LR sweep arms have been launched; those remain gated on replay results and
  the user's `R*` choice.
- `sacct` confirms every training segment requests `NNodes=16` and
  `gres/gpu=64`.
- Current first-segment states:
  - vanilla `2520960`: pending on `Resources`, soft start
    `2026-06-11T21:17:29+0200`;
  - TD R=0.35 `2520964`: pending on `Priority`, soft start
    `2026-06-11T22:44:22+0200`;
  - TD R=0.25 `2520969`: pending on `Priority`, soft start
    `2026-06-11T22:44:22+0200`;
  - TD R=0.15 `2520974`: pending on `Priority`, soft start
    `2026-06-11T22:53:11+0200`.
- Watchers `2520980`-`2520983` are still running and logging
  `waiting for checkpoints`; no training logs exist yet because first segments
  have not allocated nodes.

First production runtime signal `2026-06-11T18:35Z`:

- Vanilla first segment `2520960` allocated 16 nodes and started running at
  `2026-06-11T18:33Z`.
- Header confirms `WORLD_SIZE=64`, `nodes=16`, `gpus_per_node=4`, base
  tokenizer/data, phase-1 `hplt_only_base + replay_only_base`, and extra-valid
  enabled.
- Megatron init completed; model/dataloader setup completed; no stderr
  traceback.
- First iterations are finite:
  - iter 1: loss `1.488362`, LR `5.623750e-06`, skipped `0`, NaN `0`,
    `4376.4` tok/s/GPU;
  - iter 2: loss `1.498125`, LR `5.747500e-06`, skipped `0`, NaN `0`,
    `7570.0` tok/s/GPU;
  - iter 3: loss `1.499231`, LR `5.871250e-06`, skipped `0`, NaN `0`,
    `7630.5` tok/s/GPU;
  - iter 4: loss `1.509641`, LR `5.995000e-06`, skipped `0`, NaN `0`,
    `7624.7` tok/s/GPU.
- The printed full-run ETA after warmup is about `7h40m`; segment 1 should exit
  at iteration `952` and checkpoint every `119` iterations.

Second production allocation `2026-06-11T18:46Z`:

- TD replay R=0.35 first segment `2520964` allocated 16 nodes and started
  running at `2026-06-11T18:44Z`, so the cluster is now running 32 training
  nodes for this sweep (`2520960` vanilla + `2520964` TD R=0.35).
- Header confirms `ARM=td`, ext tokenizer
  `apertus_greek_modern_only_148480`, phase-1
  `hplt_only_ext + replay_only_ext`, `R=0.35`, extra-valid enabled, and
  tokenization `ext`.
- First TD R=0.35 iterations are finite:
  - iter 1: loss `5.979675`, LR `5.623750e-06`, skipped `0`, NaN `0`,
    `4197.7` tok/s/GPU;
  - iter 2: loss `5.888990`, LR `5.747500e-06`, skipped `0`, NaN `0`,
    `7462.7` tok/s/GPU;
  - iter 3: loss `5.765898`, LR `5.871250e-06`, skipped `0`, NaN `0`,
    `7509.0` tok/s/GPU;
  - iter 4: loss `5.558364`, LR `5.995000e-06`, skipped `0`, NaN `0`,
    `7510.2` tok/s/GPU;
  - iter 5: loss `5.409743`, LR `6.118750e-06`, skipped `0`, NaN `0`,
    `7508.1` tok/s/GPU.
- At the same snapshot, vanilla reached iteration `75` with loss `1.408932`,
  skipped `0`, NaN `0`, and about `7.62k` tok/s/GPU.
- TD R=0.25 first segment `2520969` is pending on `Resources`; TD R=0.15 first
  segment `2520974` remains pending on `Priority`.

Training checkpoint / eval cadence clarification `2026-06-11T18:59Z`:

- Vanilla `2520960` saved training checkpoint `iter_0000119` successfully:
  `.metadata` exists and training continued to iteration `120`.
- The GreekMMLU watcher did not submit sidecars at iter `119` by design:
  `eval/cadence_curriculum.tsv` starts at iter `238` (`curr-1.0B`), not at
  every `SAVE_INTERVAL=119` checkpoint.
- Updated monitoring target: first eval sidecar submission should happen after
  vanilla reaches checkpoint `iter_0000238`; iter `119` is only a durability
  checkpoint.

All replay/control first segments active `2026-06-11T19:08Z`:

- TD replay R=0.25 first segment `2520969` allocated 16 nodes at about
  `2026-06-11T18:49Z`.
- TD replay R=0.15 first segment `2520974` allocated 16 nodes at about
  `2026-06-11T19:04Z`.
- All four production first segments are now running concurrently:
  - vanilla `2520960`: 16 nodes / 64 GPUs;
  - TD R=0.35 `2520964`: 16 nodes / 64 GPUs;
  - TD R=0.25 `2520969`: 16 nodes / 64 GPUs;
  - TD R=0.15 `2520974`: 16 nodes / 64 GPUs.
- This realizes the requested parallelism: 64 training nodes / 256 GPUs active
  across the replay/control stage, plus four tiny `xfer` watchers.
- R=0.15 header confirms ext tokenizer, phase-1
  `hplt_only_ext + replay_only_ext`, `R=0.15`, extra-valid enabled, and
  tokenization `ext`.
- R=0.15 first iterations are finite; by iter 18 loss is `4.398194`, skipped
  `0`, NaN `0`, and throughput is about `7.49k` tok/s/GPU.
- TD R=0.35 has saved durability checkpoint `iter_0000119/.metadata`.

First eval sidecar submission `2026-06-11T19:21Z`:

- Vanilla `2520960` reached checkpoint `iter_0000238` at about `1.0B` tokens;
  watcher `2520980` detected it at `2026-06-11T19:18:21Z`.
- Watcher submitted eval sidecars under
  `/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/eval_curr_vanilla_r0.35_20260611T181233Z/iter_0000238`:
  - convert `2521245`, output
    `/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/eval_curr_vanilla_r0.35_20260611T181233Z/iter_0000238_hf`;
  - GreekMMLU native `2521246`, output
    `/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/eval_curr_vanilla_r0.35_20260611T181233Z/iter_0000238/native_mcq`;
  - code BPB `2521247`, output
    `/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/eval_curr_vanilla_r0.35_20260611T181233Z/iter_0000238/heldout_code_bpb.json`;
  - math BPB `2521248`, output
    `/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/eval_curr_vanilla_r0.35_20260611T181233Z/iter_0000238/heldout_math_bpb.json`;
  - checksum `2521249`, output
    `/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/eval_curr_vanilla_r0.35_20260611T181233Z/iter_0000238/checksums/curr-1.0B_iter_0000238_checksum_manifest.json`.
- `sacct` snapshot: convert `2521245`, code BPB `2521247`, and math BPB
  `2521248` completed successfully; GreekMMLU native `2521246` and checksum
  `2521249` were still running.
- Training remains healthy at this snapshot:
  - vanilla reached iter `286`, loss `1.389664`, skipped `0`, NaN `0`;
  - TD R=0.35 reached iter `213`, loss `2.279708`, skipped `0`, NaN `0`;
  - TD R=0.25 reached iter `188`, loss `2.358219`, skipped `0`, NaN `0`;
  - TD R=0.15 reached iter `96`, loss `2.683198`, skipped `0`, NaN `0`.

First TD eval sidecars `2026-06-11T19:34Z`:

- Vanilla first eval bundle completed successfully:
  - convert `2521245` completed in `00:01:34`;
  - GreekMMLU native `2521246` completed in `00:12:52`;
  - code BPB `2521247` completed in `00:01:07`;
  - math BPB `2521248` completed in `00:01:03`;
  - checksum `2521249` completed in `00:04:21`.
- TD R=0.35 watcher `2520981` detected `iter_0000238` at
  `2026-06-11T19:28:21Z` and submitted sidecars:
  - convert `2521270`;
  - GreekMMLU native `2521271`;
  - code BPB `2521272`;
  - math BPB `2521273`;
  - checksum `2521274`.
- TD R=0.35 sidecar status at this snapshot: convert, code BPB, and math BPB
  completed successfully; GreekMMLU native and checksum were still running.
- TD R=0.25 reached and saved `iter_0000238`; watcher `2520982` detected it at
  `2026-06-11T19:33:21Z` and submitted sidecars:
  - convert `2521286`;
  - GreekMMLU native `2521287`;
  - code BPB `2521288`;
  - math BPB `2521289`;
  - checksum `2521290`.
- TD R=0.15 was still before first eval at this snapshot, around iter `150`,
  with finite loss and no skipped/NaN iterations.

Sidecar code-BPB correction `2026-06-11T19:42Z`:

- Confirmed the primary curriculum data path was already StarCoderData:
  replay recipe `code_starcoderdata_subset`, in-training extra validation
  `val_forget_code_{base,ext}`, and `val_forget_code.jsonl` all come from the
  staged `bigcode/starcoderdata` subset.
- Found a remaining auxiliary sidecar default still pointing at the legacy
  200-doc CodeParrot BPB file
  `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_code_heldout_200_20260528.jsonl`.
- Created a small StarCoder BPB sidecar sample from the already-built heldout:
  `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2/val_forget_code_starcoder_200_for_bpb.jsonl`
  with `200` docs and `2,821,857` bytes.
- Patched and synced `scripts/watch_and_submit_td_checkpoint_sidecars.sbatch`
  and `scripts/submit_td_checkpoint_sidecars.sh` to prefer the StarCoder sample
  when present, with legacy CodeParrot fallback only if the StarCoder sample is
  absent.
- Restarted only the tiny watcher jobs, preserving each existing
  `*_sidecar_watch` state directory so already-submitted iter-238 sidecars are
  not duplicated:
  - vanilla watcher `2521309`;
  - TD R=0.35 watcher `2521310`;
  - TD R=0.25 watcher `2521311`;
  - TD R=0.15 watcher `2521312`.
- New watcher env files confirm
  `CODE_HELDOUT_JSONL=/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2/val_forget_code_starcoder_200_for_bpb.jsonl`.
- Caveat: the already-submitted iter-238 auxiliary code BPB sidecars for
  vanilla, TD R=0.35, and TD R=0.25 used the legacy CodeParrot 200-doc file.
  Treat those first code-BPB numbers as legacy auxiliary metrics. Future
  sidecar code BPB submissions use StarCoder.

First checkpoint eval sanity set complete `2026-06-11T20:00Z`:

- TD R=0.15 `iter_0000238` sidecars completed:
  - convert `2521334` completed in `00:01:35`;
  - GreekMMLU native `2521335` completed in `00:09:20`;
  - code BPB `2521336` completed in `00:01:11`;
  - math BPB `2521337` completed in `00:01:04`;
  - checksum `2521338` completed in `00:04:20`.
- TD R=0.15 code BPB output confirms the corrected StarCoder sample path:
  `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2/val_forget_code_starcoder_200_for_bpb.jsonl`.
- GreekMMLU `curr-1.0B` headlines:
  - vanilla: `0.5296416546` (`8809/16632`);
  - TD R=0.35: `0.4805194805` (`7992/16632`);
  - TD R=0.25: `0.4909211159` (`8165/16632`);
  - TD R=0.15: `0.4960918711` (`8251/16632`).
- These are sanity-check trajectories only; replay decision remains gated on
  later/full replay results plus forgetting-loss deltas.
- Vanilla reached `iter_0000476` (`curr-2.0B`) and watcher `2521309`
  submitted sidecars `2521385`-`2521389`. The restarted watcher env uses the
  StarCoder code BPB sample for this and future sidecars.

Parallel replay wave status `2026-06-11T20:30Z`:

- Production training wave is active and parallelized: vanilla control plus TD
  replay `R in {0.35, 0.25, 0.15}` are all running first segments on `16`
  Clariden nodes each (`64` GPUs per arm).
- Current first-segment jobs:
  - vanilla `2520960` running on `16` nodes;
  - TD R=0.35 `2520964` running on `16` nodes;
  - TD R=0.25 `2520969` running on `16` nodes;
  - TD R=0.15 `2520974` running on `16` nodes.
- Downstream segment jobs are already submitted and dependency-gated, not
  missing:
  - vanilla `2520961`-`2520963`;
  - TD R=0.35 `2520966`-`2520968`;
  - TD R=0.25 `2520971`-`2520973`;
  - TD R=0.15 `2520975`-`2520977`.
- LR sweep arms are intentionally not launched yet. They remain gated on
  collecting replay/control trajectories and old-data loss deltas, then the
  user choosing `R*`.
- Training health is clean at this snapshot: finite losses, skipped iterations
  `0`, NaN iterations `0`.
- Latest observed iterations:
  - vanilla iter `705`, loss `1.358994`;
  - TD R=0.35 iter `625`, loss `2.051710`;
  - TD R=0.25 iter `601`, loss `2.101287`;
  - TD R=0.15 iter `505`, loss `2.175694`.
- `curr-2.0B` sidecars completed for vanilla, TD R=0.35, and TD R=0.25:
  - vanilla GreekMMLU `0.5094396344` (`8473/16632`), StarCoder BPB
    `0.2638159587`;
  - TD R=0.35 GreekMMLU `0.5267556518` (`8761/16632`), StarCoder BPB
    `0.2680510668`;
  - TD R=0.25 GreekMMLU `0.5165343915` (`8591/16632`), StarCoder BPB
    `0.2693039778`.
- TD R=0.15 reached `iter_0000476` and watcher `2521312` submitted the
  `curr-2.0B` sidecar bundle:
  - convert `2521491`;
  - GreekMMLU native `2521493`;
  - code BPB `2521494`;
  - math BPB `2521495`;
  - checksum `2521496`.
  The bundle is still in progress/pending as of this snapshot.

Second checkpoint eval sanity set complete `2026-06-11T21:00Z`:

- TD R=0.15 `iter_0000476` sidecars completed:
  - convert `2521491` completed in `00:01:32`;
  - GreekMMLU native `2521493` completed in `00:08:59`;
  - code BPB `2521494` completed in `00:01:03`;
  - math BPB `2521495` completed in `00:01:01`;
  - checksum `2521496` completed in `00:04:25`.
- TD R=0.15 code BPB output confirms the corrected StarCoder sample path:
  `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2/val_forget_code_starcoder_200_for_bpb.jsonl`.
- GreekMMLU `curr-2.0B` headlines:
  - vanilla: `0.5094396344` (`8473/16632`);
  - TD R=0.35: `0.5267556518` (`8761/16632`);
  - TD R=0.25: `0.5165343915` (`8591/16632`);
  - TD R=0.15: `0.5144901395` (`8557/16632`).
- StarCoder sidecar BPB `curr-2.0B`:
  - vanilla: `0.2638159587`;
  - TD R=0.35: `0.2680510668`;
  - TD R=0.25: `0.2693039778`;
  - TD R=0.15: `0.2702046887`.
- These `curr-2.0B` points are still sanity trajectories, not the replay
  decision table. Replay choice remains gated on later/full trajectories plus
  in-training extra-validation forgetting losses.

First production segment handoff and third eval sanity set `2026-06-11T21:18Z`:

- Vanilla segment 1 `2520960` reached `iter_0000952`, saved checkpoint, and
  exited with Slurm state `COMPLETED`, exit `0:0`, elapsed `02:40:08`.
- Vanilla segment 2 `2520961` started automatically from the dependency chain
  on `16` nodes.
- Segment 2 loaded `/checkpoints/iter_0000952` and resumed at iteration `953`.
  The optimizer/schedule did not reset: observed LR stayed `5.500000E-05`,
  skipped iterations `0`, NaN iterations `0`.
- First resumed training lines were finite:
  - iter `953`, loss `1.351256`;
  - iter `954`, loss `1.343664`;
  - iter `970`, loss `1.346770`.
- Segment-1 stderr included distributed shutdown/rendezvous warnings after the
  clean save. Slurm and the run footer both marked the job done successfully;
  no action taken.
- TD R=0.35, TD R=0.25, and TD R=0.15 segment-1 jobs remain running; their
  segment-2 jobs remain dependency-gated.
- GreekMMLU `curr-3.0B` headlines:
  - vanilla: `0.5221260221` (`8684/16632`);
  - TD R=0.35: `0.5344516595` (`8889/16632`);
  - TD R=0.25: `0.5056517557` (`8410/16632`);
  - TD R=0.15: `0.5214646465` (`8673/16632`).
- StarCoder sidecar BPB `curr-3.0B`:
  - vanilla: `0.2650278562`;
  - TD R=0.35: `0.2673165931`;
  - TD R=0.25: `0.2682826044`;
  - TD R=0.15: `0.2688295269`.
- Replay decision remains gated on later/full trajectories plus in-training
  forgetting-loss deltas; LR sweep remains intentionally unlaunched.

R0.25 handoff clean, R0.35 segment-2 retry `2026-06-11T21:35Z`:

- TD R=0.35 segment 1 `2520964` completed successfully: Slurm state
  `COMPLETED`, exit `0:0`, elapsed `02:42:52`.
- Its original segment 2 `2520966` started on `16` nodes but failed during
  Torch/Inductor JIT warmup before any resumed training iteration:
  - failure rank: `rank62`;
  - error: `torch.AcceleratorError: CUDA error: unspecified launch failure`;
  - Slurm state `FAILED`, exit `1:0`, elapsed `00:01:46`.
- Because the failed job exited before loading/resuming from
  `iter_0000952`, the safe retry point remains the existing checkpoint under
  `curr_td_replay0.35_20260611T181235Z/checkpoints`.
- Submitted a replacement R0.35 chain into the same run directory:
  - `2521904` = `s2r1`, phase-1 HPLT config, `EXIT_INTERVAL=1904`;
  - `2521905` = `s3r1`, phase-1 HPLT config, `EXIT_INTERVAL=2261`;
  - `2521906` = `s4r1`, phase-2 GlossAPI config, `EXIT_INTERVAL=3218`,
    `RESET_DATA_INDEX=1`.
- Canceled the broken original dependency tail `2520967` and `2520968` to
  avoid a stale `afterok:2520966` chain.
- TD R=0.25 segment 1 `2520969` completed successfully: Slurm state
  `COMPLETED`, exit `0:0`, elapsed `02:41:51`.
- TD R=0.25 segment 2 `2520971` started on `16` nodes, loaded
  `/checkpoints/iter_0000952`, and resumed at iteration `953`.
  Optimizer/schedule continuity is clean: observed LR stayed
  `5.500000E-05`, skipped iterations `0`, NaN iterations `0`.
- First resumed TD R=0.25 training lines were finite:
  - iter `953`, loss `2.031282`;
  - iter `954`, loss `2.031653`;
  - iter `959`, loss `2.041158`.
- Resource note from live `scontrol`/`sacct`: production training jobs are
  using `16` nodes / `64` GPUs per arm as intended. The completed preprocessing
  jobs were CPU-style workloads, but they were submitted to `normal` and
  allocated full GPU nodes; this did not affect data integrity, but it is a
  real efficiency issue to fix before any future rebuild. Checksum sidecars run
  CPU-only on `xfer`; native/BPB eval sidecars use the existing GPU eval
  wrappers.
- LR sweep remains gated on replay/control results and user choice of `R*`.

R0.35 retry recovered; R0.15 first segment complete `2026-06-11T21:48Z`:

- TD R=0.35 replacement segment `2521904` started on `16` nodes, loaded
  `/checkpoints/iter_0000952`, built the 9 extra-validation loaders, and
  resumed training at iteration `953`.
- Retry health is clean: LR stayed `5.500000E-05`, skipped iterations `0`,
  NaN iterations `0`, with finite first resumed lines:
  - iter `953`, loss `1.992068`;
  - iter `954`, loss `1.974223`;
  - iter `959`, loss `2.009967`.
- TD R=0.15 segment 1 `2520974` reached `iter_0000952`, saved checkpoint, and
  completed successfully: Slurm state `COMPLETED`, exit `0:0`, elapsed
  `02:42:44`.
- TD R=0.15 segment 2 `2520975` is dependency-satisfied and pending on
  priority/resources.
- TD R=0.15 watcher `2521312` detected `iter_0000952` and submitted
  `curr-4.0B` sidecars:
  - convert `2522050`;
  - GreekMMLU native `2522051`;
  - code BPB `2522052`;
  - math BPB `2522053`;
  - checksum `2522054`.
- `curr-4.0B` GreekMMLU headlines available so far:
  - vanilla: `0.5227873978` (`8695/16632`);
  - TD R=0.35: `0.5370370370` (`8932/16632`);
  - TD R=0.25: `0.5330086580` (`8865/16632`);
  - TD R=0.15: pending sidecars.
- `curr-4.0B` StarCoder sidecar BPB available so far:
  - vanilla: `0.2668581197`;
  - TD R=0.35: `0.2672921695`;
  - TD R=0.25: `0.2686852364`;
  - TD R=0.15: pending sidecars.
- R0.15 segment-1 stderr again showed distributed rendezvous shutdown warnings
  after the successful save; Slurm marked the job `COMPLETED`, so no
  intervention was needed.
- LR sweep remains intentionally unlaunched.

R0.15 `curr-4.0B` sidecars complete; vanilla `curr-5.0B` sidecars launched `2026-06-11T22:00Z`:

- TD R=0.15 `iter_0000952` sidecars all completed successfully:
  - convert `2522050`: `COMPLETED`, exit `0:0`, elapsed `00:01:33`;
  - GreekMMLU native `2522051`: `COMPLETED`, exit `0:0`, elapsed `00:08:50`;
  - code BPB `2522052`: `COMPLETED`, exit `0:0`, elapsed `00:01:08`;
  - math BPB `2522053`: `COMPLETED`, exit `0:0`, elapsed `00:01:03`;
  - checksum `2522054`: `COMPLETED`, exit `0:0`, elapsed `00:04:13`.
- Completed `curr-4.0B` GreekMMLU table:
  - vanilla: `0.5227873978` (`8695/16632`);
  - TD R=0.35: `0.5370370370` (`8932/16632`);
  - TD R=0.25: `0.5330086580` (`8865/16632`);
  - TD R=0.15: `0.5249519000` (`8731/16632`).
- Completed `curr-4.0B` StarCoder BPB table:
  - vanilla: `0.2668581197`;
  - TD R=0.35: `0.2672921695`;
  - TD R=0.25: `0.2686852364`;
  - TD R=0.15: `0.2698321304`.
- TD R=0.15 `curr-4.0B` math BPB on the existing math held-out sample:
  `0.5448585028`.
- Vanilla watcher `2521309` detected `iter_0001190` and submitted the
  `curr-5.0B` sidecar bundle:
  - convert `2522158`;
  - GreekMMLU native `2522159`;
  - code BPB `2522160`;
  - math BPB `2522161`;
  - checksum `2522162`.
- Live training status remains healthy for the three active segment-2 arms:
  vanilla `2520961`, TD R=0.25 `2520971`, and TD R=0.35 retry `2521904` are
  all running on `16` nodes with finite losses and skipped/NaN iterations at
  `0`.
- TD R=0.15 segment 2 `2520975` remains dependency-satisfied and pending for
  `16` nodes with reason `(Resources)`.
- LR sweep remains intentionally unlaunched.

All four arms running; Clariden SSH visibility degraded `2026-06-11T22:18Z`:

- TD R=0.15 segment 2 `2520975` started on `16` nodes.
- Its startup/resume checks passed before the monitoring outage:
  - loaded `/checkpoints/iter_0000952`;
  - built all 9 extra validation datasets;
  - resumed at iteration `953`;
  - LR stayed `5.500000E-05`;
  - skipped iterations `0`, NaN iterations `0`.
- First resumed TD R=0.15 training lines were finite and then settled to the
  normal per-iteration runtime:
  - iter `953`, loss `2.059436`;
  - iter `954`, loss `2.044293`;
  - iter `963`, loss `2.062634`;
  - iter `974`, loss `2.047405`.
- Last confirmed live queue before the Clariden SSH outage had all four
  segment-2 arms running concurrently on `16` nodes each:
  - vanilla `2520961`;
  - TD R=0.35 retry `2521904`;
  - TD R=0.25 `2520971`;
  - TD R=0.15 `2520975`.
- Vanilla `curr-5.0B` sidecars at that point:
  - convert `2522158`: `COMPLETED`, exit `0:0`, elapsed `00:01:30`;
  - code BPB `2522160`: `COMPLETED`, exit `0:0`, elapsed `00:01:08`;
  - math BPB `2522161`: `COMPLETED`, exit `0:0`, elapsed `00:01:03`;
  - checksum `2522162`: `COMPLETED`, exit `0:0`, elapsed `00:04:04`;
  - GreekMMLU native `2522159`: still `RUNNING` at last visibility
    (`~12.5` minutes elapsed), headline not yet written.
- Monitoring outage evidence:
  - local CSCS cert remains valid until `2026-06-12T16:48:54`;
  - `ssh ela` works immediately;
  - from `ela`, `clariden.alps.cscs.ch` / `clariden.cscs.ch` /
    `clariden.plb.cscs.ch` / `clariden-ln004c.cscs.ch` / `172.28.39.147`
    all timed out on TCP port `22`;
  - `ela` does not provide `squeue`/`sacct`, so Slurm status is temporarily
    unavailable from there.
- Started a timeout-safe local monitor that retries Clariden SSH once per
  minute and will resume queue visibility when the login endpoint recovers.
- This log entry is local first; remote log sync is pending until Clariden SSH
  is reachable again.
- LR sweep remains intentionally unlaunched.

Clariden SSH outage still ongoing `2026-06-11T23:12Z`:

- Timeout-safe monitor continued probing Clariden once per minute.
- Repeated probes through `ela` continued to fail with
  `Connection timed out during banner exchange`.
- Fresh `ssh ela` sanity check continued to succeed (`ela5` observed), and
  `nc -zvw5 clariden.alps.cscs.ch 22` from `ela5` still timed out.
- No Slurm actions were taken blind during the outage.
- Remote log sync remains pending until Clariden SSH returns.

User-side CSCS console evidence confirms SSH/login degradation `2026-06-11T23:45Z`:

- User inspected the CSCS UI health details for Clariden and reported all
  relevant checks as `unhealthy`.
- The inspection message was:
  `TimeoutLimitExceeded: SSH connection timeout limit exceeded.`
- This matches the local `home -> ela -> Clariden` failure mode:
  repeated SSH banner timeouts and TCP port-22 timeouts from `ela`.
- Interpretation: this is confirmed as a CSCS-side Clariden SSH/login
  reachability outage. It does not by itself prove that already-running Slurm
  jobs on compute nodes have failed.
- Continued policy: do not cancel/requeue blind; keep timeout-bounded SSH
  retries active and reconcile Slurm/job/log state when Clariden SSH returns.

Second-hour outage heartbeat `2026-06-12T00:13Z`:

- Timeout-safe monitor continued probing Clariden via `ela` roughly once per
  minute.
- Probes from `2026-06-11T23:47Z` through `2026-06-12T00:13Z` all failed with
  `Connection timed out during banner exchange`.
- No new Slurm/job state could be observed because Clariden SSH remained
  unreachable and `ela` has no `squeue`/`sacct`.
- No blind Slurm actions were taken.
- Remote log sync remains pending until Clariden SSH returns.

Third-hour outage heartbeat `2026-06-12T01:00Z`:

- Timeout-safe monitor continued probing Clariden via `ela` roughly once per
  minute.
- Probes through `2026-06-12T00:59Z` continued to fail with
  `Connection timed out during banner exchange`.
- Fresh `ela` sanity checks during the outage continued to work, including
  landing on `ela6.cscs.ch`; TCP to `clariden.alps.cscs.ch:22` from `ela`
  continued to time out.
- No new Slurm/job state could be observed and no blind Slurm actions were
  taken.
- Remote log sync remains pending until Clariden SSH returns.

Fourth-hour outage heartbeat `2026-06-12T02:01Z`:

- Timeout-safe monitor continued probing Clariden via `ela`.
- Probes through `2026-06-12T02:01Z` continued to fail with
  `Connection timed out during banner exchange`.
- No new Slurm/job state could be observed and no blind Slurm actions were
  taken.
- Remote log sync remains pending until Clariden SSH returns.

Fifth-hour outage heartbeat `2026-06-12T03:01Z`:

- Timeout-safe monitor continued probing Clariden via `ela`.
- Probes through `2026-06-12T03:01Z` continued to fail with
  `Connection timed out during banner exchange`.
- No new Slurm/job state could be observed and no blind Slurm actions were
  taken.
- Remote log sync remains pending until Clariden SSH returns.

Sixth-hour outage heartbeat `2026-06-12T04:02Z`:

- Timeout-safe monitor continued probing Clariden via `ela`.
- Probes through `2026-06-12T04:02Z` continued to fail with
  `Connection timed out during banner exchange`.
- Fresh `ela` sanity check still worked (`ela6.cscs.ch` observed), while TCP
  to `clariden.alps.cscs.ch:22` from `ela` still timed out.
- No new Slurm/job state could be observed and no blind Slurm actions were
  taken.
- Remote log sync remains pending until Clariden SSH returns.

Seventh-hour outage heartbeat `2026-06-12T05:01Z`:

- Timeout-safe monitor continued probing Clariden via `ela`.
- Probes through `2026-06-12T05:01Z` continued to fail with
  `Connection timed out during banner exchange`.
- No new Slurm/job state could be observed and no blind Slurm actions were
  taken.
- Remote log sync remains pending until Clariden SSH returns.

Eighth-hour outage heartbeat `2026-06-12T06:02Z`:

- Timeout-safe monitor continued probing Clariden via `ela`.
- Probes through `2026-06-12T06:02Z` continued to fail with
  `Connection timed out during banner exchange`.
- No new Slurm/job state could be observed and no blind Slurm actions were
  taken.
- Remote log sync remains pending until Clariden SSH returns.

Ninth-hour outage heartbeat `2026-06-12T07:00Z`:

- Timeout-safe monitor continued probing Clariden via `ela`.
- Probes through `2026-06-12T06:59Z` continued to fail with
  `Connection timed out during banner exchange`.
- No new Slurm/job state could be observed and no blind Slurm actions were
  taken.
- Remote log sync remains pending until Clariden SSH returns.

Clariden login-node access recovered, but compute remains maintenance-blocked
`2026-06-12T08:05Z`:

- The load-balanced `clariden` SSH alias exposed rotated host keys after the
  outage, so monitoring was moved to explicit login node
  `clariden-ln001.cscs.ch` through `ela`.
- Verified direct login-node access with the CSCS certificate from `home`:
  `clariden-ln001` returned `date`, `hostname`, and Slurm commands.
- Reconciled the recovery queue. The fresh 16-node segment-2 recovery jobs are
  still queued, not lost:
  - vanilla `2522335`;
  - TD replay `R=0.35` `2522338`;
  - TD replay `R=0.25` `2522341`;
  - TD replay `R=0.15` `2522344`.
- Their downstream segment-3/segment-4 jobs remain pending on dependencies:
  `2522336-2522337`, `2522339-2522340`, `2522342-2522343`,
  `2522345-2522346`.
- All four segment-2 recovery jobs are pending with
  `(ReqNodeNotAvail, Reserved for maintenance)`.
- `sinfo -p normal` shows the normal partition broadly unavailable:
  `26` nodes `drain$`, `1145` nodes `down$`, and `188` nodes `maint`.
- User-side portal evidence reported the same health class earlier:
  `TimeoutLimitExceeded: SSH connection timeout limit exceeded.`
- Decision: do not cancel/requeue. Keep the recovery chains queued, monitor via
  explicit login nodes, and verify resume logs as soon as nodes return.
- Remote log sync is now possible through explicit login nodes and should be
  retried after this local entry is committed.

Clariden reservation check `2026-06-12T08:14Z`:

- `scontrol show reservation` confirms an active site/platform reservation:
  `ReservationName=poweron`, `StartTime=09:16:21`, `EndTime=Sun 09:16`,
  `Duration=2-00:00:00`.
- The reservation covers `1386` nodes / `398848` CPUs with
  `Flags=MAINT,IGNORE_JOBS,SPEC_NODES,ALL_NODES`.
- Reservation accounts are `root,csstaff`, so user jobs cannot consume those
  nodes while the reservation is active.
- Interpretation: the current bottleneck is CSCS maintenance/power-on state,
  not insufficient parallelism or a local queue-script error.
- Continued action: leave recovery chains queued, keep polling through explicit
  login nodes, and inspect training logs immediately once the first recovery
  segment enters `RUNNING`.

Watcher relaunch precheck `2026-06-12T08:29Z`:

- Checked whether the GreekMMLU-only sidecar watchers could be relaunched while
  GPU training waits.
- `sinfo -p xfer` shows both xfer nodes `down$`, so the CPU-only watcher jobs
  cannot run yet either.
- Watcher state directories are intact:
  - vanilla has submitted markers through `iter_1190`;
  - TD replay `R=0.35`, `R=0.25`, and `R=0.15` have submitted markers through
    `iter_952`.
- Implication: when xfer returns, relaunch the four watchers using the existing
  state directories so they continue from the next missing cadence checkpoints
  without duplicating prior 1B-4B sidecars.
- Separate follow-up needed: vanilla `iter_1190` GreekMMLU native sidecar failed
  during the outage but the watcher state already marks `iter_1190` submitted,
  so recover that native sidecar explicitly rather than relying on watcher
  resubmission.

CSCS certificate refresh precheck `2026-06-12T09:25Z`:

- Current local CSCS SSH certificate is valid from `2026-06-11T16:48:54` to
  `2026-06-12T16:48:54`.
- Direct `home -> auth.cscs.ch:443` is timing out, so direct `cscs-key
  --headless sign` cannot reliably poll the CSCS token endpoint from `home`.
- `macbook` relay path is unavailable from `home` right now
  (`No route to host`).
- Added a local SSH alias `clariden-ln001` in `~/.ssh/config` pointing at the
  fixed login node through `ela` with the CSCS key/certificate, avoiding the
  load-balanced `clariden` alias.
- Verified `clariden-ln001` can reach `auth.cscs.ch`.
- Started the allowlisted CSCS auth CONNECT proxy through `clariden-ln001` and
  retried `cscs-key --headless sign` with proxy env vars.
- The proxied device flow produced code `NLPH-GPLR`, but the device
  authorization expired before browser approval completed.
- Proxy was stopped cleanly. Monitoring continues with the still-valid current
  certificate; refresh should be retried with a fresh device code before
  `2026-06-12T16:48:54` if the maintenance window continues.

Recovery jobs began after maintenance release `2026-06-12T09:58Z`:

- Opened a persistent SSH master to `clariden-ln001` while the current CSCS
  certificate is still valid, to reduce risk from later cert expiry during a
  long recovery watch.
- The first recovery jobs entered `RUNNING` after the maintenance block:
  - vanilla `2522335`;
  - TD replay `R=0.35` `2522338`;
  - TD replay `R=0.15` `2522344`.
- Vanilla `2522335` health check:
  - loaded checkpoint `1190`;
  - resumed at iteration `1191`;
  - built all 9 extra-valid datasets;
  - LR remained `5.5e-5`;
  - losses finite, skipped iterations `0`, NaN iterations `0`.
- TD `R=0.35` `2522338` health check:
  - loaded checkpoint `1071`;
  - resumed at iteration `1072`;
  - built all 9 extra-valid datasets;
  - LR remained `5.5e-5`;
  - losses finite, skipped iterations `0`, NaN iterations `0`.
- TD `R=0.15` `2522344` health check:
  - loaded checkpoint `952`;
  - resumed at iteration `953`;
  - built all 9 extra-valid datasets;
  - LR remained `5.5e-5`;
  - losses finite, skipped iterations `0`, NaN iterations `0`.
- TD `R=0.25` recovery segment `2522341` failed after `00:01:42` with CUDA
  `unspecified launch failure` on rank 22, after which Slurm cancelled the
  step. This appears consistent with an unstable post-maintenance allocation,
  not a data/checkpoint failure.
- Confirmed TD `R=0.25` latest checkpoint is still `1071`.
- Cancelled broken stale downstream deps `2522342` and `2522343`.
- Submitted replacement TD `R=0.25` chain with the failed allocation's nodes
  excluded:
  - segment 2 retry `2522485`, exit iteration `1904`, phase-1 HPLT config;
  - segment 3 retry `2522486`, exit iteration `2261`, phase-1 HPLT config;
  - segment 4 retry `2522487`, exit iteration `3218`, phase-2 GlossAPI config
    with `RESET_DATA_INDEX=1`.
- Replacement segment `2522485` is pending for `Resources`.
- `xfer` remains unavailable (`down*`), so GreekMMLU watcher relaunch is still
  pending.

TD `R=0.25` replacement retry verified healthy `2026-06-12T10:18Z`:

- Replacement segment `2522485` started on a fresh allocation after excluding
  the failed node set from `2522341`.
- Health check:
  - loaded checkpoint `1071`;
  - resumed at iteration `1072`;
  - built all 9 extra-valid datasets;
  - LR remained `5.5e-5`;
  - losses finite, skipped iterations `0`, NaN iterations `0`.
- All four replay-sweep arms are now in a clean Slurm state:
  - vanilla `2522335` running;
  - TD `R=0.35` `2522338` running;
  - TD `R=0.25` `2522485` running;
  - TD `R=0.15` `2522344` running.
- Watcher relaunch still waits on `xfer`, which remains `down*`.

One-shot sidecar recovery while xfer watchers are down `2026-06-12T10:34Z`:

- Since `xfer` is still down and cannot host the long-lived watchers, submitted
  ready cadence sidecars directly from the login node with `sbatch` rather than
  waiting for watcher loops.
- Fixed an initial one-shot submission mistake: the standalone legacy
  `EVAL_DIR` lacks `run_native_greek_mcq_eval.sbatch`; the correct deployed eval
  path is under
  `/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval`.
- Submitted TD `R=0.35` `iter_1190` / `curr-5.0B` sidecars and marked watcher
  state `iter_1190.submitted` to prevent later duplication:
  - convert `2522574`;
  - native GreekMMLU `2522575`;
  - code BPB `2522576`;
  - math BPB `2522578`;
  - checksum `2522579` (`xfer`, pending until xfer nodes return).
- Submitted vanilla `iter_1190` native GreekMMLU retry only, against the
  already-converted HF checkpoint:
  - native retry `2522580`.
- Existing vanilla `iter_1190` convert/code/math/checksum jobs remain reused
  from the pre-outage sidecar run.

Additional manual sidecar catch-up while xfer remains down `2026-06-12T10:49Z`:

- Submitted ready cadence checkpoints directly and marked their watcher state
  files to avoid duplicate watcher submissions later.
- Vanilla `iter_1428` / `curr-6.0B`:
  - convert `2522586`;
  - native GreekMMLU `2522587`;
  - code BPB `2522588`;
  - math BPB `2522590`;
  - checksum `2522591`.
- TD `R=0.15` `iter_1190` / `curr-5.0B`:
  - convert `2522592`;
  - native GreekMMLU `2522594`;
  - code BPB `2522595`;
  - math BPB `2522596`;
  - checksum `2522597`.
- TD `R=0.25` `iter_1190` / `curr-5.0B`:
  - convert `2522599`;
  - native GreekMMLU `2522600`;
  - code BPB `2522601`;
  - math BPB `2522603`;
  - checksum `2522604`.
- At submission time the new conversion jobs had begun running where resources
  were available; checksum jobs remain xfer-dependent while `xfer` is down.

Manual sidecar results/catch-up `2026-06-12T11:13Z`:

- Completed native GreekMMLU one-shot results:
  - vanilla `iter_1190` / `curr-5.0B`: `0.5353535354`
    (`8904/16632`);
  - vanilla `iter_1428` / `curr-6.0B`: `0.5337902838`
    (`8878/16632`);
  - TD `R=0.35` `iter_1190` / `curr-5.0B`: `0.5491822992`
    (`9134/16632`);
  - TD `R=0.25` `iter_1190` / `curr-5.0B`: `0.5511664262`
    (`9167/16632`);
  - TD `R=0.15` `iter_1190` / `curr-5.0B`: `0.5270562771`
    (`8766/16632`).
- Code/math BPB sidecars for these manual submissions completed successfully;
  checksum jobs remain queued on `xfer`.
- `xfer` still reports both nodes `down*`, so watcher relaunch remains pending.
- Submitted next ready cadence checkpoints manually:
  - vanilla `iter_1666` / `curr-7.0B`: convert `2522698`, native
    `2522699`, code BPB `2522700`, math BPB `2522701`, checksum `2522702`;
  - TD `R=0.35` `iter_1428` / `curr-6.0B`: convert `2522703`, native
    `2522704`, code BPB `2522705`, math BPB `2522706`, checksum `2522707`.

Manual sidecar results/catch-up `2026-06-12T11:45Z`:

- Submitted additional ready cadence checkpoints while `xfer` remained down:
  - TD `R=0.15` `iter_1428` / `curr-6.0B`: convert `2522709`, native
    `2522710`, code BPB `2522711`, math BPB `2522712`, checksum `2522713`;
  - TD `R=0.25` `iter_1428` / `curr-6.0B`: convert `2522743`, native
    `2522744`, code BPB `2522745`, math BPB `2522746`, checksum `2522747`;
  - TD `R=0.35` `iter_1666` / `curr-7.0B`: convert `2522823`, native
    `2522824`, code BPB `2522825`, math BPB `2522826`, checksum `2522827`.
- Completed native GreekMMLU results:
  - vanilla `iter_1666` / `curr-7.0B`: `0.5274170274`
    (`8772/16632`);
  - TD `R=0.35` `iter_1428` / `curr-6.0B`: `0.5233886484`
    (`8705/16632`);
  - TD `R=0.25` `iter_1428` / `curr-6.0B`: `0.5367364117`
    (`8927/16632`);
  - TD `R=0.15` `iter_1428` / `curr-6.0B`: `0.5175565176`
    (`8608/16632`).
- Code/math BPB sidecars for completed items succeeded; checksums remain queued
  on `xfer`.

Segment/checkpoint catch-up `2026-06-12T11:59Z`:

- Vanilla recovery segment `2522335` completed successfully at `iter_1904`;
  downstream segment `2522336` is pending for priority.
- Submitted new ready cadence sidecars while `xfer` remained down:
  - vanilla `iter_1904` / `curr-8.0B`: convert `2522868`, native `2522869`,
    code BPB `2522870`, math BPB `2522871`, checksum `2522872`;
  - TD `R=0.25` `iter_1666` / `curr-7.0B`: convert `2522873`, native
    `2522874`, code BPB `2522875`, math BPB `2522876`, checksum `2522877`;
  - TD `R=0.15` `iter_1666` / `curr-7.0B`: convert `2522878`, native
    `2522879`, code BPB `2522880`, math BPB `2522881`, checksum `2522882`.
- TD `R=0.35` latest checkpoint at this time was `1785`; its already-submitted
  `iter_1666` sidecars were still pending.

Sidecar scheduling guard `2026-06-12T12:07Z`:

- Vanilla segment 3 (`2522336`) was pending for priority after segment 2
  completed, while many one-node normal-partition sidecars were also pending.
- To avoid sidecar jobs competing with 16-node training continuation, held
  pending normal sidecars:
  `2522823`, `2522824`, `2522825`, `2522826`, `2522868`, `2522869`,
  `2522870`, `2522871`, `2522873`, `2522874`, `2522875`, `2522876`,
  `2522878`, `2522879`, `2522880`, `2522881`, `2522901`, `2522902`,
  `2522903`, `2522904`.
- These are intentionally paused with `JobHeldUser`; release them once training
  continuation has enough room or at the next natural evaluation catch-up point.
- Checksum sidecars remain xfer-bound and were not the resource concern.

Portal degradation check `2026-06-12T12:07Z`:

- User reported `portal.cscs.ch` Clariden cards as `unhealthy`, with inspection
  message `TimeoutLimitExceeded: SSH connection timeout limit exceeded`.
- Direct login-node SSH through the pinned `clariden-ln001` master still worked;
  `squeue`, checkpoint reads, and `sinfo` all returned normally.
- Live Slurm state at check time:
  - Running 16-node training jobs: TD `R=0.25` recovery segment 2 (`2522485`,
    latest checkpoint still `1666`) and TD `R=0.15` recovery segment 2
    (`2522344`, latest checkpoint `1785`).
  - Vanilla and TD `R=0.35` had completed segment 2 at checkpoint `1904`; their
    segment-3 jobs (`2522336`, `2522339`) were pending for priority.
  - xfer remained down, so checksum sidecars and watcher relaunch still needed
    to wait.
- Decision: treat the portal cards as console/cluster health symptoms, not as a
  reason to cancel or requeue healthy Slurm jobs. Keep normal sidecars held and
  prioritize 16-node training continuation.

R0.15 recovery segment-2 completion `2026-06-12T12:28Z`:

- TD `R=0.15` recovery segment 2 (`2522344`) reached checkpoint `1904` and
  completed cleanly: `COMPLETED`, exit `0:0`, elapsed `02:42:59` on 16 nodes.
- Dependent TD `R=0.15` segment 3 (`2522345`, 1905 -> 2261) was released from
  dependency and is now pending for priority.
- Held sidecars were left untouched; `iter_1904` sidecars for TD `R=0.15` are
  intentionally deferred until they will not compete with training continuation.

Segment-3 resume verification `2026-06-12T12:46Z`:

- TD `R=0.25` recovery segment 2 (`2522485`) reached checkpoint `1904` and
  completed cleanly: `COMPLETED`, exit `0:0`, elapsed `02:23:57` on 16 nodes.
- Vanilla (`2522336`), TD `R=0.35` (`2522339`), and TD `R=0.15` (`2522345`)
  segment-3 jobs all started on 16 nodes.
- Startup log check:
  - each loaded checkpoint `1904`;
  - each resumed at iteration `1905`;
  - all nine extra validation datasets were built;
  - LR stayed at `5.5e-5`;
  - no phase-2 reset guard fired in segment 3, as expected;
  - live losses were finite with skipped/NaN iterations still `0`.
- TD `R=0.25` segment 3 (`2522486`) was released from dependency and is pending
  for 16-node resources.

R0.25 segment-3 start `2026-06-12T13:08Z`:

- TD `R=0.25` segment 3 (`2522486`) started on 16 nodes.
- Startup check matched the other segment-3 jobs: loaded checkpoint `1904`,
  resumed at iteration `1905`, built the extra validation datasets, LR remained
  `5.5e-5`, no phase-2 reset guard fired, and live losses were finite with
  skipped/NaN iterations still `0`.
- By this point all four arms were again in 16-node training: vanilla, TD
  `R=0.35`, TD `R=0.25`, and TD `R=0.15`.

Phase-boundary arrival begins `2026-06-12T13:36Z`:

- Vanilla, TD `R=0.35`, and TD `R=0.15` wrote checkpoint `2261`.
- Vanilla segment 3 (`2522336`) completed cleanly at the boundary: `COMPLETED`,
  exit `0:0`, elapsed `01:03:55` on 16 nodes.
- Vanilla segment 4 (`2522337`) was released from dependency and is pending for
  priority on 16 nodes.
- TD `R=0.35` and TD `R=0.15` were still in their segment-3 completion path at
  this check; TD `R=0.25` was still running toward checkpoint `2261`.
- Next critical check: segment 4 must load checkpoint `2261`, switch to the
  OpenArchives phase, and fire the reset guard exactly once.

Replay-sweep training completion + eval catch-up `2026-06-12T17:29Z`:

- All four fixed-total 13.5B replay-sweep training runs completed:
  - vanilla segment 4 (`2522337`): `COMPLETED`, exit `0:0`, elapsed
    `02:41:59`, 16 nodes.
  - TD `R=0.35` segment 4 (`2522340`): `COMPLETED`, exit `0:0`, elapsed
    `02:43:33`, 16 nodes.
  - TD `R=0.25` segment 4 (`2522487`): `COMPLETED`, exit `0:0`, elapsed
    `02:44:10`, 16 nodes.
  - TD `R=0.15` segment 4 (`2522346`): `COMPLETED`, exit `0:0`, elapsed
    `02:44:09`, 16 nodes.
- Latest checkpoint files for all four runs read `3218`. Final logs showed
  checkpoint `3218` saved successfully, program exit at iteration `3218`, and
  skipped/NaN iterations remained `0`.
- The held conversion/native sidecars needed for GreekMMLU catch-up were
  released: `2522823`, `2522824`, `2522868`, `2522869`, `2522873`, `2522874`,
  `2522878`, `2522879`, `2522901`, `2522902`.
- Optional code/math BPB held jobs were left held intentionally while preparing
  the fast replay-sweep decision table; old-data forgetting will be read from
  in-training extra-validation losses.
- Submitted GreekMMLU-only catch-up sidecars for missing cadence checkpoints:
  - vanilla: `2142`, `2261`, `2380`, `2618`, `2856`, `3094`, `3218`;
  - TD `R=0.35`: `2142`, `2261`, `2380`, `2618`, `2856`, `3094`, `3218`;
  - TD `R=0.25`: `1904`, `2142`, `2261`, `2380`, `2618`, `2856`, `3094`,
    `3218`;
  - TD `R=0.15`: `1904`, `2142`, `2261`, `2380`, `2618`, `2856`, `3094`,
    `3218`.
- Catch-up job range: `2524209` through `2524268` for conversion/native pairs.
  These were submitted with `NATIVE_BENCHMARKS=greekmmlu`,
  `SUBMIT_GREEK_NLP=0`, `SUBMIT_BPB=0`, `SUBMIT_RETENTION=0`, and
  `SUBMIT_CHECKSUM=0`. Watcher `.submitted` markers were written for the
  manually submitted checkpoints to avoid duplicate watcher submissions later.
- Queue health after submission: 35 eval jobs running, 30 native eval jobs
  pending on conversion dependencies, 25 GreekMMLU summary files already present
  from earlier cadence, and xfer/checksum work still noncritical for R-choice.

Peak-LR sweep implementation start `2026-06-12T18:45Z`:

- User selected the next replay split for the LR sweep: 20% foreign replay plus
  1% old-Greek replay, i.e. 79/20/1 after Megatron normalizes the blend.
- Confirmed no Slurm jobs were running or pending before starting the next
  wave.
- The old peak-LR wrapper would have reused the combined `replay_only` binary
  through a single `R=0.25` weight, which would not realize the selected
  old-Greek slot. Patched the v2 train envs to support split replay weights:
  - `FOREIGN_REPLAY_R=20/79=0.253164557`;
  - `OLD_GREEK_REPLAY_R=1/79=0.012658228`.
- Added a CPU-only split/tokenize job that reuses the already decontaminated
  and anonymized `replay_only_final.jsonl`, splitting by preserved
  `metadata.source`:
  - `foreign_replay_only_final.jsonl`;
  - `old_greek_replay_only_final.jsonl`.
- Added hard preflight checks to `sweep_peak_lr.sh` so the LR arms will not
  launch until the split replay Megatron binaries exist for both base/ext
  tokenizations.
- Synced the patched v2 scripts/log to Clariden and submitted the CPU-only
  split/tokenize job `2524425` on `normal`. The job requests one node, 64 CPUs,
  240G memory, and no GPUs.
- Submitted dependent tiny launcher `2524430` (`afterok:2524425`) to run
  `DRY_RUN=0 CONFIRM_LAUNCH=1 bash train/sweep_peak_lr.sh` once the split
  replay binaries exist. This should submit the four TD peak-LR chains at
  `LR_PEAK in {2.75e-5, 5.5e-5, 8.25e-5, 1.1e-4}` with 16 nodes per segment.
- `2524425` started on `nid007117` at about `2026-06-12T20:55Z` Clariden
  time. Early log was healthy: split progress reached 2.0M rows with no stderr.
- Split completed cleanly inside `2524425`:
  - foreign replay: 3,808,235 rows, 13.21B chars (~3.30B token estimate);
  - old-Greek replay: 1,223,498 rows, 3.04B chars (~0.76B token estimate);
  - missing source labels: 0.
- The split outputs are enough for the fixed-total 13.5B LR sweep consumption
  under the selected 79/20/1 mix (foreign needs ~2.70B tokens; old Greek needs
  ~0.135B tokens). The job then started tokenizing `foreign_replay_only` with
  the base tokenizer.

Peak-LR prerequisite progress `2026-06-12T22:10+03:00`:

- Split/tokenize job `2524425` is still healthy and CPU-only on `nid007117`
  (`RUNNING`, elapsed `00:15:16`, 1 node, no GPUs). Dependent launcher
  `2524430` remains pending on `afterok:2524425`.
- Completed split/tokenized output so far:
  - `foreign_replay_only_base_text_document.bin`: 16G;
  - `foreign_replay_only_base_text_document.idx`: 73M.
- `foreign_replay_only_ext` tokenization is in progress. Last observed stderr
  progress was about 1.516M / 3.808M documents at ~8.2k docs/s, with a 6.4G
  partial `.bin`.
- Remaining prerequisite work before the LR launcher fires: finish
  `foreign_replay_only_ext`, then tokenize `old_greek_replay_only` with base
  and ext tokenizers.

Peak-LR sweep launched `2026-06-12T22:20+03:00`:

- Split/tokenize job `2524425` completed successfully (`0:0`) in `00:24:58`.
  Dependent launcher `2524430` completed successfully (`0:0`) in `00:00:17`.
- Final split replay Megatron binaries exist for both tokenizers:
  - `foreign_replay_only_base_text_document.{bin,idx}`: 16G / 73M;
  - `foreign_replay_only_ext_text_document.{bin,idx}`: 16G / 73M;
  - `old_greek_replay_only_base_text_document.{bin,idx}`: 4.3G / 24M;
  - `old_greek_replay_only_ext_text_document.{bin,idx}`: 2.7G / 24M.
- Submitted four TD peak-LR chains at the selected replay split
  (`FOREIGN_REPLAY_R=0.253164557`, `OLD_GREEK_REPLAY_R=0.012658228`,
  metadata `R=0.25`):
  - `curr_td_f20_g1_lr2.75e-5_20260612T191957Z`, segment jobs
    `2524479` -> `2524480` -> `2524481` -> `2524482`;
  - `curr_td_f20_g1_lr5.5e-5_20260612T191957Z`, segment jobs
    `2524483` -> `2524484` -> `2524485` -> `2524486`;
  - `curr_td_f20_g1_lr8.25e-5_20260612T191957Z`, segment jobs
    `2524488` -> `2524489` -> `2524490` -> `2524491`;
  - `curr_td_f20_g1_lr1.1e-4_20260612T191957Z`, segment jobs
    `2524492` -> `2524493` -> `2524494` -> `2524495`.
- First health check of the run logs confirmed the intended split replay data
  prefix in segment 1:
  `1.0 hplt_only_ext ... 0.253164557 foreign_replay_only_ext ...
  0.012658228 old_greek_replay_only_ext ...`, with `WORLD_SIZE=64` and 16
  nodes per segment.
- Submitted GreekMMLU-only eval watchers for the four LR arms:
  `2524499`, `2524500`, `2524501`, `2524502`.

Peak-LR watcher recovery `2026-06-12T22:30+03:00`:

- The four Slurm watcher jobs `2524499`-`2524502` were stuck on `xfer`
  because the partition only had down/reserved nodes
  (`ReqNodeNotAvail, Reserved for maintenance`); cancelled them.
- Tried a consolidated one-node `debug` watcher batch (`2524567`) to avoid
  four separate watcher allocations, but the CPU-only guard rejected it because
  `debug`/`normal`/`low` are GPU-node partitions. This was the correct guard
  behavior, so no override was used.
- Added and launched a home-side watcher:
  `scripts/home_poll_curriculum_greekmmlu_sidecars.sh`. It polls Clariden over
  SSH from `home`, writes the same remote `.submitted` markers under
  `$RUN_ROOT/${RUN_TAG}_sidecar_watch`, and submits only the actual
  conversion/native GreekMMLU sidecars when checkpoints appear.
- Initial `nohup` watcher PID `350396` completed one poll but did not survive
  the command wrapper detachment, so it was superseded rather than trusted.
- Active home watcher is now tmux session `cpt_lr_watch_20260612`, log
  `logs/home_greekmmlu_lr_watch_tmux_20260612T193148Z.log`. First poll
  succeeded: `submitted_now=0`, `waiting_this_pass=60`,
  `total_required=60`.
- The home watcher explicitly disables non-GreekMMLU side work for this LR
  sweep (`SUBMIT_GREEK_NLP=0`, `SUBMIT_RETENTION=0`, `SUBMIT_BPB=0`,
  `SUBMIT_CHECKSUM=0`, and code/math BPB defaults pointed at nonexistent
  paths).

Peak-LR first checkpoint handoff `2026-06-12T23:04+03:00`:

- All four first segments reached and saved checkpoint `iter_0000238`
  (`curr-1.0B`) while training continued; latest observed training iterations
  were around `245`-`250`, with skipped/NaN counters still `0`.
- Home watcher submitted exactly the intended GreekMMLU sidecars for `iter=238`:
  - `2.75e-5`: convert `2524829`, native GreekMMLU `2524830`;
  - `5.5e-5`: convert `2524831`, native GreekMMLU `2524832`;
  - `8.25e-5`: convert `2524833`, native GreekMMLU `2524834`;
  - `1.1e-4`: convert `2524835`, native GreekMMLU `2524836`.
- Convert jobs were running and native jobs were pending on conversion
  dependencies. Submit logs confirmed `NATIVE_BENCHMARKS=greekmmlu`,
  `SUBMIT_CHECKSUM=0`, and no BPB/checksum extra jobs were submitted.
- Found and fixed a watcher environment issue: the home watcher had not exported
  `RUN_ROOT` before invoking `submit_td_checkpoint_sidecars.sh`, so the
  `iter=238` eval outputs landed under the legacy
  `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/eval_<tag>`
  root. The outputs are valid and usable there. Patched the watcher to export
  `RUN_ROOT`, then restarted tmux session `cpt_lr_watch_20260612` with log
  `logs/home_greekmmlu_lr_watch_tmux_20260612T200415Z.log`; first patched poll
  saw `already_seen_this_pass=4`, `waiting_this_pass=56`, `total_done=4`.
- All four `iter=238` native GreekMMLU jobs completed successfully:
  - `2.75e-5`: `8053/16632 = 0.4842`;
  - `5.5e-5`: `8226/16632 = 0.4946`;
  - `8.25e-5`: `8222/16632 = 0.4943`;
  - `1.1e-4`: `8262/16632 = 0.4968`.
- Result artifacts exist under the legacy eval root for this first checkpoint,
  including `*_native_mcq_aggregate.json`, `*_native_mcq_headline.json`,
  `*_native_mcq_summary.csv`, and predictions JSONL for each arm.

Peak-LR second checkpoint handoff `2026-06-12T23:00+03:00`:

- All four arms reached checkpoint `iter_0000476` (`curr-2.0B`) and the home
  watcher submitted the second GreekMMLU sidecar pair for each arm. The
  watcher fix worked: `iter=476` eval outputs landed under
  `/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/eval_<tag>`.
- Sidecar jobs all completed successfully:
  - converts: `2524931`, `2524933`, `2524942`, `2524944`;
  - native GreekMMLU: `2524932`, `2524934`, `2524943`, `2524946`.
- GreekMMLU overall at `curr-2.0B`:
  - `2.75e-5`: `8503/16632 = 0.5113`;
  - `5.5e-5`: `8532/16632 = 0.5130`;
  - `8.25e-5`: `8620/16632 = 0.5183`;
  - `1.1e-4`: `8580/16632 = 0.5159`.
- Training remained healthy past the checkpoint, with observed iterations in
  the mid-500s and skipped/NaN counters still `0`.

Peak-LR third checkpoint handoff `2026-06-12T23:42+03:00`:

- The first-segment LR arms are still running on 16 nodes each and have reached
  the `iter_0000714` (`curr-3.0B`) checkpoint window, with skipped/NaN counters
  still `0`.
- Home watcher session `cpt_lr_watch_20260612` is alive and submitting
  GreekMMLU-only sidecars from
  `logs/home_greekmmlu_lr_watch_tmux_20260612T200415Z.log`.
- Submitted GreekMMLU sidecars for `iter=714`:
  - `2.75e-5`: convert `2525016`, native GreekMMLU `2525017`;
  - `5.5e-5`: convert `2525018`, native GreekMMLU `2525019`;
  - `8.25e-5`: convert `2525153`, native GreekMMLU `2525154`;
  - `1.1e-4`: convert `2525155`, native GreekMMLU `2525156`.
- All eight `iter=714` sidecar jobs completed successfully.
- GreekMMLU overall at `curr-3.0B`:
  - `2.75e-5`: `8232/16632 = 0.4949`;
  - `5.5e-5`: `8224/16632 = 0.4945`;
  - `8.25e-5`: `8220/16632 = 0.4942`;
  - `1.1e-4`: `8282/16632 = 0.4980`.
- Eval outputs for this checkpoint exist under
  `/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/eval_<tag>/iter_0000714`.
- Training remained healthy after the checkpoint, with observed iterations
  around `850`-`859` and skipped/NaN counters still `0`.

Peak-LR fourth checkpoint and segment-2 handoff `2026-06-13T00:15+03:00`:

- All four segment-1 jobs completed successfully and their segment-2 dependency
  jobs started on 16 nodes each:
  - `2.75e-5`: `2524479` completed, `2524480` running;
  - `5.5e-5`: `2524483` completed, `2524484` running;
  - `8.25e-5`: `2524488` completed, `2524489` running;
  - `1.1e-4`: `2524492` completed, `2524493` running.
- Segment-2 logs show checkpoints loaded at iteration `952`, the split replay
  blend still set to `hplt_only_ext + 0.253164557 foreign_replay_only_ext +
  0.012658228 old_greek_replay_only_ext`, and learning rates continuing at the
  configured peak values rather than restarting.
- Home watcher submitted `iter=952` (`curr-4.0B`) GreekMMLU sidecars for all
  four arms:
  - `2.75e-5`: convert `2525341`, native GreekMMLU `2525342`;
  - `5.5e-5`: convert `2525343`, native GreekMMLU `2525344`;
  - `8.25e-5`: convert `2525352`, native GreekMMLU `2525353`;
  - `1.1e-4`: convert `2525354`, native GreekMMLU `2525355`.
- All eight `iter=952` sidecar jobs completed successfully.
- GreekMMLU overall at `curr-4.0B`:
  - `2.75e-5`: `8452/16632 = 0.5082`;
  - `5.5e-5`: `8910/16632 = 0.5357`;
  - `8.25e-5`: `8463/16632 = 0.5088`;
  - `1.1e-4`: `8722/16632 = 0.5244`.
- Segment-2 training remained healthy after the handoff, with observed
  iterations around `1006`-`1023` and skipped/NaN counters still `0`.

Peak-LR fifth checkpoint `2026-06-13T01:00+03:00`:

- Home watcher submitted `iter=1190` (`curr-5.0B`) GreekMMLU sidecars for all
  four arms:
  - `2.75e-5`: convert `2525476`, native GreekMMLU `2525477`;
  - `5.5e-5`: convert `2525478`, native GreekMMLU `2525479`;
  - `8.25e-5`: convert `2525483`, native GreekMMLU `2525484`;
  - `1.1e-4`: convert `2525485`, native GreekMMLU `2525486`.
- All eight `iter=1190` sidecar jobs completed successfully.
- GreekMMLU overall at `curr-5.0B`:
  - `2.75e-5`: `8662/16632 = 0.5208`;
  - `5.5e-5`: `8785/16632 = 0.5282`;
  - `8.25e-5`: `8551/16632 = 0.5141`;
  - `1.1e-4`: `8700/16632 = 0.5231`.
- Segment-2 training remained healthy, with observed iterations around
  `1271`-`1292` and skipped/NaN counters still `0`.

Peak-LR sixth checkpoint `2026-06-13T01:39+03:00`:

- Home watcher submitted `iter=1428` (`curr-6.0B`) GreekMMLU sidecars for all
  four arms:
  - `2.75e-5`: convert `2525556`, native GreekMMLU `2525557`;
  - `5.5e-5`: convert `2525558`, native GreekMMLU `2525559`;
  - `8.25e-5`: convert `2525565`, native GreekMMLU `2525566`;
  - `1.1e-4`: convert `2525567`, native GreekMMLU `2525568`.
- All eight `iter=1428` sidecar jobs completed successfully.
- GreekMMLU overall at `curr-6.0B`:
  - `2.75e-5`: `8667/16632 = 0.5211`;
  - `5.5e-5`: `8936/16632 = 0.5373`;
  - `8.25e-5`: `8847/16632 = 0.5319`;
  - `1.1e-4`: `8872/16632 = 0.5334`.
- Segment-2 training remained healthy, with observed iterations around
  `1511`-`1532` and skipped/NaN counters still `0`.

Peak-LR seventh checkpoint `2026-06-13T02:19+03:00`:

- Home watcher submitted `iter=1666` (`curr-7.0B`) GreekMMLU sidecars for all
  four arms:
  - `2.75e-5`: convert `2525698`, native GreekMMLU `2525699`;
  - `5.5e-5`: convert `2525700`, native GreekMMLU `2525701`;
  - `8.25e-5`: convert `2525710`, native GreekMMLU `2525711`;
  - `1.1e-4`: convert `2525712`, native GreekMMLU `2525713`.
- All eight `iter=1666` sidecar jobs completed successfully.
- GreekMMLU overall at `curr-7.0B`:
  - `2.75e-5`: `8893/16632 = 0.5347`;
  - `5.5e-5`: `9218/16632 = 0.5542`;
  - `8.25e-5`: `9413/16632 = 0.5660`;
  - `1.1e-4`: `9137/16632 = 0.5494`.
- Segment-2 training remained healthy, with observed iterations around
  `1747`-`1773` and skipped/NaN counters still `0`.

Peak-LR eighth checkpoint and segment-3 handoff `2026-06-13T02:57+03:00`:

- All four segment-2 jobs completed successfully and their segment-3 dependency
  jobs started on 16 nodes each:
  - `2.75e-5`: `2524480` completed, `2524481` running;
  - `5.5e-5`: `2524484` completed, `2524485` running;
  - `8.25e-5`: `2524489` completed, `2524490` running;
  - `1.1e-4`: `2524493` completed, `2524494` running.
- Segment-3 logs show checkpoints loaded at iteration `1904`, the split replay
  blend still set to `hplt_only_ext + 0.253164557 foreign_replay_only_ext +
  0.012658228 old_greek_replay_only_ext`, and learning rates continuing at the
  configured peak values.
- Home watcher submitted `iter=1904` (`curr-8.0B`) GreekMMLU sidecars for all
  four arms:
  - `2.75e-5`: convert `2525753`, native GreekMMLU `2525754`;
  - `5.5e-5`: convert `2525755`, native GreekMMLU `2525756`;
  - `8.25e-5`: convert `2525759`, native GreekMMLU `2525760`;
  - `1.1e-4`: convert `2525761`, native GreekMMLU `2525762`.
- All eight `iter=1904` sidecar jobs completed successfully.
- GreekMMLU overall at `curr-8.0B`:
  - `2.75e-5`: `8886/16632 = 0.5343`;
  - `5.5e-5`: `9195/16632 = 0.5529`;
  - `8.25e-5`: `9463/16632 = 0.5690`;
  - `1.1e-4`: `9022/16632 = 0.5424`.
- Segment-3 training remained healthy after the handoff, with observed
  iterations around `1921`-`1950` and skipped/NaN counters still `0`.

Peak-LR ninth checkpoint `2026-06-13T03:57+03:00`:

- Home watcher submitted `iter=2142` (`curr-9.0B`) GreekMMLU sidecars for all
  four arms:
  - `2.75e-5`: convert `2525811`, native GreekMMLU `2525812`;
  - `5.5e-5`: convert `2525813`, native GreekMMLU `2525814`;
  - `8.25e-5`: convert `2526009`, native GreekMMLU `2526010`;
  - `1.1e-4`: convert `2526011`, native GreekMMLU `2526012`.
- All eight `iter=2142` sidecar jobs completed successfully.
- GreekMMLU overall at `curr-9.0B`:
  - `2.75e-5`: `8973/16632 = 0.5395`;
  - `5.5e-5`: `9175/16632 = 0.5516`;
  - `8.25e-5`: `8915/16632 = 0.5360`;
  - `1.1e-4`: `8768/16632 = 0.5272`.
- Segment-3 training remained healthy to the phase boundary, with skipped/NaN
  counters still `0`.

Peak-LR phase-boundary checkpoint and segment-4 handoff `2026-06-13T04:03+03:00`:

- All four segment-3 jobs completed successfully and their first phase-2
  segment jobs started on 16 nodes each:
  - `2.75e-5`: `2524481` completed, `2524482` running;
  - `5.5e-5`: `2524485` completed, `2524486` running;
  - `8.25e-5`: `2524490` completed, `2524491` running;
  - `1.1e-4`: `2524494` completed, `2524495` running.
- The first phase-2 segment (`s4`) loaded `iter=2261`, switched to the split
  phase-2 blend `glossapi_only_ext + 0.253164557 foreign_replay_only_ext +
  0.012658228 old_greek_replay_only_ext`, and kept the configured LR peak.
- Reset guard fired for all four arms: each s4 stderr has 64 lines of
  `train dataloader consumed_samples 2315264 -> 0 (phase-2 binary restart;
  scheduler num_steps kept)`, one per rank.
- Training after the phase switch remained finite, with observed iterations
  around `2262`-`2323`, learning rates unchanged, and skipped/NaN counters `0`.
- Home watcher submitted `iter=2261` (`curr-phase-boundary`) GreekMMLU sidecars
  for all four arms:
  - `2.75e-5`: convert `2526122`, native GreekMMLU `2526123`;
  - `5.5e-5`: convert `2526124`, native GreekMMLU `2526125`;
  - `8.25e-5`: convert `2526129`, native GreekMMLU `2526130`;
  - `1.1e-4`: convert `2526131`, native GreekMMLU `2526132`.
- All eight `iter=2261` sidecar jobs completed successfully.
- GreekMMLU overall at the phase boundary:
  - `2.75e-5`: `8962/16632 = 0.5388`;
  - `5.5e-5`: `9170/16632 = 0.5513`;
  - `8.25e-5`: `8723/16632 = 0.5245`;
  - `1.1e-4`: `9002/16632 = 0.5412`.
- Current read: after the phase switch, `5.5e-5` is leading on GreekMMLU;
  `8.25e-5` still owns the best pre-boundary score (`0.5690` at `curr-8.0B`)
  but dipped sharply at `iter=2261`.

Peak-LR tenth checkpoint `2026-06-13T05:27+03:00`:

- Home watcher submitted `iter=2380` (`curr-10.0B`) GreekMMLU sidecars for all
  four arms:
  - `2.75e-5`: convert `2526171`, native GreekMMLU `2526173`;
  - `5.5e-5`: convert `2526177`, native GreekMMLU `2526178`;
  - `8.25e-5`: convert `2526183`, native GreekMMLU `2526184`;
  - `1.1e-4`: convert `2526185`, native GreekMMLU `2526186`.
- All eight `iter=2380` sidecar jobs completed successfully.
- GreekMMLU overall at `curr-10.0B`:
  - `2.75e-5`: `9153/16632 = 0.5503`;
  - `5.5e-5`: `9499/16632 = 0.5711`;
  - `8.25e-5`: `8993/16632 = 0.5407`;
  - `1.1e-4`: `9331/16632 = 0.5610`.
- Training remained healthy after the checkpoint, with observed iterations
  around `2459`-`2489`, learning rates unchanged, and skipped/NaN counters `0`.
- Current read: `5.5e-5` now has the best GreekMMLU score of the LR sweep
  (`0.5711`), ahead of the prior `8.25e-5` pre-boundary peak (`0.5690` at
  `curr-8.0B`).

Peak-LR eleventh checkpoint `2026-06-13T06:06+03:00`:

- Home watcher submitted `iter=2618` (`curr-11.0B`) GreekMMLU sidecars for all
  four arms:
  - `2.75e-5`: convert `2526218`, native GreekMMLU `2526219`;
  - `5.5e-5`: convert `2526221`, native GreekMMLU `2526222`;
  - `8.25e-5`: convert `2526224`, native GreekMMLU `2526225`;
  - `1.1e-4`: convert `2526227`, native GreekMMLU `2526228`.
- All eight `iter=2618` sidecar jobs completed successfully.
- GreekMMLU overall at `curr-11.0B`:
  - `2.75e-5`: `9543/16632 = 0.5738`;
  - `5.5e-5`: `9695/16632 = 0.5829`;
  - `8.25e-5`: `9542/16632 = 0.5737`;
  - `1.1e-4`: `9480/16632 = 0.5700`.
- Training remained healthy after the checkpoint, with observed iterations
  around `2699`-`2729`, LR decay underway, and skipped/NaN counters `0`.
- Current read: `5.5e-5` remains the clear leader and has raised the sweep
  best to `0.5829`.
