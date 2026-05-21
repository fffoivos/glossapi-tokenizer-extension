# Session log — 2026-05-21 CSCS execution

Complete record of everything changed locally + every job run on Clariden during the 2026-05-20→21 overnight execution. Companion to [`CSCS_OVERNIGHT_STATE.md`](CSCS_OVERNIGHT_STATE.md) (which is the *current state*; this log is the *audit trail*).

User: `fffoivos`, account `a0140`, cluster Clariden. Working uenv: `pytorch/v2.9.1:v2` (transformers 4.57.0, has `ApertusForCausalLM`).

## TL;DR

| Track | State | Evidence |
|---|---|---|
| **A — Corpus build chain** | prepare_greek_pool **on attempt 6/6** (job 2334880, RUNNING healthy past previous OOM cliff). Downstream chain (normalize -> mix -> preprocess x2) staged and audited. | `filtered_input.parquet` exists; `external_drop_filtered_input.parquet` was still in-progress/0 bytes at takeover check; DuckDB temp was advancing |
| **B — V4-HF baseline** | **PARTIAL** (job 2334245, 1h11m54s; valid for listed tasks, but missing `global_mmlu`) | [`init_bakeoff/eval/v4_baseline_20260521/`](03_4_implementation_experiments/init_bakeoff/eval/v4_baseline_20260521/) — `results.json`, `V4_RESULTS.md`, full stdout |
| **C — R1 HF→Megatron→HF roundtrip** | **PASS** (job 2333864) | std-tensor max abs diff = `0.0`, R17 changed = `128` (= 32 layers × 4 xIELU params); see [`megatron_patches/README.md` § R1 result](03_4_implementation_experiments/init_bakeoff/megatron_patches/README.md#r1-result-2026-05-21-apertus-8b-2509-job-2333864) |

## Takeover addendum — 2026-05-21 ~12:15 UTC

Codex took over the live process with permission to patch, start/stop jobs, document errors, and commit local state.

Corrections made during takeover review:

- `run_eval.sbatch` omitted `global_mmlu` even though the eval docs require Table-14 retention coverage. The V4-HF job `2334245` is therefore a valid partial baseline for its listed tasks, not the final §5.6 baseline. Patched the retention task list to include `global_mmlu`; corrected V4-HF and V4-post-conversion reruns are required.
- `mix_builder.py` said token streams are identical across all arms. Corrected the wording: all arms share the same JSONL text stream, but Vanilla is base-tokenized and ReTok/Centroid are extended-tokenized, so token IDs differ across tokenizer families.
- `bulk.json` top-level metadata still said `70/26/4` even though the source weights and validation block are `70/24/4/2`; corrected the metadata to include the math bucket.
- `normalize_nfc.sh` did not include the `cpt/` directory, so an after-prepare normalization pass would have missed the final selected parquet. Added `cpt/` to the normalized roots.
- R17 documentation previously implied q/k norm drift was measured by the same `128` count. Clarified that the count only included R17 tensors with diff > 1e-3 and added future q/k max-diff printing to `r1_roundtrip.sbatch`.

Live checks performed:

- `sacct`/`squeue` on Clariden: `2334880` is RUNNING on `nid006899`; `2334476` and `2334826` are failed.
- DuckDB temp under `/iopsstor/scratch/cscs/fffoivos/tmp/prepare_greek_pool_2334880` advanced from ~105G to ~190G during review, so the external-sort spill path is active. I did not cancel the job.
- Corpus output check: `filtered_input.parquet` exists at ~124G; `external_drop_filtered_input.parquet` was still 0 bytes at the check, so the Apertus-drop stage was not yet validated complete.
- GCP active-instance check from `home` failed because `gcloud` needs non-interactive reauthentication; no GCP instance state was verified.
- Synced corrected scripts/docs to Clariden. A parallel `rsync` of `glossapi_corpus_cli/` failed once with SSH connection close, then succeeded on a single retry; checksums matched for the key deployed files.
- Submitted corrected V4 eval jobs:
  - `2335100` V4-HF corrected baseline -> `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_baseline_v4_corrected_20260521_121639`
  - `2335101` V4-post-conversion corrected baseline -> `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_postconv_v4_corrected_20260521_121639`
- `2335101` failed with `OSError: [Errno 37] No locks available` from `datasets`/`filelock`. `run_eval.sbatch` now assigns each job its own HF/datasets cache under `/iopsstor/scratch/cscs/fffoivos/tmp/eval_cache_$SLURM_JOB_ID`; post-conversion eval was resubmitted as `2335196`.
- Queued after-prepare corpus dependency chain: `2335157` normalize_nfc -> `2335158` mix_builder_smoke -> `2335159` mix_builder_full -> `2335160` base preprocess + `2335161` extended preprocess.
- Added queueable init chain: `arms/build_init_checkpoints.sbatch`, `arms/convert_init_checkpoints.sbatch`, and `arms/submit_init_pipeline.sh`. First attempt `2335353` caught a Slurm spool-path bug; second attempt `2335371` caught the old-Transformers Apertus loader mismatch in `pytorch/v2.6.0:v1`. Final init chain `2335382` build -> `2335384` conversion completed successfully using `INIT_UENV_IMAGE=pytorch/v2.9.1:v2`.
- Patched the same Slurm spool-path bug in `preprocess_data.sbatch` and `bakeoff_train.sbatch`, then canceled old pending preprocess jobs `2335160` / `2335161` and requeued patched replacements `2335581` / `2335583` with dependency `afterok:2335159`.
- `2334880` completed successfully. Selected CPT parquet: `47,061,862` rows and `227,837,744,625` chars at `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet`.
- `2335157` normalize_nfc failed because `normalize_nfc.sh` expected unsupported directory-mode flags in `verify_and_normalize_nfc.py`. Patched the wrapper to enumerate parquets and normalize file-by-file via `--out <tmp>`, then requeued the corpus chain as `2335826` -> `2335827` -> `2335828` -> `2335829`/`2335830`.

## Files touched locally (home machine)

Code:

| Path | Change |
|---|---|
| `glossapi_corpus_cli/pipeline.py` | Added `_duckdb_connect_streaming()` helper at module top; routed all 14 `duckdb.connect()` sites through it. Sets `SET preserve_insertion_order = false` so duckdb spills large ORDER BY sorts to `temp_directory` instead of materializing in RAM. Root-cause fix for the prepare_greek_pool OOM cliff. |
| `subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/megatron_patches/loader_apertus_hf.py` | (Already on disk before session.) Empirically required additions: `--bf16` / `--fp16` registered in `add_arguments`; `ApertusForCausalLM` import wrapped with `AutoModelForCausalLM + trust_remote_code` fallback. |
| `subprojects/.../init_bakeoff/megatron_patches/r1_roundtrip.sbatch` | **New file.** Two-leg HF→Megatron→HF round-trip on Apertus-8B-2509, with the between-legs `iter_0000000` → `release` rename + `latest_checkpointed_iteration.txt = "release"` step. Compute justification block. |
| `subprojects/.../init_bakeoff/corpus_build/prepare_greek_pool.sh` | (Already on disk before session.) |
| `subprojects/.../init_bakeoff/corpus_build/prepare_greek_pool.sbatch` | **New file.** Wraps prepare_greek_pool.sh on normal partition (xfer in maintenance). CD to iopsstor tmp before invoking (duckdb writes `.tmp/duckdb_temp_storage_*` relative to CWD). `--cpus-per-task=288 --mem=800G --gpus-per-node=0`. Compute justification block. |
| `subprojects/.../init_bakeoff/corpus_build/normalize_nfc.sh` | `--workers` made env-overridable (default 64, recommended 288 on a full node). Compute justification block. |
| `subprojects/.../init_bakeoff/corpus_build/normalize_nfc.sbatch` | **New file.** Wraps normalize_nfc.sh with `WORKERS=288`. Compute justification block. |
| `subprojects/.../init_bakeoff/corpus_build/mix_builder.py` | Docstring updated — explicit note that the row-at-a-time `tokenizer.encode` loop is intentional for bakeoff determinism (not a missed batching opportunity); names the bottleneck library + thread model. |
| `subprojects/.../init_bakeoff/corpus_build/mix_builder_smoke.sbatch` | **New file.** PF3 smoke (50 M token target) on normal partition, single socket (72 CPUs), 200 G mem. Compute justification block. |
| `subprojects/.../init_bakeoff/eval/run_eval.sbatch` | Rewritten — Greek task names corrected (drop the Meltemi-only `arc_greek` / `hellaswag_greek` etc., use the swissai-native `arc_challenge_mt_el` / `belebele_ell_Grek` / `global_mmlu_full_el` / `include_base_44_greek_few_shot_en` / `xnli_el` / `xquad_el` / `global_piqa_completions_ell_grek`). Install path switched from `pip install -e .` (uenv site is read-only) to `PYTHONPATH=/iopsstor/.../python_envs/lm_eval`. `ls | head` SIGPIPE bug fixed. Compute justification block. |

Reviewer-facing docs updated:

| Path | Change |
|---|---|
| `subprojects/.../init_bakeoff/megatron_patches/README.md` | Added § "Empirically-required CLI knobs (added 2026-05-21 during R1)" + § "R1 result (2026-05-21, Apertus-8B-2509, job 2333864)" with the actual pass numbers. Procedure block now includes the release-mark step and `--loader-transformer-impl transformer_engine` on both legs. `r1_roundtrip.sbatch` added to the "files in this dir" list. |
| `subprojects/.../init_bakeoff/eval/EVAL_RECIPE.md` | Group 2 Greek table rewritten to the swissai-native task names (with a note explaining the gap between the previous version's ILSP-style names and what actually resolves). lm-eval install recipe documented under § Resources. |
| `subprojects/.../init_bakeoff/eval/v4_baseline_20260521/V4_RESULTS.md` | **New file.** Per-task V4 numbers, the SIGPIPE post-mortem, and what unblocks. |
| `subprojects/.../init_bakeoff/eval/v4_baseline_20260521/{results.json, run_metadata.json, stdout.log}` | Canonical V4 artifacts copied off Clariden for permanent record. |
| `subprojects/03_apertus_extension_*/RISKS.md` § R17 | Added empirical quantification of the R17 reset (128 keys = 32 layers × 4 xIELU params); std tensors at 0.0 max abs diff. Status remains "open" but now backed by measurement. |
| `subprojects/03_apertus_extension_*/CSCS_OVERNIGHT_STATE.md` | Prepended a "Morning update (2026-05-21 ~12:30 UTC)" table covering pulls done, R1 PASS, V4 results, prepare_greek_pool retry chain, lm_eval install recipe, glossapi_rs_noise aarch64 stub, working uenv, and a "Sbatch compute-saturation audit" table. |
| `subprojects/03_apertus_extension_*/SESSION_LOG_20260521.md` | **This file.** |

Local memories saved (auto-memory `/home/foivos/.claude/projects/-home-foivos/memory/`):

- `clariden_xfer_maintenance.md` — xfer drained till 2026-06-11; route CPU jobs to `normal` with explicit `--cpus-per-task` / `--mem` / `--gpus-per-node=0`.
- `feedback_complete_docs_as_you_go.md` — reviewer-facing docs must be updated in the same turn as the work lands.
- `feedback_compute_sweet_spot_justify.md` — every sbatch needs a 4-question Compute justification block.

## Clariden state changes

Files staged / installed at session start (still live):

| Path | Source | Notes |
|---|---|---|
| `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/` | HF `swiss-ai/Apertus-8B-2509` | ~16 GB |
| `/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/` | ship bundle | extended tokenizer |
| `/iopsstor/scratch/cscs/fffoivos/code/training/Megatron-LM-Swiss-AI/` | swiss-ai/Megatron-LM commit `c92402e3` | + `loader_apertus_hf.py` symlinked in via `install.sh` |
| `/iopsstor/scratch/cscs/fffoivos/code/training/pretrain-code/` | swiss-ai/pretrain-code commit `531cc8be` | reference only |
| `/iopsstor/scratch/cscs/fffoivos/code/eval/lm-evaluation-harness-swissai/` | swiss-ai/lm-evaluation-harness | source for our target install |
| `/iopsstor/scratch/cscs/fffoivos/code/eval/lm-evaluation-harness-eleuther/` | EleutherAI/lm-evaluation-harness | upstream fallback |
| `/iopsstor/scratch/cscs/fffoivos/repo/` | rsync of our subproject + `glossapi_corpus_cli` + `rust_reevaluate_pdf_datasets.py` (scp'd separately when CLI couldn't find it) | |
| `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/nanochat/` | `fffoivos/glossapi-greek-nanochat-pretraining-dataset` (HF, gated; token scp'd in) | 282 files, 129 G |
| `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/replay/` | FineWeb-Edu + FineWeb2-HQ + FineWeb-2 (one shard per lang) | 58 G across 24 langs |
| `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/apertus_overlap_overlay/` | `fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z` | 116 M |
| `/iopsstor/scratch/cscs/fffoivos/benchmarks/{retention, ilsp_greek, safety, other_greek}/` | HF (per `pull_benchmarks.sh`) | retention 2.1 G, safety 1.9 G, ilsp_greek 137 M, other_greek 8.4 M |
| `/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval/` | `pip install --target ... .` from the swissai harness, then `pip install --target --no-deps accelerate`, then `pip install --target --no-deps typer duckdb blake3 zstandard polars`. **Crucial:** `huggingface_hub/` and `huggingface_hub-*.dist-info` deleted from this target so uenv's compatible 0.36.0 wins over the pip-installed 1.x. | The runtime PYTHONPATH for all CSCS jobs |
| `/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval/glossapi_rs_noise/__init__.py` | tiny pure-Python stub (the real wheel is x86_64-only; Clariden is aarch64) | satisfies `import glossapi_rs_noise` in pipeline.py; raises if `score_markdown_directory_detailed` is actually called (not on our code path) |
| `/iopsstor/scratch/cscs/fffoivos/repo/rust_reevaluate_pdf_datasets.py` | scp'd from home — `pipeline.py:65` loads it dynamically | |

Outputs produced:

| Path | Source | State |
|---|---|---|
| `/capstor/scratch/cscs/fffoivos/runs/r1_roundtrip_2333864/apertus_megatron/release/` | R1 leg 1 | Megatron `release` checkpoint |
| `/capstor/scratch/cscs/fffoivos/runs/r1_roundtrip_2333864/apertus_hf_roundtrip/` | R1 leg 2 | HF-format roundtripped Apertus (R17-reset xIELU + QK-Norm) — usable for V4-post-conversion baseline |
| `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_baseline_v4_20260521_100521/` | V4 baseline | `results.json` (370 KB) + `samples_<task>.jsonl` per task + `run_metadata.json` |
| `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/glossapi_mix_prelude_*/{filtered_input.parquet, external_drop_filtered_input.parquet}` | prepare_greek_pool (in-flight) | intermediate stages 1 + 2 done; stage 3 (dedup-replay) writing now |

## Slurm jobs (chronological)

R1 path:

| Job | What | State | Walltime | Notes |
|---|---|---|---|---|
| 2333676 | R1 attempt — `saver_core` `validate_args` died on `qknorm_impl=apex` | FAILED | 11:53 | required `--loader-transformer-impl transformer_engine` |
| 2333762 | R1 retry — leg 1 wrote `iter_0000000/`, leg 2 asserts `iteration > 0 OR "release"` | FAILED | ~5 min | needed `mv iter_0000000 release` + `echo release > latest_checkpointed_iteration.txt` between legs |
| **2333864** | **R1 final** — both fixes applied | **PASS** | ~6 min | `standard max abs diff = 0.0`, R17 changed = 128 |

V4 path:

| Job | What | State | Walltime | Notes |
|---|---|---|---|---|
| 2333668 | V4 baseline first attempt | FAILED | ~5 s | pip install -e . to read-only uenv |
| 2333723 | V4 retry — target-installed lm_eval | FAILED | ~30 s | tasks not found (`arc_greek` etc. — not in swissai harness) |
| 2333949 | V4 retry — task names corrected | FAILED | ~5 s | `huggingface-hub==1.15.0 != <1.0` |
| 2334148 | V4 retry — hf_hub stripped from target | FAILED | ~5 s | `ModuleNotFoundError: accelerate` |
| **2334245** | **V4 baseline final** — accelerate installed | **PASS** (eval succeeded; sacct showed FAILED 13:0 from a SIGPIPE on `ls\|head`) | 1h11m54s | see `V4_RESULTS.md` |

prepare_greek_pool path:

| Job | What | State | Walltime | Notes |
|---|---|---|---|---|
| 2334246 | First attempt on xfer | CANCELLED | 0:00 | xfer in maintenance till 2026-06-11 |
| 2334267 | Retry on normal | FAILED | ~30 s | duckdb temp on home filesystem hit disk quota |
| 2334358 | Retry, cd-to-iopsstor for tmp | FAILED | ~10 min | duckdb OOM at 305 GiB in `materialize_doc_key_excluded_mix_input` |
| 2334476 | Retry, `--mem` 400 → 800G | FAILED | ~17 min | OOM at 610 GiB — same query, more data accumulated; not a mem-sizing issue, root cause is `preserve_insertion_order=true` materializing the full ORDER BY |
| 2334826 | Retry with `_duckdb_connect_streaming` helper (incorrectly recursive) | FAILED | ~5 s | helper called itself — my own `replace_all` rewrote the helper's `duckdb.connect()` line |
| **2334880** | **Retry with corrected helper** | **RUNNING** | 14+ min as of writing | past the OOM cliff (stages 1 + 2 written); stage 3 dedup-replay in progress |

## Errors + fixes (consolidated)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `convert.py: unrecognized arguments: --bf16` | convert.py top-level parser doesn't add `--bf16`; each loader registers its own dtype flags | added `--bf16` / `--fp16` to `loader_apertus_hf.add_arguments` |
| 2 | `ImportError: cannot import name 'ApertusForCausalLM' from transformers` | uenv `pytorch/v2.6.0:v1` has transformers 4.48.3 (no Apertus) | switched to `pytorch/v2.9.1:v2` (transformers 4.57.0) + added `AutoModelForCausalLM + trust_remote_code` fallback in loader |
| 3 | `ValueError: model_type apertus not recognized` (with trust_remote_code) | trust_remote_code doesn't backport unknown model types to old transformers | same as #2 — needed the newer uenv |
| 4 | `saver_core` `AssertionError: OP arguments are only checked with the TE transformer implementation` | Apertus checkpoint has `qknorm_impl=apex` which triggers OP-args branch; saver checked `transformer_impl == "transformer_engine"` first | pass `--loader-transformer-impl transformer_engine` on both convert.py legs |
| 5 | `loader_core` `assert iteration > 0 OR file=='release'` on the Megatron checkpoint | saver_core wrote `iter_0000000/` with iteration string `'0'`; loader_core rejects that | between legs: `mv iter_0000000 release` + `echo release > latest_checkpointed_iteration.txt` |
| 6 | `saver_swissai_hf: unrecognized argument: --bf16` | only the loader registers `--bf16`; saver_swissai_hf doesn't | drop `--bf16` from the leg-2 convert.py call |
| 7 | `pip install -e .` `OSError: Read-only file system` to uenv site-packages | uenv default view is RO | install to writable `--target=/iopsstor/.../python_envs/lm_eval` and set `PYTHONPATH` |
| 8 | `lm_eval: Tasks not found: arc_greek, hellaswag_greek, ...` | those names only exist in `LeonVouk/lighteval` / `ilsp/lm-evaluation-harness-greek`, not in `swiss-ai/lm-evaluation-harness` | rewrote task list to swissai-native names (`arc_challenge_mt_el`, `belebele_ell_Grek`, `global_mmlu_full_el`, `include_base_44_greek_few_shot_en`, `xnli_el`, `xquad_el`, `global_piqa_completions_ell_grek`); PF5 (port the ILSP YAMLs) tracked but not done |
| 9 | `transformers ImportError: huggingface-hub>=0.34.0,<1.0 ... but found 1.15.0` | swissai lm-eval install pulled hf_hub 1.x into target, which shadows uenv's compatible 0.36.0 | reinstall + `rm -rf TARGET/huggingface_hub TARGET/huggingface_hub-*.dist-info` so uenv's wins |
| 10 | `ModuleNotFoundError: accelerate` | I used `--no-deps` on the lm-eval install to dodge the hf_hub conflict, which also dropped accelerate | `pip install --target ... --no-deps accelerate` |
| 11 | V4 `sacct State=FAILED ExitCode=13` after eval visibly succeeded | `ls $OUTPUT_DIR | head -20` got SIGPIPE (exit 141 → slurm reports 13) and `set -o pipefail` propagated the fail | `ls -la "$OUTPUT_DIR" || true` instead of `\|head`; eval was already complete |
| 12 | `glossapi_corpus_cli: ModuleNotFoundError: glossapi_rs_noise` | the real wheel is x86_64 only; Clariden is aarch64 | wrote a small pure-Python stub that satisfies the import and raises if `score_markdown_directory_detailed` is called (it isn't on the `mix-prepare-selected-input` path) |
| 13 | `FileNotFoundError: rust_reevaluate_pdf_datasets.py` from `pipeline.load_reeval_module()` | not in the rsync'd subproject — sits at the repo root in our home tree | scp'd directly to `/iopsstor/.../repo/rust_reevaluate_pdf_datasets.py` |
| 14 | xfer-partition jobs hang in `PD (ReqNodeNotAvail, Reserved for maintenance)` | xfer in reservation `SD-69241-apertus-1-5-0` from 2026-05-11 → 2026-06-11 | switched all CPU-only jobs to `normal` with explicit `--cpus-per-task`, `--mem`, no `--gpus-per-node` |
| 15 | `IOException: Could not write file ".tmp/duckdb_temp_storage_*.tmp": Disk quota exceeded` | duckdb writes `.tmp/` relative to CWD; sbatch CWD defaulted to home filesystem (tiny quota) | cd to `/iopsstor/scratch/cscs/fffoivos/tmp/<JOBID>` before invoking; also set `TMPDIR` |
| 16 | `OutOfMemoryException: failed to pin block ... (305.1 GiB / 305.1 GiB used)` in `materialize_doc_key_excluded_mix_input` | duckdb's auto memory_limit ≈ 80 % of `--mem=400G`; the COPY+ORDER BY materializes the entire result | first tried `--mem=800G` (wrong fix — same query, larger ceiling) |
| 17 | `OutOfMemoryException: failed to allocate data of size 128.0 MiB (610.3 GiB / 610.3 GiB used)` in the same COPY | confirmed root cause: `preserve_insertion_order=true` (duckdb default) materializes the ORDER BY in RAM even though the query has an explicit ORDER BY | added `_duckdb_connect_streaming()` helper in pipeline.py that does `SET preserve_insertion_order = false` on every connection; replaced all 14 `duckdb.connect()` call sites |
| 18 | `RecursionError: maximum recursion depth exceeded` in `_duckdb_connect_streaming` | my `replace_all` for "con = duckdb.connect()" also rewrote the inside of the helper, so the helper recursed into itself | helper now calls `duckdb.connect()` directly (with a comment noting the recursion trap) |

Pre-CSCS-execution friction (already-debugged before the user went to sleep, listed here for completeness):

- HF `huggingface-cli download` argparse: multiple `--include` flags only honor the last (`nargs="*"`). Fix: combine all patterns into a single `--include` with space-separated args. Affected `pull_greek_corpus.sh`, `pull_replay_datasets.sh`, `pull_benchmarks.sh`.
- FineWeb2-HQ `fra_Latn` would have pulled 4.8 TB (436 shards × 11 GB). Fix: rewrote `pull_replay_datasets.sh` as single-shard-per-lang.
- `hf_transfer` not available in uenv. Fix: removed `HF_HUB_ENABLE_HF_TRANSFER=1` from pull scripts.
- nanochat HTTP 401 (gated). Fix: scp'd `~/.cache/huggingface/token` from home to Clariden.
- An earlier rsync arg accidentally created a local `setup.py` directory (mid-arg `\` escape consumed by the following token). Fix: cleaned args.

## Pending / live queue (takeover continuation)

Live jobs queued/running as of the takeover continuation:

- `2334880` prepare_greek_pool completed and produced the selected CPT parquet.
- `2335100` V4-HF corrected baseline completed successfully (`COMPLETED`, exit `0:0`, elapsed `01:10:29`). Small result copy: `03_4_implementation_experiments/init_bakeoff/eval/v4_baseline_corrected_20260521/`.
- `2335196` V4-post-conversion retry is running with per-job dataset cache.
- `2335826` -> `2335827` -> `2335828` -> `2335829`/`2335830` is the active corpus dependency chain.
- `2335382` -> `2335384` completed and produced Megatron release checkpoints for all three arms.

Next:

1. Watch `2335826` normalize_nfc.
2. Watch `2335827` mix_builder_smoke and inspect the smoke JSONL.
3. Watch `2335828` full mix -> `bulk_mix.jsonl`.
4. Watch `2335829` / `2335830` preprocess outputs: `$BASE_DATA_PREFIX{.bin,.idx}` and `$EXT_DATA_PREFIX{.bin,.idx}`.
5. Submit `submit_all_arms.sh` with `INIT_CKPT_ROOT=/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480` to fire the three 2 B-token training runs.

Independent follow-ups (deferred to after the corpus chain is unblocked):

- **PF5** — port the ILSP `*_greek` task YAMLs from `LeonVouk/lighteval` into the swissai harness clone so V4 / per-arm evals include `hellaswag_greek`, `winogrande_greek`, `mmlu_pro_greek`, `truthfulqa_greek`, `medical_mcqa_greek`. Today the V4 baseline covers seven Greek tasks; ILSP would add five more.
- **V4-post-conversion baseline** — running as retry job `2335196` with `MODEL_PATH=/capstor/scratch/cscs/fffoivos/runs/r1_roundtrip_2333864/apertus_hf_roundtrip`. Needed for §5.6 thresholds that compare against R17-reset weights.
- **§5.6 hard gates** — compute bootstrap CIs (`compute_bootstrap_cis.py`) over V4's per-sample jsonls; fill the `PENDING(V4)` cells in EVAL_RECIPE.md.
