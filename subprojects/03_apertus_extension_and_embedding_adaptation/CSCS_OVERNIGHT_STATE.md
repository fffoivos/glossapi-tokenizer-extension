# CSCS overnight state (handoff)

*Started 2026-05-21 ~03:30 EEST. Foivos going to sleep around the start of this. Cluster: Clariden, account a0140, user fffoivos.*

## Takeover update (2026-05-21 ~16:10 UTC)

Codex takeover is now the live handoff. `SESSION_LOG_20260521.md` remains the audit trail; this file is the current operating state.

| Item | Status |
|---|---|
| Active Clariden jobs | Corrected post-conversion eval `2335196` timed out at the old 4h walltime (`926/1190` generate requests). It was resubmitted as split 8h jobs: retention `2338020` and Greek `2338021`. The superseded no-code chain (`2336647` -> `2336680` -> `2336681`/`2336682`) was cancelled before concat/preprocess. The first code-included chain (`2337839` -> `2337846` -> `2337847`/`2337848`) hit HF rate limits and was cancelled before concat/preprocess. Active corrected local-replay chain is `2337911` (`0-6%7`, 1B tokens/shard), followed by concat `2337912` and preprocess jobs `2337913`/`2337914`. Corrected V4-HF eval `2335100`, normalize `2335826`, token-fair smoke `2336566`, and init build/conversion jobs `2335382`/`2335384` have COMPLETED. |
| Corpus output status | `prepare_greek_pool` job `2334880` COMPLETED `0:0` in 1h13m42s. Final selected CPT parquet exists at `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet`: `47,061,862` rows, `227,837,744,625` chars. After NFC normalization by `2335826`, the atomically replaced parquet is `129,318,720,576` bytes on disk. |
| V4-HF baseline | Job `2334245` produced valid artifacts for its task list, but the script omitted `global_mmlu`. Treat `init_bakeoff/eval/v4_baseline_20260521/` as a partial baseline until corrected V4-HF and V4-post-conversion reruns complete. |
| R1 roundtrip | PASS: job `2333864`, standard tensors max abs diff `0.0`, R17 xIELU deltas `128`. The sbatch now prints separate R17/qk max-diff values on future reruns. |
| Corrected V4 evals | `2335100` V4-HF corrected baseline completed `0:0` in `01:10:29`; small local copy is under `03_4_implementation_experiments/init_bakeoff/eval/v4_baseline_corrected_20260521/`. `2335101` V4-post-conversion failed on a datasets/filelock `No locks available` error; patched per-job HF/datasets cache and resubmitted as `2335196`. `2335196` then timed out at 4h while healthy, so `run_eval.sbatch` now defaults to 8h and the post-conversion eval is split into `2338020` retention-only and `2338021` Greek-only. |
| Queued corpus chain | First normalize job `2335157` failed because the wrapper called unsupported directory-mode CLI flags on `verify_and_normalize_nfc.py`. Patched `normalize_nfc.sh`; `2335826` then completed. Mix smoke retries exposed generator/HF interleave, eager setup, stale source labels, gated code-source issues, and finally row-weighted token-share drift. Token-fair smoke `2336566` passed (`50,000,643` tokens, recipe shares close to target). User speed challenge led to raising the old array throttle from `3` to `7`, then the direction check found the no-code workaround had drifted from the intended mix. First code-included array `2337839` failed from HF's `1000 api requests per 5 minutes` limit, so the recipe now reads replay/math from staged local parquets. Corrected active chain: `2337911` shard array -> `2337912` concat -> `2337913` base preprocess + `2337914` extended preprocess. |
| Active mix recipe deviation | Steering corrected. The live recipe is back to `70%` Greek, `24%` replay, `4%` code, `2%` math. Replay/math use staged local single-shard parquets to avoid HF metadata limits. Code uses `codeparrot/codeparrot-clean-train` as an explicit 2026-05-21 fallback because BigCode StarCoder/The Stack sources were gated or script-backed under the current auth/runtime. This preserves code-retention pressure but is not an exact StarCoder-source match. |
| Init checkpoints | Added `arms/build_init_checkpoints.sbatch`, `arms/convert_init_checkpoints.sbatch`, and `arms/submit_init_pipeline.sh`. `2335353` failed from Slurm spool-path handling; fixed with `SLURM_SUBMIT_DIR`. `2335371` failed from old Transformers in `pytorch/v2.6.0:v1`; fixed init jobs to use `pytorch/v2.9.1:v2`. Final chain passed: `2335382` build (2m43s) and `2335384` conversion (1m41s). Megatron releases exist under `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/{vanilla,retok,centroid}/megatron/release`. |
| GCP cost check | Attempted from `home`; blocked by non-interactive `gcloud` reauthentication failure. No GCP instance state was verified in this takeover turn. |
| Next actions | Monitor `2337911` shard progress and split post-conversion evals `2338020`/`2338021`. When concat and preprocess pass, submit the three 2B arms with `INIT_CKPT_ROOT=/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480 bash submit_all_arms.sh`. |

---

## Morning update (2026-05-21 ~12:30 UTC, superseded where it disagrees with takeover update)

What changed since the original handoff below:

| Item | Status |
|---|---|
| HF auth on Clariden | scp'd `~/.cache/huggingface/token` from home; no manual login needed |
| Greek nanochat pull | DONE — 282/282 files, 129 G, completed via `nohup` on the login node |
| Replay pull | DONE — 58 G across 24 single-shard langs |
| R1 HF→Megatron→HF roundtrip | **PASS** (job 2333864). `standard max abs diff = 0.0`, R17 changed = 128 (= 32 layers × 4 xIELU params). See [`03_4_implementation_experiments/init_bakeoff/megatron_patches/README.md`](03_4_implementation_experiments/init_bakeoff/megatron_patches/README.md#r1-result-2026-05-21-apertus-8b-2509-job-2333864) for the full result. |
| V4-HF baseline | **PARTIAL / artifact valid for listed tasks** (job 2334245, 1h11m54s, 1× H100, bf16). Result table + canonical artifacts saved at [`init_bakeoff/eval/v4_baseline_20260521/`](03_4_implementation_experiments/init_bakeoff/eval/v4_baseline_20260521/) — `results.json` (370 KB), `run_metadata.json`, full `stdout.log`, plus [`V4_RESULTS.md`](03_4_implementation_experiments/init_bakeoff/eval/v4_baseline_20260521/V4_RESULTS.md) summary. It omitted `global_mmlu`; rerun before final §5.6 thresholds. NB: `sacct` showed `FAILED 13:0` — that's a SIGPIPE from `ls \| head` after eval finished; patched out of run_eval.sbatch. |
| prepare_greek_pool.sh | **RUNNING** (job 2334880 after 5 failed/cancelled attempts). Issues found and patched: (1) duckdb temp on home filesystem hit disk-quota — fix: cd to `/iopsstor/scratch/cscs/fffoivos/tmp/<JOBID>`; (2) xfer partition in maintenance till 2026-06-11 (reservation `SD-69241-apertus-1-5-0`) — fix: run on `normal` with `--cpus-per-task=… --mem=…G --gpus-per-node=0`; (3) job 2334358 OOMed at 305 GiB and job 2334476 OOMed at 610 GiB inside `materialize_doc_key_excluded_mix_input`'s COPY+ORDER BY (pipeline.py:2486); (4) job 2334826 failed from a recursive helper bug. Current root-cause fix: route every `duckdb.connect()` through `_duckdb_connect_streaming()` and set `SET preserve_insertion_order = false` so the sort spills to iopsstor-backed DuckDB temp. |
| lm-evaluation-harness install | DONE — swissai clone installed via `pip install --target=/iopsstor/.../python_envs/lm_eval`; `huggingface_hub` deleted from target so uenv's compatible 0.36.0 wins. Recipe in EVAL_RECIPE.md. |
| `glossapi_corpus_cli` runtime deps | DONE on Clariden — `typer`, `duckdb`, `blake3`, `zstandard`, `polars`, `accelerate` installed via `--target --no-deps`. `glossapi_rs_noise` stubbed (aarch64; pure-Python stub raises if `score_markdown_directory_detailed` is called — not on the `mix-prepare-selected-input` code path). |
| `loader_apertus_hf.py` patches landed | `--bf16` / `--fp16` registered on the loader; `ApertusForCausalLM` import wrapped with `AutoModelForCausalLM + trust_remote_code=True` fallback; both reflected in megatron_patches/README.md. |
| sbatch fixes landed | `r1_roundtrip.sbatch` (debug, 1h), `run_eval.sbatch` (normal, 4h, target-installed lm_eval), `prepare_greek_pool.sbatch` (normal, 12h, CPU-only with iopsstor tmp). |
| `pytorch/v2.9.1:v2` is the working uenv | NOT v2.6.0:v1 — that image's transformers 4.48.3 lacks `ApertusForCausalLM`. All conversion / eval jobs use v2.9.1:v2. |

**Next blockers (in order):** normalize_nfc -> mix_builder -> preprocess x2. The selected CPT parquet and three Megatron init checkpoints already exist. Once both preprocess outputs exist, run `submit_all_arms.sh`. In parallel, corrected V4-HF and V4-post-conversion evals are running and are needed before final §5.6 thresholds.

### Sbatch compute-saturation audit (2026-05-21)

All sbatches under `03_4_implementation_experiments/init_bakeoff/` now carry an explicit `# Compute justification:` block (per memory [[feedback_compute_sweet_spot_justify]]) covering parallelism strategy / CPU+GPU saturation / memory ceiling vs observed peak / known efficiency gaps. Summary:

| Script | CPUs | Mem | GPU | Bottleneck |
|---|---|---|---|---|
| `corpus_build/prepare_greek_pool.sbatch` | **288** (was 64) | 800G | — | duckdb auto-MT; 400G OOM'd at 305 GiB |
| `corpus_build/normalize_nfc.sbatch` (new) | 288 workers | 400G | — | one Python proc per parquet shard, ~300 shards |
| `corpus_build/mix_builder_smoke.sbatch` (new) | 72 | 200G | — | streaming writer (determinism), pyarrow MT, serial tok.encode |
| `eval/run_eval.sbatch` | 72 | 200G | 1 | single-H100 8B fits comfortably; multi-GPU would idle |
| `megatron_patches/r1_roundtrip.sbatch` | 72 | 400G | 1 | convert.py per-leg streaming |

The currently running `prepare_greek_pool` (job 2334880) uses the full normal-partition node request in the local sbatch. The mix_builder.py docstring notes explicitly that the row-at-a-time tokenizer loop is intentional for bakeoff determinism (not a missed batching opportunity).

---


## What's done

| Item | State |
|---|---|
| CSCS cert | Valid until 2026-05-22 04:22 EEST (24 h) |
| Clariden dir layout | Created at `/iopsstor/scratch/cscs/fffoivos/{code,models,tokenizers,cpt_corpus,benchmarks,init_checkpoints,repo}` + `/capstor/.../runs/{eval,bakeoff,preprocess}` + `~/logs/` |
| `swiss-ai/Megatron-LM` cloned | At `c92402e3` pinned commit |
| `swiss-ai/pretrain-code` cloned | At `531cc8be` pinned commit |
| `swiss-ai/lm-evaluation-harness` cloned | HEAD (also has EleutherAI fallback) |
| Our `loader_apertus_hf.py` symlinked | Into `Megatron-LM/tools/checkpoint/` |
| Our repo rsync'd | `/iopsstor/.../repo/03_apertus_extension_and_embedding_adaptation/` + `glossapi_corpus_cli/` |
| `swiss-ai/Apertus-8B-2509` downloaded | ~16 GB at `/iopsstor/.../models/apertus-8b-2509/` |
| Extended ship bundle copied | `apertus_greek_modern_only_148480/` at `/iopsstor/.../tokenizers/` |
| Benchmarks pulled | retention (2.1 GB), ilsp_greek (137 MB), safety (1.9 GB), other_greek (8.4 MB) under `/iopsstor/.../benchmarks/` |
| Replay pull (single-shard-per-lang) | **Running in background** — should finish in ~10-30 min. ~30-50 GB total ETA. |

## What's blocked on you

**Nanochat dataset is HF-gated.** `fffoivos/glossapi-greek-nanochat-pretraining-dataset` returns HTTP 401 for the `data/*.parquet` files. You need to authenticate on Clariden:

```
ssh clariden
uenv start pytorch/v2.6.0:v1 --view=default
huggingface-cli login   # paste your HF token
exit
```

(Or set `HF_TOKEN` in `~/.bashrc` so it's permanent.)

Once authenticated:

```
ssh clariden 'nohup ~/run_in_pytorch_uenv.sh \
    /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/pull_greek_corpus.sh \
    > ~/logs/pull_greek_corpus.log 2>&1 &'
```

This pulls: nanochat parquets (Greek corpus, several GB), wave2 dedup metadata, Apertus-overlap drop overlay. ~30-60 min wall.

## Bugs found + fixed tonight

The original pull scripts had a `huggingface-cli` argparse bug: **multiple `--include` flags override each other** (nargs="*"). Only the last `--include` was honored, so the pulls fetched only README.md per dataset. **Fix**: combine all patterns into ONE `--include` arg with space-separated patterns. Patches committed to:

- `pull_greek_corpus.sh` (single-include + HF_TOKEN check + auth error message)
- `pull_replay_datasets.sh` (rewritten: **single shard per lang** — was going to download 4.8 TB just for French; now ~50 MB-3 GB per lang, ~30-50 GB total)
- `pull_benchmarks.sh` (minor)

Also: `hf_transfer` not available in the pytorch uenv → removed `HF_HUB_ENABLE_HF_TRANSFER=1` from the scripts (slower but works).

## Tomorrow morning (when you wake up)

Order of operations:

```
# 1. Authenticate HF (one time)
ssh clariden
uenv start pytorch/v2.6.0:v1 --view=default
huggingface-cli login
exit

# 2. Pull Greek nanochat (~30-60 min in background)
ssh clariden 'nohup ~/run_in_pytorch_uenv.sh \
    /iopsstor/.../init_bakeoff/corpus_build/pull_greek_corpus.sh \
    > ~/logs/pull_greek_corpus.log 2>&1 &'

# 3. Verify replay finished + benchmarks present
ssh clariden 'du -sh /iopsstor/.../cpt_corpus/replay/* /iopsstor/.../benchmarks/*'

# 4. Run prepare_greek_pool.sh (after nanochat pull done) — UNTESTED CODE PATH
ssh clariden 'bash ~/run_in_pytorch_uenv.sh \
    /iopsstor/.../corpus_build/prepare_greek_pool.sh'
# Watch for: glossapi_corpus_cli import errors, missing dedup metadata, etc.
# If errors: I documented them; we debug together.

# 5. normalize_nfc.sh, then mix_builder.py 7B tokens, then preprocess ×2 (per the chain in init_bakeoff/README.md)

# 6. Submit V4-HF baseline:
ssh clariden 'cd /iopsstor/.../init_bakeoff/eval && bash run_apertus_baseline.sh'

# 7. R1 roundtrip on Apertus-8B-2509 (megatron_patches/README.md §"Roundtrip validation procedure")

# 8. build_init_checkpoints.py + convert × 3 → submit_all_arms.sh
```

## Items I did NOT attempt tonight

- **V4-HF baseline submission**: needs lm-eval-harness to be `pip install -e .`'d inside the pytorch uenv at runtime; haven't verified it works end-to-end. Defer until morning, with eyes on first sbatch.
- **R1 HF→Megatron roundtrip**: needs Megatron Python env set up (transformer_engine, apex, etc.). Significant setup that's better done with eyes-on.
- **`prepare_greek_pool.sh` smoke test**: needs nanochat pull done (HF-auth-gated, so morning).

## Logs (read these in the morning)

```
ssh clariden 'ls -la ~/logs/'
# ~/logs/apertus_download.log       — should say APERTUS_DOWNLOAD_DONE
# ~/logs/pull_replay_datasets.log   — final state should show all langs
# ~/logs/pull_benchmarks.log        — already finished
```

## Storage state

- `/iopsstor/scratch/cscs` has ~300 TB free (1.5 PB total) — no concerns.
- `~/` is small but we don't use it for data.
- `cpt_corpus/replay/` is the biggest current consumer (~30-50 GB ETA).

## Risk reminders

- **R17** (xIELU/QK-Norm reset through HF→Megatron) is still open. Bakeoff proceeds regardless; V4 baseline runs twice (HF + post-conversion) per the round-3 doc.
- **Prepare_greek_pool.sh + mix_builder.py local_parquet path are untested**. Smoke them on small inputs before kicking off the full 7B-token build.
- **HF→Megatron loader untested at scale**. R1 roundtrip on Apertus-8B-2509 must pass before submitting the bakeoff arms.

Good night. I'll be available when you wake up.
