# CSCS overnight state (handoff)

*Started 2026-05-21 ~03:30 EEST. Foivos going to sleep around the start of this. Cluster: Clariden, account a0140, user fffoivos.*

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
