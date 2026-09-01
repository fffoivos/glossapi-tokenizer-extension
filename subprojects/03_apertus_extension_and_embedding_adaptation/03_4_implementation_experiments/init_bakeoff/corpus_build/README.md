# corpus_build — the CPT training mix

> **In one line:** turn the deduplicated Greek pool plus non-Greek replay into a single shuffled JSONL stream at fixed source weights, NFC-normalise it, and tokenize it twice — once with the base 131,072 tokenizer for Vanilla and once with the extended 148,480 tokenizer for the extension arms.
> **Period:** 2026-05-20 (`11b5ba00`) → 2026-05-26 (token accounting). **Status:** completed; the bakeoff mix and the production-safe NFC binary both exist on Clariden.
> **Feeds:** [`../bakeoff_training/`](../bakeoff_training/README.md), [`../token_distillation/`](../token_distillation/README.md), [`../production_cpt/`](../production_cpt/README.md).

## Why this existed

The bakeoff's apples-to-apples claim rests on all arms seeing the same documents in the same order. That requires one deterministic builder, one shared seed, and **two** Megatron binaries built from the same JSONL — because the arms disagree only on tokenization, not on text. It also requires the Apertus-overlap drop to be applied uniformly, and NFC normalisation to be enforced before tokenization (verification V9).

## History

| Date | What happened | Result / decision | Evidence |
|---|---|---|---|
| 2026-05-20 | Mix recipes and streaming builder written; bulk = 70 % Greek / 26 % replay / 4 % code | first draft | `11b5ba00`, [`MIX_RECIPE.md`](MIX_RECIPE.md) |
| 2026-05-21 | FineMath added as a 2 % math bucket | Mix becomes the final **70 / 24 / 4 / 2** (Greek / replay / code / math) | `16b886c8` |
| 2026-05-21 | Reviewer round-2 blocker 3: the Apertus-overlap drop was being applied per-source, so it only hit HPLT; internal dedup was missing entirely | Rewritten as a **three-step path** — pull → `prepare_greek_pool.sh` (Apertus-drop **then** `drop_intra_and_inter`) → `mix_builder.py` off the single `$SELECTED` parquet. Order matters: dropping Apertus-overlap first lets an internal duplicate family keep a fresh alternate representative | [`MIX_RECIPE.md`](MIX_RECIPE.md) §"Three-step build path", [`../../../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md`](../../../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md) |
| 2026-05-21 | `prepare_greek_pool` (job `2334880`, 1h13m42s) then NFC normalise (`2335826`) | Selected pool: **47,061,862 rows, 227,837,744,625 chars**; NFC-normalised parquet 129.3 GB | `_archive/2026-05-21_overnight_session/CSCS_OVERNIGHT_STATE.md` |
| 2026-05-21 | Five mix-builder attempts cancelled in a row | Failures found and fixed in order: generator/HF interleave, eager setup, stale source labels, gated code sources, row-weighted token-share drift, then sharded-prefix duplication across array shards, then a 6 h walltime too tight for source-row sharding. Two external constraints landed here too — **HF rate-limited at 1000 requests / 5 min**, so replay and math were re-pointed at staged local parquets; and BigCode StarCoder was gated, so code fell back to `codeparrot/codeparrot-clean-train` (documented as *not* an exact StarCoder match) | same session log; `c300c0be`, `226a15ce`, `8d12d507`, `7c7412cf` |
| 2026-05-21 | Token-fair scheduler validated | Job `2336566`: target 50,000,000 tokens, actual **50,000,643** in 33,509 rows, scheduler `token_fair_min_tokens_over_weight` | [`smoke_20260521_token_fair/`](smoke_20260521_token_fair/README.md) |
| 2026-05-21 → 05-22 | Full sharded build (`2338295`, array `0-6%7`, 1 B tokens/shard, source-row-disjoint) → concat → two Megatron preprocess jobs | the bakeoff's `.bin`/`.idx` pair per tokenizer family | `356cfa6d`, `33de0642` |
| 2026-05-24 | Production-safe base-tokenized binary rebuilt from the NFC stream | Job `2367579` (`xfer`, 16m07s): **5,754,172 rows, 9,831,704,774 tokens**, 39.3 GB `.bin`. Needed because the bakeoff binary predated NFC cleanup and because `xfer` nodes have neither `uenv` nor a torch runtime — `preprocess_hf_jsonl_to_megatron.py` writes the same format without importing torch, validated byte-for-byte on the first 7,661,264 bytes against the canonical Megatron output (job `2367575`) | [`production_base_nfc_preprocess_2367579/`](production_base_nfc_preprocess_2367579/README.md) |
| 2026-05-26 | Token accounting | Full staged HPLT clean60 slice under the extended tokenizer: **48,728,774 rows, 44,195,950,025 tokens** (44,244,678,799 with one EOD per row), job `2399397`, 3h29m, ~3.52 M tokens/s. Four earlier attempts failed on `uenv` absence, a `regex` conflict from `lm_eval` on `PYTHONPATH`, and ARM-vs-x86 wheels — fixed with a dedicated xfer-built Python 3.11 env | [`HPLT_TOKEN_COUNT_RUN_20260526.md`](HPLT_TOKEN_COUNT_RUN_20260526.md), [`TOKEN_COUNT_AUDIT_20260526.md`](TOKEN_COUNT_AUDIT_20260526.md) |

## Outcome

- **Bakeoff mix:** 70 / 24 / 4 / 2, one shared JSONL text stream, two Megatron binaries. The extended-tokenizer mix-builder budget was 7,000,141,612 tokens; the base-tokenized production binary is 9,831,704,774 tokens.
- **The anneal recipe (`recipes/anneal.json`, 85 / 12 / 3) was written and never executed** — the bakeoff had no anneal phase, and production never launched.
- Code replay is `codeparrot`, not StarCoderData; this is a documented deviation from cpt_plan v0.7 §4.4 and remains open in `../../../CPT_MASTER_20260526.md` §9 Block 5.
- All CPU work here is pinned to the `xfer` partition by [`../check_cpu_only_slurm.sh`](../check_cpu_only_slurm.sh).

## Where things are

| What | Where |
|---|---|
| Bucket allocations + per-source weights, both phases | [`MIX_RECIPE.md`](MIX_RECIPE.md), [`recipes/bulk.json`](recipes/bulk.json), [`recipes/anneal.json`](recipes/anneal.json) |
| Build path | [`pull_greek_corpus.sh`](pull_greek_corpus.sh), [`pull_replay_datasets.sh`](pull_replay_datasets.sh), [`prepare_greek_pool.sh`](prepare_greek_pool.sh)/`.sbatch`, [`normalize_nfc.sh`](normalize_nfc.sh), [`mix_builder.py`](mix_builder.py) + `mix_builder_{smoke,full}.sbatch`, [`concat_bulk_mix.sbatch`](concat_bulk_mix.sbatch) |
| Torch-free Megatron preprocessing (for `xfer`) | [`preprocess_hf_jsonl_to_megatron.py`](preprocess_hf_jsonl_to_megatron.py) |
| Token counting | [`count_hplt_tokens.py`](count_hplt_tokens.py), [`count_apertus_overlap_tokens.py`](count_apertus_overlap_tokens.py), [`count_source_tokens.py`](count_source_tokens.py) + sbatch wrappers |
| Data on Clariden | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/` — `cpt/selected_after_apertus_and_internal_dedup.parquet`, `bulk_mix.nfc.jsonl`, `bulk_mix_base_nfc_megatron/bulk_mix_text_document.{bin,idx}` |

## Working documents

- [`smoke_20260521_token_fair/`](smoke_20260521_token_fair/README.md) and [`production_base_nfc_preprocess_2367579/`](production_base_nfc_preprocess_2367579/README.md) — per-run audit copies (job id, state, elapsed, output manifest). Historical receipts; the payloads stayed on Clariden.
- [`HPLT_TOKEN_COUNT_RUN_20260526.md`](HPLT_TOKEN_COUNT_RUN_20260526.md) — run log for one counting job, including the four failed launches; useful mainly as an `xfer`-environment troubleshooting record.
