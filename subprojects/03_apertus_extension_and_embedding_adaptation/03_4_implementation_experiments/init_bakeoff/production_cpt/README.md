# production_cpt — the 15–20 B run that never fired

> **In one line:** a dry-run-validated launcher for the production Greek CPT on Vanilla Apertus with the base tokenizer and Goldfish loss restored; it was prepared on 2026-05-24 off the 2 B bakeoff conclusion and never launched from this subproject.
> **Period:** 2026-05-24 (`c4445bc5`, `29ce766e`). **Status:** prepared, never launched — superseded by the work that moved to [`../../../../04_cpt_training_regime_on_vanilla/`](../../../../04_cpt_training_regime_on_vanilla/) and [`../../../../05_token_distillation_cpt/`](../../../../05_token_distillation_cpt/).

## Why this existed

The bakeoff's purpose was to pick a starting point for a real CPT run. When the 2 B stage put Vanilla ahead, this directory captured that choice as an executable plan so it could be reviewed before any large compute was committed.

## History

| Date | What happened | Evidence |
|---|---|---|
| 2026-05-24 | Selected path recorded: **Vanilla Apertus-8B-2509 with the original base 131,072 tokenizer**, on the grounds that the 2 B bakeoff picked Vanilla on the aggregate criteria, `td_full25_layer11` was the strongest extended path but did not beat it, and Centroid/ReTok are not production defaults | this README's predecessor, `../../../_archive/synthesis_sources_20260526/PRODUCTION_DECISION_STATE.md` |
| 2026-05-24 | Launcher written and dry-run validated | [`dryrun_default_vanilla_base_15b_nfc_20260524T121007/`](dryrun_default_vanilla_base_15b_nfc_20260524T121007/) — a 14-job chain with `LOSS_OBJECTIVE=goldfish`, `TRAIN_TOKENS=15000000000`, `SAVE_INTERVAL=120`, `DEPENDENCY_MODE=afterok`, **zero Slurm jobs submitted** |
| 2026-05-26 | The premise weakened | The 3.5 B/5 B continuations put TD ahead on downstream aggregates, and the native-Greek suite put Apertus-Base ahead of every continued arm. `CPT_MASTER` §8 records the 2 B pick as "partially superseded" and "not rule-bound" |
| — | Never launched | No run directory, no Slurm job, no results doc exists for a production CPT under this path |

## What the launcher does

Reuses the bakeoff trainer with production overrides: `ARM=vanilla`, `LOSS_OBJECTIVE=goldfish`, `TRAIN_TOKENS=15000000000` (`20000000000` with `CHAIN_JOBS=18`), `SAVE_INTERVAL=120` (~503 M tokens per checkpoint), `LR_WARMUP_TOKENS = TRAIN_TOKENS/50` (a 2 % re-warmup), and `ADEMA_*_WARMUP_STEPS = ceil(2.8 % of train steps)` — restoring Apertus's own warmup fraction rather than the bakeoff's deliberately heavy short-horizon setting. One node, four GH200; the two-node path is deliberately disabled because the earlier two-node smoke failed with NCCL/OFI `NO_SPACE` while the one-node path completed every 2 B run cleanly. `CHAIN_JOBS=14` is intentionally longer than the expected runtime so walltime handoffs have room, and dependencies default to `afterok` so a genuine failure does not launch the rest of the chain.

Inputs: the R17-patched Vanilla TP=2 init checkpoint and the NFC-safe base-tokenized Megatron prefix built by [`../corpus_build/production_base_nfc_preprocess_2367579/`](../corpus_build/production_base_nfc_preprocess_2367579/README.md).

## Outcome

- **Blocked on three verifications** that were never closed here: V1 (eval-set decontamination), V8 (Goldfish hash uniformity across the new vocabulary), and applying the R17 patch to a *production* initial checkpoint. Listed in [`../../../CPT_MASTER_20260526.md`](../../../CPT_MASTER_20260526.md) §9.
- The anneal recipe stayed a design artifact — it would need rebuilding from the selected post-dedup parquet plus staged replay/code/math on `xfer` before it could be a second production phase.
- The real production CPT eventually ran, but from subproject 07 with the 148,992 tokenizer and a TD-based init — not from this launcher.

## Where things are

| What | Where |
|---|---|
| Launcher | [`submit_vanilla_base_15b_chain.sh`](submit_vanilla_base_15b_chain.sh) — `DRY_RUN=1` prints the chain and writes `submission_plan.json` + `submission_chain.tsv`; live launch needs `DRY_RUN=0 CONFIRM_PRODUCTION_LAUNCH=1` |
| Validated dry run | [`dryrun_default_vanilla_base_15b_nfc_20260524T121007/`](dryrun_default_vanilla_base_15b_nfc_20260524T121007/) |
| Init checkpoint / data prefix | Clariden `/iopsstor/.../init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched` and `/iopsstor/.../cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document` |

## Working documents

The dry-run directory is the only artifact: a submission plan and job chain for a run that was never submitted. Historical.
