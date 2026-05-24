# Production CPT launcher

Status: prepared after the 2B Vanilla/ReTok/Centroid bakeoff and the bounded
Token Distillation challenger.

## Selected path

Use Vanilla Apertus-8B-2509 with the original base tokenizer.

Why:

- The 2B bakeoff selected Vanilla on the aggregate Greek/downstream criteria.
- `td_full25_layer11` was the strongest extended-tokenizer path, but it did not
  beat Vanilla on the aggregate production gate.
- Centroid and plain ReTok are not production defaults.

## Inputs

Init checkpoint, R17 preserved and roundtrip-verified:

```text
/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched
```

Training data, NFC-safe base-tokenized Megatron prefix:

```text
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document
```

Evidence for the data build lives at:

```text
../corpus_build/production_base_nfc_preprocess_2367579/
```

## Training shape

The launcher reuses the proven bakeoff trainer, with explicit production
overrides:

- `ARM=vanilla`
- `LOSS_OBJECTIVE=goldfish`
- `TRAIN_TOKENS=15000000000` by default
- `BASE_DATA_PREFIX` set to the NFC-safe base-tokenized prefix
- `SAVE_INTERVAL=120`, about 503M tokens per checkpoint
- `LR_WARMUP_TOKENS=TRAIN_TOKENS / 50`, a 2% re-warmup
- `ADEMA_*_WARMUP_STEPS=ceil(2.8% of train steps)`, restoring the
  Apertus-like production fraction rather than the heavy short-bakeoff warmup
- one node, four GH200 GPUs

The two-node path is not enabled here: the prior two-node smoke failed before
iteration 1 with NCCL/OFI `NO_SPACE`, while the one-node path completed the
2B runs cleanly.

## Dry-run audit

Run this on the Clariden mirror:

```bash
cd /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/production_cpt
DRY_RUN=1 bash submit_vanilla_base_15b_chain.sh
```

This writes a `submission_plan.json` and `submission_chain.tsv` under the
planned output directory, and prints the full `sbatch` chain without launching.

Validated dry run:

```text
dryrun_default_vanilla_base_15b_nfc_20260524T121007/
```

It generated the default 14-job chain with `LOSS_OBJECTIVE=goldfish`,
`TRAIN_TOKENS=15000000000`, `SAVE_INTERVAL=120`, `DEPENDENCY_MODE=afterok`,
and no Slurm jobs submitted.

## Live launch

After reviewing the dry run:

```bash
DRY_RUN=0 CONFIRM_PRODUCTION_LAUNCH=1 bash submit_vanilla_base_15b_chain.sh
```

The default `CHAIN_JOBS=14` is intentionally longer than the expected one-node
runtime for 15B tokens, so walltime handoffs have room. Each continuation job
loads from the run's own `checkpoints/` directory with optimizer/RNG state.
Dependencies default to `afterok` so a real failed job does not blindly launch
the rest of the chain; the proven walltime handoff exits cleanly.

To target 20B instead:

```bash
TRAIN_TOKENS=20000000000 CHAIN_JOBS=18 DRY_RUN=1 bash submit_vanilla_base_15b_chain.sh
```

## Evaluation cadence

Saved checkpoints arrive every ~500M tokens. Use lightweight BPC/NLL and
retention checks on each saved checkpoint, and the fuller downstream suite at
least every ~2B tokens and for the final checkpoint window.

The anneal recipe remains a design artifact, not an input to this launcher. It
must be rebuilt from the selected post-dedup Greek parquet and local staged
replay/code/math sources on `xfer` before it can become a second production
phase.
