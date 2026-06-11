# Apertus CPT artifacts and hydration

Status: current repo ownership policy, 2026-05-24.

This repo is the control plane for the Apertus Greek CPT work. It should contain
the scripts, recipes, manifests, verification outputs, handoff docs, and compact
eval evidence needed to reproduce or review the work. It should not contain the
large checkpoint and dataset payloads themselves.

The current Clariden disk inventory is:
[`CLARIDEN_INVENTORY_20260524.md`](CLARIDEN_INVENTORY_20260524.md).

## What belongs in git

Keep these in the tokenizer-extension repo:

- corpus recipes, source-mix manifests, and validation summaries;
- small JSON/CSV/TSV evidence files used by reviewers;
- Slurm launchers and dry-run submission plans;
- HF-to-Megatron conversion scripts and R17/xIELU/QK-Norm verification reports;
- TD coverage/training/eval scripts and summaries;
- final eval digests, trajectory analysis, and plots;
- handoff docs that explain which remote artifact is authoritative.

Do not commit:

- `.safetensors`, `.distcp`, `.bin`, `.idx`, `.pt`, `.pth`, `.ckpt`, `.gguf`;
- raw JSONL/parquet corpora;
- per-sample eval logs unless a reviewer explicitly asks for a small excerpt;
- full run directories copied from Clariden.

The top-level `.gitignore` now blocks the common checkpoint/dataset binary
extensions so the repo records pointers and manifests, not terabytes of payload.

## Authoritative remote artifacts

| Need | Clariden path |
|---|---|
| Apertus base HF model / teacher | `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/` |
| Extended tokenizer | `/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/` |
| Production NFC base-tokenized Megatron prefix | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document` |
| Production Vanilla TP=2 R17-patched init | `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched/release/` |
| TD layer-11 TP=2 R17-patched challenger init | `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched/release/` |
| Final Vanilla/ReTok/Centroid bakeoff runs | `/capstor/scratch/cscs/fffoivos/runs/bakeoff/bakeoff_1node_chain_20260522_005620_{vanilla,retok,centroid}/` |
| Final TD 2B run | `/capstor/scratch/cscs/fffoivos/runs/bakeoff/td_full25_layer11_2b_20260523T165038Z/` |
| Eval outputs | `/capstor/scratch/cscs/fffoivos/runs/eval/` |
| Clariden repo mirror | `/iopsstor/scratch/cscs/fffoivos/repo/` |

See the inventory file for sizes, intermediate artifacts, deprecated artifacts,
and paths that are intentionally absent.

## Production hydration check

Before launching production CPT, verify the minimum required remote state:

```bash
ssh clariden 'for p in \
  /iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/config.json \
  /iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/tokenizer.json \
  /iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document.bin \
  /iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document.idx \
  /iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched/release \
  /iopsstor/scratch/cscs/fffoivos/code/training/Megatron-LM-Swiss-AI \
  /iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval ; do
  if [ -e "$p" ]; then echo "OK  $p"; else echo "MISSING  $p"; fi
done'
```

All entries should print `OK`. If any entry is missing, do not launch the
production chain.

## Repo-to-Clariden sync

The Clariden mirror is the execution copy. After committing changes locally,
sync only the small repo-owned files needed by the run:

```bash
rsync -av --delete \
  --exclude='.git/' \
  --exclude='subprojects/**/data/' \
  --exclude='subprojects/**/artifacts/' \
  --exclude='subprojects/**/outputs/' \
  --exclude='subprojects/**/staging/' \
  /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/ \
  clariden:/iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/
```

Use this sync for docs/scripts/manifests. Do not use it to pull large Clariden
payloads back into the local repo.

## Rehydrating work for review or launch

From a fresh checkout on `home`:

1. Read [`REVIEW_HANDOFF_20260524.md`](REVIEW_HANDOFF_20260524.md).
2. Read [`CLARIDEN_INVENTORY_20260524.md`](CLARIDEN_INVENTORY_20260524.md).
3. Run the production hydration check above.
4. For review, inspect compact local evidence first:
   `03_4_implementation_experiments/init_bakeoff/eval/live_summaries/` and
   `03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/`.
5. For production launch, use the dry-run-validated launcher:

```bash
cd /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/production_cpt
DRY_RUN=0 CONFIRM_PRODUCTION_LAUNCH=1 bash submit_vanilla_base_15b_chain.sh
```

The launcher points at the remote checkpoint and dataset paths above. The repo
stores the launch recipe and proof that the referenced artifacts exist; Clariden
stores the payloads.
