# Artifacts

This release is meant to tell the short artifact story first, with the
technical details in manifests and provenance.

```text
swiss-ai/Apertus-8B-2509
  |
  | continuous BPE on GlossAPI + HPLT 50/50
  v
C3 candidate tokenizer
  base 131072 + 25600 = 156672
  |
  | cutoff sweep, curation, backfill, alignment
  v
ModernGreek-148k
  base 131072 + modern Greek 17408 = 148480
  |
  | optional polytonic stack
  v
ModernGreek-Polytonic-154k
  base 131072 + modern Greek 17408 + polytonic Greek 5120 = 153600

fffoivos/glossapi-greek-nanochat-pretraining-dataset
  + nanochat internal dedup metadata
  + Apertus overlap drop overlay
  |
  v
selected_after_apertus_and_internal_dedup.parquet
  |
  | 70/24/4/2 Greek/replay/code/math mix
  v
CPT-7B-mix
  |
  | Megatron indexed dataset preprocessing
  v
bulk_mix_base_nfc_megatron/bulk_mix_text_document.{bin,idx}

ModernGreek-148k + CPT-7B-mix
  |
  | Vanilla, ReTok, Centroid, TokenDistil
  v
2B bakeoff and 3.5B continuation
  |
  v
TokenDistil-3.5B
  |
  v
3.5B-comparison
```

`checkpoints/` is reserved for actual model weights. Until weights are uploaded
to this HF repo, model artifacts are represented by `locations/*.md` files with
Clariden source paths and technical metadata.
