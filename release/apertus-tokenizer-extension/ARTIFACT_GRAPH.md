# Artifact Graph

```text
swiss-ai/Apertus-8B-2509
  |
  | continuous BPE on GlossAPI + HPLT 50/50
  v
C3 full candidate
  base 131072 + 25600 = 156672
  |
  | cutoff sweep, curation, backfill, alignment
  v
tokenizer/modern-greek-17408
  base 131072 + 17408 = 148480
  |
  | optional polytonic stack
  v
tokenizer/polytonic-plus-5120
  base 131072 + modern 17408 + polytonic 5120 = 153600

fffoivos/glossapi-greek-nanochat-pretraining-dataset
  + nanochat internal dedup metadata
  + Apertus overlap drop overlay
  |
  v
selected_after_apertus_and_internal_dedup.parquet
  |
  | 70/24/4/2 Greek/replay/code/math mix
  v
bulk_mix.nfc.jsonl
  |
  | Megatron indexed dataset preprocessing
  v
bulk_mix_base_nfc_megatron/bulk_mix_text_document.{bin,idx}

modern tokenizer + CPT mix
  |
  | Vanilla, ReTok, Centroid, Token Distillation layer 11
  v
2B bakeoff and 3.5B continuation
  |
  v
checkpoints/td-layer11-cpt-3p5b-iter834
  |
  v
results/
```

The large checkpoint and dataset payloads are represented by hydration pointers
unless explicitly uploaded as model weights.

