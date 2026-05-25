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

`experiment-checkpoints/` is the top-level checkpoint area. It contains one
folder per public experiment checkpoint; source paths and exact technical
details live in each checkpoint manifest.

Loss evidence follows the tokenizer-fair policy in
`provenance/evals/LOSS_MEASUREMENT_POLICY.md`: heldout BPC/BPB and downstream
evals decide cross-arm comparisons. Raw Megatron `lm loss` is a training health
trace because the compared arms do not all use the same tokenizer.

Checkpoint weights were uploaded from Clariden through the non-GPU `xfer`
partition:

```text
Slurm job: 2382635
state: COMPLETED
exit code: 0:0
log: /users/fffoivos/apertus_hf_upload_checkpoints_20260525_2382635.log
```
