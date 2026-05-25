---
base_model: swiss-ai/Apertus-8B-2509
library_name: tokenizers
language:
  - el
tags:
  - tokenizer
  - greek
  - apertus
  - continued-pretraining
---

# Apertus Greek Tokenizer Extension

This repo has four front-stage artifacts.

| Path | Meaning |
|---|---|
| `greek-extension-tokenizer/` | The selected modern Greek extension tokenizer, not the original Apertus tokenizer. |
| `cpt-training-dataset/` | The CPT data recipe, source graph, and hydration paths. |
| `experiment-checkpoints/` | HF-format checkpoints for the experiment arms. |
| `benchmark-evals/` | Benchmark summaries and plots. |

Everything else is under `supporting-material/`.

## Greek Extension Tokenizer

`greek-extension-tokenizer/` contains `ModernGreek-148k`, the selected tokenizer
for these experiments:

- base Apertus vocab: `131072`;
- added modern Greek C3 tokens: `17408`;
- total vocab: `148480`;
- `tokenizer.json` SHA-256:
  `358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394`.

The original Apertus tokenizer is only used by the `Vanilla-*` checkpoints as a
control. The optional polytonic tokenizer lives under
`supporting-material/optional-tokenizers/`.

## CPT Training Dataset

`cpt-training-dataset/` describes `CPT-7B-mix`, built from:

- `fffoivos/glossapi-greek-nanochat-pretraining-dataset`;
- nanochat internal dedup metadata;
- Apertus-overlap drop overlay from
  `fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z`;
- non-Greek replay, code, and math.

Bulk recipe: `70%` Greek, `24%` non-Greek replay, `4%` code, `2%` math.

## Experiment Checkpoints

`experiment-checkpoints/` contains one folder per checkpoint we care about:

| Checkpoint | Meaning |
|---|---|
| `TokenDistil-Init/` | Token Distillation initialization before CPT. |
| `TokenDistil-2B/` | Token Distillation after the 2B bakeoff. |
| `TokenDistil-3.5B/` | Selected Token Distillation checkpoint after continuation. |
| `Vanilla-2B/` | Original-tokenizer control after the 2B bakeoff. |
| `Vanilla-3.5B/` | Original-tokenizer control after continuation. |
| `ReTok-2B/` | ReTok baseline after the 2B bakeoff. |
| `ReTok-3.5B/` | ReTok baseline after continuation. |
| `Centroid-2B/` | Centroid baseline after the 2B bakeoff. |

Large model weights are uploaded to Hugging Face in these folders. They are not
mirrored in the GitHub source repository.

## Benchmark Evals

The current result anchor is:

```text
benchmark-evals/3.5B-comparison/
```

Loss-reading rule: raw Megatron `lm loss` is per-token cross entropy and is
not comparable between the original 131,072-token Vanilla tokenizer and the
148,480-token extended tokenizer arms. Cross-arm loss conclusions use heldout
BPC/BPB from the tokenizer-fair eval jobs plus downstream benchmark scores.
Raw training loss is only a health and within-arm trace unless dense `bpb`
training logs are present.

At 3.5B:

| Arm | Greek aggregate | English retention | Multilingual | Heldout BPC, lower better |
|---|---:|---:|---:|---:|
| Vanilla | 0.4339 | 0.6782 | 0.4923 | 0.4724 |
| ReTok | 0.4246 | 0.6786 | 0.4864 | 0.5390 |
| TokenDistil | 0.4344 | 0.6865 | 0.4967 | 0.5054 |

Reading: `TokenDistil-3.5B` is the strongest final benchmark arm overall after
the 3.5B continuation. `Vanilla-3.5B` remains the heldout-BPC reference.

## Source

Runnable scripts live in GitHub:

```text
https://github.com/fffoivos/glossapi-tokenizer-extension/tree/main/subprojects/03_apertus_extension_and_embedding_adaptation
```
