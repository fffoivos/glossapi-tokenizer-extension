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

This repo is organized around the four things a reviewer or user needs first.

| Top-level actor | What it is |
|---|---|
| `selected-tokenizer/` | The selected modern Greek Apertus tokenizer extension, `ModernGreek-148k`. |
| `dataset/` | The `CPT-7B-mix` recipe, source graph, and hydration paths. |
| `checkpoints/` | Checkpoint metadata, locations, and uploaded model weights. |
| `evals/` | Benchmark summaries and plots. |

Everything else is under `supporting/`: optional tokenizers, source-code links,
provenance, archive notes, and checksums.

## Selected Tokenizer

`selected-tokenizer/` contains `ModernGreek-148k`, the tokenizer selected for
the Apertus CPT experiments:

- base Apertus vocab: `131072`;
- added modern Greek C3 tokens: `17408`;
- total vocab: `148480`;
- `tokenizer.json` SHA-256:
  `358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394`.

The optional polytonic/ancient Greek stacked tokenizer is not part of the main
CPT checkpoint line. It lives at:

```text
supporting/optional-tokenizers/ModernGreek-Polytonic-154k/
```

## Dataset

`dataset/` describes `CPT-7B-mix`, built from:

- `fffoivos/glossapi-greek-nanochat-pretraining-dataset`;
- nanochat internal dedup metadata;
- Apertus-overlap drop overlay from
  `fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z`;
- non-Greek replay, code, and math.

Bulk recipe:

- Greek: `70%`;
- non-Greek replay: `24%`;
- code: `4%`;
- math: `2%`.

The final text stream is about `7.0B` extended-tokenizer-budget tokens. The
same NFC JSONL stream contains `9,831,704,774` base-tokenized Megatron tokens
after preprocessing.

## Checkpoints

`checkpoints/` is the only top-level place for model checkpoints.

Primary selected checkpoint:

```text
checkpoints/TokenDistil-3.5B/
```

Its source HF-format copy on Clariden is:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_td_layer11/iter_0000834_hf
```

Comparison and source locations live under:

```text
checkpoints/locations/
```

## Evals

The current result anchor is:

```text
evals/3.5B-comparison/
```

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

See `supporting/source-code/manifest.json` for script-family pointers.
