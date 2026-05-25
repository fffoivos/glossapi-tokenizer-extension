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

This release is organized around the artifacts that matter:

1. `ModernGreek-148k`: the selected 17,408-token modern Greek BPE extension;
2. `ModernGreek-Polytonic-154k`: the optional stacked polytonic/ancient Greek tokenizer;
3. `CPT-7B-mix`: the dataset recipe used for the bakeoff and continuation;
4. `TokenDistil-3.5B`, `Vanilla-3.5B`, and `ReTok-3.5B`: the trained comparison line;
5. `3.5B-comparison`: the compact benchmark evidence.

The runnable scripts live in GitHub:

```text
https://github.com/fffoivos/glossapi-tokenizer-extension/tree/main/subprojects/03_apertus_extension_and_embedding_adaptation
```

## Main Artifacts

| Artifact | Status | Path |
|---|---|---|
| `ModernGreek-148k` | tokenizer payload included | `tokenizers/ModernGreek-148k/` |
| `ModernGreek-Polytonic-154k` | tokenizer payload included | `tokenizers/ModernGreek-Polytonic-154k/` |
| `TokenDistil-3.5B` | location only; weights not uploaded here yet | `locations/TokenDistil-3.5B.md` |
| `TokenDistil-2B` | location only | `locations/TokenDistil-2B.md` |
| `TokenDistil-Init` | location only | `locations/TokenDistil-Init.md` |
| `Vanilla-3.5B` | location only | `locations/Vanilla-3.5B.md` |
| `ReTok-3.5B` | location only | `locations/ReTok-3.5B.md` |
| `CPT-7B-mix` | recipe and hydration pointer | `datasets/CPT-7B-mix/` |
| `3.5B-comparison` | benchmark summary and plots | `results/3.5B-comparison/` |
| Source code | GitHub pointers | `source-code/` |
| Audit trail | compact provenance | `provenance/` |

## Checkpoint Weights

`checkpoints/` is reserved for actual model weights. The current HF release does
not yet include checkpoint weight shards, so checkpoint locations live under
`locations/`.

The selected public payload to upload first is:

```text
checkpoints/TokenDistil-3.5B/
```

Its current Clariden HF-format source is:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_td_layer11/iter_0000834_hf
```

See `checkpoints/README.md` for the current weight-upload status.

## Tokenizers

`ModernGreek-148k`:

- base Apertus vocab: `131072`;
- added modern Greek C3 tokens: `17408`;
- total vocab: `148480`;
- `tokenizer.json` SHA-256:
  `358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394`.

`ModernGreek-Polytonic-154k`:

- base Apertus vocab: `131072`;
- modern Greek extension: `17408`;
- polytonic/ancient Greek extension: `5120`;
- total vocab: `153600`;
- `tokenizer.json` SHA-256:
  `b1eeb739a564b3abd33c1b85a16162b8284d98f9ab5d67528d3cbe8a82e9cbad`.

Both tokenizers preserve Apertus base ids and the first 1000 special/reserved
ids.

## Dataset

`CPT-7B-mix` is built from:

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

## Current Result Anchor

The latest compact result is `results/3.5B-comparison/`.

At 3.5B:

| Arm | Greek aggregate | English retention | Multilingual | Heldout BPC, lower better |
|---|---:|---:|---:|---:|
| Vanilla | 0.4339 | 0.6782 | 0.4923 | 0.4724 |
| ReTok | 0.4246 | 0.6786 | 0.4864 | 0.5390 |
| TokenDistil | 0.4344 | 0.6865 | 0.4967 | 0.5054 |

Reading: `TokenDistil-3.5B` is the strongest final benchmark arm overall after
the 3.5B continuation. `Vanilla-3.5B` remains the heldout-BPC reference.

## Source And Provenance

- `manifest.json`: machine-readable artifact inventory.
- `ARTIFACTS.md`: dependency graph and artifact story.
- `source-code/manifest.json`: source-code pointers.
- `provenance/`: compact documents and verification outputs for tokenizer
  selection, dataset build, Token Distillation, conversion, and eval.
- `archive/legacy-layout.md`: index of older HF folders retained for
  traceability.
