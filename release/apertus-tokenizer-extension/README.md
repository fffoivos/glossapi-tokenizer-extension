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

This release packages the Greek tokenizer-extension work around its main
artifacts:

1. a modern Greek merge-rule tokenizer extension for `swiss-ai/Apertus-8B-2509`;
2. an optional polytonic/ancient Greek stacked tokenizer;
3. the CPT dataset recipe and hydration pointers used for the bakeoff;
4. the init and Token Distillation checkpoint pointers;
5. compact benchmark evidence from the 2B bakeoff and 3.5B continuation.

The runnable scripts live in GitHub, not in this Hugging Face artifact repo:

```text
https://github.com/fffoivos/glossapi-tokenizer-extension/tree/main/subprojects/03_apertus_extension_and_embedding_adaptation
```

## Main Artifacts

| Artifact | Status | Path |
|---|---|---|
| Modern Greek tokenizer, 148,480 vocab | payload included | `tokenizer/modern-greek-17408/` |
| Polytonic-stacked tokenizer, 153,600 vocab | payload included | `tokenizer/polytonic-plus-5120/` |
| TD layer 11 CPT checkpoint at 3.5B | hydration pointer | `checkpoints/td-layer11-cpt-3p5b-iter834/` |
| TD layer 11 R17-patched init | hydration pointer | `checkpoints/td-layer11-init-r17-tp2/` |
| Vanilla/ReTok 3.5B baselines | hydration pointers | `checkpoints/baselines/` |
| CPT 7B text mix recipe | manifest and pointer | `training-data/cpt-7b-mix/` |
| 3.5B benchmark summary | compact evidence | `results/` |
| Source-code links | GitHub pointers | `code-links/` |
| Audit trail | compact provenance | `provenance/` |

## Tokenizers

Modern-only tokenizer:

- base Apertus vocab: `131072`;
- added modern Greek C3 tokens: `17408`;
- total vocab: `148480`;
- SHA-256 for `tokenizer.json`:
  `358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394`.

Polytonic-stacked tokenizer:

- base Apertus vocab: `131072`;
- modern Greek extension: `17408`;
- polytonic/ancient Greek extension: `5120`;
- total vocab: `153600`;
- SHA-256 for `tokenizer.json`:
  `b1eeb739a564b3abd33c1b85a16162b8284d98f9ab5d67528d3cbe8a82e9cbad`.

Both tokenizers preserve Apertus base ids and the first 1000 special/reserved
ids.

## Dataset

The CPT text mix is built from:

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

The latest compact result is the 3.5B continuation summary in `results/`.

At iter 834:

| Arm | Greek aggregate | English retention | Multilingual | Heldout BPC, lower better |
|---|---:|---:|---:|---:|
| Vanilla | 0.4339 | 0.6782 | 0.4923 | 0.4724 |
| ReTok | 0.4246 | 0.6786 | 0.4864 | 0.5390 |
| TD layer 11 | 0.4344 | 0.6865 | 0.4967 | 0.5054 |

Reading: TD layer 11 is the strongest final benchmark arm overall after the
3.5B continuation. Vanilla remains the heldout-BPC reference.

## Source And Provenance

- `MANIFEST.json`: machine-readable artifact inventory.
- `ARTIFACT_GRAPH.md`: dependency graph.
- `code-links/github_subproject_manifest.json`: source-code pointers.
- `provenance/`: compact documents and verification outputs for tokenizer
  selection, dataset build, TD, conversion, and eval.
- `archive/legacy-hf-layout-index.md`: index of older HF folders retained for
  traceability.
