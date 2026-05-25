# Apertus Extension Four-Actor Release Layout

Date: 2026-05-25.

Status: implemented in `release/apertus-tokenizer-extension/`.

## Goal

Make the release read from the top level as:

1. Greek extension tokenizer;
2. CPT training dataset;
3. experiment checkpoints;
4. benchmark evals.

Everything else should be supporting material.

## Previous Generic Names

```text
README.md
manifest.json

selected-tokenizer/
dataset/
checkpoints/
evals/
supporting/
```

Those generic names were replaced with the explicit public paths below.

## Implemented Top Level

```text
greek-extension-tokenizer/
cpt-training-dataset/
experiment-checkpoints/
benchmark-evals/
supporting-material/
```

## Meaning

| Path | Role |
|---|---|
| `greek-extension-tokenizer/` | Main tokenizer artifact: `ModernGreek-148k`; this is not the original Apertus tokenizer. |
| `cpt-training-dataset/` | Main dataset artifact: `CPT-7B-mix` recipe, source graph, and hydration paths. |
| `experiment-checkpoints/` | Main checkpoint area; one folder per experiment checkpoint. |
| `benchmark-evals/` | Main evaluation area; current anchor is `3.5B-comparison`. |
| `supporting-material/` | Optional tokenizer, provenance, source-code pointers, archive notes, and checksums. |

## Checkpoint Policy

`experiment-checkpoints/` contains one folder per experiment checkpoint:
`TokenDistil-Init`, `TokenDistil-2B`, `TokenDistil-3.5B`, `Vanilla-2B`,
`Vanilla-3.5B`, `ReTok-2B`, `ReTok-3.5B`, and `Centroid-2B`.

The large weight shards belong on Hugging Face in those folders, not in the
GitHub source repo. Public checkpoint names are not mixed with run tags, exact
iterations, TP size, or layer details.

## Supporting Material

The optional polytonic tokenizer is intentionally not top-level for this
Apertus CPT result:

```text
supporting-material/optional-tokenizers/ModernGreek-Polytonic-154k/
```

The implementation and audit trail are still preserved:

```text
supporting-material/source-code/
supporting-material/provenance/
supporting-material/archive/
supporting-material/ARTIFACTS.md
supporting-material/checksums.sha256
```
