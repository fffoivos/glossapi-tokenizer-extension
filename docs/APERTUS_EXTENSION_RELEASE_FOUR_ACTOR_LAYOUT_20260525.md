# Apertus Extension Four-Actor Release Layout

Date: 2026-05-25.

Status: implemented in `release/apertus-tokenizer-extension/`.

## Goal

Make the release read from the top level as:

1. selected tokenizer;
2. dataset;
3. checkpoints;
4. evals.

Everything else should be supporting material.

## Implemented Top Level

```text
README.md
manifest.json

selected-tokenizer/
dataset/
checkpoints/
evals/
supporting/
```

## Meaning

| Path | Role |
|---|---|
| `selected-tokenizer/` | Main tokenizer artifact: `ModernGreek-148k`. |
| `dataset/` | Main dataset artifact: `CPT-7B-mix` recipe, source graph, and hydration paths. |
| `checkpoints/` | Main checkpoint area; selected checkpoint is `TokenDistil-3.5B`. |
| `evals/` | Main evaluation area; current anchor is `3.5B-comparison`. |
| `supporting/` | Optional tokenizer, provenance, source-code pointers, archive notes, and checksums. |

## Checkpoint Policy

`checkpoints/TokenDistil-3.5B/` is the selected checkpoint slot. The large
weight shards belong on Hugging Face in that folder, not in the GitHub source
repo.

Comparison/source pointers live under:

```text
checkpoints/locations/
```

This keeps `checkpoints/` honest: public checkpoint names are not mixed with
run tags, exact iterations, TP size, or layer details.

## Supporting Material

The optional polytonic tokenizer is intentionally not top-level for this
Apertus CPT result:

```text
supporting/optional-tokenizers/ModernGreek-Polytonic-154k/
```

The implementation and audit trail are still preserved:

```text
supporting/source-code/
supporting/provenance/
supporting/archive/
supporting/ARTIFACTS.md
supporting/checksums.sha256
```
