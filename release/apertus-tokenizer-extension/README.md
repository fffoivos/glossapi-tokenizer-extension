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

Measured source-token count with the selected tokenizer:

| Source slice | Tokenizer | Rows | Tokens, no EOD | Tokens, +1 EOD/doc |
|---|---|---:|---:|---:|
| `HPLT/ell_Grek_ge8_no_mt_clean60` | `ModernGreek-148k` | 48,728,774 | 44,195,950,025 | 44,244,678,799 |

See `cpt-training-dataset/token-counts.json` for the exact run metadata.

## Experiment Checkpoints

`experiment-checkpoints/` contains one folder per checkpoint we care about. Each has a `manifest.json` + `README.md` with the source Clariden path. **5B is the bakeoff final endpoint**; 2B and 3.5B are iso-token snapshots.

| Checkpoint | Meaning |
|---|---|
| `TokenDistil-Init/` | Token Distillation initialization before CPT. |
| `TokenDistil-2B/` | Token Distillation after the 2B bakeoff. |
| `TokenDistil-3.5B/` | Token Distillation after the 3.5B continuation. |
| `TokenDistil-5B/` | **Token Distillation at bakeoff-final 5B endpoint.** |
| `Vanilla-2B/` | Original-tokenizer control after the 2B bakeoff. |
| `Vanilla-3.5B/` | Original-tokenizer control after the 3.5B continuation. |
| `Vanilla-5B/` | **Original-tokenizer control at bakeoff-final 5B endpoint.** |
| `ReTok-2B/` | ReTok baseline after the 2B bakeoff. |
| `ReTok-3.5B/` | ReTok baseline after the 3.5B continuation (stopped here — dominated by TD). |
| `Centroid-2B/` | Centroid baseline after the 2B bakeoff (stopped here — broken arm). |

Large model weights are uploaded to Hugging Face in these folders. They are not
mirrored in the GitHub source repository.

## Benchmark Evals

The current result anchor is:

```text
benchmark-evals/bakeoff-final/    # 5.0B endpoint, canonical headline
benchmark-evals/3.5B-comparison/  # 3.5B iso-token (Vanilla/ReTok/TD)
benchmark-evals/native-greek-suite/ # vetted native-Greek decision suite
```

Loss-reading rule: raw Megatron `lm loss` is per-token cross entropy and is
not comparable between the original 131,072-token Vanilla tokenizer and the
148,480-token extended tokenizer arms. Cross-arm loss conclusions use heldout
BPB from the tokenizer-fair eval jobs plus downstream benchmark scores. Older
files may call BPB `BPC`; that is a legacy bits-per-byte label, not bits per
character. Raw training loss is only a health and within-arm trace unless dense
`bpb` training logs are present.

Greek aggregate rule: the Greek-specific headline now uses
`benchmark-evals/native-greek-suite/`, which includes vetted native MCQ tasks
and excludes explicit MT diagnostics. The older `bakeoff-final/` Greek
aggregate is a fallback lm-eval slice and should not be treated as the
native-Greek selection headline.

At 5.0B (`bakeoff-final/`):

| Arm | Greek no-MT aggregate | English retention | Multilingual | Heldout BPB, lower better |
|---|---:|---:|---:|---:|
| Vanilla-5B | 0.4076 | 0.6799 | 0.4936 | **0.4602** |
| TokenDistil-5B | **0.4204** | **0.6903** | **0.4976** | 0.4872 |
| ReTok-3.5B (stopped) | 0.3984 | 0.6786 | 0.4864 | 0.5390 |
| Centroid-2B (stopped) | 0.2566 | 0.6836 | 0.4888 | 0.8994 |

Reading for the older fallback suite: `TokenDistil-5B` leads all three
downstream aggregates over `Vanilla-5B`. `Vanilla-5B` retains tokenizer-fair
heldout BPB leadership; gap narrowing (0.110 → 0.027 over the bakeoff).

Native-Greek suite reading:

| Arm | Native MCQ headline | MCQ + Plutus | greek-nlp supporting mean |
|---|---:|---:|---:|
| Apertus-Base | **0.4817** | **0.4902** | **0.2150** |
| Vanilla-5B | 0.4305 | 0.4329 | 0.1679 |
| TokenDistil-5B | 0.4109 | 0.4160 | 0.1733 |

For Greek-specific selection, Vanilla is ahead of TokenDistil on the native MCQ
headline, while Apertus-Base remains above all continued checkpoints. Full
native-suite tables are in `benchmark-evals/native-greek-suite/`.

**Caveat — the bakeoff was not rule-bound.** The pre-commit decision-rule thresholds from `old_experiments_plan.md` v0.12 §10 Q8 (X / M_progress / M_ext / M_van / T) were never locked before results came in. The 5B headline above is an honest description of the numbers, not an adjudicated winner. See:

```text
supporting-material/provenance/decisions/PLAN_VS_RESULTS_RECONCILIATION_20260526.md
supporting-material/provenance/decisions/CPT_MASTER_20260526.md
```

## Supporting Material

| Path | Meaning |
|---|---|
| `supporting-material/provenance/decisions/` | Plans + plan-vs-results reconciliation + master synthesis |
| `supporting-material/provenance/evals/` | Eval recipe, loss-measurement policy, per-stage result docs |
| `supporting-material/provenance/token-distillation/` | TD plan + run log |
| `supporting-material/provenance/tokenizer-selection/` | Cutoff sweep + chosen-cutoff report + firing-count audit |
| `supporting-material/provenance/dataset-build/` | Mix recipe + dataset build runbook + manifest |
| `supporting-material/provenance/conversion-roundtrip/` | R17 HF↔Megatron verification JSONs |
| `supporting-material/optional-tokenizers/` | 154k modern + polytonic tokenizer (parked) |
| `supporting-material/source-code/` | Pointer back to the GitHub source repo |
| `supporting-material/archive/` | Legacy layout artifacts + checksums |

## Source

Runnable scripts live in GitHub:

```text
https://github.com/fffoivos/glossapi-tokenizer-extension/tree/main/subprojects/03_apertus_extension_and_embedding_adaptation
```
