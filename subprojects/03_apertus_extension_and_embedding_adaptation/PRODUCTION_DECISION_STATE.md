# Production Decision State

Status: current working decision, 2026-05-23

## Current selected path

Use **Vanilla Apertus-8B-2509 with the base 131,072-token tokenizer** as the
safe production default for the next 15-20B Greek CPT run.

Do not use Centroid. Do not use ReTok as-is for production. ReTok remains alive
only as a bounded Token Distillation challenger:

1. CPU firing/coverage prepass on `xfer`.
2. Small TD smoke and layer pilot only if coverage is sufficient.
3. Full `retok_td` only if the pilot improves ReTok pre-CPT Greek BPC and
   diagnostics without base-row/xIELU/QK drift.

If TD does not pass those gates, production stays Vanilla.

## Why Vanilla is the current default

Final 2B bakeoff checkpoint:
`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_iter0000476_summary.md`

| Arm | Greek BPC | Greek MMLU | Belebele el | XQuAD el F1 | ARC el | PIQA el | MMLU | HellaSwag | PIQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | **0.4906** | **0.421** | **0.513** | **0.310** | **0.421** | **0.600** | 0.534 | 0.759 | **0.795** |
| ReTok | 0.5739 | 0.399 | 0.497 | 0.309 | 0.372 | 0.580 | **0.554** | 0.749 | 0.786 |
| Centroid | 0.8994 | 0.279 | 0.341 | 0.026 | 0.256 | 0.510 | 0.544 | **0.760** | 0.794 |

Reading:

- Vanilla is best on every Greek downstream headline except XQuAD where it is
  effectively tied with ReTok.
- ReTok gives real tokenizer benefits, but they did not overcome the Greek loss
  gap after 2B tokens.
- Centroid is eliminated: its retention-looking scores are misleading because
  Greek BPC and Greek downstream behavior collapse.

## What ReTok still proved

ReTok is not broken. It improved segmentation/compression and trained its new
rows into usable ranges:

| Signal | Vanilla | ReTok | Centroid |
|---|---:|---:|---:|
| tokens/word | 2.693 | **1.735** | **1.735** |
| chars/token | 2.557 | **3.973** | **3.973** |
| STRR | 0.270 | **0.446** | **0.446** |
| D1 new-token top1 | n/a | **0.350** | 0.100 |
| D2 avg new-token mass | 0.000 | **0.344** | 0.341 |
| D4 top1-is-new rate | n/a | **0.598** | 0.336 |
| D5 generation new-token use | 0.000 | **0.358** | 0.092 |

This is why Token Distillation is a bounded challenger rather than deleted
from the project. The specific question is now narrow: can TD move ReTok's new
rows enough to close the Greek BPC/downstream gap while keeping its compression
benefit?

## Conversion and initialization safety

The production path must use the R17-preserving conversion route documented at:
`03_4_implementation_experiments/init_bakeoff/megatron_patches/README.md`

Validated patched TP=2 init checkpoints:

| Arm | Job | Tensor diff | Logit diff |
|---|---:|---:|---:|
| Vanilla | 2341182 | 0.0 | 0.0 |
| ReTok | 2341239 | 0.0 | 0.0 |
| Centroid | 2341241 | 0.0 | 0.0 |

The raw HF -> Megatron -> HF route is not acceptable because it resets the 128
xIELU R17 values. The accepted route patches xIELU/QK-Norm extras and verifies
standard tensors, R17 tensors, xIELU, QK-Norm, and smoke logits.

## Corpus path

Use the mix documented at:
`03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md`

Current bulk recipe:

- 70% Greek from the nanochat-derived corpus after Apertus-overlap removal and
  internal deduplication.
- 24% non-Greek replay across the documented 24-language tier set.
- 4% code. The live bakeoff used `codeparrot/codeparrot-clean-train` as the
  accessible cleaned-code fallback, not exact StarCoder.
- 2% math from FineMath 3+.

All CPU-only dataset building, preprocessing, and TD coverage/snippet mining
must run on `xfer`, not on GPU partitions.

## Token Distillation next gate

The first executable TD step is now:

```bash
cd /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation
sbatch td_coverage_prepass_xfer.sbatch
```

Local source:
`03_4_implementation_experiments/init_bakeoff/token_distillation/td_coverage_prepass.py`

Do not queue a GPU TD pilot until `td_coverage_summary.json` says either
`run_full_td_100` or `run_td_25_with_flagged_tail`.

## Production CPT monitoring cadence

For the real 15-20B CPT run:

- Save/evaluate lightweight intrinsic checkpoints every 500M tokens.
- Run tokenizer-fair Greek BPC/NLL, new-token diagnostics when relevant, and
  the retention subset at each saved checkpoint.
- Run the fuller downstream Greek + retention suite at least every 2B tokens
  and at the final 3-5 checkpoints for windowed selection.
- Keep all manifests: corpus recipe, tokenizer hash, init checkpoint, R17
  roundtrip report, training config, checkpoint IDs, eval JSONs, and summaries.

## Remaining gates before declaring the full objective complete

- TD coverage prepass has not yet been run.
- If TD coverage passes, TD smoke/layer pilot and possible full TD challenger
  remain to be run and compared.
- The final 15-20B production CPT dataset manifest needs to be built or
  rehydrated from the documented corpus path.
- The selected init checkpoint for production needs a final R17 roundtrip
  report attached to the production run directory.
