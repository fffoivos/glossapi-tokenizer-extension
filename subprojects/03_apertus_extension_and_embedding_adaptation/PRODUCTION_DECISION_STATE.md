# Production Decision State

Status: current working decision, 2026-05-23

## Current selected path

Use **Vanilla Apertus-8B-2509 with the base 131,072-token tokenizer** as the
safe production default for the next 15-20B Greek CPT run until the new
`td_full25_layer11` candidate clears the downstream 2B training/eval retention
gate.

Do not use Centroid. Do not use ReTok as-is for production. The only extended
tokenizer candidate still alive is now:

- `td_full25_layer11`: ReTok plus full-token 25-snippet Token Distillation at
  `target_layer=11`.

`td_full25_layer11` has now cleared R17-preserving HF -> Megatron conversion,
exact HF roundtrip verification, and a Megatron load/train smoke. If it does
not also clear the downstream 2B training/eval retention gate, production stays
Vanilla.

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
| TD full25 layer11 | 2357565 | 0.0 | 0.0 |

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

## Token Distillation state

The CPU coverage prepass, smoke run, layer pilot, full-token `25`-snippet TD
run, preservation checks, and full-token intrinsic eval have now run.

- Training job: `2353960`, `COMPLETED`, elapsed `05:45:53`.
- Output root:
  `/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_full25_layers_20260523T092602Z`
- Candidates: `target_layer=-1` (`last`) and `target_layer=11` (`layer11`).
- Preservation checks passed on `xfer`: `2355706` (`last`) and `2355707`
  (`layer11`).
- Packed intrinsic eval passed: job `2355714`.
- R17-preserving HF -> Megatron -> HF roundtrip passed for selected
  `td_full25_layer11`: job `2357565`, exact tensor/logit diff `0.0`.
- Megatron load/train smoke passed: job `2357596`, five clean iterations,
  no skipped/NaN iterations.

Run log:
`03_4_implementation_experiments/init_bakeoff/token_distillation/RUN_LOG_20260523.md`

Full-token intrinsic eval:
`03_4_implementation_experiments/init_bakeoff/eval/td_full25_intrinsics_20260523T124000Z/TD_PILOT_INTRINSICS_SUMMARY.md`

Key result:

| Arm | BPC | D1 mean rank | D1 top1 | D1 top5 |
|---|---:|---:|---:|---:|
| ReTok | 2.9503 | 3868.27 | 0.0065 | 0.0231 |
| TD last | 1.4249 | 1756.04 | 0.0381 | 0.1596 |
| TD layer11 | **1.3846** | **1617.48** | **0.0415** | **0.1722** |

Current rule: take only `td_full25_layer11` to a decision-useful 2B training/eval
arm. Do not run a 50/100-snippet TD variant unless the 2B result is quality
ambiguous rather than clearly better or worse than Vanilla.

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

- Full-token TD challenger must finish.
- Full-token TD artifact preservation reports must pass for any candidate used
  downstream. **Done for `last` and `layer11`.**
- Full-token TD intrinsic eval must show whether `retok_td` meaningfully closes
  the ReTok gap. **Done; `layer11` clearly wins intrinsically.**
- The selected TD checkpoint passed R17-preserving HF -> Megatron conversion,
  exact roundtrip verification, and a bounded Megatron load/train smoke.
  It still needs a decision-useful 2B training/eval arm before any 15-20B
  production CPT promotion.
- The final 15-20B production CPT dataset manifest needs to be built or
  rehydrated from the documented corpus path.
- The selected init checkpoint for production needs a final R17 roundtrip
  report attached to the production run directory.
