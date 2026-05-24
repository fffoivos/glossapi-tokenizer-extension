# Production Decision State

Status: current production decision after TD 2B challenger, 2026-05-24

## Current selected path

Use **Vanilla Apertus-8B-2509 with the base 131,072-token tokenizer** as the
safe production default for the next 15-20B Greek CPT run.

Do not use Centroid. Do not use ReTok as-is for production. Do not promote
`td_full25_layer11` to the 15-20B production default unless compression/new-token
behavior is explicitly made the primary objective.

- `td_full25_layer11`: ReTok plus full-token 25-snippet Token Distillation at
  `target_layer=11`.

`td_full25_layer11` cleared R17-preserving HF -> Megatron conversion, exact HF
roundtrip verification, a Megatron load/train smoke, and a full 2B challenger
run. It is the strongest extended-tokenizer path, but the final 2B downstream
comparison remained mixed and did not beat Vanilla on the aggregate
Greek/preservation criteria.

## Why Vanilla is the current default

Final 2B bakeoff checkpoint and TD challenger digest:
`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_iter0000476_summary.md`
`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/td_full25_layer11_2b_20260523T165038Z_iter0000476_digest.md`

| Arm | Greek BPC | Greek MMLU | Belebele el | XQuAD el F1 | ARC el | PIQA el | MMLU | HellaSwag | PIQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | **0.4906** | **0.421** | **0.513** | **0.310** | **0.421** | **0.600** | 0.534 | 0.759 | **0.795** |
| TD full25 layer11 | 0.5311 | 0.386 | **0.528** | **0.326** | 0.404 | 0.550 | 0.550 | **0.761** | 0.792 |
| ReTok | 0.5739 | 0.399 | 0.497 | 0.309 | 0.372 | 0.580 | **0.554** | 0.749 | 0.786 |
| Centroid | 0.8994 | 0.279 | 0.341 | 0.026 | 0.256 | 0.510 | 0.544 | **0.760** | 0.794 |

Reading:

- Vanilla is still best on the aggregate Greek/downstream preservation criteria:
  it wins Greek MMLU, Greek ARC, Greek PIQA, global MMLU, and the base-44 Greek
  aggregate.
- TD is a real improvement over ReTok/Centroid and wins Belebele, XQuAD F1,
  MMLU, HellaSwag, XNLI, and XCOPA, but it does not dominate the Greek
  downstream headline set.
- ReTok and TD give real tokenizer benefits, but they did not overcome the
  aggregate Greek loss gap after 2B tokens.
- Centroid is eliminated: its retention-looking scores are misleading because
  Greek BPC and Greek downstream behavior collapse.

## What the extended path still proved

ReTok and TD are not broken. The extended tokenizer improves
segmentation/compression, and TD trains the new rows into more usable predictive
ranges:

| Signal | Vanilla | TD full25 layer11 | ReTok | Centroid |
|---|---:|---:|---:|---:|
| tokens/word | 2.693 | **1.735** | **1.735** | **1.735** |
| chars/token | 2.557 | **3.973** | **3.973** | **3.973** |
| STRR | 0.270 | **0.446** | **0.446** | **0.446** |
| D1 new-token top1 | n/a | **0.386** | 0.350 | 0.100 |
| D1 new-token top10 | n/a | **0.619** | 0.577 | 0.231 |
| D2 avg new-token mass | 0.000 | 0.342 | **0.344** | 0.341 |
| D4 top1-is-new rate | n/a | **0.634** | 0.598 | 0.336 |
| D5 generation new-token use | 0.000 | 0.208 | **0.358** | 0.092 |

This is why the extended-tokenizer line should be preserved as a documented
research/efficiency path. It is not the safest production default for the next
15-20B CPT run.

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

Production-safe Vanilla/base-tokenizer bulk data is now available:

- NFC JSONL:
  `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.nfc.jsonl`
- Base Megatron prefix:
  `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document`
- Preprocess job: `2367579`, `xfer`, `COMPLETED`, `0:0`, elapsed `00:16:07`
- Rows/sequences: `5,754,172`
- Base-tokenized tokens: `9,831,704,774`
- Validation: custom xfer fallback preprocessor matched the canonical Megatron
  preprocessor byte-for-byte on the first `1000` rows of the original stream.

Local evidence:
`03_4_implementation_experiments/init_bakeoff/corpus_build/production_base_nfc_preprocess_2367579/`

## Token Distillation state

The CPU coverage prepass, smoke run, layer pilot, full-token `25`-snippet TD
run, preservation checks, full-token intrinsic eval, R17 roundtrip, Megatron
smoke, and 2B downstream challenger have now run.

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
- 2B training completed cleanly:
  - initial job `2357704`, saved walltime handoff checkpoint `320`
  - resume job `2357705`, reached `476/476`, consumed `1.996B` tokens,
    loss `2.496399`, throughput `7874.3` tokens/sec/GPU, skipped `0`, NaN `0`
- Final checkpoint `476` eval chain passed:
  - conversion `2367150`, full lm-eval `2367151`, BPC `2367152`,
    diagnostics `2367153`

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

Final 2B challenger result:

| Signal | TD full25 layer11 | ReTok | Vanilla |
|---|---:|---:|---:|
| Greek BPC | 0.5311 | 0.5739 | **0.4906** |
| Greek MMLU | 0.3859 | 0.3991 | **0.4214** |
| Base-44 Greek aggregate | 0.4040 | 0.3732 | **0.4185** |
| Belebele el | **0.5278** | 0.4967 | 0.5133 |
| XQuAD el F1 | **0.3262** | 0.3092 | 0.3101 |
| MMLU | 0.5501 | **0.5542** | 0.5340 |

Decision: do not run a 50/100-snippet TD variant for the current production
choice. TD is strong enough to archive as a credible extended-tokenizer path,
but not strong enough to displace Vanilla for the next 15-20B CPT run.

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

- Full-token TD challenger finished. **Done; did not promote to production.**
- Full-token TD artifact preservation reports must pass for any candidate used
  downstream. **Done for `last` and `layer11`.**
- Full-token TD intrinsic eval must show whether `retok_td` meaningfully closes
  the ReTok gap. **Done; `layer11` clearly wins intrinsically.**
- The selected TD checkpoint passed R17-preserving HF -> Megatron conversion,
  exact roundtrip verification, and a bounded Megatron load/train smoke.
  Its 2B training/eval arm is complete. **Done; not promoted.**
- The selected Vanilla bulk data prefix is ready from the NFC-safe corpus.
  **Done for bulk.** Remaining production-data choice: decide whether the
  15-20B run repeats this bulk stream, builds a longer bulk stream, or adds the
  documented anneal stream as a separate phase.
- The selected Vanilla init checkpoint for production has local R17 roundtrip
  evidence. **Done.** Attach it to the concrete production run directory when
  that run directory is created.
