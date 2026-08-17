# Full 8B checkpoint trajectory release

This experiment-owned adapter completes the clean native-Greek benchmark
matrix for the twelve full-8B CPT exports that were not previously scored.
It reuses the frozen FP32 legacy scorer, the exact post-hoc strict
contamination subset, and the receipt format from the completed three-point
and peak-window evaluations.

It owns only the selected checkpoint list, result joining and Hugging Face
release metadata. It does not change model weights, prompts, benchmark rows,
the scorer, the training corpus, or the existing six checkpoint results.

## Checkpoint scope

The six already evaluated CPT checkpoints are 30B, 35B, 40B, 45B, 50B and
76.689B token slots. This directory evaluates the remaining twelve exports:
1.678B, 5.000B, 9.999B, 14.999B, 19.998B, 24.998B, 54.996B, 59.995B,
61.350B, 64.995B, 69.995B and 74.994B.

GreekMMLU is not rerun: every saved checkpoint already has its frozen clean
GreekMMLU accuracy and choice-NLL receipt. Greek Protipa is intentionally out
of scope because its manual dataset access gate remains unapproved.

## Evaluation population

Every native-suite score uses the exact strict post-hoc subset already
published with `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2` at
revision `987b8955fcd395c6219e39df9e64715457f69065`:

| View | Source score rows | Strong-match exclusions | Retained rows |
| --- | ---: | ---: | ---: |
| ASEP MCQA | 1,200 | 20 | 1,180 |
| DemosQA | 600 | 1 | 599 |
| GPCR | 208 | 14 | 194 |
| Medical MCQA | 432 | 13 | 419 |
| OYXOY metaphor | 3,015 | 973 | 2,042 |
| OYXOY NLI | 5,286 | 42 | 5,244 |
| OYXOY WiC | 58,831 | 4,614 | 54,217 |
| OYXOY WSD definition | 14,398 | 4,399 | 9,999 |
| **Total** | **83,970** | **10,076** | **73,894** |

The exclusion policy removes only an evaluation identity with a strong
two-surface corpus match. The published audit contains the excluded IDs and
the document/line evidence; question-only candidates are not removed.

## Execution boundary

The preflight is a one-node `debug` job. The scorer then uses the previously
qualified two-node, 44-minute debug profile in eighteen receipt-bound
segments of fourteen independent one-GPU shards. No `normal` allocation is
used. The adapter must pass an `sbatch --test-only` probe before its first
debug submission.

`evaluation/coordinate_remaining12_debug.sh` is deliberately a **one-job**
Mac-side coordinator: it runs that required test-only probe, submits one
preflight or one segment, then audits the live Slurm node count, time limit
and `debug` partition. It never prequeues the suffix. Segment `n+1` is
eligible only when segment `n` has written its receipt, so a failed shard
cannot silently turn into an invalid later result.

Hugging Face publishing is deferred until the complete 18-row native matrix
and imported GreekMMLU table pass their consistency checks. The canonical
issue-104 branch performs private per-branch xfer releases and only promotes
the repository to public automatic gating once every branch has passed Hub
inspection.
