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

The preflight is a one-node `debug` job. The scorer uses a one-node,
four-GPU, 80-minute debug profile in eighteen receipt-bound segments of
fourteen independent one-GPU shards. It preserves the same shard assignment,
FP32 scorer, prompts, examples and contamination subset as the earlier
two-node profile; Slurm simply schedules four workers concurrently rather
than eight. The completed two-node segment measured 9,946 total shard-seconds,
which implies about 41 minutes of scorer work on four GPUs before bounded
startup allowance. No `normal` allocation is used. The adapter must pass an
`sbatch --test-only` probe before its first debug submission.

After the one-time `sbatch --test-only` probe has passed, capture each
one-node debug segment visibly with `salloc`, audit the granted allocation,
then run the frozen segment via `srun` inside it. This avoids another queued
batch wrapper and leaves the allocation available for an immediate retry if a
step fails. Do not use an `sbatch` dependency graph for the suffix.

```bash
salloc --no-shell -A a0140 -p debug -N1 -n4 --ntasks-per-node=4 --gpus-per-node=4 \
  --gpus-per-task=1 --cpus-per-task=54 --mem=640G -t 01:20:00
# record the allocation id and verify its nodes with scontrol, then attach:
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
  env REMAINING12_WRAPPER_ROOT=... REMAINING12_ASSETS_ROOT=... \
  REMAINING12_OUTPUT_ROOT=... REMAINING12_SEGMENT_INDEX=N \
  REMAINING12_EXPECTED_NNODES=1 bash \
  .../evaluation/run_remaining12_native_segment.sbatch
```

## Publication sequence

Checkpoint *weight uploads* no longer wait for the final native-suite matrix.
They are repaired first as **private** immutable branches on
[`fffoivos/apertus-8b-greek-cpt`](https://huggingface.co/fffoivos/apertus-8b-greek-cpt),
using the canonical v1 checkpoint publisher plus an experiment-owned complete
Hub-inventory verifier. Each private card presents only its already-frozen
GreekMMLU point and explicitly says that the complete native-Greek matrix is
pending. No model branch is made public in this step.

The same Xfer-only release job stages two frozen training-data artifacts:

1. **Public:**
   [`fffoivos/apertus-8b-greek-cpt-modern-greek-train`](https://huggingface.co/datasets/fffoivos/apertus-8b-greek-cpt-modern-greek-train)
   contains only the exact selected Modern-Greek documents. It is reconstructed
   from the revision-pinned upstream v2 Parquet source, the selected training
   catalogs, and per-document content hashes. It performs no extra
   anonymization, deduplication, text transformation, or retokenization.
2. **Private:** `fffoivos/apertus-8b-greek-cpt-d0-full-mix` contains the exact
   portable packed 79/20/1 D0 payload, schedule, reader inputs, and provenance.
   It remains private because the replay-source redistribution matrix is not a
   public-release authorization.

The final metadata pass remains blocked on the complete 18-row native matrix.
Only it can replace the staging cards with the complete score table and only a
separate explicit decision can make the model repository public.

All preparation, hash sweeping, Parquet reconstruction, and Hugging Face
transfers run on a captured **Xfer** allocation. The normal checkpoint-
evaluation allocation is not used for release preparation.

The Xfer Python environment is an immutable x86 virtual environment whose
receipt binds the canonical replay/HF dependency lock plus the pinned Parquet
writer. It is created and verified on the captured Xfer node, never copied
from the Mac or a GPU UENV.
