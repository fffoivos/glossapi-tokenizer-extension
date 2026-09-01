# Native-Greek three-checkpoint results

Date: 2026-08-12

## Scope

The same frozen examples were scored at initialization, update 9,536
(approximately 39.997B token slots) and the terminal update 18,284
(approximately 76.689B token slots). The new matrix contains 83,970 scored
examples per checkpoint across ASEP MCQA, DemosQA, GPCR, Medical MCQA and four
OYXOY task families. GreekMMLU is included below from its existing frozen,
decontaminated 16,159-question evaluation.

Every newly audited task is reported both on its complete public evaluation
split and after applying the immutable strict filter in
`evaluation/CONTAMINATION_DROP_DECISION_20260812.md`. A table entry of
`full / filtered` gives both values in that order. Lower choice NLL is better.

## Accuracy

| Benchmark | Initialization | 39.997B | 76.689B |
| --- | ---: | ---: | ---: |
| GreekMMLU clean | 35.78% | **56.81%** | 54.85% |
| ASEP MCQA | 28.75 / 28.47% | 55.08 / 54.83% | **55.25 / 55.08%** |
| DemosQA | 33.00 / 33.06% | 46.50 / 46.41% | **46.67 / 46.58%** |
| GPCR | 52.88 / 53.09% | 60.10 / 61.34% | **60.58 / 62.89%** |
| Medical MCQA | 24.54 / 24.58% | **40.28 / 40.81%** | 38.19 / 38.42% |
| OYXOY metaphor | **64.94 / 64.94%** | 34.96 / 35.46% | 33.73 / 33.89% |
| OYXOY NLI binary | 51.23 / 51.16% | **62.32 / 62.22%** | 38.69 / 38.73% |
| OYXOY NLI exact set | 10.50 / 10.35% | **18.50 / 18.31%** | 1.70 / 1.72% |
| OYXOY WIC | **74.37 / 76.14%** | 40.37 / 39.31% | 34.81 / 33.64% |
| OYXOY WSD | 36.91 / 35.87% | **40.29 / 38.67%** | 39.50 / 38.06% |

Raw accuracy is not a sufficient interpretation for the imbalanced OYXOY
binary tasks. Their balanced accuracies are:

| Benchmark | Initialization | 39.997B | 76.689B |
| --- | ---: | ---: | ---: |
| GPCR | 53.89 / 53.84% | 60.62 / 61.80% | **61.57 / 63.75%** |
| OYXOY metaphor | **52.40 / 52.42%** | 50.88 / 51.19% | 50.00 / 50.00% |
| OYXOY NLI binary | 49.63 / 49.54% | **55.32 / 55.21%** | 49.51 / 49.53% |
| OYXOY WIC | 49.86 / 49.83% | **52.17 / 52.44%** | 50.20 / 50.40% |

## Continuous choice loss

| Benchmark | Initialization | 39.997B | 76.689B |
| --- | ---: | ---: | ---: |
| GreekMMLU clean | 1.4586 | **1.0740** | 1.1221 |
| ASEP MCQA | 1.4760 / 1.4807 | **1.2543 / 1.2548** | 1.2600 / 1.2604 |
| DemosQA | 1.3682 / 1.3688 | 1.2854 / 1.2855 | **1.2853 / 1.2854** |
| GPCR | 0.7482 / 0.7541 | 0.6703 / 0.6698 | **0.6688 / 0.6677** |
| Medical MCQA | 1.7141 / 1.7192 | **1.5029 / 1.5030** | 1.5063 / 1.5057 |
| OYXOY metaphor | **0.6430 / 0.6444** | 0.8535 / 0.8420 | 1.2088 / 1.2008 |
| OYXOY NLI binary | 0.7284 / 0.7288 | **0.6660 / 0.6663** | 0.7642 / 0.7640 |
| OYXOY WIC | **0.5837 / 0.5673** | 0.7402 / 0.7444 | 0.7615 / 0.7659 |
| OYXOY WSD | 1.4087 / 1.4480 | **1.2687 / 1.2974** | 1.2977 / 1.3319 |

## Interpretation

The 39.997B checkpoint is the strongest current single-checkpoint choice across
this three-point screen. It is best on GreekMMLU, Medical MCQA, OYXOY NLI and
OYXOY WSD. The terminal checkpoint is best on GPCR and is effectively tied on
ASEP and DemosQA, so the evidence does not say that every useful capability
peaks at 40B. It says that continued training after 40B trades away several
exam and linguistic-judgment capabilities while leaving other MCQ tasks flat
or slightly better.

The OYXOY metaphor and WIC headline accuracies are dominated by class
imbalance. Their balanced accuracies remain close to chance, and their choice
NLL worsens rather than improves. They should be treated as diagnostics of a
changing label bias or task mismatch, not as evidence that the initialization
is a generally better Greek model.

Strict filtering does not reverse the central checkpoint ordering. It does
materially reduce three OYXOY lexical panels because their source dictionary
was present in training, which is exactly why both full and filtered results
are retained. The standard MCQ filters are small and have correspondingly
small effects, except that GPCR's 14 exclusions make its filtered accuracy a
few points higher.

No unweighted macro-score is declared: the tasks differ substantially in
size, class balance, semantics and contamination rate. Selection should use
the per-task continuous metrics and the previously measured learning/retention
panels rather than averaging these rows into one number.

## Immutable authorities

- matrix root:
  `/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_3cp_20260812/matrix_v6`;
- matrix receipt SHA-256:
  `207a0ca0d0b42e92106b8b7e5c7f4bc0da42450db595b5d08bd1b686b8e69297`;
- contamination-filtered results:
  `/capstor/scratch/cscs/fffoivos/benchmark_contamination_audits/runs/20260812T171530Z-native-greek-v1/filtered_scores_v1`;
- exact drop authority:
  `evaluation/CONTAMINATION_DROP_DECISION_20260812.md`.

Greek Protipa Exams is not included: its metadata is visible, but the
authenticated Parquet download remains HTTP 403 pending manual gate approval.
