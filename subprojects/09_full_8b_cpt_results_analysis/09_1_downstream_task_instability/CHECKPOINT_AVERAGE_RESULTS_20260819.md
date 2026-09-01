# Checkpoint-average arms: built and evaluated (stage 2, first results)

Date: 2026-08-19. Status: native suite + retention complete and validated;
clean GreekMMLU in flight. **Not independently reviewed.**

## Arms

Uniform mean of five consecutive checkpoints (Nishida et al. arXiv:2510.04848),
fp32 accumulation, bf16 output (source precision), config/tokenizer identity
verified across sources; averaging receipts in each output dir:

- `avg_peak5` = iters {7152, 8344, 9536, 10728, 11920} (30–50B)
- `avg_cooldown5` = iters {14627, 15496, 16688, 17880, 18284} (the cooldown)
- Models: `/capstor/scratch/cscs/fffoivos/experiments/full8_checkpoint_averaging_20260819/{peak5_uniform,cooldown5_uniform}`
- Merge code: `code/merge_checkpoint_averages.py` (same dir).

## Native suite (frozen FP32 legacy scorer, clean 73,894-example subset)

Same code bundle (`036e1e2e…`), contract/manifest (`clean_assets_v5`), flags and
21-shard layout as the authoritative single-checkpoint matrix. Evidence:
`/iopsstor/scratch/cscs/fffoivos/evals/full8_merged_native_20260819/` (jobs
3125834/3125835; Slurm state FAILED is a wrapper shell artifact — a `&&`
short-circuit in the round-robin loop; all 42 shards completed and row counts
match the clean subset exactly).

| Benchmark (acc %) | avg_peak5 | avg_cooldown5 | iter 9536 | terminal | best single (19) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ASEP MCQA | **56.19** | 55.08 | 54.83 | 55.51* | 55.51 @11920 |
| DemosQA | 46.91 | 46.24 | 46.41 | 46.91 | 47.75 @5960 |
| GPCR | 60.82 | 61.86 | 61.34 | 62.89* | 63.40 @10728 |
| Medical MCQA | **42.48** | 39.38 | 40.81 | 41.53* | 44.87 @3576 |
| OYXOY NLI | **65.12** | 43.34 | 62.22 | 59.94* | 65.47 @14304 |
| OYXOY WiC | 54.94 | 61.66 | 39.31 | 33.77* | 77.60 @400 |
| OYXOY metaphor | 34.48 | 33.89 | 35.46 | 33.89 | 69.93 @2384 |
| OYXOY WSD | 38.48 | 38.20 | 38.67 | 38.90* | 38.90 @11920 |

(CORRECTED 2026-08-20: the terminal column previously borrowed 50B values.
Exact iter-18284 values: ASEP 55.08, DemosQA 46.58, GPCR 62.89, Medical 38.42,
metaphor 33.89, NLI 38.73, WiC 33.64, WSD 38.06. Against these, avg_cooldown5
BEATS terminal broadly: WiC +28.0, NLI +4.6, Medical +0.96. Bold = beats every
single checkpoint in the arm's own window.)

Highlights: `avg_peak5` sets a **new all-time ASEP best** (56.19 vs 55.51 over
all 19 singles) and beats every in-window single on Medical (42.48 vs 41.77)
and NLI (65.12 vs 63.96). `avg_cooldown5` **reproduces the ensemble-forecast
NLI collapse in weight space** (43.34 vs 58.12 window best — opposing label
biases average to worse-than-any-ingredient), while beating its own ensemble
forecast on WiC (61.66 vs 56.74 E2).

## Retention (Apertus Table-14 suite, lm-eval 0.4.11 locked runtime, bf16)

Full 19-checkpoint trajectory now complete (jobs 3124429/3124778/3125685/3125686);
merged arms scored in job 3126302. Macro over the 9 tasks:

| Model | Macro (%) |
| --- | ---: |
| base Apertus-8B-2509 (V4-HF anchor) | 65.82 |
| best single (iter 400, 1.7B) | 66.10 |
| **avg_peak5** | **64.58** |
| iter 9536 (40B) | 63.68 |
| avg_cooldown5 | 63.36 |
| terminal (76.7B) | 62.95 |

`avg_peak5` beats the 40B single on **all nine** retention tasks (+0.90 macro)
and the terminal checkpoint by +1.63 — recovering roughly half the total
forgetting relative to the early-checkpoint optimum. This is the basin-center
effect: a gain the prediction-level ensemble could not forecast.
Full-19 finding: retention recovers from the TD-init dip within ~2B tokens
(macro 66.10 at iter 400 — above base), then declines monotonically; the
cooldown never recovers it.

## Reading

1. **avg_peak5 is the strongest all-round model produced by this run so far**:
   at-or-above the best in-window single on 3 of 8 native tasks (one all-time
   best), ~parity on the rest, and strictly better retention than any
   checkpoint from 20B onward.
2. **avg_cooldown5 is what the forecast said**: terminal-parity stabilization
   plus a real NLI regression; no retention recovery. The "average the last 5"
   recipe underperforms the peak-window alternative on every axis measured so
   far.
3. The offline ensemble forecast (OFFLINE_ENSEMBLE_RESULTS_20260819.md) was
   predictive within ~1 pp on most cells; weight averaging exceeded it where
   basin geometry matters (WiC +5, retention).

## Clean GreekMMLU (frozen FP32 exact evaluator, 16,159-question subset)

Job 3126432 (COMPLETED 55m59s); receipts under
`/iopsstor/scratch/cscs/fffoivos/evals/full8_merged_greekmmlu_20260819/`,
same clean-subset manifest (`61ed4ac9...`) and dataset revision (`6a03aa06...`)
as every campaign number:

| Model | Clean acc (%) | Choice NLL | Correct-answer BPB |
| --- | ---: | ---: | ---: |
| **avg_peak5** | **56.78** | 1.0895 | 0.1842 |
| iter 9536 (all-time peak) | 56.81 | 1.0740 | 0.1701 |
| avg_cooldown5 | 55.05 | 1.1259 | 0.1965 |
| terminal | 54.85 | 1.1221 | 0.1926 |

`avg_peak5` statistically ties the run's GreekMMLU peak (delta 0.03 pp at
n=16,159, stderr ~0.39 pp) while beating that same checkpoint on all nine
retention tasks, ASEP (all-time best), Medical and NLI. It is the strongest
all-round artifact this run has produced and the natural candidate for
release/continuation decisions. `avg_cooldown5` edges terminal on GreekMMLU
(+0.20) and retention (+0.41 macro) - stabilization, as forecast, no more.

Still open: NLI exact-set and balanced-accuracy columns from the merged
predictions; adoption of these results into the canonical presentations;
optional greedy-soup / peak-avg-to-terminal interpolation arms.

## Publication (2026-08-20)

Uploaded as private branches `18-avg-uniform5-tokens30B-50B` and
`19-avg-uniform5-tokens61B-77B` on `fffoivos/apertus-8b-greek-cpt`, cards in
the trajectory format (+ a Retention section), choice-NLL via the gate-verified
softmax-CE-over-avg_logprob formula, per-branch average_receipt.json +
provenance/training_data.json. Root: CHECKPOINTS.md averages table;
checkpoint-index.json `averages` list; `default_revision` moved from
`08-step9536-tokens40B` to `18-avg-uniform5-tokens30B-50B`; main README
replaced (was a verbatim copy of the branch-17 card) with a trajectory
overview + selection guide. Everything remains private.
