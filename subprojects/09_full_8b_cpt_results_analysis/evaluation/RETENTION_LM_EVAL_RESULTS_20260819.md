# Downstream retention across the 8B CPT trajectory (Apertus Table-14 suite)

Date: 2026-08-19. Status: completed, first pass; **not independently reviewed**.

Five trajectory checkpoints scored on the Apertus pretraining-eval retention
set (`arc_challenge, arc_easy, hellaswag, winogrande, piqa, mmlu, global_mmlu,
xnli, xcopa`, harness-default shots, bf16, `--log_samples`). This is the
downstream counterpart of the replay-BPB forgetting evidence: the run's loss
panels showed late English/code/math/de/ru/zh degradation; this measures
whether it is behaviorally visible.

## Provenance

- Runtime: rebuilt from the frozen lock
  `evaluation/lm_eval_runtime_requirements_0_4_11.txt` (scientific bundle
  `20260804T220000Z-...-v79-sharded-retention`) = swiss-ai lm-eval fork
  0.4.11 + accelerate 1.13.0 + the campaign `global_mmlu` alias, installed at
  `/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval` (the May install found
  destroyed — dangling `__pycache__` symlinks — preserved as
  `lm_eval.broken-20260819`; filed as
  [apertus-cscs-efficiency#123](https://github.com/fffoivos/apertus-cscs-efficiency/issues/123)).
- Jobs: debug smoke `3123628`; first wave `3124429` (iter_0002384 completed;
  three slots killed by anonymous HF rate-limiting on the shared NAT IP);
  offline rerun `3124778` (COMPLETED 23m04s) reusing the completed slot's
  dataset cache with `HF_HUB_OFFLINE=1`.
- Results: `/iopsstor/scratch/cscs/fffoivos/evals/full8_retention_20260819/retention_only/iter_*/`
  (per-sample logs included); copies in
  [`../evidence/retention_lm_eval_20260819/`](../evidence/retention_lm_eval_20260819/).
- Anchor: base `Apertus-8B-2509` V4-HF baseline (May, `V4_BENCHMARK_COMPARISON.md`).
  Caveat: the anchor ran on the (now destroyed) May install; the retention
  tasks are standard and iter-0 continuity looks sane, but a ~30-minute
  re-anchor of the base model under the rebuilt runtime would close this gap.

## Results (%, Table-14 metric conventions)

| Task | base | init 0 | 10B | 40B peak | 61B cd-start | 76.7B term |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arc_challenge | 58.70 | 51.19 | **56.40** | 52.82 | 53.16 | 51.37 |
| arc_easy | 83.63 | 76.47 | **82.37** | 78.62 | 79.59 | 79.21 |
| hellaswag | 78.84 | 61.78 | **78.03** | 77.07 | 76.00 | 76.37 |
| winogrande | 69.30 | 63.06 | 69.85 | **69.93** | 69.14 | 68.51 |
| piqa | 79.92 | 74.97 | **80.30** | 79.71 | 80.25 | 79.11 |
| mmlu | 59.23 | 57.53 | **60.01** | 56.64 | 55.40 | 56.00 |
| global_mmlu (lite) | 52.46 | 51.75 | **52.40** | 50.80 | 49.10 | 49.18 |
| xcopa | 65.75 | 63.27 | **65.47** | 63.75 | 62.55 | 62.87 |
| xnli | 44.00 | 41.16 | **45.04** | 43.83 | 43.70 | 43.95 |

Language slices (acc %):

| Slice | init 0 | 10B | 40B | 61B | 76.7B |
| --- | ---: | ---: | ---: | ---: | ---: |
| xnli_el | 36.39 | 41.24 | **43.05** | 41.37 | 42.25 |
| xnli_de | 48.71 | **51.16** | 50.64 | 49.84 | 49.08 |
| xnli_ru | 46.55 | **49.16** | 42.97 | 43.01 | 43.94 |
| xnli_zh (chance .33) | 35.90 | 35.66 | 34.34 | 33.45 | 34.18 |
| xnli_en | 43.17 | 54.34 | 54.22 | 54.58 | 54.46 |
| global_mmlu_en | 62.75 | **64.25** | 59.25 | 60.00 | 58.25 |
| global_mmlu_de | 59.00 | **60.50** | 57.00 | 56.25 | 54.75 |
| global_mmlu_zh | 57.25 | 56.50 | 52.75 | 51.75 | 55.50 |
| xcopa_zh | 61.40 | **73.00** | 71.40 | 70.20 | 70.20 |

## Reading

1. **The 10B checkpoint (iter 2,384) is the retention optimum** — it matches
   or beats base Apertus on 5 of 9 tasks (mmlu 60.0 vs 59.2, piqa, winogrande,
   xnli, ~hellaswag) after fully recovering the init dip. At 10B the model had
   recovered from the extension/TD initialization and had not yet forgotten.
2. **Downstream forgetting after ~10–40B is real and measurable**: from 10B to
   terminal, mmlu −4.0, global_mmlu −3.2, arc_challenge −5.0, arc_easy −3.2,
   xcopa −2.6. The replay-BPB forgetting is behaviorally confirmed at the
   ~3–5 pp scale on knowledge/reasoning tasks.
3. **Greek is the only language that rises** (xnli_el +6 pp over init, peaking
   at 40B) while de/ru/es/it/ja knowledge slices fall 3–6 pp — the intended
   trade, now quantified on independent instruments.
4. **The init cost is task-dependent and mostly transient**: hellaswag −17 pp
   at iter 0 recovering to −0.8 by 10B; mmlu barely dips at init (−1.7).
5. Nuances: xnli_en is flat late (English NLI robust) while English *knowledge*
   (mmlu/global_mmlu_en) drops; xnli_zh sits near the chance floor throughout;
   global_mmlu is the Lite subset (~400 items/language → per-language stderr
   ≈ ±2.4 pp, so single-language deltas are ~1–2σ; trust the aggregates).

## Implication for checkpoint selection and averaging

Retention favors *earlier* checkpoints, adding a third axis to the existing
two (GreekMMLU peaks at 40B; several native-Greek tasks peak elsewhere). All
five cooldown checkpoints are approximately equally forgotten, so a last-5
average cannot recover retention; a peak-window average inherits ~40B-level
retention, ~1–2 pp better than terminal on most tasks. When merged models are
built, this suite reruns offline in one ~25-minute debug job (shared dataset
cache at `evals/full8_retention_20260819/retention_only/cache/3124429_iter_0002384`).
The iter-2384 (10B) checkpoint deserves consideration in any soup/interpolation
arm as the retention anchor.
