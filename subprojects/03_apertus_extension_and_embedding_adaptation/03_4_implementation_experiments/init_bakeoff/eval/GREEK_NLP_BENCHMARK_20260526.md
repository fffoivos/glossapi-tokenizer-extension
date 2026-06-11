# greek-nlp/benchmark checkpoint eval - 2026-05-26

Purpose: add a native-Greek benchmark pass that was missing from the original
lm-eval bakeoff suite, using the upstream `greek-nlp/benchmark` task
definitions and metrics while swapping the Ollama backend for a local
Transformers backend that can load the HF-format Apertus checkpoints.

## Runner

- Local runner:
  `run_greek_nlp_benchmark_hf.py`
- Local Slurm wrapper:
  `run_greek_nlp_benchmark_hf.sbatch`
- Remote benchmark clone:
  `/iopsstor/scratch/cscs/fffoivos/benchmarks/greek-nlp-benchmark/main_e2d4bfc`
- Upstream commit:
  `e2d4bfc5472abacb32c8a8e1cda97d5eeb7e7460`
- Remote venv:
  `/iopsstor/scratch/cscs/fffoivos/python_envs/greek_nlp_benchmark_py311`

The HF backend keeps upstream tasks/prompts/loaders/metrics unchanged and only
implements the generation backend. Required extra packages beyond the upstream
requirements: `openpyxl`, `accelerate`, `sacrebleu`, `seqeval`, `rouge-score`,
`bert-score`, `nltk`.

## Checkpoints

- Vanilla 5B:
  `/capstor/scratch/cscs/fffoivos/runs/eval/continuation_5b_td_vs_vanilla_20260525T142522Z_vanilla/iter_0001192_hf`
- TD layer11 5B:
  `/capstor/scratch/cscs/fffoivos/runs/eval/continuation_5b_td_vs_vanilla_20260525T142522Z_td_layer11/iter_0001192_hf`

## Task Scope

Upstream tasks:

- `gec`
- `intent_classification`
- `legal_classification`
- `machine_translation`
- `ner`
- `pos`
- `summarization`

The `machine_translation` task is a Greek-source translation task, not an
English benchmark machine-translated into Greek. It should still be interpreted
separately from native-Greek understanding/classification aggregates.

## Run Log

1. Smoke job `2396209` failed on `gec` dataset loading because upstream
   requirements omitted `openpyxl`.
2. Added `openpyxl`; smoke job `2396218` reached model loading but failed
   because `device_map="auto"` requires `accelerate`.
3. Added `accelerate`; smoke job `2396222` completed all seven tasks with
   `SAMPLE_SIZE=1`, `NUM_PREDICT=64`.
   Output:
   `/capstor/scratch/cscs/fffoivos/runs/eval/greek_nlp_benchmark_20260526/smoke_vanilla_5b_retry2`
4. Submitted sample-100 jobs `2396234` (Vanilla) and `2396236` (TD) with
   `NUM_PREDICT=256`. These were cancelled after proving the path because
   label-style tasks were generating long outputs and wasting GPU time.
5. Patched runner with `--task-num-predict-overrides` so free-form tasks can
   keep larger caps while classification tasks use short caps.
6. Submitted optimized sample-100 jobs:
   - Vanilla `2396595`
   - TD `2396596`

Optimized cap profile:

```text
gec=128
intent_classification=16
legal_classification=16
machine_translation=128
ner=96
pos=96
summarization=192
```

Output targets:

- Vanilla:
  `/capstor/scratch/cscs/fffoivos/runs/eval/greek_nlp_benchmark_20260526/vanilla_5b_sample100_taskcaps`
- TD:
  `/capstor/scratch/cscs/fffoivos/runs/eval/greek_nlp_benchmark_20260526/td_5b_sample100_taskcaps`

Current status: optimized jobs `2396595` and `2396596` were cancelled
intentionally at about 17 minutes elapsed. The submitted environment did not
apply the intended task-specific generation caps: label-style tasks still ran
with `num_predict=128`, which wasted GPU time. Do not use those partial outputs
as decision evidence.

Next run should use the same cap profile, but verify in stdout before leaving
the job unattended:

```text
gec=128
intent_classification=16
legal_classification=16
machine_translation=128
ner=96
pos=96
summarization=192
```

The all-checkpoint rerun belongs under the native-Greek suite plan in
`NATIVE_GREEK_EVAL_SUITE_20260526.md`.

All-checkpoint sample-100 rerun status:

- Debug cap smoke `2396967` completed and confirmed the task-specific caps
  apply (`intent_classification` used `num_predict=16`).
- First packed rerun jobs `2396935`, `2396936`, `2396937` were invalidated by
  an upstream GEC temp-directory race: `gec_benchmark.py` creates a fixed
  `repo_244` under the process working directory, so packed workers collided.
- `run_greek_nlp_benchmark_hf.py` now resolves the per-model output directory
  and changes into it before calling upstream task code. This preserves
  upstream tasks/metrics while giving each worker a private temp area.
- Retry packed jobs `2396991`, `2396992`, `2396993` completed successfully
  with exit code `0:0`.
- Final summary is folded into
  `native_greek_suite_20260526/summary/` and interpreted in
  `NATIVE_GREEK_SUITE_RESULTS_20260526.md`.

## MT-Exclusion Policy

For the original lm-eval bakeoff calculations, explicit MT diagnostics
(`arc_challenge_mt_el`, `global_piqa_completions_ell_grek`) are excluded from
headline Greek aggregates. They remain in per-task tables only.

Docs/scripts updated to use the no-explicit-MT Greek aggregate:

- `summarize_3p5b_continuation.py`
- `summarize_5b_continuation.py`
- `regenerate_plots.py`
- `plot_van_td.py`
- `summarize_bakeoff.py`
- `CONTINUATION_3P5B_RESULTS_20260525.md`
- `CONTINUATION_5B_RESULTS_20260526.md`
- `BAKEOFF_FINAL_RESULTS_20260526.md`
- `PLAN_VS_RESULTS_RECONCILIATION_20260526.md`
- release 3.5B comparison README/CSV/summary JSON
- release provenance `EVAL_RECIPE.md`

Corrected 5B no-explicit-MT Greek headline:

| Arm | Iter | Greek no-MT aggregate |
|---|---:|---:|
| Vanilla | 1192 | 0.4076 |
| TD layer11 | 1192 | 0.4204 |

## Native-Greek Benchmarks Outside This Runner

Tracked in `NATIVE_GREEK_EVAL_SUITE_20260526.md`:

- cached + MCQ runner ready: GreekMMLU, ILSP Medical MCQA, ILSP ASEP MCQA,
  Plutus QA;
- cached + adapter/scoring pending: OYXOY, GreekBarBench, Greek civics,
  Greek lyceum mathematics;
- gated: ILSP Protipa exams, ILSP modern history QA, ILSP history
  Trapeza-Thematon CO-QA, ILSP Greek PCR.

Machine-translated ILSP tasks can stay as Krikri/Meltemi comparability
diagnostics, but should not be averaged into a native-Greek headline.
