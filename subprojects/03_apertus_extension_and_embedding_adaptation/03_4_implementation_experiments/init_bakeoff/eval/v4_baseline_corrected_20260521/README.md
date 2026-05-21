# Corrected V4-HF Baseline

This directory contains the small local artifact copy for the corrected
Apertus-8B-2509 V4-HF benchmark run on Clariden.

- Slurm job: `2335100`
- State: `COMPLETED`, exit `0:0`
- Runtime: `01:10:29`
- Model: `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509`
- Remote result directory: `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_baseline_v4_corrected_20260521_121639`
- Corrected task list: includes `global_mmlu`, which the earlier `v4_baseline_20260521` artifact omitted.

Local files:

- `results.json`: lm-eval aggregated metrics.
- `run_metadata.json`: model, task list, uenv, and Slurm metadata.
- `stdout.log`: run stdout with the final harness table.
- `sacct.txt`: Slurm accounting record.

Key headline metrics from `results.json`:

| Task | Metric | Value | Stderr |
|---|---:|---:|---:|
| `arc_challenge` | acc_norm | 0.5870 | 0.0144 |
| `arc_easy` | acc_norm | 0.8363 | 0.0076 |
| `hellaswag` | acc_norm | 0.7884 | 0.0041 |
| `winogrande` | acc | 0.6930 | 0.0130 |
| `piqa` | acc_norm | 0.7992 | 0.0093 |
| `mmlu` | acc | 0.5923 | 0.0039 |
| `global_mmlu` | acc | 0.5246 | 0.0063 |
| `xnli` | acc | 0.4400 | 0.0026 |
| `xcopa` | acc | 0.6575 | 0.0063 |
| `arc_challenge_mt_el` | acc_norm | 0.4795 | 0.0146 |
| `xnli_el` | acc | 0.3984 | 0.0098 |
| `xquad_el` | exact_match | 0.2874 | 0.0131 |
| `xquad_el` | f1 | 0.5172 | 0.0117 |
| `belebele_ell_Grek` | acc | 0.6367 | 0.0160 |
| `global_mmlu_full_el` | acc | 0.5155 | 0.0040 |
| `include_base_44_greek_few_shot_en` | acc | 0.5054 | 0.0208 |
| `global_piqa_completions_ell_grek` | acc_norm | 0.6200 | 0.0488 |

Large per-task sample JSONLs remain on Clariden in the remote result directory.
