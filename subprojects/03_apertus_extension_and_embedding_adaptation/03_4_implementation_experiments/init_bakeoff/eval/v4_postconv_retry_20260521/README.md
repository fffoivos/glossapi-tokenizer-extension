# V4 Post-Conversion Retry Results - 2026-05-21

This folder keeps the compact result JSONs copied locally from the Clariden
post-conversion V4 retry. Large per-sample JSONL files remain on Clariden.

Remote source directories:

- Retention: `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_postconv_v4_retention_retry_20260521_163240`
- Greek: `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_postconv_v4_greek_retry_20260521_163240`

Slurm:

- `2338020` retention-only: `COMPLETED 0:0`, elapsed `00:25:55`
- `2338021` Greek-only: `COMPLETED 0:0`, elapsed `03:39:23`

Local files:

- `retention_results.json`
- `greek_results.json`
- `retention_run_metadata.json`
- `greek_run_metadata.json`

Greek headline metrics from `greek_results.json`:

| Task | Metric | Value |
|---|---:|---:|
| `arc_challenge_mt_el` | `acc_norm` | `0.2636518771` |
| `arc_challenge_mt_el` | `acc` | `0.2141638225` |
| `belebele_ell_Grek` | `acc` | `0.2288888889` |
| `global_mmlu_full_el` | `acc` | `0.2294544937` |
| `global_piqa_completions_ell_grek` | `acc_norm` | `0.54` |
| `global_piqa_completions_ell_grek` | `acc` | `0.42` |
| `include_base_44_greek_few_shot_en` | `acc` | `0.1974637681` |
| `xnli_el` | `acc` | `0.3333333333` |
| `xquad_el` | `exact_match` | `0.0` |
| `xquad_el` | `f1` | `0.0` |

Retention headline metrics from `retention_results.json`:

| Task | Metric | Value |
|---|---:|---:|
| `arc_challenge` | `acc_norm` | `0.2619453925` |
| `arc_easy` | `acc_norm` | `0.2613636364` |
| `global_mmlu` | `acc` | `0.2381290253` |
| `hellaswag` | `acc_norm` | `0.2674765983` |
| `piqa` | `acc_norm` | `0.5212187160` |
| `winogrande` | `acc` | `0.5106550908` |
| `xcopa` | `acc` | `0.5185454545` |
| `xnli` | `acc` | `0.3321017403` |
