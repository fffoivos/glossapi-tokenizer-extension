# V4-HF baseline results — Apertus-8B-2509 on Clariden

**Job:** 2334245 (slurm normal partition, 1× H100, bf16, batch_size=auto → 32).
**Walltime:** 1h11m54s (eval completed at 11:17 UTC). Submitted from `swiss-ai/lm-evaluation-harness` clone, target-installed at `/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval`.
**Exit code 13 in `sacct` is a `ls | head` SIGPIPE false-positive** caught by `set -o pipefail`, *after* the eval saved results. Patched in the sbatch (use `ls -la "$OUTPUT_DIR" || true` instead of `| head`).

Artifacts (canonical copies in this dir):
- `results.json` — lm-eval-harness aggregated metrics (370 KB)
- `run_metadata.json` — model + task_list + uenv image + slurm id
- `stdout.log` — the full V4 stdout including the printed group/table

**Takeover correction (2026-05-21).** This run is complete for the task list in `run_metadata.json`, but that task list accidentally omitted `global_mmlu` from the Table-14 retention group. Use these artifacts as a partial V4-HF baseline until the corrected `run_eval.sbatch` rerun lands.

## Retention (Apertus pretraining-eval table — Group 1)

| Task | Metric | V4-HF | Stderr | Apertus paper (Table 14, p.38) | Note |
|---|---|---:|---:|---:|---|
| `mmlu` (57 subj) | acc | 0.5923 | 0.0039 | reported as ~0.59 | matches |
| ↳ humanities | acc | 0.5362 | 0.0067 | — | |
| ↳ other | acc | 0.6714 | 0.0081 | — | |
| ↳ social sciences | acc | 0.7026 | 0.0080 | — | |
| ↳ stem | acc | 0.4903 | 0.0085 | — | |
| `xnli` (15 langs avg) | acc | 0.4400 | 0.0026 | — | NLI is hard for base model; ~chance is 0.33 |
| `xcopa` (11 langs avg) | acc | 0.6573 | 0.0063 | — | comfortably above chance (0.5) |
| `arc_challenge` | acc | _see samples_ | _see samples_ | — | per-sample logged |
| `arc_easy` | acc | _see samples_ | _see samples_ | — | per-sample logged |
| `hellaswag` | acc | _see samples_ | _see samples_ | — | per-sample logged |
| `winogrande` | acc | _see samples_ | _see samples_ | — | per-sample logged |
| `piqa` | acc | _see samples_ | _see samples_ | — | per-sample logged |

(For arc/hellaswag/winogrande/piqa, the table cell aggregate is in `results.json`; we'll extract them into this table when filling §5.6 thresholds. The per-sample `samples_<task>.jsonl` files are the ground truth for bootstrap CIs.)

## Greek slice — Group 2

| Task | Metric | V4-HF | Stderr | Note |
|---|---|---:|---:|---|
| `belebele_ell_Grek` | acc | _see results.json_ | _see results.json_ | reading-comp Greek |
| `global_mmlu_full_el` | acc | **0.5155** | 0.0040 | full-Greek MMLU |
| ↳ humanities | acc | 0.4748 | 0.0069 | |
| ↳ other | acc | 0.5797 | 0.0087 | |
| ↳ social sciences | acc | 0.5951 | 0.0087 | |
| ↳ stem | acc | 0.4355 | 0.0086 | |
| `include_base_44_greek_few_shot_en` | acc | **0.5054** | 0.0208 | INCLUDE-44 group (7 subjects) |
| `xnli_el` | acc | **0.3984** | 0.0098 | Greek NLI, near chance |
| `xquad_el` | exact_match | **0.2874** | 0.0131 | Greek extractive QA |
| `xquad_el` | f1 | 0.5172 | 0.0117 | Greek extractive QA |
| `arc_challenge_mt_el` | — | _see samples_ | — | per-sample logged |
| `global_piqa_completions_ell_grek` | — | _see samples_ | — | per-sample logged |

**Reading:** Apertus-8B is a strong English/multilingual base; Greek-specific scores (~0.50 on global_mmlu_full_el, ~0.50 on INCLUDE-44, ~0.40 on xnli_el) confirm the corpus has *meaningful* Greek capability but with room to grow — which is the point of the bakeoff's continued pretraining.

## What this unblocks

- **§5.6 hard-gate thresholds (HG1).** Per the EVAL_RECIPE.md plan, HG1 ("English / core retention regression > X p.p.") was `PENDING(V4)`. We can now fill X using bootstrap CIs over the per-sample jsonls (run `compute_bootstrap_cis.py samples_*.jsonl`). Suggested: HG1 trips at > 3 × (per-task bootstrap stderr) drop. Concrete numbers in a follow-up issue.
- **V4-post-conversion run.** Per the two-V4-run plan, we also need V4 evaluated AFTER going through HF → Megatron → HF. R1 PASS-ed; the conversion artifact exists at `/capstor/scratch/cscs/fffoivos/runs/r1_roundtrip_2333864/apertus_hf_roundtrip`. Re-running run_eval.sbatch against that path produces the V4-post-conversion baseline. Schedule with `MODEL_PATH=/capstor/scratch/cscs/fffoivos/runs/r1_roundtrip_2333864/apertus_hf_roundtrip`.

## Reproducing

```bash
ssh clariden 'cd /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval && \
  APERTUS_BASE=/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509 \
  OUT_ROOT=/capstor/scratch/cscs/fffoivos/runs/eval \
  bash run_apertus_baseline.sh'
```

The sbatch (run_eval.sbatch) is canonical; it carries the `Compute justification` block. uenv `pytorch/v2.9.1:v2`; task list = retention (8) + greek (7) = 15 task labels expanding to ~80 sub-task records in `results.json`.
