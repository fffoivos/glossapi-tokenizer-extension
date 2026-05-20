# Eval recipe — V4 baseline + per-arm bakeoff eval

What we run, when, and what to expect out.

Two scopes:

- **V4 baseline** — full eval suite on unmodified `swiss-ai/Apertus-8B-2509`. One run, before the bakeoff fires. Output: per-benchmark mean + bootstrap CI. Gates the §5.6 hard-gate thresholds (how much regression on each benchmark counts as a "failure" for an arm).
- **Per-arm bakeoff eval** — same suite run on each arm's saved checkpoints in the 80–100 % range of its 2 B-token budget. Output: windowed-average score + bootstrap CI per arm. Drives §5.6 selection.

Both scopes use the **same task list and same harness flags** so the V4 baseline is the apples-to-apples comparator.

## Task list

Three groups. The wrappers in this directory take `--task-group {full|greek_only|retention_only}` so you can run them independently if needed.

### Group 1: retention (non-Greek, the Apertus-reported set)

| Task | Capability | `lm-eval-harness` task name | Few-shot |
|---|---|---|---|
| ARC-Challenge | reasoning | `arc_challenge` | 25 |
| ARC-Easy | reasoning | `arc_easy` | 25 |
| HellaSwag | commonsense | `hellaswag` | 10 |
| WinoGrande | coreference | `winogrande` | 5 |
| PIQA | physical commonsense | `piqa` | 0 |
| MMLU (57 subjects) | knowledge | `mmlu` | 5 |
| HumanEval | code | `humaneval` | 0 |
| GSM8K | math | `gsm8k` | 5 |
| XNLI (15 langs) | multilingual NLI | `xnli` (all 15 languages) | 0 |
| XCOPA (11 langs) | multilingual commonsense | `xcopa` (all 11 languages) | 0 |

XNLI includes Greek (`xnli_el`). For non-Greek regression specifically, we look at `xnli_en`, `xnli_fr`, `xnli_de`, `xnli_ru`.

### Group 2: Greek (the ILSP suite + GreekMMLU + Belebele Greek)

| Task | Source | `lm-eval-harness` task name | Few-shot |
|---|---|---|---|
| Greek ARC | `ilsp/arc_greek` | `arc_greek` | 25 |
| Greek HellaSwag | `ilsp/hellaswag_greek` | `hellaswag_greek` | 10 |
| Greek WinoGrande | `ilsp/winogrande_greek` | `winogrande_greek` | 5 |
| Greek MMLU (ILSP, MT-adapted) | `ilsp/mmlu_greek` | `mmlu_greek` | 5 |
| Greek MMLU-Pro | `ilsp/MMLU-Pro_greek` | `mmlu_pro_greek` | 5 |
| Greek MGSM | `ilsp/mgsm_greek` | `mgsm_greek` | 5 |
| Greek TruthfulQA | `ilsp/truthful_qa_greek` | `truthfulqa_greek` | 0 |
| Greek Medical MCQA | `ilsp/medical_mcqa_greek` | `medical_mcqa_greek` | 0 |
| Greek IFEval | `ilsp/ifeval_greek` | `ifeval_greek` | 0 |
| Belebele Greek | `facebook/belebele`, config `ell_Grek` | `belebele_ell_Grek` | 5 |
| GreekMMLU (Zhang 2026, native) | `dascim/GreekMMLU` (if accessible) | custom task config; not yet upstream | 5 |
| AttaQ Greek (safety) | `ilsp/attaq_greek` | as separate inspect-eval | n/a |

The custom Greek evals from v0.7 §6.2 — polytonic continuation (out of bakeoff scope per the modern-only directive), morphology minimal pairs, language-ID drift, register preservation — run via Inspect AI as a separate sbatch (`run_inspect_evals.sbatch`, not in this iteration).

### Group 3: safety / other

| Task | Source | Notes |
|---|---|---|
| HarmBench | `swiss-ai/harmbench` | Apertus team's curated harmful-behavior bench |
| PolyglotToxicityPrompts | `swiss-ai/polyglotoxicityprompts` | Multilingual toxicity (incl. Greek) |
| AttaQ Greek | `ilsp/attaq_greek` | Greek-targeted safety |
| Jailbreak-StrongReject Greek | `ilsp/Jailbreak-StrongReject-el` | Jailbreak resistance |

These are inspect-style evals, run via `inspect eval` rather than lm-eval-harness. Lower priority for the bakeoff (no §5.6 hard gate ties to them directly); useful for the eventual release.

## Eval cadence

Per v0.7 §6.1:

- **Trajectory metrics** (training loss, per-bucket PPL, BPC trajectory, §5.3 new-token diagnostic suite): every 100 M tokens in the bakeoff (every ~25 global steps at 4 M tokens/step).
- **Downstream benchmarks** (Group 1 + Group 2): every 500 M tokens during training, plus a full sweep at the last 3–5 checkpoints in the 80–100 % budget range for selection.
- **V4 baseline**: once, before the bakeoff. Full suite × 1 run. Bootstrap CIs over eval samples (not over runs — most benchmark items are deterministic).

For 2 B-token bakeoff per arm: that's 4 mid-training benchmark runs (at 500 M, 1.0 B, 1.5 B, 2.0 B) and the last 3 (1.0 / 1.5 / 2.0 B) feed the windowed selection score.

## Statistical methodology

Per v0.7 §6.1 (Park et al. Oct 2025): downstream benchmark scores are noisy, and "run 3×" doesn't add information because the benchmarks are deterministic. The right way to get confidence intervals is **bootstrap over eval samples**.

`compute_bootstrap_cis.py` (this dir) takes `--log_samples` JSONL from lm-eval-harness, resamples eval items with replacement 1000 times, and reports per-task mean + 95 % CI.

**Selection scope** (per v0.7 §5.6): windowed average across the last 3–5 checkpoints in the 80–100 % range of each arm's budget. For training continuation post-winner, use the **raw** checkpoint at the end (not the averaged one), since the averaged model has no corresponding optimizer state.

## Resources per eval run

V4 baseline (one run) on Clariden:

- partition: `normal`
- shape: `-N 1 -t 4:00:00 --gres=gpu:4`
- wall: ~3-4 h end-to-end for the full Group 1 + Group 2 sweep on a 1-node 4-GPU allocation
- batch size: `auto` (lm-eval-harness picks per task)

Per-arm bakeoff eval at one checkpoint:

- same shape, ~2 h
- run inline during training via Slurm `--dependency=afterok` on the checkpoint-save step, OR run separately after the training job completes

## Files in this directory

- `EVAL_RECIPE.md` — this doc
- `pull_benchmarks.sh` — staging-time HF download for the benchmark datasets (and the swiss-ai eval scripts repo)
- `run_eval.sbatch` — parameterized eval job: `MODEL_PATH=… OUTPUT_DIR=… TASK_GROUP={full|greek_only|retention_only} sbatch run_eval.sbatch`
- `run_apertus_baseline.sh` — thin wrapper: V4 baseline on the unmodified Apertus-8B-2509 checkpoint
- `run_bakeoff_arm_eval.sh` — thin wrapper: per-arm eval, takes an arm's checkpoint dir as arg
- `compute_bootstrap_cis.py` — post-process: bootstrap CIs over the `--log_samples` outputs
