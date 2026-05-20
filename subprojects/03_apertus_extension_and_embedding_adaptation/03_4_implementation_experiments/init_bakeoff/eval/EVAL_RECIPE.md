# Eval recipe — V4 baseline + per-arm bakeoff eval

What we run, when, and what to expect out. **Engine**: [`swiss-ai/lm-evaluation-harness`](https://github.com/swiss-ai/lm-evaluation-harness) (Apertus team's fork — cited in Apertus tech report [arXiv:2509.14233](https://arxiv.org/abs/2509.14233) §5.1 footnote 45). **Citation**: Gao, Tow, Abbasi et al., Zenodo 2024, DOI [10.5281/zenodo.12608602](https://doi.org/10.5281/zenodo.12608602).

Two scopes:

- **V4 baseline** — full eval suite on unmodified `swiss-ai/Apertus-8B-2509`. One run, before the bakeoff fires. Output: per-benchmark mean + bootstrap CI. Gates the §5.6 hard-gate thresholds (how much regression on each benchmark counts as a "failure" for an arm).
- **Per-arm bakeoff eval** — same suite run on each arm's saved checkpoints in the 80–100 % range of its 2 B-token budget. Output: windowed-average score + bootstrap CI per arm. Drives §5.6 selection.

Both scopes use the **same task list and same harness flags** so the V4 baseline is the apples-to-apples comparator.

## Scope of tasks: match Apertus's pretraining-eval table, not post-training

The bakeoff compares **pretraining-stage** checkpoints (no SFT, no DPO). Apertus's tech report reports two distinct eval tables:

- **Table 14, p.38 (pretraining benchmarks)**: ARC (challenge + easy), HellaSwag, WinoGrande, PIQA, MMLU, Global-MMLU, XNLI (15 langs incl. Greek), XCOPA (11 langs). All run with harness-default shot counts per §5.1.
- **Table 22, p.44 (post-training benchmarks)**: GSM8K-CoT, HumanEval, MBPP, IFEval, BBH, BBQ, ToxiGen, HarmBench, etc. — generative / instruction-following tasks that require an SFT'd model.

Our **primary** retention metric is Table-14 tasks (Group 1 below). Table-22 tasks (GSM8K / HumanEval / MBPP / IFEval) we mark as **post-training only** and skip in the bakeoff — they shouldn't fluctuate from a CPT init choice on a non-instruction-tuned base.

## Task list

Three groups. The wrappers in this directory take `--task-group {full|greek_only|retention_only}` so you can run them independently if needed.

### Group 1: retention (Apertus pretraining-eval set per Table 14)

| Task | Capability | `lm-eval-harness` task name | Few-shot (harness default) |
|---|---|---|---|
| ARC-Challenge | reasoning | `arc_challenge` | 25 |
| ARC-Easy | reasoning | `arc_easy` | 25 |
| HellaSwag | commonsense | `hellaswag` | 10 |
| WinoGrande | coreference | `winogrande` | 5 |
| PIQA | physical commonsense | `piqa` | 0 |
| MMLU (57 subjects) | knowledge | `mmlu` | 5 |
| Global-MMLU (15 langs incl. Greek) | multilingual knowledge | `global_mmlu` | 5 |
| XNLI (15 langs incl. Greek) | multilingual NLI | `xnli` | 0 |
| XCOPA (11 langs) | multilingual commonsense | `xcopa` | 0 |

XNLI and Global-MMLU both include Greek (`xnli_el`, `global_mmlu_el`); these double as Greek-retention signals for the bakeoff. Per Apertus §5.1 footnote 45: *"All of our reported pretraining benchmarks follow the default configuration specified in lm-evaluation-harness"* — we adopt the same convention. Tasks **excluded** from the bakeoff because Apertus reports them only post-training: GSM8K, HumanEval, MBPP, IFEval, BBH (Table 22).

### Group 2: Greek (ILSP suite + Belebele Greek + native GreekMMLU)

The [ILSP Greek Evaluation Suite collection](https://huggingface.co/collections/ilsp/ilsp-greek-evaluation-suite) has **no dedicated paper** — its core 6-test-set was introduced in Voukoutis et al., *Meltemi: The first open Large Language Model for Greek* ([arXiv:2407.20743](https://arxiv.org/abs/2407.20743), 2024), and extended in the Krikri paper ([arXiv:2505.13772](https://arxiv.org/abs/2505.13772), 2025). **Apertus does not separately evaluate on ILSP tasks** — Greek shows up only as one language inside the multilingual variants (Global-MMLU, XNLI, hellaswag_multilingual). For our bakeoff, ILSP gives us **dedicated Greek signal** beyond Apertus's multilingual coverage.

| Task | Source | `lm-eval-harness` task name | Few-shot |
|---|---|---|---|
| Greek ARC | `ilsp/arc_greek` | `arc_greek` | 25 |
| Greek HellaSwag | `ilsp/hellaswag_greek` | `hellaswag_greek` | 10 |
| Greek WinoGrande | `ilsp/winogrande_greek` | `winogrande_greek` | 5 |
| Greek MMLU (ILSP, MT-adapted) | `ilsp/mmlu_greek` | `mmlu_greek` | 5 |
| Greek MMLU-Pro | `ilsp/MMLU-Pro_greek` | `mmlu_pro_greek` | 5 |
| Greek TruthfulQA | `ilsp/truthful_qa_greek` | `truthfulqa_greek` | 0 |
| Greek Medical MCQA | `ilsp/medical_mcqa_greek` | `medical_mcqa_greek` | 0 |
| Belebele Greek | `facebook/belebele`, config `ell_Grek` | `belebele_ell_Grek` | 5 |
| GreekMMLU (Zhang 2026, native) | `dascim/GreekMMLU` (if accessible) | custom task config; not yet upstream | 5 |

> **Reviewer flag**: lm-eval-harness task configs for the `*_greek` tasks above live in the **Meltemi / Krikri team's harness forks** (e.g. `LeonVouk/lighteval`, `ilsp/lm-evaluation-harness-greek`) — they have **not** landed in swiss-ai/lm-evaluation-harness or upstream EleutherAI/lm-evaluation-harness. [`pull_benchmarks.sh`](pull_benchmarks.sh) clones the swiss-ai fork as primary; the Meltemi/Krikri task YAMLs will need to be merged in at staging time. Confirm before submitting the V4 baseline.

Greek post-training-only tasks (`ilsp/mgsm_greek`, `ilsp/ifeval_greek`, `ilsp/mt-bench-greek`, `ilsp/m-ArenaHard_greek`) are out of scope for the pretraining-stage bakeoff — see "Scope of tasks" above.

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
