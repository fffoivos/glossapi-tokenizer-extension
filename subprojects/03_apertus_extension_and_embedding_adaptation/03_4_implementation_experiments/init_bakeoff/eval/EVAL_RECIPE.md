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

## §5.6 hard gates — to be filled from V4 baseline

Per v0.7 §5.6, a candidate arm **fails** if any of these gates trips. Thresholds are deliberately left as placeholders here — they're set **after** the V4 baseline run on unmodified Apertus-8B-2509 establishes the per-benchmark variance. The "fill from V4" step is on the post-V4 review checklist; until then these thresholds remain `PENDING(V4)`.

| # | Hard gate | Signal source | Failure rule | Threshold |
|---|---|---|---|---|
| HG1 | English / core retention regression | `results.json` from lm-eval-harness (ARC, HellaSwag, WinoGrande, PIQA, MMLU) | mean drop > X p.p. on any one of those 5, after windowed average over last 3-5 checkpoints in 80-100 % of budget | `X = PENDING(V4)` (suggest 3 p.p. as starting placeholder) |
| HG2 | Code retention regression (if code is a release requirement per Q A1) | lm-eval-harness code task (typically `humaneval` post-training; not in pretraining-eval table) | mean drop > Y p.p. | `Y = PENDING(V4 + Q A1)`; skip if code isn't a release requirement |
| HG3 | New-token row collapse (cosine clustering, near-zero usage) | [`compute_new_token_diagnostics.py`](compute_new_token_diagnostics.py) → `embedding.new_E_cos.mean_off_diag` AND `forward.d2_avg_prob_mass_new_per_pos` | `mean_off_diag > C_THRESH` OR `prob_mass_new < M_THRESH` | `C_THRESH = PENDING(V4)` (suggest 0.5 as starting placeholder — base Greek tokens typically sit at ~0.1-0.3 off-diag mean); `M_THRESH = PENDING(V4)` |
| HG4 | Polytonic text worsens vs base on character-normalized loss | [`compute_tokenizer_fair_metrics.py`](compute_tokenizer_fair_metrics.py) → `per_register.polytonic.nll_per_char` | arm value > base value | direct comparison; no separate threshold |
| HG5 | Throughput / memory hit disproportionate to Greek compression gain | Megatron training logs (`--log-throughput`) + tokenizer-fair `compression_ratio` | `(throughput_loss / compression_gain) > T_THRESH` | `T_THRESH = PENDING(V4)`; only fires for extended-vocab arms (ReTok/Centroid), not Vanilla |
| HG6 | Language-ID drift — model over-emits Greek in non-Greek prompts | Custom eval (§6.2; not yet implemented — Item 7 in COMPLETENESS_CHECK.md) | drift rate > D_THRESH | **`D_THRESH = PENDING(V4 + custom-eval-construction)`**; DEFERRED until §6.2 custom evals exist |

**Operational note.** HG1-HG5 can be computed from JSONs already produced by the existing eval flow. HG6 requires the custom Greek eval construction work (1-2 weeks per v0.7); it's expected to not be running in time for the first bakeoff pass and may need a substitute signal (e.g., `per_register.english.nll_per_char` from `compute_tokenizer_fair_metrics.py` on an English held-out slice — non-Greek text getting Greek-token outputs would spike English NLL).

**Why these thresholds aren't hard-coded:** the per-benchmark variance on Apertus-8B-base is unknown until we run V4. A "3 p.p. drop is a failure" rule is meaningless if the run-to-run variance of HellaSwag is itself 2 p.p. The V4 baseline + bootstrap CIs establish the noise floor, then "drop > 3× the noise floor on any benchmark" or similar becomes a defensible rule.

**Selection (not automated).** For non-failing arms, v0.7 §5.6 gives a weighted score (30-40 % Greek BPC, 25-35 % Greek benchmarks, 10-15 % polytonic, 15-25 % retention, 5-10 % efficiency). For a 3-arm bakeoff this is small enough to eyeball — see [`summarize_bakeoff.py`](summarize_bakeoff.py) for the helper that aggregates per-arm JSONs into one markdown table. **The final pick is a manual review against this table + the V4 thresholds**, not a numerical aggregate.

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
- **`compute_tokenizer_fair_metrics.py`** — primary v0.7 §5.1 intrinsic metrics (BPC, NLL/char, NLL/word, tokens/word, chars/token, compression ratio, STRR). The cross-tokenizer-fair signal for comparing Vanilla (vocab 131,072) vs ReTok/Centroid (vocab 148,480). Has a `--stats-only` mode for tokenizer-only checks (no model load).
- **`run_tokenizer_fair_metrics.sbatch`** — sbatch wrapper for the above; 1 node × 1 GPU × 2 h. Runs at each bakeoff checkpoint where downstream eval also runs.
- **`compute_new_token_diagnostics.py`** — v0.7 §5.3 new-token integration diagnostic suite: 7 diagnostics over the 17,408 new IDs (rank of new target in next-token logits, prob-mass on new IDs, per-register entropy, top-1 substitution rate at new-target positions, greedy-gen new-token utilization, embedding L2-norm distribution new-vs-existing, cosine-similarity / effective-rank of new rows). Has `--embedding-only` mode that skips the forward-pass diagnostics (D1-D5) for cheap embedding-only health checks.
- **`run_new_token_diagnostics.sbatch`** — sbatch wrapper for the diagnostic suite; 1 node × 1 GPU × 2 h.

## §5.3 new-token integration diagnostic suite — important

Per v0.7 §5.3, these 7 diagnostics are **"read at every bakeoff checkpoint"**:

| # | Diagnostic | What it catches |
|---|---|---|
| D1 | Rank of correct new token in next-token logits | New token invisible |
| D2 | Aggregate probability mass on new Greek tokens | Under- or over-emitted |
| D3 | New-token entropy by register | Polytonic rows collapsed or avoided |
| D4 | Top-k substitutions between new token and old subpieces | Model still prefers old segmentation |
| D5 | Greedy-gen new-token utilization rate | New rows exist but behaviorally dead |
| D6 | Embedding L2-norm distribution: new vs existing | Degenerate-subspace collapse |
| D7 | Cosine-similarity / effective-rank of new rows | Same-direction collapse |

`compute_new_token_diagnostics.py` implements all 7 in a single run (~10-15 min per checkpoint on one GH200). D6 + D7 are embedding-only (cheap; can run with `--embedding-only` for fast checks). D5 is the heaviest single diagnostic; `--skip-greedy` opts out.

## Primary intrinsic metrics (v0.7 §5.1) — important

Per-token PPL is **not comparable** across the Vanilla arm (vocab 131,072) and the ReTok/Centroid arms (vocab 148,480). v0.7 §5.1 specifies these tokenizer-fair metrics as the **primary** signal for the bakeoff:

| Metric | Why |
|---|---|
| **BPC (bits per byte)** | Cleanest cross-tokenizer comparison |
| **NLL per Unicode character** | More interpretable for Greek/polytonic |
| **NLL per word** | Human-facing language metric |
| **tokens/word, chars/token, compression ratio** | Quantifies tokenizer efficiency |
| **STRR** (Subword-Tokenization Recovery Rate) | Fraction of held-out words that tokenize to a single token — whole-word preservation |

`compute_tokenizer_fair_metrics.py` computes all of these from one HF-format checkpoint + a held-out JSONL. Aggregates globally + per-source + per-register.

**Held-out eval slice**: ~100-500 docs covering modern Greek registers (encyclopedic / literary / academic / dialogue / legal / dictionary). The slice should be **outside** the bakeoff training mix — currently constructed manually (deterministic doc_id-hash holdout from the dedup-audit val/test partition; reconstruction path documented in [`../../03_3_cscs_experiments_kickoff/ANALYSIS.md`](../../03_3_cscs_experiments_kickoff/ANALYSIS.md) §"Review checkpoint B" given the gcloud-access loss).
