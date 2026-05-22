# Eval recipe — V4 baseline + per-arm bakeoff eval

What we run, when, and what to expect out. **Engine**: [`swiss-ai/lm-evaluation-harness`](https://github.com/swiss-ai/lm-evaluation-harness) (Apertus team's fork — cited in Apertus tech report [arXiv:2509.14233](https://arxiv.org/abs/2509.14233) §5.1 footnote 45). **Citation**: Gao, Tow, Abbasi et al., Zenodo 2024, DOI [10.5281/zenodo.12608602](https://doi.org/10.5281/zenodo.12608602).

Two scopes:

- **V4 baseline** — full eval suite on unmodified `swiss-ai/Apertus-8B-2509`. One run, before the bakeoff fires. Output: per-benchmark mean + bootstrap CI. Gates the §5.6 hard-gate thresholds (how much regression on each benchmark counts as a "failure" for an arm).
- **Per-arm bakeoff eval** — same suite run on each arm's saved checkpoints in the 80–100 % range of its 2 B-token budget. Output: windowed-average score + bootstrap CI per arm. Drives §5.6 selection.

Both scopes use the **same task list and same harness flags** so the V4 baseline is the apples-to-apples comparator.

> **2026-05-21 takeover note.** V4 job `2334245` completed the then-current script, but that script accidentally omitted `global_mmlu` from Group 1. Treat its copied artifacts as a valid partial baseline for the listed tasks, not as the final Table-14 baseline. `run_eval.sbatch` now includes `global_mmlu`; rerun V4-HF and V4-post-conversion before filling final §5.6 thresholds.

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

### Group 2: Greek (native swissai-harness tasks + Belebele Greek)

**2026-05-21 update — empirically confirmed task availability.** The previous version of this table listed `arc_greek` / `hellaswag_greek` / `winogrande_greek` / `mmlu_greek` / `mmlu_pro_greek` / `truthfulqa_greek` / `medical_mcqa_greek` as `lm-eval-harness` task names. **Verified against the `swiss-ai/lm-evaluation-harness` clone on Clariden (2026-05-21, V4 attempt jobs 2333668 / 2333723):** those task names do not exist in the swissai harness — they live in the Meltemi/Krikri team's forks (`LeonVouk/lighteval`, `ilsp/lm-evaluation-harness-greek`) and have not landed upstream. The "reviewer flag" below was correct.

**What we actually ship at V4** (the names below resolve in `swiss-ai/lm-evaluation-harness` and were used for V4 job 2334245):

| Task | Source | `lm-eval-harness` task name | Few-shot |
|---|---|---|---|
| Greek ARC (machine-translated) | swissai harness | `arc_challenge_mt_el` | 25 |
| Greek MMLU (Global-MMLU, full Greek slice) | swissai harness | `global_mmlu_full_el` | 5 |
| Greek MMLU-44 (INCLUDE-44, native Greek, 7 subjects) | `CohereForAI/include-base-44`, config `Greek` | `include_base_44_greek_few_shot_en` (group of 7 subtasks) | 5 |
| Greek NLI | swissai harness | `xnli_el` | 0 |
| Greek QA | swissai harness | `xquad_el` | 0 |
| Greek PIQA (machine-translated) | swissai harness | `global_piqa_completions_ell_grek` | 0 |
| Belebele Greek | `facebook/belebele`, config `ell_Grek` | `belebele_ell_Grek` | 5 |

> **Reviewer flag (still open) — PF5 ILSP YAMLs.** The ILSP tasks the previous table claimed (`hellaswag_greek`, `winogrande_greek`, `mmlu_pro_greek`, `truthfulqa_greek`, `medical_mcqa_greek`) genuinely add Greek signal we don't currently have. The PF5 follow-up is to **port the YAML configs from `LeonVouk/lighteval` (or `ilsp/lm-evaluation-harness-greek`) into the swissai harness clone** so they resolve. Tracked as Task #55 in our running task list. Until PF5 lands, the bakeoff runs with the seven-task Greek list above + the multilingual coverage from Group 1.

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

### Current 2026-05-22 operational cadence

Training writes Megatron `torch_dist` checkpoints, while lm-eval and the intrinsic metric jobs require HF-format model directories. Each evaluated checkpoint therefore has two stages:

1. Convert the Megatron checkpoint to HF with [`convert_bakeoff_checkpoint_to_hf.sbatch`](convert_bakeoff_checkpoint_to_hf.sbatch).
2. Run [`run_eval.sbatch`](run_eval.sbatch) and, when a held-out JSONL is available, [`run_tokenizer_fair_metrics.sbatch`](run_tokenizer_fair_metrics.sbatch) + [`run_new_token_diagnostics.sbatch`](run_new_token_diagnostics.sbatch).

Use [`submit_bakeoff_checkpoint_eval.sh`](submit_bakeoff_checkpoint_eval.sh) for the conversion/eval chain:

```bash
RUN_TAG=bakeoff_1node_chain_20260522_005620 \
  bash submit_bakeoff_checkpoint_eval.sh vanilla 65 greek_only
```

For full bakeoff checkpoints where all three arms should be evaluated, prefer
the packed path. Clariden's normal partition can allocate/bill a whole 4-GPU
node even for a one-GPU `lm-eval` job; the packed submitter converts each arm
separately, then runs the three single-GPU full evals concurrently inside one
node allocation:

```bash
RUN_TAG=bakeoff_1node_chain_20260522_005620 \
SUBMIT_INTRINSIC=1 \
EVAL_JSONL=/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl \
  bash submit_bakeoff_checkpoint_eval_packed.sh 390 full vanilla retok centroid
```

For a live run, use [`watch_and_submit_checkpoint_evals.sh`](watch_and_submit_checkpoint_evals.sh)
to submit once per arm as soon as each checkpoint appears:

```bash
RUN_TAG=bakeoff_1node_chain_20260522_005620 \
ITER=65 \
TASK_GROUP=greek_only \
POLL_SECONDS=300 \
nohup bash watch_and_submit_checkpoint_evals.sh \
  > /capstor/scratch/cscs/fffoivos/runs/eval/watch_iter65.log 2>&1 &
```

For full checkpoints in the live bakeoff, use
[`watch_and_submit_checkpoint_evals_packed.sh`](watch_and_submit_checkpoint_evals_packed.sh)
instead, so the watcher waits until all requested arms are complete and then
submits a single packed full-eval job:

```bash
RUN_TAG=bakeoff_1node_chain_20260522_005620 \
ITER=390 \
TASK_GROUP=full \
SUBMIT_INTRINSIC=1 \
EVAL_JSONL=/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl \
POLL_SECONDS=300 \
nohup bash watch_and_submit_checkpoint_evals_packed.sh \
  > /capstor/scratch/cscs/fffoivos/runs/eval/watch_iter390_packed.log 2>&1 &
```

The live bakeoff saves every 65 iterations, which is about 273 M tokens (`65 × 1024 × 4096`). The practical cadence is:

| Iteration | Tokens/arm | Eval action |
|---:|---:|---|
| 65 | ~273 M | Greek-only downstream smoke on all three arms; proves checkpoint save, conversion, and eval before waiting for late checkpoints. |
| 130 | ~545 M | Full downstream eval + intrinsic metrics/diagnostics if held-out JSONL is staged. |
| 260 | ~1.09 B | Full downstream eval + intrinsic metrics/diagnostics. |
| 390 | ~1.64 B | Full downstream eval + intrinsic metrics/diagnostics; enters the selection window. |
| 455 or final | ~1.91-2.00 B | Full downstream eval + intrinsic metrics/diagnostics; final selection evidence. |

The final arm choice should use the late-window checkpoints (390 plus the final one, and 260 as the nearest pre-window anchor if only two late checkpoints are available). The iteration-65 Greek-only run is not selection evidence; it is an operational canary to catch broken conversion/eval early.

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
- shape: `-N 1 -t 4:00:00 --gpus-per-node=1` (V4 currently runs 1-GPU; the early 4-GPU plan was scaled back after PF5 / task-name corrections cut total work)
- wall: ~1-2 h end-to-end for the seven Greek tasks + nine retention tasks at the corrected list (V4 job 2334245 mid-run as of 2026-05-21 12:30 UTC)
- batch size: `auto` (lm-eval-harness picks per task; observed `auto:1` → `32` for Apertus-8B on bf16)

### lm-eval-harness install path (Clariden, 2026-05-21)

The uenv `pytorch/v2.9.1:v2`'s site-packages is read-only, so `pip install -e .` from the swissai clone fails with `[Errno 30] Read-only file system`. Workaround used by `run_eval.sbatch`:

```bash
# One-time, as part of CSCS staging:
uenv start pytorch/v2.9.1:v2 --view=default
cd /iopsstor/scratch/cscs/fffoivos/code/eval/lm-evaluation-harness-swissai
pip install --target=/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval --quiet .
pip install --target=/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval --no-deps --quiet accelerate
# Delete the target's copy of huggingface-hub so the uenv's 0.36.0 wins over the
# pip-installed 1.x (transformers 4.57.0 pins hf_hub<1.0):
rm -rf /iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval/huggingface_hub
rm -rf /iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval/huggingface_hub-*.dist-info
```

Then run_eval.sbatch sets `PYTHONPATH=/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval` and invokes `python3 -m lm_eval ...`. No `pip install -e .` at job-start time.

Per-arm bakeoff eval at one checkpoint:

- same shape, ~2 h
- run inline during training via Slurm `--dependency=afterok` on the checkpoint-save step, OR run separately after the training job completes

## Files in this directory

- `EVAL_RECIPE.md` — this doc
- `pull_benchmarks.sh` — staging-time HF download for the benchmark datasets (and the swiss-ai eval scripts repo)
- `run_eval.sbatch` — parameterized eval job: `MODEL_PATH=… OUTPUT_DIR=… TASK_GROUP={full|greek_only|retention_only} sbatch run_eval.sbatch`
- `run_apertus_baseline.sh` — thin wrapper: V4 baseline on the unmodified Apertus-8B-2509 checkpoint
- `run_bakeoff_arm_eval.sh` — thin wrapper: per-arm eval, takes an arm's checkpoint dir as arg
- `convert_bakeoff_checkpoint_to_hf.sbatch` — CPU-only `xfer` job; converts one Megatron `torch_dist` bakeoff checkpoint to HF format for eval
- `run_megatron_convert_with_pg.py` — initializes the single-rank process group needed by Megatron `loader core` when reading `torch_dist` checkpoints
- `submit_bakeoff_checkpoint_eval.sh` — submits conversion plus lm-eval, with optional intrinsic metrics when `SUBMIT_INTRINSIC=1`
- `watch_and_submit_checkpoint_evals.sh` — lightweight watcher that stamps per-arm submissions and prevents duplicate checkpoint eval launches
- `build_cpt_heldout_jsonl.py` / `build_cpt_heldout_jsonl.sbatch` — builds the 500-doc Greek held-out JSONL from the post-Apertus-dedup selected pool while excluding Greek doc_ids already used in `bulk_mix.jsonl`
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

**Held-out eval slice**: ~100-500 docs covering modern Greek registers (encyclopedic / literary / academic / dialogue / legal / dictionary). The slice should be **outside** the bakeoff training mix. Current operational builder:

```bash
sbatch build_cpt_heldout_jsonl.sbatch
```

This samples from `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet`, excludes every Greek `doc_id` already present in `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl`, and fills quotas from the Greek source/register buckets in `corpus_build/recipes/bulk.json` that still have training-disjoint rows. The first full scan showed the current training mix exhausted the literary and dictionary/misc filters, so the default 500-doc slice redistributes those slots to HPLT, dialogue/textbooks, academic, and legal/civic. This is more aligned with the current CPT corpus than the older C3 val/test reconstruction, but it is a **training-disjoint intrinsic-metrics slice**, not a proof that all external benchmark test items were absent from the CPT source pool. Benchmark decontamination remains a separate check.
