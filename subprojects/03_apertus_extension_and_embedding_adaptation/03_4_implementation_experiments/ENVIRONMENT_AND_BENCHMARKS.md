# Experiment Environment, Benchmarks, and CSCS Deployment

> **v0.7 supersedes prior framing.** Three propagations to flag inline:
>
> - **Init arms**: v0.7 §5 = Vanilla / ReTok / **Centroid** (Distillation bracketed in v0.7 §13). The §1.3 "Init-method reference code" entry for `konstantinjdobler/token-distillation` is **no longer required for the bakeoff** — keep on radar only if v0.7's revisit conditions trigger (ReTok-vs-Centroid bakeoff inconclusive). Centroid is closed-form (per-script-centroid + Gaussian noise per v0.7 §5); no separate code repo required.
> - **Training framework**: v0.7 §7.1 + user directive of 2026-05-20 = **Megatron-LM-Swiss-AI is the canonical trunk** (`swiss-ai/Megatron-LM` + `swiss-ai/pretrain-code`). The §1.1 framing that lists `swiss-ai/apertus-finetuning-recipes` as a "most-likely trunk" reflects v0.6-era thinking; it is now demoted to "TRL + Accelerate alternative for smaller experiments" per v0.7 §7.1 wording.
> - **Tokenizer scope**: extended ship bundle (vocab 153,600, modern + polytonic). The §3.1 `tokenizers/` staging target is the **composite bundle**, not the modern-only one — both bundles are mirrored, but the composite is the active CPT base.

*Drafted 2026-05-20. Covers the code repos and dataset locations we
pull for: (a) training the three init arms, (b) evaluating them, and
(c) what Apertus itself reports so we can match its preservation
axes. Plus the concrete plan for getting all of it on Clariden
storage before the first sbatch fires.*

> **Important: the swiss-ai team has published a substantial open
> infrastructure for Apertus training + eval.** Many of the questions
> we had open ("which harness?", "what does Apertus report?", "are
> there Apertus-aligned Greek benchmarks?") have direct answers in
> the org's GitHub + ILSP's HF account. The plan below adopts these
> first-class instead of reinventing.

## 1. Code repos we need

### 1.1 Training infrastructure (pick one trunk)

| Repo | URL | Use for | Notes |
|---|---|---|---|
| `swiss-ai/pretrain-code` | github.com/swiss-ai/pretrain-code | **Apertus's own pretraining codebase, Megatron-LM-based** | Closest match to Apertus training-recipe details (AdEMAMix, WSD, 0.1 grad-clip, xIELU, QK-Norm). Highest setup cost. |
| `swiss-ai/apertus-finetuning-recipes` | github.com/swiss-ai/apertus-finetuning-recipes | **Most-likely trunk for our CPT runs** | Updated 2026-05-15 (active); aligns with what `p-skarvelis` already uses (HF Trainer + ApertusForCausalLM). Lowest setup cost. |
| `swiss-ai/Megatron-LM` | github.com/swiss-ai/Megatron-LM | Apertus's Megatron fork | The under-recipe used by `pretrain-code`. Active (updated 2026-05-18). |
| `swiss-ai/nanotron` / `nanotron-multilingual` | github.com/swiss-ai/nanotron(-multilingual) | Alternative HF-style 3D-parallel trainer | Smaller surface than Megatron-LM; less close to Apertus pretraining recipe. |
| `swiss-ai/model-launch` | github.com/swiss-ai/model-launch | **CLI app for launching LLM/VLM models on Alps** | Updated 2026-05-20 (today). Likely simplifies sbatch authoring for us. **Look at this first.** |
| `swiss-ai/command-line-interface` | github.com/swiss-ai/command-line-interface | CLIs for the Swiss AI Research Platform (CSCS) | Sibling to `model-launch`. |
| `swiss-ai/perf-check` | github.com/swiss-ai/perf-check | Canary suite — verify GPU compute, HBM BW, NVLink/PCIe, NCCL, MPI before large runs | Run this once on a 1-node allocation before our first real CPT job, to confirm the stack isn't degraded. |
| `swiss-ai/gh200-wheels` | github.com/swiss-ai/gh200-wheels | Python wheels + images for GH200 | Pull from here when uenv image misses something. |

**My read on trunk:** start with `swiss-ai/apertus-finetuning-recipes` for the pilot (matches p-skarvelis's existing pipeline; cheap to adopt), keep `pretrain-code` + Megatron-LM warm for the post-winner main CPT if we want recipe-fidelity. See [`STORAGE_AND_EXISTING_WORK.md` § 3.4](STORAGE_AND_EXISTING_WORK.md#34-implications-for-our-plan) for the reasoning.

### 1.2 Eval harnesses

| Repo | URL | Use for | Notes |
|---|---|---|---|
| `swiss-ai/lm-evaluation-harness` | github.com/swiss-ai/lm-evaluation-harness | **Apertus team's fork of EleutherAI's lm-eval-harness** | Updated 2026-05-18 (active). This is the canonical eval framework for matching Apertus's reported numbers. |
| `swiss-ai/lm-evaluation-harness-wmt` | github.com/swiss-ai/lm-evaluation-harness-wmt | WMT-specific fork | Translation-specific; use only if we need WMT numbers. |
| `swiss-ai/lighteval` | github.com/swiss-ai/lighteval | Fork of HF lighteval (multi-backend evaluator) | Alternative front-end; same backend as lm-eval-harness. |
| `swiss-ai/olmes` | github.com/swiss-ai/olmes | Fork of Allen AI's OLMES (reproducible, flexible evals) | Third option; not currently planned. |
| `swiss-ai/evals` | github.com/swiss-ai/evals | **"Apertus evals" (their own scripts)** | Look first for the exact eval invocations the team uses. |
| `swiss-ai/evals-post-train` | github.com/swiss-ai/evals-post-train | "Updated Version of Apertus Evals for Apertus 1.5 Post-Training" | Refresh of `evals`. |
| `swiss-ai/code-eval` | github.com/swiss-ai/code-eval | Code benchmarks | If we care about preserving HumanEval / MBPP. |
| `EleutherAI/lm-evaluation-harness` | github.com/EleutherAI/lm-evaluation-harness | Upstream lm-eval-harness | Use upstream for ILSP Greek benchmarks (the ILSP team contributes tasks here). |

**My read on eval:** clone `swiss-ai/evals` + `swiss-ai/lm-evaluation-harness`. The first gives us the literal invocations Apertus used; the second is the harness backing them.

### 1.3 Init-method reference code (for ReTok + Distillation arms)

| Repo | URL | Use for | Notes |
|---|---|---|---|
| `konstantinjdobler/token-distillation` | github.com/konstantinjdobler/token-distillation | **Token Distillation reference impl (Dobler 2025, arXiv:2505.20133)** | Plan §5 Experiment 3 cites this. 8 stars, Python. Updated 2026-05-16. We port the E-side path + add pattern (2) NTP-only on U rows. |
| `EEVE-Korean` / `yanolja` family | (404 on the URLs I tried) | ReTok-style reference (subword-mean init) | Not strictly needed — ReTok is ~50 lines per plan. We'll write from scratch following plan §5 Experiment 2 spec, with norm-matching targets from Phase A §8.2. |

### 1.4 Supporting (read-but-don't-fork)

- `swiss-ai/parity-aware-bpe` — "Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness" (ACL 2026). Background reading on Mistral-Nemo tokenizer fairness reasoning; not on our critical path.
- `swiss-ai/Information-Parity` — multilingual eval metric implementation. Useful if we want to publish a fairness number.
- `swiss-ai/tokenizer-intrinsic-evals` — intrinsic tokenizer evals from the Apertus tokenization team. Sibling of our local `02_1_7_intrinsic_eval_sweep/` work.
- `swiss-ai/apertus-tech-report` — the Apertus paper (arXiv:2509.14233) repo. §5 is the canonical "what Apertus reports."
- `swiss-ai/apertus-memorization` — reproduces memorization analysis. Useful for our compliance posture.
- `swiss-ai/vocab-reduction` — inference + training optimization via vocab reduction. **Opposite direction from our extension work**; worth reading the README to know what they consider failure modes.

## 2. Benchmarks — what Apertus reports + the ILSP Greek suite + extras

### 2.1 Apertus's reported eval set (from the HF model card)

The Apertus-8B-2509 model card lists these benchmarks for "general
language understanding" pretraining eval. **These are the baseline we
must not regress on for non-Greek.**

| Benchmark | Languages reported | Role for us |
|---|---|---|
| **ARC** (challenge + easy) | English | Non-Greek regression — preserve baseline |
| **HellaSwag** | English | Non-Greek regression — preserve baseline |
| **WinoGrande** | English | Non-Greek regression — preserve baseline |
| **XNLI** | en, fr, de, es, ar, zh, ru, **el**, vi, th, ko, hi, tr, sw, ur | **The primary multilingual regression axis (incl. Greek)** |
| **XCOPA** | et, ht, id, it, qu, sw, ta, th, tr, vi, zh | Commonsense, multilingual |
| **PIQA** | English | Physical commonsense, English baseline |

The model card explicitly says: *"Many additional benchmark evaluations, for pretraining and posttraining phases, multilingual evaluations in around hundred languages, and long context evaluations are provided in Section 5 of the [Apertus Tech Report](https://arxiv.org/abs/2509.14233)."* Pulling §5 will widen this list significantly; for the *pilot* runs the 6 above are enough.

### 2.2 ILSP Greek Evaluation Suite (the goldmine)

`huggingface.co/ilsp` has the complete Greek-adapted version of every benchmark Apertus reports plus much more. **All open (no gating observed).**

| HF dataset id | What it is |
|---|---|
| **`ilsp/mmlu_greek`** | Greek MMLU (the ILSP-published version — different release from Zhang 2026 native-sourced GreekMMLU; this is MT-adapted from English MMLU) |
| `ilsp/MMLU-Pro_greek` | Greek MMLU-Pro |
| **`ilsp/hellaswag_greek`** | Greek HellaSwag |
| **`ilsp/arc_greek`** | Greek ARC |
| **`ilsp/winogrande_greek`** | Greek WinoGrande |
| **`ilsp/truthful_qa_greek`** | Greek TruthfulQA |
| **`ilsp/mgsm_greek`** | Greek GSM8K (math word problems) |
| `ilsp/medical_mcqa_greek` | Native Greek medical MCQA (Voukoutis 2024 — one of your requested) |
| `ilsp/greek_lyceum_mathematics` | Greek high-school math exam |
| `ilsp/greek_civics_qa` | Greek civics QA |
| `ilsp/mcqa_greek_asep` | Greek civil-service exam MCQA (ASEP) |
| `ilsp/mt-bench-greek` | Greek MT-Bench (instruction-following) |
| `ilsp/ifeval_greek` | Greek IFEval (instruction-following constraints) |
| `ilsp/m-ArenaHard_greek` | Greek ArenaHard (multi-turn chat eval) |
| `ilsp/vibeeval_greek` | Greek "vibe" eval |
| `ilsp/flores200_en-el` + `ilsp/flores200_el-x` | Greek↔X translation slices |
| **`ilsp/ancient-modern_greek_translations`** | **Ancient↔Modern Greek translation pairs** — directly relevant to our polytonic arm |
| `ilsp/attaq_greek` | **AttaQ Greek (safety) — exists!** (Your earlier listing was uncertain.) |
| `ilsp/Jailbreak-StrongReject-el` | Greek jailbreak / safety eval |
| `ilsp/scipar_parallel_docs` | Scientific parallel docs |
| `ilsp/greek_pcr` | Greek PCR (likely sentence-completion or similar) |

This **covers everything you asked for from the "Native Greek" tier plus Apertus's English-baseline-translated-to-Greek versions**, all in one HF org. The only thing missing here is:
- **GreekMMLU (Zhang 2026, native-sourced)** — the more methodologically-strong variant. Not visible at `ilsp/` paths; may be in a different namespace (try `mediavalvazirgiannis/` per author institutional affiliations) or gated under the paper authors' control.
- **OYXOY** (Kogkalidis 2024) — not at canonical HF paths; likely on the authors' GitHub.
- **greek-nlp/benchmark** (Pavlopoulos / Bakagianni 2025) — gated.

### 2.3 Other Greek benchmarks (non-ILSP)

| Benchmark | Source | Status | Notes |
|---|---|---|---|
| **Belebele Greek** | `facebook/belebele`, config `ell_Grek` | ✓ open | Native-speaker-created reading comprehension. |
| **Greek Legal NER** | `joelniklaus/greek_legal_ner` | ✓ open | Aligns with C3 corpus's legal weighting. |
| **UD Greek GDT, GUD** | universaldependencies.org/treebanks/el_gdt + el_gud | not on HF; `.conllu` files | POS/morphology/syntax. Tiny (~MB). |
| **GreekMMLU (Zhang 2026, native)** | arXiv:2602.05150; HF location TBD | unknown | Authors at MBZUAI / NTUA; might be on author HF. |
| **OYXOY** | arXiv:2309.07009 | check `konstantinosKokos/oyxoy` on GitHub | NLI + WSD + metaphor |
| **greek-nlp/benchmark** | `greek-nlp/benchmark` (Pavlopoulos / Bakagianni 2025) | HF gated 401 | Authorship attribution + legal text clustering |
| **elNER** | `nlpaueb/elner` | HF gated 401 | Generic Greek NER |
| **GreekSUM** | `IMISLab/GreekSUM` | HF gated 401 | Native Greek summarization |
| **GreekBarBench** | not located | TBD | Legal long-form reasoning; needs targeted search |

### 2.4 Safety / toxicity

| Benchmark | Source | Status |
|---|---|---|
| `ilsp/attaq_greek` | ILSP HF | ✓ open — preferred over the IBM-published AttaQ-English |
| `ilsp/Jailbreak-StrongReject-el` | ILSP HF | ✓ open |
| `swiss-ai/harmbench` | swiss-ai HF | ✓ open — same benchmark Apertus team uses |
| `swiss-ai/polyglotoxicityprompts` | swiss-ai HF | ✓ open — multilingual incl. Greek |
| `swiss-ai/realtoxicityprompts` | swiss-ai HF | ✓ open |
| OGTD (Greek hate speech) | not at obvious HF paths | needs targeted search |
| DACHS (Greek hate speech) | not at obvious HF paths | needs targeted search |

The ILSP + swiss-ai variants give us coverage of the "safety axis" without OGTD/DACHS; both are deferred to a "deployment readiness" pass.

### 2.5 The Greeklish probe

**No published benchmark exists.** Needs construction. Proposed scope:
- 500-1000 hand-curated triples: `(Greek-script form, Greeklish form, alternating-script form)` for common-vocabulary words and short sentences.
- Evaluate as: per-token surprisal on each variant; perplexity ratio across variants.
- Defer to post-pilot — not blocking the three-arm comparison.

### 2.6 Cross-language regression slices (for plan §10 Q8a gates)

These are perplexity slices, not MCQ benchmarks. We construct by anti-joining the corresponding FineWeb-2 slice against the Apertus-overlap doc-ids in our dedup-audit overlay.

| Language | Source slice | Approx size |
|---|---|---|
| English | `epfml/FineWeb-HQ` (since Apertus uses FineWeb-HQ for English) | 1,000 docs |
| French | `HuggingFaceFW/fineweb-2` config `fra_Latn` | 1,000 docs |
| German | `HuggingFaceFW/fineweb-2` config `deu_Latn` | 1,000 docs |
| Russian | `HuggingFaceFW/fineweb-2` config `rus_Cyrl` | 1,000 docs |
| Italian (optional) | `HuggingFaceFW/fineweb-2` config `ita_Latn` | 1,000 docs |

Build script lives in 03_4 once we get to it. Computationally cheap — runs inside the same `xfer` allocation that builds the CPT corpus.

## 3. CSCS deployment — paths, commands, sequencing

### 3.1 Storage layout (target)

```
/iopsstor/scratch/cscs/fffoivos/
├── models/
│   └── apertus-8b-2509/              # base checkpoint, ~16 GB safetensors
├── tokenizers/
│   ├── apertus_greek_modern_only_148480/   # rsync from home
│   └── apertus_greek_extended_153600/      # rsync from home
├── benchmarks/
│   ├── apertus_baseline/             # ARC, HellaSwag, WinoGrande, XNLI, XCOPA, PIQA
│   ├── ilsp_greek/                   # the full ILSP suite
│   ├── other_greek/                  # Belebele Greek, Greek Legal NER, UD treebanks, OYXOY, etc.
│   ├── safety/                       # swiss-ai harmbench + polyglotoxicityprompts + ilsp attaq + jailbreak
│   └── regression_perplexity/        # en/fr/de/ru/it held-out slices (built later)
├── cpt_corpus_v1/                    # built by xfer job, not pulled
└── prepared_datasets/                # post-tokenization, for dataloader

/capstor/scratch/cscs/fffoivos/
├── runs/                             # all training outputs land here
│   ├── perf_check_v1/
│   ├── vanilla_calibration_v1/
│   ├── vanilla_pilot_v1/
│   ├── retok_pilot_v1/
│   └── distillation_pilot_v1/
└── code/                             # cloned source repos
    ├── pretrain-code/                # swiss-ai/pretrain-code (if we use Megatron-LM trunk)
    ├── apertus-finetuning-recipes/   # swiss-ai/apertus-finetuning-recipes (if we use HF Trainer trunk)
    ├── lm-evaluation-harness/        # swiss-ai/lm-evaluation-harness
    ├── token-distillation/           # konstantinjdobler/token-distillation
    ├── evals/                        # swiss-ai/evals (Apertus's own eval invocations)
    └── model-launch/                 # swiss-ai/model-launch (Alps sbatch CLI)

/users/fffoivos/                      # keep small — configs, dotfiles only
└── (no large artifacts)

/capstor/store/cscs/swissai/a0140/    # ← TODO: confirm project store exists or request creation
```

### 3.2 Login-node staging (no slurm allocation)

Everything below runs from a normal `ssh clariden` session (no sbatch). Total wall ~30-60 min depending on HF bandwidth.

```bash
# 0. uenv setup
ssh clariden
uenv start pytorch/v2.6.0:v1 --view=default
pip install --user huggingface_hub[hf_transfer] datasets transformers
export HF_HUB_ENABLE_HF_TRANSFER=1

# 1. Apertus base checkpoint (~16 GB)
mkdir -p /iopsstor/scratch/cscs/fffoivos/models
huggingface-cli download swiss-ai/Apertus-8B-2509 \
  --local-dir /iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509

# 2. Our tokenizer ship bundles (~40 MB)
mkdir -p /iopsstor/scratch/cscs/fffoivos/tokenizers
rsync -av home:/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/ship/ \
      /iopsstor/scratch/cscs/fffoivos/tokenizers/
# (rsync from clariden→home requires reverse-direction; in practice run this push from home with `scp -r ... fffoivos@clariden:`)

# 3. Source code repos
mkdir -p /capstor/scratch/cscs/fffoivos/code
cd /capstor/scratch/cscs/fffoivos/code
git clone https://github.com/swiss-ai/apertus-finetuning-recipes.git
git clone https://github.com/swiss-ai/lm-evaluation-harness.git
git clone https://github.com/swiss-ai/evals.git
git clone https://github.com/swiss-ai/model-launch.git
git clone https://github.com/swiss-ai/perf-check.git
git clone https://github.com/konstantinjdobler/token-distillation.git
git clone https://github.com/swiss-ai/pretrain-code.git    # keep warm for later

# 4. Apertus baseline benchmarks (lm-eval-harness handles these by config name)
# These usually download on first use, but we can pre-fetch:
mkdir -p /iopsstor/scratch/cscs/fffoivos/benchmarks/apertus_baseline
for ds in allenai/ai2_arc Rowan/hellaswag winogrande facebook/xnli xcopa piqa; do
  huggingface-cli download "$ds" \
    --repo-type dataset \
    --local-dir /iopsstor/scratch/cscs/fffoivos/benchmarks/apertus_baseline/$(echo $ds | tr / _)
done

# 5. ILSP Greek suite (the goldmine — pull all open ones)
mkdir -p /iopsstor/scratch/cscs/fffoivos/benchmarks/ilsp_greek
for ds in mmlu_greek MMLU-Pro_greek hellaswag_greek arc_greek winogrande_greek \
          truthful_qa_greek mgsm_greek medical_mcqa_greek \
          greek_lyceum_mathematics greek_civics_qa mcqa_greek_asep \
          mt-bench-greek ifeval_greek m-ArenaHard_greek vibeeval_greek \
          flores200_en-el flores200_el-x ancient-modern_greek_translations \
          attaq_greek Jailbreak-StrongReject-el greek_pcr; do
  huggingface-cli download ilsp/$ds \
    --repo-type dataset \
    --local-dir /iopsstor/scratch/cscs/fffoivos/benchmarks/ilsp_greek/$ds
done

# 6. Other Greek benchmarks (open ones)
mkdir -p /iopsstor/scratch/cscs/fffoivos/benchmarks/other_greek
huggingface-cli download facebook/belebele \
  --repo-type dataset --include 'ell_Grek/**' \
  --local-dir /iopsstor/scratch/cscs/fffoivos/benchmarks/other_greek/belebele_ell_Grek
huggingface-cli download joelniklaus/greek_legal_ner \
  --repo-type dataset \
  --local-dir /iopsstor/scratch/cscs/fffoivos/benchmarks/other_greek/greek_legal_ner

# UD treebanks (CoNLL-U; small)
mkdir -p /iopsstor/scratch/cscs/fffoivos/benchmarks/other_greek/ud_greek
for tb in UD_Greek-GDT UD_Greek-GUD; do
  git clone https://github.com/UniversalDependencies/$tb.git \
    /iopsstor/scratch/cscs/fffoivos/benchmarks/other_greek/ud_greek/$tb
done

# 7. Safety / toxicity (Apertus team's curated set + ILSP)
mkdir -p /iopsstor/scratch/cscs/fffoivos/benchmarks/safety
for ds in swiss-ai/harmbench swiss-ai/polyglotoxicityprompts swiss-ai/realtoxicityprompts; do
  huggingface-cli download "$ds" --repo-type dataset \
    --local-dir /iopsstor/scratch/cscs/fffoivos/benchmarks/safety/$(echo $ds | tr / _)
done

# 8. Verify what landed
du -sh /iopsstor/scratch/cscs/fffoivos/{models,tokenizers,benchmarks}
```

### 3.3 What requires slurm (run after step 8 confirms)

| Task | Partition | Walltime | Why slurm |
|---|---|---|---|
| Perf-check canary (`swiss-ai/perf-check`) | `debug` | 30 min | Needs GPU(s) to verify the stack |
| Cross-language regression slices build (anti-join FW2 against Apertus overlap) | `xfer` | 1 h | Pyarrow scans of large parquets; CPU-bound, fits the xfer 1-node shape |
| CPT corpus build (per `CPT_DATASET_BUILD_RUNBOOK.md`) | `xfer` | 24 h cap (real ~6-10 h) | The main batch CPU job, replaces the planned GCP scratch VM |
| Smoke train of vanilla path on Apertus base (1B tokens, 12h walltime) | `normal` | 12 h | The first GPU pilot |

### 3.4 What requires user action (gated / missing)

You'll need to log into HF on the browser and request access for these (your account, not mine):

- `greek-nlp/benchmark` (Pavlopoulos / Bakagianni 2025)
- `nlpaueb/elner` (Greek NER)
- `IMISLab/GreekSUM` (Greek summarization)
- (Optional) `Zhang2026/GreekMMLU` if the native-sourced GreekMMLU release is behind a gate

And for the genuinely-missing items, decide whether we need them for the pilot or can defer:

- **OYXOY** — likely on `konstantinosKokos/oyxoy` GitHub; if so I can pull via git clone, no gate.
- **GreekBarBench** — needs a targeted search of recent Greek-NLP papers; can defer to post-pilot.
- **OGTD / DACHS** — Greek hate-speech corpora; lookup needed. swiss-ai's harmbench + polyglottoxicity already cover the safety axis for the pilot.
- **Greeklish probe** — must be constructed; defer to post-pilot.

### 3.5 What Apertus uses for *training quality* monitoring (beyond benchmarks)

Looking at p-skarvelis's existing `run_config.json`, the in-flight signals tracked during training are:

- `train_loss` per step (via HF Trainer's logging_steps=10)
- `grad_norm` per step
- `learning_rate` per step
- Token throughput (`cluster_tokens_per_second`, `tokens_per_second_per_gpu`) from `phase_metrics.json`

For benchmark eval during training, the standard Apertus pattern is:
- Save checkpoint every N steps (they used `save_steps=1000` for the 1000-step prod run)
- Construct an `eval_views/checkpoint-N/` symlink-tree per saved checkpoint
- Run `swiss-ai/lm-evaluation-harness` against each eval-view directory
- Read `eval_loss` per checkpoint from the Trainer's eval_results.json

We mirror this exactly. See [`STORAGE_AND_EXISTING_WORK.md` § 3.2](STORAGE_AND_EXISTING_WORK.md#32-measured-throughput-from-phase_metricsjson-of-the-1000-step-prod) for the validated throughput on this stack.

## 4. Recommended next actions (ordered)

1. **You decide on the trunk**: `apertus-finetuning-recipes` (HF Trainer, fast adoption) vs `pretrain-code` (Megatron-LM, recipe-fidelity). [Review checkpoint D in ANALYSIS.md](../03_3_cscs_experiments_kickoff/ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off) — but my read tilts harder toward `apertus-finetuning-recipes` now that we've seen p-skarvelis's working pipeline.
2. **You request access** to the 3-4 gated HF datasets in §3.4.
3. **I run §3.2 login-node staging** — Apertus base + tokenizer bundles + code repos + Apertus baseline + ILSP Greek suite + safety. ~30-60 min, no slurm cost. Estimated landed footprint: ~25 GB on iopsstor, ~MB on capstor for code.
4. **I run §3.3 perf-check canary** on `debug` partition (30 min slurm). One-off validation the GPU stack is healthy.
5. **I write the cross-language regression slice builder** for §2.6; runs as part of the `xfer` CPT corpus build.
6. **You give go on CPT corpus build** — the actual `xfer` job.
7. **You give go on Vanilla pilot** — the first GPU run.

Want me to start §3 step 3 (the login-node staging) now? Cost is zero (no slurm allocation), wall ~30-60 min, completely reversible (just files on iopsstor).
