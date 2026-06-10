# 03 — Training & Experiments

> **Execution entry point: [`HANDOFF.md`](HANDOFF.md)** (2026-06-09). Current
> spec: 10B unseen Greek = **70% HPLT + 30% openarchives**; **3 held-out val
> sets** (0.5B each: hplt/openarchives/greek_phd) with per-set loss at eval
> cadence (`dataset_build/EXTRA_VALID_README.md`). Final train JSONL preserves
> replay/non-new-Greek line positions and orders only the HPLT/openarchives
> slots.

Setup for the two parallel Greek-CPT experiments on Apertus-8B. This subproject
holds **configs, the chain submitter, and the runbook** — it does *not* fork the
training/TD/convert/eval code. Per the hard rule (use the most established tool
for each job), every heavy step calls an established library; the bespoke pieces
that remain are thin drivers, unavoidable gap-fillers, or acceptance gates (see
[`TOOLING_DECISIONS.md`](TOOLING_DECISIONS.md)).

## The two experiments (run in parallel)

| | Arm 1 — **Vanilla** | Arm 2 — **Modern-Greek** |
|---|---|---|
| Tokenizer | Apertus base **131,072** | extended **148,480** (+17,408 modern Greek) |
| New-token init | — (none) | **Token Distillation, layer 11** — E by MSE-distillation, U by CE ([`docs/TOKEN_DISTILLATION_E_AND_U.md`](docs/TOKEN_DISTILLATION_E_AND_U.md)) |
| Budget | **10B new Greek + replay ≈ 13.5B total** | same |
| Schedule | **full-run WSD** (warmup → stable → 20% cooldown), all schedulers over the whole run | same |
| Everything else | optimizer / LR / geometry / batch / Goldfish / mixture **identical** | identical |

The only per-arm differences are vocab, init checkpoint, tokenizer dir, and data
prefix. The shared Stage-C bulk mix is tokenized twice (same ordered JSONL →
byte-identical document stream) so the two arms differ *only* in tokenization +
init.

## Established tools per job

| Job | Tool |
|---|---|
| Train | swiss-ai/Megatron-LM fork `pretrain_gpt.py` via `bakeoff_train.sbatch` |
| New-token init | Dobler **token-distillation** (arXiv:2505.20133) `train_embeddings` |
| HF↔Megatron convert | fork `tools/checkpoint/convert.py` (+ R17 patch for xIELU/QK-Norm) |
| Data tokenize | Megatron `tools/preprocess_data.py` |
| Eval | lm-evaluation-harness (swiss-ai fork) |

## Layout

```
configs/   common_cpt.env        finalized hyperparameters (CURRENT_HYPERPARAMETERS v1.0)
           arm1_vanilla.env      arm-1 deltas (base tokenizer, base init/data)
           arm2_modern_greek.env arm-2 deltas (148480 tokenizer, TD init, ext data)
scripts/   submit_two_arm_full_run.sh   walltime-bounded chain submitter (per arm)
docs/      TOKEN_DISTILLATION_E_AND_U.md   how both E and U are produced, the repo's way
           SCHEDULER_MATH.md               13.5B → step counts, warmup floor, cooldown
BUILD_PLAN.md        end-to-end runbook: convert → TD → data → train → eval
TOOLING_DECISIONS.md established-vs-bespoke table + verdicts
```

## Quick start (after the pre-launch checklist in BUILD_PLAN)

```bash
# dry-run both chains first
bash scripts/submit_two_arm_full_run.sh vanilla
bash scripts/submit_two_arm_full_run.sh td
# then, once dataset + init checkpoints exist and the checklist passes:
DRY_RUN=0 CONFIRM_LAUNCH=1 bash scripts/submit_two_arm_full_run.sh vanilla
DRY_RUN=0 CONFIRM_LAUNCH=1 bash scripts/submit_two_arm_full_run.sh td
```

## Status — decisions made; remaining work is execution (see LAUNCH_RUNBOOK)

**Settled:** mixture (10B new + 35%-of-new = 13.5B), warmup (`2/(1−β2)`), vocab
(148,480, 256-aligned), and the entire **corpus-prep pipeline** —
`../02_corpus_preparation/PIPELINE.md` fixes order (clean→dedup→decontaminate→
anonymize-last), no global NFC (Apertus `normalizer:null`), PII as stage 4, and
HPLT = confident-only residue cleaning. Nothing here is an open dilemma.

**Remaining = execution / verification** (not decisions):
- Run the corpus-prep full-corpus stages on Clariden (clean overlay materialize+apply, decontaminate, anonymize) on the existing 129 GB `SELECTED`.
- Mix to 13.5B (+ the `apertus_overlap_drop` Greek-replay bucket) → Stage-C
  replay-fixed HPLT/OpenArchives slot order → tokenize twice.
- Build the two init checkpoints (convert+revert+R17; TD for arm2 — reuse `td_full25_layer11_r17_roundtrip` iff its tokenizer == `apertus_greek_modern_only_148480`).
- `launch_all.sh` → two parallel arms + auto-benchmarks.

Nothing has been launched.
