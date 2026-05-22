# 03 Apertus Extension And Embedding Adaptation

## Scope

Plan and later implement model-side adaptation after the tokenizer extension is frozen.

## Canonical plan

**[`cpt_plan.md`](cpt_plan.md) v0.7** is the canonical plan. Everything below is downstream of it. Live status of v0.7's verification list at [`cpt_plan_v0.7_status.md`](cpt_plan_v0.7_status.md). Critical Apertus architectural characteristics requiring CPT attention at [`apertus_fidelity_checklist.md`](apertus_fidelity_checklist.md).

**Current production decision state:** [`PRODUCTION_DECISION_STATE.md`](PRODUCTION_DECISION_STATE.md). The 2B bakeoff currently selects **Vanilla Apertus-8B with the base 131,072-token tokenizer** as the safe 15-20B CPT default. Centroid is eliminated. ReTok is not selected as-is; it remains only as a bounded Token Distillation challenger after a CPU coverage prepass on `xfer`.

Settled positions per v0.7 + 2026-05-20 user directives:

- **CPT vocab scope**: production default is now **131,072** unless Token Distillation proves the ReTok extended path. The modern-only 148,480 tokenizer and future 153,600 modern+polytonic tokenizer remain valid artifacts, but they are no longer selected automatically just because the extension exists.
- **Init arms**: Vanilla / ReTok / Centroid bakeoff completed. Current decision: Vanilla default; Centroid eliminated; ReTok only through the bounded `retok_td` challenger described in [`TOKEN_DISTILLATION_PLAN.md`](TOKEN_DISTILLATION_PLAN.md).
- **Training framework**: **Megatron-LM-Swiss-AI** (`swiss-ai/Megatron-LM` + `swiss-ai/pretrain-code`). p-skarvelis's HF-Trainer pipeline is an interesting baseline, not the scaffold we adopt.
- **Replay split**: 70/30 Greek/non-Greek (v0.7 §4.1 default).
- **Replay languages**: 24-language tier set (8 Tier-1 + 11 Tier-2 + 5 Tier-3) per v0.7 §4.2.
- **Bakeoff size**: 2 B tokens per arm × 3 arms = 6 B (init-method discrimination only). Production CPT on winner: 10–20 B (v0.7 §9, Q A2).

## Sub-subprojects (in chronological order, latest first)

- **[03_4_implementation_experiments/](03_4_implementation_experiments/README.md)** — the hands-on runs. Authors and submits sbatch on CSCS Clariden. Currently has the auth + node-finding probe and the first calibration-run sizing.
- **[03_3_cscs_experiments_kickoff/](03_3_cscs_experiments_kickoff/README.md)** — the planning + verification work that bridges the old planning era and v0.7. Reconciles the colleague's [`collegues_Apertus_plan.md`](collegues_Apertus_plan.md) and the older [`old_experiments_plan.md`](old_experiments_plan.md) with the dedup audit, the diagnostic v2 report, the 2026-05-18 tokenizer cutoff decision, the polytonic +5,120 layer, the new `cscs-key` auth tool, and the working Clariden launch pattern. Each downstream doc now carries a top-of-file callout flagging where v0.7 supersedes its body.
- [03_2_apertus_c3_dedup_audit/](03_2_apertus_c3_dedup_audit/README.md) — measures document-level overlap between Apertus's Greek pretraining sources and the C3 tokenizer-training corpus. Output: per-source `include_full / include_half_weight / replay_only` recommendations + a hard-drop overlay parquet for the CPT mix. Run completed 2026-05-19.
- [03_1_greek_embedding_diagnostic/](03_1_greek_embedding_diagnostic/README.md) — pre-extension diagnostic characterising how Apertus-8B-2509 represents Greek on its E + U matrices (centroid geometry, MP-edge spectrum, binary Greek-vs-¬Greek classifier macro F1 = 1.00, morphological clustering 5–9× tightness, cross-language semantic-cluster baseline showing no Greek↔English etymology bridge). Completed 2026-05-13 v2.3.

## Reference docs in this folder

- [`cpt_plan.md`](cpt_plan.md) — **canonical, v0.7** (2026-05-20). Supersedes everything below.
- [`PRODUCTION_DECISION_STATE.md`](PRODUCTION_DECISION_STATE.md) — current evidence-backed production path after the 2B bakeoff.
- [`TOKEN_DISTILLATION_PLAN.md`](TOKEN_DISTILLATION_PLAN.md) — parallel-ready follow-up plan for a ReTok + Token Distillation challenger if the live bakeoff leaves ReTok promising but under-initialized.
- [`cpt_plan_v0.7_status.md`](cpt_plan_v0.7_status.md) — live status answers to v0.7's V1–V16 verification questions. Confirms V14 + V16 + V9 (NFC) done; rest scheduled for Clariden.
- [`apertus_fidelity_checklist.md`](apertus_fidelity_checklist.md) — critical Apertus architectural characteristics requiring special CPT attention (AdEMAMix + 0.1 clip + xIELU scalars + Goldfish + cross-doc attention + etc). Architectural facts, version-independent.
- [`cpt_plan_v0.6_answers.md`](cpt_plan_v0.6_answers.md) — **historical** (v0.6-era status). Retained for traceability.
- [`cpt_plan_v0.6_delta_vs_prior_planning.md`](cpt_plan_v0.6_delta_vs_prior_planning.md) — **historical** (v0.6 contradictions with pre-v0.6 planning).
- [`collegues_Apertus_plan.md`](collegues_Apertus_plan.md) — colleague's parallel curriculum proposal (PPL/quality/novelty ranking). Predates cpt_plan v0.x; informs the 03_3 reconciliation work but is not canonical.
- [`old_experiments_plan.md`](old_experiments_plan.md) — the v0.12 (2026-05-12) project plan that preceded cpt_plan.md. Retained for traceability.

## What's Already Decided

- This work comes after tokenizer and corpus work, not before.
- Embeddings and `lm_head` both matter because `tie_word_embeddings = false`.
- Only the new rows need explicit initialization.
- The intended schedule is WSD with brief re-warmup → plateau → linear decay aligned with anneal (v0.7 §3.3).
- **Tokenizer cutoff: 17,408 modern Greek added** (2026-05-18, [`CHOSEN_CUTOFF.md`](../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md)); polytonic +5,120 stacked on top. These artifacts remain available, but the post-bakeoff production default is base-tokenizer Vanilla unless `retok_td` passes its gates.
- **Polytonic / Ancient Greek as separate stacked layer: +5,120 → 153,600** (2026-05-18 polytonic-extension run; budget verified against sub-1B-language scaling pattern in [POLYTONIC_VOCAB_BUDGET_CHECK.md](03_3_cscs_experiments_kickoff/POLYTONIC_VOCAB_BUDGET_CHECK.md)).
- **Two ship tokenizer bundles assembled and verified** loadable via HF `AutoTokenizer`: [`apertus_greek_modern_only_148480/`](03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_only_148480/) (for the three-arm init comparison) and [`apertus_greek_extended_153600/`](03_3_cscs_experiments_kickoff/ship/apertus_greek_extended_153600/) (for the polytonic downstream arm). Both rebuilt from the broken `TokenizersBackend` wrapper configs that the C3 + polytonic builders emit.
- **CSCS Clariden auth is live** — account `a0140`, cert refresh via `cscs-key sign --headless --duration 1d` (verified end-to-end 2026-05-20).
- **CPT corpus recipe** is dedup-audited and turned into a runnable build path — [`CPT_DATASET_BUILD_RUNBOOK.md`](03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md).
- **Init-pilot corpus = Apertus-fresh-only** (the 03_2 dedup audit's overlay drops the ~2.27 % Apertus-overlap docs); main CPT after winning init can run on the mixed pool. Reasoning in [CURRICULUM_AND_INIT_CORPUS.md](03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md).

## Still Open (and where each lives)

- **Token Distillation challenger.** The three-arm bakeoff has run; only bounded `retok_td` remains as a possible challenger. First gate: CPU firing/coverage prepass on `xfer`. → [`TOKEN_DISTILLATION_PLAN.md`](TOKEN_DISTILLATION_PLAN.md) and [`init_bakeoff/token_distillation/`](03_4_implementation_experiments/init_bakeoff/token_distillation/).
- **Production CPT dataset manifest.** The recipe is fixed at 70/24/4/2 for the current path, but the 15-20B production stream still needs its final build or rehydration manifest. CPU-only build/preprocess work belongs on `xfer`. → [`init_bakeoff/corpus_build/MIX_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md).
- **Production eval gates.** Bakeoff evidence selects the default path; the production run still needs final stop/go gates and checkpoint-window rubric attached to its run directory. → [`PRODUCTION_DECISION_STATE.md`](PRODUCTION_DECISION_STATE.md).
- **Held-out contamination check on C3 val/test** (the dedup audit's run skipped this — the C3 mix manifest lived on the now-unreachable gcloud tokenizer instance; **GCloud access was lost 2026-05-20**, so the previously-suggested "restart the instance" alternative is gone). Remaining options: re-derive the val/test partition by re-running the splitter from the published nanochat corpus with the original seed, or live with the gap. → [03_3 ANALYSIS.md § Review checkpoint B](03_3_cscs_experiments_kickoff/ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off).
