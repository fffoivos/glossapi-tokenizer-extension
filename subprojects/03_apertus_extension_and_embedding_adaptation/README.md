# 03 Apertus Extension And Embedding Adaptation

## Scope

Plan and later implement model-side adaptation after the tokenizer extension is frozen.

## Canonical plan

**[`cpt_plan.md`](cpt_plan.md) v0.7** is the canonical plan. Everything below is downstream of it. Live status of v0.7's verification list at [`cpt_plan_v0.7_status.md`](cpt_plan_v0.7_status.md). Critical Apertus architectural characteristics requiring CPT attention at [`apertus_fidelity_checklist.md`](apertus_fidelity_checklist.md).

Settled positions per v0.7 + 2026-05-20 user directives:

- **CPT vocab scope**: **153,600** (modern +17,408 + polytonic +5,120 — both extensions active in CPT). v0.7 §1's "148,480" wording is a typo; param math + §3.1 polytonic-exposure metrics imply 153,600.
- **Init arms**: Vanilla / ReTok / **Centroid** (Distillation bracketed in v0.7 §13).
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
- **Tokenizer cutoff: 17,408 modern Greek added** (2026-05-18, [`CHOSEN_CUTOFF.md`](../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md)); polytonic +5,120 stacked on top; **CPT vocab = 153,600**.
- **Polytonic / Ancient Greek as separate stacked layer: +5,120 → 153,600** (2026-05-18 polytonic-extension run; budget verified against sub-1B-language scaling pattern in [POLYTONIC_VOCAB_BUDGET_CHECK.md](03_3_cscs_experiments_kickoff/POLYTONIC_VOCAB_BUDGET_CHECK.md)).
- **Two ship tokenizer bundles assembled and verified** loadable via HF `AutoTokenizer`: [`apertus_greek_modern_only_148480/`](03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_only_148480/) (for the three-arm init comparison) and [`apertus_greek_extended_153600/`](03_3_cscs_experiments_kickoff/ship/apertus_greek_extended_153600/) (for the polytonic downstream arm). Both rebuilt from the broken `TokenizersBackend` wrapper configs that the C3 + polytonic builders emit.
- **CSCS Clariden auth is live** — account `a0140`, cert refresh via `cscs-key sign --headless --duration 1d` (verified end-to-end 2026-05-20).
- **CPT corpus recipe** is dedup-audited and turned into a runnable build path — [`CPT_DATASET_BUILD_RUNBOOK.md`](03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md).
- **Init-pilot corpus = Apertus-fresh-only** (the 03_2 dedup audit's overlay drops the ~2.27 % Apertus-overlap docs); main CPT after winning init can run on the mixed pool. Reasoning in [CURRICULUM_AND_INIT_CORPUS.md](03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md).

## Still Open (and where each lives)

- **Exact initialization method.** Three-arm comparison spec is locked (Vanilla / ReTok / Distillation per `experiments_plan.md` §5); arms not yet run. → 03_4.
- **Pre-commit decision-rule thresholds** (X, M_progress, M_ext, M_van, T) — must be locked before any arm completes CPT. → [03_3 ANALYSIS.md § Review checkpoint C](03_3_cscs_experiments_kickoff/ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off).
- **CSCS training-harness choice** (Swiss-AI Megatron-LM vs nanotron vs HF + accelerate). → [03_3 ANALYSIS.md § Review checkpoint D](03_3_cscs_experiments_kickoff/ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off).
- **CPT mix ratios + non-Greek replay percentage.** Default proposal in [CURRICULUM_AND_INIT_CORPUS.md](03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md). → [03_3 ANALYSIS.md § Review checkpoint E](03_3_cscs_experiments_kickoff/ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off).
- **Held-out contamination check on C3 val/test** (the dedup audit's run skipped this — the C3 mix manifest lived on the now-unreachable gcloud tokenizer instance; **GCloud access was lost 2026-05-20**, so the previously-suggested "restart the instance" alternative is gone). Remaining options: re-derive the val/test partition by re-running the splitter from the published nanochat corpus with the original seed, or live with the gap. → [03_3 ANALYSIS.md § Review checkpoint B](03_3_cscs_experiments_kickoff/ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off).
- **Eval-suite materialization** — cross-language regression slices (en/fr/de/ru) + first native-Greek benchmark (GreekMMLU) wired into a working harness. → 03_4.
