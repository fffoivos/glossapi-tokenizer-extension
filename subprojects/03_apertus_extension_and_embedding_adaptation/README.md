# 03 Apertus Extension And Embedding Adaptation

## Scope

Plan and later implement model-side adaptation after the tokenizer extension is frozen.

## Sub-subprojects (in chronological order, latest first)

- **[03_4_implementation_experiments/](03_4_implementation_experiments/README.md)** — the hands-on runs. Authors and submits sbatch on CSCS Clariden. Currently has the auth + node-finding probe and the first calibration-run sizing.
- **[03_3_cscs_experiments_kickoff/](03_3_cscs_experiments_kickoff/README.md)** — the planning + verification doc that connects everything else. Reconciles `experiments_plan.md` (this folder's parent plan) and the colleague's `Apertus_plan.md` with the dedup audit, the diagnostic v2 report, the 2026-05-18 tokenizer cutoff decision, the polytonic +5,120 layer, the new `cscs-key` auth tool, and the working Clariden launch pattern. **Start here for review** — read [`03_3_cscs_experiments_kickoff/REVIEW_PACKET.md`](03_3_cscs_experiments_kickoff/REVIEW_PACKET.md).
- [03_2_apertus_c3_dedup_audit/](03_2_apertus_c3_dedup_audit/README.md) — measures document-level overlap between Apertus's Greek pretraining sources and the C3 tokenizer-training corpus. Output: per-source `include_full / include_half_weight / replay_only` recommendations + a hard-drop overlay parquet for the CPT mix. Run completed 2026-05-19.
- [03_1_greek_embedding_diagnostic/](03_1_greek_embedding_diagnostic/README.md) — pre-extension diagnostic characterising how Apertus-8B-2509 represents Greek on its E + U matrices (centroid geometry, MP-edge spectrum, binary Greek-vs-¬Greek classifier macro F1 = 1.00, morphological clustering 5–9× tightness, cross-language semantic-cluster baseline showing no Greek↔English etymology bridge). Completed 2026-05-13 v2.3.

## Live anchor docs in this folder

- [`experiments_plan.md`](experiments_plan.md) (v0.12, 2026-05-12) — the parent project plan, six decision nodes, three-arm comparison spec. Not edited in this session; pending diffs collected in [03_3 ANALYSIS.md § Review checkpoints](03_3_cscs_experiments_kickoff/ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off).
- [`Apertus_plan.md`](Apertus_plan.md) — colleague's parallel curriculum proposal (PPL/quality/novelty ranking + 3-phase curriculum + Claude-review notes at the bottom). Reconciled with the dedup audit and 03_3's HPLT-broad foundation in [CURRICULUM_AND_INIT_CORPUS.md](03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md).

## What's Already Decided

- This work comes after tokenizer and corpus work, not before.
- Embeddings and `lm_head` both matter because `tie_word_embeddings = false`.
- Only the new rows need explicit initialization.
- The intended schedule is frozen-base warmup, then full continued pretraining.
- **Tokenizer cutoff: 17,408 modern Greek added → 148,480 total** (2026-05-18, [`CHOSEN_CUTOFF.md`](../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md)).
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
