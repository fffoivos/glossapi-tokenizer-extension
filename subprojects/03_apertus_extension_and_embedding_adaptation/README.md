# 03 Apertus Extension And Embedding Adaptation

## Scope

Plan and later implement model-side adaptation after the tokenizer extension is frozen.

## Canonical reference

**[`CPT_MASTER_20260526.md`](CPT_MASTER_20260526.md)** is the canonical single-doc synthesis. Read it first; it consolidates the experimental-design plan (v0.12), CPT-execution plan (v0.7), Apertus-fidelity checklist, reconciliation + discrepancy log, production decision state, and operational reference into one document. The 10 source docs it synthesizes are archived at [`_archive/synthesis_sources_20260526/`](_archive/synthesis_sources_20260526/).

## Artifacts (Hugging Face)

Model weights, the extended tokenizer, benchmark summaries, and a copy of the decision docs live on Hugging Face:

**[`fffoivos/apertus-tokenizer-extension`](https://huggingface.co/fffoivos/apertus-tokenizer-extension)**

The split is:
- **GitHub (this repo)** — scripts, recipes, sbatch launchers, verification scripts, plan / decision docs (control plane).
- **Hugging Face** — `experiment-checkpoints/` (model weights), `greek-extension-tokenizer/` (tokenizer), `benchmark-evals/` (summary tables + plots), `supporting-material/` (manifests, provenance, decisions copied from GitHub).

Decision docs (`CPT_MASTER_20260526.md`, `PLAN_VS_RESULTS_RECONCILIATION_20260526.md`) are mirrored from this repo to `supporting-material/provenance/decisions/` on HF.

**Current state (2026-05-26):**

- 4-arm bakeoff complete (Vanilla + TD to 5 B, ReTok to 3.5 B, Centroid to 2 B). 5 B endpoint in [`03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md`](03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md).
- The 2 B headline (Vanilla wins) was partially flipped by the 3.5 B + 5 B continuations: at 5 B, **TD layer11 leads all three downstream aggregates** (Greek no-MT / EN retention / Multilingual); **Vanilla still leads tokenizer-fair BPB**.
- **No `v0.12 §10 Q8` thresholds (X / M_progress / M_ext / M_van / T) were ever pre-committed**, so the bakeoff produced data, not an adjudicated winner. See [`CPT_MASTER_20260526.md`](CPT_MASTER_20260526.md) §5 for the 14-entry discrepancy log.
- The 2 B-stage "Vanilla as safe default" pick is partially superseded by 5 B continuation. Captured in `CPT_MASTER` §8. A planning-agent pass owns the next production-decision update.

**Loss measurement policy:** raw Megatron `lm loss` is per-token CE and is not
cross-tokenizer fair. Use heldout BPB and downstream evals for
Vanilla-vs-extended decisions. Older reports may call BPB `BPC`; this is a
legacy bits-per-byte label, not bits per character. See
[`LOSS_MEASUREMENT_POLICY.md`](03_4_implementation_experiments/init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md)
and the repo-wide [`docs/LOSS_MEASUREMENT_POLICY.md`](../../docs/LOSS_MEASUREMENT_POLICY.md).

Settled positions (post-5 B):

- **CPT vocab scope**: 17,408 modern-Greek extension (vocab 148,480) used for the bakeoff. The base 131,072-token tokenizer remains live as Vanilla's path. A separate production-side cutoff sweep on `{10,240, 15,360, 20,480, 25,600}` is open per `glossapi_c3_convergence.md` but downstream of the bakeoff.
- **Init arms**: Vanilla / ReTok / Centroid / TD-layer11 bakeoff complete. Centroid clearly broken (BPC ~0.90 vs ≤0.55 elsewhere). ReTok dominated by TD on every shared iter. **TD vs Vanilla is the only genuinely close call**; rule-bound winner pending Node 4 thresholds.
- **Training framework**: **Megatron-LM-Swiss-AI** (`swiss-ai/Megatron-LM` + `swiss-ai/pretrain-code`).
- **Replay split**: 70 % Greek / 24 % replay / 4 % code / 2 % math (cpt_plan v0.7 §B1). Higher Greek share than `old_experiments_plan.md` v0.12 §8.5's 10-15 % replay starting point; documented divergence.
- **Replay languages**: 24-language tier set (8 Tier-1 + 11 Tier-2 + 5 Tier-3) per cpt_plan v0.7 §4.2.
- **Bakeoff size**: 2 B per arm planned (cpt_plan v0.7 §B5), extended ad-hoc to 5 B for Vanilla + TD, 3.5 B for ReTok, 2 B for Centroid. Production CPT 15-20 B remains the working assumption (Q A2 still pending).

## Sub-subprojects (in chronological order, latest first)

- **[03_4_implementation_experiments/](03_4_implementation_experiments/README.md)** — the hands-on runs. Where the 4-arm bakeoff actually trained and was evaluated on Clariden. Contains `init_bakeoff/{arms,bakeoff_training,eval,production_cpt,token_distillation}`. Trajectory analysis + 5 B endpoint at `init_bakeoff/eval/trajectory_analysis_20260524/`.
- **[03_3_cscs_experiments_kickoff/](03_3_cscs_experiments_kickoff/README.md)** — the planning + verification work that bridges the old planning era and v0.7. Reconciles the colleague's plan and the v0.12 experimental design (both now archived at [`_archive/synthesis_sources_20260526/`](_archive/synthesis_sources_20260526/)) with the dedup audit, the diagnostic v2 report, the 2026-05-18 tokenizer cutoff decision, the polytonic +5,120 layer, the `cscs-key` auth tool, and the working Clariden launch pattern.
- [03_2_apertus_c3_dedup_audit/](03_2_apertus_c3_dedup_audit/README.md) — measures document-level overlap between Apertus's Greek pretraining sources and the C3 tokenizer-training corpus. Output: per-source `include_full / include_half_weight / replay_only` recommendations + a hard-drop overlay parquet for the CPT mix. Run completed 2026-05-19.
- [03_1_greek_embedding_diagnostic/](03_1_greek_embedding_diagnostic/README.md) — pre-extension diagnostic characterising how Apertus-8B-2509 represents Greek on its E + U matrices (centroid geometry, MP-edge spectrum, binary Greek-vs-¬Greek classifier macro F1 = 1.00, morphological clustering 5–9× tightness, cross-language semantic-cluster baseline showing no Greek↔English etymology bridge). Completed 2026-05-13 v2.3.

## Reference docs in this folder

**Canonical:**
- [`CPT_MASTER_20260526.md`](CPT_MASTER_20260526.md) — single-doc synthesis of everything (plans, fidelity, reconciliation, decision state, operational reference). **Read first.**

**Still-active companion docs:**
- [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) — production training spec (referenced by `_train_config_common.env`).
- [`TOKEN_DISTILLATION_PLAN.md`](TOKEN_DISTILLATION_PLAN.md) — TD-specific plan (4th-arm spec).
- [`RISKS.md`](RISKS.md) — 17-risk silent-failure inventory.
- [`TODO.md`](TODO.md) — current active items.

**Operational rule (lives downstream, referenced from CPT_MASTER §7):**
- [`03_4_implementation_experiments/init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md`](03_4_implementation_experiments/init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md) — canonical loss-reading rule (raw `lm loss` vs heldout BPB + the historical BPC alias).

**Archived:** [`_archive/`](_archive/README.md). Contains:
- `synthesis_sources_20260526/` — the 10 source docs CPT_MASTER was synthesized from (v0.12 + v0.7 plans, fidelity checklist, reconciliation, etc.)
- `v0.6_planning/` — v0.6 plan iterations
- `2026-05-21_overnight_session/` — operational logs from one specific CSCS execution night
- `2026-05-24_2B_bakeoff_review/` — pre-5B reviewer material

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
- **Production eval gates.** Bakeoff evidence selects the default path; the production run still needs final stop/go gates and checkpoint-window rubric attached to its run directory. → [`CPT_MASTER_20260526.md`](CPT_MASTER_20260526.md) §8-§9.
- **Held-out contamination check on C3 val/test** (the dedup audit's run skipped this — the C3 mix manifest lived on the now-unreachable gcloud tokenizer instance; **GCloud access was lost 2026-05-20**, so the previously-suggested "restart the instance" alternative is gone). Remaining options: re-derive the val/test partition by re-running the splitter from the published nanochat corpus with the original seed, or live with the gap. → [03_3 ANALYSIS.md § Review checkpoint B](03_3_cscs_experiments_kickoff/ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off).
