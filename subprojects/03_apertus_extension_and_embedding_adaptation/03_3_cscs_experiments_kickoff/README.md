# 03.3 CSCS experiments kickoff — state audit and path forward

## Scope

Bridge document between
[`../experiments_plan.md`](../experiments_plan.md) (last updated
2026-05-12, v0.12) and the next concrete step — actually starting
the three-arm comparison on Swiss-AI compute. Three jobs:

1. **State audit.** Walk through every decision node and every open
   question in `experiments_plan.md` and mark it against the 8 days
   of follow-on work (diagnostic v2 → dedup audit r1–r4 → CPT
   runbook → user-level decisions).
2. **CSCS auth + execution scheme.** Capture the daily-cert handoff
   so that this Claude session (and every future one) can run
   workloads on `clariden` / `bristen` / `ela` without ad-hoc
   re-discovery.
3. **Next-action sequence.** Order the remaining gating work between
   *here* and the first Vanilla / ReTok / Distillation pilot
   actually firing on Alps.

This subproject is **analysis only** — `experiments_plan.md` is not
edited. Anything that needs to land in the plan is recorded in
[`ANALYSIS.md` § Review checkpoints](ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off)
so we can fold them in together after review.

## Files

- [`ANALYSIS.md`](ANALYSIS.md) — main deliverable: Δ since plan,
  what the diagnostic v2 tightened, CPT-corpus shape, what's left,
  CSCS workflow, recommended sequence, review checkpoints.
- [`POLYTONIC_VOCAB_BUDGET_CHECK.md`](POLYTONIC_VOCAB_BUDGET_CHECK.md) — verifies the +5,120 polytonic
  added-token budget against the sub-1B-language scaling pattern
  derived from `02_2_2_vocab_lang_attribution/outputs/lang_metadata.json`.
  Confirms: corpus is ~223 M tokens before the polytonic extension and
  ~163 M tokens after it; the apples-to-apples post-extension pattern
  predicts ~4,000–6,300 distinctive vocab tokens; +5,120 sits inside
  that corrected band; total vocab 153,600 = 256 × 600 ✓.
- [`CSCS_AUTH_WORKFLOW.md`](CSCS_AUTH_WORKFLOW.md) — exact daily-cert
  flow using the **new `cscs-key` Rust tool v1.1.0+** (the old
  `sshservice-cli` is deprecated). Covers install, one-time keypair
  setup, daily `cscs-key sign` (or `--headless` for remote), and the
  verified Clariden launch pattern (`a0140` account,
  `pytorch/v2.6.0:v1` uenv image, working 2-node 8-GPU DDP smoke).
- [`SHIP_TOKENIZER_RECONSTRUCTION.md`](SHIP_TOKENIZER_RECONSTRUCTION.md)
  — verifies and rebuilds the two Apertus-compatible tokenizers. Ship
  bundles at [`ship/apertus_greek_modern_only_148480/`](ship/apertus_greek_modern_only_148480/) (vocab 148,480 = 256 × 580, for the
  three-arm Vanilla/ReTok/Distillation comparison) and [`ship/apertus_greek_extended_153600/`](ship/apertus_greek_extended_153600/) (vocab 153,600 = 256 × 600, for the downstream polytonic
  specialization arm). Both loadable via `AutoTokenizer`, polytonic
  fertility win 53.3 % / 66.7 % respectively (60 → 28 / 60 → 20
  tokens on NT-style text), English / Russian unchanged.
- [`CURRICULUM_AND_INIT_CORPUS.md`](CURRICULUM_AND_INIT_CORPUS.md)
  — reconciles colleague's `Apertus_plan.md` ranking with your
  HPLT-foundation curriculum and the dedup audit's per-source
  recommendations. Argues fresh-only for init pilots, mixed for main
  CPT, and lays out a 4-phase curriculum (HPLT broad → register
  diversity → academic+legal → dictionary gap restoration).
- [`REPLAY_LANGUAGE_SELECTION.md`](REPLAY_LANGUAGE_SELECTION.md) —
  picks 34 languages for the non-Greek CPT replay (3-tier
  budget allocation), using the four user-supplied criteria
  (geographic / Western-Europe-bridge / historical-Greek / global
  diversity) with explicit convergence tables and per-language
  justifications.
- (planned, post-review) `PLAN_DIFF.md` — concrete edits to apply
  back to `experiments_plan.md` once the review checkpoints are
  signed off.

## Inputs feeding this audit

- [`../experiments_plan.md`](../experiments_plan.md) v0.12 (2026-05-12)
- [`../03_1_greek_embedding_diagnostic/artifacts/results/report_v2.md`](../03_1_greek_embedding_diagnostic/artifacts/results/report_v2.md) (2026-05-13, v2.3)
- [`../03_2_apertus_c3_dedup_audit/REPORT_dedup_20260519T010924Z.md`](../03_2_apertus_c3_dedup_audit/REPORT_dedup_20260519T010924Z.md) (2026-05-19)
- [`../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md`](../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md)
- [`../../../docs/C3_CUTOFF_REPORT.md`](../../../docs/C3_CUTOFF_REPORT.md), [`../../../docs/C3_CONVERGENCE.md`](../../../docs/C3_CONVERGENCE.md), [`../../../docs/GLOBAL_DECISIONS.md`](../../../docs/GLOBAL_DECISIONS.md), [`../../../docs/CURRENT_STATUS.md`](../../../docs/CURRENT_STATUS.md)
- `~/.ssh/{config,cscs-key,cscs-key-cert.pub}` (existing CSCS setup)
