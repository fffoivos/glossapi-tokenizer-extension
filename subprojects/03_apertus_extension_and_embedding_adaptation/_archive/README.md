# _archive — superseded material from the 03 stage

> **In one line:** four groups of documents that were load-bearing at some point between 2026-05-20 and 2026-05-26 and then were replaced; nothing here is current, and everything here is kept because the conclusions that replaced it are only legible against what came before.
> **Period archived:** 2026-06-11 (`a19c136f`, "Checkpoint pending project updates" — the same commit that landed the final results and created [`../CPT_MASTER_20260526.md`](../CPT_MASTER_20260526.md)).
> **Status:** historical.

## Why this exists

The 03 stage accumulated ten overlapping planning and status documents in six days. On 2026-05-26 they were synthesised into one reference and moved here rather than deleted, so that any claim in the synthesis can be traced to the doc it came from — and so that reversals stay visible.

## What is here

### `synthesis_sources_20260526/` — the ten docs `CPT_MASTER` replaced

The two governing plans and their satellites:

- `old_experiments_plan.md` — **v0.12** (2026-05-12), the experimental-design parent: six decision nodes, the §4 hard constraints, and the §10 Q8 pre-commit decision rule whose thresholds were never locked.
- `cpt_plan.md` — **v0.7** (2026-05-20), the CPT-execution successor: recipe, replay design, §5.6 gates, V1–V16 verifications. Adopted as canonical in `01d7befa`.
- `cpt_plan_v0.7_answers.md`, `cpt_plan_v0.7_status.md` — the Q A/B/C/D decision snapshot and the V1–V16 status at the moment the bakeoff fired.
- `apertus_fidelity_checklist.md` — the 21 confirmed matches with Apertus pretraining, the three intentional deviations, and the production gates.
- `PLAN_VS_RESULTS_RECONCILIATION_20260526.md` — the reconciliation whose §10 punch list became `CPT_MASTER` §5's 14-entry discrepancy log.
- `PRODUCTION_DECISION_STATE.md` — the **2 B-stage** production pick (Vanilla). Carries its own 2026-05-26 banner noting partial supersession by the 5 B continuation; several other docs still link to it as if it were current.
- `ARTIFACTS_AND_HYDRATION.md` — the git-vs-Clariden ownership policy and the pre-launch hydration check.
- `CLARIDEN_INVENTORY_20260524.md` — the filesystem map (~6.9 TB of project state), folded into `CPT_MASTER` §7.2.
- `collegues_Apertus_plan.md` — the project's original Greek-language framing by p-skarvelis: rank GlossAPI sources by perplexity, quality and novelty before selecting CPT subsets. The framing was absorbed; its HF-Trainer scaffold was rejected in favour of Megatron-LM-Swiss-AI.

### `v0.6_planning/` — the iteration before v0.7

`cpt_plan_v0.6_answers.md` and `cpt_plan_v0.6_delta_vs_prior_planning.md`. Superseded within days. Read only to see which positions changed between v0.6 and v0.7 (framework, replay split, vocab scope).

### `2026-05-21_overnight_session/` — one execution night, ~219 KB

The corpus build, the first roundtrip and the first V4 baseline, logged in real time across 2026-05-20 → 21, then handed over mid-run to a second agent.

- `SESSION_LOG_20260521.md` — the audit trail: every local change and every Clariden job, with a three-track TL;DR.
- `TAKEOVER_LOG_20260521.md` — the handover log; records the corrections found on takeover (the missing `global_mmlu`, the "identical token streams" wording error, the `70/26/4` metadata that should have been `70/24/4/2`, the `normalize_nfc.sh` gap that would have skipped the final parquet).
- `CSCS_OVERNIGHT_STATE.md` — operating state at the time; the best single record of the six cancelled mix-builder chains and why each was abandoned.
- `SUGGESTIONS.md` — the forward-looking list from that night, tagged BLOCKER/PRIMARY/NICE/WATCH. Its open items were absorbed into the reconciliation's discrepancy log.

These are the reason several 2026-05-21 job numbers can be cited precisely; they are also the densest examples of failures being written down rather than smoothed over.

### `2026-05-24_2B_bakeoff_review/` — reviewer material whose conclusion was overturned

Produced after the 2 B bakeoff and **before** the 3.5 B and 5 B continuations, so its "Vanilla wins as the safe default" framing is pre-reversal.

- `REVIEW_HANDOFF_20260524.md` — single-map handoff for an external reviewer.
- `REVIEW_PRESENTATION.md` — narrative entry point over recipe, sbatch, eval and risks.
- `AUDIT_FINDINGS.md` — the two-round source-vs-implementation audit; still the citation for the corrected Megatron flag names (`--xielu`, `--ademamix-beta3-warmup`, `--ademamix-alpha-warmup`) and for §G, which flagged the missing HF→Megatron loader as a pre-submit blocker.
- `COMPLETENESS_CHECK.md` — inventory against cpt_plan v0.7 at 2026-05-21; honest about the gaps, including that the §5.6 gates existed only as prose and never became a pass/fail script.

## Reading notes

- Numbers in `PRODUCTION_DECISION_STATE.md` and in the 2 B review folder are **2 B-stage** numbers. The 5 B endpoint is in [`../03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md`](../03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md), and the Greek-specific headline is in [`../03_4_implementation_experiments/init_bakeoff/eval/NATIVE_GREEK_SUITE_RESULTS_20260526.md`](../03_4_implementation_experiments/init_bakeoff/eval/NATIVE_GREEK_SUITE_RESULTS_20260526.md).
- Several docs here contain absolute paths from the original working machine and links that predate the archive move. `git log --follow` on any file recovers its pre-move history.
- Older artifacts label the tokenizer-fair metric `BPC`; it is bits per UTF-8 **byte**. See [`../03_4_implementation_experiments/init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md`](../03_4_implementation_experiments/init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md).
