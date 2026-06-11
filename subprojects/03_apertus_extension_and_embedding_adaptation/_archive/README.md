# Archived material — Apertus Greek CPT (03 stage)

Historical and superseded docs from the 03 stage. Nothing here is load-bearing
for current decisions. Kept for traceability — see `git log --follow` on any
file for its history before the move.

Organized by what each group is from / for.

## `v0.6_planning/`

Pre-v0.7 cpt_plan iteration. Superseded by `../cpt_plan.md` v0.7 and
`../cpt_plan_v0.7_answers.md`.

- `cpt_plan_v0.6_answers.md` — Q A / B / C / D + V1-V16 answers at v0.6.
- `cpt_plan_v0.6_delta_vs_prior_planning.md` — v0.6's delta against the v0.12
  parent plan.

## `2026-05-21_overnight_session/`

Operational state and audit trail from one specific CSCS execution night
(2026-05-20 → 21). The bakeoff fired downstream of this session; all four
arms reached at least 2 B tokens, with Vanilla + TD continuing to 5 B.

- `SESSION_LOG_20260521.md` — full audit trail of changes + jobs.
- `TAKEOVER_LOG_20260521.md` — Codex takeover operational log.
- `CSCS_OVERNIGHT_STATE.md` — current operating state at the time. Long
  superseded by `../CLARIDEN_INVENTORY_20260524.md` and
  `03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md`.
- `SUGGESTIONS.md` — forward-looking signals from that session. Open items
  (V1 decontamination, V4 bootstrap CIs, codeparrot-vs-StarCoder, math
  2 % bucket) are absorbed into `../PLAN_VS_RESULTS_RECONCILIATION_20260526.md`
  §10 discrepancy log; the SUGGESTIONS doc itself is retained for context.

## `synthesis_sources_20260526/`

The 10 source docs synthesized into [`../CPT_MASTER_20260526.md`](../CPT_MASTER_20260526.md). Each retains its original content; the master synthesis abridges them into a single reference.

- `old_experiments_plan.md` — v0.12 (2026-05-12). Experimental-design parent plan.
- `cpt_plan.md` — v0.7 (2026-05-20). CPT-execution successor.
- `cpt_plan_v0.7_answers.md` — decision snapshot at bakeoff firing.
- `cpt_plan_v0.7_status.md` — V1-V16 verification status.
- `apertus_fidelity_checklist.md` — Apertus-pretraining fidelity items + production gates.
- `PLAN_VS_RESULTS_RECONCILIATION_20260526.md` — plan-vs-results reconciliation + 14-entry discrepancy log.
- `PRODUCTION_DECISION_STATE.md` — 2 B-stage production decision (carries 2026-05-26 banner about 5 B partial supersession).
- `ARTIFACTS_AND_HYDRATION.md` — repo ownership policy + hydration check.
- `CLARIDEN_INVENTORY_20260524.md` — Clariden filesystem map.
- `collegues_Apertus_plan.md` — original Greek-language project framing by p-skarvelis.

## `2026-05-24_2B_bakeoff_review/`

Reviewer-facing material produced after the 2 B bakeoff completed. Pre-3.5 B
and pre-5 B continuation; the conclusions in these docs (Vanilla wins as safe
default) were reversed by the 3.5 B → 5 B trajectory on downstream aggregates.
Kept for the audit / risk-inventory traceability.

- `REVIEW_HANDOFF_20260524.md` — single-map handoff doc for an external
  reviewer.
- `REVIEW_PRESENTATION.md` — narrative entry-point covering recipe + sbatch +
  eval + risk inventory.
- `AUDIT_FINDINGS.md` — 2-round source-vs-implementation audit (self-audit
  + colleague reviewer round-2).
- `COMPLETENESS_CHECK.md` — completeness inventory against `cpt_plan.md` v0.7
  at 2026-05-21.
