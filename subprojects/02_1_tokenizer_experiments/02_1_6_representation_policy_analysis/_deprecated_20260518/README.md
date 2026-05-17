# Deprecated — 2026-05-18 cleanup

Files in this folder were created during the policy-analysis effort
but are now superseded. The actual C3 cutoff decision was made
empirically in
[`../../02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](../../02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md)
on 2026-05-18 (chosen cutoff = **17,408 added units, curated +
backfilled**), using the TokEval-on-multi-metric-in-domain sweep
rather than the policy-archaeology + Gini-optimization path that
this folder represents.

Kept here (not deleted) for traceability and because some content is
still informative as a record of what was considered.

## Contents

### `synthesis_with_known_errors/` (was `_my_synthesis_set_aside/`)

Five files from my Phase 3/4 synthesis attempt:
- `03_effective_policy.md` — proposed 15-rule effective-policy synthesis
- `04_rational_policy.md` — necessary/accidental split + 3 convergence arguments
- `FAIRNESS_DEFINITION.md` — three operationalisations
- `GREEK_BUDGET.md` — proposed +5,120 recommendation
- `README.md` — set-aside rationale + known errors

**Two known arithmetic errors** in `FAIRNESS_DEFINITION.md` and
`GREEK_BUDGET.md` documented in `synthesis_with_known_errors/README.md`.
The recommendation was +5,120; the actual chosen cutoff is +17,408,
so this synthesis was both internally buggy AND wrong about the
budget. Do not revive without re-doing from scratch.

### `05_evidence_pre2024_datasets.md` through `10_evidence_reddit_proxy.md`

Six stub investigations for open hypotheses about what produced
Mistral's per-language vocab allocation:
- `05` — pre-2024 dataset landscape (OSCAR / mC4 / CulturaX / etc.)
- `06` — Mistral inherits from prior multilingual models (XLM-R, mT5, BLOOM)
- `07` — Mistral French team data bias
- `08` — commercial market footprint
- `09` — multilingual benchmark coverage
- `10` — Reddit per-language footprint

None of these were ever executed. **Deprioritized after 2026-05-18**
because the C3 cutoff decision was made empirically via the TokEval
sweep in `02_1_7`, which does not require resolving Mistral's
historical tokenizer-training policy. The hypotheses remain
academically interesting but have no decision-relevant value for
the C3 cutoff anymore.

If the user later needs to defend the cutoff against questions about
Mistral's per-language allocation policy, these stubs are still a
reasonable starting point for the investigation — they were
scoped, sourced, and ready to execute.

## What's still in the active tree

The active evidence layer in the parent directory is unchanged:

- `01_explicit_goals.md` — Phase 1 quote harvest
- `02_implicit_constraints.md` — Phase 2 structural constraints
- `03_evidence_HPLT3_FLORES_classical.md` — HPLT 3.0 baseline + FLORES+ + classical-language asymmetry
- `04_evidence_speakers.md` — speaker-count hypothesis (INVALIDATED)
- `11_tokenizer_provenance.md` — Mistral inheritance + Apertus's changes
- `12_gini_optimization.md` — Gini-on-FLORES+ experiment plan (now marked SUPERSEDED by 02_1_7 results)

Plus the standard navigation docs: `README.md`, `TODO.md`,
`INVESTIGATIONS_TRACKER.md`, `REVIEW_INTEGRATION_20260517.md`.
