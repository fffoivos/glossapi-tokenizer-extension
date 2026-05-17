# Set aside — my Phase 3/4 synthesis

These four docs were drafted by me (Claude) in one pass after writing
the Phase 1 and Phase 2 evidence. User feedback (session 2026-05-17):
the synthesis should be the user's, not mine. Setting aside so the
docs don't anchor user thinking.

- `03_effective_policy.md` — proposed 15-rule synthesis of effective
  policy with `[stated]` / `[constraint]` / `[tooling gap]` /
  `[unaccounted]` tags.
- `04_rational_policy.md` — proposed necessary/accidental split of
  the 15 rules + three convergence arguments (FLORES+, peer cluster,
  HQ-20 mainstream) for a Greek budget.
- `FAIRNESS_DEFINITION.md` — proposed three operationalisations
  (weak/mid/strong) of Apertus's multilingual commitment.
- `GREEK_BUDGET.md` — proposed +5,120 added recommendation under the
  mid operationalisation.

These are usable as reference once the user has formed an independent
view. They are **not** the subproject's deliverable; the deliverable
is `../01_explicit_goals.md` and `../02_implicit_constraints.md`.

## Known errors (flagged in review 2026-05-17)

If these docs are ever revived, fix these first:

1. **FAIRNESS_DEFINITION.md, Operationalisation 2** says Greek at
   2.41 vs HQ-20 median ~1.75 is "just outside" the ±50 % tolerance
   band. **Arithmetic error**: 1.75 × 1.5 = 2.625, and 2.41 < 2.625,
   so Greek is actually *inside* the band on the upper side. Under
   Operationalisation 2 strictly interpreted, Greek already
   satisfies fairness and **no extension is required**. The
   downstream argument for +5,120 collapses if this is fixed.

2. **GREEK_BUDGET.md TL;DR** quotes "5.03 % of vocab" for the
   recommended 6,599 total Greek tokens. **Arithmetic error**: the
   correct post-extension figure is 6,599 / 136,192 ≈ 4.85 %, which
   is what the table later in the doc correctly reports. The
   5.03 % figure appears to be a mixed-baseline slip (6,599 /
   131,072 ≈ 5.04 %, against the *pre*-extension vocab denominator).

These errors don't affect the active evidence corpus
([`../01_explicit_goals.md`](../01_explicit_goals.md) and downstream)
because the synthesis is set aside. If the synthesis is revisited,
the Operationalisation 2 math error materially undermines the
+5,120 recommendation — reconsider the recommendation from scratch
rather than patching the arithmetic.
