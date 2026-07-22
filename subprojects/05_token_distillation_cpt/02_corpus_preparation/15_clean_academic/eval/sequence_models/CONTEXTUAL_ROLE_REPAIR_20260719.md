# Contextual bibliography-role repair — 2026-07-19

## Problem and invariant

The role prompt defines `CONTINUATION` as citation content split across
physical lines and `FILLER` as non-citation material that belongs inside a
bibliography region. Neither role can exist without a bibliography entry to
which it belongs. Inspection of the sealed A/B comparison nevertheless found
large runs of `FILLER`, and a few `CONTINUATION` lines, in document regions
containing no bibliography entries.

The original prompt was consistent across the annotation runs. This is an
annotation failure to apply the contextual definition, not a missing guideline.

The repair applies one conservative invariant independently to each annotation
pass:

1. order every document using the sealed private line key;
2. form maximal contiguous components of `ENTRY`, `CONTINUATION`, `FILLER`,
   `BIB_HEADER`, and `BIB_SUBHEADER`;
3. retain every component containing at least one `ENTRY`;
4. change only `FILLER` and `CONTINUATION` in components without an `ENTRY` to
   `OTHER`;
5. never change headers, entries, unknown labels, or non-bibliography labels.

This rule deliberately preserves contextual tails in an entry-anchored
component, even when a line is distant from the nearest entry. Those tails are
profiled for later review instead of being changed using a tuned distance
threshold.

## Preservation and audit

The canonical source passes remain byte-for-byte unchanged:

- A: `26_role_sol_terra_high_a/pass.json`, SHA-256
  `9d254ed0806fcb9c83059504479bf08fe3cc19ad24a9efb04123c4040c9cc067`.
- B: `27_role_sol_terra_high_b/pass.json`, SHA-256
  `890366c3b952f599f9cb1d35adb4f5606f8581c41d306d7e6f442607bc63bb0c`.

The derived data are below the sealed root in `46_context_role_repair/`:

- repaired A: `pass-a-47791a2/pass.context-repaired.json`, SHA-256
  `d64768d782ce65aee02ad70fcc0f3c267e9e538bc259501e2630ce7e1d38ade1`;
- A audit: `pass-a-47791a2/changes.audit.jsonl`, SHA-256
  `07c171b9a431301f4b7496ceb0acb8de908be8fab8c9f3f09fa74c106a7bb5e5`;
- repaired B: `pass-b-47791a2/pass.context-repaired.json`, SHA-256
  `3001282c552c98be72e0eea3f755d4997ecc0e11ce279c30807628aebd1bf1aa`;
- B audit: `pass-b-47791a2/changes.audit.jsonl`, SHA-256
  `7eeec2057fe378b356c017ef27284d221c64a5b76c03c70690bc6683674ad7e6`.

Each audit record binds the document, row, old role, new role, and component
evidence. The repair reports `original_data_mutated: false`.

## Corrections

Pass A changed 1,040 lines across 20 documents; pass B changed 1,970 lines
across 17 documents. The 3,010 affected pass labels comprise 2,997 `FILLER`
labels and only 13 `CONTINUATION` labels.

| Source | Pass A | Pass B |
|---|---:|---:|
| Greek PhD | 233 | 78 |
| Kallipos | 99 | 15 |
| OpenArchives | 708 | 1,877 |

## Agreement before and after repair

The comparison excludes `UNKNOWN` where required by the task metrics. “Found”
measures whether both passes detected the relevant category in the union where
either pass detected it; “type” measures exact subtype agreement conditional on
both passes detecting that category.

| Metric | Original | Repaired |
|---|---:|---:|
| Bibliography / non-bibliography | 98.04% | 98.93% |
| Heading found | 85.91% | 85.91% |
| Heading type | 99.85% | 99.85% |
| Filler / continuation found | 51.02% | 74.82% |
| Filler / continuation type | 99.26% | 99.00% |

The slight conditional type decrease is expected: after removing impossible
labels, the smaller overlap contains a somewhat different set of lines. The
large detection improvement is the result relevant to the reported error.

The most visibly broken OpenArchives document,
`086aeb5912f40413cf1aaab61d55ecda816211e9f7b2a78fc9449ac8d69a48a1`,
changed by 453 labels in A and 1,173 in B. Its bibliography-membership agreement
rose from 76.06% to 100%, and its filler/continuation detection agreement rose
from 44.70% to 80.91%.

These are post-hoc annotation-audit figures. They do not retroactively pass or
replace the frozen sealed gate.

## Why “heading found” remains low

> Subsequent correction: this section described the result before enforcing
> the intended Markdown-only header eligibility rule. The drop-only repair and
> new measurements are recorded in `MARKDOWN_HEADER_ROLE_REPAIR_20260719.md`.

The contextual repair leaves heading labels unchanged, as intended. There are
1,654 heading-detection misses over 92 documents. Of these:

- 1,638 (99.0%) are `NON_BIB_HEADER` versus `NON_HEADER`;
- only 16 involve `BIB_HEADER` or `BIB_SUBHEADER` versus `NON_HEADER`;
- only seven documents contain any bibliography-header detection miss.

By source, heading-found agreement is 89.00% for Greek PhD, 92.31% for
Kallipos, and 79.79% for OpenArchives. Yet when both passes identify a heading,
heading-type agreement is 99.84%, 100%, and 99.87%, respectively.

The low figure therefore does not mean the annotators routinely confuse
bibliography headers and subheaders. It is almost entirely disagreement over
which ordinary section-like lines outside a bibliography should receive the
`NON_BIB_HEADER` hard-stop role rather than `OTHER`. Unlike an unanchored
`FILLER`, that disagreement is not a logical impossibility and must be reviewed
before another deterministic correction is defined.

## Evidence and reader

- Repair implementation commit: `47791a2`.
- Repair tests: Clariden CPU job `2795017`, 8 passed.
- Repair execution: Clariden CPU job `2795023`, completed in 11 seconds.
- Before/after reader generator commit: `ddd0f45`.
- Final reader build: Clariden CPU job `2795140`, 7 tests passed; build
  completed in 32 seconds.
- Reader: `46_context_role_repair/site-ddd0f45/`.
- Reader manifest SHA-256:
  `84a3c41292c05b7b0fb3360f7d055bc198c3d538241f075a8c14f47873669729`.
- Reader receipt SHA-256:
  `27cc21583747d99164d7fce7596573089a4532407b1eb5fa6fd9fac844597c19`.

The reader displays repaired roles side by side, marks changed labels with their
original role, and reports original-to-repaired task agreement by source and
document. Visual QA was performed at 1800 by 1200 pixels after widening the
dataset summary column to avoid overlap.
