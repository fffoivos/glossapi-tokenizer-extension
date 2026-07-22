# Markdown header-role repair — 2026-07-19

## Corrected contract

The intended header eligibility rule is deterministic: a line can receive
`BIB_HEADER`, `BIB_SUBHEADER`, or `NON_BIB_HEADER` only when it is an ATX
Markdown heading matching:

```regex
^\s{0,3}#{1,6}\s+\S
```

The sealed annotation prompt had incorrectly broadened `NON_BIB_HEADER` to any
inferred chapter or section heading and allowed some headings to remain
`OTHER`. This did not match the deterministic pipeline contract.

The repair requested here addresses the first direction only: every existing
header label on a non-Markdown line becomes `OTHER`. It does not yet promote a
Markdown line labelled `OTHER` to a header role.

## Immutable derivation

The repair is applied independently to the two context-repaired passes. It
does not mutate the canonical annotations or the preceding contextual repair.

- Input A: `46_context_role_repair/pass-a-47791a2/pass.context-repaired.json`.
- Input B: `46_context_role_repair/pass-b-47791a2/pass.context-repaired.json`.
- Output A:
  `47_markdown_header_repair/pass-a-8c864f1/pass.markdown-header-repaired.json`,
  SHA-256
  `f5e4310ba57a6ed12886d3d7c38cb98262c9180360b0ef487069da235d56329b`.
- Output B:
  `47_markdown_header_repair/pass-b-8c864f1/pass.markdown-header-repaired.json`,
  SHA-256
  `500f4df058886e4766df2a4a5e1a2865740564aca4342a787d9bcb177d7cb46e`.
- A audit:
  `47_markdown_header_repair/pass-a-8c864f1/changes.audit.jsonl`, SHA-256
  `7ce744e61d8fa48187948cf9cdab2f963777fa3273122cbb6052c71ff443e6d2`.
- B audit:
  `47_markdown_header_repair/pass-b-8c864f1/changes.audit.jsonl`, SHA-256
  `638e68426dfd06078553f2828c9d7e18c7987c595ea4ebffca25dfcd20925408`.

Every audit record contains the source text, line identity, old and new roles,
and reason.

## Changes

Pass A dropped 700 non-Markdown header labels across 69 documents. Pass B
dropped 708 across 71 documents. This affects 892 distinct source lines: both
passes changed 516 of them, A alone changed 184, and B alone changed 192.

| Original role | Pass A dropped | Pass B dropped |
|---|---:|---:|
| `BIB_HEADER` | 10 | 7 |
| `BIB_SUBHEADER` | 13 | 12 |
| `NON_BIB_HEADER` | 677 | 689 |

## Agreement

The table compares the untouched sealed annotations with the passes after both
the contextual-role repair and this Markdown-header repair.

| Metric | Original | Fully repaired |
|---|---:|---:|
| Bibliography / non-bibliography | 98.04% | 98.93% |
| Heading found | 85.91% | 88.22% |
| Heading type, conditional on both finding one | 99.85% | 99.85% |
| Filler / continuation found | 51.02% | 74.82% |
| Filler / continuation type | 99.26% | 99.00% |

By source, heading-found agreement changes from 89.00% to 90.97% for Greek
PhD, 92.31% to 100% for Kallipos, and 79.79% to 82.59% for OpenArchives.
Kallipos has no retained Markdown headings in either pass, so its 100% is empty
detection agreement rather than evidence about header typing.

## Remaining disagreement

The repair helps, but it does not eliminate the mismatch because it implements
only the requested deletion direction. There are still 1,278 Markdown lines
where one pass assigns a header role and the other assigns `OTHER`:

- 1,271 are `NON_BIB_HEADER` versus `OTHER`;
- seven involve `BIB_HEADER` or `BIB_SUBHEADER` versus `OTHER`;
- the misses occur across 83 documents.

These are not disagreements over whether the line has Markdown syntax: that is
now directly observable. They remain because the prompt allowed an eligible
Markdown heading to be labelled `OTHER`. The coherent next correction is to
make every ATX Markdown line header-typed and determine only which of the three
roles it receives. That requires a type rule or adjudication for lines on which
the two passes do not already agree; it should not be hidden inside the present
drop-only repair.

## Verification and reader

- Implementation commit: `8c864f1`.
- Clariden repair job `2795745`: five focused tests passed; execution completed
  in 15 seconds.
- Clariden reader job `2795758`: seven focused tests passed; build completed in
  27 seconds.
- Reader: `47_markdown_header_repair/site-8c864f1/`.
- Manifest SHA-256:
  `2c19564646c1b419d03c50b6f8e57fd354b34d723a0d35c71db0f4d23378d46a`.
- Receipt SHA-256:
  `691df38b173e66fdfb8aa36155a9292dcaa22f649a995c9e40d0792fa0b7233d`.

The reader compares the untouched original passes against the fully repaired
passes and displays the previous label beside every changed line. Visual QA was
performed at 1800 by 1200 pixels.
