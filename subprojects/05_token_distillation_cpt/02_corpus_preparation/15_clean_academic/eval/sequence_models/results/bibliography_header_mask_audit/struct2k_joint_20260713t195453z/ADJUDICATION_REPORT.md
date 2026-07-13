# Bibliography header-mask verification

## Decision

The broad candidate generator is **not** safe as a training mask. The verified
interim mask is deliberately narrower:

1. Mask exact multilingual bibliography headings/subheadings found inside
   silver `BIB` regions.
2. Mask a non-exact candidate only after contextual adjudication.
3. Leave every unreviewed non-exact candidate as an `ENTRY` positive.
4. Never turn a header, subheader, other structural line, disagreement, or
   uncertain line into an `O` negative.

This is sufficient to start the line-classifier label materialization without
using shortness, sparsity, or block position as exclusion rules. It is not a
claim that every header has already been found.

## Input and execution receipt

- Input: physically test-stripped STRUCT-2K LLM-silver materialization
- Input SHA-256:
  `f18ef6bf3061d932ae0aaeb2349392a2e590f2778e3205cf7cbcb5c79dffa7c0`
- Split scanned: `train` only
- Documents: 1,118
- Silver bibliography blocks: 1,632
- Silver `BIB` lines: 139,243
- Clariden CPU job: `2753862` (`COMPLETED`)
- Code commit: `672d804d0fd6624451c70ee846a1f2d8e328ccca`
- Sampling seed: `bibliography-header-mask-audit-v1`
- Context: target line plus up to three emitted lines on each side
- Candidate packet SHA-256:
  `f606ad2544ea5aa1b91e1654aba6a1cb6d2f5c5a18f941861ad2395e06ef5d06`

The scan nominated 15,427 diagnostic candidates:

| Stratum | Full train count | Audited sample |
|---|---:|---:|
| Exact heading | 586 | 30 |
| Exact subheading | 210 | 30 |
| First two lines of a silver block | 2,636 | 30 |
| Short/sparse internal probe | 11,995 | 30 |
| **Total** | **15,427** | **120** |

The 120-case sample was deterministic and source-stratified: 50 `greek_phd`,
25 `kallipos`, and 45 `openarchives` cases.

## Independent contextual review

Two fresh, ephemeral `gpt-5.6-terra` Codex executions reviewed the same packet
from different instructions. Reviewer B was explicitly told to treat
citation-key lines and OCR/wrapped fragments as entries and not to trust block
position or deterministic roles.

- Reviewer A response SHA-256:
  `07f3f95397d03751f49712a4a0a506705dcd7d8750cae2c5be5f46f7b1b56233`
- Reviewer B response SHA-256:
  `b2dbd846748af636797ae9a5117323232a53bc8b2a7722a2013da561cdc84af3`
- Structured-output schema SHA-256:
  `4a7f4500dd0ca524be2b5c5e54bb6b0f0d9af6caa717315d874bfce0243af769`
- Complete ID coverage: 120/120 for both reviewers
- Exact label agreement: 114/120 (95%)

Local evidence paths relative to the repository root:

- `outputs/bibliography-header-mask-audit-20260713t195453z/audit_sample.json`
- `outputs/bibliography-header-mask-audit-20260713t195453z/review_a.json`
- `outputs/bibliography-header-mask-audit-20260713t195453z/review_b.json`
- `outputs/bibliography-header-mask-audit-20260713t195453z/summary.json`
- `outputs/bibliography-header-mask-audit-20260713t195453z/receipt.json`

The exact adjudication prompts and rerun command are archived in
`ADJUDICATION_PROMPTS.md` beside this report.

| Stratum | Agreement | Agreed entries | Agreed header/subheader | Disputed/uncertain |
|---|---:|---:|---:|---:|
| Exact heading | 28/30 | 0 | 28 | 2 |
| Exact subheading | 30/30 | 0 | 30 | 0 |
| Block-start probe | 29/30 | 21 | 8 | 1 |
| Internal sparse probe | 27/30 | 27 | 0 | 3 |

For the two disputed exact-heading cases, both target lines were
`## Βιβλιογραφία` repeated inside continuing citation lists. Reviewer A called
them bibliography headers; reviewer B called them repeated structural labels.
Neither reviewer called either line an entry. Across all 60 exact-rule cases,
there were therefore zero `ENTRY` votes from either reviewer.

## What the verification rejected

- **Block start is not a header rule.** Twenty-one of 30 block-start probes
  were agreed citation entries.
- **Short/sparse is not a header rule.** At least 27 of 30 internal sparse
  probes were agreed entry continuations: DOI-only lines, URLs, years,
  publisher tails, author fragments, and OCR-wrapped text.
- **A candidate is not a mask.** The 14,631 non-exact probes remain diagnostic
  until individually adjudicated or replaced by a narrower audited rule.

The block-start sample also exposed real misses in the exact lexicon, including
numbered headings (`## 7. ΒΙΒΛΙΟΓΡΑΦΙΑ`), numbered references headings
(`## 7.5 Αναφορές`), OCR-spaced headings (`## BIB ΛΙΟΓΡΑΦΙΑ`), and category
headings (`## I. Ελληνική:`). These are recall work, not permission to use a
broad mask.

## Frozen interim policy

### Entry-line training

- Exact heading/subheading match inside a silver `BIB` region: `MASK`.
- Dual-agreed contextual `BIB_HEADER`, `BIB_SUBHEADER`, or
  `OTHER_STRUCTURE`: `MASK`.
- Reviewer disagreement or `UNCERTAIN`: `MASK`, no boundary cue.
- Dual-agreed `ENTRY`: positive entry example.
- Unreviewed non-exact probe: positive entry example.
- No masked line becomes an `O` negative.
- The original silver region labels remain unchanged.

### Block and header stages

The primary block detector is trained from entry evidence without header cues.
After it establishes a block, a separate H0 detector may attach a preceding
overall heading and mark internal subheadings/repeated running labels. H0 may
not create a block, so a false header decision cannot remove unrelated prose
by itself.

## Limits and remaining gate

This is LLM-assisted verification of LLM-silver material, not human gold. The
two reviewers used the same model family and are not statistically independent
humans. The sample verifies observed precision of the exact mask, not perfect
precision, and it demonstrates that exact-rule recall is incomplete. Before a
learned non-exact header detector is promoted, Foivos and Codex still need to
jointly inspect the high-risk cases and the frozen detector must be evaluated
by held-out document, not by randomly splitting lines.
