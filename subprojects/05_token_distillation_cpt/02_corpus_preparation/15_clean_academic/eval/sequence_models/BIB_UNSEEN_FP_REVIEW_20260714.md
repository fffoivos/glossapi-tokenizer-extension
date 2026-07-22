# Unseen-document bibliography false-positive review — 2026-07-14

## Scope and provenance

This is a precision-side review of the frozen source-matched unseen-document
packet. Foivos marked predicted lines as `WRONG` and marked one document
`WEIRD`. He did not perform a systematic false-negative review, so this report
must not be read as recall or as complete human gold.

- packet: 30 documents, 69 predicted blocks, 5,518 predicted lines;
- explicit corrections: 99 unique predicted lines in 16 documents;
- every exported key maps uniquely to a predicted packet line;
- no `SHOULD BE BIB` decisions were exported;
- one Greek-PhD document, `f5f47db6efbf...`, was marked weird; it contains
  1,229 predicted lines in 14 blocks among 3,301 displayed lines and is kept
  separate from ordinary precision summaries;
- review SHA-256:
  `13cbd3d3ffc822ab5ab6da1ad8b612f84c3d065555fa92e5ef0379873d4c04e7`;
- packet SHA-256:
  `0e396edc0804b19409662fe61f48514495120aa85e3010e14f0d64b62fac8ebe`.

The archived review is
`results/bibliography_unseen_fp_review_20260714/foivos_review.json`.

## What the 99 marked false positives are

| Failure shape | Blocks | Wrong lines | Share of marked FP lines |
|---|---:|---:|---:|
| Entire proposed block is wrong | 6 | 58 | 58.6% |
| Boundary spill in an otherwise retained block | 19 | 41 | 41.4% |
| — leading spill | 9 | 14 | 14.1% |
| — trailing spill | 11 | 27 | 27.3% |

One block has both a leading and trailing correction, hence 19 distinct
boundary-affected blocks but 20 affected edges. Boundary runs are small: mean
2.05 lines, median 2, maximum 5.

By source, the marked lines are 50 Greek-PhD lines in five documents, 11
Kallipos lines in four documents, and 38 OpenArchives lines in seven
documents. Excluding the weird document, 95 of 4,289 predictions were marked
wrong (2.2%). This is only an observed lower-bound false-positive rate because
unmarked lines were not independently adjudicated and false negatives were not
reviewed.

### Boundary spill

The header pattern is real, but it is not the whole error set.

- Eight marked leading lines in five blocks sit directly before an explicit or
  structurally clear bibliography/reference section heading. They are closing
  prose, images, code, or a stray bullet swallowed above the section.
- Fourteen marked trailing lines in eight blocks are a new
  chapter/abstract/summary/appendix boundary plus immediately adjacent text.
  Ten of those lines are literal Markdown headings.
- The other 19 boundary lines are headerless edge spill: footnotes after a
  reference list, a multiple-choice question after citations, a table/formula
  before the first entry, or a method section containing URLs after a selected
  article list.
- Four additional marked Markdown headings occur inside wholly spurious body
  blocks. In total, 14 marked lines are Markdown headings.

This aligns directly with the frozen decoder. Its configuration permits two
adjacent lines at each edge when contextual probability is at least 0.05.
Most erroneous edge runs are therefore not a mysterious learned-model error;
they are the intended two-line boundary allowance accepting structurally wrong
neighbours.

A packet-only diagnostic confirms that simply deleting this allowance is too
blunt:

| Adjacent expansion | Marked FP lines removed | Unmarked predicted lines also removed |
|---:|---:|---:|
| 2 (current) | 0 | 0 |
| 1 | 23 | 49 |
| 0 | 50 | 105 |

The unmarked lines are not guaranteed true positives, but these counts show why
the fix should be role- and position-aware rather than changing `2` to `0`.

### Entire spurious blocks

The 58 whole-block errors are highly concentrated. Fifty-six occur in two
documents:

1. `dcf172164a6e...` contributes three blocks / 36 lines of legal-historical
   body prose interleaved with footnote citations. It contains genuine citation
   features, but it is discussion and notes rather than a bibliography.
2. `701d19f7754e...` contributes one block / 20 lines under
   `## 1.2 Ιστορική Αναδρομή`. The prose repeatedly describes papers and years,
   so citation features are strong even though the lines are literature-review
   narrative.

The remaining two whole-block errors are isolated one-line proposals. Thus a
header-only patch can improve the boundaries but cannot solve the dominant
whole-block errors. Eighteen of the 99 marked lines are classified by the
deterministic line-role layer as entry-like (`STRONG_ENTRY_START`,
`WEAK_ENTRY_START`, or `POSSIBLE_CONTINUATION`), confirming that some errors
need component/context evidence rather than another citation regex.

## Likely same-pattern misses in the review

The 99 marks are a lower bound. The packet still contains several unmarked
predictions that appear to be the exact boundary error Foivos identified:

- `757103b14444...` L1693 `## ΚΕΦΑΛΑΙΟ 5`;
- the same document L3667–L3669, `## ΠΑΡΑΡΤΗΜΑ` and its opening prose;
- `114aa9f46d03...` L3399–L3403, `## Acknowledgments`, grant prose, and a page
  number immediately before `## References`;
- the same document L3720–L3722, a running title and
  `## 9. Βιογραφικό σημείωμα` after references;
- `f6667f451521...` L456, `## 4.3 ΜΕΘΟΔΟΣ ΣΥΛΛΟΓΗΣ ΔΕΔΟΜΕΝΩΝ`; its following
  five predicted lines were marked wrong, but the heading itself was not;
- `feac7897...` L277–L279, `## ΥΠΟΤΡΟΦΙΕΣ` and the first scholarship line after
  a publication/conference list.

These are not added to Foivos's labels. They are recorded as Codex review
candidates for confirmation and show why the export should not be treated as
complete gold.

## Recommended next experiment

Do not refit the line classifier yet. Test two changes separately.

### A. Edge-aware boundary decoder

1. Preserve the current independently anchored core.
2. Retain generic/internal headings when they lie between established citation
   anchors; bibliographies genuinely contain language, source-type, and author
   subheadings.
3. At the outer edge only, refuse unconditional expansion onto a generic
   non-bibliography heading, image/formula/code placeholder, or hard prose role.
4. Treat a non-bibliography heading after the final established anchor as a
   right boundary. Exclude the heading and do not expand beyond it.
5. Before an exact bibliography heading, trim only the short weak fringe that
   lacks an independently established preceding citation cluster. Do not cut
   all text before a heading: this packet contains valid publication/reference
   lists before later `References` headings.
6. Split a component at a generic heading and re-gate both sides independently.
   This handles the method-section case without turning every generic heading
   into a document-wide veto.

The clear structural family gives an immediate target of at least 22 marked
lines (eight above bibliography headings and fourteen around terminal
headings), while the full boundary target is 41 lines.

### B. Whole-component coherence gate

Evaluate, without text memorization:

- density and continuity of frozen entry evidence;
- fraction of hard prose/footnote roles;
- alternating body-prose and numbered-note structure;
- whether the component begins under an ordinary narrative section heading;
- citation-style consistency across the component; and
- a narrative-discussion role for multilingual forms such as “the article of
  X (year) argues/shows...”, tested across sources rather than encoded from the
  two reviewed documents.

The new 99-line review should be used as a diagnostic, not as the sole tuning
set. Freeze candidate rules on train OOF, then test them on another source-
balanced unseen sample. Report boundary and whole-block precision separately.

## Decision

The model is close on clean, conventional bibliographies, but the present
decoder is not ready for destructive corpus cleaning. A small edge-aware change
is well justified and should remove many chapter/header spills with limited
risk. The citation-dense prose/footnote components are a separate problem and
need a component gate plus fresh held-out review.
