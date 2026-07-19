# Post-repair disagreement contributors — 2026-07-19

## Scope

This analysis uses the A/B passes after both conservative repairs:

1. unanchored `FILLER` and `CONTINUATION` labels were changed to `OTHER`; and
2. header labels on non-Markdown lines were changed to `OTHER`.

Neither pass is ground truth, so the quantities below are disagreement counts,
not true classifier precision. They identify what lowers global A/B agreement
and where adjudication effort should go.

## Bibliography-membership disagreement

There are 2,068 remaining `BIB` versus `NON_BIB` disagreements among 193,718
comparable lines. They are highly concentrated:

- Greek PhD contributes 1,503 (72.68%);
- OpenArchives contributes 555 (26.84%);
- Kallipos contributes 10 (0.48%);
- the top three documents contribute 53.48%; and
- the top seven documents contribute 1,918 of 2,068, or 92.75%.

If those seven documents are omitted only as a diagnostic, the remaining
173,055 lines contain 150 disagreements: 99.91% bibliography-membership
agreement. This is not a recommendation to remove the documents. It shows that
the residual problem is localized and should be repaired in the annotations.

| Rank | Source and document | Disagreements | Global share | Main cause |
|---:|---|---:|---:|---|
| 1 | Greek PhD `be9d332ba080…` | 413 | 19.97% | Numbered scholarly footnotes distributed through body chapters. The direction switches by region: one pass calls them entries while the other calls them ordinary text. |
| 2 | Greek PhD `071463c85082…` | 353 | 17.07% | Archival and bibliographic footnotes throughout the first 70% of the thesis, between ordinary chapter headings. Both passes used Sol, so this is not a Sol-versus-Terra difference. |
| 3 | Greek PhD `2944c6c820ae…` | 340 | 16.44% | A dense footnote stretch at 20–30% of the document. Pass B calls 231 lines `ENTRY` and 107 `CONTINUATION`; pass A calls them `OTHER`. |
| 4 | OpenArchives `0bee91d555ac…` | 278 | 13.44% | Two effects: body footnotes early in the thesis and a large table/article appendix near 87–92% that one pass mistakes for a bibliography region. |
| 5 | Greek PhD `7c6b95509f09…` | 209 | 10.11% | Legal, legislative and scholarly footnotes inside body sections look like complete references. One pass labels 197 of them `ENTRY`; the other uses `OTHER`. |
| 6 | OpenArchives `de447fab30e3…` | 206 | 9.96% | Numbered legal footnotes throughout the first half, with some URLs split into percent-encoded fragments. Pass B mostly calls them entries; pass A calls them ordinary text. |
| 7 | Greek PhD `009007924306…` | 119 | 5.75% | Footnote citations occur through the body. The real `## ΒΙΒΛΙΟΓΡΑΦΙΑ` heading appears later; one pass nevertheless treats the preceding footnotes as bibliography entries. |

These are mostly good, informative documents. The dominant error is not poor
OCR or a generally unusable document. It is a semantic annotation error:
`ENTRY` was interpreted as “citation-shaped line” instead of “entry belonging
to a bibliography/list-of-references region.” Different chunk boundaries then
made the same annotation model apply that ambiguous definition differently.

The fourth document has a genuinely difficult table/article appendix, and the
sixth contains damaged URL fragments, but those extraction issues are secondary
to the footnote-versus-bibliography distinction.

## Markdown-heading disagreement

There are 1,278 remaining heading-detection disagreements. The top ten
documents account for 47.81%, so this problem is more diffuse than bibliography
membership. Every remaining candidate already has valid Markdown heading
syntax. The disagreement is whether one pass calls that line
`NON_BIB_HEADER` while the other calls it `OTHER`.

The high-contributing lines include thesis title-page fields, committee names,
contents headings, chapter headings, and visually styled labels. Examples
include `## ΜΕΛΗ ΤΡΙΜΕΛΟΥΣ ΕΠΙΤΡΟΠΗΣ`, `## Πρόλογος`, and `## Περιεχόμενα`.
This is the second direction of the known prompt/schema error: the prompt
allowed an ATX Markdown heading to remain `OTHER`.

Under the agreed contract, this is deterministic eligibility rather than a
document-quality question. Every ATX line should be header-typed; annotation or
rules should decide only `BIB_HEADER`, `BIB_SUBHEADER`, or
`NON_BIB_HEADER`.

## Filler/continuation disagreement

There are 607 remaining filler/continuation detection disagreements. The top
two documents account for 34.10%, and the top ten account for 77.92%.

The major causes are mixed:

- `2944c6c820ae…` and much of `0bee91d555ac…` inherit the footnote-region
  mistake. Once the region is correctly considered non-bibliography, its
  supposed continuations and fillers disappear.
- `00ff25a0c089…` is a short, damaged magazine/news document whose “book
  column” contains image markers, titles and fragments. One pass treats the
  column as a bibliography region; the other treats it as source content.
- `086aeb5912f4…` is the previously repaired extraction-heavy document. The
  remaining disagreements are mostly `ENTRY` versus `FILLER` or
  `CONTINUATION` inside a real bibliography region, including image markers
  and one-character fragments. These do not change binary bibliography
  membership.
- `de447fab30e3…` contains broken URL continuations attached to footnotes.
- `a6d586aecfda…` mainly disagrees about `ENTRY` versus `CONTINUATION` inside a
  real bibliography.
- `78ebbe15bb75…` contains abbreviation-table rows that one pass calls entries
  and the other calls filler.

The global filler/continuation metric therefore mixes two distinct issues:
false bibliography regions and subtype decisions inside genuine bibliography
regions. It should be reported conditionally within an accepted bibliography
region after membership adjudication.

## Recommended correction order

1. Clarify and enforce that body footnotes/endnotes are not `ENTRY`, even when
   each line independently resembles a citation. Re-adjudicate the seven
   concentrated documents with full-document/block context.
2. Complete the Markdown rule in the other direction: every ATX line must
   receive one of the three header types rather than `OTHER`.
3. Recompute bibliography membership before evaluating filler/continuation.
   Evaluate the latter only inside agreed or adjudicated bibliography regions.
4. Review the table/article appendix in `0bee91d555ac…` and the damaged book
   column in `00ff25a0c089…` separately; do not infer that all seven documents
   are low quality.

## Evidence

The sealed root is
`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/sealed_tests/bibliography_150_20260718`.

- Per-document ranking and examples:
  `47_markdown_header_repair/disagreement-contributors.json`, SHA-256
  `8af33a5f8225f157e026da269d290b29be2d717c589e54015d036828cfb9c4e5`.
- Disagreement-region profiles:
  `47_markdown_header_repair/top-bib-clusters.json`, SHA-256
  `74699f47870152933db3a74ebbf590875008f834ee31724191d17de5a6a33c7a`.
- Clariden CPU jobs: `2798561` and `2798586`, both completed successfully.
- Comparison reader: `47_markdown_header_repair/site-8c864f1/`.
