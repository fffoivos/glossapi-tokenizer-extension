# Phase-04 source license adjudication

This is a technical evidence review, not legal advice. The executable,
source-by-source matrix is
[`configs/source_license_adjudication.json`](../configs/source_license_adjudication.json).
It is default-deny, pinned to the exact `sources.json` SHA-256, and records the
exact Hugging Face card URL and revision plus any upstream terms used for every
candidate. Hugging Face gating is treated only as access control.

## Decisions

The Nanochat base is allowed only for the existing noncommercial local CPT
purpose and excluded from the new public dataset: its pinned card declares
`license: other` ([card](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset/blob/e1d54136a880ed1df2ed95a5445dabd230453207/README.md)).

| Scope | Allowed source IDs |
|---|---|
| Noncommercial local CPT | `diavgeia`, `eellak_articles`, `elocus`, `libiep`, `open_council`, `opengov_deliberations_v2`, `pergamos_sections` |
| Public cleaned-text redistribution | `diavgeia`, `eellak_articles`, `open_council`, `opengov_deliberations_v2` |

The four public decisions have explicit reuse evidence: [Diavgeia terms](https://diavgeia.gov.gr/termsOfUse),
[OpenGov terms](https://www.opengov.gr/home/%CF%8C%CF%81%CE%BF%CE%B9-%CF%87%CF%81%CE%AE%CF%83%CE%B7%CF%82),
[EELLAK's pinned publisher authorization](https://huggingface.co/datasets/glossAPI/eellak-articles/blob/59fd681c483e6bdcdabe7c1a1f8685c5eebf7883/README.md),
and [OpenCouncil terms](https://opencouncil.gr/terms). Attribution,
third-party exceptions and PII conditions in the matrix still apply.

The following remain excluded from both tracks: `amna_press`, `ellakv2`,
`psepheda`, `libduth`, `ecclesia`, `archetai`, `artos_zois`, `e_nautilia`,
`national_theatre_press`, `heinrich_boell_publications`, `new_sociology`,
`school_books_new_editions`, `openbook_v2`, `kallipos_sections`,
`greek_phd_v2`, and `openarchives_current`. The exact reason is recorded per
source: ND restrictions, conflicting or absent license text, an upstream
free-access notice without a reuse grant, or missing per-item rights filters.
E-Locus, LibIEP and Pergamos are local-only because their evidence supports
noncommercial research use but not this public aggregate.

## Enforcement

- Both cleaning passes verify and hash-bind the source registry and matrix.
- The coarse category policy is only an upper bound; the source-specific matrix
  is the final machine decision.
- Materialization revalidates the matrix and writes its receipt into the release
  manifest.
- Release validation checks distinct `acquisition_source_id` values in both the
  private and public Parquet trees against the appropriate decision.
- Publication requires the same receipt and uploads the matrix as
  `provenance/source_license_adjudication.json`.

Any source revision or evidence/decision change invalidates the existing
cleaning and release receipts.
