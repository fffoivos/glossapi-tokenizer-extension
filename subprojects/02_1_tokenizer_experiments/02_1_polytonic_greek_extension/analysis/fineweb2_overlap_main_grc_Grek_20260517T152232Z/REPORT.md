# FineWeb-2 Ancient Greek Overlap Check

This run compares our post-dedup polytonic Greek corpus against the
FineWeb-2 `grc_Grek` train parquet whose current distribution row is
`28,539` documents / `33,850,484` words / `340.80MB`.

Note: that scale matches the current FineWeb-2 distribution CSV. The
older HF `v2.0.1` README table lists a smaller `grc_Grek` snapshot:
`10,500` documents / `9,397,616` words.

## Inputs

- Our kept-text parquet:
  `polytonic_greek_training_kept_strict_w050_c010_20260517T131514Z.parquet`
- FineWeb-2 reference:
  `fineweb2_main_grc_Grek_train_000_00000.parquet`
- Cross-dedup run:
  `cross_dedup_fineweb2_main_grc_Grek_20260517T152232Z`
- Fragment run:
  `fragment_containment_main_grc_Grek_20260517T154853Z`

## Step 1: Cross-Dedup

The established `glossapi_corpus_cli dedup-text run` pipeline was run on
the tokenizer-extension instance with Greek diacritics preserved and
MinHash threshold `0.85`.

Results:

- total decisions: `47,265`
- kept: `46,728`
- dropped: `537`
- strict exact duplicates: `0`
- relaxed exact duplicate rows: `2`, dropping `1`
- near-duplicate candidate pairs: `585`
- near-duplicate drops: `536`

FineWeb/source overlap:

- cross-source families: `424`
- FineWeb + our-source families: `423`
- FineWeb + our-source rows: `901`

Rows from our sources found in FineWeb-overlap families:

- `Wikisource_Greek_texts`: `258` rows in `258` families
- `klasikh_arx_ell_grammateia`: `142` rows in `142` families
- `1000_prwta_xronia_ellhnikhs`: `37` rows in `37` families
- `Ekklisiastika_Keimena`: `3` rows in `3` families

The dedup runner selected FineWeb as the kept representative in these
cross-corpus near-duplicate cases, so these counts should be read as
"same-sized or near-same-sized duplicate evidence", not as the final
training removal policy.

## Step 2: Metadata

Our URL metadata is sparse:

- Our docs with URL-like metadata: `3,435`
- Exact canonical URL matches with FineWeb: `210`
- The visible URL metadata is entirely `el.wikisource.org`.

FineWeb `grc_Grek` is not only a Wikisource/Perseus corpus. Its source
mass by domain bucket is:

- `perseus`: `8,393`
- `biblical_church`: `7,028`
- `other`: `6,034`
- `classical_other`: `5,442`
- `wikisource`: `1,261`
- `goarch`: `381`

So the extra FineWeb document count is partly fragmentation of sources
we also have, and partly genuinely different web sources.

## Step 3: Directional Fragment Mapping

Because FineWeb has many shorter web/excerpt documents and our corpus has
larger curated works, a symmetric dedup pass can miss containment. The
directional pass maps FineWeb docs onto our larger docs using sampled
8-token anchors.

Parameters:

- shingle size: `8`
- stride: `16`
- noisy-anchor document-frequency cutoff: `25`
- minimum FineWeb tokens: `80`
- minimum anchor hits: `5`
- selected containment threshold: `0.25`

Results:

- FineWeb rows scanned: `28,539`
- eligible FineWeb rows: `26,931`
- FineWeb rows with any anchor in our corpus: `7,466`
- FineWeb rows passing selected containment threshold: `1,080`

Threshold sensitivity:

- `>=0.25`: `1,495`
- `>=0.50`: `753`
- `>=0.75`: `242`
- `>=0.90`: `79`

Matches by our source at the selected threshold:

- `Wikisource_Greek_texts`: `542`
- `klasikh_arx_ell_grammateia`: `488`
- `1000_prwta_xronia_ellhnikhs`: `47`
- `scholarios_graeca_patristic`: `2`
- `Ekklisiastika_Keimena`: `1`

Top containment examples are exactly the pattern we expected:
FineWeb/CTS or Perseus URLs for short citation ranges map into our
larger Perseus/classical documents; Wikisource page URLs map into larger
Wikisource work rows.

## Interpretation

The overlap is real, but not large enough to make our corpus redundant.

Cross-dedup finds only hundreds of same-sized/near-same-sized overlaps
out of `47,265` combined docs. Directional containment finds more, as
expected: FineWeb often breaks a larger work into citation-range or page
fragments. Even then, the selected threshold maps `1,080` FineWeb docs
into our corpus, mostly Wikisource and classical/Perseus material.

Scholarios remains almost entirely novel relative to this FineWeb
`grc_Grek` reference by both symmetric dedup and directional containment:
only `2` FineWeb docs mapped to Scholarios in the fragment pass, and no
Scholarios rows were dropped in the cross-dedup pass.

## Practical Next Step

Before tokenizer training, apply two separate filters:

1. Hygiene filter for clear bad rows already identified separately:
   empty/one-character rows, RTF/control payloads, and very short rows
   unless explicitly retained.
2. FineWeb-overlap policy:
   - remove or downweight our rows that are same-sized near duplicates
     of FineWeb (`423` FineWeb + our-source families);
   - for fragment containment, do not automatically drop whole large
     works just because a FineWeb excerpt maps into them. Instead, use
     the containment report as decontamination evidence and decide
     whether to keep the larger work, split it, or remove only heavily
     covered sections.
