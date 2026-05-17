# FineWeb-2 Cross-Dedup And Metadata Analysis

## Cross-Dedup

- Cross-source families: `424`
- Cross-source rows: `1000.0`
- FineWeb + our-source families: `423`
- FineWeb + our-source rows: `901.0`

Drops by source/stage:

- `1000_prwta_xronia_ellhnikhs` / `near_duplicate`: 37
- `1000_prwta_xronia_ellhnikhs` / `relaxed_exact`: 1
- `Ekklisiastika_Keimena` / `near_duplicate`: 3
- `Wikisource_Greek_texts` / `near_duplicate`: 258
- `fineweb2_main_grc_Grek` / `near_duplicate`: 96
- `klasikh_arx_ell_grammateia` / `near_duplicate`: 142

FineWeb-overlap rows by our source:

- `Wikisource_Greek_texts`: 258 rows in 258 families
- `klasikh_arx_ell_grammateia`: 142 rows in 142 families
- `1000_prwta_xronia_ellhnikhs`: 37 rows in 37 families
- `Ekklisiastika_Keimena`: 3 rows in 3 families

Drop direction:

- dropped `Wikisource_Greek_texts` kept-by `fineweb2_main_grc_Grek` via `near_duplicate`: 258
- dropped `klasikh_arx_ell_grammateia` kept-by `fineweb2_main_grc_Grek` via `near_duplicate`: 142
- dropped `fineweb2_main_grc_Grek` kept-by `fineweb2_main_grc_Grek` via `near_duplicate`: 96
- dropped `1000_prwta_xronia_ellhnikhs` kept-by `fineweb2_main_grc_Grek` via `near_duplicate`: 37
- dropped `Ekklisiastika_Keimena` kept-by `fineweb2_main_grc_Grek` via `near_duplicate`: 3
- dropped `1000_prwta_xronia_ellhnikhs` kept-by `fineweb2_main_grc_Grek` via `relaxed_exact`: 1

## Metadata

- Our docs with URL-like metadata: `3435`
- Exact canonical URL matches: `210`

FineWeb domain buckets:

- `perseus`: 8,393
- `biblical_church`: 7,028
- `other`: 6,034
- `classical_other`: 5,442
- `wikisource`: 1,261
- `goarch`: 381

FineWeb top domains:

- `perseus.tufts.edu`: 6,202
- `hellas.bab2min.pe.kr`: 3,521
- `cts.perseids.org`: 1,830
- `bibbiaedu.it`: 1,551
- `sacred-texts.com`: 1,068
- `el.wikisource.org`: 1,054
- `digitalathenaeus.org`: 983
- `skuolasprint.it`: 938
- `credobiblestudy.com`: 613
- `catholiclibrary.org`: 492
- `bibledatabase.net`: 436
- `studybible.info`: 385
- `goarch.org`: 362
- `stepbible.org`: 358
- `perseus.uchicago.edu`: 355
- `textusreceptusbibles.com`: 324
- `academic-bible.com`: 244
- `biblehub.com`: 221
- `agia-grafi.gr`: 213
- `versionidigreco.it`: 196
- `docplayer.gr`: 186
- `bibelwissenschaft.de`: 182
- `el.m.wikisource.org`: 170
- `tertios.com`: 149
- `orthodoxfathers.com`: 135
- `bible.by`: 135
- `septuagint.org`: 134
- `attikisti.com`: 127
- `inscriptions.packhum.org`: 123
- `epigraphy.packhum.org`: 108

Our metadata domains:

- `el.wikisource.org`: 3,435
