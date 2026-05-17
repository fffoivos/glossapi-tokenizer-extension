# FineWeb-2 Ancient Greek Comparison

## Version Note

The `28,539` documents / `33,850,484` words / `340.80MB` figure matches the
current FineWeb-2 language-distribution CSV for `grc_Grek` train, not the
older `v2.0.1` README table, which lists `10,500` documents and
`9,397,616` words for `grc_Grek`.

## Scale

| Corpus | Docs | Unicode words | Greek words | UTF-8 bytes | Parquet bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb-2 `grc_Grek` published | 28,539 | 33,850,484 | n/a | 357,352,609 | 110,536,477 |
| FineWeb-2 `grc_Grek` local count | 28,539 | 30,212,242 | 28,893,344 | 357,352,609 | 110,536,477 |
| Our kept corpus | 18,726 | 83,033,398 | 82,661,596 | 1,005,567,086 | 261,901,297 |

## Strict Polytonic Signal

| Corpus | Docs passing w>=0.50/c>=0.10 | Docs with no distinctive polytonic words | Global word ratio | Global char ratio |
| --- | ---: | ---: | ---: | ---: |
| FineWeb-2 `grc_Grek` | 25,101 | 3,137 | 0.651 | 0.147 |
| Our kept corpus | 18,711 | 4 | 0.713 | 0.160 |

## Exact Normalized Overlap

Our corpus has `0` full-document
matches against FineWeb-2 after case-folding and whitespace normalization.

This does not mean there is no source overlap. FineWeb-2's top domains
include Perseus, Wikisource, GOARCH, and biblical/patristic sources, so
the overlap is likely at the work/excerpt level rather than as identical
full documents. Before training, use the existing dedup/near-dedup logic
if the goal is to suppress content Apertus probably already saw through
FineWeb-2.

## Interpretation

- Our corpus has fewer documents than FineWeb-2 current `grc_Grek`
  train (`18,726` vs `28,539`) but much more text mass by this local
  tokenizer-independent count: `83.0M` Unicode words vs `30.2M`
  locally counted FineWeb words.
- Our average document is much larger (`4,434` Unicode words/doc vs
  FineWeb's `1,059`), which fits the source shape: complete curated
  works and Scholarios pages rather than CommonCrawl web documents.
- Our strict polytonic signal is stronger: `18,711 / 18,726` docs pass
  the `word>=0.50` and `char>=0.10` check, while FineWeb current
  `grc_Grek` has `25,101 / 28,539` passing and `3,137` docs with no
  distinctive polytonic words.

## Hygiene Flags Before Training

The kept parquet still contains a few rows that should be reviewed before
constructing training shards:

- `first1k_000000` (`pseudo-Menander`, Sententiae) is empty.
- `first1k_000095` is a one-character row: `Α`.
- `ekkl_000674` (`Κηδεία`) is a large RTF/control payload, not plain
  training text.
- `15` kept rows fail the recomputed strict `word>=0.50` /
  `char>=0.10` check. Most of these are short fragmentary Wikisource or
  classical rows near the threshold, so they are review cases rather
  than automatic rejects.
- `100` rows have fewer than `20` Unicode words. These may be valid
  fragments, but they should be explicitly accepted or filtered.

## Our Rows By Source

- `scholarios_graeca_patristic`: 12,991
- `Wikisource_Greek_texts`: 3,435
- `1000_prwta_xronia_ellhnikhs`: 983
- `Ekklisiastika_Keimena`: 673
- `klasikh_arx_ell_grammateia`: 644

## FineWeb Top Domains

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
