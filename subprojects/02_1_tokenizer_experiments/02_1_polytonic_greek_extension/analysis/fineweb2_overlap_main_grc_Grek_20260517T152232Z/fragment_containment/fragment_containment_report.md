# FineWeb-2 Directional Fragment Containment

This maps FineWeb documents onto our larger documents using sampled
word-shingle anchors. It is directional: FineWeb is treated as the
possible fragment side, our corpus as the possible containing side.

## Parameters

- `shingle_size`: `8`
- `stride`: `16`
- `noisy_doc_frequency`: `25`
- `min_fine_tokens`: `80`
- `min_anchor_hits`: `5`
- `min_containment`: `0.25`

## Results

- FineWeb rows scanned: `28539`
- Eligible FineWeb rows: `26931`
- FineWeb rows with any anchor in our corpus: `7466`
- FineWeb rows passing the selected containment threshold: `1080`

Threshold counts:

- `>=0.25`: 1,495
- `>=0.50`: 753
- `>=0.75`: 242
- `>=0.90`: 79

Matches by our source:

- `Wikisource_Greek_texts`: 542
- `klasikh_arx_ell_grammateia`: 488
- `1000_prwta_xronia_ellhnikhs`: 47
- `scholarios_graeca_patristic`: 2
- `Ekklisiastika_Keimena`: 1

Matches by FineWeb domain:

- `perseus.tufts.edu`: 309
- `hellas.bab2min.pe.kr`: 264
- `cts.perseids.org`: 97
- `digitalathenaeus.org`: 83
- `el.wikisource.org`: 73
- `perseus.uchicago.edu`: 47
- `skuolasprint.it`: 40
- `greek-language.gr`: 21
- `anastrophe.uchicago.edu`: 17
- `bibbiaedu.it`: 17
- `sacred-texts.com`: 12
- `vcar.dev`: 10
- `el.m.wikisource.org`: 6
- `remacle.org`: 5
- `misselbrook.org.uk`: 4
- `academic-bible.com`: 3
- `enciclopedia-dacica.ro`: 3
- `bibelwissenschaft.de`: 3
- `aristotelianphilosophy.com`: 3
- `ellopos.net`: 3
- `sourceviewbible.github.io`: 3
- `goarch.org`: 2
- `nestle-aland.com`: 2
- `biblehub.com`: 2
- `blogs.exeter.ac.uk`: 2
- `users.sch.gr`: 2
- `catholiclibrary.org`: 2
- `dcc.dickinson.edu`: 2
- `cts.dh.uni-leipzig.de`: 2
- `ginoskos.com`: 2
