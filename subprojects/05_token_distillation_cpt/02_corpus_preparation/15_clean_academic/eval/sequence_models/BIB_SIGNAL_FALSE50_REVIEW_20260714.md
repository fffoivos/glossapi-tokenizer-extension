# Signal-block apparent-false review — 2026-07-14

## Scope

This is a Codex first-pass review of the 50 train-OOF predicted components with
the most silver-non-BIB tokens under `signal_blocks_r1`'s highest-recall
diagnostic configuration.  It is outcome-directed diagnosis, not a replacement
validation set and not human gold.  Foivos has not yet adjudicated these cases.

The immutable full-context packet is:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/signal_block_review_r1/false_block_contexts.json
job 2758181
```

Each packet case contains the complete predicted component plus eight lines of
context on each side, silver labels, contextual TCN probabilities, document
identity, and source.  Validation rows are absent.

## Finding

The raw silver false-positive count is not an adequate correctness target for
this stage:

| First-pass class | Cases | Meaning |
|---|---:|---|
| clear silver bibliography omission | 12 | The predicted region is plainly a numbered reference/bibliography list, but silver marks it non-BIB. |
| mostly correct block with boundary overrun | 21 | The model finds a real silver or visually clear reference region but includes a small prose/header/footer tail or bridges two nearby regions. |
| genuine non-bibliography false block | 9 | Narrative prose, a research-summary table, or a chronological timeline is incorrectly included. |
| policy-sensitive structured list | 7 | CV publications, conference lists, endnotes, or assigned-reading/resource lists require an explicit removal-policy decision. |
| extraction/lineage quality problem | 1 | Repeated document/footer material makes the silver boundary and model span unreliable. |

Thus only 9/50 are straightforward whole-block classifier errors.  Twenty-one
are primarily boundary errors, and at least twelve are obvious silver misses.
The existing 99% silver-line-precision gate therefore rewards suppressing real
bibliographies.  It must remain reported for comparability, but it cannot be
the sole optimization or promotion criterion without a label-completeness
audit.

The clearest failure is Kallipos document `802cdb75649e...`: silver says the
entire document has zero bibliography blocks, while the model finds five long
regions containing explicit sequences such as `Δ. 1` through `Δ. 52`, with
authors, titles, places, years, volumes, and commentary.  These are genuine
annotated bibliographies.  The only other silver-zero component in this packet
is ordinary Greek prose and is a genuine model error.

## Case register

The numbers match `contexts` order in the immutable packet.

| # | Document / span | First-pass class | Short reason |
|---:|---|---|---|
| 1 | `802cdb75649e` 879–936 | silver miss | Greek annotated bibliography, numbered `Δ. 1`–`Δ. 52`. |
| 2 | `802cdb75649e` 766–828 | silver miss | Long numbered bibliography; final prose/table lines are a minor tail. |
| 3 | `802cdb75649e` 956–992 | silver miss | Explicit annotated bibliography; final line even describes the bibliographic compilation. |
| 4 | `802cdb75649e` 364–407 | silver miss | Repeated source and secondary-bibliography entries. |
| 5 | `802cdb75649e` 666–700 | silver miss | Numbered secondary bibliography with authors/titles/places/years. |
| 6 | `147168f3595e` 337–343 | genuine false | Continuous argumentative prose with inline citations. |
| 7 | `5d24e94b60c5` 218–388 | boundary overrun | Large standalone reference sequence with prose at the beginning/end. |
| 8 | `5407ffb61edd` 117–149 | genuine false | Literature-review prose with inline author-year citations. |
| 9 | `c957d1d77dfc` 330–414 | genuine false | Research-summary table, not a bibliography. |
| 10 | `085bb83188dd` 571–922 | boundary overrun | Begins in a silver bibliography and continues into later reference-table material. |
| 11 | `e049515d09b3` 297–341 | genuine false | Repeated narrative `author + year + reports` prose. |
| 12 | `3b6bd5e1ecd3` 1285–1567 | extraction issue | Silver bibliography is followed by repeated document/footer material and a new section. |
| 13 | `32bfcd9f0459` 960–1007 | genuine false | Tabular survey of studies and findings. |
| 14 | `19e6e95abb49` 783–881 | boundary overrun | Earlier bibliography-like book list joins the silver bibliography. |
| 15 | `cafadc9a0c8c` 827–886 | boundary overrun | Almost entirely silver bibliography; eleven boundary lines disagree. |
| 16 | `7e79a5234b68` 31–71 | genuine false | Historical narrative/timeline of translations. |
| 17 | `5d24e94b60c5` 450–489 | silver miss | Continuous numbered journal-reference list. |
| 18 | `4815fe0fe581` 644–676 | boundary overrun | Silver bibliography continues into an image-caption list and book summary. |
| 19 | `2b53974ec325` 973–1104 | policy-sensitive | Silver references continue into the author's publication/conference list. |
| 20 | `c315ea563703` 96–141 | policy-sensitive | CV, scientific works, and attended-conference list. |
| 21 | `8f90989ce4c8` 143–184 | genuine false | microRNA evidence table with citations. |
| 22 | `f5f273327994` 393–1121 | boundary overrun | Very long silver bibliography with only five disagreeing lines. |
| 23 | `07f29911ed9d` 41–65 | policy-sensitive | Professional resources/journal list immediately before silver references. |
| 24 | `15ce328fbdd3` 339–365 | silver miss | Numbered source/reference notes with titles and URLs. |
| 25 | `3dbb063c8e0f` 33–43 | policy-sensitive | Explicit scientific-conference publications. |
| 26 | `15ce328fbdd3` 192–212 | silver miss | Numbered source/reference notes. |
| 27 | `4d2d1999b638` 290–303 | policy-sensitive | Endnotes mixing bibliographic citations with biographical explanations. |
| 28 | `779ab2bd4e72` 496–507 | genuine false | Chronological technical-history timeline. |
| 29 | `839753694c20` 481–493 | policy-sensitive | Course reading/assignment bibliography. |
| 30 | `96ca7d75ce3d` 280–314 | boundary overrun | Two silver reference runs bridged across intervening body material. |
| 31 | `c02007fc0825` 43–75 | boundary overrun | Body prose joins the following silver references. |
| 32 | `15ce328fbdd3` 473–491 | silver miss | Numbered source/reference notes. |
| 33 | `cc620a9f3807` 816–1095 | boundary overrun | Legal footnotes bridge into the silver bibliography. |
| 34 | `15ce328fbdd3` 634–648 | silver miss | Numbered source/reference notes. |
| 35 | `60a1e961240d` 171–189 | boundary overrun | Silver bibliography followed by two engineering exercise lines. |
| 36 | `48d2aae9275a` 103–124 | boundary overrun | Footnotes/body citations lead into the silver bibliography. |
| 37 | `42f3603ce61a` 103–122 | boundary overrun | Two prose lines lead into the silver bibliography. |
| 38 | `c957d1d77dfc` 253–269 | genuine false | Research-summary table. |
| 39 | `8ee17209747f` 50–71 | policy-sensitive | CV publications/awards table. |
| 40 | `1948e3edf8fb` 396–406 | silver miss | Standalone numbered reference list before the next section. |
| 41 | `3ceab08822d9` 778–854 | boundary overrun | One prose lead and one prose tail around a silver bibliography. |
| 42 | `9338d20e16ef` 221–265 | boundary overrun | CV education lead directly before silver scientific publications. |
| 43 | `b9e0f4e77079` 608–619 | silver miss | Standalone numbered journal-reference list before a new section. |
| 44 | `8c2e6d9b9077` 92–113 | boundary overrun | Prose immediately before and after a silver bibliography. |
| 45 | `db71e7661068` 752–804 | boundary overrun | Image/author-copy line before a silver journal bibliography. |
| 46 | `c49a3a0bbb3c` 199–211 | boundary overrun | Image-credit line before a silver reference list. |
| 47 | `edb8d5e49e4b` 186–213 | boundary overrun | Two recommended-reading prose lines before silver references. |
| 48 | `961e5b4b3065` 340–399 | boundary overrun | One historical prose line before a silver bibliography. |
| 49 | `736b4d2c8e0e` 83–93 | boundary overrun | Formula before, exercise after, silver references in between. |
| 50 | `a9b3bca5ea69` 828–895 | boundary overrun | Two legal prose lines before a silver bibliography. |

## Consequences for the next experiment

1. Preserve the raw-silver metrics, but add a separately reported reviewed
   block metric.  Never silently rewrite the historical result.
2. Audit zero-BIB documents for missed bibliography regions before using
   `spurious blocks per zero-BIB document` as a hard gate.
3. Test barriers, not more positive citation regexes: generic body headings,
   table/caption scopes, and sustained low contextual probability should split
   regions before anchor bridging.
4. Keep publication lists, conference lists, endnotes, and assigned readings
   as an explicit policy stratum until Foivos decides whether CPT cleaning
   should remove them.
5. Do not open validation while this target-policy and label-completeness issue
   is unresolved.
