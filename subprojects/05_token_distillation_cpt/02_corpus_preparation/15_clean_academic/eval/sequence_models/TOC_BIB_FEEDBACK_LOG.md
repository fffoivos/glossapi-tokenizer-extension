# ToC/BIB feedback and response log

This is the durable register for Foivos's review feedback on the ToC/BIB
research strand. It records the feedback that changed the work, the response,
the evidence, and anything still open. New feedback should be appended here in
the same change that applies it, or marked open when no fix has been accepted.

Status meanings:

- **Applied**: implemented and backed by a test, artifact, or commit.
- **Partial**: a bounded part was implemented, but the broader request remains.
- **Open**: not implemented or not yet evaluated.
- **Superseded**: an implementation was deliberately removed or replaced after
  later feedback.

The current interactive feature presentation is served locally at
`http://127.0.0.1:8767/`. Generated HTML is ignored by git; its tracked build
receipts live under `results/bibliography_feature_explorer/`.

## Retrospective register

### Evaluation methodology

| ID | Feedback | Response and status | Evidence |
|---|---|---|---|
| EVAL-01 | There is no human-gold STRUCT-2K set, and a plan premised on manually annotating 2,000 lines is unrealistic. | **Applied as an evidence constraint.** Reports and runbooks identify the targets as GPT-generated LLM-silver and physically exclude the historical test partition during development. No metric is described here as human agreement. | `struct2k_handoff_lock.json`; `BIBLIOGRAPHY_V2.md` |
| EVAL-02 | Fresh evaluation documents should come from other books/articles in the same canonical sources, using the explicit source identity already attached to the lines. | **Applied.** Added the source-matched holdout workflow rather than sampling anonymous or unrelated text. | `a06d321`; `source_matched_holdout.py` |
| EVAL-03 | Foivos and Codex should both review the classifications, without the site exposing answers in a way that biases the human decision. | **Partial.** The human holdout interface was made blinded and historical texts are revealed only after the decision workflow. The full independent dual-review comparison is not complete. | `66b2d55`, `b8a2305`; open item O-02 |
| EVAL-04 | The existing learned classifiers were not convincing enough to justify removal; inspect their errors and build explicit deterministic evidence one feature at a time. | **Applied to research direction, still open for final policy.** The deterministic vector and feature explorer were separated from the learned line classifiers. Neither is currently authorized as a corpus-removal policy. | `2c21586`, `94a4a58`; open items O-02/O-03 |
| EVAL-05 | The sampled review documents did not provide enough Greek-language bibliographies; find a document with an actual Greek bibliography and score it directly. | **Applied.** Selected an OpenArchives document with explicit `Βιβλιογραφία` and `Ελληνόγλωσση` headings, 33 Greek references, preceding Greek prose, and a following foreign bibliography. Built a one-document explorer and a section-boundary score audit. | `GREEK_BIBLIOGRAPHY_FOCUS_V7.md`; `greek_bibliography_focus_v7_build.receipt.json`; focused site on port 8768 |

### Review workflow and presentation

| ID | Feedback | Response and status | Evidence |
|---|---|---|---|
| UX-01 | “Save and next” made the reviewer go back to reveal the comparison; navigation also failed to position the target line conveniently. | **Applied.** Review navigation and target positioning were revised. | `54e69bb` |
| UX-02 | The final review needed a table of choices with the ability to reopen the exact line. | **Applied.** Added the holdout-review choices table. | `4d65ec0` |
| UX-03 | Replace the slow form with four arrow decisions; show the line, document position, sequential list, predictions/agreement after a decision, and resume at the first undecided item. | **Applied.** Replaced the form with the rapid arrow-review feed and persistent progress. | `58e92d6` |
| UX-04 | Keep arrow labels visible, make the controls resemble the arrow directions, and place them on the right. | **Applied.** Moved the labelled controls into a right-side D-pad. | `d786fe4` |
| UX-05 | Show at least five lines of context, symmetrically distributed, with a distinct background on the classified line. | **Applied.** Added symmetric two-before/two-after context around the focus line. | `be30ea4` |
| UX-06 | For deterministic features, present the top-scoring lines with colour-coded feature boxes, a colour legend, and switches that remove features from both scoring and ranking. | **Applied, later extended.** Added the interactive unweighted feature explorer; later builds rank all lines rather than truncating to 100. | `94a4a58`, `6a3ec6d` |
| UX-07 | Boxes must show the actual detected characters; expose match offsets, and consider score normalization by line length. | **Applied.** Added exact normalized-text character offsets, count/span parity checks, match density, point density, and matched-character coverage. | `8def233`, `3c251f6` |
| UX-08 | Dense overlapping boxes made the text unreadable; hovering a feature label should isolate only that feature. | **Applied.** Feature badges and sidebar labels now spotlight one feature without changing ranking. | `9faee1e`, `9f71c1a` |
| UX-09 | These small site/regex iterations did not need Clariden. | **Applied.** The explorer can rebuild from its embedded label-blind packet locally; Clariden remains an optional clean-build wrapper. | `a57a9de` |
| UX-10 | Replace the fixed top 100 with infinite scrolling in descending order. | **Applied.** All 14,815 lines are ranked globally and appended in batches of 100. | `6a3ec6d`; `infinite_v4_build.receipt.json` |

### Detector design and feature ownership

| ID | Feedback | Response and status | Evidence |
|---|---|---|---|
| DET-01 | Explore an explicit deterministic bibliography detector: authors/names, years and dates, URL/DOI, initials, ampersands, quotes, editor/translator/thesis vocabulary, journal coordinates, pages, and identifiers. | **Applied as a research detector.** Added the countable `BibliographyFeatures` vector, reason-coded scorer, and coherent block proposal layer. It remains research-only and does not authorize deletion. | `2c21586`; `BIBLIOGRAPHY_V2.md` |
| DET-02 | Single lines are not enough: bibliography/ToC evidence occurs in blocks, and an isolated weak line between strong neighbours may belong to the block. | **Applied in the v2 decoder.** Strong anchors form blocks; bounded weak gaps may be bridged only between anchors, with scope/barrier vetoes. | `bibliography_v2.py`; `c8c559c` |
| DET-03 | A generic NN for names and places might be useful before bibliography classification. | **Open as a learned feature experiment.** The current implementation uses citation-specific Unicode name shapes and a small place lexicon. A multilingual NER model has not been accepted or evaluated against fresh-source errors. | Open item O-01 below |
| DET-04 | The first deterministic presentation should give one point per detected feature on about 20 documents, without hand-tuned weights. | **Applied.** The explorer uses 20 source-balanced documents and Boolean feature points; weighted scores and decoder decisions are absent from the page. | `94a4a58`; `BIBLIOGRAPHY_FEATURE_EXPLORER.md` |
| DET-05 | Inverted authors must capture all initials; remove the author-joiner feature; keep initials to one letter or forms such as `Ph.`; do not double-count initials as dotted words; detect numbered list entries. | **Applied.** Author forms, initial atoms, list prefixes, and feature ownership were revised; the author-joiner feature was removed. | `e0e77ff` and its tests |
| DET-06 | Dotted words must exclude spans already owned by editor, volume, or other specific detectors; proper-name words must never end at a dot; audit the other regex overlaps. | **Applied.** Specific detectors suppress broad fallbacks. The v3 audit recorded zero events for the declared accidental-overlap policy on the 14,815-line packet. | `22d1dec`; `overlap_audit_v3.json` |
| DET-07 | On document `a32563c98868...`, line 2421, direct authors were being labelled as inverted; evaluate both orientations and retain the one explaining more of the line. | **Applied.** Direct and inverted hypotheses compete by author count, covered length, and earliest position. The reviewed line has seven direct and zero inverted authors. | `22d1dec` |
| DET-08 | `pp. 44:1-44:14` is a structured article/page coordinate; do not highlight only `1-44`. | **Applied.** A specific coordinate owner captures the full repeated form and suppresses the generic range fallback. | `6a3ec6d` |
| DET-09 | The author-year composite was naïve and had not been requested. | **Applied by removal.** `author_year_count` was removed from extraction, scoring, tests, and the presentation. Years and author shapes remain separate. | `6a3ec6d` |
| DET-10 | Document `47416e0142ff...`, line 1142 exposed Lithuanian/OCR-split authors; name regexes should support Romance, Slavic Latin/Cyrillic, and monotonic/polytonic Greek. | **Applied.** Name character classes are generated from Unicode categories across Latin Extended, Greek Extended, and European Cyrillic ranges; combining marks, surname particles, and narrow author-only OCR splits are supported. | `4554b15`; multilingual regression tests |
| DET-11 | `Enzymology.` and parenthesized `(eds)`/`(ed)` were invisible. | **Applied.** The residual dotted-word owner accepts longer capitalized dotted words; editor ownership includes parenthesized singular/plural forms with optional dots. | `4554b15` |
| DET-12 | Recognize `34:27-39`, undotted `pp`, and spaced edition forms such as `5 th edn.`. | **Applied.** Volume/article page coordinates support repeated and non-repeated volume forms; page markers accept optional dots; edition and editor variants have distinct owners. | `4554b15` |
| DET-13 | In the Ding citation, `vol. 35` and `no. 10` did not include their values, and `2181-2195` had regressed. | **Applied.** Labelled volume/issue spans now own their numeric values. The page-range comma veto now rejects only comma-digit continuations, not normal citation punctuation. The fix recovers 181 additional page-range lines after date exclusions. | `4255845`; `publication_coordinates_v6_build.receipt.json` |
| DET-14 | Greek `Τόμος 64, Συμπλήρωμα : 62` should also be recognized. | **Applied.** The publication-coordinate owner now covers English/Greek volume, issue, number, and supplement labels with values and optional parenthesized issues. Labelled coordinates suppress the standalone numeric-shape fallback. | `4255845`; document `cf9352e1db72...`, line 162 |
| DET-15 | Restoring comma-terminated page ranges must not turn event dates into page evidence. | **Applied during regression analysis.** Month-first ranges such as `Oct. 1-2, 1990` and `June 6-11, 1988` now belong to the named-date owner. | `4255845`; named-date regression test |

## Open items

| ID | Item | Completion condition |
|---|---|---|
| O-01 | Evaluate multilingual person/place NER as an optional feature producer. | Fresh-source evaluation shows information gain beyond explicit citation shapes without damaging OCR-heavy sources. |
| O-02 | Convert useful deterministic evidence into a final hybrid classifier/decoder policy. | Compare deterministic-only, learned-only, and hybrid arms on untouched source-held-out documents reviewed independently by Foivos and Codex. |
| O-03 | Re-evaluate weighted bibliography-v2 metrics after the feature-definition changes. | Freeze the current feature definitions, rerun train/validation reports, and archive new immutable receipts. Historical reports must not be presented as current. |
| O-04 | Repeat the same feature-ownership review for ToC-specific patterns. | Build an equivalent inspectable ToC packet, log reviewer feedback here, and add source-held-out regression cases. |

## Maintenance rule

For each new feedback item:

1. append the example and expected ownership before or alongside the code fix;
2. add an exact regression test when the feedback is mechanically testable;
3. rebuild the label-blind site if visible matches or ranking change;
4. record the artifact hash/receipt and commit;
5. record regressions introduced by an earlier fix explicitly rather than
   silently rewriting the history.
