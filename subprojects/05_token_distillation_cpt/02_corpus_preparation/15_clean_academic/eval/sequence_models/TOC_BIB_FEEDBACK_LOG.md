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

The one-block silver bibliography document reader is served locally at
`http://127.0.0.1:8769/`. Its tracked receipt and handoff live under
`results/bibliography_one_block_reader/struct2k_joint_20260713t170910z/`.

## Retrospective register

### Evaluation methodology

| ID | Feedback | Response and status | Evidence |
|---|---|---|---|
| EVAL-01 | There is no human-gold STRUCT-2K set, and a plan premised on manually annotating 2,000 lines is unrealistic. | **Applied as an evidence constraint.** Reports and runbooks identify the targets as GPT-generated LLM-silver and physically exclude the historical test partition during development. No metric is described here as human agreement. | `struct2k_handoff_lock.json`; `BIBLIOGRAPHY_V2.md` |
| EVAL-02 | Fresh evaluation documents should come from other books/articles in the same canonical sources, using the explicit source identity already attached to the lines. | **Applied.** Added the source-matched holdout workflow rather than sampling anonymous or unrelated text. | `a06d321`; `source_matched_holdout.py` |
| EVAL-03 | Foivos and Codex should both review the classifications, without the site exposing answers in a way that biases the human decision. | **Partial.** The human holdout interface was made blinded and historical texts are revealed only after the decision workflow. The full independent dual-review comparison is not complete. | `66b2d55`, `b8a2305`; open item O-02 |
| EVAL-04 | The existing learned classifiers were not convincing enough to justify removal; inspect their errors and build explicit deterministic evidence one feature at a time. | **Applied to research direction, still open for final policy.** The deterministic vector and feature explorer were separated from the learned line classifiers. Neither is currently authorized as a corpus-removal policy. | `2c21586`, `94a4a58`; open items O-02/O-03 |
| EVAL-05 | The sampled review documents did not provide enough Greek-language bibliographies; find a document with an actual Greek bibliography and score it directly. | **Applied.** Selected an OpenArchives document with explicit `Βιβλιογραφία` and `Ελληνόγλωσση` headings, 33 Greek references, preceding Greek prose, and a following foreign bibliography. Built a one-document explorer and a section-boundary score audit. | `GREEK_BIBLIOGRAPHY_FOCUS_V7.md`; `greek_bibliography_focus_v7_build.receipt.json`; focused site on port 8768 |
| EVAL-06 | Use the silver corpus only for the bibliography strand, but do not train the current entry-line detector to reproduce bibliography headers: header detection is not its present target. | **Open target-definition constraint.** Derive a bibliography-entry view rather than treating every silver `BIB` line as a positive entry. Mask bibliography headers and structural subheadings from entry-model loss and metrics; retain them separately for a later deterministic/block boundary cue. Preserve `TOC` lines as explicit hard negatives rather than positive targets. | Required before the next bibliography-only train/evaluation materialization |
| EVAL-07 | Score and preserve every available annotated silver document by its number of continuous `BIB` blocks; report the average, median, ranked tail, and plot the distribution. | **Applied to the physically test-stripped materialization.** Clariden job `2752607` audited all 1,392 available documents / 1,198,081 labelled lines without running a classifier. Mean block count is 1.435, median 1, p95 5, and maximum 19. The sealed 608-document historical test partition was physically absent and was not opened. | `results/bibliography_block_audit/struct2k_joint_20260713t164526z/`; code commit `8505ef0` |

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
| UX-11 | For the documents with exactly one silver bibliography block, show the whole document as a continuous reader: deterministic score on the right, exact match boxes over the text, feature names out of the prose on the side, a document menu, and a red mark for previously annotated `BIB` lines. | **Applied.** Clariden job `2752692` scored all 820,686 nonblank lines in the 973 qualifying documents. The reader groups the selector by source, puts feature labels in the left rail, places the unweighted feature-point score in the right rail, highlights exact feature spans, underlines silver `BIB` lines in red, and jumps to the block on load. | `4f86eef`; `results/bibliography_one_block_reader/struct2k_joint_20260713t170910z/` |

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
| DET-16 | Greek references also use fully capitalized `ΤΟΜΟΣ` and alphabetic volume numbering, including `ΤΟΜΟΣ 37:298-304` and polytonic/OCR-spaced `ΤΟΜΟΣ Α ́ (1):5`. | **Open; examples recorded without changing the detector.** Revisit the ownership and normalization rules together before adapting the volume feature. | Exact reviewed examples below |
| DET-17 | Most genuine bibliography entries fit within roughly three wrapped lines in the current presentation; substantially longer lines often remain false positives even with a high raw feature score. | **Open calibration hypothesis.** Evaluate a length-sensitive penalty or a requirement for stronger match coverage/density on long lines. Do not impose a hard character cutoff because long author lists, OCR line joining, and embedded references are legitimate exceptions. | Reviewer observation on the focused Greek document; compare `char_length`, matched-character coverage, and block context |
| DET-18 | Do not use the roughly three-line length observation to suppress the local line classifier. Apply it at a second coherence level: a long line outside a block is excluded, while a long line inside a block established by surrounding citation evidence may be included. | **Applied and evaluated.** L0–L4/D1 receive only the 35 binary/count feature families. B0 searched seed limits 280/330/380; long lines could not anchor a block but could be absorbed after independent support. B1 retained the same hard start constraint. The frozen B1 validation recovered 72.35% of true long lines with four false-positive long lines; length never entered the line model. | `BIB_LINE_TO_BLOCK_CLASSIFIER_PLAN.md`; `BIB_ENTRY_LADDER_RUN_20260713.md`; commits `9ce273d`, `7aa9f1d` |
| DET-19 | Challenge the proposed method for excluding bibliography headers from entry-line training; a mistaken automatic mask could suppress real citation examples. | **Plan corrected.** Regex and formatting rules only nominate candidates. Contextual dual Codex judgments distinguish entries, headers, subheaders, other structure, and uncertainty; disputed/uncertain rows are masked but never made negative. Header boundary cues require agreement plus a source-balanced joint audit with no observed real-entry promotion. The original silver region target remains unchanged. | `BIB_LINE_TO_BLOCK_CLASSIFIER_PLAN.md` |
| DET-20 | Header detection can be a separate stage after block detection, but header lines must still be excluded from entry-line training; verify the exclusion method before fitting. | **Implemented conservatively; joint review pending.** Clariden job `2753862` established that exact heading/subheading rules were safe in the reviewed sample while broad block-start/sparse probes were not. The fitted table therefore masked exactly 796 silver-BIB exact structural lines, never made them negatives, and left all other unadjudicated BIB lines positive. The raw region target stayed unchanged. H0 attaches exact headers only after a block exists and cannot initiate one. B1's explicit header-feature ablation lowered D1 precision and was rejected; frozen inference uses no header feature inside B1, then applies H0 afterward. Retrospective validation completed; the 120-case joint-review site is live, but review decisions are not yet collected. | `results/bibliography_header_mask_audit/struct2k_joint_20260713t195453z/ADJUDICATION_REPORT.md`; `BIB_ENTRY_LADDER_RUN_20260713.md`; jobs `2754077`, `2754368`, `2754371`, `2754381` |
| DET-21 | Bibliography evidence should form a substantial block; do not optimize for isolated citation-like lines, and use block context to retain weak or long lines inside a real region. | **Applied as the current model-development contract.** Permissive sequence proposals can recover weak/long interior lines. A train-OOF component gate then decides whether the entire proposed region is coherent. The latest extent feature rises to 32 lines and saturates, rewarding a substantial region without rewarding arbitrarily large prose merges. | `BIBLIOGRAPHY_BLOCK_FEATURE_REFERENCE.md`; component-gate schema v2 |
| DET-22 | Inspect text segments from the 50 worst-performing validation documents and remove only genuinely unusable extractions. | **Applied as an outcome-directed diagnostic.** Beginning, quartiles, end, and missed-BIB regions were inspected for all 50. Forty-four were kept and six excluded; one of the six was newly found. Together with one prediction-blind exclusion outside the worst 50, the diagnostic-qualified set contains 267/274 documents. It remains a sensitivity analysis, not an unbiased replacement for the full or prediction-blind reports. | `BIB_VALIDATION_WORST50_REVIEW_20260714.md`; `BIB_ENTRY_LADDER_RUN_20260713.md` |
| DET-23 | Use the deterministic document-structure algorithm as a complement to the learned system, and exploit bibliography coherence rather than judging marked lines as isolated hard negatives. | **Applied as a train-only sequence experiment.** The eight mutually exclusive deterministic competing roles are supplied as one-hot B1 CRF observations. They are learned in sequence and never used as hard vetoes, allowing an occasional caption/table/prose-shaped line inside a strong bibliography while helping stop repeated non-bibliography structure. Positive citation evidence remains owned by the frozen entry model. | `bibliography_entry_role_sequence.py`; `BIBLIOGRAPHY_BLOCK_FEATURE_REFERENCE.md` |

### Pending Greek volume examples

These examples were found by Foivos in the earlier review documents on
2026-07-13. They are preserved here for the next detector-design discussion;
no adaptation has yet been applied.

1. `2. Σ . Ν . Νανάς , Α . Α . Πανταζοπούλου , Α . Χαραλαμποπούλου , Α . Χ . Ράπτη , ∆ . Σ . Μουλοπούλου , Μ . Γ . Μανδηλαρά , Α . Α . Παπαδάτου , Κ . Η . Ματσούκη , Ι . Π . Χατζηγεωργίου , Σ . Α . Κοντογιάννης , Π . Ν . Αδαμόπουλος , Σ . Π . Μουλόπουλος , (1996), « Επιδημιολογία της λιποπρωτεΐνης α σε τυχαίο δείγμα 976 ενηλίκων Αθηναίων ανδρών και γυναικών », ΕΛΛΗΝΙΚΗ ΚΑΡ∆ΙΟΛΟΓΙΚΗ ΕΠΙΘΕΩΡΗΣΗ , ΤΟΜΟΣ 37:298-304`
2. `3. Μ . Γιαλαμπουκίδης , Χ . Χουμπλιός , Α . Χαραλαμποπούλου , Α . Καπετάνιου , (1998), « Περίπτωση Ελονοσίας από Πλασμώδιο Falciparum», ΙΑΤΡΙΚΑ ΑΝΑΛΕΚΤΑ ( Τριμηνιαία Έκδοση του θεραπευτηρίου ΥΓΕΙΑ ), ΤΟΜΟΣ Α ́ (1):5 Ιούνιος -Σεπτέμβριος 1998`

## Open items

| ID | Item | Completion condition |
|---|---|---|
| O-01 | Evaluate multilingual person/place NER as an optional feature producer. | Fresh-source evaluation shows information gain beyond explicit citation shapes without damaging OCR-heavy sources. |
| O-02 | Convert useful deterministic evidence into a final hybrid classifier/decoder policy. | Compare deterministic-only, learned-only, and hybrid arms on untouched source-held-out documents reviewed independently by Foivos and Codex. |
| O-03 | Re-evaluate weighted bibliography-v2 metrics after the feature-definition changes. | Freeze the current feature definitions, rerun train/validation reports, and archive new immutable receipts. Historical reports must not be presented as current. |
| O-04 | Repeat the same feature-ownership review for ToC-specific patterns. | Build an equivalent inspectable ToC packet, log reviewer feedback here, and add source-held-out regression cases. |
| O-05 | **Implementation complete; review decision pending.** The exact-mask label contract, grouped L0–L4/D1 OOF ladder, B0/H0, B1, conditional B2 gate, train-only freeze, one-time retrospective validation, and 120-case review surface are complete. The frozen result remains research-only because train OOF failed the zero-document spurious-block gate. | Foivos and Codex complete the independent review, export both records, and choose retain/revise/shadow without tuning this frozen version. See `BIB_ENTRY_LADDER_RUN_20260713.md`. |

## Maintenance rule

For each new feedback item:

1. append the example and expected ownership before or alongside the code fix;
2. add an exact regression test when the feedback is mechanically testable;
3. rebuild the label-blind site if visible matches or ranking change;
4. record the artifact hash/receipt and commit;
5. record regressions introduced by an earlier fix explicitly rather than
   silently rewriting the history.
