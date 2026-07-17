# Bibliography CONTINUATION feature research — 2026-07-17

## Conclusion

The reviewed `CONTINUATION` lines are usually **bibliographic fragments, not
weak full entries and not layout filler**. They sit between the two:

- their median line is 78 characters / 9 tokens, versus 179 / 22 for `ENTRY`
  and 3 / 0 for `FILLER`;
- they contain recognisable citation evidence, but usually too little of the
  complete author/year/title/container pattern to pass the frozen entry model;
- joining to an adjacent line often restores evidence, especially the number
  of distinct bibliography feature families;
- their physical form is source-dependent: ordinary wrapped citation text in
  Greek PhDs, padded Markdown table rows in Kallipos, and short OCR fragments
  in OpenArchives.

The current separation of full-entry detection from continuation detection is
therefore correct. The part to reconsider is the **first connector gate**,
which currently asks one model to find both continuation and filler before the
easy continuation-versus-filler subtype head runs. Those two populations have
opposite line shapes, and the shared front gate remains the weak component.

No classifier, threshold, or block decoder was changed by this audit.

## Frozen evidence used

Clariden job `2781676` audited:

- 535 trusted `CONTINUATION` lines;
- 278 trusted `FILLER` lines;
- 5,164 trusted `ENTRY` lines and the other reviewed role classes for
  descriptive comparisons;
- the unchanged P0D entry probabilities;
- the unchanged 177-feature connector table;
- the unchanged grouped out-of-fold connector models.

The full, provenance-bound output is at:

`results/bibliography_role_pipeline/20260717/continuation_feature_audit_a47ca5c_r1/`

Validation was not opened. The audit rejoined candidate rows only to the
manifest-declared train split and verified both the P0D and source JSONL
hashes before reporting.

## What features continuation lines actually contain

The aggregate feature families overlap; these are not mutually exclusive
subtypes.

| observed evidence | continuation lines | share |
|---|---:|---:|
| author or proper-name feature | 346 | 64.7% |
| page or volume feature | 185 | 34.6% |
| deterministic table-row feature | 138 | 25.8% |
| URL or DOI feature | 76 | 14.2% |
| starts with a lowercase letter | 68 | 12.7% |
| previous line ends with opening punctuation | 81 | 15.1% |
| at most 40 characters | 141 | 26.4% |
| at most 3 characters | 41 | 7.7% |

The most common individual deterministic features are:

| feature | continuation | entry | filler | ordinary/other |
|---|---:|---:|---:|---:|
| proper-name word | 62.1% | 97.0% | 5.4% | 51.4% |
| punctuation pattern | 53.8% | 70.6% | 39.9% | 68.9% |
| year | 27.1% | 70.7% | 0.0% | 15.3% |
| table row | 25.8% | 26.8% | 35.6% | 17.8% |
| dotted word | 25.4% | 40.9% | 1.4% | 10.1% |
| page range | 21.7% | 42.2% | 0.0% | 5.9% |
| initial | 16.3% | 66.4% | 1.4% | 14.7% |
| page marker | 13.8% | 25.4% | 0.4% | 1.8% |
| URL | 8.8% | 7.5% | 0.0% | 1.8% |
| volume shape | 7.7% | 11.0% | 0.0% | 0.6% |
| DOI | 5.4% | 1.2% | 0.0% | 0.0% |

This is the expected signature of a citation fragment: names, dates, journal
abbreviations, locators, and links are present, but fewer complete feature
families co-occur than on a full entry. DOI and URL lines are particularly
important counterexamples to any assumption that continuation must resemble
prose.

## What the entry model sees

P0D is deliberately an `ENTRY` anchor model, not a generic bibliography-region
model.

| role | mean P0D | median P0D | share at P0D >= 0.25 |
|---|---:|---:|---:|
| entry | 0.3989 | 0.2105 | 47.7% |
| continuation | 0.0602 | 0.0039 | 7.5% |
| filler | 0.0004 | 0.0003 | 0.0% |
| ordinary/other | 0.0476 | 0.0005 | 6.5% |

Only 40/535 continuation lines cross the entry threshold on their own. That
is not continuation-model recall and should not be interpreted as a failure:
the entry model was intentionally trained only on good entry anchors. It does
show why the block pipeline cannot recover continuations by merely lowering
the entry threshold.

## Joining adjacent lines

- 167/535 continuations (31.2%) are below P0D 0.25 alone but reach it after
  concatenation with at least one adjacent line;
- 138 are rescued by the previous-line join, 94 by the next-line join, and 65
  by both;
- 122/535 (22.8%) gain at least 0.10 over the stronger of the two unjoined
  lines;
- the previous join adds a mean 0.877 distinct feature families on
  continuations versus 0.025 on fillers;
- the next join adds 0.626 versus 0.086.

The distinction between *joined probability* and *probability gain* matters.
A joined line can cross 0.25 merely because its neighbour was already a strong
entry. The most trustworthy continuation evidence is therefore:

1. probability gain over the stronger individual line;
2. distinct-feature gain;
3. reduction in unmatched-character coverage;
4. directional shape compatibility, such as an open previous line, indentation
   continuity, or compatible script/length.

Do not use `max(joined probability)` alone as a continuation feature.

## The continuation class has source-specific modes

| source | lines / docs | median chars / tokens | table-row | short <=40 | join-rescued | median subtype probability |
|---|---:|---:|---:|---:|---:|---:|
| Greek PhD | 209 / 12 | 67 / 9 | 0.5% | 30.1% | 26.3% | 0.9849 |
| Kallipos | 141 / 24 | 303 / 11 | 78.7% | 5.7% | 56.0% | 0.9781 |
| OpenArchives | 185 / 14 | 49 / 7 | 14.1% | 37.8% | 17.8% | 0.9777 |

Kallipos lines are often very long in characters because Markdown table cells
contain padding, despite having only 11 median tokens. Greek PhD continuations
look most like conventional citation fragments. OpenArchives contains shorter
OCR fragments and has much weaker entry probabilities: mean 0.0124, versus
0.0684 for Greek PhD and 0.1109 for Kallipos.

The class remains document-concentrated: it spans 50 documents, but the five
largest contribute 278/535 lines (52.0%). Document-grouped folds are necessary
but not sufficient; model selection should also report complete-source
holdouts.

## What the existing continuation subtype model uses

The current continuation-versus-filler subtype head is strong on train OOF:

- pooled OOF PR-AUC: **0.970882**;
- fold-weighted OOF PR-AUC: **0.959594**;
- median true-continuation probability: **0.9812**;
- 10th percentile: **0.4363**.

The pooled and fold-weighted figures differ because document folds have
different sizes and difficulty. The fold-weighted figure is the conservative
one used for permutation analysis.

| permuted feature group | features | continuation PR-AUC drop |
|---|---:|---:|
| current-line shape | 34 | 0.117289 |
| unmatched-character geometry | 7 | 0.021406 |
| adjacent-line shape pairs | 18 | 0.005163 |
| joined-line entry gains | 8 | 0.003572 |
| deterministic bibliography counts | 35 | 0.003224 |
| block-relative position | 2 | 0.002519 |
| entry-probability neighbourhoods | 30 | 0.001310 |
| deterministic bibliography presence | 35 | 0.000668 |
| heading probabilities | 4 | 0.000024 |
| nearest-entry-anchor fields | 4 | -0.002884 |

The strongest individual fields are token count, unmatched-prefix fraction,
whitespace/symbol/digit fractions, maximum token length, whether the previous
line ends open, and previous-join gains. Heading probabilities and the four
nearest-anchor fields are irrelevant or redundant for this *conditional*
subtype decision.

This high score has a strict interpretation: **given that a line is already a
trusted continuation-or-filler case, the model usually separates the two**.
It does not show that the system can find continuation lines among ordinary
text. The shared connector-versus-nonconnector front gate remains much weaker
at pooled train-OOF PR-AUC 0.6453.

## Recommended continuation feature contract

Use a compact, dedicated continuation-evidence head over nearby non-entry
candidates. Keep:

1. **Citation-fragment coverage**
   - distinct matched feature-family count;
   - matched and unmatched character fractions, especially unmatched prefix
     and suffix;
   - compact family indicators for names/authors, year/date, page/volume,
     URL/DOI, container/publisher, and table-row evidence.
2. **Line shape**
   - token and character length separately;
   - letter, digit, punctuation, symbol, and whitespace fractions;
   - starts/ends/open punctuation, indentation, script mix, and table padding.
3. **Directional join evidence**
   - previous and next probability gain over the stronger unjoined line;
   - distinct-feature and unmatched-coverage gain;
   - pairwise length, indentation, script, bracket, and quote compatibility.
4. **Ordered local context**
   - immediate and short-radius entry/continuation probabilities;
   - whether supported evidence exists on both sides;
   - relative location between supported blocks.

Remove from this head:

- heading probabilities, which belong in the boundary/block stage;
- redundant nearest-anchor fields when richer ordered windows are retained;
- duplicated per-feature presence and count fields that contribute no
  incremental information after compact family/coverage aggregates.

Because Kallipos table continuations are a genuinely different geometry, use
either explicit interactions with a table-row mode or two small experts
(`citation_fragment` and `table_fragment`) whose probabilities feed the same
block decoder. Do not encode source identity.

## Architecture implication and next experiment

The current cascade is:

`connector (continuation + filler) -> continuation versus filler`

The evidence suggests comparing it to:

`continuation evidence -> filler/context evidence -> block decoder`

The continuation head should be evaluated against nearby `OTHER`, heading,
and filler lines, while full `ENTRY` remains owned by P0D. Filler should remain
an interior/context state rather than forcing it to share a global positive
class with continuation.

Run the following ablation under identical document-grouped outer folds and
complete-source holdouts:

1. current 177-feature connector cascade;
2. compact continuation head versus nearby non-entry candidates;
3. compact head plus directional join features;
4. table-aware mixture/interaction arm;
5. the best line head inside the unchanged ordered block decoder.

Use document-balanced weights and report both micro and document-macro
precision/recall. Keep validation closed until the feature contract and
architecture are frozen.

Körner's line-based reference segmentation uses separate `B-REF` and `I-REF`
states and learns ordered transitions over text/layout features. That is the
closest external analogue to the full-entry/continuation distinction measured
here: [Reference String Extraction Using Line-Based Conditional Random
Fields](https://arxiv.org/abs/1705.08154).
