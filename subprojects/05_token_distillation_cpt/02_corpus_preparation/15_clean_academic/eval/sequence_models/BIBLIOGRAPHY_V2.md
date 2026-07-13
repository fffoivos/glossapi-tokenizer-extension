# Deterministic bibliography v2

`bibliography_v2.py` is an inspectable research detector built beside the
frozen R2 rules. It does not change the Rust implementation, mutate corpus
data, or authorise removal.

## Design

The line feature vector counts evidence instead of reducing it immediately to
one opaque citation flag:

- person-name shapes: Latin/Greek initials, initial sequences, inverted author
  forms, author-year forms, and title-cased name-word counts;
- publication dates: years, `n.d.` / `χ.χ.`, numeric dates, month dates, and
  access-date vocabulary;
- identifiers: URLs, DOI, ISBN, and ISSN are independent fields;
- citation typography: ampersands, author joiners, quoted spans, numbered
  entries, dotted abbreviations, and dotted abbreviation sequences;
- container/publication terms: editor, translator, thesis/dissertation,
  citation-container `in`, edition, publisher, and place-publisher forms in
  English and Greek;
- journal coordinates: volume/issue markers, year-volume sequences, page
  markers, and bare page ranges.

Every additive score contribution is emitted as a reason code. General NER is
not a dependency: the first experiment measures cheap name/place shapes. A
learned entity recogniser should be added only if held-out errors show that
these features add information beyond the citation patterns.

## Coherence

Local evidence is not itself a removal decision. The v2 decoder requires one
of:

- a formal bibliography heading confirmed by at least two citation anchors;
- a headerless cluster of at least three citation anchors.

Up to three weak or neutral lines may be bridged only *between* anchors, with a
45-token total limit and a 12-token limit for otherwise unsupported lines.
This rescues OCR wrapping such as a standalone surname or `and` inside a
bibliography while preventing the same fragment from starting a block. Long
unanchored prose, physical coverage gaps, CV/publication lists, notes, tables,
equations, and procedural legal text remain barriers or denied scopes.

## Evaluation contract

`bibliography_v2_eval.py` compares:

1. the deployed-safe R2 action policy;
2. all coherent R2 proposals;
3. v2 local evidence without context;
4. v2 coherent block proposals.

It reports BIB-specific line, token, span, document, and per-source metrics,
plus feature prevalence and inspectable false-positive/false-negative examples.
Only `train` and `validation` are accepted; a test-like split is not a CLI
choice. All current labels are GPT-generated LLM-silver, not human gold.

Run it on a Clariden CPU node with:

```bash
sbatch \
  --export=ALL,REPO_ROOT=/exact/clean/checkout,SILVER=/exact/test-stripped/struct2k.LLM_silver.jsonl,SPLIT=train,OUTPUT=/new/immutable/report.json \
  sequence_models/clariden/evaluate_bibliography_v2.sbatch
```

Use `SPLIT=train` while selecting features and thresholds. Run
`SPLIT=validation` only after those choices are frozen.

## 2026-07-13 result

The conservative detector was developed on the 1,118-document train split and
then frozen before one evaluation on the 274-document validation split. The
608-document historical test partition was physically absent and was not read.
All targets remain GPT-generated LLM-silver rather than human gold.

| Detector | Split | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| R2 all coherent proposals | train | 0.9917 | 0.1786 | 0.3026 |
| v2 balanced development iteration | train | 0.9607 | 0.3201 | 0.4802 |
| v2 conservative frozen detector | train | 0.9755 | 0.2575 | 0.4075 |
| R2 all coherent proposals | validation | 0.9970 | 0.1584 | 0.2734 |
| v2 conservative frozen detector | validation | 0.9919 | 0.2169 | 0.3560 |

The validation result is a real improvement in deterministic BIB agreement:
v2 recovers substantially more bibliography lines while retaining high
precision. It is not a production removal policy. Validation still contains 80
false-positive lines by the LLM-silver target, including genuine running
narrative that happens to contain dense author-year citations. Conversely,
some apparent false positives are citation lists or bibliographic-reference
metadata labelled `O`, illustrating that the silver target is not an
independent truth source.

The local-evidence-only v2 arm reached validation precision 0.9361, recall
0.4080, and F1 0.5683. That is evidence that the features carry useful signal,
but also evidence that line features must not be used directly: coherence
raises precision to 0.9919 at the cost of recall.

Exact archived reports, including per-source metrics, feature prevalence,
error summaries, and source-stratified examples:

- `results/bibliography_v2/train.report.json` — SHA-256
  `7cd24a17ba01e62a4863857cfe52c2334b3b6d5c3639f029d23f05be99667c08`;
- `results/bibliography_v2/validation.report.json` — SHA-256
  `1529c9e01d61919b0b14a1b9adafd2ea6974357eedc9dece4da4b2d05f2f6102`.

Clariden jobs and immutable originals:

- final train job `2747210`,
  `.../bibliography_v2/bib_v2_train_62ab462c/report.json`;
- frozen validation job `2748298`,
  `.../bibliography_v2/bib_v2_validation_62ab462c/report.json`;
- exact clean Clariden code commit
  `62ab462c91bf38acb49605f27543cd14036850f3` (the same patch series is local
  on branch `codex/toc-bib-agent2`).

## Interpretation and next use

The strongest next experiment is to add this explicit vector to the learned
line/sequence model and use deterministic evidence as features, vetoes, and
block priors. The deterministic detector by itself should remain a proposal or
agreement channel.

General person/place NER is feasible, but it should be an optional feature
producer rather than a direct bibliography decision. Citation-specific name
shapes were far more discriminative than generic title-cased words, and a
pretrained Greek/multilingual NER model may fail on initials, OCR spacing, and
non-Greek references. Evaluate it first on source-held-out lines; retain it
only if it improves precision/recall beyond the explicit name-pair features.

Useful deterministic follow-ups are:

1. distinguish taxonomic/statistical tables from bibliography tables;
2. add Greek/English narrative-verb evidence around author-year expressions;
3. type footnote shorthand such as `Βλ.`, `ό.π.`, `ibid.`, and `op. cit.`;
4. distinguish a book's front-matter ISBN/recommended-citation block from a
   bibliography;
5. expose conservative and balanced thresholds as named policies, then assess
   their hybrid agreement with the learned model on fresh-source documents.
