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
