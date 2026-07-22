# One-block bibliography document reader

This build selects every physically available STRUCT-2K silver document with
exactly one continuous `BIB` block and renders the document as a continuous
reader.

## Build

- Clariden job: `2752692` (`COMPLETED`)
- Code commit used by the job: `4f86eef`
- Input: 1,392 LLM-silver documents with the historical test partition
  physically absent
- Selected: 973 documents with exactly one continuous silver `BIB` block
- Sources: 299 `greek_phd`, 343 `kallipos`, 331 `openarchives`
- Coverage: 313 annotated windows, 660 full documents
- Scored nonblank lines: 820,686
- Silver `BIB` lines: 131,335
- Deterministic feature definitions: 35

The displayed line score is deliberately unweighted: one point for each
deterministic feature with at least one resolved span on that line. It is not a
classifier probability, a weighted bibliography-v2 score, or a block-decoder
decision.

## Reader semantics

- The document menu is grouped by canonical source.
- Feature labels are in the left rail so the document text remains continuous.
- Match boxes cover the exact normalized-text character spans.
- The integer deterministic score is in the right rail.
- A red underline means the line was already labelled `BIB` in the silver
  annotation; it is not a prediction from the deterministic feature extractor.
- Loading a document jumps to the start of its silver bibliography block.

## Locations

- Local generated reader:
  `outputs/bibliography-one-block-reader/`
- Local URL while the server is running:
  `http://127.0.0.1:8769/`
- Clariden immutable output:
  `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/struct2k_sources/struct2k_joint_20260712b/bibliography_one_block_reader_20260713t170910z/`

Restart the local server from the repository root with:

```bash
python3 -m http.server 8769 --bind 127.0.0.1 \
  --directory outputs/bibliography-one-block-reader
```

The local transfer was verified against all 973 per-document packet hashes,
the manifest hash, the HTML hash, and the aggregate packet-inventory hash in
`receipt.json`.
