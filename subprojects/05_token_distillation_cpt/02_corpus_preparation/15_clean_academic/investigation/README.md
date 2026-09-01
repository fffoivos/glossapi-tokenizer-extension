# investigation — the 25-agent method study (2026-06-13)

> **In one line:** the study that decided *how* to handle references in Greek academic text, and produced the finding that reframed the whole stage.
> **Period:** 2026-06-13 (commit `056396fd`). **Status:** completed; conclusions carried forward unchanged.
> **Came from / led to:** the corpus-mix decision in [`../../../PRODUCTION_MIX_DECISION_20260612.md`](../../../PRODUCTION_MIX_DECISION_20260612.md) → this → the detector in [`../reference_detector`](../reference_detector/README.md)

## What it did

Method: profile each of the four academic sources from real samples → evaluate each published
reference-handling method against that evidence → adversarially verify the load-bearing claims →
synthesize. Raw agent output is `_raw_synthesis.json`; the decision-ready write-up is
[`_markdown_report.md`](_markdown_report.md), folded into
[`../REFERENCE_CLEANING_INVESTIGATION.md`](../REFERENCE_CLEANING_INVESTIGATION.md).

## Findings

- **Verdict: REMOVE end-matter and footnote reference mass by segmentation**, realized as
  SEGMENT + per-family COUNT + reversible sidecar + a user-controlled drop knob. STRUCTURE
  (S2ORC link, MEDITRON summary) rejected — no consumer, links unrecoverable from OCR, summary
  injection fabricates non-source Greek. MASK-to-`<ref>` rejected — new mid-prose token in an
  existing tokenizer, and it corrupts OCR'd inline digits.
- **The headline:** *"the dominant reference token sink in humanities theses is NOT the
  end-matter list but the body-interleaved footnote stream"* — doc 006: 1,019 footnote lines =
  22.4% of the document against ~4.4% end-bib; doc 086: 233 footnote lines before the header
  against 2 after. The stream is dual-content (006 fn14 is discursive Greek, fn13 beside it is
  pure Latin citation), so blanket removal destroys 21–29% genuine Greek LM signal.
- **Seven verified risks with guards**, each traceable to a specific document: the TOC row
  `## Βιβλιογραφία (σελ. 1029-1163)` would amputate 99.4% of doc 000; `β` precision is only
  ~0.85–0.90 with Pergamos front-CV `β` in 36.7% of docs; the ano-teleia is U+0387 in greek_phd
  but U+00B7 in Pergamos sections and must never be NFC-folded; the four in-text citation regimes
  are mutually exclusive per document, so counters are never summed.
- **Zero `\cite`-family commands** across 100 samples, 82 k parquet rows and 3 raw shards — the
  sci-writing substitution method has no premise in this corpus.

## Contents

`_markdown_report.md` (the synthesis), `_raw_synthesis.json` (raw multi-agent output), and
`samples/` — section-labelled source excerpts and one example inline-`<match>` render.
