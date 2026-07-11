# 15_clean_academic — reference/citation cleaning for the academic CPT sources

Net-new cleaning stage for the four academic sources of the 60 B Apertus CPT mix
(openarchives.gr, greek_phd, Apothetirio_Kallipos, Apothetirio_Pergamos). Sits between
`10_clean_hplt` (web) and `20_dedup`. The GlossAPI cleaner has no reference handling;
this stage adds it.

**Read first:** [`REFERENCE_CLEANING_INVESTIGATION.md`](REFERENCE_CLEANING_INVESTIGATION.md)
— whether the peS2o/MEDITRON/S2ORC methods translate (verdict: REMOVE-by-segmentation,
not MASK/STRUCTURE), the Greek-specific crux (the footnote stream, not the end list, is the
dominant sink), the per-source design, risks, and the open drop-policy decisions.

## Contract

DETECT → SEGMENT → EMIT auditable spans → COUNT per family. **Never** hard-delete, **never**
bake a deletion threshold. Counters are split per signal family (no aggregate). Fail-closed
(keep text on disagreement). The keep/drop policy is a separate user-driven step over the
emitted spans + counters.

## Layout

```
reference_detector/   Rust crate — the detector (hot path). cargo test; cargo build --release
  src/reference_signals.rs   regex / label / codepoint inventory (accent+homoglyph fold, U+0387≠U+00B7)
  src/reference_module.rs    detect_doc + section labels (β=bib, π=ToC); split counters; tests
  src/span_line_model.rs     promoted conservative bibliography line head
  src/toc_line_model.rs      frozen ToC line head + front gate
  src/main.rs                JSONL streaming, rayon batches, combined structure-spans mode
driver/run_reference_detect.py   thin I/O driver (parquet→grouped-jsonl | jsonl.zst→binary). No per-doc text work in Python.
review/sample_refspans.py        review sampler — full-doc, post-cleaner, inline <match kind=…>, stratified, metric-prefixed
investigation/                   multi-agent synthesis (_raw_synthesis.json) + section-labelled samples + example render
out/                             spans + counters per source (gitignored scratch)
```

## Run

Production orchestration and exact token-loss accounting live in
[`../../04_full_corpus_preparation`](../../04_full_corpus_preparation/README.md). The commands below are
component-level examples only.

```bash
cargo build --release --manifest-path reference_detector/Cargo.toml

# section sources (parquet, β label available)
python3 driver/run_reference_detect.py --source kallipos --mode sections \
    --input /path/to/Apothetirio_Kallipos/Dataset_Kallipos.parquet \
    --doc-col filename --out-dir out/kallipos

# whole-doc sources (combined promoted bibliography + ToC heads)
python3 driver/run_reference_detect.py --source greek_phd --mode structure-spans \
    --input '/path/to/greek_phd/*.jsonl.zst' \
    --out-dir out/greek_phd

# review (inline <match>, stratified on a counter)
python3 review/sample_refspans.py --source greek_phd --mode wholedoc \
    --text-input '/path/to/greek_phd/*.jsonl.zst' \
    --spans out/greek_phd/refspans/greek_phd.spans.jsonl \
    --counters out/greek_phd/greek_phd.counters.jsonl \
    --metric footnote_citation_only --out-dir out/greek_phd/review_samples
```

Calibration knobs (default = emit-everything, no deletion threshold): `--bib-min-year-density`,
`--min-position-fraction`, `--footnote-cite-max-greek`, `--cv-front-max-pos`, `--emit-intext-spans`,
passed through the driver via `--knob=--flag=value`.

## Status

As of 2026-07-11, the promoted bibliography constants/operating point and ToC head are deployed to
Rust. `structure-spans` emits both span kinds, immutable text/source provenance, separate counters and
explicit overlap/conflict mass; any malformed input row is fatal. Section mode maps approved β sections
to `bib_span` and π sections to `toc_span`, skips empty spans and has trailing-newline offset tests. The
crate has 18 passing tests. The Phase-04 suite also checks both heads across every U+0370–U+03FF edge
code point at `<1e-12` Python↔Rust probability difference.

Current evidence state:

- the surviving SPAN/Opus decisions are LLM-silver BIB supervision. Their
  coordinates are tracked and the missing source text can be rehydrated from
  receipt-bound Clariden inputs without creating labels;
- the raw joint ToC+BIB `STRUCT_2K` corpus is absent, so no joint comparison or
  Stage54 evidence receipt can be claimed from it;
- add Pergamos to a fresh source-balanced validation sample;
- run the Phase-04 exact ModernGreek-148k counterfactual token-loss audit;
- complete the independent receipt-bound review of 100 high-risk predicted
  removals (exactly 50 ToC and 50 BIB) with zero catastrophic deletions.

No new 2,000-item human annotation is required or planned. For the current CPT
run the tracked cleaning policy remains `audit_only`, both structural flags are
false, and Phase-04 Stage58 records a deterministic no-op. A future application
run would need a pre-authorized policy frozen before Stage10 plus passed Stage54
evidence; editing policy during a run is not allowed.

Corpus-scale work belongs on Clariden compute nodes. `xfer` is transfer-only.
