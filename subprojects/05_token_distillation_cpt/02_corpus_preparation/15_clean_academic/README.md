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
  src/reference_module.rs    detect_doc (whole-doc) + detect_sections (β gate); split counters; tests
  src/main.rs                reference_detect CLI — zstd/JSONL streaming, rayon batches
driver/run_reference_detect.py   thin I/O driver (parquet→grouped-jsonl | jsonl.zst→binary). No per-doc text work in Python.
review/sample_refspans.py        review sampler — full-doc, post-cleaner, inline <match kind=…>, stratified, metric-prefixed
investigation/                   multi-agent synthesis (_raw_synthesis.json) + section-labelled samples + example render
out/                             spans + counters per source (gitignored scratch)
```

## Run

```bash
PY=/home/foivos/Projects/glossapi-tokenizer-extension/.venv-hplt-review/bin/python
cargo build --release --manifest-path reference_detector/Cargo.toml

# section sources (parquet, β label available)
$PY driver/run_reference_detect.py --source kallipos --mode sections \
    --input /home/foivos/data/glossapi_raw/hf/Apothetirio_Kallipos/Dataset_Kallipos.parquet \
    --doc-col filename --out-dir out/kallipos

# whole-doc sources (jsonl.zst, header+footnote detection)
$PY driver/run_reference_detect.py --source greek_phd --mode wholedoc \
    --input '/home/foivos/data/glossapi_raw/mozilla/greek_phd/phd-theses-corpus/contents/*.jsonl.zst' \
    --out-dir out/greek_phd

# review (inline <match>, stratified on a counter)
$PY review/sample_refspans.py --source greek_phd --mode wholedoc \
    --text-input '/home/foivos/data/glossapi_raw/mozilla/greek_phd/phd-theses-corpus/contents/*.jsonl.zst' \
    --spans out/greek_phd/refspans/greek_phd.spans.jsonl \
    --counters out/greek_phd/greek_phd.counters.jsonl \
    --metric footnote_citation_only --out-dir out/greek_phd/review_samples
```

Calibration knobs (default = emit-everything, no deletion threshold): `--bib-min-year-density`,
`--min-position-fraction`, `--footnote-cite-max-greek`, `--cv-front-max-pos`, `--emit-intext-spans`,
passed through the driver via `--knob=--flag=value`.

## Status

Detector built + unit-tested (7/7) + validated on real greek_phd 006 (reproduces U+0387=983, bib
boundary at L7397) and a Kallipos section slice (gate split 1657 bib / 545 kept-non-bib). Drop-policy
thresholds, the section-classifier-on-whole-doc question, and the mojibake/Pergamos-count items are
**open for the user** (see investigation §7). Production runs at corpus scale belong on CSCS, not home.
