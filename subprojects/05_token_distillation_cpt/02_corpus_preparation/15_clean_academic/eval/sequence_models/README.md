# ToC/bibliography sequence-model track

Offline, CPU-only research and evaluation code. It does not change the production Rust detector.

## Audit of the interrupted draft

Retained because they were sound:

- the sparse NumPy linear-chain CRF, including forward/backward likelihood and legal BIOES Viterbi;
- the existing Rust-mirrored features plus stable, unnormalised Unicode character n-grams;
- the compact byte CharCNN → dilated line TCN → masked BIOES CRF scaffold.

Corrected or added here:

- `UNKNOWN` is an explicit unscored label. It is segmented out of CRF fitting and excluded from metrics;
  it is never converted to `O`;
- work identities cannot cross splits, exact-text duplicates cannot cross splits, and work groups are
  deterministically balanced within source strata;
- the historical LLM-labelled `STRUCT_2K` run completed and produced the surviving fitted models. If its
  ignored raw intermediates are recovered, they remain silver only. Its locked test is
  comparison-only and may not drive architecture, feature, calibration, or threshold selection;
- C0 loads the tracked LR artifacts and hysteresis constants without fitting. Overlap between its two
  heads is retained as a conflict (fail closed);
- scoring covers lines, pinned-token counts, spans, documents, true main-text retention, unknown-label
  coverage, per-source safety, and work-clustered bootstrap confidence intervals.

The tracked annotations under `eval/annotations*` are model-produced and windowed. They are never
promoted or relabelled as human gold. The exact reconstruction audit and recovery commands are in
`SILVER_RECONSTRUCTION.md`; the machine-readable receipt is `silver_inventory.json`.

The missing SPAN text batches can be rebuilt on a CPU node with `silver_reconstruct
rehydrate-span-units` from SHA-256-bound Greek PhD/OpenArchives JSONL(Zstandard) and Kallipos section
Parquet artifacts. `span_rehydration_layout.json` preserves the base/extension batch boundary, and
`hydrate-span --unit-rehydration-receipt ...` carries snapshot status into the converted silver
receipt. Recovery never reads, invents, or alters a label. Historical snapshot equivalence is verified
only when an explicit expected snapshot artifact SHA-256 matches; otherwise it is stamped
`rehydrated_unverified_snapshot` and remains comparison-only.

SPAN labels use their historical document-union semantics: all present nonblank lines from sampled
windows are merged by document, then inclusive declared spans are intersected with those coordinates.
The silver receipt exposes adjusted nonempty and zero-effective projections instead of pretending a
blank/misaligned LLM endpoint was exact. The sole unit without an annotation decision remains in the
document union as negative-only, matching the historical loader, and is explicitly ledgered rather
than silently dropped. The quarantined current MDC v1 raw archive is available only
through the explicit `mdc_raw_forensic` route, which requires and revalidates its quarantine and v2
forensic-audit receipts plus an explicit comparison-only acknowledgement. A fresh safe extraction is
compared with the retained tree on every audit, and the final consumer requires current builder/tool
hashes plus exact selected-document text and field-use digests. It never upgrades snapshot equivalence
or production eligibility.

When using the Phase-04 Clariden acquisition, first run
`sequence_models.build_span_source_receipt` against its passed receipt and exact lock. Greek PhD and
Kallipos route choices are mandatory: Nanochat document Parquet and standalone replacement artifacts
are never treated as interchangeable or silently preferred. The explicit CPU-only dry-run/execution
script is `04_full_corpus_preparation/clariden/07_rehydrate_span_silver.sbatch`; it is intentionally not
part of the corpus-production submission DAG.

## Evidence contracts

`LLM_silver` is fit/comparison evidence only. It uses immutable identities, exact-text leak checks,
work-grouped/source-balanced splits, pinned token counts, and explicit `UNKNOWN` semantics, but
does not satisfy promotion by itself. No full-corpus human annotation is required. A future application
run must freeze an approved policy before Stage10; after a candidate is frozen, it also needs Stage54 and
a separate receipt-bound manual audit of exactly 100 high-risk predicted removals (50 ToC + 50 BIB,
zero catastrophic deletions) plus all configured retention/contamination gates. Because the
silver corpus has no independent running-prose judgment, its own reports set prose
contamination, true main-text retention, and catastrophic prose deletion metrics to `null` and retain
the production no-op fallback.

## Controlled ladder

1. `c0-rust-lr-hysteresis`: frozen Python reproduction for comparison/parity only.
2. `c1-feature-bioes-crf`: existing engineered features in a real linear-chain CRF.
3. `c2-char-ngram-feature-bioes-crf`: C1 plus stable 2–5 character n-grams for OCR variation.
4. `n1-bytecnn-tcn-masked-crf`: compact character-aware CPU shadow candidate. It is a functional model
   and CPU TorchScript emission-export scaffold; receipt-bound silver may fit it for comparison only.

The deterministic choices and exact promotion gates live in `config.json`. Training is forbidden locally
and on GPUs. Operational silver fitting, once the text recovery receipt is available, belongs on a
Clariden CPU node; the separate targeted manual safety receipt remains mandatory for production.

## Commands

Run from `eval/` (examples only; do not run fitting locally). The reconstructed file remains LLM silver
even though its legacy row schema is named `academic-structure-gold-v1`:

```bash
python3 -m sequence_models.silver_reconstruct audit
python3 -m sequence_models.contract validate-silver --silver SILVER.jsonl \
  --config sequence_models/config.json --split-manifest silver.split.json
python3 -m sequence_models.contract make-split --silver SILVER.jsonl \
  --config sequence_models/config.json --output split.json
python3 -m sequence_models.baseline --silver SILVER.jsonl --output c0.jsonl
python3 -m sequence_models.evaluate --silver SILVER.jsonl --baseline c0.jsonl \
  --candidate candidate.jsonl --config sequence_models/config.json \
  --split-manifest split.json --output report.json
python3 -m sequence_models.runtime parity --left python.jsonl --right cpu_runtime.jsonl
python3 -m sequence_models.runtime benchmark-c0 --silver SILVER.jsonl
```

For C1/C2 fitting on the current SPAN evidence use `--target bib`; this hard-disables every ToC tag
instead of learning false ToC negatives from a BIB-only annotation task. Joint comparison remains
blocked unless the separate missing STRUCT_2K LLM-silver raw artifact is recovered.

On an approved Clariden CPU allocation, the feature CRF entry point is:

```bash
python3 -m sequence_models.feature_crf --silver SILVER.jsonl --config sequence_models/config.json \
  --split-manifest split.json --architecture c2-char-ngram-feature-bioes-crf --model-out c2.npz \
  --validation-predictions c2.validation.jsonl --target bib
```

Do not request test predictions until architecture and deletion bias are frozen on validation. Silver
comparison never authorizes deployment by itself. Deployment additionally requires a pre-authorized
frozen policy, Stage54, the receipt-bound 100-case high-risk false-deletion review (50 ToC + 50 BIB,
zero catastrophes), every configured deployment gate, CPU runtime parity, and the
artifact/latency/resource receipts. The current CPT run remains `audit_only` and therefore uses
Stage58's no-op path.
