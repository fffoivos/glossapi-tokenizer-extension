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
- promotion test documents must represent every physical line, contain no `UNKNOWN`, be full-document,
  human-adjudicated gold, and use one pinned tokenizer revision;
- work identities cannot cross splits, exact-text duplicates cannot cross splits, and work groups are
  deterministically balanced within source strata;
- the historical LLM-labelled `STRUCT_2K` set, if recovered, is silver only. Its locked test is
  comparison-only and may not drive architecture, feature, calibration, or threshold selection;
- C0 loads the tracked LR artifacts and hysteresis constants without fitting. Overlap between its two
  heads is retained as a conflict (fail closed);
- scoring covers lines, pinned-token counts, spans, documents, true main-text retention, unknown-label
  coverage, per-source safety, and work-clustered bootstrap confidence intervals.

The tracked annotations under `eval/annotations*` are model-produced and windowed. They are not silently
promoted or relabelled as human gold. The missing `units/STRUCT_2K_gold.jsonl` therefore blocks fitting and
any new metrics in this worktree.

## Gold contract

Input is JSONL with `schema_version=academic-structure-gold-v1`. Each row supplies immutable
`document_id`, `work_id`, `representation_id`, `source`, locked `split`, annotation and tokenizer
provenance, `n_physical_lines`, `n_present_lines`, and every ordered nonblank line object:

```json
{"line_id":"work:42","abs_idx":42,"text":"...","label":"O|BIB|TOC|UNKNOWN","token_count":17,"is_running_prose":true}
```

Blank physical lines are omitted but still count toward `n_physical_lines` and absolute position. Missing
present lines are not negative examples. Development documents must include them as `UNKNOWN`; promotion test rows may
not contain them. `is_running_prose` is independently human-adjudicated (`null` for `UNKNOWN`), so running
prose contamination and true main-text retention are not approximated from the broader `O` class.

## Controlled ladder

1. `c0-rust-lr-hysteresis`: frozen Python reproduction for comparison/parity only.
2. `c1-feature-bioes-crf`: existing engineered features in a real linear-chain CRF.
3. `c2-char-ngram-feature-bioes-crf`: C1 plus stable 2–5 character n-grams for OCR variation.
4. `n1-bytecnn-tcn-masked-crf`: compact character-aware CPU shadow candidate. It is a functional model
   and CPU TorchScript emission-export scaffold, but remains unfitted while valid gold is absent.

The deterministic choices and exact promotion gates live in `config.json`. Training is forbidden locally
and on GPUs. Operational fitting, once gold exists, belongs on a Clariden CPU node.

## Commands

Run from `eval/` (examples only; do not run fitting locally):

```bash
python3 -m sequence_models.contract validate --gold GOLD.jsonl --config sequence_models/config.json
python3 -m sequence_models.contract make-split --gold GOLD.jsonl \
  --config sequence_models/config.json --output split.json
python3 -m sequence_models.baseline --gold GOLD.jsonl --output c0.jsonl
python3 -m sequence_models.evaluate --gold GOLD.jsonl --baseline c0.jsonl \
  --candidate candidate.jsonl --config sequence_models/config.json --split-manifest split.json \
  --output report.json
python3 -m sequence_models.runtime parity --left python.jsonl --right cpu_runtime.jsonl
python3 -m sequence_models.runtime benchmark-c0 --gold GOLD.jsonl
```

On an approved Clariden CPU allocation, the feature CRF entry point is:

```bash
python3 -m sequence_models.feature_crf --gold GOLD.jsonl --config sequence_models/config.json \
  --split-manifest split.json --architecture c2-char-ngram-feature-bioes-crf --model-out c2.npz \
  --validation-predictions c2.validation.jsonl
```

Do not request test predictions until architecture and deletion bias are frozen on validation. A candidate
may replace Rust only after all gates pass, CPU runtime parity passes, and the artifact/latency/resource
receipts satisfy `config.json`.
