# Human gold: operator and annotation runbook

This workflow turns the missing-gold blocker into a human task. It never calls an annotating model,
never imports existing LLM annotations as human work, and never adjudicates automatically.

## Locked sample

The sampler reads the canonical document-level Phase-04 Nanochat-base Parquet routes from
`04_full_corpus_preparation/configs/sources.json`:

| Phase-04 route | locked gold source | canonical selector |
|---|---|---|
| `greek_phd` | `greek_phd` | `source_dataset =~ ^greek_phd` |
| `openarchives` | `openarchives.gr` | `source_dataset =~ ^openarchives\.gr` |
| `kallipos` | `Apothetirio_Kallipos` | `source_dataset =~ ^Apothetirio_Kallipos` |
| `pergamos` | `Apothetirio_Pergamos` | `source_dataset =~ ^Apothetirio_Pergamos` |

Kallipos and Pergamos are already grouped into full documents in this canonical Parquet. Do not join
the raw section Parquets or reuse the older model-annotation windows. The Phase-04 acquisition receipt
binds the exact paths, revision, sizes and stat identities before any row is sampled.

Default output is 500 unique works and exact texts per source: 275 train, 75 validation and 150 locked
test documents. Sampling is label-blind and model-blind. A SHA-256 priority over source identity,
document identity and exact text selects documents in a streaming scan.

`state.sqlite` commits one Parquet row group at a time. Re-running with the same state skips completed
row groups; configuration, source registry, receipt or route drift is rejected. Packets and manifests
are write-once and content-checked on resume. `COMPLETED.json` binds the final packet and split manifests.
`annotation_assignments.jsonl` is the hash-bound, 2,720-row scheduling ledger for independent review and
adjudication; assigning people is a human project-management action and is never inferred by the code.

Each packet contains every physical line, including internal and trailing blank lines, with absolute
zero-based indices. It locks:

- canonical work, document and representation IDs;
- upstream `source_doc_id`, exact `source_dataset`, Phase-04 revision and artifact/row position;
- exact UTF-8 text SHA-256;
- source-balanced work-group split and annotation assignment;
- no model predictions, detector scores or model/LLM annotations.

## Human annotation

Independent annotation files are named `<packet_id>.<slot>.json`. A judgment range must cover every
physical line exactly once, with no gap or overlap:

```json
{
  "schema_version": "academic-structure-human-annotation-v1",
  "annotation_kind": "human_independent",
  "packet_id": "<from packet>",
  "document_id": "<from packet>",
  "text_sha256": "<from packet>",
  "annotator_id": "human-reviewer-stable-id",
  "human_attestation": "I attest that these judgments were made by a human reviewer without using model-generated labels as human evidence.",
  "judgments": [
    {"start_line": 0, "end_line": 27, "label": "O", "is_running_prose": false},
    {"start_line": 28, "end_line": 41, "label": "TOC", "is_running_prose": false},
    {"start_line": 42, "end_line": 90, "label": "O", "is_running_prose": true}
  ]
}
```

`label` and `is_running_prose` are independent judgments:

- `BIB`: reference-list/bibliography lines, including wrapped entries;
- `TOC`: table-of-contents navigation lines;
- `O`: everything retained;
- `is_running_prose=true`: substantive running main-text prose only. It is false for blank lines,
  front matter, headings, captions, tables, apparatus, bibliography and ToC lines.

Blank lines must be `O` and `is_running_prose=false`. Annotators must escalate uncertainty rather than
consult detector or model labels.

Exactly 30 of the 150 test documents per source require two independent annotations: 120/600 test
documents, exactly 20%. Their annotator IDs must differ. The remaining test documents require one
independent annotation. All 600 test documents require a separate human adjudication; no consensus is
computed automatically.

## Human adjudication

Adjudications are named `<packet_id>.json` and cite the exact annotation file hashes:

```json
{
  "schema_version": "academic-structure-human-adjudication-v1",
  "annotation_kind": "human_adjudication",
  "packet_id": "<from packet>",
  "document_id": "<from packet>",
  "text_sha256": "<from packet>",
  "adjudicator_id": "independent-human-adjudicator-id",
  "human_attestation": "I attest that these judgments were made by a human reviewer without using model-generated labels as human evidence.",
  "input_annotation_sha256": ["<exact annotation SHA-256>", "<second hash when assigned>"],
  "judgments": [
    {"start_line": 0, "end_line": 27, "label": "O", "is_running_prose": false}
  ]
}
```

The adjudicator must not be one of that packet's annotators and must provide a complete fresh decision,
even when the independent annotations agree. The importer rejects missing citations, gaps, overlaps,
identity drift, blank-line deletion labels, model attestations and any unassigned extra files.

## Clariden CPU handoff

Run only from a clean checkout of the intended commit. The wrapper prints a dry run unless
`CONFIRM_LAUNCH=1` is explicitly set.

Packet creation:

```bash
export INPUT_RECEIPT=/capstor/.../completed-acquisition.receipt.json
export GREEK_PHD_INPUT="$DATA_ROOT/hf/nanochat_base/e1d54136a880ed1df2ed95a5445dabd230453207/data/greek_phd*.parquet"
export OPENARCHIVES_INPUT="$DATA_ROOT/hf/nanochat_base/e1d54136a880ed1df2ed95a5445dabd230453207/data/openarchives.gr*.parquet"
export KALLIPOS_INPUT="$DATA_ROOT/hf/nanochat_base/e1d54136a880ed1df2ed95a5445dabd230453207/data/Apothetirio_Kallipos.parquet"
export PERGAMOS_INPUT="$DATA_ROOT/hf/nanochat_base/e1d54136a880ed1df2ed95a5445dabd230453207/data/Apothetirio_Pergamos.parquet"
bash eval/sequence_models/clariden/submit_sequence.sh packets
# Later, after reviewing the printed command:
CONFIRM_LAUNCH=1 bash eval/sequence_models/clariden/submit_sequence.sh packets
```

After human files are complete, pinned ModernGreek-148k counting and gold import:

```bash
export PACKET_ROOT=/capstor/.../human_gold_packets_...
export ANNOTATIONS=/capstor/.../human_annotations
export ADJUDICATIONS=/capstor/.../human_adjudications
bash eval/sequence_models/clariden/submit_sequence.sh import-gold
```

After the import receipt passes, validation-only C2 fitting (no test predictions):

```bash
export GOLD=/capstor/.../academic_structure_gold.jsonl
export SPLIT_MANIFEST="$PACKET_ROOT/split_manifest.json"
export ARCHITECTURE=c2-char-ngram-feature-bioes-crf
bash eval/sequence_models/clariden/submit_sequence.sh fit-feature
```

All three jobs request no GPU/GRES. The importer verifies the ModernGreek-148k tokenizer JSON against
SHA-256 `358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394` and records exact line token
counts. Fitting calibrates only on validation and does not emit locked-test predictions.

## Exact remaining human work at the default size

- Complete 2,120 independent full-document annotations: 1,400 train/validation singles, 600 first test
  annotations and 120 second test annotations.
- Complete 600 independent human adjudications for the locked test set.
- Keep reviewer IDs stable, ensure second reviewers differ, and ensure adjudicators differ from every
  annotator on their packet.

Only after those 2,720 human review/adjudication actions validate can the importer create promotion-grade
`academic-structure-gold-v1`. Model training and promotion evaluation remain separate later actions.
