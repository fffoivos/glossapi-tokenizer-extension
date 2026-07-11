# Two-pass cleaning and structural-last contract

The production cleaner has two different passes. They are intentionally not
replays of one script.

## Stage 50: immutable post-source/post-PII corpus

`apply_cleaning_policy.py` reads the normalized Parquet tree by row group. It
applies terminal document actions, source-specific cleanup, HTML removal where
configured, and high-confidence generic/Greek PII masking. Structural removal
is a hard no-op in this stage. The exact output is what the post-clean source
reviewer sees and what any ToC/bibliography detector must hash.

Both action sets produced by Stage 20 are mandatory inputs:

- `document_actions.jsonl` contains source-lineage/replacement decisions;
- `apertus_overlap_actions.parquet` contains the validated Apertus-overlap
  natural-key drops.

Additional JSONL, single-Parquet, or sharded-Parquet action inputs are allowed.
Every action is keyed by `stable_uid`, carries the normalized
`input_text_sha256`, and is applied only when that hash equals the recomputed
normalized text hash. Exact duplicate actions are collapsed; conflicting
duplicates or unmatched actions fail the stage. A disk-backed SQLite lookup
keeps the 2.2M-row overlap action set out of each worker's Python heap.

Parquet output is streamed in bounded batches and a bounded file-level process
pool. Equal normalized/source-cleaned/PII variants reuse the same exact token
count; changed variants use tokenizer `encode_batch`. Every shard has an atomic
file receipt with input and output hashes. `submit.sh resume 50-clean` reuses a
shard only after its config, input, output, ledger, and quarantine receipts all
verify.

## Stage 58: terminal admission and optional structure

`finalize_structural_cleaning.py` consumes Stage 50 `corpus`, `ledger`, and
`quarantine` directly. It does not rerun HTML cleanup, PII masking, source
rules, or document actions. It only:

1. applies the terminal post-clean source admission;
2. validates an optional structural model receipt and span inventory;
3. applies accepted ToC/bibliography spans to the exact Stage 50 text;
4. updates eligibility and the final ledger.

Stage58 is never submitted by the normal cleaning chains. The operator must
choose a separate explicit finalization path. A confirmed no-op produces text
that is byte-for-byte equal at the string level to Stage50. A requested apply
requires a passed exact Stage54 receipt and never silently falls back to a
no-op. The chosen mode and exact promoted-receipt hash (when applicable) are
frozen in an immutable request and revalidated on resume; changing either
requires a new pipeline run.

Structural model selection/training evidence must be declared
`LLM_silver`. No nonexistent human-gold corpus is required. A model can be
enabled only when a separate receipt-bound targeted manual audit of exactly
100 high-risk predicted deletions supplies numeric prose-deletion, main-text
retention, and catastrophic-deletion metrics and passes the tracked gates.
Unavailable silver-only safety metrics can never claim promotion. Work and
exact-text split overlap must both be zero, and code/config/checkpoint hashes
are mandatory. This 100-case deployment-safety check is the only required
manual gate for structural-model promotion (separate source-admission review
still applies); full-corpus gold or mass annotation is neither required nor
planned.

Each span row binds:

- `stable_uid` and Stage 50 `input_text_sha256`;
- the Stage 50 cleaning manifest SHA-256;
- the structural model receipt SHA-256;
- `kind`, `char_start`, `char_end`, and `rule_id`.

Unknown documents, hash drift, disallowed source profiles, duplicate spans,
overlap, or out-of-bounds offsets fail closed. The span inventory is indexed on
disk and every indexed span must be encountered exactly once in Stage 50.

## Token ledger semantics

The final ledger preserves:

- `tokens_toc_removed`: pre-structural count minus a ToC-only counterfactual;
- `tokens_bibliography_removed`: pre-structural count minus a bibliography-only
  counterfactual;
- `tokens_structural_union_removed`: pre-structural count minus the actual
  union-cleaned text;
- `tokens_structural_cleaned` and `tokens_final`.

The two per-kind values are not assumed to sum to the union value because BPE
boundaries can change. `final_text_sha256` is the emitted final text hash for a
kept row and the would-be final text hash for a dropped/quarantined audit row.
It is the cleaning-to-decontamination content-chain anchor.

Completed stage validation reopens every manifest/inventory JSON and rehashes
all attested Parquet files. A small stage receipt whose shards were later
deleted or mutated therefore cannot satisfy a downstream dependency.
