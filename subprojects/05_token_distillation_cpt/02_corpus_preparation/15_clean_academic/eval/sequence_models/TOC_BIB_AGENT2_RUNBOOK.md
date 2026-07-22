# ToC/BIB Agent 2 implementation and execution runbook

This strand develops a precision-first structural classifier for formal tables
of contents and bibliographies. It emits evidence and candidate spans only.
It does not remove corpus text, and LLM-silver agreement is not production
authorization.

## Implemented architecture

The comparison has five independently measurable components:

1. **C0/C1/C2/N1 model ladder**: the existing frozen two-head LR baseline,
   feature CRF, character-augmented CRF, and byte-CNN/TCN/CRF research arm.
2. **R0 reproduction**: an exact dependency-free reproduction of the historical
   GlossAPI section-index heuristic.
3. **R1 ToC rules**: Arabic/Roman page tails, section prefixes, leaders, table
   rows, exact Greek/English headings, page progression, front-position prior
   for headerless blocks, typed gaps, and hard barriers.
4. **R2 bibliography rules**: author/year, numbered, journal/page, publisher,
   persistent-ID, URL, and legal-source evidence; exact headings and typed
   subheadings; CV/Notes scope suppression; coherent multi-line blocks.
5. **Hybrid policy**: rules-only, base-plus-rules, base-rules-veto, and
   base-plus-rules-veto prediction adapters.

The deterministic policy is deliberately asymmetric:

- a heading is only an anchor and never means “heading to EOF”;
- isolated lines never become removal spans;
- at most two typed soft-gap lines and 80 tokens may be bridged;
- unrepresented annotated-window intervals are hard barriers;
- known full-document blank intervals are reconstructed from physical indices;
- ToC/BIB overlaps fail closed to keep;
- CV publication lists, Notes/footnotes, prose, statistical tables, legal-body
  enumerations, chronology lists, and narrative author-year prose are kept;
- a headerless bibliography proposal is **support/veto evidence only** and
  cannot independently add a removal label until it clears the false-positive
  audit gate.

Python and Rust use the component IDs
`deterministic_structural_rules_v1` and
`confirmed_blocks_typed_gaps_v1`. The Rust CLI mode is
`deterministic-structure`; its decisions and span sidecars bind the exact
input with `row_uid`, `original_sha256`, and `original_chars`.

## Validation sequence

### 0. Source-matched unseen-work holdout

The independent holdout is drawn from the same canonical representations used
to reconstruct STRUCT-2K, but from different works:

- Nanochat `source_dataset=greek_phd`;
- `kallipos_sections`, grouped to one full book per historical filename; and
- Nanochat `source_dataset=openarchives.gr`.

`source_matched_holdout.py build` requires the complete 2,000-row historical
manifest (667 Greek PhD, 666 Kallipos, 667 OpenArchives), not the recoverable
1,392-row training subset. It excludes historical source/work identities before
reading candidate text, then applies a bounded same-source bottom-k word-shingle
near-duplicate gate against historical observed text and already selected works.
The default immutable sample is 150 Greek PhD theses, 150 Kallipos books and
200 OpenArchives works.

`predict` loads the frozen C2 checkpoint and exact deletion bias, rejects model
hash drift, and performs inference only. `build-review` creates 200 blinded,
source-balanced cases: 50 ToC risks, 50 bibliography risks, 50 deterministic/C2
disagreements and 50 retained hard negatives. Model predictions and selection
strata are confined to the private key; the Codex request sees only the target
line and local context.

On Clariden, run the three stages atomically with
`clariden/run_source_matched_holdout.sbatch`. The job requires an exact clean
commit, the immutable C2 path/hash, and `CONFIRM_SOURCE_HOLDOUT=1`; it never fits
a model or mutates corpus data. After Codex review, build the local dual-review
site with `build_holdout_review_site.py`. The site hides C2, Codex, and the
selection stratum until the user has saved an independent judgment, stores user
judgments in browser local storage, and exports a packet-hash-bound JSON review.

### 1. Close the inherited model ladder

Use only the immutable published `run.receipt.json`. Staging files and log
fragments are not comparison evidence. The historical 608-document test
partition remains physically excluded; the comparison uses only the derived
274-document validation split from the 1,392 historical-train documents.

### 2. Run all deterministic/hybrid ablations

Run on an approved worker from the academic-cleaning `eval/` directory:

```bash
python3 -m sequence_models.deterministic_ablation_runner \
  --silver /explicit/validation-capable/struct2k.LLM_silver.jsonl \
  --split-manifest /explicit/struct2k.LLM_silver.split.json \
  --base-c0 /immutable/joint-ladder/c0.validation.predictions.jsonl \
  --config sequence_models/joint_config.json \
  --allowed-split validation \
  --output-dir /new/immutable/deterministic-ablation-run
```

The runner rejects test-like aliases before model/adapter materialization,
validates the silver contract, refuses overwrite, and emits four prediction
files plus `ablation.report.json`. The report binds the C0 model IDs, every
input/output SHA-256, component IDs, code revision, and per-source LLM-silver
metrics.

For Clariden, submit `clariden/run_deterministic_ablation.sbatch` only after
reviewing its plan output. Required environment:

- `PHASE04_CLARIDEN_DIR`: exact checked-out Phase-04 `clariden/` directory;
- `PHASE04_EXPECTED_COMMIT`: exact clean repository commit (normally exported
  by the Phase-04 submission wrapper);
- `DETERMINISTIC_ABLATION_RUN_ID`: new immutable run ID;
- either `STRUCT2K_SOURCE_ROOT`, or exact `DETERMINISTIC_SOURCE_RECEIPT`,
  `DETERMINISTIC_SILVER`, and `DETERMINISTIC_SPLIT_MANIFEST` paths;
- either `JOINT_LADDER_RUN_ROOT`, or exact `DETERMINISTIC_LADDER_RECEIPT` and
  `DETERMINISTIC_BASE_C0` paths.

`DETERMINISTIC_CONFIG` defaults to tracked `joint_config.json`, and
`OUTPUT_ROOT` defaults below
`$RUN_ROOT/classifier_research/deterministic_ablations/`. Exact-path overrides
must still resolve to the artifacts bound by the source and ladder receipts.
The default is plan-only; execution additionally requires
`CONFIRM_DETERMINISTIC_ABLATION=1`. The wrapper accepts only validation, checks
that the source emitted zero historical-test rows and that the ladder loaded
zero historical-test documents, verifies every ladder artifact hash, hides
CUDA/NVIDIA/ROCm/HIP visibility, checks the ablation receipt before and after
no-replace publication, and never fits a model or mutates corpus data.

### 3. Build the 200-case blinded audit

Build 50 unique cases in each of four strata: ToC high risk, BIB high risk,
model disagreement, and hard negative. Selection is risk-ranked within source
and round-robin source-balanced. Requests omit the stratum, silver label, and
model predictions; the separate key is written mode `0600`.

```bash
python3 -m sequence_models.codex56_audit build \
  --silver /explicit/validation-only.jsonl \
  --baseline-predictions /immutable/c0.validation.predictions.jsonl \
  --candidate-predictions /immutable/candidate.predictions.jsonl \
  --allowed-split validation \
  --per-stratum 50 \
  --requests-out /new/audit/requests.jsonl \
  --key-out /new/private/audit.key.jsonl \
  --manifest-out /new/audit/manifest.json
```

This is honestly described as **prompt-blinded, not access-isolated**. The Codex
CLI runs from a fresh empty directory in read-only mode and is instructed not
to use tools, but filesystem read sandboxing alone is not a formal isolation
boundary.

### 4. Preflight one Codex 5.6 batch, then run the frozen packet

```bash
python3 -m sequence_models.run_codex56_audit \
  --requests /new/audit/requests.jsonl \
  --request-manifest /new/audit/manifest.json \
  --model gpt-5.6-terra \
  --prompt sequence_models/CODEX56_AUDIT_PROMPT.md \
  --output-schema sequence_models/codex56_audit_batch.schema.json \
  --batch-dir /new/audit/batches \
  --responses-out /new/audit/responses.jsonl \
  --receipt-out /new/audit/execution.receipt.json \
  --workers 1 \
  --maximum-batches 1
```

Only after that immutable preflight validates should a new, explicitly
contracted full run omit `--maximum-batches`. Never “reroll” an unfavorable
batch. Every request self-hash is recomputed, the request manifest binds the
complete set, and BIB/TOC responses must span the selected target line.

Validate and summarize:

```bash
python3 -m sequence_models.codex56_audit validate \
  --requests /new/audit/requests.jsonl \
  --responses /new/audit/responses.jsonl \
  --expected-model gpt-5.6-terra \
  --receipt-out /new/audit/validation.receipt.json

python3 -m sequence_models.codex56_audit summarize \
  --key /new/private/audit.key.jsonl \
  --requests /new/audit/requests.jsonl \
  --responses /new/audit/responses.jsonl \
  --expected-model gpt-5.6-terra \
  --findings-out /new/audit/findings.json
```

Reported disagreement is an **audit-sample rate**, not a corpus prevalence
estimate. Expand a source/stratum slice at five high-confidence disagreements
or a rate above 10% with at least ten cases. Recommend the larger 1,392-document
re-audit at two affected sources or a risk-audit-sample rate above 5%.

### 5. Promotion boundary

A candidate may proceed to corpus-policy integration only after:

- immutable model-ladder and ablation receipts;
- exact Python/Rust role/span parity on the expanded differential fixtures;
- the 200-case Codex audit;
- a separate receipt-bound 100-case high-risk deletion review
  (50 ToC + 50 BIB) with zero catastrophic prose deletions;
- runtime/resource receipts and the existing Stage 52/54 policy gates.

Until all gates pass, the production cleaning fallback is no-op. Agent 2 hands
off candidate spans, evidence, conflicts, and receipts; the dataset-preparation
agent decides where the approved policy is applied.

## Worker note

This code is CPU-oriented. Clariden’s visible `normal` and `debug` compute
nodes are four-GH200 nodes, so even a CPU-only Slurm allocation reserves the
whole GPU node. Do not submit another long CPU-only run blindly. Bundle short
evaluation work, use the shortest suitable allocation, or use an approved true
CPU worker when one is available.
