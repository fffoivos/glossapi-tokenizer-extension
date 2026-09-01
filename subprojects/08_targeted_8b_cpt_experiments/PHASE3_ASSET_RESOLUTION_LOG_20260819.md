# Phase-3 asset-resolution log — 2026-08-19

This log records operational and scientific-data-contract corrections made
while preparing the matched HPLT-to-OpenArchives extension. It does not change
the approved model, initialization, optimization, sampling order, or dataset
families.

## 1. Exact duplicated foreign-replay documents

The strict Phase-3 unseen selector correctly stopped rather than silently
selecting repeated raw text. A read-only debug audit (`3119120`) found 166
byte-identical raw-text pairs, all within `foreign_replay`, with zero repeated
document keys and zero such pairs in OpenArchives or Old-Greek replay.

The owner authorized retaining the first source-order document and excluding
the later copy. The frozen, source-row-specific manifest is:

`/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14/receipts/phase3_duplicate_exceptions_v2.json`

It excludes 123,659 tokens across the 166 later copies. It is bound to the
read-only audit receipt and does not alter the target mixture, tokenizer,
ordering of retained rows, or any OpenArchives row. Selector job `3119131`
then passed with the manifest applied and emitted
`phase3_unseen_catalog.json`.

## 2. Candidate catalog special-token count mismatch

The first Phase-3 OpenArchives tokenization attempt (`3119163`) quarantined
its derived payload and emitted no tokenization receipt. The selector's
candidate-catalog count was 2,403,548,362 tokens, while Megatron
`preprocess_data.py` produced 2,403,602,497: an exact difference of 54,135,
one per selected document.

Cause: `build_jsonl_document_catalog.py` tokenized with
`add_special_tokens=False`, while Megatron's `_HuggingFaceTokenizer.tokenize()`
uses the Hugging Face default special-token behavior. For Apertus this adds a
per-document BOS token; `preprocess_data.py` separately appends EOD.

Correction: the Phase-3 candidate catalog builder now calls the tokenizer with
`add_special_tokens=True`, keeps `--append-eod` accounting, and records that
contract explicitly. A first v19 rebuild then correctly rejected the published
OpenArchives input because it is the already-approved
`prepared_greek_stream_v1` authority, not a redundant Stage-B receipt. The
v20 builder accepts that source only with the explicit
`--accept-published-preanonymized-source` flag and records the exception in
its receipt; it does not rerun anonymization. The final corrected code is
frozen in scientific bundle
`20260819T014000Z-hard-h2g-catalog-authority-v20` (tree
`6c140088bceec86b9f2bec6f338782723f73aa410577e78f92e96fa8d81f2e97`).

The corrected candidate catalog is rebuilt as a new immutable v2 artifact;
the prior catalog and all failed tokenization outputs are retained as failed
attempt evidence and are not used for training.

## Resource discipline

All audit, catalog, runtime and tokenization work in this log runs on one
Clariden `debug` node via `salloc`. The `salloc --no-shell` form was found not
to execute its positional command on Clariden; its idle allocation `3119292`
was immediately cancelled, and later debug work uses the supported
`salloc ... srun bash <script>` form. A frozen bundle's scripts are deliberately
non-executable, so the wrapper also invokes them through `bash` rather than
attempting direct `execve` (the resulting 8-second no-work failure was
`3119295`).

The first 1.5B normal entry (`3118586`) exited before any update or checkpoint
because its compiled candidate used V14 code while its prelaunch benchmark
receipt still bound V8. A V14-bound replacement contract was frozen on debug,
the candidate was recompiled, and the exact two-node, 12-hour `sbatch
--test-only` passed before the corrected canonical request `3119306` was
submitted. This changes no model, data, optimizer or schedule setting.

## 3. Final Phase-3 immutable assets

The v20 selector was rerun against the corrected BOS/EOD catalog and passed
as debug job `3119379`. It is the only selector receipt used for the final
Phase-3 assets:

`/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14/receipts/phase3_unseen_catalog.json`

Each selected component was then tokenized independently with the frozen v20
bundle and checked by `freeze_tokenized_stream.py`: its Megatron index total
must equal its JSONL document-catalog total, including both BOS and EOD. The
completed jobs and exact totals are:

| Component | Job | Documents | Exact tokens |
| --- | ---: | ---: | ---: |
| OpenArchives | `3119397` | 54,135 | 2,403,602,497 |
| Foreign replay | `3119406` | 1,908,795 | 1,459,450,175 |
| Old-Greek replay | `3119437` | 623,182 | 552,904,950 |

Their immutable receipts are named
`tokenized_phase3_openarchives.json`, `tokenized_phase3_foreign.json`, and
`tokenized_phase3_old_greek.json` in the stage `receipts/` directory. All
three have `status: frozen` and identical catalog/index totals. The Phase-3
data-path spec was then frozen by debug job `3119443` at
`data/phases/phase3/phase_data_path.json`; it binds those three receipts and
their binary files before cache construction.

## 4. Cache-builder and provenance corrections

Three short debug cache attempts failed closed and left no accepted cache:

| Job | Rejection | Correction |
| --- | --- | --- |
| `3119446` | base Megatron receipt lacked the required compiled dataset-helper proof | bind the existing `training_megatron_runtime_helpers_v2.json` receipt and its matching helper runtime |
| `3119450` | cache code incorrectly required the full one-epoch component arrays to equal the selector's minimum 1.005× capacity | require each built component to be **at least** its required capacity; retain the distinct no-wrap/unique-document proof |
| `3119456` | the validation-cache seed root used a republished validation copy while the existing cache descriptors bind the historical heldout root | seed only from the exact historical heldout root already used by the training argv |

The corrected cache build `3119459` completed on one debug node. It froze
`phase_3_cache_build.json` and `phase_3_blend_cache.json` with cache-tree
SHA-256 `6575b7e478a1db225facb2d2c2ea6edc62e7dc7cfb0a6a5d9d7b56698f1d2260`.
It proves one epoch with no repeated document index in each component; the
available component sample counts are 586,817 (OpenArchives), 356,311
(foreign), and 134,986 (Old Greek), all above their minimum blend margins of
386,991, 97,973, and 4,899 respectively.

The Phase-3 authority was frozen successfully by `3119462`. The final
compatibility authority `producer_bundle_compatibility_v24_phase3_capacity.json`
was frozen by `3119473`; it re-verifies historical producer receipts and
explicitly admits only the narrowly patched Phase-3 bundles. The final
scientific control bundle is
`20260819T031500Z-hard-h2g-phase3-producer-v24` (tree
`eb5223739518febc385c82a14be7c73231c64d3e0a214d58f71c5573c10234f3`).
