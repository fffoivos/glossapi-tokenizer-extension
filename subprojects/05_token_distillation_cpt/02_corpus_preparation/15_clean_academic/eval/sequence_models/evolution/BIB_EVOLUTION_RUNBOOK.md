# Bibliography model evolution runbook

This strand is a controlled development search around the frozen D1 → signal-TCN → anchored-decoder baseline. It never opens the fresh 150-document test labels during development. Every queue row changes one named parameter family, runs to an immutable `receipt.json`, and is compared on the fixed qualified 268-document development inventory.

## 1. Authoritative starting point

Run on Clariden from a clean checkout of the exact commit placed in `CODE_COMMIT`. The relevant code root is the `eval` directory containing `sequence_models/`.

The baseline lock is [baseline.lock.json](baseline.lock.json). It pins the authoritative baseline root, six source receipts, the full-prediction artifact, decoder settings, and the 268-document headline metrics. The validation feature table under the locked root has 274 documents and 259,067 lines; only the separately hashed qualified inventory contributes to headline metrics. G0 deliberately decodes all 274 documents so its prediction can be byte-identical, then metrics mask to the 268.

Do not copy a single receipt hash onto a directory row. Directory inputs use a recursive digest over every regular file name, size, and content; any symlink fails closed.

```bash
export EVAL=/path/to/repo/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval
export PYTHONPATH="$EVAL"
export CODE_COMMIT=$(git -C "$EVAL" rev-parse HEAD)
export LEAKAGE_POLICY="$EVAL/sequence_models/evolution/leakage.policy.json"
export LEAKAGE_POLICY_SHA256=$(python -m sequence_models.bibliography_evolution hash-policy --path "$LEAKAGE_POLICY")
test -z "$(git -C "$EVAL" status --porcelain --untracked-files=all)"

python -m sequence_models.bibliography_evolution hash-input \
  --path /capstor/path/to/validation_table
```

The last command reports `recursive_tree_sha256_v1` for a directory and `file_sha256` for a file.

## 2. Build typed input receipts

Create one row per consumed path. A path may appear once even when several runner flags use it, but its `data_class` must also satisfy automatic finalization. In particular, every candidate needs exactly one of each:

- `development_table`
- `qualified_development_inventory`
- `baseline_work_objectives`
- `code_test_receipt`

Example:

```bash
python -m sequence_models.bibliography_evolution make-input-row \
  --path "$VALIDATION_TABLE_DIR" \
  --data-class development_table \
  --split validation \
  --document-scope prediction_blind_extraction_qualified_268 \
  --contains-labels \
  --output rows.validation-table.json

python -m sequence_models.bibliography_evolution make-input-row \
  --path "$QUALIFIED_268_IDS" \
  --data-class qualified_development_inventory \
  --split development \
  --document-scope prediction_blind_extraction_qualified_268 \
  --output rows.qualified.json
```

Produce `baseline_work_objectives` once from the locked baseline prediction with `sequence_models.bibliography_evolution_metrics`; it must contain the same indivisible work IDs and source assignments used for every paired bootstrap.

The code-test receipt is a JSON file tied to the exact commit:

```json
{
  "status": "passed",
  "code_commit": "FULL_40_CHARACTER_COMMIT",
  "invariants": {
    "physical_gap_walls": true,
    "header_roles_non_seed": true
  }
}
```

Create it only after the contract/header tests pass under the pinned environment. Merge named rows into a mapping, for example:

```bash
jq -n \
  --slurpfile table rows.validation-table.json \
  --slurpfile qualified rows.qualified.json \
  --slurpfile baseline rows.baseline-work.json \
  --slurpfile tests rows.code-tests.json \
  '{validation_table:$table[0],qualified:$qualified[0],baseline_work:$baseline[0],code_tests:$tests[0]}' \
  > g0.inputs.json
```

Also add typed rows for every other path present in the chosen runner: lock, baseline root, line/signal/scope arrays, quality decisions, train OOF tables, and so on. `CandidateStore` rejects an undeclared path or an unapproved runner flag/module.

The binding contract is [bindings.schema.json](bindings.schema.json); [bindings.example.json](bindings.example.json) is deliberately non-executable and shows the shape only.

## 3. Parent lineage by generation

The templates use generation-specific receipt maps rather than one generic map:

- G0: `G0_INPUT_RECEIPTS`, no parent.
- G1: `G1_INPUT_RECEIPTS`, including the passed G0 receipt.
- G2: `G2_INPUT_RECEIPTS`, including the selected G1 receipt plus its owned `main` prediction and `combined_barriers.npz`.
- G3: `G3_INPUT_RECEIPTS`, including the selected G2 receipt plus the same two owned artifacts.
- G4: `G4_INPUT_RECEIPTS`, receipt-only G0 lineage. G4 is intentionally an independent signal-architecture branch from the stock frozen decoder, not a false descendant that resets G3.
- G5: `G5_INPUT_RECEIPTS`, both parent receipts, both predictions, both barrier artifacts, and the hashed development Pareto registry.

Generate verified parent rows rather than typing them:

```bash
# G1 or G4: lineage only
python -m sequence_models.bibliography_evolution make-parent-inputs \
  --receipt /candidate/g0/receipt.json --prefix g0 --receipt-only \
  --output parent.g0.json

# G2/G3/G5: lineage plus owned prediction and barriers
python -m sequence_models.bibliography_evolution make-parent-inputs \
  --receipt /candidate/parent/receipt.json --prefix parent \
  --output parent.full.json

jq -s 'reduce .[] as $packet ({}; . * $packet)' base.inputs.json parent.full.json \
  > g3.inputs.json
```

The parent verifier recursively rechecks the finalized parent, spec hash, current input hashes, prediction, barriers, and all other referenced artifacts. G5 additionally proves both IDs are Pareto members of the declared registry. Its composition unions both parents' hard and directional barriers.

## 4. Render and run G0–G5

Fill a binding JSON with `CODE_COMMIT`, the canonical parsed-JSON `LEAKAGE_POLICY_SHA256` reported by `hash-policy`, the paths required by the selected template, and exactly one of `G0_INPUT_RECEIPTS` through `G5_INPUT_RECEIPTS`. The policy digest is part of every candidate ID; `hash-input` reports a different raw-file digest and must not be used for this field. Render one generation at a time:

```bash
python -m sequence_models.bibliography_evolution render-queue \
  --templates "$EVAL/sequence_models/evolution/experiment_templates.json" \
  --generation G0 \
  --bindings bindings.g0.json \
  --output queue.g0.jsonl

QUEUE_SHA256=$(sha256sum queue.g0.jsonl | awk '{print $1}')
N=$(wc -l < queue.g0.jsonl)
sbatch --array="0-$((N-1))%8" \
  --export=ALL,CODE_ROOT="$EVAL",QUEUE_JSONL="$PWD/queue.g0.jsonl",QUEUE_SHA256="$QUEUE_SHA256",LEAKAGE_POLICY="$LEAKAGE_POLICY",CANDIDATE_ROOT="$CANDIDATE_ROOT" \
  "$EVAL/sequence_models/clariden/run_bibliography_evolution_cpu.sbatch"
```

Use the CPU launcher for G0–G3 and G5. Use `run_bibliography_evolution_signal.sbatch` only for the G4 signal-TCN branch. Both launchers verify the frozen queue SHA, exact sklearn 1.9.0 dependency path, and component class before execution.

Successful array items automatically run the backend, fixed-268 metrics, source-stratified paired work bootstrap, commit-bound test checks, wall/header invariants, result preparation, and finalization. The terminal artifact is:

```text
$CANDIDATE_ROOT/<candidate_id>/receipt.json
```

Every file under that candidate becomes read-only. A failed item has no receipt and is excluded from the registry. Do not delete or reuse it: fix the code/config, commit, render a new immutable candidate ID, and submit that row.

G3 always executes this active canonical pipeline, with one swept stage and fixed reference settings for the others:

1. internal gap connection (`0.20`, 2 lines)
2. boundary trim (`0.05`, 1 line)
3. outward edge (`0.40`, 1 line)
4. weak unseeded (`0.20`, 1 line)
5. whole-component veto (`0.02`)

The backend receipt records the executed stage trace and parameters; acceptance checks that trace rather than trusting a static order declaration.

## 5. Rebuild the development registry

```bash
python -m sequence_models.bibliography_evolution build-registry \
  --candidate-root "$CANDIDATE_ROOT" \
  --output registry.$CODE_COMMIT.json
```

The registry minimizes four objectives: token FP, token FN, spurious blocks per zero-block document, and mean emitted-line boundary error. Exact objective ties collapse deterministically by earlier generation then candidate ID. Keep the entire nondominated set; do not choose on the sealed set.

## 6. Sealed 150 boundary and prediction-only inference

The annotation lane creates `documents.private.jsonl`, `labels.private.jsonl`, its consensus receipt, and `FROZEN.receipt.json`. `freeze-pareto` binds those exact artifacts and freshly rebuilds the development Pareto frontier before any model evaluation:

```json
{
  "schema_version": "bibliography-sealed-freeze-v1",
  "status": "frozen_prediction_blind_test_set",
  "document_count": 150,
  "source_document_counts": {"greek_phd": 50, "kallipos": 50, "openarchives": 50},
  "sealed_hashes": {
    "documents_sha256": "...",
    "labels_sha256": "...",
    "consensus_receipt_sha256": "..."
  }
}
```

Freeze the development Pareto manifest only after that receipt exists:

```bash
python -m sequence_models.bibliography_evolution freeze-pareto \
  --registry registry.$CODE_COMMIT.json \
  --sealed-documents /sealed/documents.private.jsonl \
  --sealed-labels /sealed/labels.private.jsonl \
  --sealed-consensus-receipt /sealed/consensus.receipt.json \
  --sealed-freeze-receipt /sealed/FROZEN.receipt.json \
  --output pareto.frozen.json
```

`freeze-pareto` revalidates every candidate spec, leakage/input lineage, receipt, prediction, `all_rows`, artifact byte, the sealed document hash, 150 unique document IDs, and the exact 50/source quota. It binds the label and consensus hashes supplied by the terminal annotation seal without reading those two files. It also hashes the complete source-file inventory used by sealed feature extraction and G0–G5 replay; later source drift invalidates the manifest.

Next derive all predictions without passing a labels path:

```bash
python -m sequence_models.bibliography_evolution prepare-sealed-inference \
  --manifest pareto.frozen.json \
  --sealed-documents /sealed/documents.private.jsonl \
  --sealed-freeze-receipt /sealed/FROZEN.receipt.json \
  --output-root /sealed/predictions-$CODE_COMMIT
```

This command fails if `FROZEN.receipt.json` is absent, but never opens the label or consensus files. It materializes the 150-document unlabeled feature table, verifies the exact 50/source document and line inventory, derives D1 and signal-TCN probabilities from G0-bound model bytes, and recursively replays the G0–G5 candidate graph. Every frontier prediction has a derivation receipt binding the candidate spec, finalized receipt, parent prediction hashes, algorithm parameters, model artifacts, table, and annotation document hash. Caller-supplied prediction arrays are not an input. A learned-header G2 remains fail-closed unless its finalized candidate owns a deployable inference model; a validation-sized role-ID array is not accepted as a model.

## 7. One canonical final batch

Create one exact request after the prediction receipt exists:

```bash
MANIFEST_ID=$(jq -r .frozen_manifest_id pareto.frozen.json)
INFERENCE_RECEIPT=/sealed/predictions-$CODE_COMMIT/receipt.json
jq -n \
  --arg id "$MANIFEST_ID" \
  --arg path "$INFERENCE_RECEIPT" \
  --arg sha "$(sha256sum "$INFERENCE_RECEIPT" | awk '{print $1}')" \
  --slurpfile manifest pareto.frozen.json \
  '{schema_version:"bibliography-evolution-sealed-batch-request-v1",
    evaluation_mode:"one_simultaneous_batch_all_pareto_candidates",
    frozen_manifest_id:$id,
    candidate_ids:$manifest[0].candidate_ids,
    sealed_inference_receipt:{path:$path,sha256:$sha},
    bootstrap:{method:"source_stratified_work_bootstrap_bonferroni_simultaneous",
               iterations:10000,seed:20260718}}' \
  > sealed.request.json

BATCH_ROOT=$(jq -r .canonical_batch_root pareto.frozen.json)
python -m sequence_models.bibliography_evolution evaluate-sealed-batch \
  --manifest pareto.frozen.json \
  --request sealed.request.json \
  --batch-root "$BATCH_ROOT" \
  --iterations 10000 --seed 20260718
```

Before the fuse is written, the evaluator freshly rebuilds the Pareto frontier and rechecks candidate cardinality/order, all prediction derivations, bool shapes, barrier shapes, table/document hashes, FROZEN hash, and 50/source balance. It independently rematerializes the unlabeled table from sealed text and reruns every candidate and ancestor from the frozen model/spec bytes; a merely well-formed receipt with arbitrary arrays fails. It also hash-preflights the label and consensus files. The fuse has exactly one canonical location bound inside the hashed frozen manifest, so moving or copying the manifest cannot create a second final batch; its payload also binds the requested bootstrap method, iterations, and seed. Only after that fuse exists are labels parsed for the simultaneous comparison and multiplicity-adjusted source-stratified work bootstrap. Work rows support byte-exact crash recovery, and a final `sealed_results.receipt.json` binds the completed result. Repeating the exact completed request returns the existing result without parsing labels again.
