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

Fill a binding JSON with the paths required by the selected template and exactly one of `G0_INPUT_RECEIPTS` through `G5_INPUT_RECEIPTS`. Render one generation at a time:

```bash
python -m sequence_models.bibliography_evolution render-queue \
  --templates "$EVAL/sequence_models/evolution/experiment_templates.json" \
  --generation G0 \
  --bindings bindings.g0.json \
  --output queue.g0.jsonl

QUEUE_SHA256=$(sha256sum queue.g0.jsonl | awk '{print $1}')
N=$(wc -l < queue.g0.jsonl)
sbatch --array="0-$((N-1))%8" \
  --export=ALL,CODE_ROOT="$EVAL",QUEUE_JSONL="$PWD/queue.g0.jsonl",QUEUE_SHA256="$QUEUE_SHA256",LEAKAGE_POLICY="$EVAL/sequence_models/evolution/leakage.policy.json",CANDIDATE_ROOT="$CANDIDATE_ROOT" \
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

## 6. One-shot sealed 150 evaluation

Before opening labels, materialize predictions for every frozen Pareto candidate on the same 150 unlabeled documents. Freeze those prediction file hashes into one request. The sealed feature-table owner creates a FROZEN receipt with this exact schema:

```json
{
  "schema_version": "bibliography-evolution-sealed-inventory-freeze-v1",
  "status": "frozen",
  "labels_sealed": true,
  "document_count": 150,
  "inventory_path": "/sealed/inventory.json",
  "inventory_sha256": "...",
  "sealed_table_path": "/sealed/feature_table",
  "sealed_table_tree_sha256": "..."
}
```

Freeze the development Pareto manifest only after that receipt exists:

```bash
python -m sequence_models.bibliography_evolution freeze-pareto \
  --registry registry.$CODE_COMMIT.json \
  --sealed-inventory /sealed/inventory.json \
  --sealed-freeze-receipt /sealed/FROZEN.json \
  --output pareto.frozen.json
```

`freeze-pareto` revalidates every candidate spec, leakage/input lineage, receipt, prediction, `all_rows`, and artifact byte immediately before freezing. Construct one request containing the manifest ID/hash, both sealed hashes, the exact ordered `candidate_ids`, and a `prediction_inputs` mapping with a path/SHA for every ID. Subsets and missing predictions fail before the no-rerun fuse is created.

```bash
python -m sequence_models.bibliography_evolution evaluate-sealed-batch \
  --manifest pareto.frozen.json \
  --request sealed.request.json \
  --batch-root /sealed/results-once \
  --iterations 10000
```

This command evaluates all candidates in one process over exactly 150 inventory IDs, reports per-source and per-work metrics, and computes source-stratified work-bootstrap intervals with Bonferroni familywise 95% coverage across all candidate/objective cells. The batch directory is exclusive; partial, incremental, subset, drifted, or repeated evaluation is rejected.
