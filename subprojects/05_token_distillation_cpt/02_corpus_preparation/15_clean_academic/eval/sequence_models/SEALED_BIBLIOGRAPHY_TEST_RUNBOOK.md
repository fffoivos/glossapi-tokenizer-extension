# Fresh 150-document sealed bibliography test

This runbook creates the final, prediction-blind test set for bibliography
pipeline selection. It is separate from the repeatedly inspected 274-document
development split and excludes the complete earlier 500-document holdout.

The terminal label is **dual-Sol LLM-silver, not human gold**. No candidate
model may run on the test documents until `FROZEN.receipt.json` exists and all
Pareto candidates have been frozen simultaneously.

## Fixed design

- 50 works each from Greek PhD, Kallipos and OpenArchives (150 total).
- Exclude every historical STRUCT-2K identity and all 500 previous holdout
  identities before admission.
- Exclude exact normalized/materialized-text copies globally.
- Exclude bottom-k word-5-gram near copies at similarity >= 0.80 globally:
  across source names, against both excluded pools, and among fresh candidates.
- Select one canonical representation per `(source, work_key)` without looking
  at text quality or model output.
- Rank deterministically, retain a 4x oversample, then apply the quality gate.
- Run the canonical GlossAPI Rust scorer on every candidate. Only automatically
  flagged documents are sent to two independent Sol quality reviews. A/B
  disagreements receive a third de-novo review of a direct, label-blind packet
  subset.
- Quality display is bounded to a deterministic 120-line head/middle/tail
  sample, 40,000 displayed text characters, 100,000 serialized characters per
  document and 180,000 per batch. Overlong lines use an explicit display-only
  prefix/suffix marker; sealed source text is unchanged.
- Annotate every present physical line twice with independent
  `gpt-5.6-sol`/high executions. Packets contain at most 400 lines or 80,000
  text characters and 15 context-overlap lines. Every line belongs to exactly
  one core interval per pass; overlap is context only.
- Pass B uses half-chunk-staggered boundaries and reversed chunk presentation.
- A physical line longer than a packet budget is represented by a bounded
  20,000-character prefix/suffix display with full-text hash/truncation metadata;
  its full-text line ID and sealed source remain unchanged.
- A third Sol execution sees only label-blind context around A/B role
  disagreements/UNKNOWNs. Exact 2/3 role agreement wins; otherwise the line is
  `UNKNOWN`.
- Freeze only at 100% A/B coverage, >= 98% A/B binary agreement overall,
  >= 95% within every source, and <= 0.5% unresolved after adjudication.
- Do not manufacture a zero-bibliography cohort. Report the naturally occurring
  zero-BIB subset after the one-shot model evaluation.

## Pinned Clariden inputs

Use the full hashes, not prefixes:

| Input | Path | SHA-256 |
|---|---|---|
| Normalization manifest | `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/pipeline_runs/full-corpus-v2-20260712-d076a59/stages/10-normalize/normalization_manifest.json` | `ccd6c2f6212ef597a63900b8015b9a432e820fe2fea8a41d7a62ac713470bc45` |
| STRUCT-2K manifest | `/iopsstor/scratch/cscs/fffoivos/inputs/APERTUS_CLASSIFIER_HANDOFF_20260712/STRUCT_2K/manifest.jsonl` | `c08611352517ff40668eb1a74daf1c5bb645f3acf03eec4c002bb2b3f222621c` |
| Prior-500 documents | `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/source_matched_holdouts/source_matched_c2_20260713_43cf377_a/documents.jsonl` | `377b21a1cdc6a41d31264a7ad459d0539d29894285bda79c9f3cb33eb3a0dd25` |
| Prior-500 selection manifest (audit only) | `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/source_matched_holdouts/source_matched_c2_20260713_43cf377_a/selection.manifest.json` | `781096358f1c3b0fc89e309df8a2b15c124bebc848c4cf266499cbfe10f93344` |

The canonical Rust dependency is
`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/python_deps/glossapi_rs_noise_6f29a2825559c540-py312`.
The selector injects it *after* the uenv `--` boundary, verifies distribution
version `0.1.0`, and verifies `score_markdown_file_detailed` before scanning.

The default sealed root is:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/sealed_tests/bibliography_150_20260718
```

## 1. Publish a clean reviewed checkout

Set these shell variables locally. The remote checkout must be clean and at
the exact reviewed commit; every Slurm stage verifies both facts.

```bash
LOCAL_REPO=/Users/foivoskarounos-zamparloukos/Projects/train-apertus-toc-bib-annotation
LOCAL_EVAL="$LOCAL_REPO/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
REMOTE_REPO=/absolute/clean/clariden/checkout
REMOTE_EVAL="$REMOTE_REPO/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
COMMIT=$(git -C "$LOCAL_REPO" rev-parse HEAD)
ROOT=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/sealed_tests/bibliography_150_20260718
REMOTE_PY=/iopsstor/scratch/cscs/fffoivos/python_envs/full_corpus_v2/bin/python
```

Do not transfer corpus data to the MacBook. Only source code, receipts, hashes,
and bounded in-memory review envelopes cross the SSH connection.

## 2. Select and quality-score the 4x candidate pool on a CPU node

The job requests 32 CPUs and 160 GiB. It emits 200 globally de-duplicated
candidates per source plus a packet containing only Rust/text-quality flags.
It does not access classifier predictions.

```bash
ssh clariden sbatch \
  --export=ALL,SEALED_BIB_REPO_ROOT="$REMOTE_REPO",SEALED_BIB_EXPECTED_COMMIT="$COMMIT",CONFIRM_SEALED_BIB_SELECT=1 \
  "$REMOTE_EVAL/sequence_models/clariden/run_sealed_bibliography_select.sbatch"
```

Accept only `00_candidate_selection/run.receipt.json` with status `passed` and
`selection.receipt.json` with pool counts 200/200/200. Staging directories and
Slurm logs are not evidence.

## 3. Run independent quality A/B reviews from the Mac coordinator

The coordinator pins `gpt-5.6-sol`, reasoning `high`, `--ephemeral`, an empty
read-only workspace, the prompt SHA and schema SHA. It fetches at most two
flagged documents at a time over SSH, keeps them only in memory, and streams
validated label JSON back to Clariden. No packet or corpus file is written on
the Mac.

```bash
COORD="$LOCAL_EVAL/sequence_models/sealed_bibliography_sol_coordinator.py"
QPACKET="$ROOT/00_candidate_selection/quality.flagged.private.jsonl"
LPROMPT="$LOCAL_EVAL/sequence_models/SEALED_BIBLIOGRAPHY_QUALITY_PROMPT.md"
LSCHEMA="$LOCAL_EVAL/sequence_models/sealed_bibliography_quality.schema.json"
RPROMPT="$REMOTE_EVAL/sequence_models/SEALED_BIBLIOGRAPHY_QUALITY_PROMPT.md"
RSCHEMA="$REMOTE_EVAL/sequence_models/sealed_bibliography_quality.schema.json"
```

Run one immutable preflight batch for A, then resume the same contract without
`--maximum-batches`. Do not reroll the preflight.

```bash
python3 "$COORD" --kind quality --ssh-host clariden \
  --remote-uenv pytorch/v2.9.1:v2 --remote-python "$REMOTE_PY" \
  --remote-pythonpath "$REMOTE_EVAL" --remote-packet "$QPACKET" \
  --remote-run-dir "$ROOT/05_quality/a-run" \
  --remote-pass-output "$ROOT/05_quality/a.response.json" \
  --pass-id quality-a --reviewer-id sealed-quality-sol-a-v1 \
  --local-prompt "$LPROMPT" --remote-prompt "$RPROMPT" \
  --local-output-schema "$LSCHEMA" --remote-output-schema "$RSCHEMA" \
  --workers 1 --maximum-batches 1

python3 "$COORD" --kind quality --ssh-host clariden \
  --remote-uenv pytorch/v2.9.1:v2 --remote-python "$REMOTE_PY" \
  --remote-pythonpath "$REMOTE_EVAL" --remote-packet "$QPACKET" \
  --remote-run-dir "$ROOT/05_quality/a-run" \
  --remote-pass-output "$ROOT/05_quality/a.response.json" \
  --pass-id quality-a --reviewer-id sealed-quality-sol-a-v1 \
  --local-prompt "$LPROMPT" --remote-prompt "$RPROMPT" \
  --local-output-schema "$LSCHEMA" --remote-output-schema "$RSCHEMA" \
  --workers 2
```

Run B in a new run directory and with a distinct reviewer identity:

```bash
python3 "$COORD" --kind quality --ssh-host clariden \
  --remote-uenv pytorch/v2.9.1:v2 --remote-python "$REMOTE_PY" \
  --remote-pythonpath "$REMOTE_EVAL" --remote-packet "$QPACKET" \
  --remote-run-dir "$ROOT/05_quality/b-run" \
  --remote-pass-output "$ROOT/05_quality/b.response.json" \
  --pass-id quality-b --reviewer-id sealed-quality-sol-b-v1 \
  --local-prompt "$LPROMPT" --remote-prompt "$RPROMPT" \
  --local-output-schema "$LSCHEMA" --remote-output-schema "$RSCHEMA" \
  --workers 2
```

Create a direct packet subset for A/B disagreements on a CPU allocation. It
contains no earlier decisions:

```bash
ssh clariden srun --account=a0140 --partition=normal --time=00:20:00 \
  --cpus-per-task=2 --mem=8G \
  uenv run pytorch/v2.9.1:v2 --view=default -- \
  env PYTHONPATH="$REMOTE_EVAL" "$REMOTE_PY" -m sequence_models.sealed_bibliography_test \
  quality-adjudication-packet --packet "$QPACKET" \
  --response-a "$ROOT/05_quality/a.response.json" --reviewer-a sealed-quality-sol-a-v1 \
  --response-b "$ROOT/05_quality/b.response.json" --reviewer-b sealed-quality-sol-b-v1 \
  --output "$ROOT/05_quality/c.packet.private.jsonl" \
  --receipt-out "$ROOT/05_quality/c.packet.receipt.json"
```

If `document_count` is nonzero, run the coordinator with `--kind quality`,
`--pass-id quality-c`, reviewer `sealed-quality-sol-c-v1`, the C packet, and new
C paths. Then merge A/B plus C. If it is zero, merge A/B only:

```bash
# Add the C --response/--reviewer-id pair only when C ran.
ssh clariden srun --account=a0140 --partition=normal --time=00:10:00 \
  --cpus-per-task=2 --mem=8G \
  uenv run pytorch/v2.9.1:v2 --view=default -- \
  env PYTHONPATH="$REMOTE_EVAL" "$REMOTE_PY" -m sequence_models.sealed_bibliography_test \
  merge-quality --packet "$QPACKET" \
  --response "$ROOT/05_quality/a.response.json" --reviewer-id sealed-quality-sol-a-v1 \
  --response "$ROOT/05_quality/b.response.json" --reviewer-id sealed-quality-sol-b-v1 \
  --output "$ROOT/05_quality/quality.consensus.json"
```

## 4. Admit 50/source and create private A/B packets

Hash `quality.consensus.json`, then submit the packet stage. It creates the
private documents, an opaque public exclusion manifest, a mode-0600 alias
secret/key, and pass A/B packets. The public manifest has IDs/hashes only; text,
source provenance and labels remain below the sealed root.

```bash
QSHA=$(ssh clariden sha256sum "$ROOT/05_quality/quality.consensus.json" | awk '{print $1}')
ssh clariden sbatch \
  --export=ALL,SEALED_BIB_REPO_ROOT="$REMOTE_REPO",SEALED_BIB_EXPECTED_COMMIT="$COMMIT",SEALED_BIB_QUALITY_CONSENSUS="$ROOT/05_quality/quality.consensus.json",SEALED_BIB_QUALITY_CONSENSUS_SHA="$QSHA",CONFIRM_SEALED_BIB_PACKETS=1 \
  "$REMOTE_EVAL/sequence_models/clariden/run_sealed_bibliography_finalize.sbatch"
```

Require `10_sealed_inputs/run.receipt.json`, exactly 150 documents, and source
counts 50/50/50.

## 5. Run exhaustive independent role passes A and B

Set role prompt/schema paths:

```bash
LPROMPT="$LOCAL_EVAL/sequence_models/SEALED_BIBLIOGRAPHY_ROLE_PROMPT.md"
LSCHEMA="$LOCAL_EVAL/sequence_models/sealed_bibliography_role.schema.json"
RPROMPT="$REMOTE_EVAL/sequence_models/SEALED_BIBLIOGRAPHY_ROLE_PROMPT.md"
RSCHEMA="$REMOTE_EVAL/sequence_models/sealed_bibliography_role.schema.json"
```

For A, use packet `10_sealed_inputs/pass-a.packet.private.jsonl`, run directory
`20_role_a/run`, output `20_role_a/pass.json`, pass ID `pass-a`, and reviewer
`sealed-role-sol-a-v1`. For B, use the corresponding pass-B packet, `21_role_b`,
`pass-b`, and `sealed-role-sol-b-v1`. The coordinator invocation is the same as
quality review except `--kind role`. Preflight one A batch with one worker, then
resume; run B independently with a new contract. Batch size defaults to two and
workers may not exceed two.

## 6. Build and run de-novo role adjudication

After both full pass receipts exist, build the disagreement packet on a CPU
node:

```bash
ssh clariden srun --account=a0140 --partition=normal --time=01:00:00 \
  --cpus-per-task=4 --mem=32G \
  uenv run pytorch/v2.9.1:v2 --view=default -- \
  env PYTHONPATH="$REMOTE_EVAL" "$REMOTE_PY" -m sequence_models.sealed_bibliography_test \
  adjudication-packet \
  --documents "$ROOT/10_sealed_inputs/documents.private.jsonl" \
  --line-key "$ROOT/10_sealed_inputs/line-key.private.jsonl" \
  --pass-a "$ROOT/20_role_a/pass.json" --pass-b "$ROOT/21_role_b/pass.json" \
  --context-radius 30 --max-lines 400 --max-chars 80000 \
  --packet-out "$ROOT/30_adjudication/packet.private.jsonl" \
  --receipt-out "$ROOT/30_adjudication/packet.receipt.json"
```

If targets exist, run the coordinator with `--kind role`, pass ID
`adjudication`, reviewer `sealed-role-sol-c-v1`, and `30_adjudication` packet/run
paths. The C envelope identifies target offsets but contains no A/B labels; Sol
labels the displayed context from scratch.

## 7. Merge, gate and freeze

Run these commands on a CPU allocation. Omit `--adjudication` only when the
adjudication receipt says no targets.

```bash
ssh clariden srun --account=a0140 --partition=normal --time=00:30:00 \
  --cpus-per-task=4 --mem=32G \
  uenv run pytorch/v2.9.1:v2 --view=default -- \
  env PYTHONPATH="$REMOTE_EVAL" "$REMOTE_PY" -m sequence_models.sealed_bibliography_test \
  merge-labels --line-key "$ROOT/10_sealed_inputs/line-key.private.jsonl" \
  --pass-a "$ROOT/20_role_a/pass.json" --pass-b "$ROOT/21_role_b/pass.json" \
  --adjudication "$ROOT/30_adjudication/pass.json" \
  --output "$ROOT/40_frozen/labels.private.jsonl" \
  --receipt-out "$ROOT/40_frozen/consensus.receipt.json"

ssh clariden srun --account=a0140 --partition=normal --time=00:20:00 \
  --cpus-per-task=2 --mem=16G \
  uenv run pytorch/v2.9.1:v2 --view=default -- \
  env PYTHONPATH="$REMOTE_EVAL" "$REMOTE_PY" -m sequence_models.sealed_bibliography_test \
  freeze --documents "$ROOT/10_sealed_inputs/documents.private.jsonl" \
  --public-exclusions "$ROOT/10_sealed_inputs/exclusions.public.json" \
  --labels "$ROOT/40_frozen/labels.private.jsonl" \
  --consensus-receipt "$ROOT/40_frozen/consensus.receipt.json" \
  --output "$ROOT/40_frozen/FROZEN.receipt.json" --lock-inputs
```

`merge-labels` deliberately preserves a blocked receipt and exits nonzero if a
gate fails. Do not relax a gate after seeing test labels. `freeze` re-verifies
50/source, exact line coverage, public/private document identity parity and all
consensus gates. Its content hashes are the only allowed inputs to the later
one-shot Pareto evaluation.

## Artifact map

- Public future-leakage exclusion list:
  `10_sealed_inputs/exclusions.public.json` (opaque IDs/hashes only).
- Sealed text: `10_sealed_inputs/documents.private.jsonl`.
- Private alias mapping: `10_sealed_inputs/line-key.private.jsonl`.
- Independent raw pass aggregates: `20_role_a/pass.json`, `21_role_b/pass.json`.
- Third-pass aggregate when needed: `30_adjudication/pass.json`.
- Sealed labels: `40_frozen/labels.private.jsonl`.
- Terminal seal: `40_frozen/FROZEN.receipt.json`.

Never publish the sealed text, line key, pass files, or labels before the final
model comparison is complete.
