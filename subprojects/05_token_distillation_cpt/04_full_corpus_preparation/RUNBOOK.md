# Phase 04 operator runbook

Phase 04 is dry-run-first. Preparing or publishing scripts does not authorize a
corpus download, Slurm submission, policy approval or materialization.

## 1. Prepare scripts locally

From the repository root:

```bash
bash subprojects/05_token_distillation_cpt/04_full_corpus_preparation/clariden/prepare.sh
```

This validates tracked configs, compiles Python, checks shell syntax, runs Rust
tests and runs the Phase-04 Python tests. It performs no SSH, download or Slurm
operation.

## 2. Publish an exact commit

Review and commit the intended branch before Clariden execution. On Clariden,
clone or fetch that branch into a real Git worktree and record the exact commit.
Do not copy an uncommitted development directory into the execution path.

The default expected location is:

```bash
REPO=/iopsstor/scratch/cscs/fffoivos/repo/train-apertus-with-glossapi
```

Override `REPO_ROOT` explicitly if a different Git worktree is chosen. The old
`glossapi-tokenizer-extension` mirror on Clariden is not a Git checkout and must
not be used as the Phase-04 code source.

## 3. Prepare the Clariden runtime

After checking out the exact commit, submit runtime preparation as a normal
CPU-side Slurm job:

```bash
cd "$REPO/subprojects/05_token_distillation_cpt/04_full_corpus_preparation"
bash clariden/submit.sh bootstrap-runtime
CONFIRM_LAUNCH=1 bash clariden/submit.sh bootstrap-runtime
```

To replace an existing runtime intentionally, export `REBUILD_RUNTIME=1` before
the live submission. The bootstrap script refuses login-node/direct execution,
requires a clean checkout and publishes the venv by atomic directory rename.

Build the detector from the same clean checkout on a CPU-side node:

```bash
bash clariden/submit.sh build-detector
CONFIRM_LAUNCH=1 bash clariden/submit.sh build-detector
```

The build job uses a job-unique Cargo target directory and atomically publishes
an AArch64 binary, `COMPLETED` marker and `build_receipt.json` under
`$RUN_ROOT/detector_builds/<commit>-<job-id>/`. Preserve the printed
`REFERENCE_BIN` and `DETECTOR_BUILD_RECEIPT` values for structural submission.

Before a structural job can run, hydrate the private `STRUCT_2K_gold.jsonl`, pin
its SHA-256 in `configs/cleaning_policy.json`, and run all 608 held-out documents
through `eval/rust_parity_struct.py`. The resulting parity receipt is bound to
the exact detector binary, gold hash, model files and smoother. `20_structural_detect.sbatch`
fails before scanning data if that receipt is missing, partial, stale or hash-mismatched.

## 4. Resolve and acquire sources

The source job resolves every repository to the already tracked immutable
revision, writes a selected-file lock, downloads into IOPS scratch, verifies
every payload once (LFS SHA-256 or non-LFS Git blob ID), and checks local Parquet
schemas. It publishes a final `ACQUISITION_RECEIPT`; absence of that receipt means
the acquisition is incomplete.

Dry-run only:

```bash
bash clariden/submit.sh acquire
```

For a bounded first acquisition, export `SOURCES` with space-separated registry
IDs and supply `HF_TOKEN` only in the submission environment. For example, begin
with the tokenizer and one audit source. Live submission always requires:

```bash
CONFIRM_LAUNCH=1 bash clariden/submit.sh acquire
```

Do not use `xfer` for this job. Hugging Face is external; `xfer` is restricted to
`cp`/`mv`/`rsync` between CSCS filesystems.

## 5. Run source audits

Before destructive audits, build the immutable route manifest locally or on a
Clariden CPU node (this reads configs only):

```bash
python scripts/build_source_lineage.py registry \
  --output "$RUN_ROOT/lineage/registry.json"
```

After source-specific normalizers have emitted canonical JSONL envelopes, build
the base/candidate identity graph. Sectioned inputs must already have a
work-level `work_id`:

```bash
python scripts/build_source_lineage.py rows \
  --base-jsonl "$RUN_ROOT/normalized/base.jsonl" \
  --candidate-jsonl "$RUN_ROOT/normalized/candidates.jsonl" \
  --registry-manifest-out "$RUN_ROOT/lineage/registry.json" \
  --rows-out "$RUN_ROOT/lineage/rows.jsonl" \
  --relationships-out "$RUN_ROOT/lineage/relationships.jsonl" \
  --summary-out "$RUN_ROOT/lineage/summary.json"
```

Build review requests from the canonical candidate stream. This is a packet
builder, not a model invocation:

```bash
python scripts/build_source_review_packet.py \
  --candidate-jsonl "$RUN_ROOT/normalized/candidates.jsonl" \
  --requests-out "$RUN_ROOT/source_review/pre_clean_requests.jsonl" \
  --summary-out "$RUN_ROOT/source_review/pre_clean_summary.json"
```

After independent structured reviews have been collected, validate and
aggregate them. A nonzero exit means a requested review or adjudication remains
unresolved:

```bash
python scripts/aggregate_source_reviews.py \
  --requests "$RUN_ROOT/source_review/pre_clean_requests.jsonl" \
  --packet-summary "$RUN_ROOT/source_review/pre_clean_summary.json" \
  --reviews "$RUN_ROOT/source_review/pre_clean_responses.jsonl" \
  --novelty-summary "$RUN_ROOT/lineage/source_novelty.json" \
  --output "$RUN_ROOT/source_review/admission.json"
```

Do not copy review excerpts or quarantined/private rows into the HF release.
The source-quality admission report is necessary but not sufficient: candidate
survivors still pass base/candidate exact and near deduplication.

Example shape for Diavgeia after acquisition:

```bash
export INPUT="$DATA_ROOT/hf/diavgeia/ce873e49bfd8068bcf2d8692b1eb176b523ab193"
export INPUT_RECEIPT="$RUN_ROOT/source_locks/<completed-acquisition>.receipt.json"
export SOURCE=diavgeia
export TEXT_COLUMN=markdown_text
export ID_COLUMN=id
export METADATA_COLUMN=metadata_json
bash clariden/submit.sh quality
```

The job is still a dry-run unless `CONFIRM_LAUNCH=1` is present. Its outputs are
quality summaries, template candidates, reversible signing/ADA spans and
document-action candidates. It never writes cleaned text.

For academic data, run `structural-detect` one source at a time using its tracked
text and ID columns. Also export the acquisition `INPUT_RECEIPT`, the immutable
`REFERENCE_BIN`, its `DETECTOR_BUILD_RECEIPT`, and the passed `PARITY_RECEIPT`.
Feed the resulting immutable span ledger and the same `INPUT_RECEIPT` to
`structural-token-loss`. Review source-stratified random samples, near-threshold
samples and the highest-loss documents—not only aggregate precision.

Launch gates re-hash the small configs/receipts and require the exact resolved
Parquet path set, sizes and acquisition-time stat identity. They do not
re-hash the full corpus on every launch; this assumes the IOPS acquisition tree
is trusted and write-restricted. Run an explicit full payload scrub again before
any later materialization if that storage trust boundary cannot be guaranteed.

The current audit runtime pins direct Python package versions and records the
Rust toolchain in each detector-build receipt. Before materialization, replace
that bootstrap contract with a hash-locked transitive Python environment and a
pinned Rust toolchain/runtime receipt; this is a materialization-stage provenance
gate, not authorization to expand the present audit-only phase.

Raw Kallipos/Pergamos section Parquets are currently blocked by the route gate;
their grouped canonical normalizer must be implemented first. The already
concatenated Kallipos/Pergamos rows inside the nanochat base have explicit
`canonical_mixed` routes and can be audited with their exact tracked source regex.

## 6. Policy gate

Before changing `configs/cleaning_policy.json` from `audit_only`, require:

1. a fresh source-balanced holdout for Greek PhD, OpenArchives, Kallipos and
   Pergamos;
2. exact bibliography-only, ToC-only and union token deltas;
3. separate Greek, Latin and polytonic removed-character mass alongside exact
   bibliography token loss;
4. emptied/near-emptied document review;
5. a Diavgeia report separating boilerplate spans, excluded documents, PII
   replacements and template/downsampling loss;
6. a recorded decision for every source/profile pair.

This commit contains no working materializer. Passing `--materialize-dir` is a
hard error even if someone edits the policy flags. The later materialization phase
must bind source/profile, immutable input revision and hashes, detector binary and
model identity, parity receipt, span ledger, tokenizer, conflict decisions and
empty-document quarantine; it must write atomically and emit a completion manifest.

## Clariden resource policy

Clariden does not provide this project with a CPU-only production compute queue.
Production preprocessing therefore uses the CPU side of a GH200 node on `normal`,
requests no GPU/GRES, and aims to keep about 256 of 288 cores useful.
Use `debug` only for bounded smoke tests. Use `xfer` only for internal filesystem
transfer.
