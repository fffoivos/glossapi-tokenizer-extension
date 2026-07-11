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
CONFIRM_ACQUIRE=1 CONFIRM_LAUNCH=1 bash clariden/submit.sh acquire
```

Job `2735235` is the current full acquisition. Do not submit a second copy. The
extra `CONFIRM_ACQUIRE` guard exists for a deliberate future acquisition, not as
part of the downstream production chain.

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

## 7. Run the receipt-bound production CPU DAG

Choose one stable run ID. Do not change it on retries:

```bash
export PIPELINE_RUN_ID=full-corpus-v2-20260711
export ACQUISITION_RECEIPT="$RUN_ROOT/source_locks/<job-2735235>.receipt.json"
export ACQUISITION_JOB_ID=2735235  # omit after the receipt already exists
bash clariden/submit.sh chain-to-review
CONFIRM_LAUNCH=1 bash clariden/submit.sh chain-to-review
```

This creates Slurm `afterok` dependencies for `10-normalize`, `20-lineage` and
`30-review-packet`, and then stops. Lineage and review stream canonical Parquet
directly; no redundant full-corpus JSONL copy is materialized. Every normalized
row preserves the exact upstream `source_dataset`, candidate `source_id` and a
work-level `work_id`.

Review every request using the response schema. The packet job does not call a
model. After all primary, secondary and required adjudicator responses exist:

```bash
export REVIEWS_JSONL=/path/to/pre_clean_responses.jsonl
bash clariden/submit.sh review-aggregate
CONFIRM_LAUNCH=1 bash clariden/submit.sh review-aggregate
```

The aggregate is a candidate decision artifact, not an implicit authorization.
Inspect every exact `source_dataset` decision and record its displayed hash:

```bash
export SOURCE_ADMISSION="$PIPELINE_RUNS_ROOT/$PIPELINE_RUN_ID/stages/40-review-aggregate/admission_candidate.json"
export CONFIRM_ADMISSION_SHA256="$(sha256sum "$SOURCE_ADMISSION" | awk '{print $1}')"
export GREEKMMLU_QUERIES_JSONL=/immutable/path/native_greek_mcq_decontam_queries.jsonl
export GREEKMMLU_BENCHMARK_MANIFEST="$GREEKMMLU_QUERIES_JSONL.manifest.json"
bash clariden/submit.sh chain-after-admission
CONFIRM_LAUNCH=1 bash clariden/submit.sh chain-after-admission
```

If every admitted source is already `include`, this continues through
GreekMMLU, dedup and local release validation. If any source is
`include_after_cleaning`, the chain runs deterministic cleaning/PII masking,
builds a fresh packet from only those post-clean source datasets, and stops.
Review that packet, then aggregate and inspect the merged final admission:

```bash
export POST_CLEAN_REVIEWS_JSONL=/path/to/post_clean_responses.jsonl
CONFIRM_LAUNCH=1 bash clariden/submit.sh post-clean-aggregate

export FINAL_SOURCE_ADMISSION="$PIPELINE_RUNS_ROOT/$PIPELINE_RUN_ID/stages/56-post-clean-review-aggregate/final_admission_candidate.json"
export CONFIRM_FINAL_ADMISSION_SHA256="$(sha256sum "$FINAL_SOURCE_ADMISSION" | awk '{print $1}')"
bash clariden/submit.sh chain-after-post-clean
CONFIRM_LAUNCH=1 bash clariden/submit.sh chain-after-post-clean
```

The final cleaning job replays the same document actions and structural inputs
against the normalized corpus. It fails if those inputs differ from the reviewed
cleaning pass. Structural ToC/bibliography spans are off by default; enabling
them additionally requires `APPLY_STRUCTURAL=1`, a passed
`STRUCTURAL_MODEL_RECEIPT`, immutable span paths and an approved tracked policy.
The cleaner applies structural spans last.

The downstream chain runs GreekMMLU decontamination, the existing
`glossapi-corpus dedup-text run` CLI and final materialization/validation. Dedup
is explicit and receipt-bound: preserved Greek diacritics, exact plus near
dedup, 128 MinHash permutations, 32 × 4 bands, token 5-shingles, threshold 0.85
and a 5,000-member bucket ceiling. Publication is not submitted.

Check or resume without inventing a new run:

```bash
bash clariden/submit.sh status "$PIPELINE_RUN_ID"
FINAL_CLEAN_STAGE=58-final-clean bash clariden/submit.sh resume decontam
CONFIRM_LAUNCH=1 FINAL_CLEAN_STAGE=58-final-clean bash clariden/submit.sh resume decontam
```

A completed receipt makes a resubmitted stage a validated no-op. An incomplete
stage requires explicit resume; downstream stages reject missing, mismatched or
drifted receipts even if Slurm reported success.

## 8. Optional gated Hugging Face publication

Publication is standalone and never part of a chain. First inspect the release
manifest, exact token waterfall and validation receipt. Then inject the token
only into the submission environment, never a CLI argument or file:

```bash
export HF_REPO_ID=fffoivos/glossapi-greek-cpt-full-corpus-v2
export CONFIRM_PUBLISH="$HF_REPO_ID"
HF_TOKEN="$(security find-generic-password -a "$USER" -s codex-hf-token -w)" \
  CONFIRM_LAUNCH=1 bash clariden/submit.sh publish
```

On Clariden use the equivalent secure token handoff; do not copy the macOS
Keychain command literally. The publisher must verify the local validation
receipt, create/update the dataset with `gated=auto`, and write a publication
receipt. Gating does not override source redistribution restrictions.

## Script-interface handoff

The orchestration binds normalization, lineage, review and cleaning CLIs
directly. The parallel decontamination/release implementation must expose these
stable entry points, or their paths must be overridden centrally in
`clariden/paths.env` before the exact execution commit is frozen:

- `decontaminate_full_corpus.py`: `--input --output --dropped --ledger
  --manifest --queries-jsonl --benchmark-manifest --workers`;
- `materialize_release.py`: `--input --dedup-decisions --cleaning-ledger
  --decontam-ledger --output --manifest --token-waterfall
  --temporary-directory --memory-limit --threads`;
- `validate_release.py`: `--release --manifest --dedup-decisions --output
  --temporary-directory --memory-limit --threads`;
- `publish_release.py`: `--release --release-manifest --validation-receipt
  --repo-id --gate-mode auto --execute`, reading `HF_TOKEN` from the environment.

Every launcher fails before data access if its expected script is absent. This
is intentional: a renamed or half-integrated implementation cannot silently
fall back to an older corpus path.

## Clariden resource policy

Clariden does not provide this project with a CPU-only production compute queue.
Production preprocessing therefore uses the CPU side of a GH200 node on `normal`,
requests no GPU/GRES, and aims to keep about 256 of 288 cores useful.
Use `debug` only for bounded smoke tests. Use `xfer` only for internal filesystem
transfer.
