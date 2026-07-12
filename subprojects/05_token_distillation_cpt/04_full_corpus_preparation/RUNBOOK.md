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

The legacy filename `STRUCT_2K_gold.jsonl` refers to LLM-silver annotations, not
human gold. The exact historical joint ToC+BIB handoff is recovered and pinned;
its 608 historical-test documents remain physically excluded from fitting,
validation and calibration. No import, ladder or parity job has run. First run
the imported-source ladder, then record an explicit C0 selection and fresh Rust
parity with `build_joint_c0_bridge.sbatch`; non-C0 arms need a separate Rust
port/parity package. Production structural application does not
require a new full-corpus annotation effort: Stage58 requires leak-free model
evidence plus the independent receipt-bound manual audit of exactly 100
high-risk predicted removals (50 ToC + 50 BIB). Missing evidence yields a
deterministic no-op.

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
IDs and supply `HF_TOKEN` only in the submission environment. A payload-download
submission requires:

```bash
CONFIRM_ACQUIRE=1 CONFIRM_LAUNCH=1 bash clariden/submit.sh acquire
```

### Recover the completed staged payload

Clariden currently holds all 26 pinned source directories under
`/iopsstor/scratch/cscs/fffoivos/cpt_corpus/full_corpus_v2/hf`: 158 GiB on disk,
168,623,515,496 selected payload bytes, and no `.incomplete` files. Job `2735391`
downloaded and payload-verified all of it, but its old checkout rejected the two
School-books schemas. It therefore did **not** produce a passed receipt and must
not be used as an upstream job dependency.

After checking out the corrected clean commit and rebuilding the runtime, submit
exactly one existing-payload verification run:

```bash
cd "$REPO/subprojects/05_token_distillation_cpt/04_full_corpus_preparation"
export ACQUISITION_EXISTING_ONLY=1
export CONFIRM_ACQUIRE=1
export CONFIRM_LAUNCH=1
# Inject HF_TOKEN through the secure submission environment; never store it.
bash clariden/submit.sh acquire
```

Do not set `LOCK`, `DOWNLOAD_MANIFEST`, `SCHEMA_AUDIT` or
`ACQUISITION_RECEIPT`: the job creates fresh timestamp/job-ID paths for all four.
`--existing-only` performs no network payload download. It does require an HF
token to resolve exact LFS content identifiers: the anonymous API can redact an
LFS SHA-256 as 64 asterisks, which is not acceptable receipt evidence. All
158 GiB payload checks remain local. The resolver now rejects redacted hashes
before payload verification; the downloader then fails if any locked file is
missing or mismatched and rechecks the corrected schemas before writing the new
receipt. Preserve the four paths printed by this new job.

### Mozilla Data Collective routes

Istorima, the Modern Greek Dictionary and ERT Press are registered by exact MDC
dataset ID, slug, filename and size. Istorima also has a pinned archive SHA-256;
the other two fail closed until their web terms are accepted and the API exposes
their checksums. The acquisition holds presigned URLs only in memory, supports
Range resume, safely extracts regular files only, and hashes both the archive
and every selected payload file.

```bash
# Inject MOZILLA_DATA_COLLECTIVE_API_KEY only into the submission environment.
CONFIRM_LAUNCH=1 bash clariden/submit.sh acquire-mdc
```

Combine the passed HF and MDC receipts with
`scripts/merge_acquisition_receipts.py`. Normalization consumes only that merged
`full_cpt_acquisition_receipt_v1`.

For the downstream chain, use only the fresh passed receipt and omit the old job
ID entirely:

```bash
export ACQUISITION_RECEIPT=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/source_locks/sources_<fresh-timestamp>_<fresh-job>.receipt.json
unset ACQUISITION_JOB_ID
```

This recovery has no dependency on `2735391`; that job is historical evidence
that the bytes arrived, not a completion gate.

Do not use `xfer` for this job. Hugging Face is external; `xfer` is restricted to
`cp`/`mv`/`rsync` between CSCS filesystems.

## 5. Run source audits

Use the receipt-bound stages, not ad-hoc JSONL exports. `chain-to-review` runs
normalization, lineage and review-packet construction on Clariden CPU nodes and
then stops. The packet is grouped by the exact preserved `source_dataset`:
every group receives at least 100 unique redacted documents (60 random, 20
high-risk and 20 cluster representatives), while the frozen policy uses 200 for
named large/heterogeneous groups. It checks quality, markup/PDF corruption and
template variability. The packet builder invokes no model and approves no
source.

Only the small redacted packet and its receipts may be copied to the Mac for
review. Never rsync normalized, cleaned, quarantined or private source trees to
the Mac.

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

The runtime requirements pin direct Python package versions and detector builds
record the Rust toolchain. The transitive Python environment is not yet
hash-locked; record that provenance limitation explicitly rather than claiming
a fully reproducible package closure.

The production normalizer now groups raw Kallipos/Pergamos section Parquets to
work level before lineage comparison. The already concatenated rows inside the
Nanochat base retain their separate `canonical_mixed` routes; the two
representations are never silently concatenated or treated as snapshot-equal.

## 6. Policy gate

For the current CPT run, do not change `configs/cleaning_policy.json`: it is
`audit_only`, both structural materialization flags are false, and Stage58 must
record a deterministic no-op. Existing classifier supervision is LLM silver,
not human gold. The exact 2,000-document joint ToC+BIB `STRUCT_2K` handoff is
recovered and checksum-locked. The CPU importer is ready to emit a receipt-bound
1,392-document source after physically excluding all 608 historical-test
documents, but no Clariden import, N1 profile, joint ladder, C0 Rust parity or
production-selection job has run. Nobody is being asked to annotate 2,000 lines
or documents.

Any later run that proposes structural deletion must use a separately reviewed
commit whose policy was approved and frozen **before Stage10 starts**. Do not
edit policy in the middle of a run. Before that future approval, require:

1. a receipt-bound, source-balanced LLM-silver comparison split for Greek PhD,
   OpenArchives, Kallipos and Pergamos (not a new human annotation set);
2. exact bibliography-only, ToC-only and union token deltas;
3. separate Greek, Latin and polytonic removed-character mass alongside exact
   bibliography token loss;
4. emptied/near-emptied document review;
5. a Diavgeia report separating boilerplate spans, excluded documents, PII
   replacements and template/downsampling loss;
6. a recorded decision for every source/profile pair.

Before Stage52 in such a run, complete the joint C0/C1/C2/N1 ladder and record
an explicit classifier-selection receipt. C0 already has a Rust implementation,
but still needs fresh parity on the imported source. C1, C2 and N1 are Python
research arms: selecting any of them requires a separate reviewed Rust
port/export and exact parity package before detection or promotion.

That future run must also complete Stage52 detection, the Stage53 deletion-
safety packet of exactly 100 cases (50 ToC and 50 bibliography), manual review,
and Stage54 promotion. The 100 cases assess only whether predicted deletions eat
running prose/main text or catastrophically damage a document; they are not a
new training corpus and are not automatic/LLM adjudication. A clean audit alone
does not override the pre-authorized policy.

## 7. Run the receipt-bound production CPU DAG

Choose one stable run ID. Do not change it on retries:

The normalizer schedules receipt-bound tasks below 2 GiB through the ordinary
pool and tasks at or above 2 GiB through a separate pool capped at two workers.
The pools run sequentially so ordinary concurrency cannot overlap the
high-memory Parquet conversions. Override these execution-only limits with
`NORMALIZE_LARGE_TASK_BYTE_THRESHOLD` and `NORMALIZE_LARGE_TASK_WORKERS`; they
do not alter canonical bytes, immutable receipt identities or resume validity.
The global DuckDB inventory pass defaults to a 240 GB limit and 16 threads.

If normalization is interrupted after committing some file receipts, preserve
the run directory and use `clariden/submit.sh resume normalize` with the same
run ID and acquisition receipt. Resume validates committed shards, removes only
uncommitted hidden partial directories, and recreates missing file tasks. It is
safe to lower the execution-only worker caps for that retry.

```bash
export PIPELINE_RUN_ID=full-corpus-v2-<date>
export ACQUISITION_RECEIPT="$RUN_ROOT/source_locks/sources_<fresh-timestamp>_<fresh-job>.receipt.json"
unset ACQUISITION_JOB_ID  # the fresh existing-only receipt already exists
bash clariden/submit.sh chain-to-review
CONFIRM_LAUNCH=1 bash clariden/submit.sh chain-to-review
```

This creates Slurm `afterok` dependencies for `10-normalize`, `20-lineage` and
`30-review-packet`, and then stops. Lineage and review stream canonical Parquet
directly; no redundant full-corpus JSONL copy is materialized. Every normalized
row preserves the exact upstream `source_dataset`, candidate `source_id` and a
work-level `work_id`.

### Review the redacted packet on the Mac

From the authenticated Mac, copy exactly the three packet files plus the
text-free novelty summary—never a corpus directory—and run the repository's
resumable reviewer. Replace the run ID but keep the paths and model settings:

```bash
export PIPELINE_RUN_ID=full-corpus-v2-<date>
export REMOTE_PACKET_ROOT="/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/pipeline_runs/$PIPELINE_RUN_ID/stages/30-review-packet"
export LOCAL_REVIEW_ROOT="$HOME/cpt-review-runs/$PIPELINE_RUN_ID/pre_clean"
mkdir -p "$LOCAL_REVIEW_ROOT"
for name in requests.jsonl summary.json stage_receipt.json; do
  rsync -av "clariden:$REMOTE_PACKET_ROOT/$name" "$LOCAL_REVIEW_ROOT/"
done
rsync -av \
  "clariden:/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/pipeline_runs/$PIPELINE_RUN_ID/stages/20-lineage/source_novelty.json" \
  "$LOCAL_REVIEW_ROOT/"
```

Run a bounded smoke in its own immutable output directory:

```bash
cd /path/to/train-apertus-with-glossapi/subprojects/05_token_distillation_cpt/04_full_corpus_preparation
mkdir -p "$LOCAL_REVIEW_ROOT/smoke"
python3 scripts/run_codex_source_reviews.py \
  --requests "$LOCAL_REVIEW_ROOT/requests.jsonl" \
  --output "$LOCAL_REVIEW_ROOT/smoke/responses.jsonl" \
  --manifest "$LOCAL_REVIEW_ROOT/smoke/run_manifest.json" \
  --state-dir "$LOCAL_REVIEW_ROOT/smoke/state" \
  --model gpt-5.6-luna \
  --reasoning-effort low \
  --batch-size 12 \
  --workers 2 \
  --max-requests 12
```

The smoke is interface validation only and must not be aggregated. Run the full
packet with the same frozen settings:

```bash
mkdir -p "$LOCAL_REVIEW_ROOT/full"
python3 scripts/run_codex_source_reviews.py \
  --requests "$LOCAL_REVIEW_ROOT/requests.jsonl" \
  --output "$LOCAL_REVIEW_ROOT/full/responses.jsonl" \
  --manifest "$LOCAL_REVIEW_ROOT/full/run_manifest.json" \
  --state-dir "$LOCAL_REVIEW_ROOT/full/state" \
  --model gpt-5.6-luna \
  --reasoning-effort low \
  --batch-size 12 \
  --workers 2
```

If the full command is interrupted before it writes `run_manifest.json`, rerun
that exact command. Completed batches are checksum-keyed in `full/state` and are
reused after schema/identity validation. Once the immutable manifest exists,
do not rerun over it.

Before creating a Clariden stage, make a separate resolution file and run the
aggregator locally as a diagnostic:

```bash
cp "$LOCAL_REVIEW_ROOT/full/responses.jsonl" \
  "$LOCAL_REVIEW_ROOT/resolved_responses.jsonl"
python3 scripts/aggregate_source_reviews.py \
  --requests "$LOCAL_REVIEW_ROOT/requests.jsonl" \
  --packet-summary "$LOCAL_REVIEW_ROOT/summary.json" \
  --reviews "$LOCAL_REVIEW_ROOT/resolved_responses.jsonl" \
  --novelty-summary "$LOCAL_REVIEW_ROOT/source_novelty.json" \
  --review-policy configs/source_review_policy.json \
  --output "$LOCAL_REVIEW_ROOT/diagnostic_admission.json" \
  --allow-incomplete
jq '.pending_adjudications' "$LOCAL_REVIEW_ROOT/diagnostic_admission.json"
```

Do not submit Stage40 until this is zero. If it is nonzero, use only the listed
redacted samples to obtain separate schema-valid `adjudicator` responses, merge
those into a new resolved response JSONL without altering the manifest-bound
primary/secondary output, and rerun the same diagnostic command. This prevents
a failed Stage40 from binding an incomplete response-file hash.

Copy back only the responses and reviewer manifest to a new response directory:

```bash
export REMOTE_RESPONSE_ROOT="/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/reviewer_responses/$PIPELINE_RUN_ID/pre_clean"
ssh clariden "mkdir -p '$REMOTE_RESPONSE_ROOT'"
rsync -av "$LOCAL_REVIEW_ROOT/resolved_responses.jsonl" \
  "$LOCAL_REVIEW_ROOT/full/run_manifest.json" \
  "clariden:$REMOTE_RESPONSE_ROOT/"
```

Back on Clariden, submit Stage40 exactly once with the zero-pending resolved
response file:

```bash
export REMOTE_RESPONSE_ROOT="/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/reviewer_responses/$PIPELINE_RUN_ID/pre_clean"
export REVIEWS_JSONL="$REMOTE_RESPONSE_ROOT/resolved_responses.jsonl"
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

Stage50 now applies the confirmed source decisions, narrow source-specific
cleaning and high-confidence PII masking. It never applies ToC/bibliography
spans. If every admitted source is already `include`, the chain stops after
Stage50. If any source is `include_after_cleaning`, it builds a fresh redacted
packet from only those Stage50 source datasets and stops there instead. Neither
path submits Stage58 or any downstream job: the structural audit/promotion
track remains free to finish before the irreversible finalization choice.

For that optional boundary, repeat the safe Mac smoke/full/resume protocol above
with remote stage `55-post-clean-review-packet`, local phase `post_clean`, and a
distinct remote response directory ending in `/post_clean`. Its local diagnostic
omits `--novelty-summary` because novelty was a pre-clean gate. Require zero
pending adjudications before copying the resolved response file back. Then
aggregate and inspect the merged terminal admission:

```bash
export POST_CLEAN_REVIEWS_JSONL=/path/to/post_clean_responses.jsonl
CONFIRM_LAUNCH=1 bash clariden/submit.sh post-clean-aggregate

export FINAL_SOURCE_ADMISSION="$PIPELINE_RUNS_ROOT/$PIPELINE_RUN_ID/stages/56-post-clean-review-aggregate/final_admission_candidate.json"
export CONFIRM_FINAL_ADMISSION_SHA256="$(sha256sum "$FINAL_SOURCE_ADMISSION" | awk '{print $1}')"
export REUSE_STAGE50_ADMISSION=0
bash clariden/submit.sh chain-after-post-clean
CONFIRM_LAUNCH=1 bash clariden/submit.sh chain-after-post-clean
```

`chain-after-post-clean` only validates the terminal admission and prints the
finalization stop boundary; it submits no jobs. If no post-clean review was
needed, prepare the same terminal inputs from the confirmed Stage40 admission:

```bash
export FINAL_SOURCE_ADMISSION="$SOURCE_ADMISSION"
export CONFIRM_FINAL_ADMISSION_SHA256="$CONFIRM_ADMISSION_SHA256"
export REUSE_STAGE50_ADMISSION=1
```

Stage58 is required, but deliberately never automatic. It consumes the exact
Stage50 corpus and ledger and does not rerun source cleanup or PII detection.
Choose exactly one separate finalization command. For the current run's frozen
`audit_only` policy, explicitly confirm the deterministic no-op:

```bash
export CONFIRM_STRUCTURAL_NOOP=1
bash clariden/submit.sh chain-finalize-noop
CONFIRM_LAUNCH=1 bash clariden/submit.sh chain-finalize-noop
```

A run whose cleaning policy was already frozen before Stage10 as approved for
both structural heads may instead apply exact Stage54-promoted spans:

```bash
export STRUCTURAL_MODEL_RECEIPT="$PIPELINE_RUNS_ROOT/$PIPELINE_RUN_ID/stages/54-structural-promote/academic_structural_model_receipt.json"
export STRUCTURAL_SPANS="$PIPELINE_RUNS_ROOT/$PIPELINE_RUN_ID/stages/54-structural-promote/structural_spans.jsonl"
export CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256="$(sha256sum "$STRUCTURAL_MODEL_RECEIPT" | awk '{print $1}')"
bash clariden/submit.sh chain-finalize-promoted
CONFIRM_LAUNCH=1 bash clariden/submit.sh chain-finalize-promoted
```

The promoted path fails rather than degrading to a no-op if Stage54, its exact
receipt, spans, policy approval, or safety evidence is missing. Stage58 freezes
the requested mode and model-receipt SHA-256 in an immutable request and its
decision. An interrupted resume must provide the same choice and exact receipt;
a different choice requires a new `PIPELINE_RUN_ID`.

For example, resume an interrupted no-op Stage58 only with the same terminal
admission variables prepared above and the same explicit choice:

```bash
APPLY_STRUCTURAL=0 CONFIRM_STRUCTURAL_NOOP=1 \
  bash clariden/submit.sh resume final-clean
CONFIRM_LAUNCH=1 APPLY_STRUCTURAL=0 CONFIRM_STRUCTURAL_NOOP=1 \
  bash clariden/submit.sh resume final-clean
```

For a promoted resume, use `APPLY_STRUCTURAL=1` plus the same
`STRUCTURAL_MODEL_RECEIPT`, `STRUCTURAL_SPANS`, and
`CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256` values instead.

The downstream semantic order is Stage58, frozen GreekMMLU decontamination, the existing
`glossapi-corpus dedup-text run` CLI and final materialization/validation. Dedup
is explicit and receipt-bound: preserved Greek diacritics, exact plus near
dedup, 128 MinHash permutations, 32 × 4 bands, token 5-shingles, threshold 0.85
and a 5,000-member bucket ceiling. The independent GreekMMLU freeze job may run
concurrently earlier, but decontamination cannot consume it until both it and
Stage58 pass. Publication is not submitted.

Check or resume without inventing a new run:

```bash
bash clariden/submit.sh status "$PIPELINE_RUN_ID"
FINAL_CLEAN_STAGE=58-final-clean bash clariden/submit.sh resume decontam
CONFIRM_LAUNCH=1 FINAL_CLEAN_STAGE=58-final-clean bash clariden/submit.sh resume decontam
```

A completed receipt makes a resubmitted stage a validated no-op only after the
same structural finalization request is revalidated. An incomplete stage
requires explicit resume with the same no-op/apply choice (and the same exact
promoted receipt for apply); downstream stages reject missing, mismatched or
drifted receipts even if Slurm reported success.

## 8. Optional gated Hugging Face publication

Publication is standalone and never part of a chain. First inspect the release
manifest, exact token waterfall and validation receipt. Then inject the token
only into the submission environment, never a CLI argument or file:

```bash
export HF_REPO_ID=fffoivos/glossapi-greek-cpt-redistributable-delta-v2
export CONFIRM_PUBLISH="$HF_REPO_ID"
# Run on Clariden after an explicit secure HF_TOKEN handoff.
HF_TOKEN="$HF_TOKEN" CONFIRM_LAUNCH=1 bash clariden/submit.sh publish
```

Do not put the token in a file, command argument or receipt. The publisher
accepts only manual gating and a new/empty repository. It uploads the
checksum-bound `README.md`, the validated
redistribution Parquet inventory and the exact provenance receipts, then
verifies that complete remote inventory at the returned commit. This is the
license-limited **redistributable delta**, not the full private CPT corpus.
Gating does not override source redistribution restrictions.

If a failed upload left any payload, the publisher deletes nothing and refuses
to continue. The temporary upload cache is intentionally discarded, so partial
uploads are not resumed. Inspect the remote repository, then delete/recreate it
manually or choose a new empty `HF_REPO_ID` before rerunning.

## Script-interface boundary

The orchestration now binds implemented normalization, lineage, review,
cleaning, structural-finalization, decontamination, deduplication,
materialization, validation and publication entry points centrally in
`clariden/paths.env`. Every launcher fails before data access if its expected
script is absent, and receipts bind the exact clean Git commit and input
inventories. Do not override an entry point during a production run.

## Clariden resource policy

Clariden does not provide this project with a CPU-only production compute queue.
Production preprocessing therefore uses the CPU side of a GH200 node on `normal`,
requests no GPU/GRES, and aims to keep about 256 of 288 cores useful.
Use `debug` only for bounded smoke tests. Use `xfer` only for internal filesystem
transfer.
