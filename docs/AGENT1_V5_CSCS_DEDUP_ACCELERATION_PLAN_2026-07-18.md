# Agent 1 v5 CSCS deduplication acceleration plan

**Plan timestamp:** 2026-07-18 12:57 Europe/Athens
**Target:** sustainable 4–5× signature-stage speed-up with no change to dedup semantics
**Execution status:** **plan only; no live change has been applied**

## Operational baseline and scope

The combined pre-dedup dataset was constructed successfully. Acquisition
integrity, construction manifests, transform→GlossAPI→release metadata
lineage, the 44-column ZSTD Parquet release, and private HF storage all passed
their operational checks. The serial signature chain is healthy and resumable.

This plan changes **execution mechanics only**. It must not rebuild, refilter,
or otherwise change the dataset, schema, metadata, MinHash recipe, or NanoChat
protection semantics. The `greek>60` distribution and other quality counters
are deferred, non-blocking observations and are not acceleration prerequisites.

> **Execution directive:** keep the current chain running. Do not stop, cancel,
> rebuild, or refilter because of a deferred observation. Alter the chain only
> through the safe-boundary migration below after the technical preflight and
> cutover approval pass.

## Required execution changes

1. Perform one complete release audit instead of rereading all 149.746 GB
   before every rank.
2. Validate only the assigned input shard inside each signature rank.
3. Add immutable full-audit, manifest, code, assigned-input, and exact 32-output
   bindings to each new receipt while retaining legacy-receipt compatibility.
4. Benchmark 1→2→4→5 workers on one `normal` node and select four workers if
   five misses the safety gates.
5. Cut over through the held-fence procedure at a passed receipt boundary,
   without cancelling a running rank.
6. Continue downstream stages with bounded canaries and complete receipt and
   output closure.
7. Materialize and checksum-verify the final private deduplicated HF dataset.

## Technical preflight

The migration tooling must require an immutable:

```text
<run>/dedup_acceleration_preflight.json
```

It must record `status="passed"`,
`scope="current_run_dedup_acceleration_pre_fence"`,
`construction_integrity_status="passed"`,
`blocking_acceleration_findings=[]`, and explicit approval to place the fence.
These fields refer only to input/manifest/receipt/runtime/scheduler/compatibility
failures. They do not import deferred quality policy. The receipt must SHA-256
bind:

- `run_contract.json`;
- the combined input manifest;
- the private HF base publication and current metadata-publication receipts as
  construction evidence, without making live HF availability a compute-time
  dependency;
- the frozen acquisition-integrity audit;
- the exact-index manifest;
- the DataTrove runtime receipt;
- the one-time full-input audit;
- the metadata-lineage evidence;
- the acceleration code/config and candidate rank plan;
- a refreshed `debug` partition, `debug-qos`, account, queue, and ownership
  snapshot;
- the legacy-receipt compatibility validator and 32-output validation logic.

The exact final legacy boundary cannot be known at preflight time and must not
be placed in this receipt.

After the fence captures the boundary, write a second immutable receipt:

```text
<run>/dedup_acceleration_cutover.json
```

It must bind the preflight receipt, fence job, actual final legacy job/rank,
every contiguous legacy signature receipt and its 32 output hashes through that
rank, the expected successor-submit rejection, absence of a successor, the
first missing rank, and the fence cleanup result. The accelerated runner must
require both receipts. This split prevents new legacy completions between
preflight and fence placement from escaping the closure audit.

## Non-blocking observations

The following observations are kept separate from the required changes above:

- the quality audit receipt's release-oriented `status="blocked"` and its
  `greek>60`/structural counters are deferred quality-policy information;
- queue, fair-share, free-node, and elapsed-time figures below are timestamped
  measurements and must be refreshed before cutover.

Neither item is a preflight prerequisite or a stop-work instruction. The
acceleration runner must not contain a filter-policy change.

## Live CSCS evidence

At the `2026-07-18 15:20 EEST` evidence snapshot:

- passed signature receipts were contiguous ranks `0..48`: **49 / 431**;
- job `2790126`, rank 49, was healthy on `debug`;
- `normal` had a 12-hour limit and exclusive nodes;
- association was `clariden|a0140|fffoivos||normal`;
- the `debug` partition's `debug-qos` allowed one running and two submitted
  jobs per user;
- QoS `normal` exposed no explicit per-user job/TRES maximum;
- the user's fair-share factor was `0.198315`;
- a one-node allocation yielded 288 CPUs, 450 GiB memory, and four GH200 GPUs,
  even when the job requested only 16 CPUs.

Visible idle nodes are not authorization to use them. This plan deliberately
uses one node.

Storage evidence:

- Capstor user usage: about 69.3 TB / 150 TB;
- inode usage: 499,466 / 1,000,000;
- projected remaining signature output: about 52.3 GB and 12,738 files;
- Iopsstor was more heavily occupied and is not the target for this sequential
  workload.

## Measured bottleneck

Each signature task currently calls `_load_release()`, which SHA-256 validates
all 431 release files before processing its assigned rank. The release contains
149,746,029,389 bytes, so each rank rereads the full corpus.

For ranks 24–44, excluding one reused rank:

| Metric | Observed |
|---|---:|
| Median elapsed | 2,198 s (36:38) |
| p95 elapsed | 2,325 s (38:45) |
| Maximum elapsed | 2,544 s (42:24) |
| Full-corpus validation per rank | about 3m10s–3m17s |
| MaxRSS | below 0.8 GiB |
| Effective CPU use | about one core |
| Slurm allocation | exclusive 288-CPU / 4-GPU node |
| Physical read reported per rank | about 143,300 MiB |

At the rank-44 boundary, 386 ranks remained, covering 45,406,718 documents and
125.67 GB of assigned input. The serial estimate was about **9.82 days**, with
about **57.8 TB** of redundant validation reads.

## Target architecture

Use **one `normal` node** with a bounded local worker pool:

| Resource | Target |
|---|---:|
| Nodes | 1 |
| Worker processes | test 1 → 2 → 4 → 5; never exceed 5 |
| CPUs requested | 32 |
| Memory requested | 64 GiB |
| Accelerated chunk walltime | at most 11:30:00 |
| GPUs used | 0 |
| Threading per worker | 1 |

Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
equivalent limits. Five workers should remain below five effective CPU cores
and five GiB RSS based on the measured tasks. The purpose of the 32-CPU request
is bounded headroom and monitoring, not permission for hidden fan-out.

Do not submit five one-node jobs. Clariden nodes are exclusive; that would
waste 20 GPUs and most of 1,440 allocated CPU cores.

## Integrity-preserving implementation

Refactor release validation into three layers:

1. **Manifest structure:** validate schema, passed status, contiguous ranks,
   root binding, and inventory shape without reading all payload bytes.
2. **One full input audit:** read and SHA-256 all 431 inputs exactly once and
   write an immutable audit receipt bound to the combined manifest.
3. **Per-rank task validation:** verify only the assigned input's size/SHA,
   require the full-audit receipt, and process that rank.

Each signature receipt must add bindings to:

- full-input-audit SHA;
- combined-manifest SHA;
- runtime-receipt SHA;
- deployed-code SHA;
- assigned-input SHA;
- exact 32-file output inventory.

The existing receipt schema remains compatible through additive fields so that
every completed rank remains reusable by the legacy serial runner.

No semantic parameter may change:

```text
language: ell_Grek
Greek diacritics: preserved
token shingles: 5
MinHash permutations: 128
bands × hashes: 32 × 4
seed: 1
hash precision: 64
verified Jaccard: 0.85
NanoChat origin preference: protected
```

Projected remaining I/O after the change is about 453 GB: one 149.75 GB full
audit, about 251 GB of per-rank checksum/Parquet reads, and about 52 GB of
signature writes.

## Staged benchmark

Freeze 24 unstarted, homogeneous NanoChat ranks with 196,608 rows and similar
compressed sizes. Bind the exact rank list and input hashes in the experiment
receipt.

| Phase | Workers | Ranks | Waves |
|---|---:|---:|---:|
| New-code baseline | 1 | 2 | 2 |
| Concurrency canary | 2 | 4 | 2 |
| Scale canary | 4 | 8 | 2 |
| Target canary | 5 | 10 | 2 |

For every rank and wave, record:

- queue delay and wall time;
- rows, documents/s, and normalized input bytes/s;
- `/proc/<pid>/io` read/write counters;
- CPU, RSS/HWM, and open-file count;
- Slurm `ElapsedRaw`, `TotalCPU`, `MaxRSS`, `MaxDiskRead`, and `MaxDiskWrite`;
- output bytes/files/inodes and all 32 output checksums;
- metadata syscalls using `strace -c` if available, otherwise exact touched-file
  counts and a receipt noting that `strace` was unavailable;
- filesystem, node, QoS, and application warnings.

### Safety gates and worker selection

These universal gates must pass for every phase used in the decision:

- zero failed tasks, missing outputs, or receipt/checksum drift;
- aggregate RSS < 32 GiB;
- sustained five-minute read average < 1 GiB/s;
- sustained five-minute write average < 500 MiB/s;
- no uncontrolled CPU use beyond 32 cores;
- no `EIO`, `ENOSPC`, `EDQUOT`, `ESTALE`, Lustre timeout, OOM, node failure,
  or scheduler policy warning.

The immutable benchmark receipt must then select workers by this order:

1. Require the 2-worker diagnostic to reach at least 1.70× baseline.
2. Mark four workers eligible only if aggregate throughput is at least 3.40×
   baseline and median normalized per-worker elapsed is no more than 15% slower
   than baseline.
3. Select five workers only if its aggregate throughput is at least 4.25×
   baseline, its median normalized per-worker elapsed is no more than 15%
   slower, and its universal gates pass.
4. Otherwise select four if four is eligible and its universal gates pass.
5. If four is not eligible, do not accelerate; resume the legacy serial chain
   from the first missing rank through the validated rollback path.

The approved receipt must record the baseline, every phase metric/gate result,
`selected_workers` (4 or 5), and the deterministic selection reason. Never test
or run more than five workers for this run.

Repeat a small 2→4→5 canary when the ranks transition from NanoChat to candidate
shards. Run input shards larger than 1.5 GB at fan-out 1 or 2 unless a dedicated
large-shard canary proves more is safe.

## Safe migration boundary

The current helper does **not** keep a successor pending: it runs one rank and
submits the successor only after that rank's receipt passes. Polling for a
pending successor is therefore racy and is not an acceptable migration method.

After the technical preflight passes and cutover is explicitly approved,
reserve the second debug-partition submission slot with one held fence job
while the current chain rank is still running. With the partition QoS limits
`MaxJobsPU=1` and `MaxSubmitJobsPU=2`, the running rank plus the held fence fill
both submission slots. After writing and validating its receipt, the legacy
helper's successor `sbatch` is rejected as a third submitted job. No running
rank is cancelled and no receipt is lost.

### Versioned cutover script

Implement the cutover as an immutable Bash script, not as commands pasted into
an interactive shell. It must begin with:

```bash
#!/usr/bin/env bash
set -eEuo pipefail
```

Immediately before placing the fence, the script must re-confirm:

- `PartitionName=debug` is bound to `QoS=debug-qos`;
- `debug-qos` has exactly `MaxJobsPU=1` and `MaxSubmitJobsPU=2`;
- there is exactly one submitted debug job for `fffoivos`;
- that job belongs to account `a0140`, is named
  `a1v5-signature-chain-r<N>`, is `RUNNING`, and has `N < 430`;
- its output/error paths are under the expected coordination root;
- the technical preflight receipt and its cutover approval pass.

Submit exactly one held job named `a1v5-signature-chain-fence` on
`debug`/`a0140`. Immediately after obtaining `fence_id`, install
`ERR`/`INT`/`TERM`/`EXIT` cleanup. Cleanup may cancel only that exact ID after
re-verifying through `scontrol show job` that it is owned by `fffoivos`, belongs
to `a0140`, is on `debug`, has the exact fence name and coordination-root log
paths, and remains `PENDING` with reason `JobHeldUser`. On any identity
mismatch, report it and do not cancel another job. Disarm cleanup only after a
valid boundary is accepted and the fence is deliberately removed.

### Post-submission state machine

Fence submission and immediate re-resolution must handle all valid races:

1. **Fence submission rejected:** do not cancel or change any chain job. A
   predecessor/successor occupied both submission slots. Record the rejected
   attempt, let the chain continue, and retry early in a later rank.
2. **Held fence plus one chain job:** allow the chain job to be `RUNNING` or
   `PENDING`; bind its exact ID and rank as the actual fenced job. If it differs
   from the initially observed job, first validate every intervening passed
   receipt and its 32 output hashes.
3. **Held fence only:** the initially observed job may have completed, written
   its receipt, failed only at successor submission, and disappeared from
   `squeue`. Query `sacct` and accept this boundary only if that job's receipt
   passes, all 32 output hashes pass, stderr contains the expected submission
   limit rejection, and no successor exists.
4. **Any other queue state:** abort the migration path and let the guarded
   cleanup remove only the validated held fence.

While a chain job remains, continuously assert that the fence is still held.
When the actual fenced job exits, accept its expected non-zero Slurm status only
if its immutable receipt and 32 output bindings pass, its only error is the
expected successor-submit rejection, and no successor exists. Never cancel a
running or pending signature rank.

Then:

1. validate contiguous legacy receipts and 32 output hashes from rank 0 through
   the final legacy rank;
2. freeze the first missing rank as the acceleration boundary;
3. retain the validated boundary evidence in a temporary, receipt-bound staging
   object; do not finalize the cutover receipt yet;
4. cancel only the still-held, identity-verified fence and assert no
   signature-chain or fence job remains queued;
5. atomically write and validate the final immutable
   `dedup_acceleration_cutover.json`, including the fence-cancellation and
   empty-queue evidence, then disarm cleanup;
6. deploy the acceleration code into a new immutable commit-named directory;
7. run the 1→2→4→5 benchmark and approve its receipt before accelerated chunks.

Abort the migration path if the partition/QoS values change, an unrelated debug
job appears, the fence identity/hold changes, receipt/output closure fails, or
the chain failure is anything other than the expected submit-limit rejection.
The abort/rollback path preserves every valid receipt and signature file.

Never overwrite the existing `942b8d2` deployment or the separate legacy
chain helper. The new immutable helper/normal runner must support a stop
sentinel at entry and before submitting more work, so this QoS fence is only a
one-time bridge away from the legacy helper.

## Accelerated signature runner

Split the remaining ranks into immutable chunks sized from the benchmark p95,
initially no more than 60 ranks per chunk. The immutable benchmark receipt must
select four or five workers, and the chunk plan must bind that receipt and the
same value. Submit through a versioned Bash wrapper with strict mode; do not
paste the body interactively. Throttle the array to one node:

```bash
#!/usr/bin/env bash
set -eEuo pipefail

run=/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/agent1-v5-clariden-debug-20260715T111552Z-30c72e9
coord=/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/.agent1-v5-clariden-debug-20260715T111552Z-30c72e9.coord
benchmark_receipt="$run/dedup_acceleration_benchmark.json"
cutover_receipt="$run/dedup_acceleration_cutover.json"
full_audit_receipt="$run/dedup_full_input_audit.json"
combined_manifest="$run/release-pre-dedup/manifests/combined_manifest.json"
chunks="$run/dedup_acceleration_chunks.json"
submission_receipt="$run/dedup_acceleration_submission.json"
release_authorization="$run/dedup_acceleration_release_authorization.json"
submission_nonce="$(tr -d '-' < /proc/sys/kernel/random/uuid)"

selected_workers="$(jq -er 'select(.status == "passed" and .approved == true) | .selected_workers | select(. == 4 or . == 5)' "$benchmark_receipt")"
benchmark_sha256="$(sha256sum "$benchmark_receipt" | awk '{print $1}')"
cutover_sha256="$(sha256sum "$cutover_receipt" | awk '{print $1}')"
full_audit_sha256="$(sha256sum "$full_audit_receipt" | awk '{print $1}')"
combined_manifest_sha256="$(sha256sum "$combined_manifest" | awk '{print $1}')"

new_pipeline="$(realpath -e "$(jq -er '.deployed_code_root' "$chunks")")"
case "$new_pipeline" in
  /capstor/scratch/cscs/fffoivos/agent1-v5-code/*) ;;
  *) exit 1 ;;
esac
runner="$new_pipeline/slurm/agent1_v5_eiger/normal_signature_runner.sh"
test -f "$runner"
runner_sha256="$(sha256sum "$runner" | awk '{print $1}')"
last_chunk="$(jq -er '.last_chunk | select(. >= 0 and floor == .)' "$chunks")"

jq -e \
  --argjson workers "$selected_workers" \
  --arg benchmark_sha256 "$benchmark_sha256" \
  --arg cutover_sha256 "$cutover_sha256" \
  --arg full_audit_sha256 "$full_audit_sha256" \
  --arg combined_manifest_sha256 "$combined_manifest_sha256" \
  --arg runner_sha256 "$runner_sha256" \
  '.status == "passed"
   and .selected_workers == $workers
   and .benchmark_receipt_sha256 == $benchmark_sha256
   and .cutover_receipt_sha256 == $cutover_sha256
   and .full_input_audit_sha256 == $full_audit_sha256
   and .combined_manifest_sha256 == $combined_manifest_sha256
   and .runner_sha256 == $runner_sha256' \
  "$chunks" >/dev/null
chunks_sha256="$(sha256sum "$chunks" | awk '{print $1}')"

submitted="$(
  sbatch --parsable --hold --uenv-passthrough=ignore \
    --account=a0140 --partition=normal \
    --comment="agent1-v5-dedup-accel:${submission_nonce}" \
    --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G \
    --time=11:30:00 --signal=B:USR1@900 \
    --array="0-${last_chunk}%1" \
    --job-name="a1v5-signature-normal-c${selected_workers}" \
    --output="$coord/slurm/%x-%A_%a.out" \
    --error="$coord/slurm/%x-%A_%a.err" \
    --export=ALL,RUN_ROOT="$run",PIPELINE_ROOT="$new_pipeline",CHUNK_PLAN="$chunks",CHUNK_PLAN_SHA256="$chunks_sha256",WORKERS="$selected_workers",SUBMISSION_NONCE="$submission_nonce",SUBMISSION_RECEIPT="$submission_receipt",RELEASE_AUTHORIZATION="$release_authorization" \
    "$runner"
)"
array_job_id="${submitted%%;*}"
[[ "$array_job_id" =~ ^[0-9]+$ ]]
```

Before `sbatch`, the wrapper must atomically write a pending submission journal
binding every value checked above, the chunk-plan SHA, nonce, and exact array
specification. The array is submitted **held** and tagged with the same nonce in
its Slurm comment. Immediately after obtaining its ID, guarded cleanup may
cancel only the exact still-held array after re-verifying nonce, owner, account,
partition, job name, array bounds, resources, and log paths.

The wrapper must verify the held array through `scontrol`, atomically finalize
`dedup_acceleration_submission.json` with the numeric `array_job_id`, and then
atomically write `dedup_acceleration_release_authorization.json` binding the
submission-receipt SHA, job ID, nonce, chunk-plan SHA, and runner SHA. Only
after both immutable receipts exist may it release the exact array. It then
records the observed release state separately. Every array task must validate
its own `SLURM_ARRAY_JOB_ID`, nonce, submission receipt, and release
authorization before claiming a rank.

If interrupted before release, recovery finds at most one held array by the
journal-bound nonce and repeats exact identity verification. If interrupted
after release, the required authorization already exists, so running work is
not orphaned. Recovery must never infer a job from a name alone.

The runner must provide:

- atomic per-rank claims;
- non-overlapping frozen rank assignments;
- rank-specific stdout/stderr and metrics;
- idempotent reuse of passed receipts;
- a 15-minute pre-walltime drain that starts no new work;
- quarantine, never deletion, of unreceipted partial outputs;
- a final 431-receipt / 13,792-output closure check.

## Rollback

Implement rollback as a separate immutable Bash script with
`set -eEuo pipefail`; do not rely on variables from the submission shell. It
must:

1. Load `<run>/dedup_acceleration_submission.json`, verify its SHA binding and
   passed status, extract a numeric `array_job_id`, and resolve the exact run and
   coordination roots from that receipt.
2. Re-query the array and verify owner `fffoivos`, account `a0140`, partition
   `normal`, expected dynamic job name, array bounds, and chunk-plan identity
   before holding or cancelling anything.
3. Hold future array work, write the validated stop sentinel, cancel only
   pending tasks belonging to that exact array, and allow running local workers
   to drain and close their current rank receipts. Never cancel an active rank.
4. Wait until no accelerated signature worker is active, then audit every
   receipt/output set. Move only unreceipted partial outputs to a timestamped
   quarantine directory.
5. Compute `next` with the receipt validator as the first missing rank, assert
   it is numeric and in `0..430`, and prove every lower rank has a passed
   receipt with 32 matching output hashes. If no rank is missing, do not submit
   the legacy chain.
6. Verify the immutable legacy helper and pipeline identities at:

   ```text
   /capstor/scratch/cscs/fffoivos/agent1-v5-code/signature-chain/subprojects/05_token_distillation_cpt/04_full_corpus_preparation/scripts/run_signature_task_chain.sh
   /capstor/scratch/cscs/fffoivos/agent1-v5-code/942b8d2/subprojects/05_token_distillation_cpt/04_full_corpus_preparation
   ```

7. Prewrite a rollback-submission journal with a unique nonce, submit exactly
   one `a1v5-signature-chain-r${next}` job **held** on `debug`/`a0140` with that
   nonce in its Slurm comment, and verify its complete identity while held.
   Atomically write a rollback submission receipt and release authorization
   binding the array shutdown, receipt audit, computed `next`, helper/code
   hashes, nonce, and new legacy job ID; only then release it. Guarded cleanup
   and interrupted recovery may act only on the exact nonce-bound held job,
   never on a job inferred from its name.

All valid completed receipts and signature files remain reusable.

## Downstream dedup stages

Apply the same bounded one-node design only after stage-specific canaries:

- merge-signatures: serial;
- 32 bucket tasks: 1→2→4, five only if metadata traffic passes;
- pair merge and clustering: serial SQLite writers;
- 431 shingle and 431 filter tasks: up to five local workers;
- 128 verifier tasks: 2→4→5 canary because of random SQLite reads.

Node-local SQLite is allowed only after a capacity canary and must be checksummed
before atomic promotion to Capstor. Any LSH group above 5,000 documents blocks
the release until explicitly resolved; it may not be silently excluded while a
manifest passes.

## Expected result and final acceptance

The measured signature-stage estimate is **42–49 hours after migration**, versus
about **9.82 days serial**, using one exclusive node. A precise end-to-end dedup
ETA is not yet defensible because downstream pair cardinality is unknown.

Final acceptance requires:

- 431 passed signature receipts and 32 valid files per receipt;
- complete bucket, shingle, verifier, and filter manifests;
- no unresolved oversized group;
- NanoChat protection and representative-policy closure;
- a reasoned decision ledger with exact/near method, hashes, verified Jaccard,
  threshold, kept/dropped IDs, and representative;
- unchanged schema and metadata for retained rows;
- per-source row/character/token waterfalls;
- private HF final-release publication and byte/checksum verification at an
  immutable commit.
