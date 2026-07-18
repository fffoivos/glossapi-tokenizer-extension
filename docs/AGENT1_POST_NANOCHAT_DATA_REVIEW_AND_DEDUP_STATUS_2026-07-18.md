# Agent 1 post-NanoChat data review, transformation, and deduplication status

**Status timestamp:** 2026-07-18, while Clariden job `2789351` (`a1v5-signature-chain-r44`) was running.

> **Operational clarification (2026-07-18):** dataset construction through the
> combined pre-dedup release passed, document-to-metadata lineage passed, and
> private HF storage was verified object by object. Deduplication is healthy
> and still running. The `greek>60` quality flags and related content-quality
> findings are explicitly deferred and are **not** a stop condition for the
> current construction or dedup run. See
> `AGENT1_V5_DATASET_AND_HF_READINESS_AUDIT_2026-07-18.md` for the operational
> evidence and `AGENT1_V5_CSCS_DEDUP_ACCELERATION_PLAN_2026-07-18.md` for the
> performance-only acceleration plan. Execution agents must continue the live
> chain unless carrying out the plan's deliberate safe-boundary migration.

This is the operational handoff for the new GlossAPI candidate datasets: their review, structural cleaning, GFM normalization, GlossAPI preparation, NanoChat envelope, pre-dedup release, and current Datatrove deduplication run. It separates durable evidence from historical interactive review artifacts.

## Scope and source roster

The 18 configured new-corpus sources are:

`amna_press`, `archetai`, `artos_zois`, `diavgeia`, `e_nautilia`, `ecclesia`, `eellak_articles`, `elocus`, `ert_press`, `heinrich_boell_publications`, `istorima`, `libduth`, `libiep`, `modern_greek_dictionary`, `national_theatre_press`, `new_sociology`, `open_council`, and `psepheda`.

The authoritative roster and source-field mappings are in:

- `subprojects/05_token_distillation_cpt/04_full_corpus_preparation/configs/agent1_v5_eiger_pipeline.json`
- `subprojects/05_token_distillation_cpt/04_full_corpus_preparation/configs/agent1_v3_candidate_roster.json` (earlier candidate-review roster)

The two original planning documents are:

- `/Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/parallel-agent-plans/01_POST_NANOCHAT_REVIEW_AND_DATA_PREP_AGENT_PLAN.md`
- `/Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/parallel-agent-plans/04_CSCS_20_DOC_CODEX_REVIEW_AND_UNIFIED_CORPUS_PLAN.md`

## 1. Review and source admission

### What was implemented

The review stack samples candidate records, prepares per-document Codex/Terra requests, validates structured scores, aggregates results, and builds a human review site. The raw-review policy deliberately asks reviewers to identify source-logical artifacts (HTML scraping vs PDF/VLM extraction), malformed text, mojibake, VLM repetition, and image-description artifacts.

| Purpose | Source / documentation |
| --- | --- |
| Raw sample extraction and review packet | `scripts/agent1_v4_raw_review.py`, `scripts/export_agent1_v4_raw_review_packet.py` |
| Parallel Terra review requests and response validation | `scripts/run_agent1_v4_terra_reviews.py`, `scripts/validate_agent1_v4_terra_responses.py` |
| Prompt and response contract | `configs/agent1_v4_terra_review_prompt.md`, `schemas/agent1_v4_raw_review_request.schema.json`, `schemas/agent1_v4_terra_review_response.schema.json` |
| Human review site | `scripts/build_agent1_v4_review_site.py`, `scripts/serve_agent1_v4_srun_bridge.py` |
| Earlier Codex review / admission path | `scripts/run_agent1_v3_codex_reviews.py`, `scripts/agent1_v3_review.py`, `scripts/agent1_v3_review_aggregate.py`, `scripts/agent1_v3_admission.py` |
| Review tests | `tests/test_agent1_v4_raw_review.py`, `tests/test_run_agent1_v3_codex_reviews.py`, `tests/test_agent1_v3_review.py`, `tests/test_agent1_v3_review_evidence.py` |

### Evidence and limitation

The current worktree retains the complete executable review stack and its schemas. The downstream run’s durable `candidate_manifest.json` and `glossapi_manifest.json` exist in the run directory listed below, demonstrating that configured candidate rows progressed into the build. The browser-hosted review sites and local forwarded sessions used during manual inspection were transient; a durable exported HTML packet was not found in the current worktree during this review. Do not claim a particular historical review-site URL as durable evidence; regenerate it from the scripts above when needed.

## 2. Repetition removal, image cleanup, and HTML-to-GFM normalization

### Intent and transformation policy

The cleaning stage is deliberately structural rather than semantic:

1. Detect repeated VLM/OCR output, including repeated rows/blocks and non-trivial arithmetic sequences. Do not flag a single repeated character as a repetition defect.
2. Replace detected repetitive spans with `<!-- repeating-text-removed -->`.
3. Remove generated image filename tokens such as `hash_3_img.webp` and retain the inserted VLM description as `<!-- description-of-removed-image: … -->`.
4. Preserve existing Markdown, convert expressible HTML to GitHub-flavored Markdown (including tables), and remove unrepresentable syntax rather than inventing text.
5. Run GlossAPI after structural normalization; do not duplicate semantic cleanup that belongs to GlossAPI.

### Access

| Purpose | Source / documentation |
| --- | --- |
| VLM repetition detector / audit | `scripts/audit_agent1_v4_vlm_repetition.py` |
| Prototype cleaner and GFM transform | `scripts/prototype_agent1_v4_gfm_normalization.py` |
| GFM transformation specification | `docs/agent1_v4_gfm_normalization.md` |
| Unit tests | `tests/test_agent1_v4_gfm_normalization.py` |
| Cleaning / transformation orchestration | `scripts/agent1_v5_pipeline.py`, `scripts/agent1_v5_datatrove.py` |

The documented behavior is intentionally suited to mixed VLM Markdown/HTML: renderable HTML and Markdown semantics are preserved, raw image-file artifacts and unrenderable code-like markup are not.

## 3. GlossAPI processing and NanoChat envelope

### Pipeline contract

The unified record preserves NanoChat’s required top-level fields:

```text
source_dataset
source_doc_id
text
title
author
source_metadata_json
```

`source_metadata_json` retains non-canonical source fields. `title` and `author` remain null/blank if a source has no appropriate field; a missing text field is a hard admission problem rather than a fabricated value. GlossAPI-derived diagnostics are not pre-filled into the NanoChat top-level schema; they are produced by the GlossAPI stage.

| Purpose | Source / documentation |
| --- | --- |
| Source-field profiling and mapping | `scripts/profile_agent1_v4_fields.py`, `schemas/agent1_v4_field_mapping.schema.json` |
| NanoChat materialization | `scripts/materialize_agent1_v4_nanochat_envelope.py` |
| Full task runner | `scripts/agent1_v5_pipeline.py` |
| Datatrove/GlossAPI executor | `scripts/agent1_v5_datatrove.py` |
| Full-pipeline guide | `README.md`, `docs/agent1_v5_eiger_pipeline.md` |
| Runtime configuration | `configs/agent1_v5_eiger_pipeline.json`, `configs/agent1_v5_requirements.txt` |

### Durable run evidence

Remote run root:

```text
/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/agent1-v5-clariden-debug-20260715T111552Z-30c72e9
```

The following durable manifests/receipts were present there when this review was written:

```text
candidate_manifest.json
transform_tasks.json / transform_manifest.json
glossapi_runtime.json / glossapi_manifest.json
envelope_plan.json
base_tasks.json / base_manifest.json
exact_manifest.json
release-pre-dedup/
publication-pre.json
run_contract.json
license_override_receipt.json
datatrove_runtime.json
```

The presence of these files is completion evidence for the represented construction stages. Deduplication is separately in progress.

## 4. Pre-dedup release and current deduplication

### Method

Deduplication uses Hugging Face Datatrove with 128 MinHash permutations, 32 buckets, 4 hashes per bucket, 5-token shingles, and verified Jaccard threshold 0.85. NanoChat base is protected from removal. Exact dedup ran before the current MinHash/near-dedup stage.

The relevant configuration is `configs/agent1_v5_eiger_pipeline.json` under `dedup`.

| Dedup stage | Entry point |
| --- | --- |
| Exact index | `scripts/agent1_v5_datatrove.py exact-index-task` |
| MinHash signature rank | `scripts/agent1_v5_datatrove.py signature-task` |
| Signature merge | `scripts/agent1_v5_datatrove.py merge-signatures` |
| Bucket, shingle, verify, merge-verified, filter | `slurm/agent1_v5_eiger/stage.sh` and `scripts/agent1_v5_datatrove.py` |
| Release publication | `scripts/publish_private_agent1_v5.py` |
| Submission / Slurm orchestration | `scripts/submit_agent1_v5_eiger.py`, `slurm/agent1_v5_eiger/stage.sh` |

The active self-chaining signature script is:

```text
subprojects/05_token_distillation_cpt/04_full_corpus_preparation/scripts/run_signature_task_chain.sh
```

Deployed copy used by the live chain:

```text
/capstor/scratch/cscs/fffoivos/agent1-v5-code/signature-chain/subprojects/05_token_distillation_cpt/04_full_corpus_preparation/scripts/run_signature_task_chain.sh
```

### Current operational evidence

At the status timestamp:

- Signature ranks `0` through `43` had passed receipts: **44 / 431**.
- Each passed rank wrote 32 signature files and a JSON receipt under:

  ```text
  <run>/60-dedup/minhash-signatures/receipts/0000NN.json
  ```

- Active job: `2789351`, `a1v5-signature-chain-r44`, one Clariden node, partition `debug`, account `a0140`, 85-minute walltime.
- The active job had clean errors and was running normally.
- The self-chain submits one validated successor only after a passed receipt. This was intentionally used because `debug-qos` permits **one running job per user** and at most two submitted jobs.

Live status command:

```bash
ssh clariden 'squeue -u fffoivos -o "%.18i %.35j %.8T %.10M %.10l %.24R"'
```

Receipt count command:

```bash
ssh clariden '
run=/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/agent1-v5-clariden-debug-20260715T111552Z-30c72e9
for f in "$run"/60-dedup/minhash-signatures/receipts/*.json; do
  jq -e ".status == \"passed\"" "$f" >/dev/null && echo "$f"
done | wc -l
'
```

Recent completed signature durations were 35:53, 34:33, and 31:42. At that serial rate, the signature stage alone would take approximately nine more days; downstream merge/bucket/shingle/verification/filter/release steps remain after it.

## 5. CSCS access and recovery

The local Clariden project/runbook is:

```text
/Users/foivoskarounos-zamparloukos/Projects/cscs-clariden-project-understanding/README.md
```

The current CSCS certificate is `~/.ssh/cscs-key-cert.pub`. It was refreshed successfully through the official `cscs-key sign --headless --file ~/.ssh/cscs-key` device-code flow and is valid until 2026-07-19T13:24:13 UTC. Do not replace the private key to recover access; renew its signed certificate instead.

## Requested follow-up: measured 4–5× deduplication speed-up plan

Prepare, review, and then implement a plan to reduce remaining dedup wall time by **4–5×** while protecting CSCS shared infrastructure. The plan must not assume that the debug partition can be widened: `debug-qos` is explicitly limited to one running job per user.

The plan must include all of the following:

1. **Partition and resource discovery.** Record the `a0140` normal-partition association, any project/user limits, fair-share implications, job-size guidance, and storage/network policy. Current evidence shows `normal` is up with a 12-hour limit and substantial free capacity; that is not permission to saturate it.
2. **A staged bandwidth/concurrency experiment.** Use only independent, not-yet-started signature ranks. Establish a one-worker baseline from current jobs, then test 2 workers, 4 workers, and a proposed 4–5-worker steady state. Measure per-rank elapsed time, sustained read/write bandwidth, CPU and RSS, metadata-operation rate if available, queue delay, and impact on the existing run.
3. **Safety gates.** Define pre-authorized stop/back-off triggers: material per-worker slowdown, filesystem/client saturation, error-rate increase, queue-policy warnings, or evidence of adverse impact on shared services. Do not exceed the tested fan-out merely because nodes are visible as idle.
4. **Safe orchestration.** Replace the serial self-chain only at a receipt boundary using the held-QoS fence described in the reviewed acceleration plan. The helper normally has no queued successor to cancel. Never cancel a running validated rank. Submit non-overlapping rank ranges or an array with durable per-rank receipts, idempotent reruns, collision-free log paths, and a final manifest-count check.
5. **Target.** Recommend a sustainable 4–5× speed-up, not maximum possible cluster consumption. Show expected elapsed time, node count, CPU allocation, storage/network budget, and how throughput will be monitored continuously.
6. **Validation and rollback.** Require a first-wave receipt audit before continuing. Roll back to the previous serial chain if the gates are exceeded; preserve every completed receipt and signature file.

Do not enact this speed-up plan until the user has reviewed and approved it. The currently running debug chain must remain intact until a tested migration point is approved.

## Quick access map

```bash
# Local project
cd /Users/foivoskarounos-zamparloukos/Projects/train-apertus-agent1-data-prep

# Full-corpus implementation
cd subprojects/05_token_distillation_cpt/04_full_corpus_preparation

# Read the durable run contract remotely
ssh clariden 'jq . /capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/agent1-v5-clariden-debug-20260715T111552Z-30c72e9/run_contract.json | less'

# Current Slurm chain
ssh clariden 'squeue -u fffoivos -o "%.18i %.35j %.8T %.10M %.10l %.24R"'
```
