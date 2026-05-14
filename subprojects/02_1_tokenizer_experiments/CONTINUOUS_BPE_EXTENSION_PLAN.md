# Continuous BPE Extension Plan

> **Archived framing — superseded by C3 convergence.** This document
> describes the original four-arm comparison plan
> (`F1`/`F2`/`C1`/`C2`). The arm decision has been made in favor of
> **C3** (`C3_wave2_broad_glossapi_plus_hplt_50_50`); see
> [../../docs/C3_CONVERGENCE.md](../../docs/C3_CONVERGENCE.md).
>
> The sections below remain accurate for the parts that survived:
> the cutoff grid (§1.2 / §1.4), the merged-variant + intrinsic +
> fertility evaluation pattern (§8), and the publication / acceptance
> shape (§9 / §10). The four-arm-specific sections (§2–§7) describe
> work that is now done and should not be re-run.
>
> If you are reading this for the live cutoff decision, jump to §8
> and read "C3" wherever the doc says "the four arms".

## 1. Freeze The Comparison Target

### 1.1 Apertus Base Size

- Base tokenizer: `swiss-ai/Apertus-8B-2509`
- Confirmed current tokenizer length: `131072`
- `131072` is already divisible by `128`

### 1.2 Extension Ceiling For Apples-to-Apples Comparison

- The planned merged fresh-tokenizer cutoff grid already ends at:
  - `10240`
  - `15360`
  - `20480`
  - `25600`
- The maximum planned fresh merged extension is therefore `25600`
- `25600` is divisible by `128`

### 1.3 Continuous-BPE Full Training Target

- Continuous-BPE full target extension: `+25600`
- Continuous-BPE full target tokenizer size: `131072 + 25600 = 156672`
- `156672` is divisible by `128`

### 1.4 Decision

- The continuous-BPE runs must train to `156672` total tokenizer size
- The fresh-vs-continuous comparison must use the same merged cutoff grid:
  - `10240`
  - `15360`
  - `20480`
  - `25600`

## 2. Define The Four Raw Tokenizer Arms

### 2.1 Fresh Discovery Arms

- Arm `F1`: fresh discovery BPE on `GlossAPI-only`
- Arm `F2`: fresh discovery BPE on `GlossAPI + HPLT 70/30`

### 2.2 Continuous-BPE Arms

- Arm `C1`: continuous BPE from Apertus on `GlossAPI-only`
- Arm `C2`: continuous BPE from Apertus on `GlossAPI + HPLT 70/30`

### 2.3 Current State

- `F1` exists
- `F2` exists
- `C1` does not exist yet
- `C2` does not exist yet

## 3. Upload The Finished Fresh Tokenizers

### 3.1 Publish Target

- Publish the completed fresh tokenizers to a fresh HF repo with the working name:
  - `apertus-tokenizer-extension`

### 3.2 Packaging Rule

- The repo must clearly separate the tokenizer variants
- Minimum publish contents:
  - `fresh/glossapi_only_50k`
  - `fresh/glossapi_plus_hplt_70_30_50k`
- Include:
  - tokenizer files
  - per-run `training_summary.json`
  - short README describing the arm names and corpus inputs

### 3.3 Upload Gate

- Do not publish partial or ambiguous contents
- Publish only after verifying the tokenizer directories are complete and loadable

## 4. Lock Exact Apertus Replication For Continuous BPE

### 4.1 Freeze The Literal Apertus Front-End

- Freeze the exact Apertus base artifacts under:
  - `/home/foivos/data/glossapi_work/tokenizer_base_snapshots/apertus_8b_2509_20260415`
- Freeze the exact:
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - backend runtime JSON
  - normalizer
  - pre-tokenizer
  - decoder
  - model type
  - special tokens
  - special token IDs
  - added tokens behavior

### 4.2 Define The Continuation Contract

- Continuous BPE must start from the existing Apertus tokenizer state
- It must preserve the Apertus front-end exactly
- It must grow the BPE model to `156672` total tokens
- It must not silently switch to a fresh tokenizer-training path

### 4.3 Freeze What Was Proven About The Stock APIs

- `train_new_from_iterator(...)` is not continuation
- It explicitly clears the BPE vocab and merges and retrains from scratch
- Direct low-level `Tokenizer.train_from_iterator(...)` is also not usable as continuation for this case
- Therefore the continuous-BPE path must use a dedicated custom continuation implementation, not the stock retraining helpers

### 4.4 Implementation Deliverables

- Add a dedicated continuous-BPE training script
- Add a replication-check script that proves:
  - base Apertus tokenization before continuation matches exactly
  - the continued tokenizer still preserves the intended front-end
- Add a run summary schema for continuous-BPE outputs

## 5. Develop The Continuous-BPE Training Path

### 5.1 Custom Continuation Engine

- Load the frozen Apertus BPE vocab and merges as the base state
- Learn additional merge candidates from the target corpus without resetting the existing state
- Append new units until the total tokenizer size reaches `156672`
- Preserve the Apertus front-end and special-token contract exactly

### 5.2 Resumability And Output Layout

- Make continuation runs resumable from stable checkpoints
- Write a clear artifact layout for:
  - base snapshot reference
  - learned extension units
  - final tokenizer files
  - run summary
  - manifest

### 5.3 Verification Before Long Runs

- Add a small real-data smoke run for the continuation path
- Prove that the tokenizer behavior before the first new merge is identical to Apertus
- Prove that the produced tokenizer is loadable and keeps the intended front-end

## 6. Train The Two Continuous-BPE Arms

### 6.1 Continuous `GlossAPI-only`

- Train Apertus-continuation tokenizer on `GlossAPI-only`
- Output full target tokenizer at `156672`
- Save run summary and artifact manifest

### 6.2 Continuous `GlossAPI + HPLT 70/30`

- Train Apertus-continuation tokenizer on `GlossAPI + HPLT 70/30`
- Output full target tokenizer at `156672`
- Save run summary and artifact manifest

### 6.3 Operational Constraints

- Reuse the already-built mix artifacts where possible
- Keep runs resumable
- Ensure long-running jobs are detached, verified, and monitored

### 6.4 Wait And Verification Rules

- Do not call a continuous run complete just because it launched
- Wait for each run until:
  - the training process exits cleanly
  - the final tokenizer files exist
  - the run summary exists
  - the final tokenizer length is `156672`
- If a run stalls or exits early, resume it from the latest completed checkpoint instead of restarting blindly

## 7. Build The Comparable Merged Variant Grid

### 7.1 Fresh Merged Variants

- For `F1`, derive merged Apertus-compatible variants at:
  - `10240`
  - `15360`
  - `20480`
  - `25600`
- For `F2`, derive merged Apertus-compatible variants at:
  - `10240`
  - `15360`
  - `20480`
  - `25600`

### 7.2 Continuous Variants

- For `C1`, derive comparable Apertus-compatible variants at:
  - `10240`
  - `15360`
  - `20480`
  - `25600`
- For `C2`, derive comparable Apertus-compatible variants at:
  - `10240`
  - `15360`
  - `20480`
  - `25600`

### 7.3 Comparison Set

- Raw tokenizer arms to compare: `4`
- Merged cutoff variants to compare: `16`

## 8. Evaluate Apples To Apples

### 8.1 Intrinsic Metrics

- Run the same metric bundle across the merged variants:
  - `bytes_per_token`
  - `tokens_per_byte`
  - fertility
  - added-token utilization
  - vocabulary utilization
  - unreachable added tokens
  - byte-fallback rate

### 8.2 Evaluation Slices

- `GlossAPI` held-out
- `HPLT` held-out
- mixed `GlossAPI + HPLT` held-out
- `modern_greek_eval`

### 8.3 Decision Questions

- Does continuous BPE beat fresh discovery on the same corpus view?
- Does `GlossAPI + HPLT 70/30` beat `GlossAPI-only` enough to justify its extra complexity?
- Which cutoff is best after the merged Apertus-compatible comparison, not just on raw tokenizer outputs?

## 9. Publish And Operational Orchestration

### 9.1 Fresh-Tokenizer Upload And Verification

- Wait for fresh tokenizer outputs to be present and verified
- Upload them to the fresh HF tokenizer-extension repo
- Verify the repo contents after upload

### 9.2 Continuous-Tokenizer Upload And Verification

- After `C1` and `C2` finish, publish them to the same HF repo
- Keep the repo structure explicit, at minimum:
  - `fresh/...`
  - `continuous/...`
- Verify the uploaded continuous tokenizers are loadable from the published repo

### 9.3 Evaluation And Selection

- Build the merged cutoff grid
- Run intrinsic and fertility evaluation
- produce a comparison table for all `16` merged variants

### 9.4 Infrastructure Shutdown

- After upload and tokenizer work are complete:
  - verify no needed pipeline process is still running
  - stop the GCP worker instance

## 10. Required Proof Before Calling This Complete

### 10.1 Fresh Upload Proof

- HF repo exists
- both fresh tokenizer arms are present
- tokenizer files are loadable

### 10.2 Continuous-BPE Proof

- `C1` summary exists
- `C2` summary exists
- both reached total size `156672`

### 10.3 Comparison Proof

- the four raw arms are present
- the `16` merged cutoff variants are produced
- the intrinsic and fertility comparison bundle is written

### 10.4 Shutdown Proof

- no required upload/training process remains
- GCP worker is stopped
