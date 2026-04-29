# Continuous BPE Extension TODO

## 1. Freeze Target Numbers

- [x] 1.1 Confirm Apertus base tokenizer size
- [x] 1.2 Confirm maximum planned merged fresh cutoff
- [x] 1.3 Freeze continuous-BPE total target size at `156672`
- [x] 1.4 Freeze common cutoff grid at `10240`, `15360`, `20480`, `25600`

## 2. Fresh Tokenizer Publication

- [ ] 2.1 Verify both completed fresh tokenizer output directories are complete and loadable
- [ ] 2.2 Create the fresh HF repo for the tokenizer extension outputs
- [ ] 2.3 Upload `fresh/glossapi_only_50k`
- [ ] 2.4 Upload `fresh/glossapi_plus_hplt_70_30_50k`
- [ ] 2.5 Publish per-arm summaries and README notes
- [ ] 2.6 Verify the HF repo contents after upload

## 3. Apertus Replication Freeze

- [x] 3.1 Extract the exact Apertus tokenizer files used as the base
- [x] 3.2 Record the exact normalizer configuration
- [x] 3.3 Record the exact pre-tokenizer configuration
- [x] 3.4 Record the exact decoder configuration
- [x] 3.5 Record special tokens and IDs
- [ ] 3.6 Add a tokenizer-replication check script
- [ ] 3.7 Prove the replication check passes before any continuation training
- [x] 3.8 Prove the stock HF/tokenizers training APIs are retraining paths, not continuation paths
- [x] 3.9 Freeze the base snapshot under `/home/foivos/data/glossapi_work/tokenizer_base_snapshots/apertus_8b_2509_20260415`

## 4. Continuous-BPE Training Implementation

- [ ] 4.1 Add a dedicated custom continuous-BPE training script
- [ ] 4.2 Add the append-only continuation engine on top of the frozen Apertus vocab and merges
- [ ] 4.3 Add a run-summary schema for continuous-BPE outputs
- [ ] 4.4 Add resumable output layout for continuous-BPE runs
- [ ] 4.5 Add a small real-data smoke test for the continuation path
- [ ] 4.6 Add a contract test proving the continued tokenizer preserves Apertus front-end behavior
- [ ] 4.7 Add a pre-run identity check proving the continuation start state is behaviorally identical to Apertus

## 5. Continuous-BPE Execution

- [ ] 5.1 Launch `C1` on `GlossAPI-only` to total size `156672`
- [ ] 5.2 Wait for `C1` to finish cleanly
- [ ] 5.3 Verify `C1` artifacts and summary
- [ ] 5.4 Launch `C2` on `GlossAPI + HPLT 70/30` to total size `156672`
- [ ] 5.5 Wait for `C2` to finish cleanly
- [ ] 5.6 Verify `C2` artifacts and summary
- [ ] 5.7 If a run fails or stalls, resume from the latest completed checkpoint instead of restarting blindly

## 6. Merged Variant Grid

- [ ] 6.1 Build fresh merged variants for `F1` at `10240`, `15360`, `20480`, `25600`
- [ ] 6.2 Build fresh merged variants for `F2` at `10240`, `15360`, `20480`, `25600`
- [ ] 6.3 Build continuous merged variants for `C1` at `10240`, `15360`, `20480`, `25600`
- [ ] 6.4 Build continuous merged variants for `C2` at `10240`, `15360`, `20480`, `25600`
- [ ] 6.5 Verify all `16` merged variants are loadable

## 7. Apples-To-Apples Evaluation

- [ ] 7.1 Run the intrinsic metric bundle on all merged variants
- [ ] 7.2 Run fertility tests on all merged variants
- [ ] 7.3 Run the common evaluation slices:
  - [ ] 7.3.1 `GlossAPI` held-out
  - [ ] 7.3.2 `HPLT` held-out
  - [ ] 7.3.3 mixed `GlossAPI + HPLT` held-out
  - [ ] 7.3.4 `modern_greek_eval`
- [ ] 7.4 Write a single comparison table across all `16` merged variants
- [ ] 7.5 Identify the best one or two candidates for downstream use

## 8. Operational Completion

- [ ] 8.1 Verify dedup replacement upload is complete
- [ ] 8.2 Create and verify the fresh HF tokenizer-extension repo
- [ ] 8.3 Upload the two fresh tokenizer arms
- [ ] 8.4 Upload the two continuous tokenizer arms
- [ ] 8.5 Verify the published repo contents are loadable
- [ ] 8.6 Verify no required tokenizer process is still running
- [ ] 8.7 Stop the GCP worker instance

## 9. Final Acceptance

- [ ] 9.1 Four raw tokenizer arms exist: `F1`, `F2`, `C1`, `C2`
- [ ] 9.2 Sixteen merged cutoff variants exist
- [ ] 9.3 HF tokenizer-extension repo is published
- [ ] 9.4 Comparison report is written
- [ ] 9.5 Worker shutdown is confirmed
