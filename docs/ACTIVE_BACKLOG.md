# Active Backlog

## Tokenizer Critical Path

1. Freeze the current pathological dedup run before more work is lost:
- stop the live process cleanly
- snapshot `state.sqlite`, `state.sqlite-wal`, `state.sqlite-shm`, and the run root
- preserve the strict exact outputs already produced

2. Repair the dedup exact-stage export path according to:
- [PIPELINE_RECOVERY_AND_SCALE_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_RECOVERY_AND_SCALE_PLAN.md)
- [DEDUP_SCRIPT_REPAIR_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/DEDUP_SCRIPT_REPAIR_PLAN.md)
- golden rule:
  - dedup functionality must remain the same
  - only efficiency, storage strategy, and parallelism may change

3. Resume the current dedup run from saved state instead of restarting from raw corpus input.

4. Treat the failed `16`-worker `near_candidates` run as an explicit plan diversion:
- compare the current implementation against Hugging Face/DataTrove MinHash
- preserve our semantics, but replace the current near-candidate execution shape with a more streaming, merge-based design
- reference:
  - [HF_DEDUP_INVESTIGATION.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/HF_DEDUP_INVESTIGATION.md)
  - [NEAR_DEDUP_REDESIGN_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/NEAR_DEDUP_REDESIGN_PLAN.md)

5. Rebuild the HPLT slice on a GCP worker with the real final filter path:
- quality bins `>=8`
- exclude `Machine translated or generated`
- run real `Corpus.clean(..., write_cleaned_files=False, drop_bad=False)` scoring
- drop rows with `greek_badness_score > 60`

6. Freeze the worker-side downstream builder inputs for:
- `GlossAPI-only`
- `GlossAPI + HPLT` at `70/30` by training-token mass

7. Verify that `openarchives.gr` rows with `needs_ocr == true` are still excluded from the CPT-ready dataset used for tokenizer work.

8. Freeze the held-out eval manifests from the same prepared source-parquet tree.

9. Lock the literal Apertus tokenizer-replication checklist, including the exact tokenizer files and a toy extension proof.

10. Keep the contract-verification suite green as the pipeline changes:
- synthetic tests for schema, markers, and stage-to-stage contracts
- uploader-handoff contract tests
- tiny real-document smoke runs so the contracts are also exercised on real HPLT/GlossAPI records
- exact-stage golden equivalence tests between old and repaired code
- exact-stage resume equivalence tests
- downstream contract equivalence tests after repaired dedup outputs

11. Export BPE-training text for the two training views from the CPT-ready dataset on the chosen GCP worker.

12. Start true Greek `BPE` discovery experiments from the frozen worker-side manifests.

13. Diff learned Greek units against Apertus `model.vocab` and `model.merges`.

14. Run the analytic cutoff sweep on merged variants at:
- `10240`
- `15360`
- `20480`
- `25600`

15. Only after the elbow is known, choose the shipped `128`-aligned extension size.

16. Implement and test the merge-rule extension.

## Dataset Operational Sidetrack

1. Replace the stopped score-only HPLT upload attempt with a rebuilt GCP-side slice that includes the real `Corpus.clean` gate.

2. Rerun the full prepared-source dataset with HPLT included, using the existing dataset scripts rather than inventing a new release path.

3. Keep tokenizer work moving off the prepared dataset without waiting for the HF upload to finish.

4. Refresh published `dedup_metadata` only after the intended dataset state is settled; do not block tokenizer work on that refresh.

5. Provision a separate cheap uploader instance for HF publication work.

6. Use the repo-owned uploader handoff under `ops/upload/` to stage the complete filtered HPLT source parquet slice on that uploader instance, without applying physical dataset deduplication to the published source parquets.

7. Stage the refreshed `dedup_metadata` bundle on that uploader instance so downstream builder-time dedup works after HF download.

8. Publish from that uploader instance with the official HF large-folder upload path through [publish_hf_release.py](/home/foivos/data/glossapi_work/publish_hf_release.py), keeping this upload track fully independent of the tokenizer worker.

9. Verify the uploader instance is configured for the best officially recommended HF large-dataset path, including Xet-backed uploads when available, before the next publication run.

10. Keep the downstream builder/tokenizer efficiency work explicit:
- use [BUILDER_TOKENIZER_EFFICIENCY_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/BUILDER_TOKENIZER_EFFICIENCY_PLAN.md)
- preserve builder semantics while reducing unnecessary bundle loads
- benchmark tokenizer throughput on worker hardware before freezing runtime defaults

## Immediate Risks

- the exploratory HPLT review sample is not the same thing as the final upload-ready HPLT slice
- the exact tokenizer-replication spec is still missing some literal details and a proof-of-mechanism test
- the current HF uploader strategy is poor for observability and recovery on very large patches
- the old baseline workflow still exists for reference, so active work must stay within the new canonical files
- the current dedup exact-stage implementation is not scaling to the full working release
- SQLite is acting as both control plane and heavy data plane during exact-stage finalization
