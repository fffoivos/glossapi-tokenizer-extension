# Active Backlog

## Tokenizer Critical Path

1. Complete the live downstream continuation from the dedup-complete worker state:
- keep the current full-size mix build healthy
- verify tokenizer training actually launches after the two mix outputs exist
- decide whether the live training launcher should stay inline or return to the canonical dual-service path
- publish a final downstream run report once the full-size live chain clears tokenizer launch

2. Run the explicit pipeline verification pass before trusting the downstream chain again:
- execute the real repo-owned worker path end to end from dedup completion through tokenizer launch
- audit each remaining stage for hidden serial bottlenecks
- audit each remaining stage for trustworthy progress reporting
- reference:
  - [PIPELINE_E2E_VERIFICATION_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_VERIFICATION_PLAN.md)
  - [PIPELINE_E2E_VERIFICATION_TODO.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_VERIFICATION_TODO.md)
  - [PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md)
  - [PIPELINE_STAGE_PARALLELISM_REVIEW_20260415.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_STAGE_PARALLELISM_REVIEW_20260415.md)
  - [PIPELINE_STAGE_PROGRESS_REVIEW_20260415.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_STAGE_PROGRESS_REVIEW_20260415.md)

3. Continue the near-dedup redesign as an explicit plan diversion:
- compare the current implementation against Hugging Face/DataTrove MinHash
- preserve our semantics, but replace the current near-candidate execution shape with a more streaming, merge-based design
- reference:
  - [HF_DEDUP_INVESTIGATION.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/HF_DEDUP_INVESTIGATION.md)
  - [NEAR_DEDUP_REDESIGN_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/NEAR_DEDUP_REDESIGN_PLAN.md)

4. Rebuild the HPLT slice on a GCP worker with the real final filter path:
- quality bins `>=8`
- exclude `Machine translated or generated`
- run real `Corpus.clean(..., write_cleaned_files=False, drop_bad=False)` scoring
- drop rows with `greek_badness_score > 60`

5. Freeze the worker-side downstream builder inputs for:
- `GlossAPI-only`
- `GlossAPI + HPLT` at `70/30` by training-token mass

6. Verify that `openarchives.gr` rows with `needs_ocr == true` are still excluded from the CPT-ready dataset used for tokenizer work.

7. Freeze the held-out eval manifests from the same prepared source-parquet tree.

8. Lock the literal Apertus tokenizer-replication checklist, including the exact tokenizer files and a toy extension proof.

9. Keep the contract-verification suite green as the pipeline changes:
- synthetic tests for schema, markers, and stage-to-stage contracts
- uploader-handoff contract tests
- tiny real-document smoke runs so the contracts are also exercised on real HPLT/GlossAPI records
- downstream contract equivalence tests after repaired dedup outputs
- near-stage resumability and chunk-progress tests for the redesigned path

10. Run the builder/tokenizer efficiency plan on worker hardware:
- builder duplicate-subset replay benchmark
- tokenizer throughput sweep for `RAYON_NUM_THREADS` and batch size
- freeze runtime defaults after the sweep

11. Export BPE-training text for the two corpus views from the CPT-ready dataset on the chosen GCP worker:
- `GlossAPI-only`
- `GlossAPI + HPLT` at `70/30`

12. Run the full four-arm tokenizer experiment matrix from the frozen worker-side manifests:
- fresh discovery `BPE` on `GlossAPI-only`
- fresh discovery `BPE` on `GlossAPI + HPLT`
- continuous `BPE` from Apertus on `GlossAPI-only`
- continuous `BPE` from Apertus on `GlossAPI + HPLT`

13. Compare all four tokenizer arms on the same evaluation bundle:
- fertility/compression metrics
- tokenization behavior on the primary Greek eval set
- practical extension quality relative to Apertus compatibility

14. Diff the best one or two learned Greek-unit sets against Apertus `model.vocab` and `model.merges`.

15. Run the analytic cutoff sweep on merged variants at:
- `10240`
- `15360`
- `20480`
- `25600`

16. Only after the elbow is known, choose the shipped `128`-aligned extension size.

17. Implement and test the merge-rule extension.

## Dataset Operational Sidetrack

1. Replace the stopped score-only HPLT upload attempt with a rebuilt GCP-side slice that includes the real `Corpus.clean` gate.

2. Rerun the full prepared-source dataset with HPLT included, using the existing dataset scripts rather than inventing a new release path.

3. Keep tokenizer work moving off the prepared dataset without waiting for the HF upload to finish.

4. Refresh published `dedup_metadata` only after the intended dataset state is settled; do not block tokenizer work on that refresh.

5. Provision a separate cheap uploader instance for HF publication work.

6. Use the repo-owned uploader handoff under `ops/upload/` to stage the complete filtered HPLT source parquet slice on that uploader instance, without applying physical dataset deduplication to the published source parquets.

7. Stage the refreshed `dedup_metadata` bundle on that uploader instance so downstream builder-time dedup works after HF download.

8. Publish from that uploader instance with the official HF large-folder upload path through [publish_hf_release.py](/home/foivos/Projects/glossapi-tokenizer-extension/publish_hf_release.py), keeping this upload track fully independent of the tokenizer worker.

9. Verify the uploader instance is configured for the best officially recommended HF large-dataset path, including Xet-backed uploads when available, before the next publication run.

10. Keep the downstream builder/tokenizer efficiency work explicit:
- use [BUILDER_TOKENIZER_EFFICIENCY_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/BUILDER_TOKENIZER_EFFICIENCY_PLAN.md)
- preserve builder semantics while reducing unnecessary bundle loads
- benchmark tokenizer throughput on worker hardware before freezing runtime defaults

11. Treat the current worker-side source-only upload as an explicit temporary fallback:
- the intended permanent target is still the separate cheap uploader host
- while that host is unreachable, keep the worker upload low-priority and isolated from dedup
- once the cheap uploader host is reachable again, move the publication path back there

## Immediate Risks

- the exploratory HPLT review sample is not the same thing as the final upload-ready HPLT slice
- the exact tokenizer-replication spec is still missing some literal details and a proof-of-mechanism test
- the current HF uploader strategy is poor for observability and recovery on very large patches
- the old baseline workflow still exists for reference, so active work must stay within the new canonical files
- the live full-size downstream chain is still blocked on a long serial mix stage before tokenizer launch
- tokenizer training still lacks a trustworthy mid-run progress signal
- the cheap uploader host is still unreachable, so the worker-side source-only upload fallback must remain temporary
