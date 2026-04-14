# Active Backlog

## Tokenizer Critical Path

1. Complete the live repo-backed near-dedup continuation from the preserved state:
- keep the current prefix-chunk `near_candidates` run healthy
- record memory profile, completed-chunk rate, and time-to-band readiness
- continue into `near_clusters`, final exports, dedup overlay, and builder metadata publication

2. Continue the near-dedup redesign as an explicit plan diversion:
- compare the current implementation against Hugging Face/DataTrove MinHash
- preserve our semantics, but replace the current near-candidate execution shape with a more streaming, merge-based design
- reference:
  - [HF_DEDUP_INVESTIGATION.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/HF_DEDUP_INVESTIGATION.md)
  - [NEAR_DEDUP_REDESIGN_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/NEAR_DEDUP_REDESIGN_PLAN.md)

3. Rebuild the HPLT slice on a GCP worker with the real final filter path:
- quality bins `>=8`
- exclude `Machine translated or generated`
- run real `Corpus.clean(..., write_cleaned_files=False, drop_bad=False)` scoring
- drop rows with `greek_badness_score > 60`

4. Freeze the worker-side downstream builder inputs for:
- `GlossAPI-only`
- `GlossAPI + HPLT` at `70/30` by training-token mass

5. Verify that `openarchives.gr` rows with `needs_ocr == true` are still excluded from the CPT-ready dataset used for tokenizer work.

6. Freeze the held-out eval manifests from the same prepared source-parquet tree.

7. Lock the literal Apertus tokenizer-replication checklist, including the exact tokenizer files and a toy extension proof.

8. Keep the contract-verification suite green as the pipeline changes:
- synthetic tests for schema, markers, and stage-to-stage contracts
- uploader-handoff contract tests
- tiny real-document smoke runs so the contracts are also exercised on real HPLT/GlossAPI records
- downstream contract equivalence tests after repaired dedup outputs
- near-stage resumability and chunk-progress tests for the redesigned path

9. Run the builder/tokenizer efficiency plan on worker hardware:
- builder duplicate-subset replay benchmark
- tokenizer throughput sweep for `RAYON_NUM_THREADS` and batch size
- freeze runtime defaults after the sweep

10. Export BPE-training text for the two training views from the CPT-ready dataset on the chosen GCP worker.

11. Start true Greek `BPE` discovery experiments from the frozen worker-side manifests.

12. Diff learned Greek units against Apertus `model.vocab` and `model.merges`.

13. Run the analytic cutoff sweep on merged variants at:
- `10240`
- `15360`
- `20480`
- `25600`

14. Only after the elbow is known, choose the shipped `128`-aligned extension size.

15. Implement and test the merge-rule extension.

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
- the live near-candidate path still needs more throughput validation on real corpus scale
- the cheap uploader host is still unreachable, so the worker-side source-only upload fallback must remain temporary
