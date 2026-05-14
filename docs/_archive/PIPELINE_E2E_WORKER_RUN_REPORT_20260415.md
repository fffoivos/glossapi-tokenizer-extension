# Pipeline E2E Worker Run Report 2026-04-15

## Standard

This report uses the stricter verification rule from [PIPELINE_E2E_VERIFICATION_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_VERIFICATION_PLAN.md):

- real worker environment
- real repo-owned wrapper or command
- real stage input artifact from the previous step
- real output artifact for the next step

Component tests are not counted here unless the real worker-side orchestration path was exercised.

## Result

The worker-side downstream chain is now verified end to end through:

1. dedup completion
2. dedup overlay publish
3. tokenizer mix build
4. tokenizer training
5. uploader handoff prep
6. uploader-ready local stage

This proof comes from two complementary runs:

- a live large-chain rearm on the active worker, which verified the real post-dedup production handoff path up to the long-running mix stage
- a bounded real-doc worker smoke run, which exercised the same repo-owned downstream wrappers through tokenizer training completion and uploader-ready local staging

## Run A: Live Large-Chain Rearm

Worker roots:

- working release root:
  - `/home/foivos/data/glossapi_work/hf_release_publish_working`
- dedup state root:
  - `/home/foivos/data/glossapi_work/analysis/dedup/text_publish/state/gcp_refresh_20260413`
- mix root:
  - `/home/foivos/data/glossapi_work/tokenizer_mixes_20260413`
- training root:
  - `/home/foivos/data/glossapi_work/tokenizer_training_runs_20260413`
- handoff root:
  - `/home/foivos/data/glossapi_work/uploader_handoff_20260414`
- rearm logs:
  - `/home/foivos/data/glossapi_work/logs/e2e_verify_20260415_v1`

Verified transitions:

1. Dedup completion already existed:
  - `/home/foivos/data/glossapi_work/analysis/dedup/text_publish/state/gcp_refresh_20260413/latest_success.json`
2. Overlay publish completed from the repo-owned wrapper:
  - wrapper: [wait_for_dedup_and_publish_overlay.sh](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/scripts/wait_for_dedup_and_publish_overlay.sh)
  - completion artifact:
    - `/home/foivos/data/glossapi_work/hf_release_publish_working/dedup_metadata/latest.json`
  - publish summary:
    - `/home/foivos/data/glossapi_work/hf_release_publish_working/dedup_metadata/exact_stage_20260413T025237Z/publish_summary.json`
3. Uploader handoff prep completed from the repo-owned wrapper:
  - wrapper: [wait_for_dedup_overlay_and_prepare_handoff.sh](/home/foivos/Projects/glossapi-tokenizer-extension/ops/upload/wait_for_dedup_overlay_and_prepare_handoff.sh)
  - completion artifacts:
    - `/home/foivos/data/glossapi_work/uploader_handoff_20260414/uploader_handoff.json`
    - `/home/foivos/data/glossapi_work/uploader_handoff_20260414/handoff_summary.json`
4. Uploader-ready local stage completed from the repo-owned launcher path:
  - wrapper: [wait_for_uploader_handoff_and_launch.sh](/home/foivos/Projects/glossapi-tokenizer-extension/ops/upload/wait_for_uploader_handoff_and_launch.sh)
  - launcher: [launch_hf_uploader_handoff.py](/home/foivos/Projects/glossapi-tokenizer-extension/ops/upload/launch_hf_uploader_handoff.py)
  - completion summary:
    - `/home/foivos/data/glossapi_work/uploader_handoff_20260414/launch_summary.json`
  - staged root:
    - `/home/foivos/data/glossapi_work/local_uploader_stage_20260415/hf_release_publish`
5. Live tokenizer mix build started from the published overlay:
  - wrapper: [wait_for_dedup_overlay_and_build_tokenizer_mixes.sh](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_2_training_dataset_mix/scripts/wait_for_dedup_overlay_and_build_tokenizer_mixes.sh)
  - active process observed:
    - `566412`
  - first durable progress artifact:
    - `/home/foivos/data/glossapi_work/tokenizer_mixes_20260413/glossapi_only/glossapi_mix_t9v_tbid/.filtered_input.parquet.tmp`

Current live-chain status at verification time:

- overlay branch: verified
- handoff branch: verified
- uploader-ready local stage: verified
- mix branch: actively progressing
- tokenizer training: waiting correctly on the two `mix.parquet` outputs

Important note:

- the rearmed training watcher was intentionally set to `TOKENIZER_TRAINING_LAUNCH_MODE=inline` for direct verification, so once both mixes exist the two tokenizer runs will execute serially in that specific live rearm
- the canonical wrapper default is still `systemd`

## Run B: Bounded Real-Doc Worker Smoke

This run closed the true E2E gap without waiting for the full live production mix to finish.

Worker smoke root:

- `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z`

Smoke log:

- `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z.log`

The smoke used:

- source release root:
  - `/home/foivos/data/glossapi_work/hf_release_publish_working`
- raw HPLT root:
  - `/home/foivos/apertus_greek_tokenizer_runs/20260408T160000Z/data/hplt_ell_grek`
- corpus python:
  - `/home/foivos/venvs/glossapi-corpus-clean/bin/python`
- tokenizer python:
  - `/home/foivos/venvs/tokenizer-training/bin/python`

Completed stages from the log:

1. `stage=sample_release_rows`
2. `stage=sample_real_hplt_rows`
3. `stage=build_hplt_slice`
4. `stage=integrate_hplt_slice`
5. `stage=dedup`
6. `stage=publish_overlay`
7. `stage=prepare_upload_handoff`
8. `stage=local_stage_upload_handoff`
9. `stage=build_mixes`
10. `stage=train_tokenizers`

Smoke completion summary:

- `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z/smoke_summary.json`

Verified smoke artifacts:

- dedup success:
  - `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z/dedup_state/latest_success.json`
- overlay publish:
  - `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z/tiny_working_release/dedup_metadata/latest.json`
- uploader handoff:
  - `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z/upload_handoff/handoff_summary.json`
- uploader-ready local stage:
  - `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z/upload_handoff/launch_summary.json`
- mix outputs:
  - `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z/mixes/glossapi_only/mix.parquet`
  - `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z/mixes/glossapi_plus_hplt_70_30/mix.parquet`
- mix summaries:
  - `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z/mixes/glossapi_only/mix.summary.csv`
  - `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z/mixes/glossapi_plus_hplt_70_30/mix.summary.csv`
- tokenizer training completion:
  - `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z/training/glossapi_only_1k/training_summary.json`
  - `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z/training/glossapi_plus_hplt_70_30_1k/training_summary.json`

Key smoke outcomes:

- sampled release datasets:
  - `openarchives.gr`: `2`
  - nine other real source datasets plus one old-HPLT row
- sampled HPLT rows:
  - `40`
- mixed-HPLT character ratio:
  - `0.2940`
- glossapi-only tokenizer actual vocab:
  - `1252`
- mixed tokenizer actual vocab:
  - `1252`

## Conclusion

What is now genuinely proven:

- the repo-owned downstream worker path after dedup is no longer blocked by the dead overlay script path
- the overlay, handoff, uploader-ready local stage, mix build, and tokenizer training wrappers work in sequence on the real worker
- a bounded real-doc worker run completed through tokenizer training and uploader-ready handoff

What is still operationally in flight:

- the live large-chain downstream continuation is still inside the first long `mix` build under `/home/foivos/data/glossapi_work/tokenizer_mixes_20260413`
- tokenizer training has not launched yet in that live large-chain run because both full-size mixes are not finished

So the repo-owned worker chain is now truly end-to-end verified, but the full-size live run is still bottlenecked on mix build throughput and visibility.
