# Latest 5B Report Status

Generated from `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/reports/latest_5b_report_state.json`.

Canonical task lockdown: `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/goal/canonical_eval_tasks.json`.

## Canonical Eval Tasks

Render is filtered against the canonical lockdown file. Non-canonical task data stays on disk; the renderer never surfaces it.

| Field | Value |
| --- | --- |
| canonical_tasks_source | `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/goal/canonical_eval_tasks.json` |
| schema | canonical-eval-tasks-v1 |
| generated_utc | 2026-05-30T16:00:41Z |
| greek_native_mcq.headline_tasks | greekmmlu, ilsp_medical_mcqa, ilsp_mcqa_asep |
| greek_native_mcq.diagnostic_separate | plutus_qa |
| greek_native_mcq.excluded_mt_derived | xnli_el, xcopa_el, arc_challenge_mt_el, global_piqa_completions_ell_grek |
| retention.languages | en, fr, de, ru |
| retention.tasks_present | en::mmlu, en::arc_challenge, en::arc_easy, en::hellaswag, en::piqa, en::global_mmlu_en, en::xnli_en, fr::global_mmlu_fr, fr::xnli_fr, de::global_mmlu_de, de::xnli_de, ru::xnli_ru |
| retention.tasks_absent_on_disk | en::xstorycloze_en, ru::xstorycloze_ru |
| heldout_bpb | greek_heldout_500, code_heldout_200, math_heldout_200 |

## Latest Training Snapshot

| Field | Value |
| --- | --- |
| Collected at UTC | 2026-05-30T16:03:12Z |
| Run tag | 04_vanilla_goldfish_5b_20260528T112539Z |
| Current iter | 300 |
| Target iter | 300 |
| Consumed tokens | 1.258B |
| LM loss | 1.50459 |
| Skipped iterations | 0 |
| NaN iterations | 0 |
| Tokens/sec/GPU | 8000.5 |
| Iter ms | 131064.1 |
| Next incomplete checkpoint | none |
| Next checkpoint iter | none |
| Remaining iters to next checkpoint | none |
| Next checkpoint tokens | none |
| Next checkpoint ETA | none |
| ETA iter 119 | 2026-05-30T16:03:12Z |
| ETA iter 238 | 2026-05-30T16:03:12Z |
| ETA iter 300 | 2026-05-30T16:03:12Z |
| ETA iter 477 | 2026-05-30T22:29:51Z |
| ETA iter 834 | 2026-05-31T11:29:41Z |
| ETA iter 1192 | 2026-06-01T00:31:42Z |

Latest line:

```text
3:  [2026-05-29 03:52:45] iteration      300/     300 | consumed samples:       307200 | consumed tokens: 1.258B | elapsed time per iteration (ms): 131064.1 | eta: 0:00:00 | tokens/sec/gpu: 8000.5 | throughput per GPU (TFLOP/s/GPU): 412.3 | learning rate: 1.100000E-05 | global batch size:  1024 | lm loss: 1.504590E+00 | loss scale: 1.0 | grad norm: 0.623 | params norm: 7092.506 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

## Artifact Counts And Health

| Check | Value |
| --- | --- |
| Checkpoint iter dirs | 18 |
| Sidecar files | 14 |
| Eval items | 1207 |
| Watcher stderr exists | yes |
| Watcher stderr bytes | 0 |
| Nonzero skipped/NaN lines | 0 |
| Severe log terms | 0 |

## Queue

| Job ID | Name | Partition | State | Elapsed | Limit | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| none |  |  |  |  |  |  |

## Sidecar Watcher Coverage

| Watcher job | Started UTC | Max watch seconds | Resubmit if incomplete | Code/math heldouts required | Env mtime |
| --- | --- | ---: | --- | --- | --- |
| 2417289 | 2026-05-28T14:41:42Z | 82800 | 1 | 1 | 2026-05-28T14:41:42Z |
| 2417305 | 2026-05-28T14:45:34Z | 82800 | 1 | 1 | 2026-05-28T14:45:34Z |
| 2417454 | 2026-05-28T14:55:32Z | 82800 | 1 | 1 | 2026-05-28T14:55:32Z |
| 2424477 | 2026-05-29T13:55:57Z | 82800 | 1 | 1 | 2026-05-29T13:55:57Z |

## Checkpoints

| Label | Iter | Tokens | Megatron | Metadata | HF | Checksum manifest | Eval root | Sidecar manifest | Archived sidecar attempts | Handoff verified | Review critique |
| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | --- | --- |
| Vanilla-0.5B | 119 | 499122176 | yes | yes | yes | yes | yes | yes | 1 | yes | yes |
| Vanilla-1B | 238 | 998244352 | yes | yes | yes | yes | yes | yes | 0 | yes | yes |
| Vanilla-2B | 477 | 2000683008 | yes | yes | yes | yes | yes | yes | 0 | yes | yes |
| Vanilla-3.5B | 834 | 3498049536 | yes | yes | yes | yes | yes | yes | 0 | yes | yes |
| Vanilla-5B | 1192 | 4999610368 | yes | yes | yes | yes | yes | yes | 0 | yes | yes |

## Handoff Verification

| Label | Iter | Latest check | Checkpoint metadata | Sidecars submitted | Manifest complete | Outputs ready | Active sidecar jobs | Slurm done | Checksum ready | Handoff ready | Pass snapshot |
| --- | ---: | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| Vanilla-0.5B | 119 | 2026-05-28T21:28:38Z | yes | yes | yes | yes | 0 | yes | yes | yes | yes |
| Vanilla-1B | 238 | 2026-05-29T00:47:26Z | yes | yes | yes | yes | 0 | yes | yes | yes | yes |
| Vanilla-2B | 477 | 2026-05-29T11:14:32Z | yes | yes | yes | yes | 0 | yes | yes | yes | yes |
| Vanilla-3.5B | 834 | 2026-05-29T23:42:43Z | yes | yes | yes | yes | 0 | yes | yes | yes | yes |
| Vanilla-5B | 1192 | 2026-05-30T15:33:56Z | yes | yes | yes | yes | 0 | yes | yes | yes | yes |

## Checkpoint Metrics

| Label | Iter | MCQ headline | MCQ + Plutus | Greek BPB | Greek trunc | Code BPB | Math BPB | Retention en (mmlu/arc_challenge/arc_easy/hellaswag/piqa/global_mmlu_en/xnli_en) | Retention fr (global_mmlu_fr/xnli_fr) | Retention de (global_mmlu_de/xnli_de) | Retention ru (xnli_ru) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| Vanilla-0.5B | 119 | 0.439098 | 0.429323 | 0.604936 | 0.292 | 0.417785 | 0.727075 | 0.567/0.509/0.811/0.571/0.786/0.605/0.519 | 0.535/0.455 | 0.525/0.486 | 0.482 |
| Vanilla-1B | 238 | 0.448670 | 0.439836 | 0.468373 | 0.292 | 0.306476 | 0.602232 | 0.551/0.529/0.823/0.578/0.795/0.600/0.512 | 0.525/0.494 | 0.540/0.509 | 0.480 |
| Vanilla-2B | 477 | 0.479152 | 0.481586 | 0.431274 | 0.292 | 0.280681 | 0.558946 | 0.596/0.525/0.803/0.592/0.791/0.637/0.490 | 0.580/0.468 | 0.588/0.485 | 0.488 |
| Vanilla-3.5B | 834 | 0.478966 | 0.481447 | 0.419664 | 0.292 | 0.269746 | 0.549082 | 0.589/0.519/0.811/0.591/0.782/0.615/0.546 | 0.560/0.504 | 0.603/0.502 | 0.491 |
| Vanilla-5B | 1192 | 0.497331 | 0.481887 | 0.413192 | 0.292 | 0.264617 | 0.544837 | 0.580/0.537/0.819/0.591/0.789/0.650/0.549 | 0.588/0.505 | 0.595/0.495 | 0.473 |

## Report Artifacts

| Artifact | Exists | Updated UTC | Path |
| --- | --- | --- | --- |
| config_geometry_audit_iter_0000119_json | yes | 2026-05-28T21:01:59Z | `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/reports/config_geometry_audit_iter_0000119.json` |
| config_geometry_audit_iter_0000119_md | yes | 2026-05-28T21:01:59Z | `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/reports/config_geometry_audit_iter_0000119.md` |
| goal_doc | yes | 2026-05-28T21:11:39Z | `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/goal/goal.md` |
| goal_hyperparameters_json | yes | 2026-05-29T12:20:37Z | `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/goal/hyperparameters.json` |
| report_draft_5b | yes | 2026-05-28T21:25:25Z | `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/reports/5B_REPORT_DRAFT.md` |
| run_log_20260528 | yes | 2026-05-30T15:50:57Z | `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/RUN_LOG_20260528.md` |
| submit_checkpoint_sidecars | yes | 2026-05-29T11:21:44Z | `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/scripts/submit_checkpoint_sidecars.sh` |
| watch_and_run_adversarial_reviews | yes | 2026-05-28T21:24:58Z | `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/scripts/watch_and_run_adversarial_reviews.sh` |
| watch_and_submit_checkpoint_sidecars | yes | 2026-05-28T12:02:09Z | `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/scripts/watch_and_submit_checkpoint_sidecars.sbatch` |
| watch_checkpoint_sidecar_verification | yes | 2026-05-28T21:24:14Z | `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/scripts/watch_checkpoint_sidecar_verification.sh` |

## Key Paths

| Name | Path |
| --- | --- |
| train_run_dir | `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z` |
| eval_root | `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04_vanilla_goldfish_5b_20260528T112539Z` |
| watch_dir | `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z_sidecar_watch` |
| submit_state_dir | `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z_submit_state` |
| active_log | `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04van5b_i300-2417446.out` |
