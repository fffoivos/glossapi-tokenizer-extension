# Targeted Apertus 8B CPT experiments

## Current active track (2026-08-14)

The active experiment is the matched **8B + 1.5B HPLT-to-OpenArchives** study
with goals A/B/C in
[`HARD_H_TO_G_8B_1P5B_REPLICATION_PLAN_20260814.md`](HARD_H_TO_G_8B_1P5B_REPLICATION_PLAN_20260814.md).
The machine authority is
[`configs/hard_h_to_g_replication_v1.json`](configs/hard_h_to_g_replication_v1.json),
and the R2 review plus disposition record are
[`REVIEW_ULTRACODE_R2_HARD_H_TO_G_PLAN_20260814.md`](REVIEW_ULTRACODE_R2_HARD_H_TO_G_PLAN_20260814.md)
and
[`ULTRACODE_R2_REMEDIATION_20260814.md`](ULTRACODE_R2_REMEDIATION_20260814.md).

Implementation is authorized. Production is not. The launch contract retains
`production_launch_authorized=false` until every pre-main receipt and explicit
owner authorization passes. Operational preparation belongs on one-node
`debug` or the allocation-free login/control path. Load paths, model/config
geometry, tokenizer/vocabulary shape, cache indexes, runtime imports, world-size
arithmetic and the exact expanded segment command must be checked before
`sbatch`; they are never a reason to request `normal`. Only actual scientific
work (1.5B Token Distillation, 1.5B LR pilots and approved production training)
belongs on `normal`.

Current executable entry points are:

- `clariden/deploy_targeted_bundle.sh` — freeze the immutable scientific code
  bundle;
- `clariden/build_data_runtime_debug.sbatch` and
  `scripts/verify_data_runtime.py` — build a new immutable AArch64 data runtime
  on `debug` and verify its imports, exact versions, critical file hashes,
  requirements lock and scientific-bundle binding;
- `clariden/build_td_xfer_runtime.sbatch` and
  `clariden/build_td_snippets_xfer.sbatch` — build a separately frozen x86_64
  runtime and execute the exact-order 2B-token TD coverage scan on the CPU-only
  `xfer` partition.  A 2026-08-15 debug measurement reached 50,002,836 tokens
  in 7m48s (about 5.2 hours projected), proving that this one operational scan
  cannot fit the 90-minute debug limit; it consumes no `normal` GPU capacity;
  the scan audits NFC form but intentionally preserves the exact historical
  Stage-B Unicode bytes. The selected replication pipeline did not contain an
  NFC transform, so non-NFC rows are counted and receipted rather than silently
  normalized or used to change the training corpus;
- `clariden/validate_and_inspect_debug.sbatch` — run the full PyArrow-aware
  unit/static gate on one debug node using only that verified runtime;
- `clariden/freeze_modern_mix_recipes_debug.sbatch` and
  `clariden/build_modern_mix_selection_debug.sbatch` — bind the exact
  benchmark-clean source-view files and reproduce the historical 8.5B HPLT /
  3.7B OpenArchives 16-shard selection geometry before E001 and the fresh
  GreekMMLU scan; failed payloads are retained outside the stable authority
  paths;
- `clariden/split_replay_stage_b_debug.sbatch` and
  `clariden/tokenize_h2g_stream_debug.sbatch` — reproduce the historical
  foreign/Old-Greek replay split and create four separately receipted
  historical-148,480 Megatron indexed streams on one `debug` node, with exact
  input-row/document reconciliation and transactional failure quarantine;
- `clariden/anonymize_training_stream_debug.sbatch`,
  `clariden/filter_replay_greekmmlu_debug.sbatch`,
  `clariden/audit_replay_stage_b_greekmmlu_debug.sbatch`, and
  `clariden/materialize_replay_postfilter_scan_debug.sbatch` — apply Stage B
  only after the selected replay has passed both native-suite and GreekMMLU
  filtering, then bind both post-filter audits to the exact Stage-B bytes that
  will be split and tokenized;
- `clariden/build_phase_data_path_spec_debug.sbatch`,
  `clariden/build_phase_gptdataset_cache_debug.sbatch`, and
  `clariden/freeze_phase_blend_cache_debug.sbatch` — build and freeze the exact
  ordered three-prefix data/cache contract;
- `clariden/train_hard_h_to_g_segment.sbatch` — fail-closed segmented trainer;
  it requires the promoted-profile/LR run permit, exact source/target phase
  caches, either an initialization or checkpoint permit, and a byte-bound
  preallocation static-preflight receipt for the exact same segment; and
- `clariden/audit_training_checkpoint_debug.sbatch` — freeze and structurally
  audit a completed DCP checkpoint before a successor permit can be issued;
- `clariden/run_prelaunch_benchmark.sbatch` plus the debug finalizers — retained
  for genuine profile research only. They are not used to catch load/config
  mistakes. The already-promoted 8B profile is reused; 1.5B dynamic evidence is
  collected by its actual LR-pilot work rather than a separate proof-only
  allocation;
- `clariden/prove_uenv10_srun_debug.sbatch` — prove on one debug node that the
  CSCS uenv-v10 Slurm plugin mounts PyTorch and `torchrun` inside remote srun
  tasks before any multi-node profile or production allocation is requested;
- `clariden/freeze_production_timing_and_allocation_debug.sbatch` — run exact
  `sbatch --test-only` checks without changing the artifact manifest, then
  derive measured conservative segment times and the bounded one-successor
  schedule; and
- `clariden/freeze_pre_main_artifact_manifest_debug.sbatch`,
  `clariden/freeze_owner_authorization_debug.sbatch`, and
  `clariden/freeze_pre_main_launch_gate_debug.sbatch` — freeze the pre-timing,
  pre-authorization and final 26-role manifests in order, record owner
  authorization only after the other 25 roles pass, and produce the immutable
  launch-ready pre-main gate; every manifest accepts only explicitly audited
  producer bundles; and
- Phase-3 cursor and constant-floor continuation are checked from checkpoint
  metadata, frozen cache identities and deterministic scheduler/data-loader
  simulation before submission. No one-update `normal` proof job is required.

Every production segment, including the first, must pass the authorization
gate appropriate to its stage inside `preflight_train_segment.py`. Phase 1/2
requires `pre_main`; Phase 3 production requires `pre_extension` and then
`pre_second_extension`. GreekMMLU sentinel calibration is evaluation-only: it
runs concurrently from the saved checkpoints and gates `pre_finalization`, not
Phase-3 optimizer entry. The one-update Phase-3 proof jobs consume the previous
stage's gate so they can produce, rather than circularly require, the next
gate. A direct `sbatch` therefore cannot bypass the staged artifact manifest or
owner authorization.

The legacy public GreekMMLU compatibility result is executed from the frozen
16,632-question snapshot through
`scripts/run_legacy_greekmmlu_snapshot_eval.py`. A separate loader-parity
receipt proves that the compatibility wrapper replaces dataset loading only;
the historical scorer and prompt arithmetic remain at commit `cfdd0e7b`.

The older targeted A/B material below is retained as historical machinery and
is not the current launch authority.

This subproject prepares one active targeted 8B experiment without changing the
approved Apertus 8B model, tokenizer, optimizer, loss, masking, or DP32
execution geometry. A previously prepared continuation experiment is retained
only as immutable historical machinery.

- **A — academic/polytonic mixture:** one pass over `openarchives.gr` and
  `greek_phd`, the same number of HPLT active tokens, one pass over the existing
  release-internal polytonic source datasets, and stationary 79/20/1
  modern/foreign/Old-Greek replay.
- **B — retired:** the update-9,536 unseen-non-HPLT continuation was dropped by
  the owner on 2026-08-12. Pending jobs `3061757` and `3061758` were cancelled
  before allocation or training (`00:00:00`). Its builder and receipts remain
  available, but the recipe is explicitly not launch-authorized.

The authoritative design and allocation contract is
[`CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md`](CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md).

The proposed replacement research program—independent replication of the
13.5B-token 8B HPLT-to-GlossAPI result followed by a 0.5B/1.5B proxy-validity
study—is specified in
[`SCALE_PREDICTIVITY_STUDY_20260812.md`](SCALE_PREDICTIVITY_STUDY_20260812.md).
It is design-only and not launch-authorized.

The focused execution plan has three goals: reproduce the historical-horizon
8B result, test whether a matched 1.5B model mirrors the 8B trajectory, and
continue the OpenArchives phase for two additional approximately 1B-token
checkpoints at both scales. It freezes update 3,218 as the replication endpoint
before continuing to updates 3,456 and 3,694. The benchmark-clean corpus is
built by directly joining the published HF document-overlap table while
freshly regenerating GreekMMLU queries and scanning every rebuilt stream.
Nested frozen GreekMMLU sentinels provide the dense trajectory only after
same-stack calibration in both the early and late four-checkpoint windows; the
plan falls back to the complete panel if neither subset resolves the relevant
adjacent-checkpoint changes. Phase 3 is a separate unseen-document blend/cache
consumed from cursor zero, not an appended Phase-2 index. Complete endpoint and
mechanical plateau confirmation remains mandatory:
[`HARD_H_TO_G_8B_1P5B_REPLICATION_PLAN_20260814.md`](HARD_H_TO_G_8B_1P5B_REPLICATION_PLAN_20260814.md).
R2 findings and their code/doc dispositions are tracked in
[`ULTRACODE_R2_REMEDIATION_20260814.md`](ULTRACODE_R2_REMEDIATION_20260814.md).
Implementation is authorized, but production remains fail-closed and not
launch-authorized.

The exact continuation data builder is retained for future, explicitly
versioned mix experiments. Its implementation hashes, proven B invocation,
fixed-policy boundary and extension rules are frozen in
[`CONTINUATION_DATA_BUILDER_HANDOFF_20260812.md`](CONTINUATION_DATA_BUILDER_HANDOFF_20260812.md)
and `configs/continuation_data_builder_v1.json`.

The implementation is deliberately fail-closed:

1. A cannot freeze until every configured polytonic source is audited and
   extracted from the pinned anonymized Hugging Face release, then counted with
   the production tokenizer. No external dataset, separate historical Parquet,
   reconstruction, or second global deduplication is permitted.
2. A cannot pack until the selected modern data has passed the pinned
   GreekMMLU decontamination scan. The final anonymized Hugging Face release is
   used as-is; no second global deduplication is permitted.
3. B cannot launch: its owner-retirement flag is checked by the production
   submitter before Slurm test-only or submission. Its old artifacts are
   preservation evidence only.
4. No production-horizon GPU submission is allowed until the debug preparation
   receipts, initial validation, GreekMMLU conversion smoke, and one bounded
   16-node DP32 two-update/restart parity allocation all pass. The distributed
   parity smoke executes three optimizer updates in one allocation: two in an
   uninterrupted control trajectory, then one resumed from the control's exact
   intermediate checkpoint. It is not a substitute for a launch gate.
5. Operational jobs use `debug`; production uses only the proven
   16-node/64-GPU `dp32_16node` profile.

The active A data path is release-internal and fail-closed:
`clariden/audit_release_polytonic_sources_debug.sbatch` verifies actual
polytonic Unicode evidence in every configured source, then
`clariden/extract_release_polytonic_sources_debug.sbatch` copies exactly those
release rows without transformation or deduplication. The old standalone
`poly_train` recovery scripts are retained only as superseded history; they are
not an input or a launch gate. `home` is not an execution, storage, or recovery
dependency for this subproject.

Useful entry points:

```bash
python3 scripts/freeze_experiment_contract.py \
  --experiment-a-config configs/experiment_a_recipe.json \
  --experiment-b-config configs/experiment_b_recipe.json \
  --allocation-config configs/allocation_plan.json \
  --output /path/to/contracts_receipt.json

python3 scripts/build_continuation_b_schedule.py --help
python3 scripts/validate_static_contracts.py
```

`clariden/inspect_hf_release_debug.sbatch` is the first remote job. It only
reads schemas and receipts and therefore belongs on the one-node `debug`
partition.

Large binary-encoding and packing task sets use
`clariden/run_parallel_task_batch_debug.sbatch`. It runs a frozen positional
range concurrently on one debug node and emits one receipt for the whole
range; it does not change task identity, document order, tokenization, packing,
or the selected schedule.

The sole non-production `normal` allocation is launched through
`clariden/submit_targeted_restart_smoke.sh` after all debug-built data,
schedule, recipe and execution-profile assets are frozen. The submitter always
runs `sbatch --test-only`, hard-pins one leaf with an audited exclusion, and
refuses to proceed before those assets exist. `--switches=1` alone is not
treated as a hard placement guarantee because Slurm may relax it after its
wait threshold. All other preparation and evaluation-control scripts declare
`#SBATCH --partition=debug`.

For experiment B, `clariden/prepare_continuation_b_assets_debug.sbatch` runs
the pool-view, training-asset and experiment-contract freezes sequentially in
one debug allocation after the exact continuation schedule exists. Each step
still writes and validates its own immutable receipt; this only removes two
extra scheduler waits.

`clariden/run_all_per_document_groups_debug.sbatch` evaluates all 13 frozen
validation panels at one checkpoint in four sequential groups inside one
four-GPU debug allocation. It calls the already proven resource-aware group
runner unchanged, verifies both immutable bundles before scoring, and publishes
the complete output tree transactionally only after all 13 receipts pass. This
removes three scheduler waits without changing panel identity or evaluation
arithmetic.

The aggregate in-training source panel is evaluated every 25 optimizer updates
by the inherited launcher. `freeze_training_assets.py` writes the same
25-update / 104,857,600-token cadence into each targeted recipe, and the launch
gate rejects any receipt/executable mismatch.

GreekMMLU receipt metadata is derived together with its checkpoint list: A
records approximately every 2B active tokens (477 updates), while B records the
parent anchor and approximately every 1B continuation tokens (238 updates),
plus the final checkpoint. Inherited 5B cadence labels are overwritten.

Because the launch gate binds the nested-submission proof to the executing
scientific bundle, each new targeted bundle uses
`clariden/prove_nested_sbatch_debug.sbatch` once. Both its parent and submitted
child run on `debug`, and the child verifies the immutable bundle plus the
rank-local uenv/torchrun/Megatron path. This replaces the inherited proof's
stale code-root binding without consuming a `normal` allocation.
