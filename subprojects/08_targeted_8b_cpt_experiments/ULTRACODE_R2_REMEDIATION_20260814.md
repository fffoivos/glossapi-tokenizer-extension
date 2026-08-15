# Ultracode R2 remediation record

Date: 2026-08-14
Status: R2 design/code remediation and the subsequent executable audit are
locally complete; the prior v8-v12 remote evidence is retained only as
superseded diagnostic evidence. A replacement immutable bundle, bundle-bound
debug runtime and complete remote test/evidence cycle remain pending;
production remains unauthorized.

Review authority:
`REVIEW_ULTRACODE_R2_HARD_H_TO_G_PLAN_20260814.md`, against plan SHA-256
`a47c152bb47c5646196a4e42525f8cd7b731dad83318356d3a055ba733497fe5`.

Current local authorities after remediation:

- plan SHA-256: `1aa1560bbaae69ae8ab0594151c5ef4e9fc2f25ae2a62c1d43b618eadf19c4c2`;
- machine contract SHA-256:
  `38008be0830c658b8cbc66d34ed1f94270fb5dbcf8fecdafbd828ed93e4e77e6`;
- launch freezer SHA-256:
  `4c2d982c78e74d41aa72ea27875db8e26cf73650c21eb3ad458b294767bfc6d0`;
- statistics freezer SHA-256:
  `441d4fa150b2c15921ebd67cebb5ed34902fc9f3593ed4bf91b260fc7fb33b59`.

These hashes are local review evidence, not an immutable CSCS code-bundle
receipt. A new bundle must be frozen after all local checks pass.

Superseded immutable CSCS authority (retained as failure evidence):

- bundle root:
  `/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260814T154000Z-hard-h2g-r2-v8`;
- bundle tree SHA-256:
  `2a62bd6d0a396aa8440b977f61d07533fa7dc4061024c267839a4ee136069fcc`;
- immutable data runtime:
  `/iopsstor/scratch/cscs/fffoivos/python_envs/targeted_h2g_py312_20260814_v8`.

The subsequent v9 validation freeze failed closed in job `3082594` before it
wrote a receipt. The inherited neutral external Greek panel uses the explicit
row schema `text` / `source_doc_id` / `cluster_id` / `source_id`, while the
other twelve panels use `text` / `doc_id` / `source_dataset`. The freezer had
incorrectly assumed the latter schema for every panel. It now has an explicit
panel adapter, records the selected row schema in the receipt, and forbids a
fallback from a missing neutral `source_doc_id` to an unrelated `doc_id`.
Because the scientific code changed, v9 cannot be relabelled: a new immutable
bundle, runtime, full test gate, and all bundle-bound data-authority receipts
must be produced.

Bundle v10 subsequently proved the panel adapter end to end: job `3082694`
passed 85/85 targeted and 4/4 source-local packing tests, and job `3082837`
froze all 13 panels (59,749 rows; 59,742 unique exact-text hashes) with the
neutral panel's explicit 345-document cluster schema recorded in its receipt.
That audit also found that fifteen remaining R2 data wrappers still named the
known-incomplete legacy PyArrow environment as a default or used an ambient
uenv Python. Although the v10 invocations explicitly overrode those defaults,
leaving them in code would allow a later manual run to regress. They now fail
closed unless `H2G_DATA_PYTHON` is supplied, and all PyArrow/source-transform
entrypoints execute that exact interpreter. This source change requires a
final new immutable bundle/runtime/test/evidence cycle; v10 remains positive
test evidence but is not the launch authority.

## Finding disposition

| R2 finding | Disposition | Enforced by |
| --- | --- | --- |
| Phase-3 append/unseen/randomized contradiction | confirmed; fixed by a separate Phase-3-only blend/cache consumed from cursor zero | plan Sections 5–7; `build_phase3_unseen_catalog.py`; pre-extension gate |
| Deleted GreekMMLU queries and incomplete old Stage-A exclusions | confirmed; regenerate pinned queries and freshly scan every rebuilt stream | query contract/builder; `run_fresh_greekmmlu_stream_scan.py`; debug wrapper |
| Longer horizon reanchors WSD and optimizer ramps | confirmed; freeze `train_samples=3,782,656`, constant nominal floor, zero LR warmup and 3,218-update alpha/beta3 denominators | machine contract; scheduler resume pre-extension receipt |
| Phase-2 is a repeated shuffled stream, not an unseen prefix | confirmed; realized Phase-2 document ids are measured, not inferred | plan Section 6; realized-ledger and Phase-3 anti-join gates |
| Phase-local cursor code and Phase-2 restart gap | confirmed; generalized to every Phase-2/3 entry and restart | `phase_local_data_index_guard.py`; profile-matched restart receipts |
| v2 HPLT population/Greek-replay overlap ambiguity | adjusted; exact `HPLT/ell_Grek_ge8_no_mt_clean60` label, explicit population difference and natural-key anti-join | machine contract; source-label/count receipt |
| External replay scanner unnamed/gameable | adjusted; named heterogeneous adapter and mandatory content scan, no disjointness escape | `build_replay_benchmark_scan_input.py`; debug wrapper; pre-main receipt |
| Early-only sentinel cannot resolve plateau | confirmed; independent early and late full-panel calibration, both required | evaluation/statistics contracts |
| Monolithic launch checklist deadlocks | confirmed; split into pre-main, pre-extension, pre-second-extension and pre-finalization gates | launch freezer and Section 13 |
| Spearman threshold likely unreachable | confirmed; calibrated point threshold 0.45, lower bound above zero, fail only when upper bound below zero | statistics contract/freezer |
| Forgetting margin undefined | confirmed; `2 * median(document-bootstrap SE)` over 0/238/476/714, minimum recomputed per replicate | statistics contract/freezer |
| Goal-C margins/panels undefined | confirmed; exact calibration updates, per-panel formula, panel list and directions frozen | statistics contract/freezer |
| Goal-B aggregation undefined | confirmed; exact primary/secondary, plateau and pass/fail/inconclusive aggregation | statistics contract/freezer |
| Batch-16 clean scorer unsupported | confirmed; clean FP32 scorer fixed at batch 1; batch 16 only for the separate legacy BF16 run | machine and legacy evaluator contracts |
| Extension save walls off 119 grid | confirmed; force exact save-and-exit at 3,456 and 3,694 | schedule contract and plan |
| Published normalized-text hash nonexistent/ordered after E001 | confirmed; verify published raw UTF-8 `document_text_sha256` on pristine v2 rows before E001 | native-exclusion builder and machine contract |
| No-rescan/Protipa contradiction | confirmed; Protipa frozen excluded; native-v2 rows use direct join while fresh GreekMMLU and external replay scans are explicitly required | data contract and plan |
| Goal-A margin rationale arithmetic | confirmed; rewritten as a predeclared operational tolerance, not an exact CI derivation | plan Section 9 |
| Sentinel determinism/nesting gaps | confirmed; stable subject floor, salted stratum tie-break, incremental remaining-capacity Hamilton | `build_greekmmlu_sentinels.py`; tests |
| Goal-C direction/saturation ambiguity | confirmed; explicit BPB directions; inconclusive second interval cannot establish saturation | statistics contract/freezer |
| Bootstrap scope unstated | confirmed; panel-sampling uncertainty conditional on one run per scale, not training-seed variance | statistics contract/freezer |
| Legacy evaluator cannot forward dataset revision | confirmed; pre-materialized pinned snapshot plus loader-only parity wrapper required | legacy evaluator contract/freezer |
| `physical_order` chronology wrong | confirmed; wording now states it predates the run but conflicts with receipted runtime mode | plan Section 2 |
| WSD floor not exact at update 3,218 | confirmed; nominal floor occurs near update 3,218.65 and extension behavior is deliberate | plan/training contract |
| Inventory incorrectly implied query survival | confirmed; inventory distinguishes deleted and surviving evidence | plan and asset inventory contract |
| Legacy BF16 under fallback tokenizer unclear | confirmed; skip or mark non-comparable; Goal A not testable | plan and legacy contract |
| Plateau trigger judgment-laden | confirmed; earliest conclusion-changing plateau member is a mechanical full-panel trigger | evaluation contract and plan |
| Sweep audit over-cited for new recipe fields | confirmed; field-specific authority map added; sweep audit limited to arm selection | plan Section 2 |
| Full-8B D0 wording conflates two runs | confirmed; 8B production and 0.5B D0 predictions are named separately and non-binding | plan Section 8 |

## New executable surfaces

- `scripts/phase_local_data_index_guard.py`
- `scripts/build_phase3_unseen_catalog.py`
- `scripts/build_replay_benchmark_scan_input.py`
- `scripts/run_fresh_greekmmlu_stream_scan.py`
- `clariden/build_phase3_unseen_catalog_debug.sbatch`
- `clariden/build_replay_scan_input_debug.sbatch`
- `clariden/run_fresh_greekmmlu_stream_scan_debug.sbatch`
- `scripts/constant_floor_resume.py`
- `scripts/patch_bakeoff_scale_geometry.py`
- `scripts/finalize_training_megatron.py`
- `scripts/freeze_phase_blend_cache.py`
- `clariden/prepare_training_megatron_debug.sbatch`
- `clariden/freeze_phase_blend_cache_debug.sbatch`
- `scripts/freeze_online_validation_binaries.py`
- `clariden/freeze_online_validation_binaries_debug.sbatch`
- `scripts/verify_data_runtime.py`
- `clariden/build_data_runtime_debug.sbatch`
- `scripts/freeze_modern_mix_recipes.py`
- `scripts/finalize_modern_mix.py`
- `clariden/freeze_modern_mix_recipes_debug.sbatch`
- `clariden/build_modern_mix_selection_debug.sbatch`
- `scripts/freeze_legacy_greekmmlu_loader_parity.py`
- `scripts/run_legacy_greekmmlu_snapshot_eval.py`
- `scripts/finalize_lr_pilot_arm.py`
- `scripts/freeze_production_timing_and_allocation.py`
- `scripts/freeze_submission_dry_run.py`
- `clariden/freeze_production_timing_and_allocation_debug.sbatch`
- `scripts/finalize_phase3_resume_smoke.py`
- `scripts/freeze_post_checkpoint_authorities.py`
- `clariden/run_phase3_resume_smoke.sbatch`
- `clariden/finalize_phase3_resume_smoke_debug.sbatch`
- `clariden/freeze_post_checkpoint_authorities_debug.sbatch`

Every operational builder above is assigned to a one-node `debug` job. No
production or distributed `normal` allocation was submitted during this
remediation.

## GPTDataset cache-builder runtime failures

The first Phase-1 cache build, job `3085644`, failed before receipt publication
because the pinned Megatron source had not compiled its canonical
`helpers_cpp` extension.  The partial cache was retained under `_failed`, the
dependent Phase-2 job was cancelled without running, and the replacement
runtime compiled and import-tested the exact c92402e helper.

The helper-qualified retry, job `3085687`, then advanced through all three
component GPTDataset index builds and exposed a second independent upstream
assumption: `BlendedDataset._build_indices()` calls
`torch.distributed.get_rank()` unconditionally on a cache miss.  The wrapper
had correctly launched one process, but the cache-only Python entry point had
not initialized a process group.  No stable receipt was written; the wrapper
quarantined the partial cache, and dependent job `3085688` performed no work.

The cache builder now creates an ephemeral CPU/Gloo process group with exact
rank 0 and world size 1 only around canonical blend construction, destroys it
before publication, and records the backend, rank, world size and teardown in
the cache-build receipt.  This supplies runtime context required by upstream
Megatron without changing component weights, sample requests, seeds, indices,
tokenization or training semantics.

The first pre-main data-authority freeze, job `3085753`, then rejected the
already-receipted Stage-B no-op streams because its verifier compared complete
file bindings, including their deliberately different prepared and Stage-B
paths.  The HPLT and OpenArchives receipts both prove equal byte counts and
equal SHA-256 digests, zero changed rows, equal row counts and the explicit
`asserted_byte_noop` invariant.  The verifier now compares byte count plus
SHA-256 for no-op identity, independently requires those row/invariant proofs,
and retains path-sensitive comparisons wherever lineage requires the exact
same file.  The failure wrote no authority output; dependent initialization
job `3085754` was cancelled before allocation.

The bundle-v25 retry, job `3085801`, then exposed the same schema-boundary
class at the next replay edge: the native-filter output binding included its
producer-side `rows` field while the GreekMMLU-filter input binding recorded
only path, bytes and SHA-256.  Both receipts bind the exact same
`replay_native_clean.jsonl` path, 24,933,360,322 bytes and SHA-256
`2ad5c14ea0822628c0734f3235e8f6ecc8c463b321223bb604d4f4f846aa9ccc`.
The verifier now removes only producer count metadata before the path-sensitive
comparison.  It additionally requires the replay Stage-B input to match the
GreekMMLU-clean output's exact path, byte count and SHA-256, in addition to the
existing receipt-file binding.  Job `3085801` published no authority; dependent
8B initialization job `3085809` was cancelled before allocation.

The v26 retry, job `3085861`, verified the full replay chain and then exposed
that the reusable phase-cache validator still required the cache receipt's
producer to equal the currently executing bundle.  That contradicted the
separate, fail-closed producer-compatibility authority used for immutable data
adoption.  The cache validator continues to require exact-current production
by default.  Its pre-main adoption call now supplies only the bundle identities
already admitted by the fully verified compatibility receipt and additionally
runs the full receipt-bound `require_accepted_producer` check on each phase
cache.  Job `3085861` published no pre-main authority.

The first regenerated-TD coverage attempt, job `3085916`, used the pinned
historical single-document encoder and exercised only one of its 64 allocated
CPUs.  After 7 minutes 38 seconds it had not reached the first 50M-token marker,
projecting well beyond the 90-minute `debug` limit; it and its unstarted
dependent TD job `3085919` were cancelled.  The temporary directory was removed
and no stable TD-input receipt was published.  The replacement keeps document
order, token-budget truncation, firing counts and seeded reservoir updates
sequential, but batches the independent Rust tokenizer calls so its established
internal CPU parallelism is used.  It compares batched IDs and offsets against
individual encoding for the first 256 documents and has a byte-for-byte fixture
against the pinned sequential prepass for coverage and snippet outputs.  The TD
input freezer now requires and records that parity authority.

## Post-review implementation audit

The R2 plan review was followed by a command-level audit of the executable
wrappers, not only the prose. It found and corrected three additional issues
before their affected stages launched:

1. The reconstructed replay recipe still pointed at deleted historical
   placeholder Parquets. It now derives its exact file set from the completed
   2026-07-31 acquisition receipt, full-hashes the 355 selected files, and
   states explicitly that historical replay document identity is not claimed.
2. The first reconstruction wrapper had collapsed the historical 16-way
   eligible-row-modulo replay selection into one mix-builder call. It now
   reproduces the 5B aggregate target as 16 independent 312.5M-token shards,
   seed `20260611`, concatenated in shard-index order. Failed output is retained
   under an exact `_failed_replay_mix_<job>` path and cannot poison the stable
   authority path.
3. The Greek source wrapper would have processed every eligible v2 HPLT and
   OpenArchives row rather than reproducing the historical 8.5B/3.7B selected
   capacities. The corrected pipeline first freezes exact-file recipes over
   the benchmark-clean source views and then applies the historical 16-shard
   scheduler: 531.25M HPLT tokens per shard and 231.25M OpenArchives tokens per
   shard, seed `20260611`, ascending concatenation. The mix builder has a
   lineage-only output extension for exact source files, release coordinates
   and metadata; a unit fixture proves the modulo-selected row identities are
   unchanged. E001 and the fresh GreekMMLU scan now consume these selected
   streams rather than the complete v2 pools.

The same audit also corrected the 1.5B Token Distillation batch from 16 to the
historically receipted batch 8, froze seed `20260523` and BF16 execution, bound
the 2B-token coverage prepass, and made the TD output transactional so a failed
training process cannot leave an apparently reusable stable initialization.

These changes postdate bundle v12. Therefore v12 remains intermediate positive
evidence only: the launch authority must be a new immutable bundle, a new
bundle-bound runtime, the complete debug test gate, and regenerated or
cryptographically adopted receipts. No replay or modern-stream selection may
run from v12.

A second executable audit then found and corrected five further fail-closed
gaps before the replacement bundle was frozen:

1. Megatron resume had been handed `iter_000xxxx` instead of the checkpoint
   parent containing `latest_checkpointed_iteration.txt`; permits now bind both
   the exact iteration directory and the parent load root/tracker.
2. The prelaunch profile test proved only a Phase-1 resume. It now also binds
   the Phase-2 cache, enters it at cursor zero, resumes its exact update-2
   checkpoint at cursor 1,024, and requires next-update parity. The test-only
   phase-start override is explicitly cleared by the production trainer.
3. The 1.5B reference-profile finalizer used the nonexistent identifier
   `1p5b_1node_dp4`; it now requires the frozen ID `1p5b_tp1_1node`.
4. The historical BF16 GreekMMLU compatibility path previously asserted a
   loader-only reconstruction without proving it. Query freezing now compares
   all 16,632 pinned raw-row hashes, reconstructed fields and prompts through
   both evaluator versions, and the final contract binds the loader-only
   adapter and parity receipt.
5. `production_timing` and `allocation_schedule` were required artifact roles
   without a producer. Their producer now requires production-cadence wall
   measurements, promoted profiles, exact run permits and mutation-free Slurm
   test-only evidence before deriving bounded holder triggers.

## Local evidence

- Python compile: passed for all subproject scripts and evaluation utilities.
- New PyArrow-independent R2 orchestration suite: 5/5 passed locally.
- The full suite cannot run under the Mac's broken/missing PyArrow installs;
  it remains a mandatory gate in the fresh bundle-bound debug runtime rather
  than being waived or replaced by the local subset.
- New Slurm wrapper syntax: passed `bash -n`.
- The constant-floor unit gate proves restored WSD scheduler and parameter-group
  state are both rewritten to the nominal floor and remain idempotent.
- The training runtime is now a read-only c92402e clone with only the
  SHA-256-pinned named-validation patch; the unrelated exact-evaluation patch
  is explicitly forbidden. Segment preflight revalidates the live git diff,
  patched-file hashes, phase cache and bundle bindings before GPU work.
- A post-review command-level audit found and corrected a machine-contract
  transcription error: the historical launcher enables
  `--reset-attention-mask`, `--reset-position-ids`, and `--eod-mask-loss`.
  The contract now states no cross-EOD attention, reset positions, and masked
  EOD targets, matching the executable environment exactly.
- The inherited trainer's permissive missing-validation warning is fenced by a
  preflight receipt that full-hashes all 18 files for the exact nine historical
  panels. Training cannot start if any named panel would be silently skipped.
- The executed phase-cursor path now uses the strict cache-bound guard rather
  than the older unbound wrapper. Source-phase and target-phase cache receipts
  are separate at the 2,261 and 3,218 boundaries, so a checkpoint is bound to
  the cache that produced it while the resumed segment is independently bound
  to the cache it will consume.
- The trainer now passes an explicit receipt-bound `--data-cache-path` instead
  of relying on Megatron's implicit cache location beside read-only data.
- Phase data is represented by the actual ordered weighted three-prefix
  `--data-path`, not a fictitious single binary. Phase 3 decouples its 487,424
  local GPTDataset samples from the 3,782,656 global scheduler horizon and
  rejects any document-index epoch wrap.
- Training preflight now requires a per-scale run permit that binds the exact
  promoted profile and selected peak/floor LR. Successor permits come only
  from a structural DCP checkpoint audit; the audit checks model, optimizer,
  RNG, scheduler/cursor, training-log stability, storage ranges and read-only
  checkpoint files.
- Planning contract: correctly reports `blocked`, stage `pre_main`; the current
  machine contract requires all 26 pre-main artifact roles, including the
  not-yet-granted production authorization.
- Statistical decision contract: freezes successfully as schema v2.
- Remote jobs `3082097` and `3082119` failed before dataset work because the
  inherited runtime advertised PyArrow 21.0.0 in package metadata while its
  installed `pyarrow` package lacked all Python files, including
  `__init__.py` and `compute.py`. The validation path no longer accepts the
  old existence-only runtime receipt. A fresh immutable runtime must be built
  on `debug`, must import every critical module, and must pass
  `verify_data_runtime.py` against its exact package versions, required-file
  hashes, requirements lock and executing scientific bundle.
- Runtime job `3082335` completed in 50 seconds. Its verifier passed every
  package, import, critical-file, requirements-lock and bundle-binding check;
  the runtime root is read-only.
- Full debug validation job `3082346` completed in 71 seconds: 84/84 targeted
  tests and 4/4 source-local packing tests passed. Static contracts passed;
  release inspection passed without writing data rows; the hard H-to-G
  statistical decision contract froze as schema v2. The pre-main launch
  contract remains intentionally blocked on missing downstream receipts and
  owner production authorization.
- Bundle v9 passed the same 84/84 plus 4/4 debug gate in job `3082443` and
  regenerated the query/native/replay authorities. Validation job `3082594`
  then exposed the neutral-panel identity mismatch above. No validation
  exclusion payload or receipt was accepted from that failed job.
- Bundle v13 validation job `3083811` failed before data production and exposed
  three executable test/preflight defects: two incomplete unit fixtures and
  incomplete enforcement of the bundle-bound data runtime. Bundle v14 job
  `3083829` passed 114/114 targeted tests plus 4/4 source-local packing tests
  after those defects were corrected.
- The first rebuilt replay inventory then exposed one legitimate heterogeneous
  schema case: the exact SHA-bound FineMath Parquet inputs have no source
  identity column. The inventory now permits a synthetic identity only for
  `math_finemath`, defined as immutable input-file SHA-256 plus zero-based row
  index; every other source still fails if its configured identity is absent.
  Bundle v15 job `3083862` passed 115/115 targeted tests plus 4/4 packing tests
  with that narrow fallback.
- A command-level audit before modern-mix execution found that
  `freeze_modern_mix_recipes_debug.sbatch` and
  `build_modern_mix_selection_debug.sbatch` accepted `H2G_DATA_PYTHON` without
  verifying its immutable runtime-root binding. Neither wrapper had yet run.
  Both now require `H2G_DATA_RUNTIME_ROOT` and source the shared verifier only
  after authenticating the scientific bundle. Bundle v17 is frozen at
  `/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260814T205000Z-hard-h2g-r2-v17`
  with tree SHA-256
  `e4862154ca7adfd50a6c2d2fc4fa9c3fc579384329718efff65642107209d7b4`;
  its fresh runtime and complete debug gate remain mandatory before those two
  wrappers are used.

## Remaining fail-closed work

1. Inventory the actual replay schemas and freeze the complete adapter config.
2. Run query regeneration and fresh scans on one-node `debug` jobs.
3. Rebuild/receipt data, validation, tokenizer and initialization assets.
4. Run debug unit/scheduler smokes and bounded `normal` restart/profile/LR
   pilots.
5. Re-run the staged launch freezer. Main production remains blocked until all
   pre-main receipts and explicit owner production authorization exist.

## 2026-08-15 native-suite scan runtime correction

The first replay native-suite scan, job `3084724`, failed before scanning any
document because `run_native_suite_replay_scan_debug.sbatch` invoked the frozen
scanner with `pytorch/v2.6.0:v1`'s Python 3.13, whose worker processes could not
import PyArrow. The normalized 32-shard corpus and all earlier receipts passed;
the failed job published no scan receipt or exclusion authority.

The wrapper now verifies and uses the same bundle-bound Python 3.12/PyArrow 21
data runtime as the other R2 data wrappers, inside `pytorch/v2.9.1:v2`. Scanner
code, query bytes, corpus bytes, matching thresholds, worker count and
finalizer arithmetic are unchanged. The wrapper was added to the mandatory
runtime-enforcement test set, and its path is an explicitly allowed operational
producer-bundle difference. A fresh immutable bundle, runtime, complete debug
test gate and producer-compatibility authority are required before retrying.

## 2026-08-15 historical-tokenizer Capstor metadata correction

Historical-tokenizer materialization job `3085408` failed before publishing a
tokenizer directory or receipt. `shutil.copy2` copied the tokenizer bytes and
then attempted to propagate a read-only bundle extended attribute onto
Capstor, which rejected the metadata write with `PermissionError`. The
scientific contract requires byte-exact files, not source filesystem metadata.
The materializer therefore uses `shutil.copyfile` and continues to verify the
predeclared SHA-256 of every source and every published file. A regression test
requires byte-only copying and forbids `copy2`. A fresh bundle, runtime, full
test gate and producer-compatibility authority are mandatory before retrying.

## 2026-08-15 GPTDataset compiled-helper correction

Phase-1 GPTDataset cache job `3085644` failed before publishing a cache receipt
because the pinned Megatron clone did not contain the compiled
`megatron.core.datasets.helpers_cpp` extension. Preprocessing had succeeded
without this extension, so the omission was first exercised by high-level
GPTDataset index construction. Dependent Phase-2 job `3085646` never allocated
compute and was cancelled in `DependencyNeverSatisfied`; the partial Phase-1
cache was retained under `_failed/phase_1_cache_build_3085644`.

The training-runtime freezer now compiles the canonical c92402e dataset helper
from `megatron/core/datasets/Makefile` inside the pinned PyTorch 2.9.1 uenv,
records the binary hash and Python ABI, and performs an import smoke before
freezing a new runtime. Cache construction and production preflight require
that helper authority. Cache builds also quarantine partial cache roots on any
failure. The earlier training runtime and completed tokenized-stream receipts
remain immutable; a new runtime, code bundle, complete debug test gate and
producer-compatibility authority are required for the retry.

## 2026-08-15 initialization lineage closure

The first initialization wrappers authenticated their executing code bundle but
only tested several upstream model and receipt paths for existence. That was
not sufficient evidence that the exact pinned Hugging Face materializations,
benchmark-clean TD inputs, corrected geometry view and round-trip output formed
one continuous immutable chain.

The initialization path now fails closed on that lineage:

- the regenerated TD-input receipt binds the pre-main dataset authority, both
  Stage-B stream receipts and their exact ordered input identities;
- the 1.5B ReTok reference verifies the complete parent-model inventory and
  the complete frozen tokenizer receipt before producing a complete output
  inventory;
- the 1.5B TD verifier binds the parent materialization, ReTok reference,
  benchmark-clean TD inputs, tokenizer, 8B TD materialization and predeclared
  row-norm contract, and inventories the entire TD output tree;
- each training-geometry view verifies the complete upstream authority and
  inventories every output file; and
- each HF/Megatron/HF round trip verifies and binds the geometry receipt before
  accepting exact tensor and logit parity.

These changes alter only evidence and failure handling. They do not change the
148,480-token tokenizer, selected token ids, snippets, target layers, TD
optimizer, model weights, RoPE target, conversion revision, or training
hyperparameters. A fresh immutable bundle and its full Clariden test gate are
required before any initialization job resumes.
# 2026-08-15 TD coverage execution-route evidence

The regenerated 2B-token TD snippet corpus remains a mandatory initialization
input.  Four bounded debug probes established that the scientific scan cannot
fit the debug wall-time.  The final v33 probe eliminated unbounded task
submission and held RSS near 6.2 GB, passed the exact sequential/batched parity
guard, and measured 50,002,836 tokens after 7m48s.  Linear completion is about
5.2 hours, versus the 90-minute debug limit.

The scan therefore moves to the CPU-only Clariden `xfer` partition, where the
historical scan ran and where the live limit is 24 hours.  This is an
operational routing change only: input order, tokenizer bytes, target token
budget, new-ID range, seed, reservoir state-update order and parity guard are
unchanged.  `build_td_xfer_runtime.sbatch` freezes an x86_64 runtime bound to
the scientific bundle, and `build_td_snippets_xfer.sbatch` verifies both before
running.  No `normal` allocation and no GPU are consumed.

## 2026-08-15 exact Stage-B Unicode preservation

The first complete `xfer` coverage run, job `3086029`, scanned the full
2B-token budget in 48m29s but refused to publish its TD-input authority because
93 scanned documents were not already in NFC form.  The failure was safe: no
stable coverage or TD-input receipt was written, and the dependent 1.5B TD job
`3086117` consumed no allocation before it was cancelled.

The selected historical Stage-A/Stage-B curriculum pipeline did not normalize
text to NFC.  Adding normalization only for this reconstruction would make the
TD snippets differ from the exact Stage-B strings that the matched training
streams consume, while normalizing the training streams would introduce a new
scientific data transform outside the approved near-replication design.
Accordingly, the corrected scan preserves the exact Stage-B Unicode bytes and
audits NFC status independently.  Its immutable receipt must record the total
documents scanned, the non-NFC count and fraction, `input_bytes_transformed =
false`, and policy `audit_and_preserve_exact_stage_b_text`.  This is recorded
as a named initialization reconstruction difference; it does not modify the
training data, tokenizer, selected token ids, reservoir algorithm, TD recipe,
or token order.

## 2026-08-15 prelaunch producer-authority propagation

The first v41 8B profile-contract freeze, job `3086334`, failed before any
`normal` allocation was requested. Its phase-cache validator accepted only the
executing bundle by default, even though the phase caches had been built by an
older bundle that the explicit producer-compatibility authority had already
audited. The benchmark-contract and in-allocation preflight paths now bind that
authority, require full receipt-level producer membership for the
initialization, both phase caches, the Megatron runtime and online validation,
and pass only the resulting accepted root/tree set into live payload
validators. This neither weakens provenance to arbitrary historical bundles
nor rebuilds any scientific artifact.

## 2026-08-15 CSCS uenv-v10 Slurm-step mounting

The first 16-node 8B profile job, `3086397`, passed its complete provenance
preflight but failed in the first training subrun after 2m12s. Every rank
reported `torchrun: not found`; a one-node reproduction (`3086409`) showed the
Slurm task resolving `/usr/bin/python3`, with no `torch` import, even though
the parent `uenv run ... srun` process exported a `/user-environment` PATH.
No optimizer update or profile receipt was produced.

The subsequent v43 producer-compatibility freeze (`3086431`) also failed
closed because the compatibility allowlist did not yet name the exact frozen
trainer path changed by that operational patch.  The allowlist now names only
`frozen_training_tools/bakeoff_training/bakeoff_train.sbatch`; it does not
permit a directory wildcard.  A fresh immutable bundle and complete debug
gate are required before compatibility or distributed qualification is
retried.

CSCS uenv v10 requires the image and view to be attached through the Slurm
plugin for remote steps. The frozen trainer is therefore patched after copying
the clean historical source so both launch modes use `srun
--uenv="$UENV_IMAGE" --view=default`. A one-node debug proof must resolve both
Python and `torchrun` inside `/user-environment`, import PyTorch 2.9.1, and bind
the result to the immutable code bundle before another multi-node profile job
is submitted. This changes only runtime mounting, not the training command,
scientific recipe, rank geometry, data, initialization or checkpoint state.

## 2026-08-15 1.5B reserved-token compatibility

The first 1.5B TD job (`3086329`) failed safely in 22 seconds, before TD
training or publication, because the pinned 1.5B HF tokenizer's special-token
map differs from the shared historical tokenizer. Exact inspection found 14
base-vocabulary content differences, all in reserved/special slots, plus four
same-content special-flag changes; 131,058 ordinary token strings retain their
ids and all 269,443 base merges are an exact prefix of the target tokenizer.
The source package also leaves padding undeclared while its model config says
`pad_token_id=3`; the shared target tokenizer declares `<pad>` at id 3.

`configs/1p5b_tokenizer_compatibility_v1.json` now freezes every allowed
difference. The ReTok builder verifies that contract and preserves all base
input/output rows by id without permutation. Any additional vocabulary,
pipeline, merge, special-record or pad-id difference remains fatal.

The first contract-aware retry (`3086502`) passed that tokenizer gate but
failed before output because the architecture bridge loaded the canonical
`retok.py` outside its frozen tool directory, so its top-level `from _common`
could not resolve. The loader now temporarily prepends only the immutable
tool directory, restores both `sys.path` and any pre-existing `_common` module
after import, and has an executable sibling-import isolation test.

The next retry (`3086546`) proved that the immutable bundle itself omitted
`_common.py`; it failed at the same pre-output import boundary. The deployer
now requires the clean canonical `_common.py`, copies it beside `retok.py`,
and the complete remote test gate imports the exact frozen pair before any TD
job can run.
