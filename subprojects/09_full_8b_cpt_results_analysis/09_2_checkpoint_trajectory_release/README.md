# 09.2 — Checkpoint trajectory release

> **In one line:** scored the twelve saved 8B exports that had never been evaluated, closing the matrix at 19/19 checkpoints, and staged the whole trajectory as private, receipt-bound Hugging Face branches with two frozen companion datasets.
> **Period:** 2026-08-17 → 2026-08-20 (`894fca3f` … `ae2acd30`; the 08-20 averaged-branch publication is recorded in [`../09_1_downstream_task_instability/CHECKPOINT_AVERAGE_RESULTS_20260819.md`](../09_1_downstream_task_instability/CHECKPOINT_AVERAGE_RESULTS_20260819.md), not here). **Status:** completed; repository remained **private**.
> **Came from / led to:** the three-checkpoint screen and peak-window scoring → this → the complete 19-checkpoint report in [`../presentations/`](../presentations/).

## Why this existed

After the three-point screen (2026-08-12) and the four-point peak window (2026-08-17), six of the run's checkpoints had native-Greek scores and twelve did not. Without them, any statement about *when* a capability peaks rested on six samples of an 18,284-update trajectory. This adapter owned exactly three things — the checkpoint list, the result joining, and the Hugging Face release metadata. It changed no weights, prompts, benchmark rows, scorer or corpus, and did not touch the six existing results.

## History

**2026-08-17 (`894fca3f`)** — the twelve missing exports were bound (1.678, 5.000, 9.999, 14.999, 19.998, 24.998, 54.996, 59.995, 61.350, 64.995, 69.995, 74.994 B token slots) to the exact strict population already published with `glossapi-greek-nanochat-pretraining-dataset-v2` at revision `987b8955…`: 83,970 source rows, 10,076 strong-match exclusions, **73,894 retained**. GreekMMLU was deliberately not rerun — every saved checkpoint already had a frozen clean GreekMMLU receipt — and Protipa stayed out of scope behind its unapproved access gate.

**2026-08-17 → 08-18 — four rewrites of the allocation shape, no change to the science.** The plan moved from a two-node debug profile, to a held `salloc --no-shell` captured visibly and attached with `srun` (`d411cb01`, `bdb9fc5a` — explicitly to avoid another queued batch wrapper and keep the allocation available for retry), to a one-node four-GPU 80-minute profile in eighteen receipt-bound segments (`9b17b6c5`), to allowing an explicit `normal` partition (`73f527d9`). `test_remaining12_resource_profile.py` exists to guard that these are resource changes only. Two commits (`c171b891`, `3933e2f6`) were spent keeping scheduler logs out of the frozen evaluation bundle so its hash stayed meaningful.

**2026-08-18 — the publication adapters** (`779ca47d` and the eight commits after it). Weight uploads were decoupled from the evaluation: checkpoints were repaired first as **private** immutable branches on `fffoivos/apertus-8b-greek-cpt`, each card carrying only its already-frozen GreekMMLU point and saying that the native matrix was pending. Two frozen datasets were staged over an Xfer allocation: a **public** `…-modern-greek-train` reconstructed from the revision-pinned upstream Parquet plus per-document content hashes and the already-approved PII masker — no extra policy, dedup or retokenization pass (`def7a027`) — and a **private** `…-d0-full-mix` holding the exact packed 79/20/1 D0 payload, kept private because replay-source redistribution was never authorized. The Xfer Python environment took three commits to pin (`b41b0903`, `29a1d304`, `2f4f4ed7`) and is built on the Xfer node itself, never copied from the Mac or a GPU uenv.

**2026-08-19 (`b18930bf`)** — a nested `srun` attach step exposed a single CPU to the frozen segment wrapper, making a normal-allocation resume impossible. Rather than edit the frozen wrapper, `workaround_resume_remaining12_normal.py` was written as an experiment-owned recovery *driver*: it relaunches the already-frozen per-shard command, skips completed receipts and runs the existing aggregate verifier. The matrix completed and was independently verified at **252/252 shards** (21 shards × 12 checkpoints).

**2026-08-19 (`ae2acd30`)** — with the matrix closed, the metadata-only release ran. Eighteen ordered branch aliases `00-step400-tokens2B` … `17-step18284-tokens77B` were created over the existing private commits, the model-file inventory was verified unchanged, and only then were the old unprefixed aliases removed. Receipts for the dry run, the recovery dry run and the completed pass are in [`publication/receipts/`](publication/receipts/); the completed one records `status: "completed"` with the final ref → commit map and `index_sha256 5aa27ab8…`. The published index status is `complete_private_metadata_release`.

## Outcome

- **All 19 saved checkpoints scored on one strict 73,894-example population** — the basis of [`../presentations/FULL8_ALL_CHECKPOINT_NATIVE_BENCHMARKS_20260819.html`](../presentations/FULL8_ALL_CHECKPOINT_NATIVE_BENCHMARKS_20260819.html).
- **18 ordered private model branches plus `main`** on `fffoivos/apertus-8b-greek-cpt`, with per-branch provenance bound to the two frozen companion datasets.
- **Two companion datasets frozen**: the public exact Modern-Greek training text, and the private full D0 mixture.
- The repository was **never made public**; the release code says explicitly that only a separate, explicit decision could do that. On 2026-08-20 two checkpoint-average branches (`18-…`, `19-…`) were added by the 09_1 work and `default_revision` moved to `18-avg-uniform5-tokens30B-50B`.
- Left open at the end: public release of the model repository, and Protipa.

## Where things are

| What | Path |
| --- | --- |
| The twelve checkpoints and their bindings | [`evaluation/remaining12_checkpoint_bindings.json`](evaluation/remaining12_checkpoint_bindings.json) |
| Rebind the clean population onto them | [`evaluation/rebind_remaining12_native_suite.py`](evaluation/rebind_remaining12_native_suite.py) |
| The frozen scorer segment / preflight | [`evaluation/run_remaining12_native_segment.sbatch`](evaluation/run_remaining12_native_segment.sbatch), [`evaluation/freeze_and_preflight_remaining12.sbatch`](evaluation/freeze_and_preflight_remaining12.sbatch) |
| The allocation-recovery driver | [`evaluation/workaround_resume_remaining12_normal.py`](evaluation/workaround_resume_remaining12_normal.py) |
| Release metadata assembly | [`evaluation/assemble_trajectory_release_metadata.py`](evaluation/assemble_trajectory_release_metadata.py), [`publication/update_ordered_hf_checkpoint_metadata.py`](publication/update_ordered_hf_checkpoint_metadata.py) |
| What actually got published | [`publication/receipts/ordered_hf_checkpoint_metadata_20260819.completed.json`](publication/receipts/ordered_hf_checkpoint_metadata_20260819.completed.json) |
| Dataset staging and upload | [`publication/export_public_modern_greek_train.py`](publication/export_public_modern_greek_train.py), [`publication/prepare_full8_d0_private_stage.py`](publication/prepare_full8_d0_private_stage.py), [`publication/upload_frozen_dataset.py`](publication/upload_frozen_dataset.py) |
| Pinned Xfer runtime | [`publication/xfer_requirements.txt`](publication/xfer_requirements.txt) |
| Tests | [`evaluation/test_*.py`](evaluation/), [`publication/test_publication_adapters.py`](publication/test_publication_adapters.py) |
