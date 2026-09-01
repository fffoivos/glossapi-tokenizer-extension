# 05 — Training-dataset bridge (corpus → Megatron binaries)

> **In one line:** built to turn a validated Phase-04 private release into fresh Megatron `.bin`/`.idx` shards for a 25B Token-Distillation probe; that probe was never launched from here, but the bridge's shard builder and receipt library became the shared tokenisation layer for every later CPT subproject.
> **Period:** 2026-07-12 → 2026-08-07 (commits touching this directory). **Status:** superseded as a launch path by [`../06_25b_midtraining_probe`](../06_25b_midtraining_probe), which calls it "the stale single-blend `05_training_dataset_bridge` launch path"; still live as a code dependency of subprojects [06](../../06_dataset_scheduling_experiments), [07](../../07_full_8b_cpt) and [08](../../08_targeted_8b_cpt_experiments).
> **Came from / led to:** [`../04_full_corpus_preparation`](../04_full_corpus_preparation) → this → [`../06_25b_midtraining_probe`](../06_25b_midtraining_probe), [`../../07_full_8b_cpt`](../../07_full_8b_cpt), [`../../08_targeted_8b_cpt_experiments`](../../08_targeted_8b_cpt_experiments).

## Why this existed

By 2026-07-11 the curriculum-v2 Megatron `.bin`/`.idx` payloads on Clariden were gone — "zero files" ([`../LOG.md`](../LOG.md), 2026-07-11) — so the next run could not reuse them. Something had to stand between a cleaned Parquet corpus and a GPU launch and prove, before any GPU time was spent, that: every document was encoded exactly once, held-out evaluation documents were excluded from training, replay sources were restaged from pinned revisions rather than refetched from a moving `main`, and no pool reached its sample target by repeating rows.

## History

### 2026-07-12 — built in one pass, then hardened twice

`01cba0ee` created the whole directory (27 files, ~6,050 lines): the frozen recipe [`configs/frozen_25b_td.json`](configs/frozen_25b_td.json), replay restaging [`configs/replay_acquisition.json`](configs/replay_acquisition.json), nine Slurm stages, ten scripts and the launcher. `76a44479` and `3d063bfd` then closed the launch and acquisition contracts, growing `train/full_corpus_25b.env` from 44 to ~226 lines so that every full-run semantic is asserted *after* the shared environments are sourced.

The frozen recipe: 79% new Greek / 20% Apertus-family foreign replay / 1% old-Greek replay; 25,000,000,000 nominal tokens which integer batches make **5,960 steps / 24,998,051,840 effective tokens** (floor residual 1,948,160); sequence 4,096, global batch 1,024; ModernGreek-148480 tokenizer at revision `a4826df7…`; TD layer 11; LR `5.5e-5` → `5.5e-6` with 400-step warmup and a final-20% `1-sqrt` cooldown; AdEMAMix `0.9/0.999/0.999`, alpha 4; Goldfish 50/50.

Three rules were designed in from the start and never relaxed:

- **Encode once.** Mixture shares are Megatron `--data-path` sampling weights, not row multiplicity. Finalisation discounts exact-content duplicates and requires every pool, weighted foreign source and physical prefix to hold `ceil(planned_samples × 1.005) + 1` non-repeating samples. If capacity is short, `finalize_bridge.py` writes `bridge_capacity_failure.json` and stops — the remedy is more data or an explicit recipe review, never silent reuse.
- **Identity is composite.** `full-cpt-document-identity-v2`: shard-local upstream IDs are file-scoped, and old Greek is keyed by `(source_dataset, source_doc_id)` — `source_doc_id` alone is forbidden.
- **Nine deterministic heldouts**, rebuilt rather than inherited: `hplt`, `openarchives`, `greek_phd`, `english`, `de`, `ru`, `zh`, `code`, `old_greek`, selected by a domain-separated SHA-256 threshold with a 2 GB character budget per set and a 0.25 pool-fraction cap. Every exclusion is checksum-bound and mandatory; finalisation proves each selected ID was excluded exactly once.

`configs/replay_acquisition.json` recorded the awkward part honestly: the historical FineWeb-Edu, FineWeb-2/HQ, FineMath and StarCoderData commits were recovered from retained acquisition-day cache refs and a completed staging log because the payload copies were gone. Restaging deliberately does **not** expand the pinned FineWeb2-HQ globs to all 5,285 matching shards (5.36 TB); it keeps the full 10BT English sample and old-Greek inputs and uses a seed-20260609 domain-separated SHA-256 file ranking for capacity-sized multilingual and FineMath samples (`62d4aac3`).

### 2026-07-12 → 07-31 — the launch never happened here

The README written at creation listed the external prerequisites bluntly: Clariden had empty replay and Megatron skeletons, the ModernGreek-148480 tokenizer tree had to be restored, the Swiss-AI Megatron checkout at `c92402e…` had to be restored, the patched TD layer-11 checkpoint directories were "empty skeletons", and a passed Phase-04 private release did not yet exist. It never did in the form this bridge expected — [`../04_full_corpus_preparation`](../04_full_corpus_preparation) shipped its corpus through the Agent 1 v5 lane and Hugging Face, not through the v2 Stage-80 materialisation path that `freeze_inputs.py` demands.

### 2026-07-31 — repurposed as a library

`8dbb6d25` created [`../06_25b_midtraining_probe`](../06_25b_midtraining_probe) and, in the same commit, taught this directory's shard builder to serve it:

- `build_binary_shard.py` gained an optional **receipt-bound phase partition**. A task may carry `phase_partition` naming a corpus (`new_greek` or `replay`), a phase, a seed and a logical pool; the implementation module is loaded only after its declared SHA-256 matches, and rows outside the selected phase are counted as `phase_excluded_rows` rather than silently dropped. This is what makes the probe's disjoint two-phase document allocation auditable.
- `build_heldouts.py` gained `selector_not_regex` and applied GreekMMLU decontamination while selecting heldout documents, counting `contaminated_selected_documents_dropped`.

`5ede1f62`, `25c6375d` and `62d4aac3` hardened replay staging on the same day: malformed benchmark receipts tolerated, Hugging Face snapshot symlinks resolved before staging, and acquisition bounded to deterministic capacity samples.

### 2026-08-01 — heldouts may overlap

`86c1b8fe` replaced a hard error with a documented policy. Evaluation slices intentionally overlap (a broad HPLT slice and a historical/source-specific slice can select the same document), so the training exclusion is now their **set union**: one identity is retained, every duplicate membership is receipted (`duplicate_memberships`, `overlapping_documents`, `merge_policy: set_union_across_heldout_components_v1`). A repeat *within* a single component is still a hard error.

### 2026-08-07 — adopted by the full-8B sanitized restart

`5b6dd260` ("Prepare receipt-gated sanitized full-8B restart") was authored on a branch that did not contain this directory and re-added `bridge_common.py` and `build_binary_shard.py` under this path so the 8B anonymisation pipeline could import them. On the consolidation branch the two lineages converged: today's `build_binary_shard.py` is that version plus the phase-partition hunk, merged on 2026-09-01 (`600148e6`).

## Outcome

- **The 25B probe was not launched from here.** [`../06_25b_midtraining_probe/README.md`](../06_25b_midtraining_probe/README.md) opens by calling this the "stale single-blend launch path" it replaces, and pins the public HF v2 corpus and a two-phase blend instead of a Phase-04 local release and a single 79/20/1 blend.
- **The code outlived the launch path.** `scripts/build_binary_shard.py` and `scripts/bridge_common.py` are imported or invoked by: `../06_25b_midtraining_probe/clariden/{build_train_shards,build_heldout_shards}.sbatch` and `dataset/freeze_inputs.py`; `../../06_dataset_scheduling_experiments/clariden/{build_train_shards,build_heldout_shards,pack_catalog_bucket,prepare_data_mixes,validate_partition_group}.sbatch`; `../../07_full_8b_cpt/dataset/anonymization/*`; and `../../08_targeted_8b_cpt_experiments/clariden/*` plus `scripts/run_parallel_task_batch.py`.
- **Contracts that propagated:** encode-once with a 1.005× unique-capacity floor, checksum-bound heldout exclusion proven at finalisation, composite document identity, and a launcher that re-verifies every `.bin`/`.idx`, the init checkpoint, the Megatron tree and both training environments *twice* — once before submission and again at job start inside the pinned uenv.
- **Left open:** the replay restaging and old-Greek rebuild described here were prerequisites, not receipts; there is no run record in this tree for `BRIDGE_RUN_ID=full-corpus-25b-v1`.

## Sub-subprojects

| Dir | Role | Period | Status | Result |
|---|---|---|---|---|
| [`scripts/`](scripts) | The ten Python tools: restage, freeze, build heldouts, encode shards, finalise, freeze assets, verify launch | 2026-07-12 → 08-07 | two of ten still in production use | `build_binary_shard.py` + `bridge_common.py` became the shared library |
| [`clariden/`](clariden) | Nine sbatch stages and the `restage` / `freeze` / `after-freeze` dispatcher | 2026-07-12 | unused as a chain | Dry-run default; `CONFIRM_BUILD=1` / `CONFIRM_RESTAGE=PINNED_REPLAY_V1` to submit |
| [`configs/`](configs) | The frozen 25B recipe and the replay-acquisition pin set | 2026-07-12 → 07-31 | frozen | 5,960 steps / 24,998,051,840 tokens; nine heldouts; recovered replay revisions |
| [`train/`](train) | The receipt-gated 25B launcher and its effective-recipe environment | 2026-07-12 | never launched | Superseded by `../06_25b_midtraining_probe/clariden/train_segment.sbatch` |
| `tests/` | `test_training_bridge.py`, the single suite covering identity, heldouts, capacity, receipts and the launch gate | 2026-07-12 → 08-01 | maintained | Grew from 332 to ~779 lines alongside each hardening pass |

## Working documents

There are no dated plan or status files in this directory — its history lives entirely in commits, in the frozen configs, and in the README states summarised above. The 2026-07-12 "Current external prerequisites" list has been folded into the History section as the reason the launch path went unused.
