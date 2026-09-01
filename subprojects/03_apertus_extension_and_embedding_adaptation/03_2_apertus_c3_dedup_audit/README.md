# 03_2 — Apertus × C3 dedup audit

> **In one line:** measure how much of the Greek CPT corpus Apertus-8B-2509 had already seen during pretraining, so the init bakeoff could train on genuinely fresh text; it ran on 2026-05-19 and found **~2.27 % document overlap**, publishing a hard-drop overlay — but the held-out contamination check it was also supposed to do was skipped and became unrecoverable a day later.
> **Period:** 2026-05-18 (plan + 3 review rounds, first partial run) → 2026-05-19 (full run, report, HF publish). Committed into the repo on 2026-05-21 (`3aa2cf71`).
> **Status:** completed with one permanent gap (held-out contamination).
> **Came from / led to:** the frozen 17,408 cutoff in [`02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](../../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md) → this → the CPT corpus build in [`../03_4_implementation_experiments/init_bakeoff/corpus_build/`](../03_4_implementation_experiments/init_bakeoff/corpus_build/README.md).

## Why this existed

The parent subproject had two open questions that both turn on the same unknown: how much *new* Greek signal does continued pretraining actually deliver? The Apertus Greek-share number already known (0.023 % of realised pretraining tokens) gives total exposure, not per-document overlap with *our* corpus. Without it, the replay ratio is a guess and the held-out eval slices could be leakage-contaminated. This is measurement only — no corpus is rewritten and no model is trained.

## History

| Date | What happened | Result / decision | Evidence |
|---|---|---|---|
| 2026-05-18 | `PLAN.md` written, then reviewed four times before execution | r1: split the overlap into *source-universe* vs *consumed-estimate* axes, and run the full ladder (not md5) on the held-out check. r2: `consumed-estimate` renamed `consumed_exposure_estimate` with an honesty caveat — it is probability-weighted, not a per-document "seen" claim; tier verb changed from SEEN to **EXPOSED**. r3: user decision **`greek_diacritic_policy = preserve`**, because Apertus's tokenizer has `normalizer: null` and encodes ά (U+03AC) differently from α, so stripping diacritics would conflate texts Apertus distinguishes. r4: scope split into `hf_source_pool_overlap` (this run) vs `c3_exact_mix_overlap` (never run — needs the C3 mix manifest, which lived on a terminated GCloud instance) | [`REVIEW_INTEGRATION_20260518.md`](REVIEW_INTEGRATION_20260518.md) → [`_round2`](REVIEW_INTEGRATION_20260518_round2.md) → [`_round3`](REVIEW_INTEGRATION_20260518_round3.md) → [`_round4`](REVIEW_INTEGRATION_20260518_round4.md) |
| 2026-05-18 14:01 | First run `dedup_20260518T140127z` fires after 9 green pre-flight checks | **partial** — only 4 workers, 149,908 pool docs, 2 matches. Archived, superseded | [`READY_TO_SPIN_UP.md`](READY_TO_SPIN_UP.md), [`REPORT_dedup_20260518T140127z.md`](REPORT_dedup_20260518T140127z.md), [`manifests/run_dedup_20260518T140127z_archived_partial_4w/`](manifests/run_dedup_20260518T140127z_archived_partial_4w/) |
| 2026-05-18 23:15 | Second attempt `dedup_20260518T231526Z`, 8 workers | torn down 2026-05-19 01:08, zero instances remaining; no report retained | [`manifests/run_dedup_20260518T231526Z/teardown_log.txt`](manifests/run_dedup_20260518T231526Z/teardown_log.txt) |
| 2026-05-19 01:09 | **Canonical run `dedup_20260519T010924Z`** — 8 × `c4-highcpu-192` spot workers in `europe-west4-c`, 77 shards each, per-worker validation all `pass` | **98,203,721 HF source-pool docs audited, 2,223,781 matched (~2.27 %)**. All 20 sources except `HuggingFaceFW/finewiki` (48.4 % fresh → `include_half_weight`) recommended `include_full`. HPLT clean60 — the bulk of the pool at 48.7 M docs — is 95.7 % fresh | [`REPORT_dedup_20260519T010924Z.md`](REPORT_dedup_20260519T010924Z.md), [`manifests/run_dedup_20260519T010924Z/`](manifests/run_dedup_20260519T010924Z/) |
| 2026-05-19 04:32 / 09:02 | Workers and the joins VM torn down by the exit-trap driver; zero remaining both times | cost control worked as designed | [`teardown_log.txt`](manifests/run_dedup_20260519T010924Z/teardown_log.txt) |
| 2026-05-19 06:38 | Artifacts published | HF dataset `fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z`, `status: uploaded_complete` (overlap parquets + manifests + report; intermediate hash tables excluded) | [`manifests/run_dedup_20260519T010924Z/hf_repo.json`](manifests/run_dedup_20260519T010924Z/hf_repo.json) |
| 2026-05-21 | [`CPT_DATASET_BUILD_RUNBOOK.md`](CPT_DATASET_BUILD_RUNBOOK.md) written — the repeatable path from published corpus to a CPT-ready pool | Established the **order that matters**: hard-exclude the Apertus overlay *first*, then replay internal dedup, so an internal duplicate family can still keep a fresh alternate representative | `3aa2cf71` |
| 2026-05-20 (after) | GCloud access lost | the `c3_exact_mix_overlap` follow-on and the held-out re-run both became unreachable; the runbook's "GCP scratch VM" steps were re-hosted on Clariden `xfer` | [`../03_3_cscs_experiments_kickoff/ANALYSIS.md`](../03_3_cscs_experiments_kickoff/ANALYSIS.md) constraint update |

## Outcome

- **The load-bearing artifact:** `apertus_overlap_drop_docs.parquet`, consumed by every downstream corpus build. Its effect on the pool is ~2.27 % of documents at pool stage (98.2 M → ~95.98 M); after internal `drop_intra_and_inter` dedup the trainable pool is ~14.4 M docs.
- **Decision it enabled:** init pilots train on the Apertus-fresh-only pool, main CPT may use the mixed pool — argued in [`../03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md`](../03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md) §1.
- **Two things it did not answer.** (1) Held-out C3 val/test contamination — **SKIPPED**, no `holdout_doc_ids.parquet` was supplied; both reports say so explicitly. It reappears as gap **D5 / V1 (decontamination NOT DONE)** in [`../CPT_MASTER_20260526.md`](../CPT_MASTER_20260526.md) §5.1 and was still open when the subproject closed. (2) The exact sampled C3 mix overlap — the number that would actually be load-bearing for the replay recipe.
- **Scope limits stated in the report:** Apertus's long-context phase (FineWeb-Long, Institutional Books) was not measured; only Greek-bearing Apertus slices were audited.
- Later token accounting (2026-05-26) confirmed the audit's document counts but noted the published HF repo deliberately excludes raw text, so the *token* mass of the dropped documents needed a separate Clariden job — see [`../03_4_implementation_experiments/init_bakeoff/corpus_build/TOKEN_COUNT_AUDIT_20260526.md`](../03_4_implementation_experiments/init_bakeoff/corpus_build/TOKEN_COUNT_AUDIT_20260526.md).

## Where things are

| What | Where |
|---|---|
| Canonical result | [`REPORT_dedup_20260519T010924Z.md`](REPORT_dedup_20260519T010924Z.md) |
| Corpus build path that consumes it | [`CPT_DATASET_BUILD_RUNBOOK.md`](CPT_DATASET_BUILD_RUNBOOK.md) |
| Full plan (methodology, sources, sensitivity grid, risks) | [`PLAN.md`](PLAN.md) — 42 KB, the design of record |
| Coordinator pipeline (11 steps + teardown trap) | [`scripts/coordinator/`](scripts/coordinator/) — `00_pre_flight.py` … `10_build_cpt_final_overlay.py`, `run_all_with_teardown_trap.sh` |
| Worker fan-out | [`scripts/worker/`](scripts/worker/) |
| Run receipts (partition, per-worker configs, pre-flight, validation, teardown) | [`manifests/`](manifests/) — active run id in `manifests/CURRENT_RUN_ID` |
| Overlap parquets + published report | HF dataset `fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z` (public, gated: manual) |

Methodology pins used by all runs: `text_dedup.py` at commit `9a6b039` / file-hash `6b9bfdb0…`, blake3 exact hashing, 128-perm MinHash, token 5-shingles, Jaccard ≥ 0.85, short docs < 20 tokens skipped, `greek_diacritic_policy = preserve`.

## Working documents

Historical.

- **Superseded run:** [`REPORT_dedup_20260518T140127z.md`](REPORT_dedup_20260518T140127z.md) and [`manifests/run_dedup_20260518T140127z_archived_partial_4w/`](manifests/run_dedup_20260518T140127z_archived_partial_4w/) — the partial 4-worker attempt; its 149,908-doc numbers are not the audit's result.
- **Pre-launch snapshot:** [`READY_TO_SPIN_UP.md`](READY_TO_SPIN_UP.md) — go/no-go state for the 2026-05-18 run, including the single-command driver. Its cost estimate (~$30 for 8 spot workers, ~45 min) is the only cost figure recorded.
- **Review trail:** the four `REVIEW_INTEGRATION_20260518*.md` files — read them for *why* the scope is `hf_source_pool` and not `c3_exact_mix`; they are also the record of the diacritic-policy decision.
- **Run logs:** `manifests/run_*/teardown_log.txt`, `preflight_status.md`, `validation.json`, `bucket.txt` — receipts, one set per run.
