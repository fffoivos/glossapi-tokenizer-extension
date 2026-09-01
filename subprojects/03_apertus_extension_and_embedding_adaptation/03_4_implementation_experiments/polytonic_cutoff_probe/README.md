# polytonic_cutoff_probe — choosing the production polytonic cutoff

> **In one line:** two months after the bakeoff closed, the polytonic layer that had been dropped from it came back as a properly pre-committed choice between **+512 and +1,024** appended merges; +512 won and became the production tokenizer (vocab **148,992**) for the cleaned Greek CPT corpus.
> **Period:** 2026-07-29 (run, decision, freeze). Scripts first committed 2026-07-31 (`f0dc31a0`); the rest recovered from the owner's working tree on 2026-09-01 (`2aec4a66`). **Status:** completed and frozen.
> **Came from / led to:** [`../init_bakeoff/token_distillation/`](../init_bakeoff/token_distillation/README.md) (the TD machinery it reuses) → this → the 148,992 bundle at [`../../03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_polytonic_148992/`](../../03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_polytonic_148992/) and the production init in `05_token_distillation_cpt/06_25b_midtraining_probe/initialization/build_production_init.sbatch`.

## Why this existed

The 2026-05-20 scope decision took polytonic out of the bakeoff, and the old +5,120 / 153,600 bundle was never trained. When a cleaned Greek CPT corpus needed a tokenizer, the question returned in a smaller and much better-posed form: how many polytonic merges are worth appending to the modern 148,480 tokenizer, given that every added row costs adaptation budget and risks degrading modern Greek? Unlike the bakeoff, this one had its decision rule written down **before** the numbers existed.

## The pre-committed contract

- Preserve every ID and merge below 148,480 exactly.
- Keep structural ByteLevel merge fragments (not standalone UTF-8, but their compositions decode to valid polytonic Greek); merge-chain initialise them and exclude them only from independent token-distillation targets.
- Run the same fixed NFC FineWeb-2 ancient and modern Greek streams for baseline / +512 / +1,024. Reject any evaluation with tokenizer-specific document truncation.
- Reject a candidate whose modern-Greek BPB regresses more than **0.5 %** against baseline.
- Choose +1,024 only if it passes that guard **and** improves ancient BPB by ≥ **1 %** over +512. Otherwise take +512 if it passes; if +512 fails, select nothing.

## History (2026-07-29)

| Stage | What happened | Evidence |
|---|---|---|
| Assets + coverage (job `2922887`) | Candidate tokenizers built, evaluation streams frozen, added-token firing frequencies counted over a 30 M-token scan | [`prepare_probe_assets.sbatch`](prepare_probe_assets.sbatch), [`count_token_frequencies.py`](count_token_frequencies.py) |
| First model probe (job `2922926`) | **Failed both modern guards by ~26 %**, despite almost unchanged modern token counts. Cause: positive-only token distillation had made the new output rows overconfident, inflating the expanded softmax denominator. The gate failing here is the gate working | `PRODUCTION_DECISION_20260729.md` §"Runtime issue found and fixed" |
| Calibration data (job `2924310`) + corrected probe (job `2924312`) | A bounded pass that freezes the whole model, alternates disjoint ancient/modern calibration blocks, updates only IDs ≥ 148,480, exact-checks every old input and output row, and uses 2,000 calibration documents with zero eval-text overlap. **+512 modern regression fell from 26.29 % to 0.138 %** | [`calibrate_new_output_rows.py`](calibrate_new_output_rows.py) |
| Suspicious-token review | Three historically flagged IDs inside the cutoff (148924 `ἷς` 3,568 firings; 148979 `ὓς` 3,619; 148987 `Ἦχος`/`Ὦχος` 438) reviewed and **kept** — each is a partial UTF-8 ByteLevel merge component firing inside a valid polytonic surface, not mojibake; removing one would break a live merge dependency | [`build_suspicious_review.py`](build_suspicious_review.py), `suspicious_token_review.json` |
| Decision | **+512 selected** | [`PRODUCTION_DECISION_20260729.md`](PRODUCTION_DECISION_20260729.md), [`selection.json`](../../03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_polytonic_148992/selection.json) |

### The gate table

| Metric | Modern-148480 | **+512** | +1,024 |
|---|---:|---:|---:|
| Ancient tokens (311 docs) | 221,695 | **204,809** | 198,148 |
| Ancient tokens vs baseline | — | **−7.62 %** | −10.62 % |
| Ancient single-token word rate | 20.13 % | **41.05 %** | 43.23 % |
| Ancient BPB after bounded adaptation | 0.8533 | **0.9124** | 0.9283 |
| Modern BPB (1,000 docs) | 0.5178 | **0.5185** | 0.5190 |
| Modern BPB ratio to baseline | 1.0000 | **1.00138** | 1.00231 |

Both candidates passed the 0.5 % modern guard. +1,024 modelled ancient text **1.74 % worse** than +512, so it did not earn its extra 512 rows.

## Outcome

- **Frozen production tokenizer:** vocab 148,992 = 256 × 582 (131,072 base + 17,408 modern + 512 polytonic), 512 appended merges, zero orphan entries, zero external padding, `tokenizer.json` sha256 `bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b`. Published on HF as `fffoivos/apertus-tokenizer-extension`, subfolder `greek-modern-polytonic-tokenizer`.
- **The +5,120 / 153,600 bundle is retired** to historical specialization artifact and is not the tokenizer for the corpus materialization that followed.
- **Scope discipline:** no new-dataset tokenization was run as part of this decision, and the adapted checkpoint under `results/calibrated_0512/` is evidence for tokenizer selection, **not** a release model. The ancient BPB numbers are a short adaptation probe — the next CPT run still has to train the selected rows.
- **Reused as a build step:** `build_incremental_checkpoint.py` (merge-chain init of appended rows) and `calibrate_new_output_rows.py` are called directly by subproject 05's production-init job, sandwiching a layer-11 TD pass.

## Where things are

| What | Where |
|---|---|
| The decision | [`PRODUCTION_DECISION_20260729.md`](PRODUCTION_DECISION_20260729.md) |
| Frozen tokenizer + audits | [`../../03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_polytonic_148992/`](../../03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_polytonic_148992/) — `manifest.json`, `release_audit.json` (status `passed`), `selection.json`, `suspicious_token_review.json` |
| Submit the probe | `RUN_ROOT=<clariden run root> bash submit_probe.sh` — refuses duplicate submission, chains assets → probe with `afterok`, writes `submission/jobs.tsv` |
| Init + calibration (also used by 05) | [`build_incremental_checkpoint.py`](build_incremental_checkpoint.py), [`calibrate_new_output_rows.py`](calibrate_new_output_rows.py) |
| Probe pipeline | [`prepare_probe_jsonl.py`](prepare_probe_jsonl.py), [`prepare_calibration_jsonl.py`](prepare_calibration_jsonl.py), [`run_cutoff_probe.sbatch`](run_cutoff_probe.sbatch), [`run_calibrated_probe.sbatch`](run_calibrated_probe.sbatch), [`aggregate_probe.py`](aggregate_probe.py) |
| Release verification | [`verify_production_tokenizer.py`](verify_production_tokenizer.py) + sbatch |
| Remote receipts | Clariden `/iopsstor/scratch/cscs/fffoivos/tokenizer_finalization/20260729T094000Z-poly512-1024/` — both the failed uncalibrated pass and the corrected selection are retained under `production_cutoff_candidates/model_probe/` |

## Working documents

Nothing superseded here — the decision doc and this history are the record. Note that the entire directory beyond the two initialisation scripts, plus the 148,992 ship bundle, was never committed during the work and was recovered from the owner's working tree on 2026-09-01 (`2aec4a66`, "Recover uncommitted working-tree files"); the content dates from 2026-07-29.
