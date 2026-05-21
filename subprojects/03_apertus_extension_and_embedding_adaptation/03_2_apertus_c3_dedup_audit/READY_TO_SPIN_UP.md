# READY TO SPIN UP — `dedup_20260518T140127z`

Generated 2026-05-18 by autonomous prep run. **All pre-flight checks PASS.**

## State at a glance

| Item | Value |
|---|---|
| `RUN_ID` | `dedup_20260518T140127z` |
| Pre-flight | **all 9 checks PASS** — see `manifests/run_<RUN_ID>/preflight_status.md` |
| GCS bucket | `gs://eellak-glossapi-20251008-dedup_20260518t140127z` (europe-west4) — created and writable |
| Pinned `text_dedup.py` commit | `9a6b039` (verified) |
| Pinned file-hash | `6b9bfdb0bd9923349c348f80866c472101ab8fcf` (verified) |
| `greek_diacritic_policy` | `preserve` (user decision) |
| Worker scripts | 5 files in `scripts/worker/` (bash -n + py_compile pass) |
| Coordinator scripts | 11 files in `scripts/coordinator/` (bash -n + py_compile pass) |
| Plan | `PLAN.md` (after 3 review rounds) |
| Pin manifest | `manifests/run_dedup_20260518T140127z/text_dedup_pin.json` |

## Single-command go-ahead (recommended, post-r4-refactor)

After r4 review the pipeline is wrapped in a **single driver with a
teardown trap** that runs on any exit path. Sequence:

```bash
cd /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_2_apertus_c3_dedup_audit
export RUN_ID="$(cat manifests/CURRENT_RUN_ID)"
export HF_TOKEN  # already set in shell

# Step 01: enumerate shards (HF API only, no cost). Writes per-worker configs.
/home/foivos/.venvs/glossapi-merge-docling/bin/python3 scripts/coordinator/01_bytes_balanced_partition.py

# Step 02: COST EVENT — create 8 × c4-highcpu-192 SPOT workers (~$30).
bash scripts/coordinator/02_spin_up_workers.sh

# Step 03-09 + teardown trap — single driver:
bash scripts/coordinator/run_all_with_teardown_trap.sh
```

The `run_all_with_teardown_trap.sh` wrapper sets
`trap teardown EXIT INT TERM` so workers are torn down **whether
the run succeeds, errors, or is SIGINT'd**. No manual cleanup needed
afterward.

**Alternative single command (includes spin-up):**
```bash
bash scripts/coordinator/run_all_with_teardown_trap.sh --include-spinup
```
This bundles steps 01 + 02 + 03-09 + teardown into one command. Use
only when you're sure pre-flight is green and don't want a stop-point
between partition and spin-up.

## Or if you trust me to drive it

After you return + check `preflight_status.md`, just say:

> "go: spin up dedup workers"

and I will run steps 1 → 99 above with supervision.

## Estimated timing

| Step | Wall-clock |
|---|---:|
| 01 partition | ~3 min |
| 02 spin up | ~5 min |
| 03 dispatch + worker pipeline | ~30-45 min (parallel) |
| 04 poll loop | (covers step 3) |
| 05-09 joins + report | ~15 min |
| 99 teardown | ~2 min |
| **Total** | **~55-70 min** |

## Estimated cost

8 × `c4-highcpu-192` spot in europe-west4-b × ~50 min × ~$4.5/hr ≈ **$30**
+ GCS storage of run artifacts (~10 GB × 1 month) ≈ ~$0.20
+ Egress on download (~5 GB) ≈ ~$0.60
**Total ~$31**

## Kill-switch (immediate teardown)

```bash
gcloud compute instances delete \
  $(gcloud compute instances list \
      --filter="labels.workload=apertus-c3-dedup AND labels.run=dedup_20260518t140127z" \
      --format="value(name)") \
  --zone=europe-west4-b --quiet
```

## What to look at after the run

- `REPORT_dedup_20260518T140127z.md` — synthesised summary at subproject root
- `artifacts/dedup_20260518T140127z/per_c3_source_actionable.parquet` — the load-bearing CPT-recipe input
- `artifacts/dedup_20260518T140127z/holdout_contamination.parquet` — eval-integrity gating check
- `manifests/run_dedup_20260518T140127z/progress.jsonl` — append-only supervisor log

## Scope this run measures (post r4-reframe)

This run produces **`hf_source_pool_overlap`** — the broader HF source pool
(GlossAPI nanochat + HPLT clean60) vs Apertus pretraining. It does NOT
produce **`c3_exact_mix_overlap`** — the exact 1:1 sampled C3 training
mix vs Apertus. The c3_exact_mix scope requires the C3 mix manifest
(on the TERMINATED `apertus-greek-tokenizer-20260408t160000z` instance);
a separate audit run when that instance is reachable.

All output names are `hf_pool_*` not `c3_*` to keep downstream consumers
from confusing the universes.

## Known soft spots

1. **`08_holdout_contamination_check.py` needs `manifests/run_<RUN_ID>/holdout_doc_ids.parquet`** — a list of C3 val + test doc_keys. The C3 train/val/test splits live on the TERMINATED `apertus-greek-tokenizer-20260408t160000z` instance. Step 08 is non-fatal if the holdout list is missing; REPORT.md explicitly says "Held-out contamination check: SKIPPED" in that case.
2. **C3-exact-mix overlap is NOT produced by this run** — see scope clarification above. To produce it, start the gcloud instance, scp the C3 mix manifest to home as `manifests/run_<RUN_ID>/c3_mix_doc_ids.parquet`, and run a separate audit step (not yet scripted).
3. **`07_minhash_overlap_lsh.py` has an O(N×M) loop** over LSH candidate pairs after the band-join. Fine for typical near-dup recall but worth profiling on actual data sizes. If slow, switch to a polars `streaming` join.
4. **C3 GlossAPI side could SHORTCUT strict_exact** via the published bundle's `doc_dedup_metadata.parquet` (per PLAN §6.9 reuse decision). Currently the workers RECOMPUTE strict_exact on all HF-pool docs — adds ~5 min wall-clock. The recompute path produces verifiable hashes that should match the published bundle's `exact_strict_norm_v1` (full 64-char blake3 hex, per r4-fixed worker).

## Reference docs

- Plan: `PLAN.md` (after 3 review rounds)
- Review tracking: `REVIEW_INTEGRATION_20260518.md`, `_round2.md`, `_round3.md`
- Pin manifest: `manifests/run_dedup_20260518T140127z/text_dedup_pin.json`
- Pre-flight status: `manifests/run_dedup_20260518T140127z/preflight_status.md`
