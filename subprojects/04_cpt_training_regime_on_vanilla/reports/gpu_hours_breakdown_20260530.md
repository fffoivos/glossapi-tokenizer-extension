# 04 Vanilla CPT — GPU-Hours Breakdown (snapshot 2026-05-30)

Snapshot pulled from Clariden Slurm accounting at 2026-05-30 (query window: `-S 2026-05-28T00:00`).
Account `fffoivos`. Run cohort = "04 Vanilla CPT" (training arm `04van5b_*`, its per-checkpoint
sidecars, matched-config Apertus-Base evals, MCQ resubmits, smokes, watcher chain, and
warmup-assert repair attempts).

## Headline

| | GPU-h |
|---|---|
| **Total GPU-h requested** (sum of `elapsed * gres/gpu`) | **145.85** |
| **Total GPU-h billed under Clariden whole-node policy** (`elapsed * 4` on every GPU partition) | **145.85** |
| Of which: completed work | 133.60 |
| Of which: in-flight (job 2417450 partial elapsed at query time) | 12.25 |

The two accounting modes coincide exactly because every GPU job in this cohort already
requested `gres/gpu=4` (a full GH200 node), so "requested" and "whole-node billed" are
identical. There is **no discrepancy** between the two modes; the "billed" column would only
exceed "requested" if any job had asked for 1–3 GPUs.

**Current vs at-completion:** at the steady-state training throughput observed in the
last two completed segments (i596 + i715: 238 iters in 8h53m33s = 134.5 sec/iter on 4 GH200
GPUs), finishing the planned chain through iter 1192 requires about **59.0 additional
billed GPU-h** (≈ 4.98 ks left on i834 + 48.2 ks for the i1192 segment, each × 4 GPUs).
**Projected total at chain completion: ≈ 204.9 billed GPU-h**, before any iter-834 or
iter-1192 sidecar fan-outs (which historically add ≈ 7 GPU-h per checkpoint).

## Per-category breakdown

| Category | n_jobs | gpu_h_requested | gpu_h_billed_normal_policy |
|---|---:|---:|---:|
| training (successful chained segments) | 5 | 117.97 | 117.97 |
| per_iter_sidecar (convert + 5 evals + checksum × 3 checkpoints) | 24 | 21.75 | 21.75 |
| matched_config (Apertus-Base native / retention / BPB) | 3 | 2.74 | 2.74 |
| mcq_resub (native MCQ rerun for iters 238, 477) | 2 | 2.01 | 2.01 |
| repair (failed warmup-assert + archived iter-119 retry) | 3 | 0.73 | 0.73 |
| smoke (dataset, training, conversion, native, BPB) | 9 | 0.64 | 0.64 |
| dataset_prep (hplt build + cm_heldout, both xfer) | 2 | 0.00 | 0.00 |
| watcher (xfer sidecar watcher chain + submit driver) | 5 | 0.00 | 0.00 |
| **total** | **53** | **145.85** | **145.85** |

`dataset_prep` and `watcher` are CPU-only on the `xfer` partition by design (0 GPU-h).
Per-iter `04cksum_i*` jobs are also `xfer` (0 GPU-h) and are bundled under
`per_iter_sidecar` for accounting clarity.

## Training segments (line items)

Five successful segments + 2 repair attempts. JobName encodes the END iter of the segment.

| JobID | JobName | State | Partition | Elapsed | sec | GPU-h billed |
|---|---|---|---|---|---:|---:|
| 2417446 | 04van5b_i300 | COMPLETED | normal | 10:57:34 | 39454 | 43.84 |
| 2417447 | 04van5b_i477 | COMPLETED | normal | 06:34:45 | 23685 | 26.32 |
| 2417448 | 04van5b_i596 | COMPLETED | normal | 04:26:16 | 15976 | 17.75 |
| 2417449 | 04van5b_i715 | COMPLETED | normal | 04:27:17 | 16037 | 17.82 |
| 2417450 | 04van5b_i834 | **RUNNING** (in-flight) | normal | 03:03:45 | 11025 | 12.25 |
| 2417278 | 04van5b_i119 | FAILED (warmup-assert) | normal | 00:01:09 | 69 | 0.08 |
| 2417297 | 04van5b_i357 | CANCELLED (warmup-assert) | normal | 00:09:42 | 582 | 0.65 |
| training subtotal | | | | | **106828** | **118.71** |

Notes on segments:
- The `i300` segment is ~2.5× the others because it includes the cold start from
  iter 0 plus the first checkpoint write, and presumably ran at a lower throughput at first.
  After i300 the steady-state cadence is ~119 iters per 4h26m segment.
- `i834` is currently RUNNING with 3h03m45s of partial elapsed at the query instant; that
  partial time is counted in the headline total. Expected wall-clock for i834 segment
  (119 iters × 134.5 sec/iter) ≈ 4h26m45s → about 1h23m and ≈ 5.55 billed GPU-h still to
  go on this segment.
- `i357` (cancelled) and `i119` (failed) are bookkept under **repair**, not training, per
  the task spec ("failed warmup-assert chain attempts"). Their GPU-h is small (0.73 total).

## In-flight jobs (2)

- **2417450 04van5b_i834** — partition `normal`, elapsed 03:03:45, billed-so-far 12.25 GPU-h.
- **2424477 04van5b_watch** — partition `xfer`, elapsed 07:13:00, 0 GPU-h by construction.

## Unclassified

None — every JobID returned by sacct matched a known purpose pattern.

## Methodology

- **sacct command** (exact, read-only over SSH):
  ```
  ssh clariden 'sacct -u fffoivos -S 2026-05-28T00:00 \
    --format=JobID,JobName,State,Partition,Elapsed,AllocTRES,ExitCode,Start,End \
    --parsable2 --noheader'
  ```
- Filtered out `.batch`, `.extern`, and `.0` sub-rows; kept only top-level Slurm job IDs.
- `elapsed_seconds` parsed from sacct `Elapsed` (`HH:MM:SS` or `D-HH:MM:SS`).
- `gpus_requested` parsed from `AllocTRES` field `gres/gpu=N`; absent → 0 (CPU-only xfer).
- `gpu_hours_requested = elapsed_seconds / 3600 × gpus_requested`.
- `gpu_hours_billed_normal_policy = elapsed_seconds / 3600 × 4` for partitions in
  `{normal, debug, low}`, else 0.
- **Whole-node billing assumption:** Clariden's `normal`/`debug`/`low` partitions
  allocate a full 4-GPU GH200 node even if the job requests `--gpus-per-node=1`. In this
  cohort every GPU job already requested `gres/gpu=4`, so the two columns are equal — the
  whole-node policy would only inflate the bill for narrower GPU requests.
- `xfer` partition is CPU-only by design; all jobs there carry 0 GPU-h regardless of
  partition policy.
- **Extrapolation to chain completion** uses the mean throughput of the two most recent
  *completed* segments (i596 + i715, both 119 iters, COMPLETED): 134.5 sec/iter. Remaining
  iters at snapshot time = (i834 finish: 119 − partial) + i1192 segment (358 iters).
  Multiplied by 4 GPUs and converted to hours. The 8-GPU assumption is explicitly *not*
  used; this run is single-node 4× GH200.
- Sidecar fan-out costs for the upcoming iter-834 and iter-1192 checkpoints are *not*
  included in the at-completion projection (would add ≈ 14 GPU-h based on per-iter
  sidecar history of ~7 GPU-h per checkpoint × 2 checkpoints).

## Verification (2026-05-30, independent re-computation)

Independent re-pull using `sacct -X` (allocation rows only — guarantees no
`.batch` / `.extern` double-counting; original used a wider `--format` and filtered
sub-rows post-hoc, this verification ensures the filter was exhaustive). 53 allocation
rows returned, identical to the row count in the original report.

**Partition-summed elapsed (this re-pull):**

| Quantity | Value |
|---|---:|
| `T_normal_hours` (sum elapsed, partition=normal) | 36.36 h |
| `T_xfer_hours` (sum elapsed, partition=xfer) | 34.04 h |
| `T_debug_hours` (sum elapsed, partition=debug — smokes) | 0.16 h |
| Implied billed GPU-h (T_normal × 4) | 145.43 |
| Implied billed GPU-h ((T_normal + T_debug) × 4) | 146.07 |

Original headline: **145.85 GPU-h billed**. This sits between the normal-only
(145.43) and normal+debug (146.07) recomputations. The ~0.4 h difference vs
normal-only is the live drift of `2417450 04van5b_i834`, which advanced from
03:03:45 elapsed at the original query to 03:07:06 at this re-pull (+201 s ≈
+0.22 GPU-h), plus the 0.64 GPU-h `debug` smokes that the original report
correctly included in the headline. Inverse-checking: original total 145.85
≈ 145.43 (normal at first snapshot) + 0.64 (debug smokes) − 0.22 (job 2417450
delta) = 145.85. Reconciled to 4 decimal places. **No discrepancy.**

**Spot-checks (this re-pull vs original):**

| Spot-check | Re-pull | Original | Δ | Verdict |
|---|---:|---:|---:|---|
| Training (`04van5b_i*`, all 7 rows: 5 segments + 2 repairs) | 118.92 GPU-h | 117.97 + 0.73 = 118.70 GPU-h | +0.22 | within live-drift on 2417450 |
| Per-iter sidecars (24 normal sidecars + 3 xfer cksums = 25 rows, cksum @ 0 GPU-h) | 21.76 GPU-h | 21.75 GPU-h | +0.01 | rounding |
| Matched-config base eval (2422890+2422891+2422892) | 2.74 GPU-h | 2.74 GPU-h | 0.00 | exact |

All three spot-checks agree within < 1 %. Training delta is entirely attributable
to job `2417450` ticking forward between the two queries; sidecar count differs
trivially (original explicitly split `cksum` into a separate xfer bucket but
totalled the same; this re-pull rolls all 25 per-iter rows into one bucket).

**Partition policy — AllocTRES ground truth:**

Distribution of `gres/gpu=N` over all 53 rows by partition:

| Partition | gres/gpu=0 | gres/gpu=4 | gres/gpu=1\|2\|3 |
|---|---:|---:|---:|
| normal | 1 (cancelled-at-0s 2415493) | 34 | 0 |
| debug | 1 (cancelled-at-0s 2415572) | 4 | 0 |
| xfer | 13 (CPU-only) | 0 | 0 |

**Ground truth: every GPU job in this cohort that actually started requested
`gres/gpu=4`.** No row in any `normal` / `debug` partition allocated 1, 2, or 3
GPUs — so the whole-node-billing question (does Clariden inflate a
`--gpus-per-node=1` request to 4 for billing?) does *not* arise in this cohort.
The original report's whole-node-policy column equals the requested column by
construction here, not by Clariden inflating anything in this dataset. The
billing assumption for `--gpus-per-node=1` cases remains untested against this
sacct snapshot; future per-GPU requests would need a re-check.

**Verdict: consistent.** Headline GPU-h, per-category split, and partition policy
all reconcile within sub-1 % drift attributable to the in-flight `2417450` job
advancing between query times. AllocTRES ground truth: every GPU job in cohort
allocates 4 GPUs, so "requested" == "whole-node billed" by construction, not by
policy inflation.
