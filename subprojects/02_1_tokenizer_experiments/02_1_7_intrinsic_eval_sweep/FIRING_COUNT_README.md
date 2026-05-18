# Firing-count tokenization — runbook

Companion to `FIRING_COUNT_PLAN.md`. The plan documents *why*; this
README documents *how to run + how to verify locally*.

## Test environment

Unit tests depend on `tokenizers`, `pyarrow`, `numpy`. Use the TokEval
venv that already exists in the parent sub-subproject:

```bash
cd subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep
export VENV=$PWD/vendor/tokenizer-intrinsic-evals/.venv/bin
```

If the venv doesn't exist, build it:

```bash
cd vendor/tokenizer-intrinsic-evals
python3 -m venv .venv
.venv/bin/pip install -e .   # installs the suite + tokenizers/pyarrow/numpy
```

## Unit tests (no cloud, synthetic data)

```bash
$VENV/python scripts/test_firing_count_worker.py      # 4 tests
$VENV/python scripts/test_aggregate_firing_counts.py  # 2 tests
$VENV/python scripts/test_build_shard_manifests.py    # 1 test
```

The three suites build synthetic train.parquet + manifest.parquet pairs,
monkey-patch `gcloud` to a local-copy stub, run the real worker /
aggregator / sharder end-to-end, and verify:

- Output schemas (column names, file presence)
- Share-sum invariant (`sum_over_sources(count[s, i]) == combined[i]` for every id)
- Component invariant (`glossapi_nanochat_only + hplt_only == combined`)
- Full vocab coverage (148,480 ids, including zero-firing tail)
- Fail-hard on row-misaligned manifest
- Fail-hard on missing `source_dataset` column
- Fail-hard on missing `_DONE` marker
- Smoke benchmark records `source_datasets_seen`, `source_dataset_rows`,
  `component_row_counts` (gates source accounting before fleet)
- Sharder produces bit-identical component count vectors vs an
  unsharded baseline on a multi-row-group synthetic input

Hard expectation: all 7 tests pass before any cloud spawn.

## Cloud run

Top-level command (assumes preflight from FIRING_COUNT_PLAN.md §3 is
done — train.parquet + train_manifest.parquet on GCS, HF token in
Secret Manager, IAM granted):

```bash
bash scripts/run_firing_count.sh
```

The coordinator runs 5 stages:

1. **Preflight** — auth, bucket access, secret existence, train.parquet
   + manifest on GCS, stage tokenizer + worker.py to the run prefix.
2. **Sharder** — `build_shard_manifests.py` splits train.parquet +
   train_manifest.parquet into K paired files in
   `$GCS_RUN_PREFIX/shards/`.
3. **Smoke** — spawns ONE instance with `--smoke
   --max-total-rows-per-component 10000`. The worker terminates the
   smoke pass once **either** (a) every component has reached the
   per-component cap, **or** (b) the per-shard hard ceiling
   (`4 × per_component_cap × n_components` = 80,000 rows by default)
   is hit. (b) bounds smoke cost when shard 0 is source-homogeneous —
   in that case (a) would otherwise never trigger and smoke would
   process the entire shard. Polls for `_smoke/_DONE_shard_00_of_KK`.
   Reads the smoke benchmark and gates the fleet on:
   - non-empty `source_datasets_seen` (**FATAL** — accounting broken)
   - both components in `component_row_counts` (**FATAL** unless overridden — see below)
   - mean_cpu_pct ≥ 70 (warn only — instance/parallelism tunable)

   **Possible false-fail on the "both components" gate:** train.parquet
   may be sorted by stable_key, so shard 0 can be source-homogeneous —
   meaning a 10k-row smoke might see only one component even though the
   fleet sees both. If you've verified the shard balance is OK
   (`gcloud storage ls $GCS_RUN_PREFIX/shards/` + sanity-check shard
   sizes), override with:
   ```bash
   ALLOW_SMOKE_SINGLE_COMPONENT=1 bash scripts/run_firing_count.sh
   ```
4. **Fleet** — spawns K workers in parallel via startup script. Polls
   for K `_DONE_shard_NN` markers (45-min timeout). Any missing marker
   triggers a respawn message (does NOT auto-respawn — human in the loop).
5. **Aggregate** — `aggregate_firing_counts.py` downloads the K
   partials, sums per-source counts to full 148,480-id vectors, runs
   the share-sum invariant check, writes 5 final parquets plus
   `run_summary.json` to
   `variants/c3_added_17408_curated_padded.firing_counts/`.

The completed 2026-05-18 run is summarized in
[`FIRING_COUNT_RUN_20260518.md`](FIRING_COUNT_RUN_20260518.md). Its
compact machine-readable provenance is tracked at
[`manifests/firing_count_20260518_run_summary_augmented.json`](manifests/firing_count_20260518_run_summary_augmented.json).

### Knobs (env vars)

| var | default | meaning |
|---|---|---|
| `K` | 8 | number of worker instances |
| `ZONE` | europe-west4-a | GCE zone |
| `MACHINE` | c4-highcpu-32 | instance type |
| `BUCKET` | gs://testbucketglossapi | GCS bucket prefix |
| `HF_SECRET` | hf-token-firing-count | Secret Manager secret name |
| `TIMESTAMP` | now() | run id; if rerunning a partial run, reuse the previous TS |
| `SMOKE_MAX_ROWS` | 10000 | smoke per-component row cap |
| `DO_PREFLIGHT` | 1 | skip with =0 if preflight already done |
| `DO_SHARD` | 1 | skip with =0 if shards already on GCS |
| `DO_SMOKE` | 1 | skip with =0 to go straight to fleet |
| `DO_FLEET` | 1 | skip with =0 if all _DONE markers already present |
| `DO_AGGREGATE` | 1 | skip with =0 to defer aggregation |

### Recovery

If a shard fails:

```bash
SHARD=3 K=8 TIMESTAMP=20260518t012345 \
    bash scripts/respawn_shard.sh
```

Once the missing _DONE is present, re-run the coordinator with
`DO_PREFLIGHT=0 DO_SHARD=0 DO_SMOKE=0 DO_FLEET=0 DO_AGGREGATE=1`.

## Outputs

Local (in this sub-subproject):

```
variants/c3_added_17408_curated_padded.firing_counts/
  source_dataset_token_counts.parquet   long-format sparse: source_dataset,
                                          source_group, id, decoded, fire_count,
                                          fire_rate_within_source_dataset,
                                          share_of_component_token_firings
  source_dataset_summary.parquet         one row per source_dataset:
                                          rows, chars, tokenized_tokens,
                                          n_nonzero_added_ids,
                                          total_added_firings,
                                          added_firing_share_of_total
  glossapi_nanochat_only.parquet         component: 148,480 rows, id, decoded,
                                          fire_count, fire_rate
  hplt_only.parquet                       component (if HPLT was counted)
  glossapi_nanochat_plus_hplt.parquet    derived sum of the two components
  run_summary.json                       full run metadata + per-component
                                          tail stats (n_zero, n_lt_100, ...,
                                          percentiles p10..p99)
  run_summary_augmented.json             run_summary plus input GCS hashes,
                                          smoke benchmark, shard summary,
                                          fleet config, and cleanup status
  provenance/                            small run provenance bundle
```

GCS:

```
$BUCKET/firing_counts_$TIMESTAMP/
  tokenizer.tar.gz
  worker.py
  shards/shard_NN_of_KK_{text,manifest}.parquet
  shards/shards_index.json
  per_source_counts/shard_NN_of_KK.parquet
  per_source_denominators/shard_NN_of_KK.json
  _DONE_shard_NN_of_KK                   (markers, gating)
  _smoke/...                             (smoke artifacts if applicable)
```

## Cost containment

- Worker startup script enforces `timeout 30m` on the Python entry point —
  stuck worker self-kills before it can run away with the bill.
- Workers attempt to self-delete via `gcloud compute instances delete
  $(hostname) ...` on success (RC=0); however, the default compute SA
  in `eellak-glossapi-20251008` lacks `compute.instances.delete` so
  this fails silently. **Always verify VMs are cleaned up post-run**
  with `gcloud compute instances list --filter="labels.run=$TS"` and
  batch-delete any survivors.
- Coordinator does NOT auto-respawn failed shards; it prints the exact
  respawn command and exits non-zero. Human is in the loop.

## Pre-run checklist (do these manually before `bash run_firing_count.sh`)

- [ ] `gcloud auth login` — refresh creds if expired
- [ ] IAM grant: compute SA needs `objectAdmin` on the bucket (one-time
      `gcloud storage buckets add-iam-policy-binding`)
- [ ] One-time ship from apertus (resume → set `--scopes=cloud-platform`
      → ssh → `gcloud storage cp train.parquet + train_manifest.parquet`
      → stop)
- [ ] Verify `manifests/train_manifest.csv` exists on apertus, convert
      to parquet, confirm aligned row count + `source_dataset` column
- [ ] `$VENV/python scripts/test_firing_count_worker.py` → 4/4 pass
- [ ] `$VENV/python scripts/test_aggregate_firing_counts.py` → 2/2 pass
- [ ] `$VENV/python scripts/test_build_shard_manifests.py` → 1/1 pass

Then `bash scripts/run_firing_count.sh` is one-button. (Note: stage 2
sharder reads/writes 42 GB via `pafs.GcsFileSystem` and runs locally by
default; for the actual completed run we spawned a transient
`c4-highcpu-32` in europe-west4-a to keep the I/O intra-zone — see
"Completed run" below for the recipe.)

## Completed run — 2026-05-18 (`run_id=20260518t044858`)

End-to-end firing-count tokenization shipped. Headline numbers:

| metric | glossapi_nanochat_only | hplt_only | combined |
|---|---:|---:|---:|
| rows | 517,791 | 13,883,763 | 14,401,554 |
| chars (B) | 47.00 | 52.26 | 99.26 |
| tokenized tokens (B) | 12.39 | 12.50 | 24.89 |
| added tokens with 0 firings (of 17,408) | 0 | 27 | 0 |
| added tokens with <100 firings | 2 | 43 | 1 |
| added tokens with <1k firings | 40 | 80 | 9 |
| added tokens with ≥10k firings | 17,015 (97.7%) | 17,131 (98.4%) | 17,316 (99.5%) |
| total added-token firings (B) | 3.36 | 4.74 | 8.10 |
| added-token share of total | 27.1% | 37.9% | 32.5% |

**50/50 design realized**: GA-nanochat is 3.6% of rows but **49.8% of
tokenized tokens** (char-budgeted). HPLT is 96.4% of rows but only
50.2% of tokens. Within GA, openarchives.gr + greek_phd account for
~87% of GA token mass.

**Validated invariants (post-aggregate):**
- All 3 component parquets have exactly 148,480 rows, ids 0..148479.
- `glossapi_nanochat_only + hplt_only == glossapi_nanochat_plus_hplt`
  exactly (every id).
- `sum_over_sources(per_source_counts) == combined` exactly.
- `fire_rate = fire_count / component_denominator` exactly.
- `source_dataset_summary.parquet`, `source_dataset_token_counts.parquet`,
  and `run_summary.json` agree on all 18 sources.

**Outputs:**
- Local: `variants/c3_added_17408_curated_padded.firing_counts/`
  - `glossapi_nanochat_only.parquet` (2.7 MB)
  - `hplt_only.parquet` (2.6 MB)
  - `glossapi_nanochat_plus_hplt.parquet` (2.8 MB)
  - `source_dataset_token_counts.parquet` (20.8 MB; 1,069,908 non-zero
    `(source, id)` rows)
  - `source_dataset_summary.parquet` (4.4 KB; 18 source rows)
  - `run_summary.json` (the original aggregator output) and
    `run_summary_augmented.json` (with smoke benchmark, GCS hashes,
    fleet config, mixture formula realized, per-component top sources,
    cleanup status)
  - `provenance/`: `shards_index.json`, `smoke_benchmark.json`,
    `worker.py.snapshot`, `input_hashes.txt`, `run_metadata.txt`
- GCS (transient intermediates, OK to delete after this README is
  saved): `gs://testbucketglossapi/firing_counts_20260518t044858/` —
  48.7 GB of pre-sharded inputs + per-shard partials. Final outputs
  do not depend on this path; it exists only for shard respawn.
- GCS (kept): `gs://testbucketglossapi/c3_train_mix/{train,train_manifest}.parquet`
  (45.2 GB). Kept so a re-run with a different tokenizer doesn't need
  another apertus resume + 42 GB ship.

**Recipe (for future re-runs of this exact pipeline):**

1. `apertus-greek-tokenizer-20260408t160000z` resume → `set-service-account
   --scopes=cloud-platform` → ssh → convert `train_manifest.csv` → `.parquet`
   → `gcloud storage cp` both to `gs://testbucketglossapi/c3_train_mix/`
   → stop apertus. **~10 min wall + ~$1 apertus.**
2. Spawn transient `c4-highcpu-32` in europe-west4-a (cloud-platform
   scope), scp `scripts/build_shard_manifests.py`, run with
   `--text-parquet gs://.../train.parquet --manifest gs://.../train_manifest.parquet`,
   delete the VM. **~35 min wall + ~$0.30 sharder.**
3. `TIMESTAMP=<from step 2> DO_SHARD=0 bash scripts/run_firing_count.sh`
   — runs preflight + smoke + fleet (8× c4-highcpu-32) + aggregator.
   **~25 min fleet wall + ~$0.50 fleet.**
4. Verify cleanup: `gcloud compute instances list --filter="labels.run=$TIMESTAMP"`
   → batch-delete any survivors (self-delete fails due to missing IAM).
5. Optional: delete the run prefix on GCS once `run_summary_augmented.json`
   is committed:
   `gcloud storage rm -r gs://testbucketglossapi/firing_counts_$TIMESTAMP/`.
