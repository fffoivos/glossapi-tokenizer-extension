# Firing-count tokenization — plan (v2.4, corpus = train.parquet + train_manifest.csv)

**Date**: 2026-05-18. **Status**: run completed; final artifacts are in `variants/c3_added_17408_curated_padded.firing_counts/`, with compact provenance tracked in `manifests/firing_count_20260518_run_summary_augmented.json` and summarized in `FIRING_COUNT_RUN_20260518.md`.

## v2.4 change — exact BPE training data via train.parquet + train_manifest.csv (per reviewer round 2)

Reviewer round 2 (2026-05-18) flagged six issues with v2.3. The big one
(finding #2): **mix.parquet is upstream of the train/val/test split**,
so it includes 0.1 % more data (val + test rows) than the C3 BPE actually
saw. The exact BPE training corpus is:

| file | role |
|---|---|
| `splits/.../exports/train.parquet` | the exact 14,401,554-row / ~100 B-char **text** the BPE saw |
| `manifests/train_manifest.csv` (or `.parquet` — verify on next apertus session) | per-row `source_dataset` (and other metadata) in matching deterministic order — written by `export_text_budgeted_splits.py:233` |

Worker becomes a **paired streaming read**: pyarrow yields a record
batch from train.parquet (text-only) while a parallel csv/pandas
iterator yields the matching N rows from train_manifest.csv. The
batches are zipped on row index to assign each text to its source.

**Open verification items** (must be confirmed on next apertus session
before this plan is safe to run):

- ☐ `manifests/train_manifest.csv` (or `.parquet`) exists in the C3
  50/50 run dir on apertus
- ☐ schema contains a `source_dataset` column
- ☐ row count equals `train.parquet` exactly (14,401,554)
- ☐ row order matches train.parquet (spot-check by joining 10 random
  rows and confirming text/length is consistent)

Fallback if the manifest is missing or non-aligned: use **mix.parquet**
(the v2.3 plan) with explicit acknowledgement that we're counting on a
0.1 % superset of the BPE training data.

### Other reviewer findings (round 2) addressed in v2.4

- **#1 wrong path in §3.4 preflight**: §3.4 now ships `train.parquet`
  + `train_manifest.csv` (not the upstream mix). Path is grounded in
  `splits/.../exports/` / `splits/.../manifests/`.
- **#3 source-component rule**: NOT "starts with a GlossAPI source name"
  (would miss `AI-team-UoA/greek_legal_code`, `HuggingFaceFW/finewiki`, …).
  Robust rule:
  ```
  component = 'hplt_only' iff source_dataset == 'HPLT/ell_Grek_ge8_no_mt_clean60'
              else 'glossapi_nanochat_only'
  ```
  Exact single-string match for HPLT; everything else is GlossAPI-nanochat.
- **#4 denominators**: `train.parquet` is **14,401,554 rows / ~100 B
  chars / N_tokenized_tokens** (computed by worker; not hard-coded).
  The plan body MUST NOT mix train and mix figures.
- **#5 sharding**: `gcloud storage cp` does NOT support row-group slice
  copies. Two options:
  - **A (preferred)**: pre-shard train.parquet on home into K physical
    parquet files using `pyarrow.parquet.write_to_dataset` or a
    row-group-based slice loop; ship K files to GCS; each worker
    downloads its single file (~5 GB for K=8).
  - **B (alternative)**: `pyarrow.fs.GcsFileSystem` + `pq.ParquetFile`
    supports HTTP range reads → workers can `iter_batches(row_groups=...)`
    on the GCS object directly. Lower storage cost; more fragile
    (range-read retries on transient HTTP).

  Recommended path is A.
- **#6 apertus state**: was SUSPENDED at preflight verify time; resume
  is now part of §3.4 with an explicit "suspend immediately after"
  step (cost-containment).
- **Naming rename**: all body refs `glossapi_only → glossapi_nanochat_only`
  for consistency.

### Also, IAM gap noted 2026-05-18

The apertus default compute service account
(`457132319455-compute@developer.gserviceaccount.com`) currently lacks
`storage.objects.get` on `gs://testbucketglossapi/`. Attempting
`gcloud storage cp $TRAIN gs://testbucketglossapi/c3_train_mix/...`
fails with HTTP 403 because the cp pre-flight `stat`-style check needs
GET. **Required preflight fix** (one-time admin op):

```bash
gcloud storage buckets add-iam-policy-binding gs://testbucketglossapi \
    --member=serviceAccount:457132319455-compute@developer.gserviceaccount.com \
    --role=roles/storage.objectAdmin
```

Or grant a narrower role + prefix policy. This is a §3.2 prerequisite.

---

## (historical) v2.3 attempt — mix.parquet

(Superseded by v2.4. The v2.3 mix.parquet path is kept as the fallback
if `train_manifest.csv` turns out to be missing or non-aligned.)

Inspected apertus directly (2026-05-18). **`splits/.../exports/train.parquet`
is text-only** (the export step dropped all metadata). The per-source
`source_dataset` column lives
one stage upstream in `mix.parquet`:

| file | size | columns | usable for per-source? |
|---|---:|---|---|
| `splits/.../exports/mix.parquet` | 42 GB | `text` only | **NO** |
| `splits/.../exports/val.parquet`, `test.parquet` | 22 MB each | `text` only | NO |
| `splits/.../exports/{val,test}_clean.parquet` | 35 MB each | `text` only | NO |
| **`mixes/.../mix.parquet`** | **44 GB** | `source_dataset`, `source_doc_id`, `text`, + ~30 metadata cols | **YES — canonical** |
| `mixes/.../mix.source_mix_summary.json` | 1 KB | per-source row + char counts already computed at mix time | YES — reference |

The mix→split step (`export_text_budgeted_splits.py`) deterministically
partitions mix rows into train/val/test by `stable_key`. mix.parquet is
the **superset** of mix.parquet by exactly 14k rows (val + test = 7,654 +
7,282) — 0.1% larger. Tokenizing mix.parquet gives essentially the same
distribution as tokenizing mix.parquet, with the bonus that val+test
rows ARE part of the C3 training-time char budget the model will eventually
see during CPT. So mix.parquet is at least as accurate, and gives us
per-source breakdown for free.

**The framing** (per user 2026-05-18): the C3 tokenizer was trained on
the 1:1 mix of `fffoivos/glossapi-greek-nanochat-pretraining-dataset` +
`HPLT/ell_Grek_ge8_no_mt_clean60`. The primary question is "GlossAPI-
nanochat alone vs +HPLT", and within GlossAPI-nanochat the per-sub-source
contribution matters for picking a minimum sufficient set.

mix.parquet's `source_dataset` column carries:
- ~19 sub-source values from inside the GlossAPI-nanochat HF dataset
  (e.g. `Apothetirio_Kallipos`, `AI-team-UoA/greek_legal_code`,
  `1000_prwta_xronia_ellhnikhs`, …)
- 1 value `HPLT/ell_Grek_ge8_no_mt_clean60` for the HPLT slice

Component rollups for the plan:
- `glossapi_nanochat_only` = sum over all `source_dataset` values that
  are NOT `HPLT/ell_Grek_ge8_no_mt_clean60` (the entire HF nanochat
  dataset post-cleaner + quality gates)
- `hplt_only` = sum where `source_dataset == HPLT/ell_Grek_ge8_no_mt_clean60`
- `glossapi_nanochat_plus_hplt` = mix.parquet total (derived sum of
  the two components)

(Renamed from v2.2's `glossapi_only` → `glossapi_nanochat_only` for clarity.
The semantics are unchanged: the union of the 19 GlossAPI-nanochat
sub-sources, which is exactly the HF nanochat dataset as filtered into
C3.)

**Preflight change**: §3.4's apertus ssh now ships **mix.parquet** (not
mix.parquet) to GCS at `gs://testbucketglossapi/c3_train_mix/mix.parquet`,
together with the sibling `mix.source_mix_summary.json` for sanity
cross-check.

**Throughput change**: 44 GB compressed parquet is ~10× larger than
v1's HF stream-tokenize attempt, but a single intra-project gsutil cp
runs at ~500 MB/s = ~90 sec for the full file. Workers then download
their row-group slice from GCS (also intra-project, fast).

## v2 changes — addressing reviewer findings

| # | finding | resolution |
|---|---|---|
| 1 | "C3 training mix" ≠ full HF source pools | **§2**: corpus is now explicitly `mix.parquet` (the actual C3 mix on apertus disk); alternatives documented and rejected. |
| 2 | Runnable scripts don't exist yet | **§9**: scripts marked as TODO with exact paths; plan is spec-only until they land. |
| 3 | GCP auth is failing | **§3**: explicit preflight checks (auth refresh, IAM, secret, apertus disk read) gate the run. |
| 4 | Worker silently swallows errors → false `_DONE` | **§5**: worker is fail-hard — any download/parse error raises `sys.exit(1)`; `_DONE` only on full success. Coordinator detects missing `_DONE` per shard and respawns. |
| 5 | `--max-rows` is per-file; smoke `_DONE` path mismatched | **§7**: smoke uses a component-scoped `$PREFIX/_smoke/` namespace, `--max-total-rows-per-component` flag, and component `_DONE` is the gate. |
| 6 | `--out-dir ../variants/...` escapes the subproject | **§6**: corrected to `variants/c3_added_17408_curated_padded.firing_counts/`. |
| 7 | HF token uploaded to GCS; SVC_ACCOUNT undefined | **§4**: HF token stays in Secret Manager; workers fetch via `gcloud secrets versions access`; SVC_ACCOUNT + IAM bindings are explicit preflight items. |
| 8 | Cost/time model optimistic + unvalidated | **§8**: cost/wall labelled "estimate pending smoke benchmark"; the smoke instance reports download MB/s, tokenize tok/s, and CPU% which calibrate the fleet sizing **before** each component fleet. |

## v2.1 change — component counts before mixture rates

Downstream may train/adapt on the minimal selected GlossAPI set only.
Therefore the run must keep **GlossAPI-selected** and **HPLT-selected**
as separate first-class firing-count vectors, then derive any combined
view from those two components. We do not only compute
`glossapi_only` + `glossapi_plus_hplt` and subtract HPLT as an
afterthought; explicit HPLT partials make rate denominators, shard
debugging, and mixture comparisons cleaner.

## v2.2 change — per-dataset contribution accounting

The run should also calculate each exact `source_dataset` value's
contribution. This is not a separate expensive tokenization pass: while
the worker is already tokenizing rows for `glossapi_only` or `hplt_only`,
it also accumulates per-`source_dataset` count vectors and per-source
denominators. That lets us answer both:

- Which tokens are supported by the minimal GlossAPI set as a whole?
- Within GlossAPI, which input datasets actually contribute those token
  firings?

## 1. Goal

Per-token-id firing-count vectors for the chosen
`c3_added_17408_curated_padded` tokenizer (vocab 148,480) on the
**actual C3 training mix** (not the full current HF source pools), in
two source components plus exact per-dataset contributions via the
`source_dataset` column in `train_manifest.parquet` joined to
`train.parquet` by row order:

1. **`glossapi_nanochat_only`** — rows where `source_dataset` is **NOT**
   the exact string `HPLT/ell_Grek_ge8_no_mt_clean60` (i.e. any of the
   ~19 sub-source values inside the `fffoivos/glossapi-greek-nanochat-pretraining-dataset`
   HF dataset: `Apothetirio_Kallipos`, `AI-team-UoA/greek_legal_code`,
   `HuggingFaceFW/finewiki`, `1000_prwta_xronia_ellhnikhs`, …).
2. **`hplt_only`** — rows where `source_dataset == HPLT/ell_Grek_ge8_no_mt_clean60`
   (exact string match).
3. **`glossapi_nanochat_plus_hplt`** — derived locally as
   `glossapi_nanochat_only + hplt_only`.

This supports both hypotheses:

- **minimal GlossAPI-nanochat-only training/adaptation**:
  use `glossapi_nanochat_only.fire_count` and `.fire_rate`.
- **original C3-style mixed training/adaptation**:
  use the derived `glossapi_nanochat_plus_hplt` counts/rates.

Rollup output schema (up to ×3 parquets):

```
id          int32
decoded     string         (best-effort string from tokenizer.decode([id]))
fire_count  int64
fire_rate   float64        (fire_count / total_tokenized_tokens for this corpus)
```

Each parquet's metadata, and `run_summary.json`, records
`total_rows`, `total_chars`, and `total_tokenized_tokens`. Those
denominators are required; rates cannot be interpreted from counts
alone. In particular, `hplt_only.fire_rate` must be computed from the
HPLT denominator, and `glossapi_nanochat_plus_hplt.fire_rate` must be
computed from the combined denominator; never subtract or average
component rates.

Per-dataset contribution outputs:

```
source_dataset_token_counts.parquet   (sparse long format)
source_dataset_summary.parquet
```

`source_dataset_token_counts.parquet` schema:

```
source_dataset                      string
source_group                        string    (glossapi_nanochat_only or hplt_only)
id                                  int32
decoded                             string
fire_count                          int64
fire_rate_within_source_dataset      float64   (count / source_dataset tokenized tokens)
share_of_component_token_firings     float64   (count / component count for this token)
```

Only non-zero `(source_dataset, id)` rows need to be stored; missing rows
mean zero firings. `source_dataset_summary.parquet` records rows, chars,
tokenized tokens, tokenized-token share, non-zero vocab size, non-zero
added-token count, and total added-token firings per source dataset.

## 2. Corpus decision (resolves finding #1; updated for v2.4)

| option | what it is | accuracy for "what the LM will see during CPT" | cost / complexity |
|---|---|---|---|
| **A** | `train.parquet` (~42 GB, text-only) **+** `manifests/train_manifest.parquet` (per-row `source_dataset`) joined by row order; both on apertus at `/home/foivos/runs/c3_wave2_broad_latest_cleaner_20260506/50_50/splits/glossapi_plus_hplt_50_50/` | exact: this IS the BPE training data (14,401,554 rows / ~100 B chars) | requires one-time apertus → GCS ship of two files |
| B | `mix.parquet` (44 GB, upstream of the train/val/test split) | 0.1% superset of A (includes val+test rows); marginal mismatch with what BPE saw | one-time apertus → GCS ship; fallback when train_manifest is missing/non-aligned |
| C | Current HF source pools full (the v1-plan behaviour) | wrong — counts include filtered-out + unsampled rows + post-C3 additions | low — public download |

**Decision: A.** Exact match to BPE training data. The export script at
`subprojects/_archive/01_2_training_dataset_mix/scripts/export_text_budgeted_splits.py:233`
writes `train.parquet` (text only) AND `train_manifest.csv/.parquet`
(source_dataset etc., row-aligned) in the same deterministic order.

Sizes per `docs/C3_TRAINING_DATASETS.md`:
- 14,401,554 rows
- ~100 B chars
- on-disk: `train.parquet` 42 GB, `train_manifest.parquet` TBD (a few hundred MB)

The worker does a **paired streaming read**: pyarrow yields a record
batch from `train.parquet` (text) while a parallel pyarrow csv/parquet
iterator yields the matching N source_dataset values from the manifest.
Batches are zipped on row index. Any row-count or order mismatch →
fail-hard (worker §5).

## 3. Preflight checks (manual gate, before any worker spawns)

```bash
# 3.1 — GCP auth refresh (reviewer finding #3)
gcloud auth login
gcloud config set project eellak-glossapi-20251008
gcloud compute instances list --limit 1 > /dev/null   # MUST succeed
gcloud storage ls gs://testbucketglossapi/ > /dev/null # MUST succeed

# 3.2 — IAM verification (reviewer finding #7)
#   Service account that workers use needs:
#     - roles/storage.objectAdmin  on the firing-count bucket prefix
#     - roles/compute.instanceAdmin.v1  on its own instance (self-delete)
#     - roles/secretmanager.secretAccessor  on the HF token secret
#   Either:
#     a) reuse the default compute service account with these bindings, OR
#     b) create a dedicated svc account (preferred for isolation)
SVC=firing-count-worker@eellak-glossapi-20251008.iam.gserviceaccount.com
gcloud iam service-accounts describe $SVC || echo "create me"

# 3.3 — HF token → Secret Manager (reviewer finding #7)
gcloud secrets describe hf-token-firing-count 2>/dev/null \
  || gcloud secrets create hf-token-firing-count \
       --replication-policy=automatic \
       --data-file=$HOME/.config/secrets/hf_token

# 3.4 — apertus read access + verify train.parquet + manifest + ship to GCS
TRAIN_DIR=/home/foivos/runs/c3_wave2_broad_latest_cleaner_20260506/50_50/splits/glossapi_plus_hplt_50_50
gcloud compute ssh apertus-greek-tokenizer-20260408t160000z \
    --zone europe-west4-b --command "
        set -e
        ls -lh $TRAIN_DIR/exports/train.parquet
        ls -lh $TRAIN_DIR/manifests/train_manifest.*  || echo 'MANIFEST NOT FOUND — fallback to mix.parquet'
        # Verify alignment: text row count must equal manifest row count
        python3 -c \"
import pyarrow.parquet as pq
t = pq.ParquetFile('$TRAIN_DIR/exports/train.parquet')
print('text rows:', t.metadata.num_rows)
# Try parquet manifest first, then csv
import os
for cand in ['$TRAIN_DIR/manifests/train_manifest.parquet',
             '$TRAIN_DIR/manifests/train_manifest.csv']:
    if os.path.exists(cand):
        if cand.endswith('.parquet'):
            m = pq.ParquetFile(cand)
            print('manifest rows:', m.metadata.num_rows, 'path:', cand)
            cols = m.schema_arrow.names
        else:
            import pyarrow.csv as pacsv
            m = pacsv.read_csv(cand)
            print('manifest rows:', m.num_rows, 'path:', cand)
            cols = m.column_names
        print('source_dataset in cols:', 'source_dataset' in cols)
        break
else: print('NO MANIFEST FOUND')
\"
        sha256sum $TRAIN_DIR/exports/train.parquet
    "

# Then ONE-TIME ship of train.parquet + manifest to GCS. Read-only on
# apertus; safe to run concurrently with any active workload there.
gcloud compute ssh apertus-greek-tokenizer-20260408t160000z \
    --zone europe-west4-b --command "
        gcloud storage cp $TRAIN_DIR/exports/train.parquet \
            gs://testbucketglossapi/c3_train_mix/train.parquet
        # Ship the manifest (whichever extension exists)
        if [ -f $TRAIN_DIR/manifests/train_manifest.parquet ]; then
            gcloud storage cp $TRAIN_DIR/manifests/train_manifest.parquet \
                gs://testbucketglossapi/c3_train_mix/train_manifest.parquet
        elif [ -f $TRAIN_DIR/manifests/train_manifest.csv ]; then
            # Convert csv → parquet on apertus before ship (smaller + faster
            # downstream); fall back to csv if conversion fails
            python3 -c \"
import pyarrow.csv as pacsv, pyarrow.parquet as pq
tbl = pacsv.read_csv('$TRAIN_DIR/manifests/train_manifest.csv')
pq.write_table(tbl, '/tmp/train_manifest.parquet', compression='zstd')
            \"
            gcloud storage cp /tmp/train_manifest.parquet \
                gs://testbucketglossapi/c3_train_mix/train_manifest.parquet
        else
            echo 'FATAL: no train_manifest found; fall back to mix.parquet path'
            exit 1
        fi
    "

# Verify outputs
gcloud storage ls -L gs://testbucketglossapi/c3_train_mix/train.parquet \
    | grep -E "size|md5"
gcloud storage ls -L gs://testbucketglossapi/c3_train_mix/train_manifest.parquet \
    | grep -E "size|md5"
```

If any preflight step fails, **STOP**. Fix and re-run preflight. Do not
proceed to §4.

**Output of preflight**: train.parquet + train_manifest.parquet on GCS;
sha256s recorded. The first stage of `scripts/run_firing_count.sh`
(coordinator) re-verifies these are present before any worker spawn.

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ home (coordinator) — `scripts/run_firing_count.sh`                   │
│                                                                      │
│  1. preflight  (§3 — auth, IAM, secret, train.parquet + manifest on  │
│                 GCS, stage tokenizer + worker.py to GCS run prefix)  │
│  2. shard       — scripts/build_shard_manifests.py: read             │
│                   train.parquet + train_manifest.parquet from GCS,   │
│                   greedy bin-pack row groups across K shards,        │
│                   write K paired (text, manifest) parquets to GCS    │
│  3. smoke       — ONE worker w/ --smoke --max-total-rows-per-component│
│                   N; poll _smoke/_DONE_shard_00; read benchmark;     │
│                   gate fleet on:                                     │
│                     - source_datasets_seen non-empty (FATAL)         │
│                     - both components in row_counts (FATAL by        │
│                       default; ALLOW_SMOKE_SINGLE_COMPONENT=1        │
│                       overrides for source-homogeneous shard 0)      │
│                     - mean_cpu_pct < 70 (warn only)                  │
│  4. fleet       — spawn K workers in parallel; poll K _DONE markers; │
│                   missing markers print respawn cmd + fatal exit     │
│  5. aggregate   — scripts/aggregate_firing_counts.py: download K     │
│                   partials, sum to full 148,480-id per-source        │
│                   vectors, derive components, write 6 output files   │
│                   to variants/c3_added_17408_curated_padded.firing_counts/│
│                                                                      │
│ Coordinator never SSHes into workers after spawn.                    │
└──────────────────────────────────────────────────────────────────────┘
                              │            ▲
                              │ create     │ partials + _DONE
                              ▼            │
┌──────────────────────────────────────────────────────────────────────┐
│ K × c4-highcpu-32 in europe-west4-a (worker instances)               │
│                                                                      │
│ scripts/worker_startup.sh (runs as root via instance metadata):      │
│   1. apt + pip install (tokenizers, pyarrow, numpy, psutil)          │
│   2. fetch HF token from Secret Manager (NOT GCS):                   │
│        gcloud secrets versions access latest                          │
│          --secret=hf-token-firing-count > /work/hf_token              │
│   3. fetch from GCS run prefix:                                      │
│        $PREFIX/tokenizer.tar.gz                                      │
│        $PREFIX/worker.py                                             │
│        $PREFIX/shards/shard_NN_of_KK_text.parquet                    │
│        $PREFIX/shards/shard_NN_of_KK_manifest.parquet                │
│   4. timeout 30m python worker.py (paired-stream read, fail-hard)    │
│      - pq.iter_batches on text.parquet + paired manifest iterator    │
│      - per (source, batch): group ids, bincount once (NOT per-doc)   │
│      - upload source_dataset_token_counts partial + denominators     │
│      - write _DONE only on full success                              │
│   5. on success: self-delete via                                     │
│        gcloud compute instances delete $(hostname) --zone $ZONE      │
│      on failure: leave alive for ssh-based forensics                 │
└──────────────────────────────────────────────────────────────────────┘
```

## 5. Worker error semantics (resolves finding #4)

**Hard rule**: a worker's `_DONE` marker is written **iff** every byte
of its assigned shard was successfully tokenized and counted.

Concrete checks inside the worker (any failure → `sys.exit(1)`):

| failure mode | detection |
|---|---|
| HF token unreadable | exit at startup |
| Tokenizer load failure | exit at startup |
| Manifest read failure | exit at startup |
| GCS download of train.parquet / train_manifest.parquet failed | exit |
| `pq.ParquetFile` open failed | exit |
| Row-group read raised | exit |
| `tokenizer.encode_batch` returned shorter list than input | exit |
| Output parquet write failed | exit |
| GCS upload of partial failed | exit |
| `_DONE` marker upload failed | exit (so coordinator sees it missing) |

The worker NEVER catches an exception and continues. The coordinator
treats "no `_DONE` after N minutes" as a respawn signal for that exact
shard ID (idempotent).

## 6. Output paths (resolves finding #6; updated v2.4)

All paths are relative to `subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/`.

```
variants/c3_added_17408_curated_padded.firing_counts/
  glossapi_nanochat_only.parquet         (148,480 rows: id, decoded, fire_count, fire_rate)
  hplt_only.parquet                       (148,480 rows: id, decoded, fire_count, fire_rate)
  glossapi_nanochat_plus_hplt.parquet    (derived sum of the two components)
  source_dataset_token_counts.parquet    (sparse long: source_dataset, source_group, id,
                                          decoded, fire_count, fire_rate_within_source_dataset,
                                          share_of_component_token_firings)
  source_dataset_summary.parquet          (per-source aggregate: rows, chars, tokenized_tokens,
                                          n_nonzero_added_ids, total_added_firings, ...)
  run_summary.json                        (per-component denominators + tail stats
                                          + per-source contribution summary)
```

Coordinator writes via `--out-dir variants/c3_added_17408_curated_padded.firing_counts/`.

## 7. Smoke test (resolves finding #5; updated v2.4)

**Hard gate.** Spawn ONE instance only on shard 0. Use a separate GCS
namespace (`$PREFIX/_smoke/`) so smoke artifacts don't collide with
the full-fleet outputs.

```bash
# Worker invocation (via worker_startup.sh on the smoke instance):
python worker.py --shard 0 --total $K \
    --text-parquet /work/shard_00_of_KK_text.parquet \
    --manifest     /work/shard_00_of_KK_manifest.parquet \
    --tokenizer-dir /work/c3_added_17408_curated_padded \
    --smoke \
    --max-total-rows-per-component 10000 \
    --gcs-out-prefix $PREFIX

# Worker writes (smoke namespace) to:
#   $PREFIX/_smoke/per_source_counts/shard_00_of_KK.parquet
#   $PREFIX/_smoke/per_source_denominators/shard_00_of_KK.json
#   $PREFIX/_smoke/smoke_benchmark.json
#   $PREFIX/_smoke/_DONE_shard_00_of_KK
```

`--max-total-rows-per-component 10000` is a GLOBAL cap (not per-file):
the worker stops processing rows once each component has reached
10,000 rows. The benchmark JSON the worker emits:

```json
{
  "wall_seconds": 84.3,
  "n_rows_processed": 14213,
  "total_tokens": 1845231,
  "tokenize_tokens_per_sec": 21887,
  "peak_rss_mb": 6210,
  "mean_cpu_pct": 87.4,
  "n_sources": 12,
  "vocab_size": 148480,
  "source_datasets_seen": ["Apothetirio_Kallipos", "greek_phd",
                            "HPLT/ell_Grek_ge8_no_mt_clean60", ...],
  "source_dataset_rows": {"Apothetirio_Kallipos": 89, ...},
  "component_row_counts": {"glossapi_nanochat_only": 10000, "hplt_only": 10000}
}
```

The coordinator gates the full fleet on:

- **FATAL** if `source_datasets_seen` is empty → accounting broken
- **FATAL** if `component_row_counts` is missing either component →
  shard 0 may be source-homogeneous (train.parquet is sorted by
  stable_key; some shards may be dominated by one component). Override
  with `ALLOW_SMOKE_SINGLE_COMPONENT=1` if you've verified shard balance.
- **WARN** if `mean_cpu_pct < 70` → consider larger instance or
  `--batch-size` tweak
- **WARN** if `tokenize_tokens_per_sec × K × expected_wall < total_tokens`
  → bump K or MACHINE before fleet spawn

**Only after the smoke benchmark is recorded + gates pass** does the
coordinator spawn K workers.

## 8. Sizing & cost (resolves finding #8)

**All numbers here are estimates pending the smoke benchmark.**

Inputs (to be confirmed by §3 preflight):
- `train.parquet` ~42 GB compressed (text-only, 14.4 M rows)
- `train_manifest.parquet` ~150 MB (per-row `source_dataset` column;
  exact size depends on whether nanochat builder wrote csv or parquet)
- ~100 B chars
- ~30-50 B tokens (at typical Greek 2-3 chars/token)

Sizing scenarios (all in europe-west4-a, c4-highcpu-32):

| K | per-worker rows | est. wall | est. cost | comment |
|---|---|---|---|---|
| 4 | 3.6 M | ~25 min | ~$1.40 | minimum viable |
| **8** (recommended) | **1.8 M** | **~12-15 min** | **~$1.50** | sweet spot |
| 16 | 0.9 M | ~10 min | ~$2.10 | startup overhead dominates |

These numbers will be VERIFIED via smoke benchmark before fleet spawn.
If smoke shows the worker is much slower than estimated (e.g.
tok/s × 8 × 15 min < total tokens), we bump K or switch to
c4-highcpu-96 BEFORE spawning the fleet.

Note: in v2.4 the worker counts both components in a single pass (rows
are interleaved in `train.parquet`), so there is no separate "Stage B"
smoke for HPLT. However, train.parquet may be sorted by `stable_key`
which can make shard 0 source-homogeneous — if smoke saw only one
component, sanity-check shard balance under `$PREFIX/shards/` before
overriding with `ALLOW_SMOKE_SINGLE_COMPONENT=1`.

## 9. Scripts (now implemented + unit-tested — 2026-05-18)

All scripts now exist under `scripts/`. Test commands and the cloud-run
runbook live in [`FIRING_COUNT_README.md`](FIRING_COUNT_README.md).

```
scripts/
  build_shard_manifests.py        ← pre-shard train.parquet + manifest into K
                                     paired physical files (reviewer #5 fix)
  firing_count_worker.py          ← paired-stream worker; bincount batched per
                                     (source, batch) for ~200× perf (reviewer #2);
                                     fail-hard semantics; vocab-id overflow check
  aggregate_firing_counts.py      ← downloads K shard partials, sums with
                                     full-vocab vectors (reviewer #1 fix —
                                     allocates from tokenizer, not max(observed_id));
                                     writes 6 output files including
                                     source_dataset_token_counts.parquet and
                                     source_dataset_summary.parquet
  worker_startup.sh               ← instance metadata startup script; fetches HF
                                     token from Secret Manager (not GCS),
                                     downloads inputs, runs worker, self-deletes
                                     on success; 30-min timeout (reviewer #7 +
                                     §15 cost containment)
  run_firing_count.sh             ← coordinator: 5 stages (preflight → shard
                                     → smoke → fleet → aggregate); per-stage
                                     skip flags; uses smoke benchmark to gate
                                     fleet sizing
  respawn_shard.sh                ← respawn ONE shard after a missing _DONE
  test_firing_count_worker.py     ← 4 tests; happy path + 3 fail-hard cases +
                                     smoke validates source-accounting fields
  test_aggregate_firing_counts.py ← 2 tests; end-to-end through worker × K + agg;
                                     verifies share-sum, component invariants,
                                     full-vocab coverage
  test_build_shard_manifests.py   ← 1 test; multi-row-group synthetic input
                                     sharded → workers × K → aggregator vs an
                                     unsharded baseline; component count vectors
                                     must be bit-identical (reviewer round 4 #4)
```

**Test status (all 7 pass on home, in TokEval venv):**

```
$VENV/python scripts/test_firing_count_worker.py      → 4/4 PASS
$VENV/python scripts/test_aggregate_firing_counts.py  → 2/2 PASS
$VENV/python scripts/test_build_shard_manifests.py    → 1/1 PASS
```

See `FIRING_COUNT_README.md` § Test environment for which Python venv
to use (the existing `vendor/tokenizer-intrinsic-evals/.venv/`).

Invariants validated by tests:

- `sum_over_sources(per_source_counts[s, i]) == combined_counts[i]` for every id
- `glossapi_nanochat_only[i] + hplt_only[i] == combined[i]` for every id
- Per-source share-sum: `sum(share_of_component_token_firings)` over rows of
  a given (component, id) equals 1.0 within float tolerance
- Component parquets cover full 148,480 vocab (zero-firing tail tokens visible)
- Misaligned manifest / missing source_dataset column / missing _DONE → fail-hard

## 10. Failure recovery

In v2.4 the worker counts both components in one pass, so there is one
`_DONE` marker per shard (not per component-per-shard). The marker is
written only after the per-source counts partial AND the per-source
denominators partial have both been uploaded:

```
$PREFIX/_DONE_shard_NN_of_KK
```

Coordinator polls. After a timeout (e.g. 25 min), any missing marker
identifies a failed shard. Respawn that shard ID with the same
shard pair (`scripts/respawn_shard.sh SHARD=NN K=KK TIMESTAMP=TS`);
the worker is idempotent — if the partial already exists in GCS, it's
overwritten safely; if it doesn't, the worker computes it from scratch.

For systemic failures (multiple shards missing), inspect the worker
log on a *living* respawned instance:

```bash
gcloud compute ssh firing-w-NN-TS --zone $ZONE \
    --command 'cat /var/log/firing_worker.log'
```

(Workers self-delete on SUCCESS only; on failure they leave their log
and exit non-zero so the coordinator can SSH in for forensics.)

## 11. Compute saturation strategies

Same as v1 plan but now justified by the smoke benchmark:

| stage | bottleneck | mitigation | smoke metric |
|---|---|---|---|
| Download | network | `gcloud storage cp` of this worker's pre-sharded `$PREFIX/shards/shard_NN_of_KK_{text,manifest}.parquet` pair (sharder uses greedy bin-pack of row groups) | `download_mbps` |
| Read parquet | disk + arrow | `pq.iter_batches(batch_size=8192, columns=['text','source_dataset'])` | implicit |
| Tokenize | CPU | `tokenizer.encode_batch` (Rust rayon) | `tokenize_tokens_per_sec`, `mean_cpu_pct` |
| Aggregate | numpy | `np.bincount(ids, minlength=V)` per batch | (negligible) |

If smoke `mean_cpu_pct < 70 %`, bump `--download-inflight` and re-smoke
before fleet.

## 12. Outputs landing on home

After aggregator runs:

```
subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/
  variants/c3_added_17408_curated_padded.firing_counts/
    glossapi_nanochat_only.parquet
    hplt_only.parquet                          (written iff any HPLT row seen)
    glossapi_nanochat_plus_hplt.parquet        (derived sum of the two)
    source_dataset_token_counts.parquet        (sparse long format)
    source_dataset_summary.parquet
    run_summary.json     {wall_clock, cost_usd, K, machine, smoke_benchmark,
                          component_statuses,
                          source_dataset_statuses,
                          per_shard_wall, total_tokens_per_component,
                          total_chars_per_component,
                          total_tokens_per_source_dataset,
                          total_chars_per_source_dataset,
                          tokens_with_zero_firings_per_component,
                          tokens_with_lt_100_firings_per_component,
                          tokens_with_lt_1k_firings_per_component,
                          top_source_datasets_by_added_token_firings,
                          percentiles_p10_p25_p50_p75_p90_p99_per_component,
                          mixture_formula}
```

`run_summary.json` is the canonical output that gets cited from
`CHOSEN_CUTOFF.md` (informing the embedding-signal section).

## 13. Reproduction

Scripts now exist + are tested. The end-to-end one-button command lives
in [`FIRING_COUNT_README.md`](FIRING_COUNT_README.md); summary:

```bash
cd subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep
# 1. Verify environment (one-time)
$VENV/python scripts/test_firing_count_worker.py       # 4/4 PASS expected
$VENV/python scripts/test_aggregate_firing_counts.py   # 2/2 PASS expected
$VENV/python scripts/test_build_shard_manifests.py     # 1/1 PASS expected
# 2. Run (after preflight: train.parquet + manifest on GCS, IAM, secret)
bash scripts/run_firing_count.sh                       # K=8 default
```

Knobs (env vars): `K`, `ZONE`, `MACHINE`, `BUCKET`, `HF_SECRET`,
`TIMESTAMP`, `SMOKE_MAX_ROWS`, and per-stage skip flags
`DO_PREFLIGHT`/`DO_SHARD`/`DO_SMOKE`/`DO_FLEET`/`DO_AGGREGATE`. Full
documentation in `FIRING_COUNT_README.md`.

## 14. Hard gates before this is safe to execute

In order:

1. ✅ **Decision on corpus**: option A (train.parquet + train_manifest)
   confirmed. mix.parquet kept as fallback if manifest is missing.
2. ✅ **Scripts implemented + unit-tested** with synthetic
   parquet (no instance spawn needed), including the per-source
   contribution identity check.
3. ☐ **§3 preflight passes**: auth refresh, IAM bindings, Secret
   Manager secret, `train.parquet` + `train_manifest.parquet` shipped
   to GCS, sha256 of both recorded, row counts confirmed aligned.
4. ☐ **GlossAPI smoke instance succeeds** and the benchmark JSON looks
   reasonable (mean_cpu_pct ≥ 70 %, tokenize tok/s within 2× of
   expectation, no errors).
5. ☐ **Sizing calibrated** from smoke benchmark (K and MACHINE may
   change; `mean_cpu_pct ≥ 70`, `tokens_per_sec × K × wall ≥ total`).
6. ☐ **Both components seen in smoke** (or `ALLOW_SMOKE_SINGLE_COMPONENT=1`
   override if shard 0 is source-homogeneous).
7. ☐ **Fleet spawns**, all K partials land, aggregator writes the 6
   output files including `glossapi_nanochat_only.parquet`,
   `hplt_only.parquet`, `glossapi_nanochat_plus_hplt.parquet`,
   `source_dataset_token_counts.parquet`, `source_dataset_summary.parquet`.

Only proceed if ALL gates pass. Stop at the first failure.

(There is no longer a "Stage A: GlossAPI first, Stage B: HPLT later"
two-pass execution. The worker counts BOTH components in one pass
because both rows are interleaved in train.parquet; the
glossapi_nanochat_only-vs-combined comparison is done at analysis time
on the same output. This is cheaper and simpler than the v2.1 two-stage
plan.)

## 15. Cost containment

- Hard cap on instance lifetime via startup script: `timeout 30m` on
  the worker entry point. If the worker doesn't finish in 30 min, the
  startup script kills it AND triggers self-delete. Prevents runaway
  instances burning $.
- Idempotent design: a failed-shard respawn is a known operation, not
  a "throw more instances at it and hope" scramble.
- Coordinator MUST verify all `_DONE` markers landed before exiting;
  if any are missing, it prints a respawn command and exits non-zero
  so the human is in the loop for the failure-recovery decision (no
  auto-respawn without supervision).

---

**TL;DR (v2.4)**: target is `train.parquet` + `train_manifest.parquet`
on apertus (exact C3 BPE training data — 14.4 M rows, ~100 B chars),
shipped to GCS once via a read-only apertus ssh. Workers are
self-destructing startup-script-driven c4-highcpu-32s in europe-west4-a
running `firing_count_worker.py` as a paired-stream read; in ONE pass
they count both components AND every individual `source_dataset` value,
writing sparse long-format per-source counts + per-component rollups.
HF token in Secret Manager; fail-hard error semantics; smoke gates the
fleet on source-accounting + both-components-seen; cost/wall pending
smoke benchmark. **All 7 unit tests pass on home; orchestration shell
scripts written + syntax-checked. Ready for a guarded smoke run on
gcloud once preflight (train.parquet ship + manifest verification on
apertus) completes.** See `FIRING_COUNT_README.md` for the runbook.
