#!/usr/bin/env bash
# Coordinator: preflight → smoke → fleet → aggregate.
# Per FIRING_COUNT_PLAN.md v2.4 §13.
#
# Knobs (env vars):
#   K=8                                  number of workers
#   ZONE=europe-west4-a
#   MACHINE=c4-highcpu-32
#   BUCKET=gs://testbucketglossapi
#   TIMESTAMP=$(date -u +%Y%m%dt%H%M%S)
#   SMOKE_MAX_ROWS=10000
#
# Sub-stage gates (env vars, all default true):
#   DO_PREFLIGHT=1  DO_SHARD=1  DO_SMOKE=1  DO_FLEET=1  DO_AGGREGATE=1
#
# Inputs assumed already on GCS (from a prior one-time ship from apertus):
#   gs://.../c3_train_mix/train.parquet
#   gs://.../c3_train_mix/train_manifest.parquet
#   gs://.../c3_train_mix/train.parquet.sha256  (optional, recorded only)
#
# Outputs land at:
#   gs://.../firing_counts_TIMESTAMP/{shards,per_source_counts,...}
#   variants/c3_added_17408_curated_padded.firing_counts/{*.parquet,run_summary.json}

set -euo pipefail

# --- knobs ------------------------------------------------------------------

K=${K:-8}
ZONE=${ZONE:-europe-west4-a}
MACHINE=${MACHINE:-c4-highcpu-32}
BUCKET=${BUCKET:-gs://testbucketglossapi}
TIMESTAMP=${TIMESTAMP:-$(date -u +%Y%m%dt%H%M%S)}
SMOKE_MAX_ROWS=${SMOKE_MAX_ROWS:-10000}
DO_PREFLIGHT=${DO_PREFLIGHT:-1}
DO_SHARD=${DO_SHARD:-1}
DO_SMOKE=${DO_SMOKE:-1}
DO_FLEET=${DO_FLEET:-1}
DO_AGGREGATE=${DO_AGGREGATE:-1}

SSP=$(cd "$(dirname "$0")/.." && pwd)
SCRIPTS="$SSP/scripts"
VENV="$SSP/vendor/tokenizer-intrinsic-evals/.venv/bin"
GCS_RUN_PREFIX="$BUCKET/firing_counts_$TIMESTAMP"
TOKENIZER_LOCAL="$SSP/variants/c3_added_17408_curated_padded"
OUT_DIR="$SSP/variants/c3_added_17408_curated_padded.firing_counts"

echo "==================================================================="
echo "FIRING-COUNT RUN"
echo "  K               : $K"
echo "  ZONE            : $ZONE"
echo "  MACHINE         : $MACHINE"
echo "  GCS_RUN_PREFIX  : $GCS_RUN_PREFIX"
echo "  TOKENIZER_LOCAL : $TOKENIZER_LOCAL"
echo "  OUT_DIR         : $OUT_DIR"
echo "==================================================================="

# --- spawn_worker helper (used by smoke + fleet + respawn) ------------------

spawn_worker() {
    # Args: shard_idx is_smoke instance_name
    local SHARD="$1" IS_SMOKE="$2" NAME="$3"
    local METADATA="shard=$SHARD,total=$K,gcs-prefix=$GCS_RUN_PREFIX"
    if [ "$IS_SMOKE" == "true" ]; then
        METADATA="$METADATA,smoke=true,smoke-max-rows=$SMOKE_MAX_ROWS"
    fi
    gcloud compute instances create "$NAME" \
        --zone "$ZONE" \
        --machine-type "$MACHINE" \
        --boot-disk-size 200GB \
        --boot-disk-type hyperdisk-balanced \
        --image-family debian-12 --image-project debian-cloud \
        --scopes cloud-platform \
        --metadata "$METADATA" \
        --metadata-from-file "startup-script=$SCRIPTS/worker_startup.sh" \
        --labels "owner=foivos,purpose=firing-count,run=$TIMESTAMP,shard=$SHARD" \
        --quiet >/dev/null
    echo "  spawned $NAME (shard $SHARD, smoke=$IS_SMOKE)"
}

# --- poll_done_marker helper -----------------------------------------------

poll_done_marker() {
    # Args: marker_uri timeout_seconds
    local MARKER="$1" TIMEOUT="$2"
    local START=$(date +%s)
    while true; do
        if gcloud storage ls "$MARKER" >/dev/null 2>&1; then
            return 0
        fi
        local NOW=$(date +%s)
        local ELAPSED=$((NOW - START))
        if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
            echo "  TIMEOUT waiting for $MARKER"
            return 1
        fi
        echo "  ... still waiting ($ELAPSED/${TIMEOUT}s) for $MARKER"
        sleep 60
    done
}

# ============================================================================
# Stage 1 — Preflight (auth, IAM, tokenizer + worker + train.parquet on GCS)
# ============================================================================

if [ "$DO_PREFLIGHT" == "1" ]; then
    echo
    echo "==== [Stage 1] preflight ===="
    # Auth check (fail-fast if creds expired)
    gcloud auth print-access-token >/dev/null
    gcloud storage ls "$BUCKET/" >/dev/null
    echo "  ✓ auth + bucket access"

    # train.parquet + manifest must already be on GCS
    # (one-time ship from apertus is a separate manual operation)
    for f in train.parquet train_manifest.parquet; do
        if ! gcloud storage ls "$BUCKET/c3_train_mix/$f" >/dev/null 2>&1; then
            echo "  FATAL: $BUCKET/c3_train_mix/$f not found."
            echo "  Run the one-time apertus → GCS ship first (see plan §3.4)."
            exit 1
        fi
    done
    echo "  ✓ train.parquet + train_manifest.parquet are on GCS"

    # Stage tokenizer + worker.py to the run prefix
    tar -czf /tmp/tokenizer.tar.gz -C "$SSP/variants" c3_added_17408_curated_padded
    gcloud storage cp /tmp/tokenizer.tar.gz "$GCS_RUN_PREFIX/tokenizer.tar.gz"
    gcloud storage cp "$SCRIPTS/firing_count_worker.py" "$GCS_RUN_PREFIX/worker.py"
    rm -f /tmp/tokenizer.tar.gz
    echo "  ✓ staged tokenizer + worker.py to $GCS_RUN_PREFIX"
fi

# ============================================================================
# Stage 2 — Sharder (pre-split train.parquet + manifest into K paired files)
# ============================================================================

if [ "$DO_SHARD" == "1" ]; then
    echo
    echo "==== [Stage 2] sharding train.parquet into K=$K paired shards ===="
    "$VENV/python" "$SCRIPTS/build_shard_manifests.py" \
        --text-parquet  "$BUCKET/c3_train_mix/train.parquet" \
        --manifest      "$BUCKET/c3_train_mix/train_manifest.parquet" \
        --k             "$K" \
        --gcs-out-prefix "$GCS_RUN_PREFIX" \
        --local-tmp      /tmp/firing_sharder
    echo "  ✓ shards uploaded to $GCS_RUN_PREFIX/shards/"
fi

# ============================================================================
# Stage 3 — Smoke (ONE instance, shard 0, --max-total-rows-per-component N)
# ============================================================================

if [ "$DO_SMOKE" == "1" ]; then
    echo
    echo "==== [Stage 3] smoke on shard 0 (max $SMOKE_MAX_ROWS rows/component) ===="
    SMOKE_NAME="firing-smoke-$TIMESTAMP"
    spawn_worker 0 true "$SMOKE_NAME"
    SMOKE_MARKER="$GCS_RUN_PREFIX/_smoke/_DONE_shard_00_of_$(printf '%02d' "$K")"
    if ! poll_done_marker "$SMOKE_MARKER" 1800; then
        echo "  FATAL: smoke didn't produce _DONE in 30 min."
        echo "  Inspect: gcloud compute ssh $SMOKE_NAME --zone $ZONE"
        exit 1
    fi
    # Validate smoke benchmark — fatal gates per FIRING_COUNT_PLAN.md §7.
    # ALLOW_SMOKE_SINGLE_COMPONENT=1 makes a missing component a warning
    # (e.g. when shard 0 is source-homogeneous because train.parquet is
    # sorted by stable_key and one component dominates the first row groups).
    gcloud storage cp "$GCS_RUN_PREFIX/_smoke/smoke_benchmark.json" /tmp/smoke_benchmark.json
    ALLOW_SMOKE_SINGLE_COMPONENT="${ALLOW_SMOKE_SINGLE_COMPONENT:-0}" \
    "$VENV/python" - <<PY
import json, os, sys
b = json.load(open("/tmp/smoke_benchmark.json"))
print(f"  smoke benchmark: tokens/s={b['tokenize_tokens_per_sec']:,} "
      f"cpu={b.get('mean_cpu_pct')}% rows={b['n_rows_processed']:,}")
print(f"  sources seen: {len(b['source_datasets_seen'])} — "
      f"sample {b['source_datasets_seen'][:3]}")
print(f"  component rows: {b['component_row_counts']}")

# Gate 1: source accounting wired (FATAL)
if not b['source_datasets_seen']:
    print("  FATAL: smoke saw no source_datasets — accounting broken")
    sys.exit(1)

# Gate 2: CPU saturation (warn only — instance/parallelism tunable)
if (b.get('mean_cpu_pct') or 0) < 70:
    print(f"  WARN: mean_cpu_pct {b['mean_cpu_pct']} < 70 — consider --batch-size or larger instance")

# Gate 3: both components seen (FATAL by default; overridable)
ga_rows = b['component_row_counts'].get('glossapi_nanochat_only', 0)
hp_rows = b['component_row_counts'].get('hplt_only', 0)
allow_single = os.environ.get("ALLOW_SMOKE_SINGLE_COMPONENT", "0") == "1"
if not (ga_rows > 0 and hp_rows > 0):
    if allow_single:
        print(f"  WARN: smoke saw only one component "
              f"(ga={ga_rows} hp={hp_rows}); ALLOW_SMOKE_SINGLE_COMPONENT=1 so proceeding")
    else:
        print(f"  FATAL: smoke saw only one component (ga={ga_rows} hp={hp_rows}).")
        print(f"  This usually means shard 0 is source-homogeneous because")
        print(f"  train.parquet is sorted by stable_key. The fleet WILL see")
        print(f"  both components, but the smoke can't validate that here.")
        print(f"  If you've verified shard balance is OK, override with:")
        print(f"    ALLOW_SMOKE_SINGLE_COMPONENT=1 bash scripts/run_firing_count.sh ...")
        sys.exit(1)
PY
    echo "  ✓ smoke passed"
fi

# ============================================================================
# Stage 4 — Full fleet (K parallel workers)
# ============================================================================

if [ "$DO_FLEET" == "1" ]; then
    echo
    echo "==== [Stage 4] spawning $K workers in parallel ===="
    for i in $(seq 0 $((K-1))); do
        NAME="firing-w-$i-$TIMESTAMP"
        spawn_worker "$i" false "$NAME" &
    done
    wait
    echo "  ✓ $K workers spawned"
    echo
    echo "==== polling for $K _DONE markers (timeout 45 min) ===="
    MISSING=()
    for i in $(seq 0 $((K-1))); do
        TAG=$(printf 'shard_%02d_of_%02d' "$i" "$K")
        MARKER="$GCS_RUN_PREFIX/_DONE_$TAG"
        if ! poll_done_marker "$MARKER" 2700; then
            MISSING+=("$i")
        else
            echo "  ✓ shard $i done"
        fi
    done
    if [ "${#MISSING[@]}" -gt 0 ]; then
        echo
        echo "  FATAL: ${#MISSING[@]} shards missing _DONE: ${MISSING[*]}"
        echo "  Respawn with:"
        for s in "${MISSING[@]}"; do
            echo "    SHARD=$s K=$K TIMESTAMP=$TIMESTAMP ZONE=$ZONE MACHINE=$MACHINE \\"
            echo "      bash $SCRIPTS/respawn_shard.sh"
        done
        echo
        echo "  Stopping; do not aggregate partial results."
        exit 1
    fi
fi

# ============================================================================
# Stage 5 — Aggregate (download K partials, sum, write final parquets)
# ============================================================================

if [ "$DO_AGGREGATE" == "1" ]; then
    echo
    echo "==== [Stage 5] aggregate ===="
    "$VENV/python" "$SCRIPTS/aggregate_firing_counts.py" \
        --gcs-prefix     "$GCS_RUN_PREFIX" \
        --k              "$K" \
        --tokenizer-dir  "$TOKENIZER_LOCAL" \
        --out-dir        "$OUT_DIR"
    echo
    echo "  ✓ outputs at: $OUT_DIR"
    ls -la "$OUT_DIR"
fi

echo
echo "==================================================================="
echo "DONE  GCS run prefix: $GCS_RUN_PREFIX"
echo "      local outputs : $OUT_DIR"
echo "==================================================================="
