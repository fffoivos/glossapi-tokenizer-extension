#!/usr/bin/env bash
# Explicit two-step CPU build submitter. Defaults to a non-submitting dry run.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.env"
MODE=${1:-status}
case "$MODE" in
    restage) bridge_require_base ;;
    freeze|after-freeze) bridge_require_paths ;;
    status) bridge_require_run ;;
    *)
        echo "usage: $0 {restage|freeze|after-freeze|status}" >&2
        exit 2
        ;;
esac
DRY_RUN=${DRY_RUN:-1}
CONFIRM_BUILD=${CONFIRM_BUILD:-0}
CONFIRM_RESTAGE=${CONFIRM_RESTAGE:-}
MAX_PARALLEL_TRAIN=${MAX_PARALLEL_TRAIN:-12}
MAX_PARALLEL_HELDOUT=${MAX_PARALLEL_HELDOUT:-6}

case "$DRY_RUN" in 0|1) ;; *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
if [[ "$DRY_RUN" == 0 ]]; then
    if [[ "$MODE" == restage ]]; then
        [[ "$CONFIRM_RESTAGE" == "PINNED_REPLAY_V1" ]] || {
            echo "ERROR: restaging requires CONFIRM_RESTAGE=PINNED_REPLAY_V1" >&2; exit 2;
        }
        : "${HF_TOKEN:?inject HF_TOKEN for pinned Hugging Face restaging}"
    elif [[ "$CONFIRM_BUILD" != 1 ]]; then
        echo "ERROR: live CPU submission requires CONFIRM_BUILD=1" >&2
        exit 2
    fi
fi

submit() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'DRY:'; printf ' %q' sbatch --parsable "$@"; printf '\n'
        echo "DRY$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

common=(--account="$BRIDGE_ACCOUNT" --partition="$BRIDGE_PARTITION" --export="ALL,BRIDGE_CLARIDEN_DIR=$HERE")
case "$MODE" in
    restage)
        acquire_job=$(submit "${common[@]}" "$HERE/05_acquire_replay.sbatch" | tail -n 1)
        old_greek_job=$(submit "${common[@]}" --dependency="afterok:$acquire_job" \
            "$HERE/07_build_old_greek.sbatch" | tail -n 1)
        printf 'acquisition_job=%s\nold_greek_job=%s\n' "$acquire_job" "$old_greek_job"
        ;;
    freeze)
        test -s "$REPLAY_ACQUISITION_RECEIPT" || { echo "ERROR: restage first: $REPLAY_ACQUISITION_RECEIPT" >&2; exit 3; }
        test -s "$OLD_GREEK_BUILD_RECEIPT" || { echo "ERROR: build old Greek first: $OLD_GREEK_BUILD_RECEIPT" >&2; exit 3; }
        job=$(submit "${common[@]}" "$HERE/10_freeze_inputs.sbatch" | tail -n 1)
        echo "freeze_job=$job"
        echo "After it completes, run: DRY_RUN=$DRY_RUN CONFIRM_BUILD=$CONFIRM_BUILD $0 after-freeze"
        ;;
    after-freeze)
        test -s "$INPUT_RECEIPT" || { echo "ERROR: freeze first: $INPUT_RECEIPT" >&2; exit 3; }
        read -r train_count heldout_count < <(bridge_python - "$INPUT_RECEIPT" "$BRIDGE_CONFIG" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8")); c=json.load(open(sys.argv[2],encoding="utf-8"))
print(r["task_count"], sum(len(c["heldouts"][p]) for p in ("new_greek","foreign_replay","old_greek_replay")))
PY
)
        (( train_count > 0 && heldout_count > 0 ))
        heldout_job=$(submit "${common[@]}" "$HERE/20_build_heldouts.sbatch" | tail -n 1)
        train_job=$(submit "${common[@]}" --dependency="afterok:$heldout_job" \
            --array="0-$((train_count-1))%$MAX_PARALLEL_TRAIN" "$HERE/30_build_train_shards.sbatch" | tail -n 1)
        val_job=$(submit "${common[@]}" --dependency="afterok:$heldout_job" \
            --array="0-$((heldout_count-1))%$MAX_PARALLEL_HELDOUT" "$HERE/40_build_heldout_shards.sbatch" | tail -n 1)
        final_job=$(submit "${common[@]}" --dependency="afterok:$train_job:$val_job" "$HERE/50_finalize_bridge.sbatch" | tail -n 1)
        assets_job=$(submit "${common[@]}" --dependency="afterok:$final_job" "$HERE/60_freeze_training_assets.sbatch" | tail -n 1)
        printf 'heldouts_job=%s\ntrain_array_job=%s\nheldout_array_job=%s\nfinalize_job=%s\nassets_job=%s\n' \
            "$heldout_job" "$train_job" "$val_job" "$final_job" "$assets_job"
        ;;
    status)
        for path in "$REPLAY_ACQUISITION_RECEIPT" "$OLD_GREEK_BUILD_RECEIPT" \
                    "$INPUT_RECEIPT" "$HELDOUT_MANIFEST" "$BRIDGE_MANIFEST" \
                    "$TRAINING_ASSETS_RECEIPT"; do
            if [[ -s "$path" ]]; then echo "READY $path"; else echo "MISSING $path"; fi
        done
        ;;
    *)
        echo "usage: $0 {restage|freeze|after-freeze|status}" >&2
        exit 2
        ;;
esac
