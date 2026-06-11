#!/usr/bin/env bash
# Copy lightweight 5B continuation eval artifacts from Clariden into the local
# trajectory bundle. This intentionally copies JSON summaries/logs only, not
# checkpoint weights.

set -euo pipefail

REMOTE="${REMOTE:-clariden}"
TAG="${TAG:-continuation_5b_td_vs_vanilla_20260525T142522Z}"
REMOTE_EVAL_ROOT="${REMOTE_EVAL_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
DEST_ROOT="${DEST_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/per_iter_results}"
ITER_LIST="${ITER_LIST:-1013 1192}"

mkdir -p "$DEST_ROOT/intrinsic" "$DEST_ROOT/diagnostics" "$DEST_ROOT/training_logs"

copy_remote_file() {
    local remote_path="$1"
    local local_path="$2"
    if ssh "$REMOTE" "test -s '$remote_path'"; then
        mkdir -p "$(dirname "$local_path")"
        scp -q "$REMOTE:$remote_path" "$local_path"
        printf 'copied %s -> %s\n' "$remote_path" "$local_path"
        return 0
    fi
    printf 'missing %s\n' "$remote_path" >&2
    return 1
}

latest_results_json() {
    local arm="$1"
    local iter_pad="$2"
    ssh "$REMOTE" "find '$REMOTE_EVAL_ROOT/${TAG}_${arm}/iter_${iter_pad}_full' -maxdepth 1 -type f -name 'results_*.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-"
}

for iter in $ITER_LIST; do
    if ! [[ "$iter" =~ ^[0-9]+$ ]]; then
        echo "ERROR: non-numeric iteration: $iter" >&2
        exit 2
    fi
    iter_pad="$(printf "%07d" "$iter")"

    for arm in vanilla td_layer11; do
        case "$arm" in
            vanilla) local_arm="vanilla" ;;
            td_layer11) local_arm="td" ;;
        esac

        results_path="$(latest_results_json "$arm" "$iter_pad")"
        if [ -n "$results_path" ]; then
            copy_remote_file "$results_path" "$DEST_ROOT/${local_arm}_iter${iter}.json" || true
        else
            echo "missing results json for $arm iter $iter" >&2
        fi

        copy_remote_file \
            "$REMOTE_EVAL_ROOT/${TAG}_${arm}/iter_${iter_pad}_tokenizer_fair_metrics.json" \
            "$DEST_ROOT/intrinsic/${local_arm}_iter${iter}_fair.json" || true

        if [ "$arm" = "td_layer11" ]; then
            copy_remote_file \
                "$REMOTE_EVAL_ROOT/${TAG}_${arm}/iter_${iter_pad}_new_token_diagnostics.json" \
                "$DEST_ROOT/diagnostics/${local_arm}_iter${iter}_new_token_diagnostics.json" || true
        fi
    done
done

copy_remote_file "$REMOTE_LOG_ROOT/5b_vanilla_1013-2382982.out" "$DEST_ROOT/training_logs/5b_vanilla_1013-2382982.out" || true
copy_remote_file "$REMOTE_LOG_ROOT/5b_td_layer11_1013-2382984.out" "$DEST_ROOT/training_logs/5b_td_layer11_1013-2382984.out" || true
copy_remote_file "$REMOTE_LOG_ROOT/5b_vanilla_1192-2382983.out" "$DEST_ROOT/training_logs/5b_vanilla_1192-2382983.out" || true
copy_remote_file "$REMOTE_LOG_ROOT/5b_td_layer11_1192-2382985.out" "$DEST_ROOT/training_logs/5b_td_layer11_1192-2382985.out" || true
