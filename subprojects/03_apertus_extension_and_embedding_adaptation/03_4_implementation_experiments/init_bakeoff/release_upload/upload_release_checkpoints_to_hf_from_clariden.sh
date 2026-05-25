#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; echo "[$(date -Is)] EXIT rc=$rc"; exit $rc' EXIT

# Run this on a Clariden xfer node with HF_TOKEN in the environment.
# Example submission from a login shell after setting HF_TOKEN:
#
#   sbatch --partition=xfer --time=24:00:00 --ntasks=1 --cpus-per-task=4 \
#     --mem=16G --job-name=apertus_hf_ckpts \
#     --output="$HOME/apertus_hf_upload_checkpoints_%j.log" \
#     --export=ALL,HF_TOKEN \
#     --wrap="bash $PWD/upload_release_checkpoints_to_hf_from_clariden.sh"

REPO="${REPO:-fffoivos/apertus-tokenizer-extension}"
HF_BIN="${HF_BIN:-$HOME/.venvs/hfupload/bin/hf}"
STATUS_DIR="${STATUS_DIR:-$HOME/apertus_hf_upload_status}"
mkdir -p "$STATUS_DIR"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set" >&2
  exit 2
fi
if [[ ! -x "$HF_BIN" ]]; then
  echo "ERROR: HF CLI missing at $HF_BIN" >&2
  exit 2
fi

upload_one() {
  local name="$1"
  local src="$2"
  local dst="$3"
  echo "[$(date -Is)] START $name"
  if [[ ! -d "$src" ]]; then
    echo "[$(date -Is)] ERROR missing source for $name: $src" >&2
    exit 3
  fi

  local missing=0
  for f in config.json generation_config.json model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer_config.json; do
    if [[ ! -f "$src/$f" ]]; then
      echo "[$(date -Is)] ERROR $name missing $f" >&2
      missing=1
    fi
  done
  if [[ "$missing" != 0 ]]; then
    exit 4
  fi

  du -sh "$src"
  find "$src" -maxdepth 1 -type f -printf "%f %s\n" | sort
  "$HF_BIN" upload "$REPO" "$src" "$dst" \
    --type model \
    --commit-message "Upload $name checkpoint" \
    --format quiet
  echo "[$(date -Is)] DONE $name"
  touch "$STATUS_DIR/$name.done"
}

upload_one "TokenDistil-Init" "/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/hf_roundtrip" "experiment-checkpoints/TokenDistil-Init"
upload_one "TokenDistil-2B" "/capstor/scratch/cscs/fffoivos/runs/eval/td_full25_layer11_2b_20260523T165038Z/iter_0000476_hf" "experiment-checkpoints/TokenDistil-2B"
upload_one "TokenDistil-3.5B" "/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_td_layer11/iter_0000834_hf" "experiment-checkpoints/TokenDistil-3.5B"
upload_one "Vanilla-2B" "/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_vanilla/iter_0000476_hf" "experiment-checkpoints/Vanilla-2B"
upload_one "Vanilla-3.5B" "/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_vanilla/iter_0000834_hf" "experiment-checkpoints/Vanilla-3.5B"
upload_one "ReTok-2B" "/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_retok/iter_0000476_hf" "experiment-checkpoints/ReTok-2B"
upload_one "ReTok-3.5B" "/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_retok/iter_0000834_hf" "experiment-checkpoints/ReTok-3.5B"
upload_one "Centroid-2B" "/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_centroid/iter_0000476_hf" "experiment-checkpoints/Centroid-2B"

echo "[$(date -Is)] ALL_DONE"
touch "$STATUS_DIR/all.done"
