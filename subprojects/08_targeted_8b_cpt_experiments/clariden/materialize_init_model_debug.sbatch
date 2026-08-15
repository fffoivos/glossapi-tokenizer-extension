#!/usr/bin/env bash
#SBATCH --job-name=h2g-init-hf
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_INIT_SCALE:?set 8b or 1p5b}"
: "${HF_TOKEN:?inject HF_TOKEN without persisting it}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "HF model materialization requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
script="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/materialize_pinned_hf_model.py"
cache="$H2G_STAGE_ROOT/cache/huggingface"
mkdir -p "$H2G_STAGE_ROOT/assets/init" "$H2G_STAGE_ROOT/receipts" "$cache"
export H2G_CODE_ROOT H2G_CODE_RECEIPT HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=900 HF_HUB_ETAG_TIMEOUT=120
case "$H2G_INIT_SCALE" in
  8b)
    output="$H2G_STAGE_ROOT/assets/init/8b_td_hf_raw"
    receipt="$H2G_STAGE_ROOT/receipts/8b_td_hf_materialization.json"
    args=(
      --repo-id fffoivos/apertus-tokenizer-extension
      --revision fcd33ec09fb7d86bc072b3a4b3e890efa6473b66
      --subfolder experiment-checkpoints/TokenDistil-Init
      --expected-vocab-size 148480 --expected-hidden-size 4096 --expected-hidden-layers 32
      --expected-attention-heads 32 --expected-kv-heads 8
      --expected-safetensors 4 --expected-safetensor-bytes 16391939480
      --expected-td-target-layer 11
    )
    ;;
  1p5b)
    output="$H2G_STAGE_ROOT/assets/init/1p5b_parent_hf"
    receipt="$H2G_STAGE_ROOT/receipts/1p5b_parent_hf_materialization.json"
    args=(
      --repo-id swiss-ai/Apertus-v1.1-1.5B
      --revision dbe8919b2f0389888bada6b3a19e81e0ef4286c1
      --expected-vocab-size 131072 --expected-hidden-size 2048 --expected-hidden-layers 16
      --expected-attention-heads 32 --expected-kv-heads 8
    )
    ;;
  *) echo "invalid H2G_INIT_SCALE=$H2G_INIT_SCALE" >&2; exit 2 ;;
esac
[[ ! -e "$receipt" ]] || { echo "receipt already exists: $receipt"; exit 0; }
uenv run pytorch/v2.9.1:v2 --view=default -- python3 "$script" \
  "${args[@]}" --output-root "$output" --cache-dir "$cache" --output-receipt "$receipt"
