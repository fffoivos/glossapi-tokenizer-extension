#!/usr/bin/env bash
#SBATCH --job-name=h2g-init-geometry
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_INIT_SCALE:?set 8b or 1p5b}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
export H2G_CODE_ROOT H2G_CODE_RECEIPT
case "$H2G_INIT_SCALE" in
  8b)
    source="$H2G_STAGE_ROOT/assets/init/8b_td_hf_raw"
    source_authority="$H2G_STAGE_ROOT/receipts/8b_td_hf_materialization.json"
    output="$H2G_STAGE_ROOT/assets/init/8b_td_hf_training_geometry"
    receipt="$H2G_STAGE_ROOT/receipts/8b_td_hf_training_geometry.json"
    hidden=4096; layers=32
    ;;
  1p5b)
    source="$H2G_STAGE_ROOT/assets/init/1p5b_td_hf_raw"
    source_authority="$H2G_STAGE_ROOT/receipts/1p5b_td_initialization_v2.json"
    output="$H2G_STAGE_ROOT/assets/init/1p5b_td_hf_training_geometry"
    receipt="$H2G_STAGE_ROOT/receipts/1p5b_td_hf_training_geometry.json"
    hidden=2048; layers=16
    ;;
  *) echo "invalid H2G_INIT_SCALE=$H2G_INIT_SCALE" >&2; exit 2 ;;
esac
[[ -s "$source/config.json" && -s "$source_authority" ]] || { echo "initialization source or authority missing: $source" >&2; exit 2; }
[[ ! -e "$output" && ! -e "$receipt" ]] || { echo "immutable geometry output exists" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/prepare_training_geometry_hf.py" \
  --source-root "$source" --source-authority "$source_authority" \
  --output-root "$output" --output-receipt "$receipt" \
  --expected-vocab-size 148480 --expected-hidden-size "$hidden" --expected-hidden-layers "$layers"
