#!/usr/bin/env bash
# Relaunch the three v4 tails after the optimizer param-group min_lr bug.
set -euo pipefail

RUN_ID="${LR13_RUN_ID:-20260801T171214Z-lr-floor-13b-v4}"
CODE="${LR13_CODE_ROOT:-/iopsstor/scratch/cscs/fffoivos/experiments/lr-floor-13b-v4/code}"
DATA="${LR13_DATA_ROOT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/lr_floor_13b/$RUN_ID}"
RUN="${LR13_RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/lr_floor_13b/$RUN_ID}"
SOURCE="${LR13_SOURCE_REPO:-/iopsstor/scratch/cscs/fffoivos/repo/train-apertus-with-glossapi-cpt25b-86c1b8fe}"
MEGATRON="${MEGATRON_LM_SWISSAI_DIR:-/iopsstor/scratch/cscs/fffoivos/experiments/lr-floor-13b-v4/megatron-extra-valid}"
FIX="$RUN/submissions/lr_floor_resume_fix_bundle"
FREEZE="$RUN/submissions/freeze_bundle"
BRANCH_RECEIPT="$RUN/checkpoint_receipts/iteration_2574.json"

common="ALL,LR13_CODE_ROOT=$CODE,LR13_ASSETS_RECEIPT=$DATA/training_assets_receipt.json,LR13_TRAINING_DATA_ENV=$DATA/training_data.env,LR13_SOURCE_REPO=$SOURCE,MEGATRON_LM_SWISSAI_DIR=$MEGATRON,LR13_RECOVERY_BUNDLE=$FIX"
freeze_common="ALL,LR13_CODE_ROOT=$FREEZE,LR13_ASSETS_RECEIPT=$DATA/training_assets_receipt.json"
tails=()
freezes=()
for floor in 10 20 30; do
  branch="$RUN/T$floor"
  job=$(sbatch --parsable --account=a0140 --partition=normal --nodes=16 --time=03:00:00 \
    --job-name="${RUN_ID}_T${floor}r1" --output="$branch/%x-%j.out" --error="$branch/%x-%j.err" \
    --export="$common,LR13_START_ITERATION=2574,LR13_END_ITERATION=3218,LR13_PHASE=2,LR13_LR_FLOOR_PERCENT=$floor,LR13_SAVE_INTERVAL=107,LR13_LOAD_CHECKPOINT=$RUN/shared_prefix/checkpoints,LR13_RESUME_RECEIPT=$BRANCH_RECEIPT,LR13_OUTPUT_DIR=$branch" \
    "$FIX/train_segment_lr_floor_resume_recovery.sbatch")
  frozen=$(sbatch --parsable --account=a0140 --partition=xfer --dependency="afterok:$job" \
    --job-name="${RUN_ID}_T${floor}r1_freeze" --output="$branch/%x-%j.out" --error="$branch/%x-%j.err" \
    --export="$freeze_common,LR13_CHECKPOINT_DIR=$branch/checkpoints,LR13_CHECKPOINT_ITERATION=3218,LR13_CHECKPOINT_RECEIPT=$RUN/checkpoint_receipts/T${floor}_iteration_3218.json" \
    "$FREEZE/clariden/freeze_checkpoint.sbatch")
  tails+=("$job")
  freezes+=("$frozen")
  printf 'T%s=%s freeze=%s\n' "$floor" "$job" "$frozen"
done

python3 - "$RUN/submissions/launch_graph_lr_floor_fix_20260802.json" "${tails[@]}" "${freezes[@]}" <<'PY'
import datetime
import json
import os
import sys
import tempfile

out, t10, t20, t30, z10, z20, z30 = sys.argv[1:]
value = {
    "schema_version": "apertus8b_lr_floor_resume_fix_submission_v1",
    "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "reason": "optimizer parameter-group min_lr from shared checkpoint overrode the branch scheduler floor",
    "invalid_jobs": {"T10": "2972676", "T20": "2972678", "T30": "2972680"},
    "invalid_final_freezes": {"T10": "2974836", "T20": "2974837", "T30": "2974838"},
    "branch_checkpoint_receipt": "/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/lr_floor_13b/20260801T171214Z-lr-floor-13b-v4/checkpoint_receipts/iteration_2574.json",
    "fix_bundle": {
        "root": "/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/lr_floor_13b/20260801T171214Z-lr-floor-13b-v4/submissions/lr_floor_resume_fix_bundle",
        "files": {
            "lr_floor_resume.py": "76f4637e9b9495830339e5705516453944586cc6766b8a4b5a5b1de7ab665297",
            "lr_floor_config_recovery.env": "58bb48fcef29b7a295d387f139099b58d45811118ed49afa498ed29babed8760",
            "train_segment_lr_floor_resume_recovery.sbatch": "2e7879e44664005ca9d43d42861ae54be9db7dc0d4a632a9ffbd6b6799e66d0d",
        },
    },
    "jobs": {
        "tails": {"T10": t10, "T20": t20, "T30": t30},
        "freeze_final": {"T10": z10, "T20": z20, "T30": z30},
    },
}
fd, tmp = tempfile.mkstemp(prefix=".launch_fix.", suffix=".partial", dir=os.path.dirname(out))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, out)
print(out)
PY

squeue -j "${tails[0]},${tails[1]},${tails[2]},${freezes[0]},${freezes[1]},${freezes[2]}" \
  -o "%.18i %.2t %.12M %.4D %R"
