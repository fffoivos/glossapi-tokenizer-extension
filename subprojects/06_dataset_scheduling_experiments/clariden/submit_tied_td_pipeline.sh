#!/usr/bin/env bash
# Submit the receipt-gated tied-TD initialization chain. Dry-run is default.
set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}

: "${MINI_CODE_BUNDLE:?set immutable Mini code bundle}"
: "${MINI_BASE_MODEL_DIR:?set frozen base Mini model}"
: "${MINI_OVERLAY_TOKENIZER_DIR:?set frozen Mini tokenizer overlay}"
: "${MINI_FVT_INIT_DIR:?set frozen tied FVT initialization}"
: "${TD_CANONICAL_ADAPTER:?set frozen canonical Token Distillation adapter}"
: "${TD_FAIR_METRICS_SCRIPT:?set frozen tokenizer-fair metrics script}"
: "${TD_COVERAGE_JSONL:?set normalized complete TD coverage JSONL}"
: "${TD_SNIPPETS_JSONL:?set frozen TD snippet index}"
: "${TD_HPLT_EVAL_JSONL:?set frozen HPLT pilot heldout JSONL}"
: "${TD_NON_HPLT_EVAL_JSONL:?set frozen non-HPLT pilot heldout JSONL}"
: "${TD_POLYTONIC_EVAL_JSONL:?set frozen polytonic pilot heldout JSONL}"
: "${MEGATRON_DIR:?set pinned SwissAI Megatron checkout}"
: "${MEGATRON_PATCH_DIR:?set pinned Apertus conversion patch directory}"
: "${TD_RUN_ROOT:?set new initialization run root}"
DRY_RUN=${DRY_RUN:-1}
CONFIRM_GPU_LAUNCH=${CONFIRM_GPU_LAUNCH:-}
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
if [[ "$DRY_RUN" == 0 && "$CONFIRM_GPU_LAUNCH" != MINI_TIED_TD ]]; then
  echo "live launch requires CONFIRM_GPU_LAUNCH=MINI_TIED_TD" >&2
  exit 2
fi

TD_PILOT_ROOT="$TD_RUN_ROOT/pilots"
MINI_FULL_TD_DIR="$TD_RUN_ROOT/full_td"
TD_CONVERSION_ROOT="$TD_RUN_ROOT/conversion"
TD_INITIALIZATION_RECEIPT="$TD_RUN_ROOT/initialization_receipt.json"
TD_PILOT_SELECTION="$TD_PILOT_ROOT/selected_pilot.json"
scripts=(
  "$MINI_CODE_BUNDLE/clariden/run_tied_td_pilots.sbatch"
  "$MINI_CODE_BUNDLE/clariden/evaluate_tied_td_pilots.sbatch"
  "$MINI_CODE_BUNDLE/clariden/run_full_tied_td.sbatch"
  "$MINI_CODE_BUNDLE/clariden/verify_full_tied_td.sbatch"
  "$MINI_CODE_BUNDLE/clariden/convert_full_td_to_megatron.sbatch"
  "$MINI_CODE_BUNDLE/clariden/finalize_tied_td_initialization.sbatch"
)
for path in "${scripts[@]}"; do test -s "$path"; done
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY RUN tied-TD root=$TD_RUN_ROOT"
  echo "pilots(4 GPUs) -> 3-slice evaluation(4 GPUs) -> full TD -> full verification -> canonical Megatron roundtrip -> final receipt"
  exit 0
fi
[[ ! -e "$TD_RUN_ROOT" ]] || { echo "refusing to reuse $TD_RUN_ROOT" >&2; exit 3; }
mkdir -p "$TD_RUN_ROOT/logs"
common="ALL,MINI_CODE_BUNDLE=$MINI_CODE_BUNDLE,MINI_BASE_MODEL_DIR=$MINI_BASE_MODEL_DIR,MINI_OVERLAY_TOKENIZER_DIR=$MINI_OVERLAY_TOKENIZER_DIR,MINI_FVT_INIT_DIR=$MINI_FVT_INIT_DIR,TD_CANONICAL_ADAPTER=$TD_CANONICAL_ADAPTER,TD_FAIR_METRICS_SCRIPT=$TD_FAIR_METRICS_SCRIPT,TD_COVERAGE_JSONL=$TD_COVERAGE_JSONL,TD_SNIPPETS_JSONL=$TD_SNIPPETS_JSONL,TD_HPLT_EVAL_JSONL=$TD_HPLT_EVAL_JSONL,TD_NON_HPLT_EVAL_JSONL=$TD_NON_HPLT_EVAL_JSONL,TD_POLYTONIC_EVAL_JSONL=$TD_POLYTONIC_EVAL_JSONL,TD_PILOT_ROOT=$TD_PILOT_ROOT,TD_PILOT_SELECTION=$TD_PILOT_SELECTION,MINI_FULL_TD_DIR=$MINI_FULL_TD_DIR,MEGATRON_DIR=$MEGATRON_DIR,MEGATRON_PATCH_DIR=$MEGATRON_PATCH_DIR,TD_CONVERSION_ROOT=$TD_CONVERSION_ROOT,TD_INITIALIZATION_RECEIPT=$TD_INITIALIZATION_RECEIPT"
pilots=$(sbatch --parsable --export="$common" --output="$TD_RUN_ROOT/logs/%x-%j.out" --error="$TD_RUN_ROOT/logs/%x-%j.err" "${scripts[0]}")
pilot_eval=$(sbatch --parsable --dependency="afterok:$pilots" --export="$common" --output="$TD_RUN_ROOT/logs/%x-%j.out" --error="$TD_RUN_ROOT/logs/%x-%j.err" "${scripts[1]}")
full=$(sbatch --parsable --dependency="afterok:$pilot_eval" --export="$common" --output="$TD_RUN_ROOT/logs/%x-%j.out" --error="$TD_RUN_ROOT/logs/%x-%j.err" "${scripts[2]}")
full_verify=$(sbatch --parsable --dependency="afterok:$full" --export="$common" --output="$TD_RUN_ROOT/logs/%x-%j.out" --error="$TD_RUN_ROOT/logs/%x-%j.err" "${scripts[3]}")
conversion=$(sbatch --parsable --dependency="afterok:$full_verify" --export="$common" --output="$TD_RUN_ROOT/logs/%x-%j.out" --error="$TD_RUN_ROOT/logs/%x-%j.err" "${scripts[4]}")
finalize=$(sbatch --parsable --dependency="afterok:$conversion" --export="$common" --output="$TD_RUN_ROOT/logs/%x-%j.out" --error="$TD_RUN_ROOT/logs/%x-%j.err" "${scripts[5]}")
"$HOST_PYTHON" - "$TD_RUN_ROOT/submission_receipt.json" "$pilots" "$pilot_eval" "$full" "$full_verify" "$conversion" "$finalize" <<'PY'
import datetime,json,os,sys,tempfile
out,*ids=sys.argv[1:]
names=("pilots","pilot_eval","full_td","full_verify","conversion","finalize")
data={"schema_version":"apertus_mini_td_submission_v1","status":"submitted","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"jobs":dict(zip(names,ids))}
fd,tmp=tempfile.mkstemp(prefix=".submission.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f: json.dump(data,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out); print(json.dumps(data,sort_keys=True))
PY
