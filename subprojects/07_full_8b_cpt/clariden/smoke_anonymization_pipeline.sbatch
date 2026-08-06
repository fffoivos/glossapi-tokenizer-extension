#!/usr/bin/env bash
#SBATCH --job-name=full8-anon-smoke
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=460G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
DATA_PYTHON=${FULL8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
DATA_RUNTIME_RECEIPT=${FULL8_DATA_RUNTIME_RECEIPT:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/runtime_receipt.json}
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code-bundle receipt}"
: "${FULL8_ANON_STAGE_ROOT:?set anonymized derivative root}"
: "${FULL8_PARENT_SOURCE_ROOT:?set parent tokenized-corpus root}"
test -s "$DATA_RUNTIME_RECEIPT"
"$HOST_PYTHON" "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" -c \
  'import pyarrow,tokenizers; assert pyarrow.__version__ == "21.0.0"'
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$FULL8_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/scripts/test_pii_local.py"

mapfile -t indices < <("$HOST_PYTHON" - "$FULL8_ANON_STAGE_ROOT/anonymization_overlay.json" "$FULL8_PARENT_SOURCE_ROOT" <<'PY'
import json, os, sys
value=json.load(open(sys.argv[1])); parent=sys.argv[2]; found={}
for index, task in enumerate(value["tasks"]):
    pool=task["output_prefix"].split("/",2)[1]
    legacy=os.path.join(parent,"megatron",task["output_prefix"]+".bin")
    if not os.path.isfile(legacy) or os.path.getsize(legacy) == 0:
        continue
    candidate=(os.path.getsize(legacy),index)
    if pool not in found or candidate < found[pool]:
        found[pool]=candidate
for pool in ("hplt_new_greek","non_hplt_new_greek","foreign_replay","old_greek_replay"):
    if pool not in found: raise SystemExit(f"missing non-empty smoke pool: {pool}")
    print(found[pool][1])
PY
)
[[ "${#indices[@]}" -eq 4 ]]
pids=()
for index in "${indices[@]}"; do
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=72 --mem=110G \
    uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
    "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/anonymization/build_anonymization_inventory.py" \
    --overlay "$FULL8_ANON_STAGE_ROOT/anonymization_overlay.json" \
    --heldout-manifest "$FULL8_ANON_STAGE_ROOT/heldouts/heldout_manifest.json" \
    --stage-root "$FULL8_ANON_STAGE_ROOT" --task-index "$index" --workers 64 --chunksize 16 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || exit "$status"
"$HOST_PYTHON" - "$FULL8_ANON_STAGE_ROOT" "${indices[@]}" <<'PY'
import json, os, sys
root=sys.argv[1]
for raw in sys.argv[2:]:
    index=int(raw)
    path=os.path.join(root,"inventory",f"task_{index:05d}.manifest.json")
    value=json.load(open(path))
    counts=value.get("counts",{})
    if counts.get("candidate_rows",0) <= 0 or counts.get("inventory_rows",0) <= 0:
        raise SystemExit(f"smoke task {index} did not exercise retained rows: {value}")
print(json.dumps({"ok":True,"nonempty_inventory_tasks":[int(x) for x in sys.argv[2:]]},sort_keys=True))
PY
