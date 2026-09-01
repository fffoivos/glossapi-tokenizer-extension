#!/usr/bin/env bash
#SBATCH --job-name=full8b-eval-overlap-proof
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=00:05:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_EVAL_OVERLAP_SMOKE_ROOT:?set unique overlap-smoke root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 4 ]] || {
  echo "evaluation-overlap proof requires four debug nodes" >&2
  exit 2
}
[[ ! -e "$FULL8_EVAL_OVERLAP_SMOKE_ROOT" ]] || {
  echo "evaluation-overlap smoke root already exists" >&2
  exit 2
}
mkdir -p "$FULL8_EVAL_OVERLAP_SMOKE_ROOT"

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#hosts[@]}" == 4 ]] || { echo "expected four hosts" >&2; exit 2; }
pids=()
labels=()
for group in 0 1 2; do
  labels+=("doc_group_$group")
  srun --nodes=1 --ntasks=1 --nodelist="${hosts[$group]}" \
    --gpus-per-node=4 --gres=gpu:4 --cpus-per-task=8 --mem=8G \
    --exact bash -lc \
    "printf '%s %s %s %s\\n' doc_group_$group \"\$SLURMD_NODENAME\" \"\$CUDA_VISIBLE_DEVICES\" \"\$SLURM_STEP_GPUS\"; sleep 15" \
    >"$FULL8_EVAL_OVERLAP_SMOKE_ROOT/doc_group_$group.out" 2>&1 &
  pids+=("$!")
done
labels+=("doc_group_3")
srun --nodes=1 --ntasks=1 --nodelist="${hosts[3]}" \
  --gpus-per-node=1 --gres=gpu:1 --cpus-per-task=8 --mem=8G \
  --exact bash -lc \
  "printf '%s %s %s %s\\n' doc_group_3 \"\$SLURMD_NODENAME\" \"\$CUDA_VISIBLE_DEVICES\" \"\$SLURM_STEP_GPUS\"; sleep 15" \
  >"$FULL8_EVAL_OVERLAP_SMOKE_ROOT/doc_group_3.out" 2>&1 &
pids+=("$!")
labels+=("greekmmlu")
srun --nodes=1 --ntasks=1 --nodelist="${hosts[3]}" \
  --gpus-per-node=1 --gres=gpu:1 --cpus-per-task=8 --mem=8G \
  --exact bash -lc \
  "printf '%s %s %s %s\\n' greekmmlu \"\$SLURMD_NODENAME\" \"\$CUDA_VISIBLE_DEVICES\" \"\$SLURM_STEP_GPUS\"; sleep 15" \
  >"$FULL8_EVAL_OVERLAP_SMOKE_ROOT/greekmmlu.out" 2>&1 &
pids+=("$!")

codes=()
failure=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then codes+=(0); else code=$?; codes+=("$code"); failure=1; fi
done

/usr/bin/python3.11 - \
  "$FULL8_EVAL_OVERLAP_SMOKE_ROOT/evaluation_overlap_smoke_receipt.json" \
  "$SLURM_JOB_ID" "${hosts[*]}" "${labels[*]}" "${codes[*]}" \
  "$FULL8_CODE_ROOT" "$FULL8_CODE_BUNDLE_RECEIPT" \
  "$FULL8_OPS_ROOT" "$FULL8_OPS_BUNDLE_RECEIPT" <<'PY'
import datetime,json,os,sys,tempfile
out,job,hosts,labels,codes,code_root,code_receipt,ops_root,ops_receipt=sys.argv[1:]
codes=[int(x) for x in codes.split()]
root=os.path.dirname(out)
observed={}
for label in labels.split():
 parts=open(os.path.join(root,label+".out")).read().strip().split()
 if len(parts)!=4 or parts[0]!=label: raise SystemExit({"bad_step_output":label,"parts":parts})
 observed[label]={"host":parts[1],"cuda_visible_devices":parts[2],"step_gpus":parts[3]}
checks={
 "partition_is_debug":os.environ.get("SLURM_JOB_PARTITION")=="debug",
 "nodes_are_four":os.environ.get("SLURM_NNODES")=="4" and len(hosts.split())==4,
 "five_steps_completed":len(codes)==5 and all(code==0 for code in codes),
 "shared_fourth_node_exercised":(
   len(labels.split())==5
   and observed["doc_group_3"]["host"]==observed["greekmmlu"]["host"]
 ),
 "shared_node_steps_got_distinct_gpus":(
   observed["doc_group_3"]["step_gpus"]
   != observed["greekmmlu"]["step_gpus"]
 ),
}
if not all(checks.values()): raise SystemExit(checks)
value={
 "schema_version":"apertus_full_8b_evaluation_overlap_smoke_v1",
 "status":"passed","completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "job_id":job,"partition":"debug","nodes":4,"hosts":hosts.split(),
 "steps":[{"label":label,"exit_code":code} for label,code in zip(labels.split(),codes)],
 "observed_step_resources":observed,
 "scientific_root":os.path.realpath(code_root),
 "scientific_receipt":os.path.realpath(code_receipt),
 "operational_root":os.path.realpath(ops_root),
 "operational_receipt":os.path.realpath(ops_receipt),
 "checks":checks,
}
fd,tmp=tempfile.mkstemp(prefix=".overlap.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f:
 json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
print(json.dumps({"ok":True,"job":job,"steps":len(codes)},sort_keys=True))
PY
exit "$failure"
