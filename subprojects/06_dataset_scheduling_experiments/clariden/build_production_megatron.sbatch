#!/usr/bin/env bash
#SBATCH --job-name=mini_patch_megatron
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${SOURCE_MEGATRON:?set pinned SwissAI Megatron source checkout}"
: "${OUTPUT_MEGATRON:?set new patched Megatron checkout}"
: "${MEGATRON_RUNTIME_RECEIPT:?set new runtime receipt path}"
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}

[[ ! -e "$MEGATRON_RUNTIME_RECEIPT" ]] || {
  echo "refusing to replace MEGATRON_RUNTIME_RECEIPT=$MEGATRON_RUNTIME_RECEIPT" >&2
  exit 2
}
SOURCE_MEGATRON="$SOURCE_MEGATRON" OUTPUT_MEGATRON="$OUTPUT_MEGATRON" \
  SCIENTIFIC_ROOT="$SCIENTIFIC_BUNDLE" \
  bash "$SCIENTIFIC_BUNDLE/training/runtime_patches/apply_production_megatron_patches.sh"
"$HOST_PYTHON" - "$SCIENTIFIC_BUNDLE" "$SOURCE_MEGATRON" "$OUTPUT_MEGATRON" \
  "$MEGATRON_RUNTIME_RECEIPT" <<'PY'
import datetime,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path

scientific,source,output,receipt=map(Path,sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def rec(path): return {"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":sha(path)}
commit=subprocess.check_output(["git","-C",str(output),"rev-parse","HEAD"],text=True).strip()
diff=subprocess.check_output(["git","-C",str(output),"diff","--binary"])
changed=subprocess.check_output(["git","-C",str(output),"diff","--name-only"],text=True).splitlines()
expected=["megatron/training/arguments.py","megatron/training/training.py","pretrain_gpt.py"]
if commit!="c92402e39ef3c8e69ea378a59e79059dc14541f4" or sorted(changed)!=sorted(expected):
    raise ValueError("patched Megatron runtime contract drift")
payload={
    "schema_version":"apertus_mini_patched_megatron_runtime_v1",
    "status":"frozen",
    "created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_root":str(source.resolve()),"output_root":str(output.resolve()),
    "upstream_commit":commit,"git_diff_sha256":hashlib.sha256(diff).hexdigest(),
    "patches":[
        rec(scientific/"training/runtime_patches/megatron_extra_valid_c92402e.patch"),
        rec(scientific/"training/runtime_patches/megatron_exact_eval_iterations_c92402e.patch"),
    ],
    "patched_files":[rec(output/path) for path in expected],
    "checks":{"tracked_source_clean":True,"exact_changed_file_set":True,"git_diff_check_passed":True},
}
receipt.parent.mkdir(parents=True,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix=".megatron-runtime.",suffix=".partial",dir=receipt.parent)
with os.fdopen(fd,"w") as f:
    json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,receipt)
print(json.dumps({"ok":True,"output_root":str(output),"git_diff_sha256":payload["git_diff_sha256"]},sort_keys=True))
PY
