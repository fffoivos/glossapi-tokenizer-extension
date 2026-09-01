#!/usr/bin/env bash
set -euo pipefail

# Experiment-local binding adapter: recompile the unchanged 8B scientific
# campaign against the frozen canonical runner that supports direct salloc,
# then reuse the already-proven 16-node runtime through the canonical
# rebind-proven-runtime contract. This performs no model or dataset work.

: "${H2G_REBIND_OUTPUT_ROOT:?set a new immutable output root}"

ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
BASE="$ST/canonical/pre_main_data_v9_safepath_5caa614d_v103"
RUNNER=/iopsstor/scratch/cscs/fffoivos/orchestration/apertus-cscs-efficiency/20260818T211000Z-6b7796a-sequence-range
RUN_ROOT="$ST/runs/efficiency_bound_proven_8b_v112_recovery_codebinding_v3_staticready"

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "debug allocation required" >&2; exit 2; }
[[ "${SLURM_JOB_NUM_NODES:-0}" == 1 ]] || { echo "exactly one debug node required" >&2; exit 2; }
[[ ! -e "$H2G_REBIND_OUTPUT_ROOT" ]] || { echo "immutable output root exists" >&2; exit 2; }

mkdir -p "$H2G_REBIND_OUTPUT_ROOT"
source_campaign="$BASE/campaign_8b_s3_recovery_e9e2921.json"
campaign="$H2G_REBIND_OUTPUT_ROOT/campaign-v3-entrypoint-bound.json"
candidate="$H2G_REBIND_OUTPUT_ROOT/compiled-candidate.json"
qualification_root="$H2G_REBIND_OUTPUT_ROOT/operational-rebind"
proven_runtime="$H2G_REBIND_OUTPUT_ROOT/runtime-proven.json"
proven_manifest="$H2G_REBIND_OUTPUT_ROOT/manifest-proven.json"

# The source campaign predates the canonical runner's v3 closure rule. Derive
# the newly required entrypoint binding from the exact file already executed by
# train_argv. This adds evidence only: train_argv and every science value remain
# byte-for-byte unchanged.
python3 - "$source_campaign" "$campaign" \
  "$BASE/compiled_8b_s3_recovery_e9e2921.json" \
  "$RUN_ROOT/permits/s4/from_s3_attempt_000003_ca9dd64_postprocess_fix.json" <<'PY'
import hashlib
import json
import os
import shutil
import sys

source, output, source_manifest, recovery_permit = sys.argv[1:]
with open(source, "r", encoding="utf-8") as handle:
    campaign = json.load(handle)
# Preserve every portable campaign-level file binding beside the derivative.
# The recursive walk is mechanical and verifies source existence; the current
# runner re-verifies each copied byte count and digest during compile.
source_dir = os.path.dirname(source)
output_dir = os.path.dirname(output)
def copy_portable_bindings(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            relative = value["path"]
            if not os.path.isabs(relative):
                source_path = os.path.join(source_dir, relative)
                output_path = os.path.join(output_dir, relative)
                if not os.path.isfile(source_path):
                    raise SystemExit(f"portable binding source absent: {source_path}")
                os.makedirs(os.path.dirname(output_path) or output_dir, exist_ok=True)
                shutil.copyfile(source_path, output_path)
        for child in value.values():
            copy_portable_bindings(child)
    elif isinstance(value, list):
        for child in value:
            copy_portable_bindings(child)
copy_portable_bindings(campaign)
# Portable bindings are relative to the campaign file and absolute paths are
# forbidden. Copy the already-frozen manifest and its prepared-gate receipts
# byte-for-byte beside the audited derivative, preserving every binding value.
data_binding = campaign["science"].get("training_data_manifest")
if data_binding and not os.path.isabs(data_binding["path"]):
    source_manifest_path = os.path.join(source_dir, data_binding["path"])
    output_manifest_path = os.path.join(output_dir, data_binding["path"])
    shutil.copyfile(source_manifest_path, output_manifest_path)
    with open(source_manifest_path, "r", encoding="utf-8") as handle:
        training_manifest = json.load(handle)
    for row in training_manifest["datasets"]:
        relative = row["prepared_gate"]["path"]
        shutil.copyfile(os.path.join(source_dir, relative), os.path.join(output_dir, relative))
argv = campaign["science"]["train_argv"]
try:
    entrypoint_path = next(
        token for token in argv
        if token.endswith("/run_canonical_train_segment.py")
    )
except StopIteration as exc:
    raise SystemExit("canonical training entrypoint absent from train_argv") from exc
with open(entrypoint_path, "rb") as handle:
    payload = handle.read()
campaign["science"]["entrypoint"] = {
    "kind": "file",
    "path": entrypoint_path,
    "bytes": len(payload),
    "sha256": hashlib.sha256(payload).hexdigest(),
    "verify_at_submit": True,
}
def binding(path):
    with open(path, "rb") as handle:
        payload = handle.read()
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

# The canonical v3 migration escape hatch is deliberately narrow: it skips
# closure of historical argv only when both the historical compiled manifest
# and the exact recovery permit are digest-bound, and preserve_train_argv is
# true. The runner's validator enforces all three conditions.
campaign["legacy_continuation"] = {
    "source_manifest": binding(source_manifest),
    "recovery_permit": binding(recovery_permit),
    "preserve_train_argv": True,
}
with open(output, "x", encoding="utf-8") as handle:
    json.dump(campaign, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
PY

"$RUNNER/bin/apertus-campaign" compile \
  --campaign "$campaign" \
  --runtime "$BASE/runtime_8b_candidate.json" \
  --evaluation "$BASE/evaluation_8b_s3_recovery_e9e2921.json" \
  --output "$candidate"

"$RUNNER/bin/apertus-campaign" rebind-proven-runtime \
  --source-manifest "$BASE/compiled_8b_s3_recovery_e9e2921.json" \
  --candidate-manifest "$candidate" \
  --output-root "$qualification_root"

"$RUNNER/bin/apertus-campaign" promote-runtime \
  --manifest "$candidate" \
  --qualification "$qualification_root/runtime-qualification.json" \
  --runtime-output "$proven_runtime" \
  --output "$proven_manifest"

"$RUNNER/bin/apertus-campaign" select-run-root \
  --manifest "$proven_manifest" --run-root "$RUN_ROOT" \
  --output "$H2G_REBIND_OUTPUT_ROOT/run-root-selection.json"

"$RUNNER/bin/apertus-campaign" status \
  --manifest "$proven_manifest" --run-root "$RUN_ROOT" \
  > "$H2G_REBIND_OUTPUT_ROOT/status.json"

chmod -R a-w "$H2G_REBIND_OUTPUT_ROOT"
printf '%s\n' "$proven_manifest"
