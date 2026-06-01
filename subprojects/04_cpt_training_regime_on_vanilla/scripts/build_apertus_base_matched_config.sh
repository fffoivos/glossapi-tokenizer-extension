#!/usr/bin/env bash
# Build a matched-geometry copy of the local Apertus-Base HF snapshot so the
# 5B Vanilla CPT comparison baseline is apples-to-apples on positional
# geometry.
#
# Background.
#   /iopsstor/.../models/apertus-8b-2509/config.json declares
#     max_position_embeddings = 65536
#     rope_theta              = 12000000
#     rope_scaling            = {rope_type: llama3, factor: 8.0,
#                                original_max_position_embeddings: 8192,
#                                low_freq_factor: 1.0, high_freq_factor: 4.0}
#   This is the Apertus paper §2.5 long-context-extension geometry. The 04
#   Vanilla CPT run trained with rope_theta=500000, max_position=4096,
#   rope_scaling=null (matching paper §2.3 initial pretraining, which is the
#   only geometry the bakeoff and the 04 run ever exposed). Vanilla-0.5B
#   Critical-1 (still live at iter 238) and reports/config_geometry_audit_*
#   document that any "CPT vs Apertus-Base" delta is currently confounded by
#   positional geometry.
#
# What this script does.
#   - Stage a sibling HF directory at
#     /iopsstor/.../models/apertus-8b-2509-matched-rope500k-seq4096/.
#   - Symlink every weight + tokenizer file from the original snapshot so we
#     avoid duplicating ~16 GB of safetensors.
#   - Write a config.json that overrides ONLY three fields:
#       rope_theta              = 500000
#       max_position_embeddings = 4096
#       rope_scaling            = null
#     All other fields (vocab_size, hidden_size, num_attention_heads, dtype,
#     tie_word_embeddings, etc.) are copied verbatim from the source.
#   - Emit a MANIFEST.json next to the new config with: source path, source
#     SHA-256, target SHA-256, the three overridden fields, the intent
#     string, and a timestamp.
#
# Why a config-overridden symlinked dir (Option A) instead of runtime
# argument injection on each eval driver (Option B):
#   - Zero eval-code changes. Every existing sbatch (run_native_greek_mcq_eval,
#     run_eval, run_tokenizer_fair_metrics) takes a single MODEL_PATH /
#     MODEL_SPEC argument; HF loads its config.json. No driver has a
#     rope_theta or max_position override flag today, and threading one
#     through three independent scripts (plus the underlying lm-eval-harness
#     hf-LM wrapper) would touch each one's argparse + INNER block.
#   - Reproducible. The matched config is materialized to disk with a SHA;
#     anyone can `diff` it against the source and verify exactly the three
#     overridden keys.
#   - Cheap. Symlinks for the four safetensors shards + tokenizer artifacts;
#     only config.json is a real file.
#
# Idempotent: rerunning the script regenerates the config + manifest but
# leaves the safetensors symlinks in place (ln -sfn).
#
# This script MUST be run on Clariden (paths are absolute IO/storage
# locations). It is safe to dry-run with `DRY_RUN=1 bash build_...sh`.

set -euo pipefail

SOURCE_DIR="${SOURCE_DIR:-/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509}"
TARGET_DIR="${TARGET_DIR:-/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509-matched-rope500k-seq4096}"

# Three matched-geometry overrides.
TARGET_ROPE_THETA="${TARGET_ROPE_THETA:-500000}"
TARGET_MAX_POSITION="${TARGET_MAX_POSITION:-4096}"
# rope_scaling=null is enforced unconditionally below.

# Intent string baked into the manifest.
INTENT="${INTENT:-Apples-to-apples baseline for 04 Vanilla CPT comparison. Apertus-Base re-exposed under the same short-context geometry (rope_theta=500000, max_position_embeddings=4096, rope_scaling=null) as the CPT run, which trains under paper §2.3 initial-pretraining geometry, not §2.5 long-context extension geometry. Removes the positional-geometry confound from Vanilla-1B/2B/3.5B/5B vs Apertus-Base deltas on native MCQ, retention, and BPB.}"

DRY_RUN="${DRY_RUN:-0}"

echo "=== build_apertus_base_matched_config.sh ==="
date -u
echo "SOURCE_DIR:          $SOURCE_DIR"
echo "TARGET_DIR:          $TARGET_DIR"
echo "TARGET_ROPE_THETA:   $TARGET_ROPE_THETA"
echo "TARGET_MAX_POSITION: $TARGET_MAX_POSITION"
echo "DRY_RUN:             $DRY_RUN"
echo

test -d "$SOURCE_DIR" || { echo "ERROR: SOURCE_DIR does not exist: $SOURCE_DIR" >&2; exit 2; }
test -f "$SOURCE_DIR/config.json" || { echo "ERROR: missing $SOURCE_DIR/config.json" >&2; exit 2; }

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY_RUN: %s\n' "$*"
  else
    eval "$@"
  fi
}

# Compute source SHA-256 BEFORE any writes.
SOURCE_CONFIG_SHA="$(sha256sum "$SOURCE_DIR/config.json" | awk '{print $1}')"
echo "SOURCE config.json SHA-256: $SOURCE_CONFIG_SHA"

run "mkdir -p '$TARGET_DIR'"

# Symlink every non-config file from SOURCE_DIR into TARGET_DIR.
# Includes:
#   - model.safetensors.index.json
#   - model-0000?-of-0000?.safetensors  (the four shards)
#   - tokenizer.json, tokenizer_config.json, special_tokens_map.json
#   - generation_config.json
#   - LICENSE.txt, USAGE_POLICY.md, README.md
# Excludes:
#   - config.json (rewritten below)
#   - MANIFEST.json (newly written below)
echo
echo "Symlinking source files (excluding config.json, MANIFEST.json):"
while IFS= read -r -d '' src; do
  base="$(basename "$src")"
  case "$base" in
    config.json|MANIFEST.json) continue ;;
  esac
  echo "  $base"
  run "ln -sfn '$src' '$TARGET_DIR/$base'"
done < <(find "$SOURCE_DIR" -maxdepth 1 -mindepth 1 -print0)

# Write the matched config.json. Python is the deterministic JSON writer:
# it preserves int types (rope_theta stays int, not "500000.0") and gives
# us a stable sort_keys=False round-trip.
echo
echo "Writing matched config.json with overrides:"
echo "  rope_theta              = $TARGET_ROPE_THETA"
echo "  max_position_embeddings = $TARGET_MAX_POSITION"
echo "  rope_scaling            = null"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN: would write $TARGET_DIR/config.json"
else
  python3 - "$SOURCE_DIR/config.json" "$TARGET_DIR/config.json" \
    "$TARGET_ROPE_THETA" "$TARGET_MAX_POSITION" <<'PY'
import json
import sys
from pathlib import Path

src_path = Path(sys.argv[1])
dst_path = Path(sys.argv[2])
target_rope_theta = int(sys.argv[3])
target_max_position = int(sys.argv[4])

cfg = json.loads(src_path.read_text())

before = {
    "rope_theta": cfg.get("rope_theta"),
    "max_position_embeddings": cfg.get("max_position_embeddings"),
    "rope_scaling": cfg.get("rope_scaling"),
}

cfg["rope_theta"] = target_rope_theta
cfg["max_position_embeddings"] = target_max_position
cfg["rope_scaling"] = None

# Preserve original key order, but ensure the three overridden keys are
# present (they already are in the Apertus-8B-2509 config; the assignment
# above mutates them in-place).
dst_path.write_text(json.dumps(cfg, indent=2) + "\n")

print(f"OVERRIDES before -> after:")
for k, v in before.items():
    print(f"  {k:25s} {v!r:60s} -> {cfg[k]!r}")
PY
fi

# Write the manifest with source SHA, target SHA, intent, timestamp.
if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN: would write $TARGET_DIR/MANIFEST.json"
else
  TARGET_CONFIG_SHA="$(sha256sum "$TARGET_DIR/config.json" | awk '{print $1}')"
  echo "TARGET config.json SHA-256: $TARGET_CONFIG_SHA"

  python3 - "$SOURCE_DIR" "$TARGET_DIR" \
    "$SOURCE_CONFIG_SHA" "$TARGET_CONFIG_SHA" \
    "$TARGET_ROPE_THETA" "$TARGET_MAX_POSITION" "$INTENT" <<'PY'
import json
import os
import sys
import socket
import time
from pathlib import Path

(src_dir, dst_dir, src_sha, dst_sha,
 rope_theta, max_position, intent) = sys.argv[1:]

manifest = {
    "kind": "apertus_base_matched_config_manifest",
    "schema_version": 1,
    "intent": intent,
    "source": {
        "dir": src_dir,
        "config_sha256": src_sha,
    },
    "target": {
        "dir": dst_dir,
        "config_sha256": dst_sha,
        "overrides": {
            "rope_theta": int(rope_theta),
            "max_position_embeddings": int(max_position),
            "rope_scaling": None,
        },
        "fields_unchanged_from_source": [
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "hidden_act",
            "qk_norm",
            "attention_bias",
            "tie_word_embeddings",
            "rms_norm_eps",
            "dtype",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "architectures",
            "model_type",
        ],
    },
    "weight_files": "symlinked from source (model-*.safetensors, *.index.json)",
    "tokenizer_files": "symlinked from source (tokenizer*.json, special_tokens_map.json)",
    "host": socket.gethostname(),
    "build_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "rationale_doc": (
        "subprojects/04_cpt_training_regime_on_vanilla/reports/"
        "config_geometry_audit_iter_0000119.md"
    ),
    "adversarial_critique_finding": (
        "subprojects/04_cpt_training_regime_on_vanilla/adversarial_reviews/"
        "Vanilla-1B/adversarial_critique.md Critical-2 (RoPE/seqlen mismatch)"
    ),
}

Path(os.path.join(dst_dir, "MANIFEST.json")).write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print("wrote MANIFEST.json")
PY
fi

echo
echo "=== matched-config build complete ==="
date -u
echo "TARGET_DIR contents:"
ls -la "$TARGET_DIR" || true
echo
echo "Next: submit evals against MODEL_PATH=$TARGET_DIR via:"
echo "  sbatch scripts/eval_apertus_base_matched_config.sbatch"
