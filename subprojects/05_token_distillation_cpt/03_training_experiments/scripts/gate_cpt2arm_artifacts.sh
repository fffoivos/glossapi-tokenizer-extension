#!/usr/bin/env bash
# Gate documented artifacts before launching the two-arm 13.5B CPT runs.
set -euo pipefail

SC="${SC:-/iopsstor/scratch/cscs/fffoivos}"
REPO_ROOT="${REPO_ROOT:-$SC/repo/glossapi-tokenizer-extension}"
EXP_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt/03_training_experiments"
STAGE="${STAGE:-$SC/cpt_corpus/cpt_2arm_13b}"
MEGATRON_STAGE="$STAGE/megatron"
TRAIN_DIR="${TRAIN_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training}"
MEGATRON_DIR="${MEGATRON_DIR:-$SC/code/training/Megatron-LM-Swiss-AI}"

COMMON_ENV="$EXP_DIR/configs/common_cpt.env"
ARM1_ENV="$EXP_DIR/configs/arm1_vanilla.env"
ARM2_ENV="$EXP_DIR/configs/arm2_modern_greek.env"

failures=0

check_file() {
  local path="$1" label="$2"
  if [ -s "$path" ]; then
    printf 'OK   %-40s %s bytes  %s\n' "$label" "$(stat -c%s "$path")" "$path"
  else
    printf 'MISS %-40s %s\n' "$label" "$path" >&2
    failures=$((failures + 1))
  fi
}

check_text_equals() {
  local path="$1" expected="$2" label="$3" got
  if [ ! -f "$path" ]; then
    printf 'MISS %-40s %s\n' "$label" "$path" >&2
    failures=$((failures + 1))
    return
  fi
  got="$(tr -d '[:space:]' < "$path")"
  if [ "$got" = "$expected" ]; then
    printf 'OK   %-40s %s\n' "$label" "$got"
  else
    printf 'BAD  %-40s expected=%s got=%s path=%s\n' "$label" "$expected" "$got" "$path" >&2
    failures=$((failures + 1))
  fi
}

check_prefix() {
  local prefix="$1" label="$2"
  check_file "${prefix}.bin" "$label.bin"
  check_file "${prefix}.idx" "$label.idx"
}

check_grep() {
  local path="$1" pattern="$2" label="$3"
  if grep -q -- "$pattern" "$path"; then
    printf 'OK   %-40s %s\n' "$label" "$path"
  else
    printf 'BAD  %-40s missing pattern=%s path=%s\n' "$label" "$pattern" "$path" >&2
    failures=$((failures + 1))
  fi
}

resolve_config_var() {
  local config="$1" var="$2"
  env -u INIT_CKPT -u BASE_DATA_PREFIX -u EXT_DATA_PREFIX \
    -u BASE_TOKENIZER_DIR -u EXT_TOKENIZER_DIR \
    bash -c 'source "$1" >/dev/null 2>&1; printf "%s\n" "${!2}"' bash "$config" "$var"
}

printf '=== CPT 2-arm artifact gate ===\n'
printf 'SC=%s\nSTAGE=%s\nEXP_DIR=%s\n\n' "$SC" "$STAGE" "$EXP_DIR"

printf '%s\n' '--- config invariants ---'
# shellcheck disable=SC1090
source "$COMMON_ENV"
declare -A expected=(
  [TRAIN_TOKENS]=13500000000
  [TRAIN_ITERS]=3218
  [GLOBAL_BATCH_TOKENS]=4194304
  [LR_PEAK]=5.5e-5
  [LR_FINAL]=5.5e-6
  [LR_WARMUP_ITERS]=400
  [LR_WSD_DECAY_SAMPLES]=659179
  [ADEMA_BETA2]=0.995
  [ADEMA_BETA3]=0.999
  [ADEMA_ALPHA]=4.0
  [SEQ_LENGTH]=4096
  [ROTARY_BASE]=500000
  [USE_ROPE_SCALING]=1
  [MAKE_VOCAB_SIZE_DIVISIBLE_BY]=256
  [LOSS_OBJECTIVE]=goldfish
  [GOLDFISH_K]=50
  [GOLDFISH_H]=50
  [DATA_SEED]=20260609
  [EVAL_INTERVAL]=25
  [EVAL_ITERS]=1
  [NODES]=16
  [LAUNCH_MODE]=torchrun
)
for key in "${!expected[@]}"; do
  if [ "${!key}" = "${expected[$key]}" ]; then
    printf 'OK   %-40s %s\n' "$key" "${!key}"
  else
    printf 'BAD  %-40s expected=%s got=%s\n' "$key" "${expected[$key]}" "${!key}" >&2
    failures=$((failures + 1))
  fi
done

printf '\n%s\n' '--- tokenizer invariants ---'
python3 - "$BASE_TOKENIZER_DIR" "$EXT_TOKENIZER_DIR" <<'PY'
import json
import pathlib
import sys

ok = True
for name, root, expected in [
    ("base", pathlib.Path(sys.argv[1]), 131072),
    ("ext", pathlib.Path(sys.argv[2]), 148480),
]:
    tok = json.load(open(root / "tokenizer.json"))
    cfg = json.load(open(root / "tokenizer_config.json"))
    vocab = tok["model"]["vocab"]
    checks = [
        (len(vocab) == expected, f"vocab={len(vocab)} expected={expected}"),
        (len(vocab) % 256 == 0, f"divisible_by_256={len(vocab) % 256 == 0}"),
        (vocab.get(cfg.get("bos_token", "")) == 1, "bos_id=1"),
        (vocab.get(cfg.get("eos_token", "")) == 2, "eos_id=2"),
        (vocab.get(cfg.get("pad_token", "")) == 3, "pad_id=3"),
    ]
    for passed, msg in checks:
        print(("OK   " if passed else "BAD  ") + f"tokenizer:{name:<29} {msg}")
        ok = ok and passed
sys.exit(0 if ok else 1)
PY

printf '\n%s\n' '--- init checkpoints ---'
VANILLA_INIT="$(resolve_config_var "$ARM1_ENV" INIT_CKPT)"
TD_INIT="$(resolve_config_var "$ARM2_ENV" INIT_CKPT)"
for label in vanilla td; do
  init_var="${label^^}_INIT"
  init="${!init_var}"
  check_text_equals "$init/latest_checkpointed_iteration.txt" release "$label latest"
  check_file "$init/release/mp_rank_00/model_optim_rng.pt" "$label rank00"
  check_file "$init/release/mp_rank_01/model_optim_rng.pt" "$label rank01"
done

printf '\n%s\n' '--- trainer/runtime patches ---'
check_file "$TRAIN_DIR/../megatron_patches/runtime/pretrain_gpt_te_guard.py" "te guard wrapper"
check_grep "$TRAIN_DIR/bakeoff_train.sbatch" "NCCL_NET=.*AWS Libfabric" "nccl aws-libfabric env"
check_grep "$TRAIN_DIR/bakeoff_train.sbatch" "FI_CXI_DEFAULT_CQ_SIZE" "nccl cxi cq env"
check_grep "$TRAIN_DIR/bakeoff_train.sbatch" "FI_CXI_RDZV_EAGER_SIZE" "nccl eager workaround"
check_grep "$TRAIN_DIR/bakeoff_train.sbatch" "RUNTIME_ENV_PREFIX" "runtime env reapplied inside uenv"
if [ "${NCCL_NET:-}" = "Socket" ]; then
  check_grep "$TRAIN_DIR/bakeoff_train.sbatch" "NCCL_SOCKET_IFNAME" "socket iface preserved inside uenv"
fi
check_grep "$MEGATRON_DIR/megatron/training/training.py" "validation_name = None" "extra-valid tb prefix"
check_grep "$MEGATRON_DIR/megatron/training/training.py" "tb_key = '\\[{}\\] {}'.format(validation_name, key)" "extra-valid tb key"

printf '\n%s\n' '--- held-out validation binaries ---'
for tok in base ext; do
  for val in hplt openarchives greek_phd; do
    check_prefix "$MEGATRON_STAGE/val_${val}_${tok}_text_document" "val $val $tok"
  done
done

printf '\n%s\n' '--- full training binaries ---'
check_file "$STAGE/bulk_mix_final.jsonl" "bulk final jsonl"
check_file "$STAGE/bulk_mix_ordered_replay_fixed_final.jsonl" "bulk ordered replay-fixed jsonl"
check_file "$STAGE/bulk_mix_ordered_replay_fixed_manifest.json" "bulk ordered manifest"
if [ -s "$STAGE/bulk_mix_ordered_replay_fixed_manifest.json" ]; then
  if ! python3 - "$STAGE/bulk_mix_ordered_replay_fixed_manifest.json" <<'PY'
import json
import sys
path = sys.argv[1]
m = json.load(open(path))
checks = [
    ("non_new_positions_preserved", m["replay_preservation"]["non_new_positions_preserved"] is True),
    ("ordered_violation_count", m["new_greek_order"]["ordered_violation_count"] == 0),
    ("hplt_before_openarchives", m["new_greek_order"]["last_hplt_new_greek_slot"] < m["new_greek_order"]["first_openarchives_new_greek_slot"]),
]
ok = True
for label, passed in checks:
    print(("OK   " if passed else "BAD  ") + f"ordered_manifest:{label:<23} {path}")
    ok = ok and passed
sys.exit(0 if ok else 1)
PY
  then
    failures=$((failures + 1))
  fi
fi
check_prefix "$MEGATRON_STAGE/bulk_mix_ordered_replay_fixed_base_text_document" "bulk ordered base"
check_prefix "$MEGATRON_STAGE/bulk_mix_ordered_replay_fixed_ext_text_document" "bulk ordered ext"

printf '\n'
if [ "$failures" -eq 0 ]; then
  echo "ARTIFACT GATE PASSED"
else
  echo "ARTIFACT GATE FAILED: $failures issue(s)" >&2
  exit 1
fi
