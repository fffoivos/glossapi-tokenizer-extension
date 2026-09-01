#!/usr/bin/env bash
# Freeze a new targeted scientific bundle by overlaying this subproject on the
# already proven full-8B scientific/runtime bundle. Transfers source only.
set -euo pipefail

[[ "$#" == 1 || "$#" == 2 ]] || {
  echo "usage: $0 REMOTE_ROOT [PROVEN_BASE_ROOT]" >&2
  exit 2
}
remote_root=$1
proven_base=${2:-/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260811T201000Z-targeted8b-v14}
case "$remote_root" in
  /iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/*) ;;
  *) echo "refusing unexpected remote bundle root: $remote_root" >&2; exit 2 ;;
esac
case "$proven_base" in
  /iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/*|/iopsstor/scratch/cscs/fffoivos/orchestration/full8b-cpt/*) ;;
  *) echo "refusing unexpected proven base root: $proven_base" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "$0")/../../.." && pwd -P)
relative=subprojects/08_targeted_8b_cpt_experiments
native_eval_worktree=${H2G_NATIVE_EVAL_WORKTREE:-/Users/foivoskarounos-zamparloukos/Projects/.codex-worktrees/train-apertus-full8-results}
native_eval_relative=subprojects/09_full_8b_cpt_results_analysis/evaluation
native_eval_revision=2a7eb9d8de342129f379575ec031f631cde304bc
native_audit_relative=subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts
native_audit_files=(
  build_decontamination_queries.py
  audit_benchmark_contamination_parquet.py
)
legacy_eval_revision=cfdd0e7b00761a736be660867bf3d09733e24a92
legacy_eval_relative=subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval
legacy_eval_files=(
  run_native_greek_mcq_eval.py
  run_native_greek_mcq_eval.sbatch
  native_greek_benchmark_registry.json
)
query_builder_revision=86c1b8fe362233bba6e4e2ca92eb4535287fb240
query_builder_root=frozen_greekmmlu_query_builder
query_builder_sources=(
  subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/build_decontamination_queries.py
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/native_greek_benchmark_registry.json
)
historical_tokenizer_source=$repo_root/subprojects/03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_only_148480
pii_masker_source=$repo_root/subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/scripts/pii_masker.py
pii_masker_sha256=8f489a175aeb47f2c0996431a9d1c6f93ec03d4f52d9ea33621b76facfc0e83c
init_bakeoff_root=$repo_root/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff
td_tools_source=$init_bakeoff_root/token_distillation
megatron_patches_source=$init_bakeoff_root/megatron_patches
historical_td_manifest_source=$td_tools_source/full_td_20260523T092602Z/layer11/retok_td_manifest.json
historical_td_preservation_source=$td_tools_source/full_td_20260523T092602Z/layer11/td_preservation_report.json
retok_source=$init_bakeoff_root/arms/retok.py
retok_common_source=$init_bakeoff_root/arms/_common.py
train_common_source=$init_bakeoff_root/bakeoff_training/_train_config_common.env
trainer_source=$init_bakeoff_root/bakeoff_training/bakeoff_train.sbatch
runtime_guard_source=$megatron_patches_source/runtime/pretrain_gpt_te_guard.py
common_cpt_source=$repo_root/subprojects/05_token_distillation_cpt/03_training_experiments/configs/common_cpt.env
historical_lr_decision_source=$repo_root/subprojects/05_token_distillation_cpt/PRODUCTION_LR_DECISION_20260613.md
extra_valid_patch_source=$repo_root/subprojects/06_dataset_scheduling_experiments/training/runtime_patches/megatron_extra_valid_c92402e.patch
extra_valid_patch_sha256=2e6810fa8b6c25597ccb3bcb9dc1ff5bf843ead2337e3edde0344605a23ec4c6
[[ -d "$historical_tokenizer_source" ]] || {
  echo "historical tokenizer source is missing: $historical_tokenizer_source" >&2
  exit 2
}
for init_source in "$td_tools_source/td_coverage_prepass.py" "$td_tools_source/train_retok_td.py" \
  "$td_tools_source/external/token-distillation" "$megatron_patches_source" \
  "$historical_td_manifest_source" "$historical_td_preservation_source" "$retok_source" "$retok_common_source" "$train_common_source"; do
  [[ -e "$init_source" ]] || { echo "initialization source is missing: $init_source" >&2; exit 2; }
done
for training_source in "$common_cpt_source" "$historical_lr_decision_source"; do
  [[ -f "$training_source" ]] || { echo "training source is missing: $training_source" >&2; exit 2; }
done
for training_source in "$trainer_source" "$runtime_guard_source" "$extra_valid_patch_source"; do
  [[ -f "$training_source" ]] || { echo "training runtime source is missing: $training_source" >&2; exit 2; }
done
for init_relative in \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation/td_coverage_prepass.py \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation/train_retok_td.py \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation/external/token-distillation \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation/full_td_20260523T092602Z/layer11/retok_td_manifest.json \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation/full_td_20260523T092602Z/layer11/td_preservation_report.json \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/arms/retok.py \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/arms/_common.py \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/megatron_patches \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training/_train_config_common.env; do
  git -C "$repo_root" diff --quiet -- "$init_relative" || {
    echo "initialization source is dirty: $init_relative" >&2; exit 2;
  }
done
for training_relative in \
  subprojects/05_token_distillation_cpt/PRODUCTION_LR_DECISION_20260613.md \
  subprojects/05_token_distillation_cpt/03_training_experiments/configs/common_cpt.env \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training/bakeoff_train.sbatch \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/megatron_patches/runtime/pretrain_gpt_te_guard.py \
  subprojects/06_dataset_scheduling_experiments/training/runtime_patches/megatron_extra_valid_c92402e.patch; do
  git -C "$repo_root" diff --quiet -- "$training_relative" || {
    echo "training source is dirty: $training_relative" >&2; exit 2;
  }
done
[[ "$(shasum -a 256 "$extra_valid_patch_source" | awk '{print $1}')" == "$extra_valid_patch_sha256" ]] || {
  echo "named extra-validation patch drift: $extra_valid_patch_source" >&2
  exit 2
}
[[ "$(shasum -a 256 "$pii_masker_source" | awk '{print $1}')" == "$pii_masker_sha256" ]] || {
  echo "published v2 Stage-B masker drift: $pii_masker_source" >&2
  exit 2
}
[[ -d "$native_eval_worktree/$native_eval_relative" ]] || {
  echo "native-suite evaluation source is missing: $native_eval_worktree/$native_eval_relative" >&2
  exit 2
}
[[ "$(git -C "$native_eval_worktree" rev-parse HEAD)" == "$native_eval_revision" ]] || {
  echo "native-suite worktree revision drift" >&2
  exit 2
}
[[ -z "$(git -C "$native_eval_worktree" status --short -- "$native_eval_relative")" ]] || {
  echo "native-suite evaluation source is dirty" >&2
  exit 2
}
for native_audit_file in "${native_audit_files[@]}"; do
  git -C "$native_eval_worktree" diff --quiet -- "$native_audit_relative/$native_audit_file" || {
    echo "native contamination audit source is dirty: $native_audit_file" >&2
    exit 2
  }
done
legacy_tmp=$(mktemp -d)
cleanup_legacy_tmp() { rm -rf -- "$legacy_tmp"; }
trap cleanup_legacy_tmp EXIT
for legacy_file in "${legacy_eval_files[@]}"; do
  mkdir -p "$legacy_tmp/frozen_legacy_greekmmlu_eval"
  git -C "$repo_root" show "$legacy_eval_revision:$legacy_eval_relative/$legacy_file" \
    > "$legacy_tmp/frozen_legacy_greekmmlu_eval/$legacy_file"
done
mkdir -p "$legacy_tmp/$query_builder_root/eval"
for query_source in "${query_builder_sources[@]}"; do
  case "$query_source" in
    */build_decontamination_queries.py)
      query_target="$legacy_tmp/$query_builder_root/build_decontamination_queries.py"
      ;;
    *)
      query_target="$legacy_tmp/$query_builder_root/eval/$(basename "$query_source")"
      ;;
  esac
  git -C "$repo_root" show "$query_builder_revision:$query_source" > "$query_target"
done
dataset_sources=(
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/mix_builder.py
  subprojects/05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/dataset/make_phase_recipes.py
  subprojects/05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/dataset/split_replay_final_for_lr.py
  subprojects/05_token_distillation_cpt/03_training_experiments/dataset_build/bulk_13b.json
  subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/decontaminate.py
  subprojects/05_token_distillation_cpt/03_training_experiments/dataset_build/hplt_clean.py
  subprojects/06_dataset_scheduling_experiments/dataset/build_five_schedules.py
  subprojects/06_dataset_scheduling_experiments/dataset/build_packing_plan.py
  subprojects/06_dataset_scheduling_experiments/dataset/pack_catalog_bucket.py
  subprojects/06_dataset_scheduling_experiments/dataset/finalize_packed_corpus.py
)
packing_builder_test=subprojects/06_dataset_scheduling_experiments/tests/test_source_local_packing.py
receipt="$remote_root.receipt.json"

ssh -o BatchMode=yes clariden /usr/bin/env \
  REMOTE_ROOT="$remote_root" PROVEN_BASE="$proven_base" RECEIPT="$receipt" \
  bash -s <<'REMOTE'
set -euo pipefail
test -d "$PROVEN_BASE"
test ! -e "$REMOTE_ROOT"
test ! -e "$RECEIPT"
cp -a "$PROVEN_BASE" "$REMOTE_ROOT"
chmod -R u+w "$REMOTE_ROOT"
REMOTE

rsync -a --delete --delete-excluded --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.ruff_cache/' --exclude='.pytest_cache/' \
  "$repo_root/$relative/" "clariden:$remote_root/$relative/"
ssh -o BatchMode=yes clariden mkdir -p "$remote_root/$native_eval_relative"
rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' \
  "$native_eval_worktree/$native_eval_relative/" "clariden:$remote_root/$native_eval_relative/"
mkdir -p "$legacy_tmp/frozen_native_contamination_audit"
for native_audit_file in "${native_audit_files[@]}"; do
  git -C "$native_eval_worktree" show "$native_eval_revision:$native_audit_relative/$native_audit_file" \
    > "$legacy_tmp/frozen_native_contamination_audit/$native_audit_file"
done
rsync -a --delete "$legacy_tmp/frozen_native_contamination_audit/" \
  "clariden:$remote_root/frozen_native_contamination_audit/"
rsync -a --delete "$legacy_tmp/frozen_legacy_greekmmlu_eval/" \
  "clariden:$remote_root/frozen_legacy_greekmmlu_eval/"
rsync -a --delete "$legacy_tmp/$query_builder_root/" \
  "clariden:$remote_root/$query_builder_root/"
rsync -a --delete "$historical_tokenizer_source/" \
  "clariden:$remote_root/frozen_historical_tokenizer_148480/"
ssh -o BatchMode=yes clariden mkdir -p \
  "$remote_root/frozen_td_tools/external" \
  "$remote_root/frozen_init_tools/megatron_patches" \
  "$remote_root/frozen_init_tools/bakeoff_training" \
  "$remote_root/frozen_training_tools/bakeoff_training" \
  "$remote_root/frozen_training_tools/megatron_patches/runtime"
rsync -a "$td_tools_source/td_coverage_prepass.py" \
  "clariden:$remote_root/frozen_td_tools/td_coverage_prepass.py"
rsync -a "$td_tools_source/train_retok_td.py" \
  "clariden:$remote_root/frozen_td_tools/train_retok_td.py"
rsync -a "$retok_source" "clariden:$remote_root/frozen_td_tools/retok.py"
rsync -a "$retok_common_source" "clariden:$remote_root/frozen_td_tools/_common.py"
rsync -a "$historical_td_manifest_source" \
  "clariden:$remote_root/frozen_td_tools/historical_layer11_retok_td_manifest.json"
rsync -a "$historical_td_preservation_source" \
  "clariden:$remote_root/frozen_td_tools/historical_layer11_td_preservation_report.json"
rsync -a --delete "$td_tools_source/external/token-distillation/" \
  "clariden:$remote_root/frozen_td_tools/external/token-distillation/"
rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' "$megatron_patches_source/" \
  "clariden:$remote_root/frozen_init_tools/megatron_patches/"
rsync -a "$train_common_source" \
  "clariden:$remote_root/frozen_init_tools/bakeoff_training/_train_config_common.env"
rsync -a "$common_cpt_source" "clariden:$remote_root/frozen_training_tools/common_cpt.env"
ssh -o BatchMode=yes clariden mkdir -p "$remote_root/subprojects/05_token_distillation_cpt"
rsync -a "$historical_lr_decision_source" \
  "clariden:$remote_root/subprojects/05_token_distillation_cpt/PRODUCTION_LR_DECISION_20260613.md"
rsync -a "$train_common_source" \
  "clariden:$remote_root/frozen_training_tools/bakeoff_training/_train_config_common.env"
rsync -a "$trainer_source" \
  "clariden:$remote_root/frozen_training_tools/bakeoff_training/bakeoff_train.sbatch"
rsync -a "$runtime_guard_source" \
  "clariden:$remote_root/frozen_training_tools/megatron_patches/runtime/pretrain_gpt_te_guard.py"
rsync -a "$extra_valid_patch_source" \
  "clariden:$remote_root/frozen_training_tools/megatron_extra_valid_c92402e.patch"
for source in "${dataset_sources[@]}"; do
  ssh -o BatchMode=yes clariden mkdir -p "$(dirname "$remote_root/$source")"
  rsync -a "$repo_root/$source" "clariden:$remote_root/$source"
done
ssh -o BatchMode=yes clariden mkdir -p "$remote_root/frozen_historical_dataset_tools"
rsync -a "$repo_root/subprojects/05_token_distillation_cpt/03_training_experiments/dataset_build/hplt_clean.py" \
  "clariden:$remote_root/frozen_historical_dataset_tools/hplt_clean.py"
rsync -a "$pii_masker_source" "clariden:$remote_root/frozen_historical_dataset_tools/pii_masker.py"
rsync -a "$repo_root/$packing_builder_test" "clariden:$remote_root/$packing_builder_test"

ssh -o BatchMode=yes clariden /usr/bin/env \
  REMOTE_ROOT="$remote_root" RECEIPT="$receipt" \
  bash -s <<'REMOTE'
set -euo pipefail
target_train="$REMOTE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch"
if ! grep -q 'FULL8_BENCHMARK_BASE_ITERATION' "$target_train"; then
  patch -d "$REMOTE_ROOT" -p1 --forward --batch \
    < "$REMOTE_ROOT/subprojects/08_targeted_8b_cpt_experiments/patches/train_segment_targeted_benchmark_offset.patch"
fi
target_bakeoff="$REMOTE_ROOT/frozen_training_tools/bakeoff_training/bakeoff_train.sbatch"
/usr/bin/python3.11 \
  "$REMOTE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/patch_bakeoff_scale_geometry.py" \
  --trainer "$target_bakeoff"
/usr/bin/python3.11 \
  "$REMOTE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/patch_bakeoff_uenv10_srun.py" \
  --trainer "$target_bakeoff"
/usr/bin/python3.11 \
  "$REMOTE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/patch_bakeoff_runtime_compat.py" \
  --trainer "$target_bakeoff"
if find "$REMOTE_ROOT/subprojects/08_targeted_8b_cpt_experiments" -type d \
  \( -name '.ruff_cache' -o -name '.pytest_cache' -o -name '__pycache__' \) \
  -print -quit | grep -q .; then
  echo "runtime or test cache directory entered the candidate subproject" >&2
  exit 2
fi
find "$REMOTE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden" -type f \
  \( -name '*.sh' -o -name '*.sbatch' \) -print0 |
  while IFS= read -r -d '' file; do bash -n "$file"; done
/usr/bin/python3.11 - "$REMOTE_ROOT/subprojects/08_targeted_8b_cpt_experiments" <<'PY'
import ast,sys
from pathlib import Path
root=Path(sys.argv[1])
paths=sorted(root.rglob("*.py"))
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
print({"ok":True,"python_files":len(paths)})
PY
/usr/bin/python3.11 \
  "$REMOTE_ROOT/subprojects/06_dataset_scheduling_experiments/production/freeze_code_bundle.py" \
  --root "$REMOTE_ROOT" --kind scientific --output "$RECEIPT"
/usr/bin/python3.11 \
  "$REMOTE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$REMOTE_ROOT" --receipt "$RECEIPT" --kind scientific
chmod -R a-w "$REMOTE_ROOT"
REMOTE
printf '%s\n%s\n' "$remote_root" "$receipt"
