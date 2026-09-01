#!/usr/bin/env bash
#SBATCH --job-name=h2g-data-authorities
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_PRODUCER_COMPATIBILITY:?set exact current producer compatibility receipt}"
: "${H2G_PHASE1_CACHE_RECEIPT:?set exact clean Phase-1 cache receipt}"
: "${H2G_PHASE2_CACHE_RECEIPT:?set exact clean Phase-2 cache receipt}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "data authorities require one debug node" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
r="$H2G_STAGE_ROOT/receipts"
out="$r/pre_main_data_authorities"
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 "$subproject/scripts/freeze_pre_main_data_authorities.py" \
  --query-receipt "$r/greekmmlu_query_regeneration.json" --native-query-authority "$r/native_suite_scan_authority.json" \
  --source-views "$r/source_views.json" \
  --hplt-selected "$r/hplt_selected_raw.manifest.json" --openarchives-selected "$r/openarchives_selected_raw.manifest.json" \
  --hplt-prepared "$r/hplt_prepared_stream.json" --openarchives-prepared "$r/openarchives_prepared_stream.json" \
  --hplt-stage-b "$r/hplt_stage_b.json" --openarchives-stage-b "$r/openarchives_stage_b.json" \
  --hplt-post-greekmmlu "$r/greekmmlu_scan_hplt_post.json" --openarchives-post-greekmmlu "$r/greekmmlu_scan_openarchives_post.json" \
  --replay-selected "$r/replay_selected_raw.manifest.json" --replay-validation-filter "$r/replay_validation_exclusion.json" \
  --replay-scan-input "$r/replay_scan_input.json" --replay-native-filter "$r/replay_native_filter.json" \
  --replay-greekmmlu-filter "$r/greekmmlu_scan_replay_selected.json" --replay-stage-b "$r/replay_selected_stage_b.json" \
  --replay-post-greekmmlu "$r/greekmmlu_scan_replay_selected_post.json" --replay-post-native "$r/native_suite_replay_scan_post_filter.json" \
  --replay-split "$r/replay_split.json" --validation-panels "$r/reused_validation_panels.json" \
  --hplt-tokenized "$r/tokenized_hplt.json" --openarchives-tokenized "$r/tokenized_openarchives.json" \
  --foreign-tokenized "$r/tokenized_foreign.json" --old-greek-tokenized "$r/tokenized_old_greek.json" \
  --phase1-cache "$H2G_PHASE1_CACHE_RECEIPT" --phase2-cache "$H2G_PHASE2_CACHE_RECEIPT" \
  --producer-compatibility "$H2G_PRODUCER_COMPATIBILITY" \
  --output-dir "$out"
