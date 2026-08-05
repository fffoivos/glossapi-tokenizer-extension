#!/usr/bin/env bash
set -euo pipefail

: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${NEUTRAL_RUN_ROOT:?set new neutral pipeline root}"
CONFIRM_NEUTRAL_BUILD=${CONFIRM_NEUTRAL_BUILD:-}
[[ "$CONFIRM_NEUTRAL_BUILD" == GREEK_PARLIAMENT_2587904 ]] || {
  echo "live neutral build requires CONFIRM_NEUTRAL_BUILD=GREEK_PARLIAMENT_2587904" >&2
  exit 2
}
AGENT1_RUN_ROOT=${AGENT1_RUN_ROOT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/agent1-v5-clariden-debug-20260715T111552Z-30c72e9}
POOL_CORPUS_RECEIPT=${POOL_CORPUS_RECEIPT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/dataset_scheduling_0p5b/20260802T104539Z-mini-schedule-v1/pool_corpus_receipt.json}
TOKENIZER_DIR=${TOKENIZER_DIR:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_v1_1_0p5b_greek_overlay_fcd33ec}
MAX_PARALLEL_BUCKETS=${MAX_PARALLEL_BUCKETS:-2}
[[ "$MAX_PARALLEL_BUCKETS" =~ ^[1-2]$ ]] || { echo "MAX_PARALLEL_BUCKETS must be 1 or 2 to bound normal-node I/O" >&2; exit 2; }
[[ ! -e "$NEUTRAL_RUN_ROOT" ]] || { echo "refusing to replace $NEUTRAL_RUN_ROOT" >&2; exit 2; }
for required in "$SCIENTIFIC_BUNDLE" "$AGENT1_RUN_ROOT" "$POOL_CORPUS_RECEIPT" "$TOKENIZER_DIR"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 2; }
done
mkdir -p "$NEUTRAL_RUN_ROOT/logs" "$NEUTRAL_RUN_ROOT/submissions"
exports="ALL,SCIENTIFIC_BUNDLE=$SCIENTIFIC_BUNDLE,NEUTRAL_RUN_ROOT=$NEUTRAL_RUN_ROOT,AGENT1_RUN_ROOT=$AGENT1_RUN_ROOT,POOL_CORPUS_RECEIPT=$POOL_CORPUS_RECEIPT,TOKENIZER_DIR=$TOKENIZER_DIR"
source_job=$(sbatch --parsable --export="$exports" \
  --output="$NEUTRAL_RUN_ROOT/logs/%x-%j.out" --error="$NEUTRAL_RUN_ROOT/logs/%x-%j.err" \
  "$SCIENTIFIC_BUNDLE/clariden/prepare_neutral_external_source.sbatch")
signature_job=$(sbatch --parsable --dependency="afterok:$source_job" --export="$exports" \
  --output="$NEUTRAL_RUN_ROOT/logs/%x-%j.out" --error="$NEUTRAL_RUN_ROOT/logs/%x-%j.err" \
  "$SCIENTIFIC_BUNDLE/clariden/build_neutral_candidate_signatures.sbatch")
match_job=$(sbatch --parsable --dependency="afterok:$signature_job" \
  --array="0-31%$MAX_PARALLEL_BUCKETS" --export="$exports" \
  --output="$NEUTRAL_RUN_ROOT/logs/%x-%A_%a.out" --error="$NEUTRAL_RUN_ROOT/logs/%x-%A_%a.err" \
  "$SCIENTIFIC_BUNDLE/clariden/match_neutral_minhash_bucket.sbatch")
dedup_job=$(sbatch --parsable --dependency="afterok:$match_job" --export="$exports" \
  --output="$NEUTRAL_RUN_ROOT/logs/%x-%j.out" --error="$NEUTRAL_RUN_ROOT/logs/%x-%j.err" \
  "$SCIENTIFIC_BUNDLE/clariden/finalize_neutral_cross_dedup.sbatch")
heldout_job=$(sbatch --parsable --dependency="afterok:$dedup_job" \
  --export="$exports,NEUTRAL_DEDUP_RECEIPT=$NEUTRAL_RUN_ROOT/final-cross-dedup/neutral_external_dedup_receipt.json,NEUTRAL_OUTPUT_ROOT=$NEUTRAL_RUN_ROOT/heldout" \
  --output="$NEUTRAL_RUN_ROOT/logs/%x-%j.out" --error="$NEUTRAL_RUN_ROOT/logs/%x-%j.err" \
  "$SCIENTIFIC_BUNDLE/clariden/build_neutral_external_heldout.sbatch")
python3 - "$NEUTRAL_RUN_ROOT/submissions/neutral_external_pipeline.json" "$source_job" "$signature_job" "$match_job" "$dedup_job" "$heldout_job" <<'PY'
import datetime as dt,json,os,sys
out,source,signatures,matches,dedup,heldout=sys.argv[1:]
payload={
 "schema_version":"apertus_mini_neutral_external_submission_v1",
 "status":"submitted",
 "created_at":dt.datetime.now(dt.timezone.utc).isoformat(),
 "source":{"zenodo_record":2587904,"license":"CC-BY-4.0","cluster_unit":"complete_parliamentary_sitting"},
 "run_root":os.environ["NEUTRAL_RUN_ROOT"],
 "scientific_bundle":os.environ["SCIENTIFIC_BUNDLE"],
 "jobs":{"source_snapshot":source,"candidate_signatures":signatures,"minhash_bucket_array":matches,"cross_dedup_finalizer":dedup,"heldout_builder":heldout},
 "expected":{"minhash_buckets":32,"target_tokens_including_eos":15000000,"token_range":[10000000,20000000]},
}
open(out,"x",encoding="utf-8").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
printf 'source=%s\nsignatures=%s\nmatches=%s\ndedup=%s\nheldout=%s\n' \
  "$source_job" "$signature_job" "$match_job" "$dedup_job" "$heldout_job"
