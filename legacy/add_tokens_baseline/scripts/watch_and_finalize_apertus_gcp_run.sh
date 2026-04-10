#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-eellak-glossapi-20251008}"
ZONE="${ZONE:-europe-west4-b}"
INSTANCE="${INSTANCE:?INSTANCE is required}"
REMOTE_ROOT="${REMOTE_ROOT:?REMOTE_ROOT is required}"
LOCAL_ROOT="${LOCAL_ROOT:?LOCAL_ROOT is required}"
POLL_SECONDS="${POLL_SECONDS:-600}"

mkdir -p "${LOCAL_ROOT}"
LOG="${LOCAL_ROOT}/finalize.log"
exec >>"${LOG}" 2>&1

echo "[$(date -Is)] finisher started"

while true; do
  if /home/foivos/google-cloud-sdk/bin/gcloud compute ssh "${INSTANCE}" --project "${PROJECT_ID}" --zone "${ZONE}" --command "test -f '${REMOTE_ROOT}/outputs/nanochat_plus_hplt/summary.json'"; then
    break
  fi
  echo "[$(date -Is)] summary not ready yet"
  sleep "${POLL_SECONDS}"
done

echo "[$(date -Is)] summary detected"
/home/foivos/google-cloud-sdk/bin/gcloud compute ssh "${INSTANCE}" --project "${PROJECT_ID}" --zone "${ZONE}" --command "jq '{best_variant: .variant_results.best_variant.size, base_tokens_per_100_chars: .variant_results.base_evaluation.aggregate.text_metrics.tokens_per_100_chars, variants: [.variant_results.variants[] | {size, tokens_per_100_chars: .evaluation.aggregate.text_metrics.tokens_per_100_chars, avg_tokens_per_greek_word: .evaluation.aggregate.greek_word_metrics.avg_tokens_per_greek_word, single_token_share: .evaluation.aggregate.greek_word_metrics.single_token_greek_word_share}] }' '${REMOTE_ROOT}/outputs/nanochat_plus_hplt/summary.json'"

until /home/foivos/google-cloud-sdk/bin/gcloud compute scp --project "${PROJECT_ID}" --zone "${ZONE}" --recurse \
  "${INSTANCE}:${REMOTE_ROOT}/outputs" \
  "${INSTANCE}:${REMOTE_ROOT}/logs" \
  "${LOCAL_ROOT}/"; do
  echo "[$(date -Is)] scp failed, retrying in 60s"
  sleep 60
done

/home/foivos/google-cloud-sdk/bin/gcloud compute instances stop "${INSTANCE}" --project "${PROJECT_ID}" --zone "${ZONE}"
echo "[$(date -Is)] finisher completed"
