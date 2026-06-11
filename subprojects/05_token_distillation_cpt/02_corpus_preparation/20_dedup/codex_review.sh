#!/usr/bin/env bash
# Adversarial review of a pipeline-stage script via `codex exec`, BEFORE it runs.
# Usage: bash codex_review.sh <script.py> [extra-context-string]
# Writes the review to <script>.codex_review.md and prints it.
#
# Strips the codex agent prompt + user config per feedback_codex_strip_instructions
# (-c model_instructions_file + --ignore-user-config), so codex acts purely as
# the adversarial reviewer defined in codex_reviewer_instructions.md.
set -euo pipefail

SCRIPT="${1:?usage: codex_review.sh <script.py> [context]}"
CONTEXT="${2:-}"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${SCRIPT%.*}.codex_review.md"
MODEL="${CODEX_MODEL:-gpt-5.5}"
EFFORT="${CODEX_EFFORT:-high}"

[ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT" >&2; exit 2; }

PROMPT="Adversarially review the script below (path: $SCRIPT).
${CONTEXT:+Extra context: $CONTEXT
}
Find correctness bugs, silent data-loss/wrong-drop risks, parquet/library
misuse, scale/memory problems, reproducibility gaps, and edge cases. Output a
prioritized [BLOCKER]/[MAJOR]/[MINOR] list with concrete fixes.

--- BEGIN SCRIPT ($SCRIPT) ---
$(cat "$SCRIPT")
--- END SCRIPT ---"

echo "[codex_review] reviewing $SCRIPT with $MODEL (effort=$EFFORT)…" >&2
printf '%s' "$PROMPT" | codex exec \
  --ephemeral --ignore-user-config --skip-git-repo-check \
  -m "$MODEL" \
  -c model_reasoning_effort="$EFFORT" \
  -c model_instructions_file="$DIR/codex_reviewer_instructions.md" \
  -o "$OUT" \
  - >/dev/null 2>"$OUT.log" || { echo "[codex_review] codex exec failed; see $OUT.log" >&2; tail -5 "$OUT.log" >&2; exit 1; }

echo "[codex_review] review saved to $OUT" >&2
echo "============================================================"
cat "$OUT"
echo "============================================================"
