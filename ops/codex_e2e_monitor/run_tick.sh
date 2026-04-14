#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_ROOT="/home/foivos/data/glossapi_work/logs/codex_e2e_monitor"
LOCK_FILE="${STATE_ROOT}/tick.lock"
THREAD_FILE="${STATE_ROOT}/thread_id"
LAST_MESSAGE_FILE="${STATE_ROOT}/last_message.txt"
LATEST_LOG="${STATE_ROOT}/latest.jsonl"
RUNS_DIR="${STATE_ROOT}/runs"
NOTES_DIR="${STATE_ROOT}/notes"
LATEST_NOTE="${STATE_ROOT}/latest.md"
INITIAL_PROMPT_FILE="${SCRIPT_DIR}/initial_prompt.md"
RESUME_PROMPT_FILE="${SCRIPT_DIR}/resume_prompt.md"

mkdir -p "${STATE_ROOT}" "${RUNS_DIR}" "${NOTES_DIR}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "monitor tick skipped: lock held"
  exit 0
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="${RUNS_DIR}/${STAMP}.jsonl"
STATUS_JSON="${RUNS_DIR}/${STAMP}.status.json"
NOTE_MD="${NOTES_DIR}/${STAMP}.md"

MODEL="gpt-5.4"
REASONING_CFG='reasoning_effort="xhigh"'
CODEX_BIN="/home/foivos/.npm-global/bin/codex"

cd "${REPO_ROOT}"

if [ -f "${THREAD_FILE}" ]; then
  THREAD_ID="$(cat "${THREAD_FILE}")"
  PROMPT="$(cat "${RESUME_PROMPT_FILE}")"
  MODE="resume"
  CMD=(
    "${CODEX_BIN}" exec resume
    "${THREAD_ID}"
    -m "${MODEL}"
    -c "${REASONING_CFG}"
    --dangerously-bypass-approvals-and-sandbox
    --json
    -o "${LAST_MESSAGE_FILE}"
    "${PROMPT}"
  )
else
  PROMPT="$(cat "${INITIAL_PROMPT_FILE}")"
  MODE="initial"
  CMD=(
    "${CODEX_BIN}" exec
    -m "${MODEL}"
    -c "${REASONING_CFG}"
    --dangerously-bypass-approvals-and-sandbox
    --json
    -o "${LAST_MESSAGE_FILE}"
    "${PROMPT}"
  )
fi

set +e
timeout 13m "${CMD[@]}" | tee "${RUN_LOG}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

cp "${RUN_LOG}" "${LATEST_LOG}"

THREAD_ID_VALUE=""
if grep -q '"type":"thread.started"' "${RUN_LOG}" 2>/dev/null; then
  THREAD_ID_VALUE="$(python3 - "${RUN_LOG}" <<'PY'
import json
import sys
from pathlib import Path

for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    try:
        payload = json.loads(line)
    except Exception:
        continue
    if payload.get("type") == "thread.started":
        print(payload.get("thread_id", ""))
        break
PY
)"
elif [ -f "${THREAD_FILE}" ]; then
  THREAD_ID_VALUE="$(cat "${THREAD_FILE}")"
fi

if [ -n "${THREAD_ID_VALUE}" ]; then
  printf '%s\n' "${THREAD_ID_VALUE}" > "${THREAD_FILE}"
fi

python3 - "${STATUS_JSON}" "${MODE}" "${EXIT_CODE}" "${THREAD_ID_VALUE}" "${RUN_LOG}" <<'PY'
import json
import sys
from datetime import datetime, timezone

out = sys.argv[1]
payload = {
    "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "mode": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "thread_id": sys.argv[4] or None,
    "run_log": sys.argv[5],
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps(payload, ensure_ascii=False))
PY

python3 - "${NOTE_MD}" "${STAMP}" "${MODE}" "${EXIT_CODE}" "${THREAD_ID_VALUE}" "${RUN_LOG}" "${LAST_MESSAGE_FILE}" <<'PY'
import sys
from pathlib import Path

note_path = Path(sys.argv[1])
stamp = sys.argv[2]
mode = sys.argv[3]
exit_code = sys.argv[4]
thread_id = sys.argv[5] or "unknown"
run_log = sys.argv[6]
last_message_path = Path(sys.argv[7])
message = ""
if last_message_path.exists():
    message = last_message_path.read_text(encoding="utf-8", errors="replace").strip()

header = [
    f"# Codex Monitor Tick {stamp}",
    "",
    f"- mode: `{mode}`",
    f"- exit_code: `{exit_code}`",
    f"- thread_id: `{thread_id}`",
    f"- run_log: `{run_log}`",
    "",
    "## Findings",
    "",
]

if message:
    body = message
else:
    body = "_No final monitor message was captured for this tick._"

note_path.write_text("\n".join(header) + body + "\n", encoding="utf-8")
PY

cp "${NOTE_MD}" "${LATEST_NOTE}"
