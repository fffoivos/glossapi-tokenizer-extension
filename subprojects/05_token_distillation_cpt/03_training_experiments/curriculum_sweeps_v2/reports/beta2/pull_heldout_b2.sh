#!/usr/bin/env bash
# Pull in-training extra-valid heldout losses for the beta2 comparison.
#
# The losses live in the training .out logs on Clariden and are extracted by
# analysis/collect_forgetting_loss.py. This writes heldout_b2.csv locally.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SSH_TARGET="${SSH_TARGET:-clariden}"
V2_REMOTE="${V2_REMOTE:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/curriculum_v2}"
OUT="${OUT:-$HERE/heldout_b2.csv}"

RUN_TAGS_DEFAULT=(
  curr_td_b20p99_b3p999_13b_20260616T093527Z
  curr_td_f20_g1_lr5.5e-5_a4_20260613T090652Z
  curr_td_b20p999_b3p999_13b_20260616T093527Z
)

if [ "$#" -gt 0 ]; then
  RUN_TAGS=("$@")
else
  RUN_TAGS=("${RUN_TAGS_DEFAULT[@]}")
fi

remote_dirs=()
for tag in "${RUN_TAGS[@]}"; do
  remote_dirs+=("$RUN_ROOT/$tag")
done

tmp_out="${OUT}.tmp"
rm -f "$tmp_out"

printf 'pulling heldout loss via %s\n' "$SSH_TARGET" >&2
printf 'remote collector: %s/analysis/collect_forgetting_loss.py\n' "$V2_REMOTE" >&2
printf 'runs:\n' >&2
printf '  %s\n' "${remote_dirs[@]}" >&2

ssh -o BatchMode=yes -o ConnectTimeout=20 "$SSH_TARGET" \
  "V2='$V2_REMOTE' bash -s" -- "${remote_dirs[@]}" > "$tmp_out" <<'REMOTE'
set -euo pipefail
v2="$V2"
shift 0
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
python3 "$v2/analysis/collect_forgetting_loss.py" "$@" >&2
cat forgetting_loss.csv
REMOTE

python3 - "$tmp_out" <<'PY'
import csv, sys
path = sys.argv[1]
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"no rows parsed from {path}")
tags = sorted({r["run_tag"] for r in rows})
names = sorted({r["name"] for r in rows})
print(f"parsed {len(rows)} rows across {len(tags)} runs and {len(names)} heldout sets", file=sys.stderr)
for tag in tags:
    final = {}
    for r in rows:
        if r["run_tag"] == tag:
            final[r["name"]] = r
    missing = [n for n in names if n not in final]
    if missing:
        raise SystemExit(f"{tag}: missing final rows for {missing}")
    print(f"  {tag}: final_iter={max(int(r['iter']) for r in final.values())}", file=sys.stderr)
PY

mv "$tmp_out" "$OUT"
printf 'wrote %s\n' "$OUT" >&2
