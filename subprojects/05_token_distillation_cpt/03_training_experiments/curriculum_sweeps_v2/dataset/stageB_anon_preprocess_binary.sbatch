#!/usr/bin/env bash
#SBATCH --job-name=v2_B_anon_preprocess
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%A_%a.err
# array 0-2. Per-binary StageB: PII anonymize (split into 64 line-shards first — datatrove shards
# by FILE) then Megatron preprocess_data.py x2 (base 131072 + ext 148480). Mirrors stageB_anon_preprocess.sbatch.
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "$V2_DIR/paths.env" ] && [ -f "$V2_DIR/../paths.env" ]; then
  V2_DIR="$(cd "$V2_DIR/.." && pwd)"
fi
source "$V2_DIR/paths.env"

BINS=(hplt_only glossapi_only replay_only)
B=${BINS[${SLURM_ARRAY_TASK_ID:-0}]}
test -s "$STAGE/${B}_decontam.jsonl"; check_build_py
mkdir -p "$STAGE/${B}_indir" "$MEGOUT"

# datatrove shards by FILE -> 64 line-shards so all 64 tasks work
rm -f "$STAGE/${B}_indir"/part_*.jsonl
rm -rf "$STAGE/${B}_masked"
split -n l/64 -d -a 2 --additional-suffix=.jsonl "$STAGE/${B}_decontam.jsonl" "$STAGE/${B}_indir/part_"

echo "=== anonymize (email/IP/IBAN) $B ==="
run_build_py -u "$ANON" \
  --input "$STAGE/${B}_indir" --reader jsonl --glob '*.jsonl' \
  --text-key text --id-key doc_id \
  --output "$STAGE/${B}_masked" --writer jsonl --tasks 64
# datatrove writes gzipped ${rank}.jsonl.gz
zcat "$STAGE/${B}_masked"/*.jsonl.gz > "$STAGE/${B}_final.jsonl"
echo "masked docs ($B): $(wc -l < "$STAGE/${B}_final.jsonl")"

cd "$MEG"
for pair in "base $BASE_TOK" "ext $EXT_TOK"; do
  set -- $pair; TAG=$1; TOK=$2
  echo "=== tokenize $B $TAG ==="
  rm -f "$MEGOUT/${B}_${TAG}_text_document.bin" "$MEGOUT/${B}_${TAG}_text_document.idx"
  run_build_py tools/preprocess_data.py --input "$STAGE/${B}_final.jsonl" \
    --output-prefix "$MEGOUT/${B}_${TAG}" \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOK" \
    --append-eod --workers 64 --json-keys text
  test -s "$MEGOUT/${B}_${TAG}_text_document.bin"
  test -s "$MEGOUT/${B}_${TAG}_text_document.idx"
done
echo "stageB DONE $B $(date -u)"
ls -lah "$MEGOUT/${B}_"*text_document* 2>&1 | head
