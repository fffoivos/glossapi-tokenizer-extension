#!/usr/bin/env bash
# Fetch the CPT/tokenizer-extension paper library: arXiv PDFs -> pdftotext.
# Idempotent: skips a paper if its .txt already exists and is non-trivial.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
UA="Mozilla/5.0 (paper-library-fetcher; +local)"

# slug|source  where source is an arXiv id or an ACL-anthology URL
MANIFEST=$(cat <<'EOF'
ademamix|2409.03137
apertus|2509.14233
hagele-wsd-scaling|2405.18392
ibrahim-cpt|2403.08763
gupta-rewarm-cpt|2308.04014
practitioner-multimodal-cpt|2408.14471
stability-gap-cpt|2406.14833
ma-yarats-warmup|1910.04209
goldfish-loss|2406.10209
token-distillation|2505.20133
artetxe-crosslingual-transfer|1910.11856
eeve-vocab-expansion|2402.14714
jiang-tokenizer-aware-adaptation|ACL:2026.eacl-long.357
optimal-embedding-lr|2506.15025
allam-arabic-cpt|2407.15390
tokenization-bottleneck|2511.14365
ulmfit|1801.06146
llrd-bert-finetuning|2006.05987
tao-vocab-scaling|2407.13623
magikarp-undertrained-tokens|2405.05417
EOF
)

fetch_one() {
  local slug="$1" src="$2" url pdf="pdf/$1.pdf" txt="txt/$1.txt"
  if [ -s "$txt" ] && [ "$(wc -c < "$txt")" -gt 4000 ]; then
    echo "SKIP  $slug (txt exists)"; return 0
  fi
  if [[ "$src" == ACL:* ]]; then
    url="https://aclanthology.org/${src#ACL:}.pdf"
  else
    url="https://arxiv.org/pdf/${src}"
  fi
  curl -sL -A "$UA" --max-time 120 -o "$pdf" "$url"
  if [ ! -s "$pdf" ] || ! head -c 4 "$pdf" | grep -q '%PDF'; then
    echo "FAIL  $slug ($url) — not a PDF"; rm -f "$pdf"; return 1
  fi
  pdftotext -q "$pdf" "$txt" 2>/dev/null
  local n; n=$(wc -c < "$txt" 2>/dev/null || echo 0)
  if [ "$n" -lt 2000 ]; then echo "WARN  $slug — txt only ${n}B"; return 1; fi
  echo "OK    $slug  (pdf $(du -h "$pdf"|cut -f1), txt ${n}B)"
}
export -f fetch_one

echo "$MANIFEST" | while IFS='|' read -r slug src; do
  [ -z "$slug" ] && continue
  fetch_one "$slug" "$src" &
  # cap concurrency at 6
  while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done
done
wait
echo "=== done; library at $DIR ==="
ls -1 txt/ | sed 's/^/  txt: /'
