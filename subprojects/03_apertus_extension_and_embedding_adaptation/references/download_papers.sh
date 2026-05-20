#!/usr/bin/env bash
# Idempotent paper PDF download. Skips already-present files.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p papers
cd papers

# arXiv PDF URLs use /pdf/<id>v<N>.pdf; we use the latest available version.
ARXIV=(
    "apertus_2509.14233|2509.14233"
    "ademamix_2409.03137|2409.03137"
    "goldfish_2406.10209|2406.10209"
    "retok_2410.04335|2410.04335"
    "mundra_2407.05841|2407.05841"
    "qknorm_2010.04245|2010.04245"
    "wsd_minicpm_2404.06395|2404.06395"
    "fineweb_2406.17557|2406.17557"
    "finewebhq_2502.10361|2502.10361"
    "starcoder_2305.06161|2305.06161"
    "megatron_1909.08053|1909.08053"
    "meltemi_2407.20743|2407.20743"
    "krikri_2505.13772|2505.13772"
)

for entry in "${ARXIV[@]}"; do
    IFS='|' read -r filename arxivid <<< "$entry"
    target="${filename}.pdf"
    if [ -f "$target" ]; then
        echo "  skip (have): $target"
        continue
    fi
    echo "  download: $target  ($arxivid)"
    curl -sSL "https://arxiv.org/pdf/${arxivid}.pdf" -o "$target"
    sleep 1  # be nice
done

# Hewitt's vocab-expansion is an HTML technical note, not arxiv.
if [ ! -f hewitt_vocab_expansion.html ]; then
    echo "  download: hewitt_vocab_expansion.html"
    curl -sSL "https://www.cs.columbia.edu/~johnhew//vocab-expansion.html" \
        -o hewitt_vocab_expansion.html
fi

# FVT is EMNLP Industry; ACL Anthology PDF.
if [ ! -f fvt_emnlp2022_industry_41.pdf ]; then
    echo "  download: fvt_emnlp2022_industry_41.pdf"
    curl -sSL "https://aclanthology.org/2022.emnlp-industry.41.pdf" \
        -o fvt_emnlp2022_industry_41.pdf
fi

echo
echo "=== papers/ summary ==="
ls -la *.pdf *.html 2>/dev/null | awk '{print "  " $NF, "  " $5}'
echo
echo "Total: $(du -sh . | cut -f1)"
