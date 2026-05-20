#!/usr/bin/env bash
# Idempotent paper download. For arxiv papers, prefer the HTML version when
# available (text-based; ~10× smaller; easier to grep + cite by section ID),
# fall back to PDF when HTML isn't published (older papers especially).
#
# Apertus 2509.14233 specifically stays PDF — we cite by page number ("§C
# Table C.4 p.82") across the recipe doc, which the HTML version can't
# preserve. The HTML version of arxiv 2509.14233 is also not available
# (2026-05-21 probe: all versions 404).
#
# Non-arxiv references (Hewitt note = HTML, FVT = ACL Anthology PDF) stay
# in their canonical format.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p papers
cd papers

# (filename-base, arxiv-id, format-preference)
#   pdf-only: arxiv has no HTML version (typically pre-2023, or rendering failed)
#   html-pdf-fallback: try HTML first; PDF if HTML 404s
#   pdf-cite-by-page: cite-heavy paper; keep PDF for page-number precision
ARXIV=(
    "apertus_2509.14233|2509.14233|pdf-cite-by-page"
    "ademamix_2409.03137|2409.03137|html-pdf-fallback"
    "goldfish_2406.10209|2406.10209|html-pdf-fallback"
    "retok_2410.04335|2410.04335|html-pdf-fallback"
    "mundra_2407.05841|2407.05841|html-pdf-fallback"
    "qknorm_2010.04245|2010.04245|pdf-only"
    "wsd_minicpm_2404.06395|2404.06395|html-pdf-fallback"
    "fineweb_2406.17557|2406.17557|html-pdf-fallback"
    "finewebhq_2502.10361|2502.10361|html-pdf-fallback"
    "starcoder_2305.06161|2305.06161|pdf-only"
    "megatron_1909.08053|1909.08053|pdf-only"
    "meltemi_2407.20743|2407.20743|html-pdf-fallback"
    "krikri_2505.13772|2505.13772|pdf-only"
)

# Skip if either format is already present (idempotent).
have() {
    [ -f "$1.html" ] || [ -f "$1.pdf" ]
}

fetch_pdf() {
    local filename="$1"; local arxivid="$2"
    echo "  PDF $arxivid → $filename.pdf"
    curl -sSL "https://arxiv.org/pdf/${arxivid}.pdf" -o "${filename}.pdf"
}

fetch_html() {
    # Try v1; if 404 try v2/v3; return 0 on success.
    local filename="$1"; local arxivid="$2"
    for v in v1 v2 v3 ""; do
        local url="https://arxiv.org/html/${arxivid}${v}"
        if curl -fsSL "$url" -o "${filename}.html" 2>/dev/null; then
            # Sanity: must be > 5 KB (arxiv serves a tiny placeholder for missing HTML)
            local size=$(stat -c '%s' "${filename}.html" 2>/dev/null || echo 0)
            if [ "$size" -gt 5000 ]; then
                echo "  HTML $arxivid$v → $filename.html ($size bytes)"
                return 0
            fi
            rm -f "${filename}.html"
        fi
    done
    return 1
}

for entry in "${ARXIV[@]}"; do
    IFS='|' read -r filename arxivid mode <<< "$entry"
    if have "$filename"; then
        echo "  skip (have): $filename"
        continue
    fi
    case "$mode" in
        pdf-only|pdf-cite-by-page)
            fetch_pdf "$filename" "$arxivid"
            ;;
        html-pdf-fallback)
            if ! fetch_html "$filename" "$arxivid"; then
                echo "  (HTML not available; falling back) "
                fetch_pdf "$filename" "$arxivid"
            fi
            ;;
        *)
            echo "  ERROR: unknown mode $mode for $filename" >&2; exit 2 ;;
    esac
    sleep 1   # be nice
done

# Hewitt vocab-expansion is an HTML technical note (not arxiv).
if [ ! -f hewitt_vocab_expansion.html ]; then
    echo "  HTML hewitt_vocab_expansion"
    curl -sSL "https://www.cs.columbia.edu/~johnhew//vocab-expansion.html" \
        -o hewitt_vocab_expansion.html
fi

# FVT is EMNLP 2022 Industry — ACL Anthology has PDF only.
if [ ! -f fvt_emnlp2022_industry_41.pdf ]; then
    echo "  PDF fvt_emnlp2022_industry_41"
    curl -sSL "https://aclanthology.org/2022.emnlp-industry.41.pdf" \
        -o fvt_emnlp2022_industry_41.pdf
fi

echo
echo "=== summary ==="
ls -la *.pdf *.html 2>/dev/null | awk '{printf "  %-40s %s\n", $NF, $5}' | sort
echo
echo "Total: $(du -sh . | cut -f1)"
