"""Derive the sub-1B-language vocab-vs-tokens scaling slope for script-isolated
languages, and project at the polytonic-Greek training-corpus token count.

Reproduces the numbers in POLYTONIC_VOCAB_BUDGET_CHECK.md.

Run as:
  /home/foivos/.venvs/glossapi-merge-docling/bin/python3 verify_polytonic_budget.py
"""
import json
import math
from pathlib import Path

REPO = Path("/home/foivos/Projects/glossapi-tokenizer-extension")
LANG_META = REPO / "subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/outputs/lang_metadata.json"
# Polytonic corpus shape: read from the firing-count summary (the split_manifest stores
# per-split chars; the firing summary aggregates them under char_counts).
POLY_SPLIT_MANIFEST = REPO / "subprojects/02_1_tokenizer_experiments/02_1_polytonic_greek_extension/analysis/c3p_polytonic_20260518T_impl/splits/split_manifest.json"
POLY_FIRING_SUMMARY = REPO / "subprojects/02_1_tokenizer_experiments/02_1_polytonic_greek_extension/analysis/c3p_polytonic_20260518T_impl/token_firing_counts/polytonic_token_firing_summary.json"

# Script-isolated languages (Latin-script langs over-fire vocab due to shared
# basic-Latin substrate, so they're not the right cohort for polytonic Greek).
SCRIPT_ISOLATED = {
    "Grek", "Cyrl", "Hani", "Hira", "Kana", "Hang", "Hebr", "Arab",
    "Deva", "Beng", "Tamil", "Telu", "Knda", "Mlym", "Guru", "Gujr",
    "Orya", "Sinh", "Thai", "Lao", "Mymr", "Khmr", "Tibt", "Armn",
    "Geor", "Ethi", "Yiii", "Mong", "Syrc", "Nkoo", "Cher", "Cans",
    "Vaii", "Adlm", "Olck", "Saur", "Tale", "Talu", "Lana", "Khar",
}


def fit_power_law(xs, ys):
    """Log-log linear fit: y ≈ a * x^b. Returns (a, b, r2, n)."""
    n = len(xs)
    log_x = [math.log(x) for x in xs]
    log_y = [math.log(y) for y in ys]
    mx, my = sum(log_x) / n, sum(log_y) / n
    num = sum((log_x[i] - mx) * (log_y[i] - my) for i in range(n))
    den = sum((log_x[i] - mx) ** 2 for i in range(n))
    b = num / den
    a = math.exp(my - b * mx)
    ss_tot = sum((log_y[i] - my) ** 2 for i in range(n))
    pred = [math.log(a) + b * lx for lx in log_x]
    ss_res = sum((log_y[i] - pred[i]) ** 2 for i in range(n))
    return a, b, 1 - ss_res / ss_tot, n


def main():
    meta = json.load(open(LANG_META))
    split = json.load(open(POLY_SPLIT_MANIFEST))
    firing = json.load(open(POLY_FIRING_SUMMARY))

    # Total chars across the hygiene-kept polytonic corpus (train + val + test)
    poly_chars_total = firing["char_counts"]["hygiene_kept_all"]
    poly_tokens_est = poly_chars_total / 2.29  # base C3 chars/token at +0 polytonic
    bytes_total = sum(split["outputs"][k]["utf8_bytes"] for k in ("poly_train", "poly_val", "poly_test"))
    print(f"Polytonic training corpus: {poly_chars_total:,} chars "
          f"(rows: {split['hygiene']['rows_after']:,}, bytes UTF-8: {bytes_total:,})")
    print(f"At Apertus+C3 chars/token = 2.29 → {poly_tokens_est / 1e6:.1f}M tokens "
          f"({'sub-1B' if poly_tokens_est < 1e9 else 'OVER 1B'})")
    print()

    cohort = [
        (k, v["sample_tokens_total"], v["vocab_entries_fired_geq_100"], v["script_iso15924"])
        for k, v in meta.items()
        if v.get("sample_tokens_total", 0) > 0
        and v.get("vocab_entries_fired_geq_100", 0) > 0
        and v.get("sample_tokens_total", 0) < 999_000_000
        and v.get("script_iso15924", "") in SCRIPT_ISOLATED
    ]
    xs = [t for _, t, _, _ in cohort]
    ys = [v100 for _, _, v100, _ in cohort]
    a, b, r2, n = fit_power_law(xs, ys)
    print(f"Script-isolated sub-1B cohort: n = {n}")
    print(f"Fit: vocab_fired_geq_100 ≈ {a:.4f} × tokens^{b:.4f}  (R² = {r2:.3f})")
    print()

    grc = meta["grc_Grek"]
    fit_at_grc = a * grc["sample_tokens_total"] ** b
    print(f"grc_Grek anchor: actual {grc['vocab_entries_fired_geq_100']:,} @ "
          f"{grc['sample_tokens_total']/1e6:.0f}M tokens vs fit predicts {fit_at_grc:,.0f}")

    print()
    print(f"Projection at polytonic corpus size ({poly_tokens_est/1e6:.0f}M tokens):")
    print(f"  pure fit:       {a * poly_tokens_est**b:,.0f}")
    print(f"  grc_Grek-scaled: {grc['vocab_entries_fired_geq_100'] * (poly_tokens_est/grc['sample_tokens_total'])**b:,.0f}")
    print()
    print("Candidate 256-aligned totals (base 148,480 = C3 17,408-curated-padded ship):")
    for add in [3584, 4096, 4608, 5120, 6144, 7168, 7680, 10240]:
        total = 148_480 + add
        assert total % 256 == 0
        in_range = 4800 <= add <= 7500
        print(f"  +{add:>5}  →  total {total:>7,} = 256 × {total // 256:<3}"
              f"  {'← inside pattern range (4800-7500)' if in_range else ''}")


if __name__ == "__main__":
    main()
