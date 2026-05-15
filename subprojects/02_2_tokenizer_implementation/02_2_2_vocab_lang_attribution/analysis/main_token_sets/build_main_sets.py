"""Main token sets for English vs German (FineWeb pair).

Uses the rate-distinctiveness technique from
`02_2_4_language_category_promotion/METHODOLOGY.md` § Method F (filter).

Pool (per target language L, competitor L'):
  - bitmask_and has L-bit set (chars admit L)
  - popcount(bitmask_and) < N_LANG_BITS  (not universal substrate)
  - bitmask_and != 0                    (chars admit at least one in-scope locale)
  - status NOT in {partial_utf8, byte_unmapped, special}

Per-token effect:
  score(t, L) = log10((count_L + α) / (count_L' + α))    α=0.5 Laplace
                ↑ positive → L is more frequent;  negative → L' is more frequent

Promotion criteria:
  count_L      >= MIN_COUNT     (evidence floor)
  score(t, L)  >= DELTA         (effect-size threshold)

Defaults below: MIN_COUNT=100, DELTA=1.0  (≥10× more in target than competitor).

Outputs:
  main_tokens_en.txt   one line per promoted en token: {id: 'decoded'}
  main_tokens_de.txt   one line per promoted de token: {id: 'decoded'}
  summary.json         numbers + parameter values + at-a-glance comparison
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=100,
                    help="evidence floor — token must fire >= this many times in the target language")
    ap.add_argument("--delta",     type=float, default=1.0,
                    help="effect-size threshold — log10 of target/competitor rate must be >= this")
    ap.add_argument("--en-key", default="eng_Latn_fineweb_hq")
    ap.add_argument("--de-key", default="deu_Latn")
    args = ap.parse_args()

    HERE = Path(__file__).resolve().parent
    VLA  = HERE.parent.parent
    CLM  = VLA.parent / "02_2_1_char_language_membership"

    # ── load firing counts ────────────────────────────────────────────────
    z = np.load(VLA / "outputs/histogram_matrix.npz", allow_pickle=True)
    H = z["H"].astype(np.int64); ck = list(z["canonical_keys"])
    en = H[ck.index(args.en_key)]
    de = H[ck.index(args.de_key)]
    total_en, total_de = int(en.sum()), int(de.sum())
    V = H.shape[1]

    # ── load char masks (v4) ──────────────────────────────────────────────
    clm = pl.read_parquet(CLM / "artifacts/token_language_bitmask.parquet")
    def to_int(b): return int.from_bytes(b, "little") if b is not None else 0
    bm_lang = np.array([to_int(b) for b in clm["bitmask_and"].to_list()], dtype=object)
    status  = clm["status"].to_list()
    decoded = clm["decoded_text"].to_list()

    manifest = json.load(open(CLM / "artifacts/manifest.json"))
    lang_bit = {L["code"]: L["bit"] for L in manifest["languages"]}
    EN_BIT, DE_BIT = lang_bit["en"], lang_bit["de"]
    N_LANG_BITS    = manifest["levels"]["language"]["bits_used"]
    UNKNOWN = {"partial_utf8", "byte_unmapped", "special"}

    popcount = np.array([bin(int(x)).count("1") for x in bm_lang], dtype=np.int32)
    en_cap = np.array([((int(x) >> EN_BIT) & 1) for x in bm_lang], dtype=bool)
    de_cap = np.array([((int(x) >> DE_BIT) & 1) for x in bm_lang], dtype=bool)
    eval_ok = np.array([s not in UNKNOWN for s in status], dtype=bool)

    # T0/T2 pool per language: in-scope L-admissible, no substrate, no T4/T5
    pool_en = en_cap & (popcount > 0) & (popcount < N_LANG_BITS) & eval_ok
    pool_de = de_cap & (popcount > 0) & (popcount < N_LANG_BITS) & eval_ok

    # ── effect size per token ────────────────────────────────────────────
    ALPHA = 0.5
    score_en = np.log10((en.astype(np.float64) + ALPHA) / (de.astype(np.float64) + ALPHA))
    # score for de is the negative of score_en, computed independently
    score_de = np.log10((de.astype(np.float64) + ALPHA) / (en.astype(np.float64) + ALPHA))

    # ── promotion ─────────────────────────────────────────────────────────
    promoted_en = pool_en & (en >= args.min_count) & (score_en >= args.delta)
    promoted_de = pool_de & (de >= args.min_count) & (score_de >= args.delta)

    en_idx = np.where(promoted_en)[0]
    de_idx = np.where(promoted_de)[0]
    en_idx = en_idx[np.argsort(-en[en_idx])]   # rank by target firing count
    de_idx = de_idx[np.argsort(-de[de_idx])]

    # ── write the two txt files ──────────────────────────────────────────
    out_en = HERE / "main_tokens_en.txt"
    out_de = HERE / "main_tokens_de.txt"
    with out_en.open("w") as f:
        f.write(f"# en main tokens — fired >= {args.min_count} in {args.en_key}, "
                f"log10(en/de) >= {args.delta}, en-admissible, non-substrate\n")
        f.write(f"# {len(en_idx):,} tokens, sorted by count_en desc\n")
        for i in en_idx:
            f.write("{%d: %r}\n" % (int(i), decoded[i]))
    with out_de.open("w") as f:
        f.write(f"# de main tokens — fired >= {args.min_count} in {args.de_key}, "
                f"log10(de/en) >= {args.delta}, de-admissible, non-substrate\n")
        f.write(f"# {len(de_idx):,} tokens, sorted by count_de desc\n")
        for i in de_idx:
            f.write("{%d: %r}\n" % (int(i), decoded[i]))

    # ── activation-fingerprint summary ────────────────────────────────────
    fired_en, fired_de = (en > 0), (de > 0)
    fired_both = fired_en & fired_de
    fired_only_en = fired_en & ~fired_de
    fired_only_de = fired_de & ~fired_en
    # Per-token rate (normalised firing rates per language) — used for the
    # cosine of the activation fingerprint.
    p_en = en / max(total_en, 1)
    p_de = de / max(total_de, 1)
    norm_en = float(np.linalg.norm(p_en))
    norm_de = float(np.linalg.norm(p_de))
    cosine = float((p_en * p_de).sum() / (norm_en * norm_de)) if norm_en and norm_de else float("nan")
    # L1 distance of the rate vectors — half the total-variation distance
    tvd = float(0.5 * np.abs(p_en - p_de).sum())
    # Spearman-like top-K rank overlap at a few cutoffs
    rk_overlap = {}
    en_sorted = np.argsort(-en)
    de_sorted = np.argsort(-de)
    for K in (10, 100, 1000, 10000):
        rk_overlap[K] = int(len(set(en_sorted[:K].tolist()) & set(de_sorted[:K].tolist())))

    promoted_en_mass = int(en[en_idx].sum())
    promoted_de_mass = int(de[de_idx].sum())

    summary = {
        "params": {"min_count": args.min_count, "delta": args.delta, "alpha": ALPHA,
                   "en_key": args.en_key, "de_key": args.de_key},
        "totals": {"total_en": total_en, "total_de": total_de,
                   "fired_en": int(fired_en.sum()), "fired_de": int(fired_de.sum()),
                   "fired_both": int(fired_both.sum()),
                   "fired_only_en": int(fired_only_en.sum()),
                   "fired_only_de": int(fired_only_de.sum())},
        "activation_fingerprint_distance": {
            "cosine_similarity_p_en_p_de": cosine,
            "total_variation_distance":   tvd,
            "top_K_rank_intersection":    rk_overlap,
        },
        "main_sets": {
            "en": {"tokens": int(promoted_en.sum()),
                   "mass_count": promoted_en_mass,
                   "mass_pct_of_total_en": 100 * promoted_en_mass / total_en},
            "de": {"tokens": int(promoted_de.sum()),
                   "mass_count": promoted_de_mass,
                   "mass_pct_of_total_de": 100 * promoted_de_mass / total_de},
        },
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"=== Activation fingerprint distance (en={args.en_key} vs de={args.de_key}) ===")
    print(f"  total_en = {total_en:,}")
    print(f"  total_de = {total_de:,}")
    print(f"  cosine(p_en, p_de)        = {cosine:.4f}")
    print(f"  total-variation distance  = {tvd:.4f}     (0 = identical, 1 = disjoint)")
    print(f"  top-10  rank intersection = {rk_overlap[10]}/10")
    print(f"  top-100 rank intersection = {rk_overlap[100]}/100")
    print(f"  top-1k  rank intersection = {rk_overlap[1000]}/1,000")
    print(f"  top-10k rank intersection = {rk_overlap[10000]:,}/10,000")
    print()
    print(f"=== Main token sets (min_count={args.min_count}, delta={args.delta}) ===")
    print(f"  English main: {len(en_idx):>6,} tokens · "
          f"{promoted_en_mass:>14,} mass · "
          f"{100*promoted_en_mass/total_en:>6.2f}% of total_en")
    print(f"  German  main: {len(de_idx):>6,} tokens · "
          f"{promoted_de_mass:>14,} mass · "
          f"{100*promoted_de_mass/total_de:>6.2f}% of total_de")
    print()
    print(f"wrote {out_en.name} and {out_de.name}")


if __name__ == "__main__":
    main()
