"""Build per-language isolation sets at three strictness levels.

For each (canonical_key, lang_code) pair, partition the fired tokens into
the same tier hierarchy as `german_review/tiered_attribution.py` and emit
three filter sets:

  strict       = T0 only                                    (no premise)
  premise      = T0 ∪ T2                                    (with premise)
  premise_sub  = T0 ∪ T2 ∪ T3 (substrate included)          (with premise + infra)

Uses v4 char masks (script_and / family_and / bitmask_and). N_LANG_BITS is
read from the char_language_membership manifest, not hardcoded.

Output:
  tables/<L>/<canonical_key>__<mode>.jsonl
      one record per token: {id, decoded, count, tier, basis}
  tables/<L>/<canonical_key>__summary.tsv
      one row per mode: {mode, tokens, mass, mass_pct}
  manifest.json
      per-run parameters + provenance
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-key", required=True,
                    help="e.g. deu_Latn / eng_Latn / eng_Latn_fineweb_hq")
    ap.add_argument("--lang-code", required=True,
                    help="ISO bit code for the target language, e.g. de / en / el")
    ap.add_argument("--vla-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent.parent,
                    help="Path to 02_2_2_vocab_lang_attribution/")
    ap.add_argument("--clm-dir", type=Path, default=None,
                    help="Path to 02_2_1_char_language_membership/. Defaults to sibling.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "tables",
                    help="Where to write tables/<L>/*.jsonl.")
    args = ap.parse_args()

    clm_dir = args.clm_dir or (args.vla_dir.parent / "02_2_1_char_language_membership")
    out_root = args.out_dir / args.lang_code
    out_root.mkdir(parents=True, exist_ok=True)

    # ── load char masks (v4) ───────────────────────────────────────────────
    clm = pl.read_parquet(clm_dir / "artifacts/token_language_bitmask.parquet")
    def to_int(b): return int.from_bytes(b, "little") if b is not None else 0
    bm_lang = [to_int(b) for b in clm["bitmask_and"].to_list()]
    status  = clm["status"].to_list()
    decoded = clm["decoded_text"].to_list()
    V = len(bm_lang)
    manifest_path = clm_dir / "artifacts/manifest.json"
    cl_manifest = json.loads(manifest_path.read_text())
    lang_bit_of = {L["code"]: L["bit"] for L in cl_manifest["languages"]}
    if args.lang_code not in lang_bit_of:
        raise SystemExit(f"unknown lang_code {args.lang_code!r}; available: "
                         f"{sorted(lang_bit_of.keys())}")
    L_BIT = lang_bit_of[args.lang_code]
    N_LANG_BITS = cl_manifest["levels"]["language"]["bits_used"]
    LANG_MASK_ALL = (1 << N_LANG_BITS) - 1
    UNKNOWN_STATUSES = {"partial_utf8", "byte_unmapped", "special"}

    # ── load firing histogram for the target canonical key ────────────────
    z = np.load(args.vla_dir / "outputs/histogram_matrix.npz", allow_pickle=True)
    keys = list(z["canonical_keys"])
    if args.canonical_key not in keys:
        raise SystemExit(f"unknown canonical_key {args.canonical_key!r}; "
                         f"closest: {[k for k in keys if args.lang_code in k][:5]}")
    H = z["H"].astype(np.int64)
    row = H[keys.index(args.canonical_key)]
    total_L = int(row.sum())

    # ── tier per fired token ───────────────────────────────────────────────
    fired_idx = np.where(row > 0)[0]
    tier_of_token = {}
    for i in fired_idx:
        i = int(i)
        if status[i] in UNKNOWN_STATUSES:
            tier_of_token[i] = "T5"
            continue
        bm = bm_lang[i]
        pc = bin(bm).count("1")
        has_L = ((bm >> L_BIT) & 1) == 1
        if pc == 1 and has_L:
            tier_of_token[i] = "T0"
        elif pc == N_LANG_BITS:
            tier_of_token[i] = "T3"
        elif not has_L:
            tier_of_token[i] = "T4"
        else:
            tier_of_token[i] = "T2"

    # ── three modes ────────────────────────────────────────────────────────
    MODES = {
        "strict":       {"T0"},
        "premise":      {"T0", "T2"},
        "premise_sub":  {"T0", "T2", "T3"},
    }

    mode_summary = []
    for mode, tiers in MODES.items():
        ids = [i for i, t in tier_of_token.items() if t in tiers]
        ids.sort(key=lambda i: -int(row[i]))
        mass = int(row[ids].sum()) if ids else 0
        out_jsonl = out_root / f"{args.canonical_key}__{mode}.jsonl"
        with out_jsonl.open("w") as f:
            for i in ids:
                rec = {
                    "id":    int(i),
                    "decoded": decoded[i],
                    "count": int(row[i]),
                    "tier":  tier_of_token[i],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mode_summary.append({
            "mode":     mode,
            "tiers":    sorted(tiers),
            "tokens":   len(ids),
            "mass":     mass,
            "mass_pct": 100 * mass / total_L if total_L else 0.0,
            "out_jsonl": str(out_jsonl.relative_to(out_root.parent)),
        })

    # ── per-tier breakdown (for the summary) ──────────────────────────────
    tier_breakdown = {}
    for t in ("T0", "T2", "T3", "T4", "T5"):
        ids_t = [i for i, tt in tier_of_token.items() if tt == t]
        tier_breakdown[t] = {
            "tokens": len(ids_t),
            "mass":   int(row[ids_t].sum()) if ids_t else 0,
            "mass_pct": 100 * int(row[ids_t].sum()) / total_L if (ids_t and total_L) else 0.0,
        }

    # ── write summary TSV + manifest ──────────────────────────────────────
    sum_tsv = out_root / f"{args.canonical_key}__summary.tsv"
    flat = [{**m, "tiers": ",".join(m["tiers"])} for m in mode_summary]
    pl.DataFrame(flat).write_csv(sum_tsv, separator="\t")

    manifest = {
        "canonical_key": args.canonical_key,
        "lang_code":     args.lang_code,
        "lang_bit":      L_BIT,
        "n_lang_bits":   N_LANG_BITS,
        "total_sample":  total_L,
        "tier_breakdown": tier_breakdown,
        "modes":         mode_summary,
        "premise_text":  (f"In the {args.canonical_key} dataset, tokens whose chars "
                          f"are all {args.lang_code}-admissible (bitmask_and has bit "
                          f"{L_BIT}) default to {args.lang_code} under the 'premise' "
                          f"mode. T0 is char-evidenced (no premise needed). T3 is "
                          f"substrate (universal). The 'premise_sub' mode adds T3 "
                          f"to capture full corpus infrastructure."),
        "char_membership_schema_version": cl_manifest.get("schema_version"),
    }
    mf_path = out_root / f"{args.canonical_key}__manifest.json"
    mf_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"{args.canonical_key} × {args.lang_code} (bit {L_BIT}):")
    print(f"  total sample tokens: {total_L:,}")
    print(f"  {'mode':<14} {'tokens':>8} {'mass':>14} {'mass%':>7}")
    for m in mode_summary:
        print(f"  {m['mode']:<14} {m['tokens']:>8,} {m['mass']:>14,} "
              f"{m['mass_pct']:>6.2f}%")
    print(f"  -> {out_root}")


if __name__ == "__main__":
    main()
