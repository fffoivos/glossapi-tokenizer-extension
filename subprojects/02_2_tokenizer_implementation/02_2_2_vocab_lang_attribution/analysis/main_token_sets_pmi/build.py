"""Multi-language PMI promotion — implementation of PMI_PROMOTION_SPEC.md.

For the cap-hit canonical keys (Σ firings ≥ 1 B; currently 87 keys
after the eng_Latn_fineweb_hq rerun), emit:

  tables/<key>__masked.txt    — Variant A (char-mask + PMI ≥ δ + count ≥ min_count)
  tables/<key>__unmasked.txt  — Variant B (PMI ≥ δ + count ≥ min_count only)
  tables/<key>__delta.txt     — B \\ A (tokens rejected by the char mask)

Plus:
  summary.tsv                 — one row per target key
  overlap_matrix.tsv          — square |masked_i ∩ masked_j|
  per_token_pmi.parquet       — per (key, token, pmi ≥ 0) audit row
  manifest.json               — provenance + knobs

Consumes char-tool schema v5 (78+ language bits, etc. — exact count
read from the char-tool manifest). PMI base log10. The canonical
marginal is **count-pooled**: `p_marg(t) = Σ count_L(t) / Σ total_L`,
where the sums run over the cap-hit keys. This is equivalent to a
language-mass-weighted average of per-language rates — keys with
larger samples contribute proportionally more to the marginal. Since
all cap-hit `total_L` are ~1 B (within ~4 % of each other), the
count-pooled marginal is numerically close to a 1/K average, but the
two formulas are not identical.

The diagnostic `pmi_training` column uses a different marginal:
`p_marg_training(t) = Σ_L w_L · p_L(t)`. When no weights are
supplied, `w_L = 1/K` (uniform), so this gives the equal-weighted
average of per-language rates — a DIFFERENT formula from canonical
PMI, even if numerically close at our scale. Feed
`--training-weights weights.json` to supply non-uniform weights
(e.g., approximate Apertus training shares).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


# Canonical-key → char-tool-code lookup is published by the char tool
# in manifest.json["canonical_key_to_char_tool_code"] (v5 schema, v3.2.1+).
# We read it directly — no consumer-side ISO mapping, no derived fallback,
# no hardcoded patches. Suffixed keys (e.g. `eng_Latn_fineweb_hq`) fall
# back to their iso_iso15924 root (`eng_Latn`) since the suffix is a
# source tag, not a script disambiguation.
def make_lookup(cl_manifest: dict):
    """Returns a function key → char-tool-code (or None). Reads the
    published canonical_key_to_char_tool_code map and strips source-tag
    suffixes from canonical keys before lookup."""
    ck_map = cl_manifest["canonical_key_to_char_tool_code"]

    def lookup(key: str) -> str | None:
        if key in ck_map:
            return ck_map[key]
        parts = key.split("_")
        if len(parts) >= 2:
            root = f"{parts[0]}_{parts[1]}"
            return ck_map.get(root)
        return None
    return lookup


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha",          type=float, default=0.5)
    ap.add_argument("--delta",          type=float, default=1.0)
    ap.add_argument("--min-count",      type=int,   default=100)
    ap.add_argument("--marginal-floor", type=int,   default=1_000_000_000)
    ap.add_argument("--training-weights", type=Path, default=None,
                    help="JSON file: {canonical_key: float_share}. "
                         "If absent, uses equal weights (training-weighted PMI "
                         "collapses to equal-weighted).")
    args = ap.parse_args()

    HERE   = Path(__file__).resolve().parent
    VLA    = HERE.parent.parent
    CLM    = VLA.parent / "02_2_1_char_language_membership"
    TABLES = HERE / "tables"; TABLES.mkdir(parents=True, exist_ok=True)
    PARAM = {"alpha": args.alpha, "delta": args.delta,
             "min_count": args.min_count, "marginal_floor": args.marginal_floor,
             "pmi_base": 10}

    # ── 1. Load histogram + token-mask + char-tool manifest ────────────────
    hist_path = VLA / "outputs/histogram_matrix.npz"
    tbm_path  = CLM / "artifacts/token_language_bitmask.parquet"
    cm_path   = CLM / "artifacts/manifest.json"
    print(f"[load] histogram_matrix.npz ...")
    z = np.load(hist_path, allow_pickle=True)
    H = z["H"].astype(np.int64)
    keys = list(z["canonical_keys"])
    V = H.shape[1]

    print(f"[load] token_language_bitmask.parquet ...")
    tbm = pl.read_parquet(tbm_path)
    def to_int(b): return int.from_bytes(b, "little") if b is not None else 0
    bm_lang = np.array([to_int(b) for b in tbm["bitmask_and"].to_list()], dtype=object)
    status  = tbm["status"].to_list()
    decoded = tbm["decoded_text"].to_list()
    cl_manifest = json.loads(cm_path.read_text())
    lang_bit_of = {L["code"]: L["bit"] for L in cl_manifest["languages"]}
    N_LANG_BITS = cl_manifest["levels"]["language"]["bits_used"]
    UNKNOWN     = {"partial_utf8", "byte_unmapped", "special"}

    popcount = np.array([bin(int(x)).count("1") for x in bm_lang], dtype=np.int32)
    eval_ok  = np.array([s not in UNKNOWN for s in status], dtype=bool)

    # ── 2. Pick marginal-contributing keys (totals ≥ floor) ────────────────
    totals = H.sum(axis=1)
    marg_idx = np.where(totals >= args.marginal_floor)[0]
    marg_keys = [keys[i] for i in marg_idx]
    print(f"[scope] {len(marg_idx)} marginal keys "
          f"(threshold Σ ≥ {args.marginal_floor:,})")

    # Map each canonical key to its lang_code from the manifest map
    lookup = make_lookup(cl_manifest)
    key_to_lang = {k: lookup(k) for k in marg_keys}
    unmapped = [k for k, c in key_to_lang.items() if c is None]
    if unmapped:
        print(f"[warn] {len(unmapped)} keys have no char-tool lang_code mapping: "
              f"{unmapped[:5]}{'...' if len(unmapped) > 5 else ''}")

    # ── 3. Compute marginal count + total ──────────────────────────────────
    count_marg = H[marg_idx].sum(axis=0).astype(np.float64)        # (V,)
    total_marg = int(totals[marg_idx].sum())
    p_marg     = (count_marg + args.alpha) / (total_marg + args.alpha * V)
    print(f"[marg] total_marg = {total_marg:,}  (over {len(marg_idx)} keys)")

    # ── 3b. Training-weighted marginal (defaults to equal weights) ────────
    if args.training_weights:
        weights_raw = json.loads(args.training_weights.read_text())
    else:
        weights_raw = {}
    weights = np.zeros(len(marg_idx), dtype=np.float64)
    for i, k in enumerate(marg_keys):
        weights[i] = float(weights_raw.get(k, 0.0))
    if weights.sum() == 0:
        weights = np.ones(len(marg_idx), dtype=np.float64) / len(marg_idx)
        weights_source = "equal (no overrides provided)"
    else:
        # missing keys keep weight 0 — normalise the rest
        if weights.sum() < 1.0 - 1e-6:
            # Don't auto-fill — fail loud so the user has to supply complete weights
            raise SystemExit(f"weights.json sums to {weights.sum():.4f}, "
                             f"need 1.0 (or close). Missing keys default to 0.")
        weights = weights / weights.sum()
        weights_source = f"from {args.training_weights}"
    # Training-weighted p(t) = Σ_L w_L · p(t|L)
    p_per_key = np.array([(H[marg_idx[i]].astype(np.float64) + args.alpha)
                           / (totals[marg_idx[i]] + args.alpha * V)
                           for i in range(len(marg_idx))])           # (88, V)
    p_marg_training = (weights[:, None] * p_per_key).sum(axis=0)     # (V,)
    print(f"[marg] training-weighted marginal source: {weights_source}")

    # ── 4. For each target key, compute PMI + masks + emit ─────────────────
    summary_rows = []
    masked_sets  = []   # for overlap matrix
    pmi_audit_rows = []

    print(f"[run] computing PMI for {len(marg_idx)} target keys ...")
    for i, L_idx in enumerate(marg_idx):
        key = keys[L_idx]
        lang_code = key_to_lang[key]
        row = H[L_idx]
        total_L = int(totals[L_idx])
        p_L = (row.astype(np.float64) + args.alpha) / (total_L + args.alpha * V)
        pmi          = np.log10(p_L / p_marg)
        pmi_training = np.log10(p_L / p_marg_training)

        # Variant B — unmasked
        unmasked_mask = (row >= args.min_count) & (pmi >= args.delta)
        # Variant A — masked (requires lang_code resolvable)
        if lang_code is not None and lang_code in lang_bit_of:
            L_BIT = lang_bit_of[lang_code]
            admissible = (np.array([((int(x) >> L_BIT) & 1) for x in bm_lang], dtype=bool)
                          & (popcount > 0) & (popcount < N_LANG_BITS) & eval_ok)
        else:
            admissible = np.zeros(V, dtype=bool)
        masked_mask = unmasked_mask & admissible
        delta_mask  = unmasked_mask & ~admissible       # in B not A

        masked_idx   = np.where(masked_mask)[0]
        unmasked_idx = np.where(unmasked_mask)[0]
        delta_idx    = np.where(delta_mask)[0]
        # Sort by firing count descending
        masked_idx   = masked_idx  [np.argsort(-row[masked_idx])]
        unmasked_idx = unmasked_idx[np.argsort(-row[unmasked_idx])]
        delta_idx    = delta_idx   [np.argsort(-row[delta_idx])]

        # ── write the three txt files ────────────────────────────────────
        def write_set(out_path, idx, header):
            with out_path.open("w") as f:
                f.write(header + "\n")
                f.write(f"# {len(idx):,} tokens, sorted by count desc\n")
                for t in idx:
                    f.write("{%d: %r}\n" % (int(t), decoded[t]))

        hdr_common = (f"# target_key={key}  lang_code={lang_code}\n"
                      f"# Σ_target={total_L:,}  marg_total={total_marg:,}  "
                      f"marg_keys={len(marg_idx)}\n"
                      f"# alpha={args.alpha}  delta={args.delta}  "
                      f"min_count={args.min_count}")
        write_set(TABLES / f"{key}__masked.txt",   masked_idx,
                  hdr_common + f"\n# variant=A (char-mask ON: bit set, popcount<{N_LANG_BITS}, evaluable)")
        write_set(TABLES / f"{key}__unmasked.txt", unmasked_idx,
                  hdr_common + "\n# variant=B (rate test only — no char mask)")
        write_set(TABLES / f"{key}__delta.txt",    delta_idx,
                  hdr_common + "\n# variant=B\\A (rejected by char mask)")

        masked_sets.append(set(masked_idx.tolist()))

        # ── summary row ───────────────────────────────────────────────────
        promoted_pmi = pmi[masked_idx] if masked_idx.size else np.array([])
        unmasked_pmi = pmi[unmasked_idx] if unmasked_idx.size else np.array([])
        masked_mass   = int(row[masked_idx].sum())   if masked_idx.size   else 0
        unmasked_mass = int(row[unmasked_idx].sum()) if unmasked_idx.size else 0
        summary_rows.append({
            "target_key":         key,
            "lang_code":          lang_code or "unmapped",
            "total_L":            total_L,
            "masked_count":       int(masked_idx.size),
            "unmasked_count":     int(unmasked_idx.size),
            "delta_count":        int(delta_idx.size),
            "masked_mass":        masked_mass,
            "unmasked_mass":      unmasked_mass,
            "masked_mass_pct":    100 * masked_mass   / total_L,
            "unmasked_mass_pct":  100 * unmasked_mass / total_L,
            "max_pmi_masked":     float(promoted_pmi.max())    if promoted_pmi.size else float("nan"),
            "min_pmi_masked":     float(promoted_pmi.min())    if promoted_pmi.size else float("nan"),
            "median_pmi_masked":  float(np.median(promoted_pmi))if promoted_pmi.size else float("nan"),
        })

        # ── per_token_pmi audit rows for pmi >= 0 ─────────────────────────
        positive_pmi_idx = np.where((pmi >= 0) & (row >= 1))[0]
        for t in positive_pmi_idx:
            t = int(t)
            pmi_audit_rows.append({
                "target_key":       key,
                "token_id":         t,
                "decoded":          decoded[t],
                "count_L":          int(row[t]),
                "pmi":              float(pmi[t]),
                "pmi_training":     float(pmi_training[t]),
                "admissible_for_L": bool(admissible[t]),
                "popcount_lang":    int(popcount[t]),
            })

        if (i + 1) % 10 == 0 or i == len(marg_idx) - 1:
            print(f"  [{i+1:>3}/{len(marg_idx)}] {key:<35} "
                  f"A={int(masked_idx.size):>6,} B={int(unmasked_idx.size):>6,} "
                  f"Δ={int(delta_idx.size):>5,}")

    # ── 5. summary.tsv ─────────────────────────────────────────────────────
    summary_df = pl.DataFrame(summary_rows)
    summary_df.write_csv(HERE / "summary.tsv", separator="\t")

    # ── 6. overlap_matrix.tsv ──────────────────────────────────────────────
    n = len(masked_sets)
    overlap = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i, n):
            inter = len(masked_sets[i] & masked_sets[j]) if i != j else len(masked_sets[i])
            overlap[i, j] = inter
            overlap[j, i] = inter
    df_overlap = pl.DataFrame({"target_key": marg_keys}
                              | {marg_keys[j]: overlap[:, j].tolist() for j in range(n)})
    df_overlap.write_csv(HERE / "overlap_matrix.tsv", separator="\t")

    # ── 7. per_token_pmi.parquet ───────────────────────────────────────────
    if pmi_audit_rows:
        df_pmi = pl.DataFrame(pmi_audit_rows)
        df_pmi.write_parquet(HERE / "per_token_pmi.parquet", compression="zstd")
        print(f"  per_token_pmi.parquet: {len(pmi_audit_rows):,} rows, "
              f"{(HERE / 'per_token_pmi.parquet').stat().st_size/1e6:.1f} MB")

    # ── 8. weights_used.json ───────────────────────────────────────────────
    (HERE / "weights_used.json").write_text(
        json.dumps({k: float(weights[i]) for i, k in enumerate(marg_keys)},
                   indent=2, ensure_ascii=False)
    )

    # ── 9. validation assertions (best-effort: warn rather than fail) ──────
    print("\n[validate]")
    # Sanity 1: substrate PMI within ±log10(2) ≈ ±0.30
    sub_idx = np.where(popcount == N_LANG_BITS)[0]
    if sub_idx.size:
        # Recompute PMI for a representative key (first marg_idx) and check
        L0 = marg_idx[0]
        row0 = H[L0]; tL0 = int(totals[L0])
        p_L0 = (row0.astype(np.float64) + args.alpha) / (tL0 + args.alpha * V)
        pmi0_sub = np.log10(p_L0[sub_idx] / p_marg[sub_idx])
        sus = int((np.abs(pmi0_sub) > 0.30).sum())
        print(f"  substrate |PMI| within 0.30 for {marg_keys[0]}: "
              f"{sub_idx.size - sus}/{sub_idx.size} pass (suspicious: {sus})")

    # Sanity 2: T0 tokens that fire ≥ min_count should be in masked set
    print("  T0 tokens fired ≥ min_count → all should be in masked set:")
    for i, L_idx in enumerate(marg_idx[:5]):
        key = keys[L_idx]
        lang_code = key_to_lang[key]
        if not lang_code or lang_code not in lang_bit_of: continue
        L_BIT = lang_bit_of[lang_code]
        t0 = (popcount == 1) & np.array([((int(x) >> L_BIT) & 1) == 1 for x in bm_lang], dtype=bool)
        t0_fired = t0 & (H[L_idx] >= args.min_count)
        if t0_fired.sum() == 0: continue
        t0_ids = set(np.where(t0_fired)[0].tolist())
        missed = t0_ids - masked_sets[i]
        if missed:
            print(f"    {key:<30} {len(missed)} T0 tokens fired ≥ min but NOT in masked set!")
        else:
            print(f"    {key:<30} all {t0_fired.sum()} T0 tokens in masked set ✓")

    # ── 10. manifest.json ──────────────────────────────────────────────────
    manifest = {
        "schema_version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "histogram_matrix":    str(hist_path),
            "histogram_matrix_md5": md5_of(hist_path),
            "token_bitmask":       str(tbm_path),
            "token_bitmask_md5":   md5_of(tbm_path),
            "char_tool_schema_version": cl_manifest.get("schema_version"),
        },
        "parameters": PARAM,
        "n_marginal_keys":  len(marg_idx),
        "marginal_keys":    marg_keys,
        "marginal_total":   total_marg,
        "double_counted_languages": ["en (eng_Latn + eng_Latn_fineweb_hq)"],
        "target_keys":      marg_keys,
        "keys_without_lang_code_mapping": unmapped,
        "training_weights_source": weights_source,
        "outputs": {
            "tables_dir":            "tables/",
            "per_token_pmi_parquet": "per_token_pmi.parquet",
            "overlap_matrix_tsv":    "overlap_matrix.tsv",
            "summary_tsv":           "summary.tsv",
            "weights_used_json":     "weights_used.json",
        },
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n[done] outputs in {HERE}")
    print(f"  tables/                 — {len(marg_idx) * 3} txt files (masked/unmasked/delta)")
    print(f"  summary.tsv             — per-key counts/mass/PMI stats")
    print(f"  overlap_matrix.tsv      — {n}×{n} pairwise overlap of masked sets")
    print(f"  per_token_pmi.parquet   — audit table")
    print(f"  manifest.json           — provenance")
    print(f"  weights_used.json       — training weights actually applied")


if __name__ == "__main__":
    main()
