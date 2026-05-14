"""Methodology comparison: char-language-membership bitmask vs script-flag, for Greek.

Outputs:
  tables/leakage_top200.tsv       — top 200 decoded-text tokens that fired in
                                    Greek but are NOT el-capable (bit 16 = 0).
                                    These are the loanwords / names / code /
                                    contamination.
  tables/unknown_standalone_top200.tsv
                                  — top 200 partial-UTF8 / non-text tokens that
                                    fired in Greek. These are not hard
                                    membership rejections; the char-level tool
                                    cannot evaluate them as standalone chars.
  tables/dormant_el_capable.tsv   — tokens that ARE el-capable but did not fire
                                    in our 1 B Greek sample. The "Greek-capable
                                    but unused" set.
  tables/confusion.tsv            — 2×2 confusion: el-capable × fired.
  membership_report.md            — short methodology write-up.
"""
import json
from pathlib import Path
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
VLA  = HERE.parent.parent     # 02_2_2_vocab_lang_attribution/
CLM  = VLA.parent / "02_2_1_char_language_membership"
TABLES = HERE / "tables"; TABLES.mkdir(exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────
z = np.load(VLA / "outputs/histogram_matrix.npz", allow_pickle=True)
H = z["H"].astype(np.int64); ck = list(z["canonical_keys"])
ELL = ck.index("ell_Grek")
greek = H[ELL]
total_greek = int(greek.sum())

tok = pl.read_parquet(VLA / "outputs/token_metadata.parquet")
is_gs = (tok["has_greek_mono"] | tok["has_greek_poly"]).to_numpy()
decoded = tok["decoded_string"].to_list()
has_latin = (tok["has_latin_basic"] | tok["has_latin_extended"]).to_numpy()
is_struct = tok["is_structural_only"].to_numpy()

clm = pl.read_parquet(CLM / "artifacts/token_language_bitmask.parquet")
def to_int(b): return int.from_bytes(b, "little") if b is not None else 0
bm_and = [to_int(b) for b in clm["bitmask_and"].to_list()]
status = clm["status"].to_list()
manifest = json.load(open(CLM / "artifacts/manifest.json"))
bit2code = {L["bit"]: L["code"] for L in manifest["languages"]}
N_BITS = len(bit2code)
EL_BIT  = 16
ELP_BIT = 22

V = H.shape[1]
el_capable  = np.array([(bm_and[i] >> EL_BIT)  & 1 for i in range(V)], dtype=bool)
elp_capable = np.array([(bm_and[i] >> ELP_BIT) & 1 for i in range(V)], dtype=bool)
either_el   = el_capable | elp_capable
popcount    = np.array([bin(bm_and[i]).count("1") for i in range(V)], dtype=np.int32)

fired = greek > 0
unknown_statuses = {"partial_utf8", "byte_unmapped", "special"}
unknown = np.array([s in unknown_statuses for s in status], dtype=bool)
evaluable = ~unknown
evaluable_fired = fired & evaluable

# ── confusion: el-capable × fired ──────────────────────────────────────────
def conf(predicate, fired_mask, universe):
    predicate = predicate & universe
    fired_mask = fired_mask & universe
    tp = int((predicate & fired_mask).sum())
    fp = int((predicate & ~fired_mask & universe).sum())
    fn = int((~predicate & fired_mask & universe).sum())
    tn = int((~predicate & ~fired_mask & universe).sum())
    return tp, fp, fn, tn

scenarios = [
    ("is_greek_script (script flag)", is_gs),
    ("el-capable (bit 16)",            el_capable),
    ("el-poly-capable (bit 22)",       elp_capable),
    ("either el or el-poly capable",   either_el),
    ("el-capable AND NOT substrate",   el_capable & (popcount < N_BITS)),
]

print("\n=== Confusion vs 'fired in Greek' (decoded/evaluable tokens only) ===")
print(f"{'classifier':<40} {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>9} {'P':>6} {'R':>6} {'Mass%':>7}")
rows = []
for label, pred in scenarios:
    pred_eval = pred & evaluable
    tp, fp, fn, tn = conf(pred_eval, evaluable_fired, evaluable)
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    mass = int(greek[pred_eval].sum()) / total_greek
    print(f"{label:<40} {tp:>7,} {fp:>7,} {fn:>7,} {tn:>9,} {P:>6.3f} {R:>6.3f} {100*mass:>6.2f}%")
    rows.append({"classifier": label, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                 "precision": P, "recall": R, "mass_fraction": mass})
pl.DataFrame(rows).write_csv(TABLES / "confusion.tsv", separator="\t")

# ── leakage: fired AND NOT el-capable ──────────────────────────────────────
print("\n=== Leakage table (decoded text fired in Greek but NOT el-capable, top 200) ===")
leak_mask = evaluable_fired & ~either_el
leak_idx  = np.where(leak_mask)[0]
leak_top  = leak_idx[np.argsort(-greek[leak_idx])][:200]
with open(TABLES / "leakage_top200.tsv", "w") as f:
    f.write("rank\ttoken_id\tdecoded\tcount\tshare_pct\tstatus\tcompat_popcount\tsignature\n")
    for rank, i in enumerate(leak_top, 1):
        sig = ",".join(bit2code[b] for b in range(N_BITS) if (bm_and[i] >> b) & 1)
        if not sig: sig = "(none)"
        if len(sig) > 80: sig = sig[:80] + "…"
        f.write(f"{rank}\t{i}\t{decoded[i]!r}\t{int(greek[i])}\t"
                f"{100*float(greek[i])/total_greek:.4f}\t{status[i]}\t{popcount[i]}\t{sig}\n")
print(f"  total leakage rows: {leak_mask.sum():,}, mass {int(greek[leak_mask].sum()):,} "
      f"({100*greek[leak_mask].sum()/total_greek:.2f}% of Greek-sample)")

# ── unknown standalone tokens: fired, but char-membership cannot decide ─────
print("\n=== Unknown standalone tokens (fired, but NOT hard rejections) ===")
unknown_fired = fired & unknown
unknown_idx = np.where(unknown_fired)[0]
unknown_top = unknown_idx[np.argsort(-greek[unknown_idx])][:200]
with open(TABLES / "unknown_standalone_top200.tsv", "w") as f:
    f.write("rank\ttoken_id\tdecoded\tcount\tshare_pct\tstatus\n")
    for rank, i in enumerate(unknown_top, 1):
        f.write(f"{rank}\t{i}\t{decoded[i]!r}\t{int(greek[i])}\t"
                f"{100*float(greek[i])/total_greek:.4f}\t{status[i]}\n")
print(f"  unknown standalone rows: {unknown_fired.sum():,}, mass {int(greek[unknown_fired].sum()):,} "
      f"({100*greek[unknown_fired].sum()/total_greek:.2f}% of Greek-sample)")

# ── dormant: el-capable AND NOT fired ──────────────────────────────────────
print("\n=== Dormant el-capable tokens (in vocab, did not fire in 1 B Greek sample) ===")
dorm_mask = either_el & ~fired
dorm_idx  = np.where(dorm_mask)[0]
with open(TABLES / "dormant_el_capable.tsv", "w") as f:
    f.write("token_id\tdecoded\tstatus\tcompat_popcount\tsignature\n")
    for i in dorm_idx:
        sig = ",".join(bit2code[b] for b in range(N_BITS) if (bm_and[i] >> b) & 1)
        if not sig: sig = "(none)"
        f.write(f"{i}\t{decoded[i]!r}\t{status[i]}\t{popcount[i]}\t{sig}\n")
print(f"  dormant: {dorm_mask.sum():,} tokens "
      f"({either_el.sum():,} el-capable in vocab, {dorm_idx.size:,} unused)")

# ── breakdown of the leakage by category ──────────────────────────────────
print("\n=== Leakage breakdown by char-membership status ===")
leak_status = {}
for i in leak_idx:
    s = status[i]
    leak_status.setdefault(s, [0, 0])
    leak_status[s][0] += 1
    leak_status[s][1] += int(greek[i])
print(f"  {'status':<32} {'tokens':>10} {'mass':>16} {'mass%':>8}")
for s, (n, m) in sorted(leak_status.items(), key=lambda x: -x[1][1]):
    print(f"  {s:<32} {n:>10,} {m:>16,} {100*m/total_greek:>7.3f}%")

# ── breakdown by Latin / Cyrillic / Han / other token-script ─────────────
print("\n=== Leakage breakdown by token-metadata category ===")
leak_cat = {"latin": [0,0], "structural":[0,0], "digit":[0,0], "other":[0,0]}
for i in leak_idx:
    if has_latin[i] and not is_gs[i]:
        k = "latin"
    elif is_struct[i] and not is_gs[i]:
        k = "structural"
    elif tok["is_pure_digits"].to_numpy()[i]:
        k = "digit"
    else:
        k = "other"
    leak_cat[k][0] += 1
    leak_cat[k][1] += int(greek[i])
for k, (n, m) in sorted(leak_cat.items(), key=lambda x: -x[1][1]):
    print(f"  {k:<20} {n:>10,} {m:>16,} {100*m/total_greek:>7.3f}%")

print("\n[done]")
