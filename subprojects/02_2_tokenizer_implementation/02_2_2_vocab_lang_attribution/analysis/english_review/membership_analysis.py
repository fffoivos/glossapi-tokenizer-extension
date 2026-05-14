"""Methodology comparison for English (eng_Latn), mirroring greek_review.

Joins `02_2_1_char_language_membership/artifacts/token_language_bitmask.parquet`
with the eng_Latn slice of the firing histogram. English = bit 0 (en).

Outputs:
  tables/popcount_distribution.tsv  — token-type and mass counts by
                                       popcount of bitmask_and for tokens
                                       that fired in English.
  tables/leakage_top200.tsv         — top 200 decoded-text tokens that
                                       fired in English but are NOT
                                       en-capable. These are diacritic
                                       loanwords, foreign names, foreign-
                                       script text, etc.
  tables/unknown_standalone_top200.tsv
                                    — top 200 partial_utf8/non-text tokens
                                       that fired in English but cannot
                                       be evaluated as standalone chars.
  tables/dormant_en_capable.tsv     — en-capable tokens that did not fire.
  tables/confusion.tsv              — 5-row confusion table.
  tables/rule_summary.tsv           — three rule definitions + coverage.
"""
import json
from pathlib import Path
import numpy as np
import polars as pl

HERE   = Path(__file__).resolve().parent
VLA    = HERE.parent.parent           # 02_2_2_vocab_lang_attribution/
CLM    = VLA.parent / "02_2_1_char_language_membership"
TABLES = HERE / "tables"; TABLES.mkdir(exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────
z   = np.load(VLA / "outputs/histogram_matrix.npz", allow_pickle=True)
H   = z["H"].astype(np.int64)
ck  = list(z["canonical_keys"])
ENG = ck.index("eng_Latn")
eng = H[ENG]
total_eng = int(eng.sum())
V = H.shape[1]

tok = pl.read_parquet(VLA / "outputs/token_metadata.parquet")
decoded = tok["decoded_string"].to_list()
has_latin = (tok["has_latin_basic"] | tok["has_latin_extended"]).to_numpy()
has_greek = (tok["has_greek_mono"]  | tok["has_greek_poly"] ).to_numpy()
has_cyr   = tok["has_cyrillic"].to_numpy()
is_struct = tok["is_structural_only"].to_numpy()
is_digit  = tok["is_pure_digits"].to_numpy()

clm = pl.read_parquet(CLM / "artifacts/token_language_bitmask.parquet")
def to_int(b): return int.from_bytes(b, "little") if b is not None else 0
bm_and = [to_int(b) for b in clm["bitmask_and"].to_list()]
status = clm["status"].to_list()
manifest = json.load(open(CLM / "artifacts/manifest.json"))
bit2code = {L["bit"]: L["code"] for L in manifest["languages"]}
N_BITS = len(bit2code)
EN_BIT = 0  # English

en_capable = np.array([(bm_and[i] >> EN_BIT) & 1 for i in range(V)], dtype=bool)
popcount   = np.array([bin(bm_and[i]).count("1") for i in range(V)], dtype=np.int32)
fired      = eng > 0
unknown_statuses = {"partial_utf8", "byte_unmapped", "special"}
unknown    = np.array([s in unknown_statuses for s in status], dtype=bool)
evaluable  = ~unknown
evaluable_fired = fired & evaluable

# ── confusion: en-capable × fired ──────────────────────────────────────────
def conf(predicate, fired_mask, universe):
    predicate = predicate & universe
    fired_mask = fired_mask & universe
    tp = int((predicate & fired_mask).sum())
    fp = int((predicate & ~fired_mask & universe).sum())
    fn = int((~predicate & fired_mask & universe).sum())
    tn = int((~predicate & ~fired_mask & universe).sum())
    return tp, fp, fn, tn

scenarios = [
    ("en-capable (bit 0)",               en_capable),
    ("en-capable AND NOT substrate",     en_capable & (popcount < N_BITS)),
    ("en-capable AND popcount == 28",    en_capable & (popcount == 28)),
    ("Latin-script (has_latin)",         has_latin),
    ("Latin-script AND NOT structural",  has_latin & ~is_struct & ~is_digit),
]

print("\n=== Confusion vs 'fired in English' (decoded/evaluable tokens only) ===")
print(f"{'classifier':<40} {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>9} {'P':>6} {'R':>6} {'Mass%':>7}")
rows = []
for label, pred in scenarios:
    pred_eval = pred & evaluable
    tp, fp, fn, tn = conf(pred_eval, evaluable_fired, evaluable)
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    mass = int(eng[pred_eval].sum()) / total_eng
    print(f"{label:<40} {tp:>7,} {fp:>7,} {fn:>7,} {tn:>9,} {P:>6.3f} {R:>6.3f} {100*mass:>6.2f}%")
    rows.append({"classifier": label, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                 "precision": P, "recall": R, "mass_fraction": mass})
pl.DataFrame(rows).write_csv(TABLES / "confusion.tsv", separator="\t")

# ── popcount distribution among fired tokens ───────────────────────────────
print("\n=== Popcount distribution of FIRED tokens (en-capable view) ===")
print(f"  {'popcount':>8} {'tokens':>8} {'mass':>14} {'mass%':>7}  meaning")
rows_pc = []
def meaning(pc):
    if pc == 0:  return "(no in-scope language admits any char)"
    if pc == 1:  return "exactly one locale admits — language-distinctive"
    if pc == 27: return "Latin minus one locale (often Turkish / Polish)"
    if pc == 28: return "all 28 Latin-script locales admit — pan-Latin"
    if pc == 55: return "every locale admits — substrate (punct/digit/ws)"
    return ""
for pc_val in sorted(set(popcount[fired].tolist())):
    mask = fired & (popcount == pc_val)
    n = int(mask.sum())
    m = int(eng[mask].sum())
    print(f"  {pc_val:>8} {n:>8,} {m:>14,} {100*m/total_eng:>6.3f}%  {meaning(pc_val)}")
    rows_pc.append({"popcount": pc_val, "tokens": n, "mass": m,
                    "mass_pct": 100*m/total_eng, "meaning": meaning(pc_val)})
pl.DataFrame(rows_pc).write_csv(TABLES / "popcount_distribution.tsv", separator="\t")

# ── leakage: decoded text fired AND NOT en-capable ─────────────────────────
print("\n=== Leakage table (decoded text fired in English but NOT en-capable, top 200) ===")
leak_mask = evaluable_fired & ~en_capable
leak_idx  = np.where(leak_mask)[0]
leak_top  = leak_idx[np.argsort(-eng[leak_idx])][:200]
with open(TABLES / "leakage_top200.tsv", "w") as f:
    f.write("rank\ttoken_id\tdecoded\tcount\tshare_pct\tstatus\tcompat_popcount\tsignature\n")
    for rank, i in enumerate(leak_top, 1):
        sig = ",".join(bit2code[b] for b in range(N_BITS) if (bm_and[i] >> b) & 1)
        if not sig: sig = "(none)"
        if len(sig) > 100: sig = sig[:100] + "…"
        f.write(f"{rank}\t{i}\t{decoded[i]!r}\t{int(eng[i])}\t"
                f"{100*float(eng[i])/total_eng:.4f}\t{status[i]}\t{popcount[i]}\t{sig}\n")
print(f"  total leakage rows: {leak_mask.sum():,}, mass {int(eng[leak_mask].sum()):,} "
      f"({100*eng[leak_mask].sum()/total_eng:.3f}% of English-sample)")

# ── unknown standalone tokens ──────────────────────────────────────────────
print("\n=== Unknown standalone tokens (fired in English, can't evaluate standalone) ===")
unknown_fired = fired & unknown
unknown_idx = np.where(unknown_fired)[0]
unknown_top = unknown_idx[np.argsort(-eng[unknown_idx])][:200]
with open(TABLES / "unknown_standalone_top200.tsv", "w") as f:
    f.write("rank\ttoken_id\tdecoded\tcount\tshare_pct\tstatus\n")
    for rank, i in enumerate(unknown_top, 1):
        f.write(f"{rank}\t{i}\t{decoded[i]!r}\t{int(eng[i])}\t"
                f"{100*float(eng[i])/total_eng:.4f}\t{status[i]}\n")
print(f"  unknown standalone rows: {unknown_fired.sum():,}, mass {int(eng[unknown_fired].sum()):,} "
      f"({100*eng[unknown_fired].sum()/total_eng:.3f}% of English-sample)")

# ── dormant: en-capable AND NOT fired ──────────────────────────────────────
print("\n=== Dormant en-capable tokens (in vocab, did not fire in 1 B English sample) ===")
dorm_mask = en_capable & ~fired
dorm_idx  = np.where(dorm_mask)[0]
with open(TABLES / "dormant_en_capable.tsv", "w") as f:
    f.write("token_id\tdecoded\tstatus\tcompat_popcount\tsignature\n")
    for i in dorm_idx:
        sig = ",".join(bit2code[b] for b in range(N_BITS) if (bm_and[i] >> b) & 1)
        if not sig: sig = "(none)"
        f.write(f"{i}\t{decoded[i]!r}\t{status[i]}\t{popcount[i]}\t{sig}\n")
print(f"  dormant: {dorm_mask.sum():,} tokens "
      f"({en_capable.sum():,} en-capable in vocab, {dorm_idx.size:,} unused)")

# ── leakage breakdown by category ──────────────────────────────────────────
print("\n=== Leakage breakdown by char-membership status ===")
leak_status = {}
for i in leak_idx:
    s = status[i]
    leak_status.setdefault(s, [0, 0])
    leak_status[s][0] += 1
    leak_status[s][1] += int(eng[i])
print(f"  {'status':<32} {'tokens':>10} {'mass':>16} {'mass%':>8}")
for s, (n, m) in sorted(leak_status.items(), key=lambda x: -x[1][1]):
    print(f"  {s:<32} {n:>10,} {m:>16,} {100*m/total_eng:>7.3f}%")

print("\n=== Leakage breakdown by token-metadata script ===")
leak_cat = {"latin_diacritic":[0,0], "greek":[0,0], "cyrillic":[0,0],
            "structural":[0,0], "digit":[0,0], "other":[0,0]}
for i in leak_idx:
    if has_greek[i]:               k = "greek"
    elif has_cyr[i]:               k = "cyrillic"
    elif has_latin[i]:             k = "latin_diacritic"
    elif is_struct[i]:             k = "structural"
    elif is_digit[i]:              k = "digit"
    else:                          k = "other"
    leak_cat[k][0] += 1
    leak_cat[k][1] += int(eng[i])
for k, (n, m) in sorted(leak_cat.items(), key=lambda x: -x[1][1]):
    print(f"  {k:<20} {n:>10,} {m:>16,} {100*m/total_eng:>7.3f}%")

# ── three rule definitions ─────────────────────────────────────────────────
print("\n=== Rule summary (token types + mass coverage) ===")
rule_rows = [
    ("English MAXIMAL", "bit 0 set in bitmask_and",
     en_capable),
    ("English DISTINCTIVE (no substrate)", "bit 0 set AND popcount < 55",
     en_capable & (popcount < N_BITS)),
    ("English pure-pan-Latin", "bit 0 set AND popcount == 28",
     en_capable & (popcount == 28)),
    ("English near-pan-Latin", "bit 0 set AND popcount in {27, 28}",
     en_capable & ((popcount == 27) | (popcount == 28))),
    ("script-flag-only (has_latin)", "any Latin codepoint in decoded",
     has_latin),
]
rule_summary = []
print(f"{'rule':<38} {'vocab':>8} {'fired':>8} {'mass%':>7}")
for name, rule, pred in rule_rows:
    vocab_n = int(pred.sum())
    fired_n = int((pred & fired).sum())
    mass    = int(eng[pred & fired].sum())
    print(f"{name:<38} {vocab_n:>8,} {fired_n:>8,} {100*mass/total_eng:>6.2f}%")
    rule_summary.append({"name": name, "predicate": rule,
                         "vocab_size": vocab_n, "fired": fired_n,
                         "mass_pct": 100*mass/total_eng})
pl.DataFrame(rule_summary).write_csv(TABLES / "rule_summary.tsv", separator="\t")

# ── summary.json ───────────────────────────────────────────────────────────
summary = {
    "lang": "eng_Latn",
    "name": "English",
    "total_sample_tokens": total_eng,
    "vocab_size": V,
    "fired_in_eng": int(fired.sum()),
    "en_capable_in_vocab": int(en_capable.sum()),
    "en_capable_fired": int((en_capable & fired).sum()),
    "en_capable_fired_mass": int(eng[en_capable & fired].sum()),
    "en_capable_fired_mass_pct": 100*int(eng[en_capable & fired].sum())/total_eng,
    "leakage_decoded_tokens": int(leak_mask.sum()),
    "leakage_decoded_mass": int(eng[leak_mask].sum()),
    "leakage_decoded_mass_pct": 100*int(eng[leak_mask].sum())/total_eng,
    "unknown_standalone_tokens": int(unknown_fired.sum()),
    "unknown_standalone_mass": int(eng[unknown_fired].sum()),
    "unknown_standalone_mass_pct": 100*int(eng[unknown_fired].sum())/total_eng,
    "dormant_en_capable": int(dorm_mask.sum()),
    "substrate_fired_tokens": int((fired & (popcount == N_BITS)).sum()),
    "substrate_fired_mass_pct": 100*int(eng[fired & (popcount == N_BITS)].sum())/total_eng,
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nwrote summary.json")
print("[done]")
