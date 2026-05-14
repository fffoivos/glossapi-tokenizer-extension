"""Generate data.json for the Greek-tokens interactive app.

Now joins the 02_2_1_char_language_membership bitmasks (55 (language, script,
encoding) triples) into each record:

  el       — bitmask_and has bit 16 set (Modern Greek, monotonic).
  elp      — bitmask_and has bit 22 set (Greek polytonic).
  pc       — popcount of bitmask_and (how many in-scope locales admit this token).
  pco      — popcount of bitmask_or (relaxed: at least one char admits).
  cm_st    — char-language-membership status (text / partial_utf8 / special / …).
  sig      — short comma-joined list of locale codes whose bit is set in bitmask_and,
              up to 4 codes; longer signatures truncated with "…".

Output: data.json — JSON object with `summary` and `tokens` (100,014 firing records).
"""
import json
from pathlib import Path
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
VLA  = HERE.parent.parent.parent             # 02_2_2_vocab_lang_attribution/
CLM  = VLA.parent / "02_2_1_char_language_membership"  # sibling sub-subproject

# ── load vocab-attribution Greek slice ──────────────────────────────────────
z = np.load(VLA / "outputs/histogram_matrix.npz", allow_pickle=True)
H = z["H"].astype(np.int64)
ck = list(z["canonical_keys"])
ELL = ck.index("ell_Grek")
greek = H[ELL]

tok = pl.read_parquet(VLA / "outputs/token_metadata.parquet")
is_gs = (tok["has_greek_mono"] | tok["has_greek_poly"]).to_numpy()
decoded = tok["decoded_string"].to_list()
has_latin = (tok["has_latin_basic"] | tok["has_latin_extended"]).to_numpy()
is_struct = tok["is_structural_only"].to_numpy()
is_digit = tok["is_pure_digits"].to_numpy()
is_ws = tok["is_pure_whitespace"].to_numpy()
is_special_meta = tok["is_special"].to_numpy()
is_byte = tok["is_byte_fragment"].to_numpy()

# ── load 02_2_1_char_language_membership bitmasks ──────────────────────────────────
clm = pl.read_parquet(CLM / "artifacts/token_language_bitmask.parquet")
clm_manifest = json.load(open(CLM / "artifacts/manifest.json"))
bit2code = {L["bit"]: L["code"] for L in clm_manifest["languages"]}
N_BITS = len(bit2code)  # 55

EL_BIT  = 16   # el (modern Greek)
ELP_BIT = 22   # el-polyton
UNKNOWN_STATUSES = {"partial_utf8", "byte_unmapped", "special"}

def as_int(buf):
    return int.from_bytes(buf, "little") if buf is not None else 0

# Build per-token int representations, indexed by token_id
bm_and = [as_int(b) for b in clm["bitmask_and"].to_list()]
bm_or  = [as_int(b) for b in clm["bitmask_or"].to_list()]
cm_st  = clm["status"].to_list()
# clm is already 0..131071 in order, but be safe:
ids = clm["token_id"].to_list()
assert ids == list(range(len(ids))), "token_language_bitmask not in order!"


def popcount(x: int) -> int:
    return bin(x).count("1")


def signature(mask: int, max_codes: int = 4) -> str:
    if mask == 0:
        return "(none)"
    pc = popcount(mask)
    if pc == N_BITS:
        return "(all)"
    codes = [bit2code[b] for b in range(N_BITS) if (mask >> b) & 1]
    if len(codes) <= max_codes:
        return ",".join(codes)
    return ",".join(codes[:max_codes]) + f"…+{len(codes)-max_codes}"


def cat_of(i):
    if is_gs[i]: return "greek"
    if is_special_meta[i]: return "special"
    if is_byte[i]: return "byte_frag"
    if is_struct[i]: return "structural"
    if is_digit[i]: return "digit"
    if is_ws[i]: return "ws"
    if has_latin[i]: return "latin"
    return "other"


# ── build records ───────────────────────────────────────────────────────────
firing = np.where(greek > 0)[0]
order = firing[np.argsort(-greek[firing])]

records = []
for rank, i in enumerate(order, 1):
    ba = bm_and[i]
    bo = bm_or[i]
    records.append({
        "r":   rank,
        "i":   int(i),
        "d":   decoded[i],
        "c":   int(greek[i]),
        "cat": cat_of(i),
        "gs":  bool(is_gs[i]),
        "el":  bool((ba >> EL_BIT) & 1),
        "elp": bool((ba >> ELP_BIT) & 1),
        "pc":  popcount(ba),
        "pco": popcount(bo),
        "sig": signature(ba),
        "st":  cm_st[i],
        "unk": cm_st[i] in UNKNOWN_STATUSES,
    })


# ── summary ─────────────────────────────────────────────────────────────────
def count_where(predicate):
    return sum(1 for r in records if predicate(r))

def mass_where(predicate):
    return sum(r["c"] for r in records if predicate(r))

summary = {
    "lang":                       "ell_Grek",
    "name":                       "Modern Greek (1453-)",
    "total_greek_tokens":         int(greek.sum()),
    "vocab_size":                 int(H.shape[1]),
    "tokens_fired":               int(len(records)),
    "tokens_zero":                int(H.shape[1] - len(records)),
    "greek_script_total":         int(is_gs.sum()),
    "greek_script_fired":         int((is_gs & (greek > 0)).sum()),
    "non_greek_script_fired":     int((~is_gs & (greek > 0)).sum()),
    "greek_mass_in_greek_script": int(greek[is_gs].sum()),
    # bitmask-based:
    "el_capable_in_vocab":        sum(1 for i in range(H.shape[1]) if (bm_and[i] >> EL_BIT) & 1),
    "elp_capable_in_vocab":       sum(1 for i in range(H.shape[1]) if (bm_and[i] >> ELP_BIT) & 1),
    "el_capable_fired":           count_where(lambda r: r["el"]),
    "el_capable_fired_mass":      mass_where(lambda r: r["el"]),
    "elp_capable_fired":          count_where(lambda r: r["elp"]),
    "either_greek_capable_fired": count_where(lambda r: r["el"] or r["elp"]),
    "substrate_fired":            count_where(lambda r: r["pc"] == 55),
    "substrate_fired_mass":       mass_where(lambda r: r["pc"] == 55),
    "unknown_standalone_fired":          count_where(lambda r: r["unk"]),
    "unknown_standalone_fired_mass":     mass_where(lambda r: r["unk"]),
    "no_language_compatible_fired":      count_where(lambda r: r["pc"] == 0 and not r["unk"]),
    "no_language_compatible_fired_mass": mass_where(lambda r: r["pc"] == 0 and not r["unk"]),
    "category_counts":            {},
    "category_mass":              {},
    "bit_to_code":                bit2code,
    "membership_source":          "02_2_1_char_language_membership · "
                                  f"CLDR {clm_manifest.get('cldr_release', '?')} · "
                                  f"{N_BITS} (language, script, encoding) triples",
}
for c in ["greek","structural","digit","ws","latin","byte_frag","special","other"]:
    mask = np.array([r["cat"] == c for r in records])
    if mask.any():
        summary["category_counts"][c] = int(mask.sum())
        summary["category_mass"][c]   = int(sum(r["c"] for r in records if r["cat"] == c))

out = {"summary": summary, "tokens": records}
(HERE / "data.json").write_text(json.dumps(out, ensure_ascii=False))
sz = (HERE / "data.json").stat().st_size
print(f"wrote data.json — {len(records):,} firing tokens, {sz/1e6:.1f} MB")
print(f"  el-capable in vocab:   {summary['el_capable_in_vocab']:,}")
print(f"  el-capable fired:      {summary['el_capable_fired']:,}  ({100*summary['el_capable_fired_mass']/summary['total_greek_tokens']:.2f}% of mass)")
print(f"  el-poly capable fired: {summary['elp_capable_fired']:,}")
print(f"  either Greek capable:  {summary['either_greek_capable_fired']:,}")
print(f"  substrate fired:       {summary['substrate_fired']:,}  ({100*summary['substrate_fired_mass']/summary['total_greek_tokens']:.2f}% of mass)")
print(f"  unknown standalone:    {summary['unknown_standalone_fired']:,}  ({100*summary['unknown_standalone_fired_mass']/summary['total_greek_tokens']:.2f}% of mass)")
print(f"  decoded zero-language-bits fired: {summary['no_language_compatible_fired']:,}")
