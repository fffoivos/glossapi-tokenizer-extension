"""Generate data.json for the Greek-tokens interactive app.

Output: data.json — JSON array of every vocab entry that fired in Greek (>=1).
Each record: {r:rank, i:token_id, d:decoded, c:count, cat:category, gs:is_greek_script}
"""
import json
from pathlib import Path
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent      # vocab_lang_attribution/

z = np.load(ROOT / "outputs/histogram_matrix.npz", allow_pickle=True)
H = z["H"].astype(np.int64)
ck = list(z["canonical_keys"])
ELL = ck.index("ell_Grek")
greek = H[ELL]

tok = pl.read_parquet(ROOT / "outputs/token_metadata.parquet")
is_gs = (tok["has_greek_mono"] | tok["has_greek_poly"]).to_numpy()
decoded = tok["decoded_string"].to_list()
has_latin = (tok["has_latin_basic"] | tok["has_latin_extended"]).to_numpy()
is_struct = tok["is_structural_only"].to_numpy()
is_digit = tok["is_pure_digits"].to_numpy()
is_ws = tok["is_pure_whitespace"].to_numpy()
is_special = tok["is_special"].to_numpy()
is_byte = tok["is_byte_fragment"].to_numpy()

def cat_of(i):
    if is_gs[i]: return "greek"
    if is_special[i]: return "special"
    if is_byte[i]: return "byte_frag"
    if is_struct[i]: return "structural"
    if is_digit[i]: return "digit"
    if is_ws[i]: return "ws"
    if has_latin[i]: return "latin"
    return "other"

# All firing tokens sorted by count desc
firing = np.where(greek > 0)[0]
order = firing[np.argsort(-greek[firing])]

records = []
for rank, i in enumerate(order, 1):
    records.append({
        "r": rank,
        "i": int(i),
        "d": decoded[i],
        "c": int(greek[i]),
        "cat": cat_of(i),
        "gs": bool(is_gs[i]),
    })

# Also a tiny summary
summary = {
    "lang": "ell_Grek",
    "name": "Modern Greek (1453-)",
    "total_greek_tokens": int(greek.sum()),
    "vocab_size": int(H.shape[1]),
    "tokens_fired": int(len(records)),
    "tokens_zero": int(H.shape[1] - len(records)),
    "greek_script_total": int(is_gs.sum()),
    "greek_script_fired": int((is_gs & (greek > 0)).sum()),
    "non_greek_script_fired": int((~is_gs & (greek > 0)).sum()),
    "greek_mass_in_greek_script": int(greek[is_gs].sum()),
    "category_counts": {},
    "category_mass": {},
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
