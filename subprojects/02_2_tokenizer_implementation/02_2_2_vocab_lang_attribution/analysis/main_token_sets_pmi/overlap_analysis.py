"""Inspect overlap_matrix.tsv — does the per-language overlap make sense?

Expectations to test:
1. Closely-related languages overlap a lot (de/nl/da/nb/sv, es/it/pt/ca, ru/uk/bg/mk, …).
2. Cross-script pairs overlap near 0 (jpn vs deu, ell vs rus, etc.).
3. Same-script-distant-family pairs overlap weakly (en/tr — both Latin but different families).
4. The two English samples (eng_Latn wiki vs eng_Latn_fineweb_hq) overlap massively
   — they're the same language under different sources.
"""
import json
from pathlib import Path
import polars as pl
import numpy as np

HERE = Path(__file__).resolve().parent
mf = json.loads((HERE / "manifest.json").read_text())
keys = mf["marginal_keys"]

# Load the overlap matrix
df = pl.read_csv(HERE / "overlap_matrix.tsv", separator="\t")
M = df.select(keys).to_numpy()      # square (n × n)
n = len(keys)
key_to_idx = {k: i for i, k in enumerate(keys)}
diag = np.diag(M).astype(np.int64)

# Lang-code lookup — use the same manifest-driven path as build.py
import sys
sys.path.insert(0, str(HERE))
from build import make_lookup
CLM = HERE.parent.parent.parent / "02_2_1_char_language_membership"
cl_manifest = json.loads((CLM / "artifacts/manifest.json").read_text())
lookup = make_lookup(cl_manifest)
key_to_lang = {k: lookup(k) for k in keys}

# Helper: extract script from canonical key
def script_of(k):
    parts = k.split("_")
    return parts[1] if len(parts) >= 2 else None

# ── 1. Largest off-diagonal pairs ─────────────────────────────────────────
print("=== Top 25 overlapping pairs (off-diagonal, by absolute overlap) ===")
print(f"  {'overlap':>8}  {'jaccard':>7}  {'pair'}")
pairs = []
for i in range(n):
    for j in range(i+1, n):
        if M[i, j] > 0:
            union = diag[i] + diag[j] - M[i, j]
            jaccard = M[i, j] / max(union, 1)
            pairs.append((M[i, j], jaccard, keys[i], keys[j]))
pairs.sort(key=lambda x: -x[0])
for ovl, jac, ki, kj in pairs[:25]:
    print(f"  {ovl:>8,}  {jac:>7.3f}  {ki} ↔ {kj}")

# ── 2. Largest by Jaccard (relative overlap) ───────────────────────────────
print("\n=== Top 25 pairs by Jaccard similarity (relative overlap) ===")
print(f"  {'jaccard':>7}  {'overlap':>8}  {'pair'}")
pairs.sort(key=lambda x: -x[1])
for ovl, jac, ki, kj in pairs[:25]:
    print(f"  {jac:>7.3f}  {ovl:>8,}  {ki} ↔ {kj}")

# ── 3. Cross-script pairs — should overlap near 0 ─────────────────────────
print("\n=== Cross-script pairs with non-zero overlap (sanity: should be tiny) ===")
script_pairs = []
for i in range(n):
    for j in range(i+1, n):
        s_i = script_of(keys[i]); s_j = script_of(keys[j])
        if s_i and s_j and s_i != s_j and M[i, j] > 0:
            union = diag[i] + diag[j] - M[i, j]
            jaccard = M[i, j] / max(union, 1)
            script_pairs.append((M[i, j], jaccard, keys[i], keys[j], s_i, s_j))
script_pairs.sort(key=lambda x: -x[0])
print(f"  {'overlap':>8}  {'pair'}")
for ovl, jac, ki, kj, si, sj in script_pairs[:15]:
    print(f"  {ovl:>8,}  {ki} ({si}) ↔ {kj} ({sj})")
print(f"  (total cross-script pairs with non-zero overlap: {len(script_pairs):,})")

# ── 4. Same-family clusters (derived from char-tool families.yaml) ────────
print("\n=== Within-family overlap (derived from char-tool manifest) ===")
# Build SCRIPT_FAMILIES from the char-tool manifest's `families` list +
# `canonical_key_to_char_tool_code` map. Each char-tool family gives us
# the list of language codes (BCP47) that belong to it; we invert the
# canonical_key map to find which canonical keys map to each code.
CLM_DIR = HERE.parent.parent.parent / "02_2_1_char_language_membership"
char_manifest_path = CLM_DIR / "artifacts/manifest.json"
if char_manifest_path.exists():
    char_mf = json.loads(char_manifest_path.read_text())
    ck_map = char_mf.get("canonical_key_to_char_tool_code", {})
    # inverse: BCP47 code -> [canonical_keys that map to it].
    # Also add cap-hit keys with source-tag suffixes (e.g.
    # eng_Latn_fineweb_hq) that resolve to the same code as their root.
    code_to_cks = {}
    for ck, code in ck_map.items():
        code_to_cks.setdefault(code, []).append(ck)
    for k in keys:
        if k in ck_map: continue
        parts = k.split("_")
        if len(parts) >= 3:
            root = f"{parts[0]}_{parts[1]}"
            root_code = ck_map.get(root)
            if root_code:
                code_to_cks.setdefault(root_code, []).append(k)
    SCRIPT_FAMILIES = {}
    for fam in char_mf.get("families", []):
        members = []
        for locale in fam.get("locales", []):
            members.extend(code_to_cks.get(locale, []))
        # Also include any cap-hit canonical keys that mapped to a code with
        # this family via suffix-stripped lookup (e.g. eng_Latn_fineweb_hq).
        # Skip families with <2 cap-hit members.
        members_in_caphit = [m for m in members if m in key_to_idx]
        if len(members_in_caphit) >= 2:
            SCRIPT_FAMILIES[fam["code"]] = members
else:
    # Fallback if char manifest isn't reachable — minimal hardcoded set
    SCRIPT_FAMILIES = {"Germanic-Latn": ["eng_Latn", "deu_Latn", "nld_Latn"]}
for fam, members in SCRIPT_FAMILIES.items():
    present = [m for m in members if m in key_to_idx]
    if len(present) < 2: continue
    print(f"\n  {fam} ({len(present)} members in cap-hit):")
    for ii, a in enumerate(present):
        for b in present[ii+1:]:
            i, j = key_to_idx[a], key_to_idx[b]
            ovl = int(M[i, j]); union = diag[i] + diag[j] - ovl
            jaccard = ovl / max(union, 1)
            print(f"    {a:<22} ↔ {b:<22}  overlap={ovl:>5,}  jaccard={jaccard:.3f}")

# ── 5. Two English samples — the "same language" sanity check ─────────────
print("\n=== eng_Latn (wiki) vs eng_Latn_fineweb_hq (FineWeb-HQ) — same language, different sources ===")
ka, kb = "eng_Latn", "eng_Latn_fineweb_hq"
i, j = key_to_idx[ka], key_to_idx[kb]
ovl = int(M[i, j]); union = diag[i] + diag[j] - ovl; jaccard = ovl / max(union, 1)
print(f"  set sizes: {diag[i]:,} (wiki) vs {diag[j]:,} (FineWeb-HQ)")
print(f"  overlap:   {ovl:,}")
print(f"  jaccard:   {jaccard:.4f}")
print(f"  wiki-only: {diag[i] - ovl:,}")
print(f"  fwhq-only: {diag[j] - ovl:,}")
