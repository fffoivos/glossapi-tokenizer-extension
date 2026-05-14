"""Apply char-language-membership as a rejection signal over all rows.

This joins:

  - outputs/histogram_matrix.npz
  - outputs/lang_metadata.json
  - ../02_2_1_char_language_membership/artifacts/token_language_bitmask.parquet

The important contract: a zero mask is only a hard rejection for decoded text.
`partial_utf8`, `byte_unmapped`, and `special` tokens are reported as
unknown/non-text because the char-level membership table cannot evaluate them
as standalone decoded strings.

Outputs:

  - language_membership_rejections.tsv
      One row per canonical dataset language/script row.
  - direct_rejection_top_tokens.tsv
      Top incompatible decoded-text tokens for rows with direct CLM bits.
  - summary.json
      Machine-readable run metadata and aggregate totals.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CLM = ROOT.parent / "02_2_1_char_language_membership"

OUT_SUMMARY = HERE / "language_membership_rejections.tsv"
OUT_TOP = HERE / "direct_rejection_top_tokens.tsv"
OUT_JSON = HERE / "summary.json"

UNKNOWN_STATUSES = {"partial_utf8", "byte_unmapped", "special"}

# Direct language/script/encoding rows in the 1,933-row empirical histogram
# that correspond to CLM bits. Values are CLM language codes, not bit ids.
#
# Greek is intentionally grouped as modern + polytonic for the broad "Greek"
# row; Ancient Greek gets only the polytonic bit. Mandarin/Hani is grouped as
# zh-Hans + zh-Hant because the empirical row does not distinguish simplified
# vs traditional.
DIRECT_CANONICAL_TO_CLM_CODES: dict[str, list[str]] = {
    "eng_Latn": ["en"],
    "ces_Latn": ["cs"],
    "dan_Latn": ["da"],
    "deu_Latn": ["de"],
    "spa_Latn": ["es"],
    "fra_Latn": ["fr"],
    "hun_Latn": ["hu"],
    "ind_Latn": ["id"],
    "ita_Latn": ["it"],
    "nld_Latn": ["nl"],
    "pol_Latn": ["pl"],
    "por_Latn": ["pt"],
    "swe_Latn": ["sv"],
    "tur_Latn": ["tr"],
    "vie_Latn": ["vi"],
    "rus_Cyrl": ["ru"],
    "ell_Grek": ["el", "el-polyton"],
    "grc_Grek": ["el-polyton"],
    "arb_Arab": ["ar"],
    "fas_Arab": ["fa"],
    "cmn_Hani": ["zh-Hans", "zh-Hant"],
    "jpn_Jpan": ["ja"],
    "kor_Hang": ["ko"],
    "hin_Deva": ["hi"],
    "heb_Hebr": ["he"],
    "tha_Thai": ["th"],
    "hye_Armn": ["hy"],
    "kat_Geor": ["ka"],
    "ben_Beng": ["bn"],
    "tam_Taml": ["ta"],
    "tel_Telu": ["te"],
    "kan_Knda": ["kn"],
    "mal_Mlym": ["ml"],
    "guj_Gujr": ["gu"],
    "pan_Guru": ["pa"],
    "mya_Mymr": ["my"],
    "urd_Arab": ["ur"],
    "ron_Latn": ["ro"],
    "ukr_Cyrl": ["uk"],
    "bul_Cyrl": ["bg"],
    "mkd_Cyrl": ["mk"],
    "srp_Latn": ["sr-Latn"],
    "srp_Cyrl": ["sr-Cyrl"],
    "azj_Latn": ["az"],
    "fin_Latn": ["fi"],
    "nob_Latn": ["nb"],
    "slv_Latn": ["sl"],
    "hrv_Latn": ["hr"],
    "slk_Latn": ["sk"],
    "ekk_Latn": ["et"],
    "lit_Latn": ["lt"],
    "lvs_Latn": ["lv"],
    "cat_Latn": ["ca"],
    "isl_Latn": ["is"],
}


SCRIPT_TO_CLM_SCRIPTS: dict[str, set[str]] = {
    "Latn": {"Latn"},
    "Cyrl": {"Cyrl"},
    "Grek": {"Grek"},
    "Arab": {"Arab"},
    "Hani": {"Hans", "Hant", "Jpan"},
    "Hans": {"Hans"},
    "Hant": {"Hant"},
    "Jpan": {"Jpan"},
    "Kana": {"Jpan"},
    "Hang": {"Hang"},
    "Deva": {"Deva"},
    "Hebr": {"Hebr"},
    "Thai": {"Thai"},
    "Armn": {"Armn"},
    "Geor": {"Geor"},
    "Beng": {"Beng"},
    "Taml": {"Taml"},
    "Telu": {"Telu"},
    "Knda": {"Knda"},
    "Mlym": {"Mlym"},
    "Gujr": {"Gujr"},
    "Guru": {"Guru"},
    "Mymr": {"Mymr"},
}


def decode_mask(buf: bytes | None) -> int:
    return int.from_bytes(buf, "little") if buf is not None else 0


def mask_for_codes(codes: list[str], code_to_bit: dict[str, int]) -> int:
    out = 0
    for code in codes:
        out |= 1 << code_to_bit[code]
    return out


def mask_for_scripts(scripts: set[str], manifest_languages: list[dict]) -> int:
    out = 0
    for lang in manifest_languages:
        if lang["script"] in scripts:
            out |= 1 << lang["bit"]
    return out


def signature(mask: int, bit_to_code: dict[int, str], max_codes: int = 8) -> str:
    if mask == 0:
        return "(none)"
    bits = [bit for bit in sorted(bit_to_code) if mask & (1 << bit)]
    codes = [bit_to_code[bit] for bit in bits]
    if len(codes) <= max_codes:
        return ",".join(codes)
    return ",".join(codes[:max_codes]) + f"...+{len(codes) - max_codes}"


def fmt_pct(part: int, total: int) -> str:
    if total == 0:
        return "0.000000"
    return f"{100 * part / total:.6f}"


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((CLM / "artifacts" / "manifest.json").read_text())
    manifest_languages = manifest["languages"]
    code_to_bit = {lang["code"]: lang["bit"] for lang in manifest_languages}
    bit_to_code = {lang["bit"]: lang["code"] for lang in manifest_languages}
    all_bits = mask_for_codes(list(code_to_bit), code_to_bit)

    z = np.load(ROOT / "outputs" / "histogram_matrix.npz", allow_pickle=True)
    H = z["H"].astype(np.int64)
    canonical_keys = [str(k) for k in z["canonical_keys"]]

    lang_meta = json.loads((ROOT / "outputs" / "lang_metadata.json").read_text())

    clm = pq.read_table(
        CLM / "artifacts" / "token_language_bitmask.parquet",
        columns=["token_id", "decoded_text", "bitmask_and", "status"],
    )
    token_ids = clm["token_id"].to_pylist()
    if token_ids != list(range(len(token_ids))):
        raise RuntimeError("token_language_bitmask.parquet is not token_id ordered")
    decoded = clm["decoded_text"].to_pylist()
    status = np.array(clm["status"].to_pylist(), dtype=object)
    bm_and = np.array(
        [decode_mask(buf) for buf in clm["bitmask_and"].to_pylist()],
        dtype=object,
    )

    unknown = np.isin(status, list(UNKNOWN_STATUSES))
    evaluable = ~unknown
    substrate = np.array([mask == all_bits for mask in bm_and], dtype=bool)
    zero_decoded = evaluable & np.array([mask == 0 for mask in bm_and], dtype=bool)

    rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []

    direct_count = 0
    script_proxy_count = 0
    no_basis_count = 0

    for row_index, key in enumerate(canonical_keys):
        counts = H[row_index]
        fired = counts > 0
        total = int(counts.sum())
        meta = lang_meta[key]
        script = meta.get("script_iso15924", "?")

        direct_codes = DIRECT_CANONICAL_TO_CLM_CODES.get(key)
        if direct_codes:
            allowed_mask = mask_for_codes(direct_codes, code_to_bit)
            basis = "direct_clm_bits"
            rejection_kind = "hard_language_impossible"
            direct_count += 1
        else:
            clm_scripts = SCRIPT_TO_CLM_SCRIPTS.get(script, set())
            allowed_mask = mask_for_scripts(clm_scripts, manifest_languages)
            if allowed_mask:
                basis = "script_proxy_bits"
                rejection_kind = "script_proxy_incompatible"
                script_proxy_count += 1
            else:
                basis = "no_clm_script_bits"
                rejection_kind = "not_available"
                no_basis_count += 1

        if allowed_mask:
            compatible = evaluable & np.array(
                [(int(mask) & allowed_mask) != 0 for mask in bm_and],
                dtype=bool,
            )
            rejected = fired & evaluable & ~compatible
            compatible_fired = fired & compatible
        else:
            compatible = np.zeros(H.shape[1], dtype=bool)
            rejected = np.zeros(H.shape[1], dtype=bool)
            compatible_fired = np.zeros(H.shape[1], dtype=bool)

        unknown_fired = fired & unknown
        substrate_fired = fired & substrate
        zero_decoded_fired = fired & zero_decoded

        rejected_mass = int(counts[rejected].sum())
        unknown_mass = int(counts[unknown_fired].sum())
        compatible_mass = int(counts[compatible_fired].sum())
        substrate_mass = int(counts[substrate_fired].sum())
        zero_decoded_mass = int(counts[zero_decoded_fired].sum())

        rows.append(
            {
                "canonical_key": key,
                "row_index": row_index,
                "iso_639_3": meta.get("iso_639_3", ""),
                "script": script,
                "name": meta.get("name", ""),
                "basis": basis,
                "rejection_kind": rejection_kind,
                "allowed_codes": ",".join(direct_codes or []),
                "allowed_mask_signature": signature(allowed_mask, bit_to_code),
                "sample_total": total,
                "fired_token_types": int(fired.sum()),
                "compatible_mass": compatible_mass,
                "compatible_mass_pct": fmt_pct(compatible_mass, total),
                "rejected_decoded_mass": rejected_mass,
                "rejected_decoded_mass_pct": fmt_pct(rejected_mass, total),
                "unknown_standalone_mass": unknown_mass,
                "unknown_standalone_mass_pct": fmt_pct(unknown_mass, total),
                "substrate_mass": substrate_mass,
                "substrate_mass_pct": fmt_pct(substrate_mass, total),
                "zero_decoded_mask_mass": zero_decoded_mass,
                "zero_decoded_mask_mass_pct": fmt_pct(zero_decoded_mass, total),
                "rejected_decoded_token_types": int(rejected.sum()),
                "unknown_standalone_token_types": int(unknown_fired.sum()),
            }
        )

        if basis == "direct_clm_bits" and rejected_mass:
            idx = np.where(rejected)[0]
            idx = idx[np.argsort(-counts[idx])][:50]
            for rank, token_id in enumerate(idx, 1):
                mask = int(bm_and[token_id])
                top_rows.append(
                    {
                        "canonical_key": key,
                        "rank": rank,
                        "token_id": int(token_id),
                        "decoded": repr(decoded[token_id]),
                        "count": int(counts[token_id]),
                        "share_pct": fmt_pct(int(counts[token_id]), total),
                        "status": str(status[token_id]),
                        "bitmask_popcount": int(mask.bit_count()),
                        "bitmask_signature": signature(mask, bit_to_code),
                    }
                )

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        header = list(rows[0])
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row[col]) for col in header) + "\n")

    with OUT_TOP.open("w", encoding="utf-8") as f:
        header = list(top_rows[0]) if top_rows else [
            "canonical_key", "rank", "token_id", "decoded", "count", "share_pct",
            "status", "bitmask_popcount", "bitmask_signature",
        ]
        f.write("\t".join(header) + "\n")
        for row in top_rows:
            f.write("\t".join(str(row[col]) for col in header) + "\n")

    direct_rows = [row for row in rows if row["basis"] == "direct_clm_bits"]
    summary = {
        "n_rows": len(rows),
        "n_direct_clm_rows": direct_count,
        "n_script_proxy_rows": script_proxy_count,
        "n_no_clm_script_rows": no_basis_count,
        "outputs": {
            "summary_tsv": OUT_SUMMARY.name,
            "direct_top_tokens_tsv": OUT_TOP.name,
        },
        "methodology": (
            "For rows with direct CLM bits, rejected_decoded_mass is a hard "
            "language-membership rejection: decoded token text fired in the "
            "dataset but bitmask_and has none of that language group's bits. "
            "partial_utf8/byte_unmapped/special tokens are separated as "
            "unknown_standalone, not counted as rejections. Rows without direct "
            "CLM bits use script_proxy_bits when the script is modeled; that is "
            "a coarse script-family incompatibility signal, not a language-level "
            "proof."
        ),
        "direct_rows_top_rejected_mass_pct": [
            {
                "canonical_key": row["canonical_key"],
                "rejected_decoded_mass_pct": row["rejected_decoded_mass_pct"],
                "unknown_standalone_mass_pct": row["unknown_standalone_mass_pct"],
            }
            for row in sorted(
                direct_rows,
                key=lambda r: float(r["rejected_decoded_mass_pct"]),
                reverse=True,
            )[:20]
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"wrote {OUT_SUMMARY}")
    print(f"wrote {OUT_TOP}")
    print(f"wrote {OUT_JSON}")
    print(
        f"basis counts: direct={direct_count}, "
        f"script_proxy={script_proxy_count}, no_basis={no_basis_count}"
    )


if __name__ == "__main__":
    main()
