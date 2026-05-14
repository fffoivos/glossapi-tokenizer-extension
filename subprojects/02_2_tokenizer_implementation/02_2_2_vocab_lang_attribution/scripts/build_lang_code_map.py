"""Build the canonical-language code map for the vocab-attribution run.

Canonical form: `<ISO 639-3>_<ISO 15924>` (FW-2's convention).

Inputs:
  - FW-2 distribution CSV (gives the authoritative 1811-language list)
  - Wikipedia 319 codes (hand-mapped where they differ from ISO conventions)
  - EuroParl 21 two-letter codes (ISO 639-1 → 639-3 + script)
  - ParaDocs 18 two-letter codes (same convention as EuroParl)

Output:
  - lang_code_map.json — { canonical_key: { sources: { source_name: local_code }, name, family } }
  - partition.json     — { worker_idx (0..7): [canonical_key, ...] }
"""

import csv
import json
import hashlib
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Wikipedia local-code → canonical (ISO 639-3 + script). Manually curated for
# the quirky codes; common ISO-639-1 codes get mechanical mapping.
# ---------------------------------------------------------------------------
WIKI_OVERRIDE = {
    # Code in Wikipedia ↔ canonical key
    "ab":  "abk_Cyrl",  "ace": "ace_Latn",  "ady": "ady_Cyrl",  "af":  "afr_Latn",
    "als": "als_Latn",  "alt": "alt_Cyrl",  "am":  "amh_Ethi",  "ami": "ami_Latn",
    "an":  "arg_Latn",  "ang": "ang_Latn",  "anp": "anp_Deva",  "ar":  "arb_Arab",
    "arc": "arc_Syrc",  "ary": "ary_Arab",  "arz": "arz_Arab",  "as":  "asm_Beng",
    "ast": "ast_Latn",  "atj": "atj_Latn",  "av":  "ava_Cyrl",  "avk": "avk_Latn",
    "awa": "awa_Deva",  "ay":  "aym_Latn",  "az":  "azj_Latn",  "azb": "azb_Arab",
    "ba":  "bak_Cyrl",  "ban": "ban_Latn",  "bar": "bar_Latn",  "bat-smg": "sgs_Latn",
    "bcl": "bcl_Latn",  "be":  "bel_Cyrl",  "be-x-old": "bel_Cyrl",
    "bg":  "bul_Cyrl",  "bh":  "bho_Deva",  "bi":  "bis_Latn",  "bjn": "bjn_Latn",
    "blk": "blk_Mymr",  "bm":  "bam_Latn",  "bn":  "ben_Beng",  "bo":  "bod_Tibt",
    "bpy": "bpy_Beng",  "br":  "bre_Latn",  "bs":  "bos_Latn",  "bug": "bug_Latn",
    "bxr": "bxr_Cyrl",  "ca":  "cat_Latn",  "cbk-zam": "cbk_Latn", "cdo": "cdo_Hani",
    "ce":  "che_Cyrl",  "ceb": "ceb_Latn",  "ch":  "cha_Latn",  "chr": "chr_Cher",
    "chy": "chy_Latn",  "ckb": "ckb_Arab",  "co":  "cos_Latn",  "cr":  "cre_Cans",
    "crh": "crh_Latn",  "cs":  "ces_Latn",  "csb": "csb_Latn",  "cu":  "chu_Cyrl",
    "cv":  "chv_Cyrl",  "cy":  "cym_Latn",  "da":  "dan_Latn",  "dag": "dag_Latn",
    "de":  "deu_Latn",  "din": "din_Latn",  "diq": "diq_Latn",  "dsb": "dsb_Latn",
    "dty": "dty_Deva",  "dv":  "div_Thaa",  "dz":  "dzo_Tibt",  "ee":  "ewe_Latn",
    "el":  "ell_Grek",  "eml": "egl_Latn",  "en":  "eng_Latn",  "eo":  "epo_Latn",
    "es":  "spa_Latn",  "et":  "ekk_Latn",  "eu":  "eus_Latn",  "ext": "ext_Latn",
    "fa":  "fas_Arab",  "ff":  "ful_Latn",  "fi":  "fin_Latn",  "fiu-vro": "vro_Latn",
    "fj":  "fij_Latn",  "fo":  "fao_Latn",  "fr":  "fra_Latn",  "frp": "frp_Latn",
    "frr": "frr_Latn",  "fur": "fur_Latn",  "fy":  "fry_Latn",  "ga":  "gle_Latn",
    "gag": "gag_Latn",  "gan": "gan_Hani",  "gcr": "gcr_Latn",  "gd":  "gla_Latn",
    "gl":  "glg_Latn",  "glk": "glk_Arab",  "gn":  "grn_Latn",  "gom": "gom_Deva",
    "gor": "gor_Latn",  "got": "got_Goth",  "gu":  "guj_Gujr",  "guc": "guc_Latn",
    "gur": "gur_Latn",  "guw": "guw_Latn",  "gv":  "glv_Latn",  "ha":  "hau_Latn",
    "hak": "hak_Hani",  "haw": "haw_Latn",  "he":  "heb_Hebr",  "hi":  "hin_Deva",
    "hif": "hif_Latn",  "hr":  "hrv_Latn",  "hsb": "hsb_Latn",  "ht":  "hat_Latn",
    "hu":  "hun_Latn",  "hy":  "hye_Armn",  "hyw": "hyw_Armn",  "ia":  "ina_Latn",
    "id":  "ind_Latn",  "ie":  "ile_Latn",  "ig":  "ibo_Latn",  "ik":  "ipk_Latn",
    "ilo": "ilo_Latn",  "inh": "inh_Cyrl",  "io":  "ido_Latn",  "is":  "isl_Latn",
    "it":  "ita_Latn",  "iu":  "iku_Cans",  "ja":  "jpn_Jpan",  "jam": "jam_Latn",
    "jbo": "jbo_Latn",  "jv":  "jav_Latn",  "ka":  "kat_Geor",  "kaa": "kaa_Cyrl",
    "kab": "kab_Latn",  "kbd": "kbd_Cyrl",  "kbp": "kbp_Latn",  "kcg": "kcg_Latn",
    "kg":  "kon_Latn",  "ki":  "kik_Latn",  "kk":  "kaz_Cyrl",  "kl":  "kal_Latn",
    "km":  "khm_Khmr",  "kn":  "kan_Knda",  "ko":  "kor_Hang",  "koi": "koi_Cyrl",
    "krc": "krc_Cyrl",  "ks":  "kas_Arab",  "ksh": "ksh_Latn",  "ku":  "kmr_Latn",
    "kv":  "kpv_Cyrl",  "kw":  "cor_Latn",  "ky":  "kir_Cyrl",  "la":  "lat_Latn",
    "lad": "lad_Latn",  "lb":  "ltz_Latn",  "lbe": "lbe_Cyrl",  "lez": "lez_Cyrl",
    "lfn": "lfn_Latn",  "lg":  "lug_Latn",  "li":  "lim_Latn",  "lij": "lij_Latn",
    "lld": "lld_Latn",  "lmo": "lmo_Latn",  "ln":  "lin_Latn",  "lo":  "lao_Laoo",
    "lt":  "lit_Latn",  "ltg": "ltg_Latn",  "lv":  "lvs_Latn",  "mad": "mad_Latn",
    "mai": "mai_Deva",  "map-bms": "jav_Latn",  "mdf": "mdf_Cyrl",  "mg":  "mlg_Latn",
    "mhr": "mhr_Cyrl",  "mi":  "mri_Latn",  "min": "min_Latn",  "mk":  "mkd_Cyrl",
    "ml":  "mal_Mlym",  "mn":  "khk_Cyrl",  "mni": "mni_Beng",  "mnw": "mnw_Mymr",
    "mr":  "mar_Deva",  "mrj": "mrj_Cyrl",  "ms":  "zsm_Latn",  "mt":  "mlt_Latn",
    "mwl": "mwl_Latn",  "my":  "mya_Mymr",  "myv": "myv_Cyrl",  "mzn": "mzn_Arab",
    "nah": "nah_Latn",  "nap": "nap_Latn",  "nb":  "nob_Latn",  "nds": "nds_Latn",
    "nds-nl": "nds_Latn", "ne":  "npi_Deva",  "new": "new_Deva",  "nia": "nia_Latn",
    "nl":  "nld_Latn",  "nn":  "nno_Latn",  "no":  "nob_Latn",  "nov": "nov_Latn",
    "nqo": "nqo_Nkoo",  "nrm": "nrf_Latn",  "nso": "nso_Latn",  "nv":  "nav_Latn",
    "ny":  "nya_Latn",  "oc":  "oci_Latn",  "olo": "olo_Latn",  "om":  "gaz_Latn",
    "or":  "ory_Orya",  "os":  "oss_Cyrl",  "pa":  "pan_Guru",  "pag": "pag_Latn",
    "pam": "pam_Latn",  "pap": "pap_Latn",  "pcd": "pcd_Latn",  "pcm": "pcm_Latn",
    "pdc": "pdc_Latn",  "pfl": "pfl_Latn",  "pi":  "pli_Deva",  "pih": "pih_Latn",
    "pl":  "pol_Latn",  "pms": "pms_Latn",  "pnb": "pnb_Arab",  "pnt": "pnt_Grek",
    "ps":  "pbt_Arab",  "pt":  "por_Latn",  "pwn": "pwn_Latn",  "qu":  "quz_Latn",
    "rm":  "roh_Latn",  "rmy": "rmy_Latn",  "rn":  "run_Latn",  "ro":  "ron_Latn",
    "roa-rup": "rup_Latn", "roa-tara": "ita_Latn",
    "ru":  "rus_Cyrl",  "rue": "rue_Cyrl",  "rw":  "kin_Latn",  "sa":  "san_Deva",
    "sah": "sah_Cyrl",  "sat": "sat_Olck",  "sc":  "srd_Latn",  "scn": "scn_Latn",
    "sco": "sco_Latn",  "sd":  "snd_Arab",  "se":  "sme_Latn",  "sg":  "sag_Latn",
    "sh":  "hbs_Latn",  "shi": "shi_Latn",  "shn": "shn_Mymr",  "si":  "sin_Sinh",
    "sk":  "slk_Latn",  "skr": "skr_Arab",  "sl":  "slv_Latn",  "sm":  "smo_Latn",
    "smn": "smn_Latn",  "sn":  "sna_Latn",  "so":  "som_Latn",  "sq":  "als_Latn",
    "sr":  "srp_Cyrl",  "srn": "srn_Latn",  "ss":  "ssw_Latn",  "st":  "sot_Latn",
    "stq": "stq_Latn",  "su":  "sun_Latn",  "sv":  "swe_Latn",  "sw":  "swh_Latn",
    "szl": "szl_Latn",  "szy": "szy_Latn",  "ta":  "tam_Taml",  "tay": "tay_Latn",
    "tcy": "tcy_Knda",  "te":  "tel_Telu",  "tet": "tet_Latn",  "tg":  "tgk_Cyrl",
    "th":  "tha_Thai",  "ti":  "tir_Ethi",  "tk":  "tuk_Latn",  "tl":  "tgl_Latn",
    "tly": "tly_Latn",  "tn":  "tsn_Latn",  "to":  "ton_Latn",  "tpi": "tpi_Latn",
    "tr":  "tur_Latn",  "trv": "trv_Latn",  "ts":  "tso_Latn",  "tt":  "tat_Cyrl",
    "tum": "tum_Latn",  "tw":  "twi_Latn",  "ty":  "tah_Latn",  "tyv": "tyv_Cyrl",
    "udm": "udm_Cyrl",  "ug":  "uig_Arab",  "uk":  "ukr_Cyrl",  "ur":  "urd_Arab",
    "uz":  "uzn_Latn",  "ve":  "ven_Latn",  "vec": "vec_Latn",  "vep": "vep_Latn",
    "vi":  "vie_Latn",  "vls": "vls_Latn",  "vo":  "vol_Latn",  "wa":  "wln_Latn",
    "war": "war_Latn",  "wo":  "wol_Latn",  "wuu": "wuu_Hani",  "xal": "xal_Cyrl",
    "xh":  "xho_Latn",  "xmf": "xmf_Geor",  "yi":  "ydd_Hebr",  "yo":  "yor_Latn",
    "za":  "zha_Latn",  "zea": "zea_Latn",  "zh":  "cmn_Hani",
    "zh-classical": "lzh_Hani", "zh-min-nan": "nan_Latn", "zh-yue": "yue_Hani",
    "zu":  "zul_Latn",
}


# EuroParl + ParaDocs use 2-letter ISO 639-1; same convention as Wikipedia. Reuse.
EP_2LETTER = ["bg","cs","da","de","el","en","es","et","fi","fr","hu","it","lt","lv","nl","pl","pt","ro","sk","sl","sv"]
PD_2LETTER = ["cs","de","es","fr","hi","hu","id","it","km","lo","my","ne","nl","pl","pt","sv","th","vi"]


def main():
    fw_csv = "/tmp/fw2_dist.csv"
    here = Path(__file__).resolve().parent
    out_map = here / "lang_code_map.json"
    out_part = here / "partition.json"

    # Build the canonical-key set seeded from FW-2 + FW-2-HQ.
    canonical = {}
    fw2_hq_subset = {"rus_Cyrl","cmn_Hani","deu_Latn","spa_Latn","jpn_Jpan","fra_Latn",
                     "ita_Latn","por_Latn","pol_Latn","nld_Latn","ind_Latn","tur_Latn",
                     "ces_Latn","arb_Arab","vie_Latn","swe_Latn","fas_Arab","hun_Latn",
                     "ell_Grek","dan_Latn"}

    with open(fw_csv) as f:
        for row in csv.DictReader(f):
            if row["split"] != "train": continue
            if row["subset"].endswith("_removed"): continue
            key = row["subset"]
            if key in canonical: continue
            try:
                docs = int(row["documents"])
                words = int(row["words"]) if row["words"] not in ("", "-") else 0
            except ValueError:
                docs = 0; words = 0
            in_hq = key in fw2_hq_subset
            canonical[key] = {
                "name": row["name"],
                "family": row["family"],
                "iso_639_3": row["code"],
                "script_iso15924": row["script"],
                "fw2_docs": docs,
                "fw2_words": words,
                "sources": {
                    "fineweb_2": key,
                }
            }
            if in_hq:
                canonical[key]["sources"]["fineweb_2_hq"] = key

    # Wikipedia: map every code we have an override for; collect orphans.
    wiki_orphans = []
    for wcode, ckey in WIKI_OVERRIDE.items():
        if ckey not in canonical:
            # Wikipedia code maps to a canonical key not in FW-2 — add it.
            iso, script = ckey.split("_", 1)
            canonical[ckey] = {
                "name": f"<wiki-only: {wcode}>",
                "family": "<unknown>",
                "iso_639_3": iso,
                "script_iso15924": script,
                "fw2_docs": 0,
                "fw2_words": 0,
                "sources": {}
            }
        canonical[ckey]["sources"].setdefault("clean_wikipedia", []).append(wcode)

    # EuroParl: map 2-letter codes via the same WIKI_OVERRIDE table.
    for code in EP_2LETTER:
        ckey = WIKI_OVERRIDE.get(code)
        if not ckey:
            print(f"WARN EuroParl code {code!r} has no canonical mapping", file=sys.stderr)
            continue
        if ckey not in canonical:
            iso, script = ckey.split("_", 1)
            canonical[ckey] = {
                "name": f"<europarl-only: {code}>",
                "family": "<unknown>",
                "iso_639_3": iso,
                "script_iso15924": script,
                "fw2_docs": 0,
                "fw2_words": 0,
                "sources": {}
            }
        canonical[ckey]["sources"].setdefault("europarl", []).append(code)

    # ParaDocs: same shape — non-English side per pair.
    for code in PD_2LETTER:
        ckey = WIKI_OVERRIDE.get(code)
        if not ckey:
            print(f"WARN ParaDocs code {code!r} has no canonical mapping", file=sys.stderr)
            continue
        if ckey not in canonical:
            iso, script = ckey.split("_", 1)
            canonical[ckey] = {
                "name": f"<paradocs-only: {code}>",
                "family": "<unknown>",
                "iso_639_3": iso,
                "script_iso15924": script,
                "fw2_docs": 0,
                "fw2_words": 0,
                "sources": {}
            }
        canonical[ckey]["sources"].setdefault("paradocs", []).append(code)

    # Special: English from FineWeb-Edu (separate of FW-2). Already keyed as eng_Latn from
    # WIKI_OVERRIDE 'en'; mark fineweb_edu source.
    canonical.setdefault("eng_Latn", {
        "name": "English",
        "family": "Indo-European",
        "iso_639_3": "eng",
        "script_iso15924": "Latn",
        "fw2_docs": 0,
        "fw2_words": 0,
        "sources": {}
    })
    canonical["eng_Latn"]["sources"]["fineweb_edu"] = "default"
    canonical["eng_Latn"]["sources"]["fineweb_hq"] = "default"
    canonical["eng_Latn"]["sources"]["dclm_edu"] = "default"

    # Output the map (sorted for stability).
    out = {k: canonical[k] for k in sorted(canonical.keys())}
    out_map.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {out_map} with {len(out)} canonical keys.")

    # Build the worker partition.
    N_WORKERS = 8
    partition = {str(i): [] for i in range(N_WORKERS)}
    for ckey in out.keys():
        h = hashlib.sha1(ckey.encode()).hexdigest()
        worker_idx = int(h, 16) % N_WORKERS
        partition[str(worker_idx)].append(ckey)
    out_part.write_text(json.dumps(partition, indent=2))
    sizes = {w: len(v) for w, v in partition.items()}
    print(f"Wrote {out_part}; per-worker counts: {sizes}")


if __name__ == "__main__":
    main()
