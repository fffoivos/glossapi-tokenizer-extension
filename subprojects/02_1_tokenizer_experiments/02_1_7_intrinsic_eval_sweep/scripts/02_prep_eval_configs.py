"""Build the four TokEval language/tokenizer configs the sweep needs.

Outputs (all under configs/):
  - cutoff_sweep_tokenizers.json  — 15 tokenizers (13 raw + 2 curated)
  - apertus55_lang_config.json    — 55-language multilingual config (for TFG)
  - greek_only_lang_config.json   — single-language Greek (for per-lang Greek)
  - our_holdouts_lang_config.json — 3 in-house Greek slices wrapped as a
                                    TokEval language config

Apertus-55 selection: the Apertus paper §2.2 does not publish the
exact 55-language list. We use a defensible proxy: the union of
(a) the 13 in TokEval's core_lang_config.json,
(b) Modern Greek (ell_Grek) — added explicitly per reviewer High-2,
(c) the highest-firing languages from our PMI promotion pass
    (02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/summary.tsv)
filtered to those with masked_count > 100, capped at 55.

The exact list is recorded in the manifest at build time so the choice
is reproducible. Future refinement to a paper-exact 55 only requires
swapping the input list; the rest of the pipeline is invariant.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path("/home/foivos/Projects/glossapi-tokenizer-extension")
SSP = REPO / "subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
TOKEVAL = SSP / "vendor/tokenizer-intrinsic-evals"
CONFIGS_OUT = SSP / "configs"
VARIANTS_DIR = SSP / "variants"
VARIANTS_ARCHIVE = SSP / "variants/_archive"


def find_variant(name: str) -> Path | None:
    """Look in variants/ first; fall back to variants/_archive/.
    Returns the path to the dir if found, else None.
    """
    for root in (VARIANTS_DIR, VARIANTS_ARCHIVE):
        p = root / name
        if p.exists():
            return p
    return None
MANIFESTS = SSP / "manifests"

PMI_SUMMARY = REPO / (
    "subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/"
    "analysis/main_token_sets_pmi/summary.tsv"
)
FLORES_PLUS_CFG = TOKEVAL / "configs/flores+_lang_config.json"
CORE_CFG = TOKEVAL / "configs/core_lang_config.json"

OUR_HOLDOUTS = {
    "virgin_hplt": "/home/foivos/data/glossapi_work/runs/c3_holdouts_20260511/virgin_hplt.parquet",
    "C3_val_clean": "/home/foivos/data/glossapi_work/runs/c3_holdouts_20260511/C3_val_clean.parquet",
    "C3_test_clean": "/home/foivos/data/glossapi_work/runs/c3_holdouts_20260511/C3_test_clean.parquet",
}


def build_tokenizers_config() -> None:
    out_path = CONFIGS_OUT / "cutoff_sweep_tokenizers.json"
    # TokEval expects dict-of-dicts keyed by display name. We also emit a
    # sibling metadata file (`cutoff_sweep_tokenizers_meta.json`) that the
    # aggregator reads to recover added_tokens / curated for each name.
    cfg: dict = {}
    meta: list = []
    p = find_variant("apertus_base")
    if p:
        cfg["apertus_base"] = {"class": "hf", "path": str(p)}
        meta.append({"name": "apertus_base", "added_tokens": 0, "curated": False})
    RAW_CUTOFFS = [
        1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192,
        9216, 10240, 11264, 12288,
        13312, 14336, 15360, 16384, 17408, 18432, 19456, 20480,
        21504, 22528, 23552, 24576, 25600,
    ]
    for n in RAW_CUTOFFS:
        p = find_variant(f"c3_added_{n}")
        if p is None:
            continue
        cfg[f"add_{n}"] = {"class": "hf", "path": str(p)}
        meta.append({"name": f"add_{n}", "added_tokens": n, "curated": False})
    for n in [11264, 12288, 15360, 17408, 20480, 25600]:
        p = find_variant(f"c3_added_{n}_curated")
        if p is None:
            continue
        cfg[f"add_{n}_curated"] = {"class": "hf", "path": str(p)}
        meta.append({"name": f"add_{n}_curated", "added_tokens": n, "curated": True})
    # The canonical ship artifact (curated+backfilled at 17,408)
    p = find_variant("c3_added_17408_curated_padded")
    if p is not None:
        cfg["add_17408_curated_padded"] = {"class": "hf", "path": str(p)}
        meta.append({"name": "add_17408_curated_padded", "added_tokens": 17408, "curated": True})
    out_path.write_text(json.dumps(cfg, indent=2))
    (CONFIGS_OUT / "cutoff_sweep_tokenizers_meta.json").write_text(
        json.dumps({"tokenizers": meta}, indent=2)
    )
    print(f"wrote {out_path}  ({len(cfg)} tokenizers)")


def build_apertus55_config() -> None:
    """Apertus-55 proxy: see module docstring."""
    flores_plus = json.loads(FLORES_PLUS_CFG.read_text())["languages"]
    core = json.loads(CORE_CFG.read_text())["languages"]

    pmi_keys: list[str] = []
    with PMI_SUMMARY.open() as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for r in rdr:
            if int(r["masked_count"]) >= 100:
                pmi_keys.append(r["target_key"])

    # Build selection: start with core (13), add Greek, then PMI-promoted
    # keys in decreasing masked-count order, capped at 55
    selected: list[str] = list(core.keys())
    if "ell_Grek" not in selected:
        selected.append("ell_Grek")
    for k in pmi_keys:
        if k in selected:
            continue
        if k not in flores_plus:
            # PMI key has no FLORES+ slice — skip
            continue
        selected.append(k)
        if len(selected) >= 55:
            break
    # Greek must be in
    assert "ell_Grek" in selected, "ell_Grek somehow not in apertus55 — bug"
    # Constrain to entries that actually exist in flores+
    selected = [k for k in selected if k in flores_plus]
    out_path = CONFIGS_OUT / "apertus55_lang_config.json"
    out = {
        "languages": {k: flores_plus[k] for k in selected},
        "analysis_groups": {
            # TFG benefits from a single "all" group across all selected langs
            "all": {"all": selected},
        },
    }
    # Rewrite data_path to be relative to the running TokEval root
    for k, v in out["languages"].items():
        if v.get("data_path", "").startswith("parallel/"):
            v["data_path"] = str(TOKEVAL / v["data_path"])
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}  ({len(selected)} languages including ell_Grek)")
    (MANIFESTS / "apertus55_languages.txt").write_text(
        "\n".join(selected) + "\n"
    )


def build_greek_only_config() -> None:
    flores_plus = json.loads(FLORES_PLUS_CFG.read_text())["languages"]
    out_path = CONFIGS_OUT / "greek_only_lang_config.json"
    entry = dict(flores_plus["ell_Grek"])
    if entry["data_path"].startswith("parallel/"):
        entry["data_path"] = str(TOKEVAL / entry["data_path"])
    out = {
        "languages": {"ell_Grek": entry},
        "analysis_groups": {"all": {"all": ["ell_Grek"]}},
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}  (1 language)")


def build_our_holdouts_config() -> None:
    """Wrap our 3 in-house held-outs as TokEval 'languages'.

    TokEval needs a `data_path` per language pointing at a text file
    (one document/sentence per line). Our held-outs are parquet. The
    runner script (03b_run_our_suite.sh) uses our own 02_1_3 harness
    for the in-house slices instead — TokEval reads only the FLORES+
    text files.

    This config is emitted only for documentation / future-use.
    """
    out_path = CONFIGS_OUT / "our_holdouts_lang_config.json"
    out = {
        "_note": (
            "Our held-outs are parquet, not txt. TokEval is not run on "
            "them; the 02_1_3 harness is used instead. This file "
            "documents the slice locations for 03b_run_our_suite.sh."
        ),
        "slices": OUR_HOLDOUTS,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}  ({len(OUR_HOLDOUTS)} in-house slices)")


def main() -> None:
    CONFIGS_OUT.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    build_tokenizers_config()
    build_apertus55_config()
    build_greek_only_config()
    build_our_holdouts_config()


if __name__ == "__main__":
    main()
