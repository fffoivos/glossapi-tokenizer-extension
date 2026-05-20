"""Build the two Apertus-compatible ship tokenizer bundles and run all the
structural-integrity + AutoTokenizer-load + fertility checks documented in
SHIP_TOKENIZER_RECONSTRUCTION.md.

Two bundles are built:
  - apertus_greek_modern_only_148480/   — base + C3 modern Greek (+17,408).
    Use this for the three-arm Vanilla/ReTok/Distillation comparison.
  - apertus_greek_extended_153600/      — modern + polytonic (+5,120 on top of 148,480).
    Use this for the downstream polytonic specialization arm.

Both source variants on disk (c3_added_17408_curated_padded and
c3p_poly_added_5120) have a broken tokenizer_config.json with
`tokenizer_class: TokenizersBackend` that HF AutoTokenizer can't load.
We fix this by copying Apertus's canonical `tokenizer_config.json` over.

Run as:
  /home/foivos/.venvs/glossapi-merge-docling/bin/python3 build_and_verify_ship_tokenizer.py
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = Path("/home/foivos/Projects/glossapi-tokenizer-extension")
SHIP_ROOT = REPO / "subprojects/03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/ship"
SHIP_MODERN = SHIP_ROOT / "apertus_greek_modern_only_148480"
SHIP_EXTENDED = SHIP_ROOT / "apertus_greek_extended_153600"
POLY_SRC = REPO / "subprojects/02_1_tokenizer_experiments/02_1_polytonic_greek_extension/analysis/c3p_polytonic_20260518T_impl/variants/c3p_poly_added_5120"
C3_SRC = REPO / "subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/variants/c3_added_17408_curated_padded"
APERTUS = Path("/home/foivos/.cache/huggingface/hub/models--swiss-ai--Apertus-8B-2509/snapshots/3162c99675aa588097cecd4a24b9aa1f712af477")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build_bundle(ship_dir, src_dir, name, vocab_size, added_modern, added_poly, purpose):
    ship_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_dir / "tokenizer.json", ship_dir / "tokenizer.json")
    # Replace broken tokenizer_config.json with Apertus's canonical one
    shutil.copy(APERTUS / "tokenizer_config.json", ship_dir / "tokenizer_config.json")
    shutil.copy(APERTUS / "special_tokens_map.json", ship_dir / "special_tokens_map.json")

    manifest = {
        "name": name,
        "purpose": purpose,
        "vocab_size": vocab_size,
        "base_apertus_vocab": 131072,
        "added_modern_greek_c3": added_modern,
        "added_polytonic_greek": added_poly,
        "alignment": {"128": vocab_size % 128 == 0, "256": vocab_size % 256 == 0, "factor_256": vocab_size // 256},
        "first_1000_preserved": True,
        "front_end_contract": {
            "normalizer": None,
            "pre_tokenizer": "Sequence(Split(GPT-2 regex) → ByteLevel)",
            "decoder": "ByteLevel",
            "model_type": "BPE",
            "special_tokens": {"unk": 0, "bos": 1, "eos": 2, "pad": 3, "reserved_count": 1000},
        },
        "loadable_via": "transformers.AutoTokenizer.from_pretrained() → PreTrainedTokenizerFast",
        "sha256": {fn: sha256(ship_dir / fn) for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")},
    }
    (ship_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  ✓ {ship_dir.name}/ assembled (vocab {vocab_size:,})")


def verify_structure(ship_dir, expected_vocab, has_polytonic):
    ap = json.load(open(APERTUS / "tokenizer.json"))
    c3 = json.load(open(C3_SRC / "tokenizer.json"))
    sh = json.load(open(ship_dir / "tokenizer.json"))

    def idx_map(t):
        return {i: tok for tok, i in t["model"]["vocab"].items()}

    ap_idx, c3_idx, sh_idx = idx_map(ap), idx_map(c3), idx_map(sh)

    checks = []

    def chk(label, ok):
        checks.append((label, ok))
        print(f"  {'✓' if ok else 'FAIL'}  {label}")

    print(f"\n=== Structural integrity: {ship_dir.name} ===")
    chk(f"vocab_size = {expected_vocab:,}", len(sh["model"]["vocab"]) == expected_vocab)
    chk(f"256-aligned ({expected_vocab:,} = 256 × {expected_vocab // 256})", expected_vocab % 256 == 0)
    chk("first 1000 ids identical to Apertus", all(ap_idx.get(i) == sh_idx.get(i) for i in range(1000)))
    chk("ids 0..131,071 identical to Apertus", all(ap_idx.get(i) == sh_idx.get(i) for i in range(131072)))
    chk("ids 131,072..148,479 identical to C3 17,408 ship",
        all(c3_idx.get(i) == sh_idx.get(i) for i in range(131072, 148480)))
    if has_polytonic:
        chk("5,120 polytonic ids are new (no overlap with C3 base)",
            sum(1 for i in range(148480, 153600) if sh_idx.get(i) in c3["model"]["vocab"]) == 0)
    chk("normalizer = null (Apertus contract)", sh.get("normalizer") is None)
    chk("pre_tokenizer identical to Apertus", sh.get("pre_tokenizer") == ap.get("pre_tokenizer"))
    chk("decoder identical to Apertus", sh.get("decoder") == ap.get("decoder"))
    chk("model.type = BPE", sh["model"].get("type") == "BPE")
    chk("added_tokens count = 1000", len(sh.get("added_tokens", [])) == 1000)
    chk("added_tokens identical to Apertus",
        {(t["id"], t["content"]) for t in sh.get("added_tokens", [])}
        == {(t["id"], t["content"]) for t in ap.get("added_tokens", [])})
    expected_merges_delta = 17408 + (5120 if has_polytonic else 0)
    chk(f"merges: +{expected_merges_delta:,} over Apertus base",
        len(sh["model"]["merges"]) == len(ap["model"]["merges"]) + expected_merges_delta)

    return all(ok for _, ok in checks)


def verify_load_and_fertility(ship_dir, label, expected_vocab):
    print(f"\n=== AutoTokenizer load + fertility: {ship_dir.name} ===")
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(str(ship_dir))
    ap = AutoTokenizer.from_pretrained(str(APERTUS))

    assert tk.vocab_size == expected_vocab, f"vocab mismatch: {tk.vocab_size} != {expected_vocab}"
    print(f"  class={type(tk).__name__}  vocab={tk.vocab_size}  "
          f"bos={tk.bos_token_id} eos={tk.eos_token_id} pad={tk.pad_token_id} unk={tk.unk_token_id}")

    samples = [
        ("modern Greek (web)", "Η ελληνική γλώσσα είναι μια από τις παλαιότερες γλώσσες που μιλιούνται."),
        ("polytonic (NT)", "Ἐν ἀρχῇ ἦν ὁ Λόγος, καὶ ὁ Λόγος ἦν πρὸς τὸν Θεόν, καὶ Θεὸς ἦν ὁ Λόγος."),
        ("Katharevousa", "Ἡ φιλοσοφία τῶν ἀρχαίων Ἑλλήνων ἐθεωρεῖτο ὡς πηγὴ σοφίας."),
        ("academic", "Σύμφωνα με την έρευνα του πανεπιστημίου, η μεθοδολογία αναπτύχθηκε από τους ερευνητές."),
        ("legal", "Σύμφωνα με το άρθρο 5 του νόμου 1234/2020, η εφαρμογή της διάταξης ορίζεται από τον Υπουργό."),
        ("English (control)", "The Greek language is one of the oldest languages still spoken today."),
        ("Russian (control)", "Греческий язык — один из древнейших живых языков мира."),
    ]
    print(f"  {'register':<22}  {'apertus':>7}  {label:>11}  {'% saved':>7}")
    for sample_label, txt in samples:
        a = len(ap.encode(txt, add_special_tokens=False))
        p = len(tk.encode(txt, add_special_tokens=False))
        pct = ((a - p) / a * 100) if a else 0
        print(f"  {sample_label:<22}  {a:>7}  {p:>11}  {pct:>+6.1f}%")


def main():
    print("=== Assembly ===")
    build_bundle(
        SHIP_MODERN, C3_SRC,
        name="apertus_greek_modern_only_148480",
        vocab_size=148480,
        added_modern=17408,
        added_poly=0,
        purpose=(
            "Apertus-compatible modern-Greek-only tokenizer: base 131,072 + C3 modern "
            "Greek (+17,408 curated+backfilled) = 148,480 = 256 × 580. Use this for the "
            "three-arm Vanilla/ReTok/Distillation init comparison."
        ),
    )
    build_bundle(
        SHIP_EXTENDED, POLY_SRC,
        name="apertus_greek_extended_153600",
        vocab_size=153600,
        added_modern=17408,
        added_poly=5120,
        purpose=(
            "Apertus-compatible composite tokenizer: base 131,072 + C3 modern Greek "
            "(+17,408) + polytonic/ancient Greek (+5,120) = 153,600 = 256 × 600. Use "
            "this for the downstream polytonic specialization arm."
        ),
    )

    all_ok = True
    all_ok &= verify_structure(SHIP_MODERN, 148480, has_polytonic=False)
    all_ok &= verify_structure(SHIP_EXTENDED, 153600, has_polytonic=True)
    if not all_ok:
        print("\nFAIL: structural checks did not all pass.")
        sys.exit(1)

    verify_load_and_fertility(SHIP_MODERN, "modern148480", 148480)
    verify_load_and_fertility(SHIP_EXTENDED, "extended153600", 153600)
    print("\n✓ Both ship bundles are ready.")


if __name__ == "__main__":
    main()
