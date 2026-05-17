"""Build the curated + backfilled ship variant.

Goal: 17,408 added tokens, all non-curated, vocab = 148,480 (Apertus
front-end contract preserved + 256-aligned), built by walking the full
C3 merge sequence in order and:
  1. SKIPPING any merge whose result is in `02_1_5/manifests/removal_list.jsonl`
  2. SKIPPING any merge whose left or right component is in the skipped
     set (BPE cascade — a merge that depends on a removed token can't
     be applied at tokenization time)
  3. Accepting the first 17,408 surviving merges

The skipped+cascaded merges are reported. We expect the survivors to
come from a slightly deeper slice of the c3_full merge order than
17,408 — say, position ~17,500 — because we're consuming the
removal_list (104 over the full 25,600 vocab) at a roughly proportional
rate and a small number of cascade-skips.

Output: variants/c3_added_17408_curated_padded/tokenizer.json
        manifests/curated_padded_at_17408_manifest.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path("/home/foivos/Projects/glossapi-tokenizer-extension")
SSP = REPO / "subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
APERTUS_BASE = SSP / "tokenizers_local/apertus_base"
C3_FULL = SSP / "tokenizers_local/c3_full"
REMOVAL_LIST = REPO / (
    "subprojects/02_1_tokenizer_experiments/02_1_5_added_token_curation/"
    "manifests/removal_list.jsonl"
)
OUT_DIR = SSP / "variants/c3_added_17408_curated_padded"
MANIFEST_OUT = SSP / "manifests/curated_padded_at_17408_manifest.json"

BASE_VOCAB = 131_072
TARGET_ADDED = 17_408


def split_merge(m):
    if isinstance(m, str):
        parts = m.split(" ", 1)
        return parts if len(parts) == 2 else None
    if isinstance(m, list) and len(m) == 2:
        return m
    return None


def main() -> None:
    # Load full C3 tokenizer
    full = json.loads((C3_FULL / "tokenizer.json").read_text())
    full_vocab: dict[str, int] = full["model"]["vocab"]
    full_merges: list = full["model"]["merges"]
    id_to_tok = {v: k for k, v in full_vocab.items()}

    # Load Apertus-base merge count (we keep everything <= that index)
    base = json.loads((APERTUS_BASE / "tokenizer.json").read_text())
    base_n_merges = len(base["model"]["merges"])
    print(f"Apertus base merges : {base_n_merges:,}")
    print(f"C3 full merges      : {len(full_merges):,}")
    print(f"C3 full vocab       : {len(full_vocab):,}")
    print(f"target added merges : {TARGET_ADDED:,}")

    # Read removal_list — full 104 tokens — and convert id → string
    removed_strings: set[str] = set()
    removed_ids: set[int] = set()
    with REMOVAL_LIST.open() as fh:
        for line in fh:
            r = json.loads(line)
            tid = int(r["id"])
            removed_ids.add(tid)
            if tid in id_to_tok:
                removed_strings.add(id_to_tok[tid])
    print(f"removal-list ids    : {len(removed_ids)}  (any-cutoff)")
    print(f"removal-list strings: {len(removed_strings)}")

    # Build the new added-merge sequence by walking c3_full added merges in
    # order, skipping removed-result + cascade-skipping.
    accepted_merges: list = []
    accepted_tokens: set[str] = set(full_vocab.keys())
    # Start with all base-vocab strings as "live"; we'll subtract removed
    # strings + any merge we skip.
    # Actually: live = base-vocab. Then each accepted merge adds a token.
    base_vocab_strings = {tok for tok, tid in full_vocab.items()
                          if tid < BASE_VOCAB}
    live: set[str] = set(base_vocab_strings)

    skipped_removed = 0
    skipped_cascade = 0
    skipped_strings: set[str] = set()
    walked = 0
    for m in full_merges[base_n_merges:]:
        walked += 1
        s = split_merge(m)
        if s is None:
            continue
        left, right = s
        result = left + right
        # Skip if this merge's result is in the removal list
        if result in removed_strings:
            skipped_removed += 1
            skipped_strings.add(result)
            continue
        # Cascade skip: result depends on at least one removed/skipped token
        if left not in live or right not in live:
            skipped_cascade += 1
            skipped_strings.add(result)
            continue
        # Accept
        accepted_merges.append(m)
        live.add(result)
        if len(accepted_merges) >= TARGET_ADDED:
            break

    print()
    print(f"walked added merges : {walked:,}")
    print(f"  skipped (removed) : {skipped_removed}")
    print(f"  skipped (cascade) : {skipped_cascade}")
    print(f"  accepted          : {len(accepted_merges):,}  (target {TARGET_ADDED:,})")
    if len(accepted_merges) < TARGET_ADDED:
        print(
            f"  ! short by {TARGET_ADDED - len(accepted_merges)} — "
            "ran out of c3_full merges; would need a deeper source vocab"
        )
        return

    # Build new vocab: Apertus base verbatim + accepted-merge results
    new_vocab: dict[str, int] = {}
    # First: all base vocab entries with original ids (0..131,071)
    for tok, tid in full_vocab.items():
        if tid < BASE_VOCAB:
            new_vocab[tok] = tid
    # Add accepted merges as new ids starting at BASE_VOCAB
    next_id = BASE_VOCAB
    for m in accepted_merges:
        s = split_merge(m)
        result = s[0] + s[1]
        new_vocab[result] = next_id
        next_id += 1
    new_total = next_id
    print(f"new total vocab     : {new_total:,}  "
          f"(/128 = {new_total/128}, /256 = {new_total/256})")
    assert new_total == BASE_VOCAB + TARGET_ADDED, \
        f"vocab math: expected {BASE_VOCAB + TARGET_ADDED}, got {new_total}"

    # Build the new tokenizer.json
    out = json.loads(json.dumps(full))  # deep-copy
    out["model"]["vocab"] = new_vocab
    out["model"]["merges"] = full["model"]["merges"][:base_n_merges] + accepted_merges

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tokenizer.json").write_text(json.dumps(out, ensure_ascii=False))
    for sib in C3_FULL.iterdir():
        if sib.name == "tokenizer.json":
            continue
        if sib.is_file():
            shutil.copy2(sib, OUT_DIR / sib.name)

    # Manifest
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps({
        "variant_id": "add_17408_curated_padded",
        "added_tokens": TARGET_ADDED,
        "total_vocab": new_total,
        "alignment_128": new_total % 128 == 0,
        "alignment_256": new_total % 256 == 0,
        "apertus_base_preserved_verbatim": True,  # ids 0..131,071 unchanged
        "removal_list_source": str(REMOVAL_LIST),
        "removal_list_size_full": len(removed_ids),
        "merges_walked_from_c3_full": walked,
        "merges_skipped_for_removal": skipped_removed,
        "merges_skipped_for_cascade": skipped_cascade,
        "merges_accepted": len(accepted_merges),
        "ship_artifact_path": str(OUT_DIR / "tokenizer.json"),
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT_DIR / 'tokenizer.json'}")
    print(f"wrote {MANIFEST_OUT}")

    # Spot-check via transformers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(OUT_DIR))
    print(f"\nloaded via AutoTokenizer: vocab_size = {tok.vocab_size:,}")
    for s in ["Καλημέρα, ο κόσμος είναι μεγάλος.",
             "Η νεοελληνική γλώσσα είναι όμορφη.",
             "Το θέμα μας είναι σήμερα η εκπαίδευση.",
             "Ο νεοδιοριζόμενος υπουργός ψηφίζει.",  # check post-cutoff content
             ]:
        enc = tok.encode(s, add_special_tokens=False)
        print(f"  {repr(s):<55s} → {len(enc):>2d} tokens")


if __name__ == "__main__":
    main()
