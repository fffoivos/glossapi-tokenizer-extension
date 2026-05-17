"""Build curated twin variants for the candidate ship cutoffs.

For each of {11264, 12288}, take the raw cutoff variant produced by
01_build_variants.sh and apply 02_1_5/manifests/removal_list.jsonl to
the merge table (Option 2 from 02_1_5/CURATION_REPORT.md).

A merge-graph validator runs first: for each removed token, walk the
merges of every kept token at any later id and confirm none of them
step through the removed merge pair. If validation fails for a
cutoff, that curated variant is SKIPPED (logged, not raised) so the
rest of the sweep can still complete.

The output tokenizer dirs:
  variants/c3_added_11264_curated/
  variants/c3_added_12288_curated/

Each is loadable via transformers.AutoTokenizer.from_pretrained.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path("/home/foivos/Projects/glossapi-tokenizer-extension")
SSP = REPO / "subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
VARIANTS_DIR = SSP / "variants"
REMOVAL_LIST = REPO / (
    "subprojects/02_1_tokenizer_experiments/02_1_5_added_token_curation/"
    "manifests/removal_list.jsonl"
)

CANDIDATE_CUTOFFS = [11264, 12288, 15360, 17408, 20480, 25600]


def load_removal_ids() -> set[int]:
    """Token ids the curation policy removes (from the 02_1_5 manifest)."""
    ids: set[int] = set()
    with REMOVAL_LIST.open() as fh:
        for line in fh:
            r = json.loads(line)
            ids.add(int(r["id"]))
    return ids


def validate_merge_graph(tokenizer_json: dict, removed_ids: set[int]) -> list[str]:
    """Return a list of hazards: kept tokens whose merge chain depends
    on a removed merge pair. Empty list = safe to prune."""
    model = tokenizer_json["model"]
    vocab: dict[str, int] = model["vocab"]
    merges: list[str] = model["merges"]
    id_to_tok = {v: k for k, v in vocab.items()}

    # Build the set of merge pairs whose RESULT token is removed
    removed_toks = {id_to_tok[i] for i in removed_ids if i in id_to_tok}
    # A "merge" entry is either "<left> <right>" or ["<left>","<right>"]
    # depending on tokenizers library version. Normalize.
    def split_merge(m):
        if isinstance(m, str):
            parts = m.split(" ", 1)
            return parts if len(parts) == 2 else None
        if isinstance(m, list) and len(m) == 2:
            return m
        return None

    removed_merge_pairs: set[tuple[str, str]] = set()
    for m in merges:
        s = split_merge(m)
        if s is None:
            continue
        left, right = s
        result = left + right
        if result in removed_toks:
            removed_merge_pairs.add((left, right))

    # Now retokenize every kept token via greedy left-to-right BPE
    # using the FULL merge order. If a kept token's tokenization passes
    # through one of removed_merge_pairs, it's a hazard.
    #
    # Lightweight check: a kept token whose surface string contains a
    # removed surface string as a "merge atom" is at risk. We use a
    # simpler structural test: for each kept token T, does its surface
    # form contain a removed token T' as a strict prefix such that
    # removing the merges that produce T' would orphan T? In BPE that
    # happens iff at some merge step T' must form first.
    #
    # We approximate this by walking each kept token's char sequence and
    # applying the merge prefix UP TO (but not removing) the removed
    # merges; if the result still equals the token id, kept token is
    # safe. Otherwise it's a hazard.
    hazards: list[str] = []
    merge_priority: dict[tuple[str, str], int] = {}
    for i, m in enumerate(merges):
        s = split_merge(m)
        if s is not None:
            merge_priority[(s[0], s[1])] = i
    kept_pairs = {p for p in merge_priority.keys() if p not in removed_merge_pairs}

    for tid, tok in id_to_tok.items():
        if tid in removed_ids:
            continue
        # Try to retokenize tok using only kept merges. We do a simple
        # iterative greedy: start with one-char tokens (after special
        # handling for the ByteLevel prefix) and apply the next kept
        # merge that matches. If we can't reach a single piece, it's a
        # hazard.
        # For efficiency, only check tokens whose decoded form contains
        # the surface of any removed token (small subset).
        if not any(rt in tok for rt in removed_toks):
            continue
        pieces = list(tok)
        applied = True
        while applied and len(pieces) > 1:
            applied = False
            best_idx = None
            best_priority = None
            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                if pair in kept_pairs:
                    p = merge_priority[pair]
                    if best_priority is None or p < best_priority:
                        best_priority = p
                        best_idx = i
            if best_idx is not None:
                pieces = (
                    pieces[:best_idx]
                    + [pieces[best_idx] + pieces[best_idx + 1]]
                    + pieces[best_idx + 2:]
                )
                applied = True
        if len(pieces) != 1 or pieces[0] != tok:
            hazards.append(tok)
    return hazards


def build_curated(cutoff: int, removed_ids: set[int]) -> None:
    src = VARIANTS_DIR / f"c3_added_{cutoff}"
    dst = VARIANTS_DIR / f"c3_added_{cutoff}_curated"
    if not src.exists():
        print(f"[skip] {src} missing — run 01_build_variants.sh first")
        return

    tok_json_path = src / "tokenizer.json"
    tok_json = json.loads(tok_json_path.read_text())

    # Scope removed_ids to the in-cutoff slice
    BASE_VOCAB = 131_072
    in_cutoff = {i for i in removed_ids if i < BASE_VOCAB + cutoff}
    print(f"\n[cutoff {cutoff}] removable in-cutoff: {len(in_cutoff)}")

    if not in_cutoff:
        print(f"  → nothing to remove; copying src verbatim")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return

    # Merge-graph validation
    hazards = validate_merge_graph(tok_json, in_cutoff)
    if hazards:
        print(
            f"  ✗ {len(hazards)} merge-graph hazards — skipping curated build."
        )
        print("    Sample hazards:", hazards[:5])
        return

    # Build the pruned vocab + merges
    model = tok_json["model"]
    vocab: dict[str, int] = model["vocab"]
    id_to_tok = {v: k for k, v in vocab.items()}
    removed_toks = {id_to_tok[i] for i in in_cutoff if i in id_to_tok}

    new_vocab = {tok: tid for tok, tid in vocab.items() if tid not in in_cutoff}
    # Renumber to be contiguous starting at 0
    sorted_items = sorted(new_vocab.items(), key=lambda kv: kv[1])
    new_vocab = {tok: i for i, (tok, _) in enumerate(sorted_items)}
    model["vocab"] = new_vocab

    # Drop merges whose result is a removed token
    new_merges = []
    for m in model["merges"]:
        if isinstance(m, str):
            parts = m.split(" ", 1)
        elif isinstance(m, list) and len(m) == 2:
            parts = m
        else:
            new_merges.append(m)
            continue
        if (parts[0] + parts[1]) in removed_toks:
            continue
        new_merges.append(m)
    model["merges"] = new_merges

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "tokenizer.json").write_text(
        json.dumps(tok_json, ensure_ascii=False)
    )
    # Copy the rest of the config files
    for sib in src.iterdir():
        if sib.name == "tokenizer.json":
            continue
        if sib.is_file():
            shutil.copy2(sib, dst / sib.name)
    print(f"  ✓ wrote {dst}  (vocab {len(new_vocab)}, merges {len(new_merges)})")


def main() -> None:
    removed_ids = load_removal_ids()
    print(f"Total removable tokens at full 25,600: {len(removed_ids)}")
    for c in CANDIDATE_CUTOFFS:
        build_curated(c, removed_ids)


if __name__ == "__main__":
    main()
