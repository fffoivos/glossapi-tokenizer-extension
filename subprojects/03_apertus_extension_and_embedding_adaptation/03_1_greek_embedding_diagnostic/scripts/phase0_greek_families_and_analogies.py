"""v2 §3.12 — Long-token families + analogy arithmetic for Greek.

§3.12.1: cluster Greek tokens of decoded length ≥ 4 by 3-letter root key;
compare intra-family cosine similarity to size-matched random baseline.

§3.12.2: hand-picked Mikolov-style analogy tests on Greek pairs.

Outputs:
  geometry/v2/greek_families_{E,U}.json     family stats + member lists
  geometry/v2/greek_families_{E,U}.md       human-readable preview
  geometry/v2/greek_analogies_{E,U}.json    per-analogy results
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
V2_DIR = ROOT / "geometry" / "v2"
CLASS_PATH = Path(
    "/home/foivos/runs/apertus_greek_diagnostic_20260511_v2/token_classification.jsonl"
)

MIN_LEN = 4
ROOT_KEY_LEN = 3
MIN_FAMILY_SIZE = 4
N_BOOTSTRAP = 10
SEED = 20260512


# Hand-picked analogy probes — these are CANDIDATES; many will be skipped
# if Apertus's tokenizer doesn't keep the relevant word as a single token.
ANALOGY_CANDIDATES = [
    # (label, a, b, c, expected_d) — meaning: vector(a) − vector(b) + vector(c) ≈ vector(expected_d)
    ("article-gender-masc-fem",  " ο", " το", " η", " ο"),       # trivial sanity
    ("singular-plural-1",        " παιδί",  " παιδιά",  " βιβλίο",  " βιβλία"),
    ("singular-plural-2",        " κόρη",   " κόρες",   " γυναίκα", " γυναίκες"),
    ("masc-fem-1",               " γιος",   " κόρη",    " αδελφός", " αδελφή"),
    ("verb-person-1",            " γράφω",  " γράφεις", " διαβάζω", " διαβάζεις"),
    ("verb-person-2",            " γράφω",  " γράφει",  " διαβάζω", " διαβάζει"),
    ("noun-adj-1",               " ελληνικά", " Ελλάδα", " γαλλικά", " Γαλλία"),
    ("place-day-1",              " Δευτέρα", " Τρίτη",  " Πέμπτη",  " Παρασκευή"),
]


def load_decoded_all_greek():
    groups = json.loads((ROOT / "geometry" / "groups_greek_vs_not.json").read_text())
    want = set(int(x) for x in groups["Greek"])
    decoded: dict[int, dict] = {}
    with CLASS_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            tid = int(r["id"])
            if tid in want:
                decoded[tid] = {
                    "raw_token": r.get("raw_token", ""),
                    "decoded_text": r.get("decoded_text", ""),
                }
    return groups["Greek"], decoded


def find_token_id(target_surface: str, decoded_all: dict, tok=None):
    """Return token-id whose decoded_text matches target exactly, else None."""
    for tid, info in decoded_all.items():
        if info["decoded_text"] == target_surface:
            return tid
    # Fall back to live tokenisation
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")
    ids = tok(target_surface, add_special_tokens=False)["input_ids"]
    if len(ids) == 1:
        return ids[0]
    return None


def family_analysis(M: np.ndarray, matrix: str, greek_ids, decoded):
    print(f"=== {matrix}: long-token families ===", flush=True)
    # Filter to decoded length ≥ MIN_LEN (after stripping leading space marker)
    long_tokens: list[tuple[int, str, str]] = []
    for tid in greek_ids:
        d = decoded[int(tid)]["decoded_text"].strip()
        if len(d) >= MIN_LEN:
            long_tokens.append((int(tid), d, decoded[int(tid)]["raw_token"]))
    print(f"  long tokens (len≥{MIN_LEN}): {len(long_tokens)}", flush=True)

    # Group by root key
    families: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for tid, dec, raw in long_tokens:
        key = dec[:ROOT_KEY_LEN]
        families[key].append((tid, dec, raw))
    families = {k: v for k, v in families.items() if len(v) >= MIN_FAMILY_SIZE}
    print(f"  families with ≥{MIN_FAMILY_SIZE} members: {len(families)}", flush=True)

    # For each family, compute intra-family cosine + baseline
    rng = np.random.default_rng(SEED)
    long_token_ids = np.asarray([t[0] for t in long_tokens], dtype=np.int64)
    long_token_rows = M[long_token_ids]
    long_token_rows_unit = long_token_rows / (np.linalg.norm(long_token_rows, axis=1, keepdims=True) + 1e-12)

    out_families = {}
    for key, members in sorted(families.items(), key=lambda kv: -len(kv[1])):
        mem_ids = np.asarray([m[0] for m in members], dtype=np.int64)
        rows = M[mem_ids]
        rows_unit = rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12)
        # Mean pairwise cosine (upper triangle)
        sim = rows_unit @ rows_unit.T
        n = sim.shape[0]
        if n < 2:
            continue
        iu = np.triu_indices(n, k=1)
        intra = float(sim[iu].mean())
        # Random size-matched baseline
        baseline_sims = []
        for _ in range(N_BOOTSTRAP):
            pick = rng.choice(long_token_rows_unit.shape[0], size=n, replace=False)
            r = long_token_rows_unit[pick]
            s = r @ r.T
            baseline_sims.append(s[iu].mean())
        baseline = float(np.median(baseline_sims))
        out_families[key] = {
            "n_members": n,
            "intra_family_cosine_mean": intra,
            "random_baseline_cosine_median": baseline,
            "tightness_ratio": intra / baseline if baseline != 0 else float("inf"),
            "members_preview": [
                {"id": int(m[0]), "decoded_text": m[1], "raw_token": m[2]}
                for m in members[:10]
            ],
            "n_total_members": n,
        }
    return out_families


def render_families_md(out_dict: dict, matrix: str) -> str:
    lines = [f"# Greek long-token families ({matrix} matrix)", ""]
    lines.append(f"_Filtered to tokens with decoded length ≥ {MIN_LEN}; "
                 f"grouped by first {ROOT_KEY_LEN} letters._")
    lines.append("")
    for key, fam in sorted(out_dict.items(), key=lambda kv: -kv[1]["tightness_ratio"])[:50]:
        lines.append(f"## family `{key}*`  ({fam['n_members']} members)")
        lines.append(f"- intra-family cosine: **{fam['intra_family_cosine_mean']:+.3f}**")
        lines.append(f"- random-baseline cosine: {fam['random_baseline_cosine_median']:+.3f}")
        lines.append(f"- tightness ratio: **{fam['tightness_ratio']:.2f}**")
        lines.append("")
        lines.append("members preview:")
        for m in fam["members_preview"]:
            lines.append(f"  - `{m['decoded_text']}` (id={m['id']}, raw=`{m['raw_token']}`)")
        lines.append("")
    return "\n".join(lines)


def analogy_test(M: np.ndarray, matrix: str, decoded, classified_ids):
    print(f"=== {matrix}: analogy arithmetic ===", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")

    # Pre-compute unit-normalised rows for all classified tokens (fast cosine search)
    classified_ids_arr = np.asarray(classified_ids, dtype=np.int64)
    all_rows = M[classified_ids_arr]
    all_unit = all_rows / (np.linalg.norm(all_rows, axis=1, keepdims=True) + 1e-12)

    decoded_lookup_by_id = decoded.copy()

    results = []
    for label, a, b, c, expected_d in ANALOGY_CANDIDATES:
        ida = find_token_id(a, decoded_lookup_by_id, tok)
        idb = find_token_id(b, decoded_lookup_by_id, tok)
        idc = find_token_id(c, decoded_lookup_by_id, tok)
        idd = find_token_id(expected_d, decoded_lookup_by_id, tok)
        if any(x is None for x in (ida, idb, idc, idd)):
            results.append({
                "label": label, "a": a, "b": b, "c": c, "expected_d": expected_d,
                "skipped": True,
                "reason": "one of {a,b,c,d} is not a single token in Apertus's vocab",
                "ids": {"a": ida, "b": idb, "c": idc, "d": idd},
            })
            continue
        va = M[ida]
        vb = M[idb]
        vc = M[idc]
        target = va - vb + vc
        target_unit = target / (np.linalg.norm(target) + 1e-12)
        # cosine search excluding a, b, c themselves
        sims = all_unit @ target_unit
        exclude = {ida, idb, idc}
        # rank
        order = np.argsort(-sims)
        top5 = []
        rank_of_expected = None
        for j, idx in enumerate(order):
            tid = int(classified_ids_arr[idx])
            if tid in exclude:
                continue
            cos = float(sims[idx])
            if len(top5) < 5:
                top5.append({
                    "id": tid,
                    "decoded_text": decoded_lookup_by_id.get(tid, {}).get("decoded_text", ""),
                    "cos": cos,
                })
            if tid == idd and rank_of_expected is None:
                rank_of_expected = len([s for s in order[:j+1] if int(classified_ids_arr[s]) not in exclude])
            if len(top5) >= 5 and rank_of_expected is not None:
                break
        results.append({
            "label": label, "a": a, "b": b, "c": c, "expected_d": expected_d,
            "skipped": False,
            "ids": {"a": ida, "b": idb, "c": idc, "d": idd},
            "rank_of_expected_d": rank_of_expected,
            "top5": top5,
            "success": (rank_of_expected is not None and rank_of_expected <= 5),
        })
    return results


def main():
    V2_DIR.mkdir(parents=True, exist_ok=True)
    greek_ids, decoded = load_decoded_all_greek()
    # also need decoded for the classified ¬Greek so that analogy search can name results
    groups = json.loads((ROOT / "geometry" / "groups_greek_vs_not.json").read_text())
    other_ids = groups["not_Greek"]
    want = set(int(x) for x in other_ids)
    with CLASS_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            tid = int(r["id"])
            if tid in want:
                decoded[tid] = {
                    "raw_token": r.get("raw_token", ""),
                    "decoded_text": r.get("decoded_text", ""),
                }

    for matrix in ("E", "U"):
        M = np.load(ROOT / "arrays" / f"{matrix}_fp32.npy", mmap_mode="r")
        fam = family_analysis(np.asarray(M), matrix, greek_ids, decoded)
        (V2_DIR / f"greek_families_{matrix}.json").write_text(
            json.dumps(fam, indent=2, ensure_ascii=False))
        (V2_DIR / f"greek_families_{matrix}.md").write_text(render_families_md(fam, matrix))
        ana = analogy_test(np.asarray(M), matrix, decoded,
                            groups["all_classified"])
        (V2_DIR / f"greek_analogies_{matrix}.json").write_text(
            json.dumps(ana, indent=2, ensure_ascii=False))
        print(f"  [{matrix}] {len(fam)} families, {len(ana)} analogy candidates "
              f"({sum(1 for a in ana if not a['skipped'])} tested, "
              f"{sum(1 for a in ana if a.get('success'))} successes)", flush=True)


if __name__ == "__main__":
    main()
