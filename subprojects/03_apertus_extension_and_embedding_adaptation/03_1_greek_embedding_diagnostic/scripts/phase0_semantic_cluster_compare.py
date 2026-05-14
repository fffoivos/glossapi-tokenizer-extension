"""Semantic cluster tightness — English baseline vs Greek vs cross-language.

For each word in a "concept cluster":
  - tokenize with the Apertus tokenizer (with leading space)
  - get its mean E embedding (mean over its sub-tokens)
  - normalise (unit cosine)

Within each cluster, compute the mean pairwise cosine between the unit
mean-embeddings. Compare to a random-pair baseline drawn from the same
vocab universe (classified subset, sized to match).

Cross-language: for each "Greek-loanword" concept, build a cluster
spanning {en, fr, de, es, it, el}. Compute within-cluster cosine
(all-languages-of-same-concept).

Output:
  geometry/v2_2/semantic_cluster_compare.json — machine-readable
  geometry/v2_2/semantic_cluster_compare.md   — human-readable
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path("/home/foivos/runs/apertus_embedding_init_test_20260512")
OUT_DIR = ROOT / "geometry" / "v2_2"

# ============================================================
# Concept clusters
# ============================================================

ENGLISH_CLUSTERS = {
    "family": ["father", "mother", "brother", "sister", "son", "daughter", "parent"],
    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "small_numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "colors": ["red", "blue", "green", "yellow", "black", "white"],
    "size_adjectives": ["big", "large", "huge", "small", "tiny", "enormous"],
    "common_verbs_motion": ["run", "walk", "jump", "fly", "swim", "drive"],
}

GREEK_CLUSTERS = {
    "greek_family": ["πατέρας", "μητέρα", "αδελφός", "αδελφή", "γιος", "κόρη"],
    "greek_days":   ["Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή"],
    "greek_small_numbers": ["ένα", "δύο", "τρία", "τέσσερα", "πέντε"],
    "greek_colors": ["κόκκινο", "μπλε", "πράσινο", "κίτρινο", "μαύρο", "λευκό"],
    "greek_size_adj": ["μεγάλο", "μικρό", "τεράστιο"],
}

# Cross-language Greek-loanword concepts (en, fr, de, es, it, el)
LOANWORD_CONCEPTS = {
    "democracy":   ["democracy",   "démocratie",    "Demokratie",  "democracia",   "democrazia",   "δημοκρατία"],
    "philosophy":  ["philosophy",  "philosophie",   "Philosophie", "filosofía",    "filosofia",    "φιλοσοφία"],
    "mathematics": ["mathematics", "mathématiques", "Mathematik",  "matemáticas",  "matematica",   "μαθηματικά"],
    "history":     ["history",     "histoire",      "Geschichte",  "historia",     "storia",       "ιστορία"],
    "music":       ["music",       "musique",       "Musik",       "música",       "musica",       "μουσική"],
    "theatre":     ["theatre",     "théâtre",       "Theater",     "teatro",       "teatro",       "θέατρο"],
    "energy":      ["energy",      "énergie",       "Energie",     "energía",      "energia",      "ενέργεια"],
    "biology":     ["biology",     "biologie",      "Biologie",    "biología",     "biologia",     "βιολογία"],
    "geography":   ["geography",   "géographie",    "Geographie",  "geografía",    "geografia",    "γεωγραφία"],
    "logic":       ["logic",       "logique",       "Logik",       "lógica",       "logica",       "λογική"],
}

# Non-Greek-origin concepts for contrast (English-native or common Germanic/Romance)
NATIVE_CONCEPTS = {
    "house":    ["house",    "maison",   "Haus",     "casa",     "casa",     "σπίτι"],
    "water":    ["water",    "eau",      "Wasser",   "agua",     "acqua",    "νερό"],
    "fire":     ["fire",     "feu",      "Feuer",    "fuego",    "fuoco",    "φωτιά"],
    "love":     ["love",     "amour",    "Liebe",    "amor",     "amore",    "αγάπη"],
    "freedom":  ["freedom",  "liberté",  "Freiheit", "libertad", "libertà",  "ελευθερία"],
}

LANG_ORDER = ["en", "fr", "de", "es", "it", "el"]


# ============================================================
# Embedding helpers
# ============================================================

def get_word_embedding(word: str, tok, E: np.ndarray, with_space: bool = True,
                        verbose: bool = False) -> tuple[np.ndarray, list[int]]:
    """Tokenize the word (with leading space by default) and return mean E embedding."""
    s = " " + word if with_space else word
    ids = tok(s, add_special_tokens=False)["input_ids"]
    if not ids:
        return None, []
    rows = E[ids]
    mean_vec = rows.mean(axis=0)
    if verbose:
        decoded = [tok.decode([i]) for i in ids]
        print(f"    {word!r:<28s} → ids={ids} → {decoded}")
    return mean_vec, ids


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def random_baseline_pairwise(E: np.ndarray, ids_pool: np.ndarray,
                              n_pairs: int = 5000, n_tokens_per_word: int = 2,
                              seed: int = 20260513) -> float:
    """Random baseline: mean cosine between random pairs of mean-of-N-random-tokens."""
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_pairs):
        a_idx = rng.choice(ids_pool, size=n_tokens_per_word, replace=False)
        b_idx = rng.choice(ids_pool, size=n_tokens_per_word, replace=False)
        a = E[a_idx].mean(axis=0)
        b = E[b_idx].mean(axis=0)
        sims.append(cos(a, b))
    return float(np.median(sims))


def cluster_pairwise_stats(words: list[str], tok, E: np.ndarray,
                            verbose: bool = False) -> dict:
    embs = []
    tokenisations = []
    for w in words:
        v, ids = get_word_embedding(w, tok, E, verbose=verbose)
        if v is None:
            continue
        embs.append((w, v, len(ids)))
        tokenisations.append((w, ids, len(ids)))
    if len(embs) < 2:
        return {"n_members": len(embs), "members": tokenisations,
                 "pairwise": [], "mean_pairwise_cosine": float("nan")}
    sims = []
    pairs_detail = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            c = cos(embs[i][1], embs[j][1])
            sims.append(c)
            pairs_detail.append({
                "a": embs[i][0], "b": embs[j][0], "cos": c,
                "a_n_tokens": embs[i][2], "b_n_tokens": embs[j][2],
            })
    return {
        "n_members": len(embs),
        "mean_pairwise_cosine": float(np.mean(sims)),
        "median_pairwise_cosine": float(np.median(sims)),
        "min_pairwise_cosine": float(np.min(sims)),
        "max_pairwise_cosine": float(np.max(sims)),
        "n_tokens_per_word": [t[2] for t in tokenisations],
        "members_tokenisation": [{"word": w, "ids": ids, "n_tokens": n_t}
                                   for w, ids, n_t in tokenisations],
        "pairs": pairs_detail,
    }


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")

    E = np.load(ROOT / "arrays" / "E_fp32.npy")  # load to RAM for fast pairwise
    groups = json.loads((ROOT / "geometry" / "groups_greek_vs_not.json").read_text())
    classified_ids = np.asarray(groups["all_classified"], dtype=np.int64)

    print("== Random baseline ==", flush=True)
    base_2tok = random_baseline_pairwise(E, classified_ids, n_pairs=5000, n_tokens_per_word=2)
    base_3tok = random_baseline_pairwise(E, classified_ids, n_pairs=5000, n_tokens_per_word=3)
    print(f"  random pair, 2-token-mean: median cos = {base_2tok:+.4f}")
    print(f"  random pair, 3-token-mean: median cos = {base_3tok:+.4f}")

    results = {
        "random_baseline_median_cos": {"2_token_mean": base_2tok, "3_token_mean": base_3tok},
        "english_clusters": {}, "greek_clusters": {},
        "loanword_clusters": {}, "native_clusters": {},
    }

    print("\n== English clusters ==", flush=True)
    for name, words in ENGLISH_CLUSTERS.items():
        print(f"  {name}: {words}")
        results["english_clusters"][name] = cluster_pairwise_stats(words, tok, E, verbose=True)
        m = results["english_clusters"][name]["mean_pairwise_cosine"]
        avg_tok = mean(results["english_clusters"][name]["n_tokens_per_word"])
        print(f"    mean pairwise cos = {m:+.4f}  (avg tokens/word = {avg_tok:.1f})")

    print("\n== Greek-only clusters ==", flush=True)
    for name, words in GREEK_CLUSTERS.items():
        print(f"  {name}: {words}")
        results["greek_clusters"][name] = cluster_pairwise_stats(words, tok, E, verbose=True)
        m = results["greek_clusters"][name]["mean_pairwise_cosine"]
        avg_tok = mean(results["greek_clusters"][name]["n_tokens_per_word"])
        print(f"    mean pairwise cos = {m:+.4f}  (avg tokens/word = {avg_tok:.1f})")

    print("\n== Greek-loanword cross-language clusters ==", flush=True)
    for name, words in LOANWORD_CONCEPTS.items():
        results["loanword_clusters"][name] = cluster_pairwise_stats(words, tok, E, verbose=True)
        # Per-language pair listing — most useful is cos(en, el) for each
        d = results["loanword_clusters"][name]
        print(f"  {name}: mean pairwise cos = {d['mean_pairwise_cosine']:+.4f}  "
              f"(avg tokens/word = {mean(d['n_tokens_per_word']):.1f})")
        # Extract specific en-vs-el cosine
        for p in d["pairs"]:
            if (p["a"] == LOANWORD_CONCEPTS[name][0]
                and p["b"] == LOANWORD_CONCEPTS[name][-1]):
                print(f"    en-vs-el ({p['a']!r} / {p['b']!r}) cos = {p['cos']:+.4f}")

    print("\n== Non-Greek-origin cross-language clusters (contrast) ==", flush=True)
    for name, words in NATIVE_CONCEPTS.items():
        results["native_clusters"][name] = cluster_pairwise_stats(words, tok, E, verbose=True)
        d = results["native_clusters"][name]
        print(f"  {name}: mean pairwise cos = {d['mean_pairwise_cosine']:+.4f}  "
              f"(avg tokens/word = {mean(d['n_tokens_per_word']):.1f})")
        for p in d["pairs"]:
            if (p["a"] == NATIVE_CONCEPTS[name][0]
                and p["b"] == NATIVE_CONCEPTS[name][-1]):
                print(f"    en-vs-el ({p['a']!r} / {p['b']!r}) cos = {p['cos']:+.4f}")

    # Constructed-word probe: build Greek compounds + check distance to plain English
    # Concept: combine a Greek root with a Greek suffix and see if its mean embedding
    # sits near the English-loanword target.
    print("\n== Constructed-word probe ==", flush=True)
    constructed = {
        "demos+kratia → democracy":   ([" δήμος", " κράτος"], "democracy"),
        "phil+sophia  → philosophy":  ([" φιλ", " σοφία"],     "philosophy"),
        "geo+graphia  → geography":   ([" γεω", " γραφία"],     "geography"),
        "bio+logia    → biology":     ([" βίος", " λογία"],     "biology"),
        "musi+ki      → music":       ([" μουσι", " κή"],       "music"),
    }
    constructed_results = {}
    for label, (parts, target) in constructed.items():
        part_embs = []
        for p in parts:
            v, ids = get_word_embedding(p.strip(), tok, E, verbose=True)
            if v is not None:
                part_embs.append(v)
        if not part_embs:
            continue
        construct_vec = np.mean(part_embs, axis=0)
        target_vec, target_ids = get_word_embedding(target, tok, E, verbose=True)
        c = cos(construct_vec, target_vec)
        constructed_results[label] = {
            "parts": parts, "target": target, "cos_construct_vs_target": c,
            "target_n_tokens": len(target_ids),
        }
        print(f"  {label}: cos(construct, {target!r}) = {c:+.4f}")
    results["constructed_probes"] = constructed_results

    out_path = OUT_DIR / "semantic_cluster_compare.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
