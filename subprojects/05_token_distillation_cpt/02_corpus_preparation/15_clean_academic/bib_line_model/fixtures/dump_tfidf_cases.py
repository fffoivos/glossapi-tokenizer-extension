#!/usr/bin/env python3
"""Transform real corpus lines with the *fitted* TF-IDF vectorizers.

The Rust port reimplements sklearn's `char_wb` and word analyzers from a reading of
the source. That is the single least-verified assumption in the whole port: the
n-gram boundary rules are fiddly (each whitespace-split word is padded with one
space either side, and a word shorter than n contributes exactly one padded n-gram
rather than several), and a subtle disagreement would shift every heading
probability without ever raising an error.

So rather than test the port against another reading of the source, transform real
lines with the actual fitted vectorizer and dump the sparse rows. `tests/tfidf_parity.rs`
then requires the port to reproduce them exactly.

    python3 dump_tfidf_cases.py --heading-model-dir <dir> --documents <jsonl> \
        --out tfidf_cases.json [--limit 4000]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path


def load_bundle(path: Path):
    import __main__

    from sequence_models.bibliography_role_experts import (
        ConnectorBundle,
        HeadingBundle,
        HeadingTransform,
    )

    __main__.HeadingTransform = HeadingTransform
    __main__.HeadingBundle = HeadingBundle
    __main__.ConnectorBundle = ConnectorBundle
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heading-model-dir", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=4000)
    args = parser.parse_args()

    bundle = load_bundle(sorted(Path(args.heading_model_dir, "models").glob("fold*.pkl"))[0])
    transform = getattr(bundle, "transform", bundle)
    char_vec = getattr(transform, "char_vectorizer", None) or transform.char_tfidf
    word_vec = getattr(transform, "word_vectorizer", None) or transform.word_tfidf

    # Sample across all documents, not the first few: the analyzers must survive
    # OCR wreckage and Greek diacritics, which cluster by source.
    corpus: list[str] = []
    with open(args.documents, encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            doc = json.loads(raw)
            corpus.extend(str(doc.get("text", "")).split("\n"))
    stride = max(1, len(corpus) // max(1, args.limit))
    texts = corpus[::stride][: args.limit]
    print(f"{len(corpus)} lines -> stride {stride} -> {len(texts)} sampled", file=sys.stderr)

    cases = []
    for name, vec in (("char", char_vec), ("word", word_vec)):
        matrix = vec.transform(texts)
        print(f"  {name}: {matrix.shape}, {matrix.nnz} nonzeros", file=sys.stderr)
    char_m = char_vec.transform(texts)
    word_m = word_vec.transform(texts)
    for row, text in enumerate(texts):
        cs, ce = char_m.indptr[row], char_m.indptr[row + 1]
        ws, we = word_m.indptr[row], word_m.indptr[row + 1]
        cases.append(
            {
                "text": text,
                "char_indices": [int(i) for i in char_m.indices[cs:ce]],
                "char_values": [float(v) for v in char_m.data[cs:ce]],
                "word_indices": [int(i) for i in word_m.indices[ws:we]],
                "word_values": [float(v) for v in word_m.data[ws:we]],
            }
        )

    payload = {
        "schema_version": "bib-tfidf-cases-v1",
        "n_cases": len(cases),
        "char_analyzer": char_vec.analyzer,
        "char_ngram_range": list(char_vec.ngram_range),
        "word_analyzer": word_vec.analyzer,
        "word_ngram_range": list(word_vec.ngram_range),
        "cases": cases,
    }
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    print(f"wrote {len(cases)} cases -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
