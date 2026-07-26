#!/usr/bin/env python3
"""Dump the connector stage's candidate mask and 177-feature matrix for cohort-2.

The connector bundle is the last feature stage and the widest: 177 features per
candidate, built from neighbour-joined text, window statistics over the entry and
heading probabilities, and a second pass through the P0D model on the joined lines.

Porting that against the four output columns alone would make every disagreement a
needle in a 177-dimensional haystack. Dumping the intermediate matrix instead means a
mismatch names the feature.

Emits, aligned to the full 210,704-line inventory:

    candidate_mask.npy   (n,)      bool   -- which lines the bundle scores at all
    connector_rows.npy   (m, 177)  f32    -- the feature matrix, candidates only
    connector_index.npy  (m,)      int64  -- absolute line index of each row
    feature_names.json                    -- the 177 names, in order

    python3 dump_connector.py --documents <jsonl> --entry-model-dir <dir> \
        --heading-model-dir <dir> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--entry-model-dir", required=True)
    parser.add_argument("--heading-model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from sequence_models import bibliography_nextgen_unseen_features as unseen
    from sequence_models.bibliography_role_features import (
        candidate_window_mask,
        connector_feature_names,
        connector_feature_row,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(connector_feature_names())

    # Rebuild the documents the same way the driver does, so line inventory and
    # abs_indices match the deployed run exactly.
    documents = []
    metadata = []
    cursor = 0
    with open(args.documents, encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            doc = json.loads(raw)
            texts = str(doc.get("text", "")).split("\n")
            lines = [
                {"text": t, "line_id": f"{doc['source_doc_id']}:{i}", "abs_idx": cursor + i}
                for i, t in enumerate(texts)
            ]
            documents.append(
                unseen._extract_document(
                    {
                        "document_id": str(doc["source_doc_id"]),
                        # The cohort file carries no work_id; identity is only used
                        # for validation inside the extractor, not for any feature.
                        "work_id": str(doc["source_doc_id"]),
                        "source": str(doc.get("source", "unknown")),
                        "lines": lines,
                    }
                )
            )
            metadata.append({"line_start": cursor, "line_end": cursor + len(texts)})
            cursor += len(texts)

    total = cursor
    print(f"{len(documents)} documents, {total} lines", file=sys.stderr)

    p0d_root = Path(args.entry_model_dir)
    p0d_models = [
        unseen._load_pickle(p) for p in sorted((p0d_root / "models").glob("P0D.fold*.pkl"))
    ]
    counts = np.concatenate([d.counts for d in documents])
    entry = unseen._batched_predict(p0d_models, unseen.p0d_matrix(counts))
    heading, heading_candidate = unseen._heading_probabilities(
        documents, entry, metadata, Path(args.heading_model_dir)
    )

    mask_all = np.zeros(total, dtype=bool)
    rows: list[np.ndarray] = []
    index: list[int] = []
    for document, row in zip(documents, metadata, strict=True):
        start, end = row["line_start"], row["line_end"]
        local_entry = entry[start:end]
        mask = candidate_window_mask(
            local_entry,
            heading_candidate[start:end],
            document.abs_indices,
            entry_threshold=0.25,
            radius=30,
        )
        mask_all[start:end] = mask
        wanted = np.flatnonzero(mask)
        joined: dict[bytes, np.ndarray] = {}
        for i in wanted:
            for neighbour, left_first in ((i - 1, True), (i + 1, False)):
                if not 0 <= neighbour < len(document.texts):
                    continue
                if abs(
                    int(document.abs_indices[i]) - int(document.abs_indices[neighbour])
                ) > unseen.MAX_PHYSICAL_GAP:
                    continue
                text = (
                    document.texts[neighbour].rstrip() + " " + document.texts[i].lstrip()
                    if left_first
                    else document.texts[i].rstrip() + " " + document.texts[neighbour].lstrip()
                )
                values = unseen.extract_positional_line(text).counts.reshape(1, -1)
                joined.setdefault(values.tobytes(), values)
        joined_scores: dict[bytes, float] = {}
        if joined:
            keys = list(joined)
            matrix = np.concatenate([joined[k] for k in keys])
            scores = unseen._batched_predict(p0d_models, unseen.p0d_matrix(matrix))
            joined_scores = {k: float(v) for k, v in zip(keys, scores, strict=True)}

        def score_counts(values: np.ndarray) -> float:
            return joined_scores[values.astype(np.uint32, copy=False).tobytes()]

        local_heading = heading[start:end, 1:]
        for i in wanted:
            feature = connector_feature_row(
                index=int(i),
                texts=document.texts,
                counts=document.counts,
                gap_summaries=document.gaps,
                abs_indices=document.abs_indices,
                entry_probability=local_entry,
                heading_probability=local_heading,
                candidate_mask=mask,
                score_counts=score_counts,
                entry_threshold=0.25,
            )
            rows.append(feature.values)
            index.append(start + int(i))

    matrix = np.stack(rows).astype(np.float32) if rows else np.zeros((0, len(names)), np.float32)
    np.save(out_dir / "candidate_mask.npy", mask_all)
    np.save(out_dir / "connector_rows.npy", matrix)
    np.save(out_dir / "connector_index.npy", np.asarray(index, dtype=np.int64))
    (out_dir / "feature_names.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    print(f"  candidates {int(mask_all.sum())}, matrix {matrix.shape}", file=sys.stderr)


if __name__ == "__main__":
    main()
