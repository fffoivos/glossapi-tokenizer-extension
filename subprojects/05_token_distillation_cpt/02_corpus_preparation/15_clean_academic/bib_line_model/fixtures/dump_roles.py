#!/usr/bin/env python3
"""Dump the deterministic negative roles and header kinds for a document set.

These are 9 of the signal TCN's 10 inputs, and they come from
`analyze_bibliography_line_v2` -> reason codes -> `_role_index`, a ~235-line
branching cascade. Porting that blind and hoping is not the method that has worked
here; every other stage was got right by building the gate first and letting it
find the disagreements.

So this emits, for the same documents the Rust side will read:

    roles.npy         (n_lines, 8) uint8, one-hot and mutually exclusive
    header_kinds.npy  (n_lines,)   uint8, 1 = HEADING, 2 = SUBHEADING, 0 = neither
    reason_codes.jsonl            per line, for diagnosing *which* branch differs

The reason codes matter as much as the roles: `_role_index` maps codes to a role by
substring, so a port can land the right role for the wrong reason and only diverge
later on some other corpus.

    python3 dump_roles.py --documents <jsonl> --out-dir <dir>
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
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from sequence_models.bibliography_deterministic_roles import (
        ROLE_NAMES,
        _analyze_document,
    )
    from sequence_models.bibliography_v2 import analyze_bibliography_line_v2
    from sequence_models.deterministic_structure import BibRole, analyze_bib_line

    all_roles = []
    header_kinds: list[int] = []
    codes_path = Path(args.out_dir) / "reason_codes.jsonl"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    n_docs = 0
    with codes_path.open("w", encoding="utf-8") as codes_out:
        with open(args.documents, encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                doc = json.loads(raw)
                texts = str(doc.get("text", "")).split("\n")
                lines = [{"text": t} for t in texts]
                _id, roles, _counts = _analyze_document(
                    (str(doc.get("source_doc_id", n_docs)), lines)
                )
                all_roles.append(roles)
                for index, text in enumerate(texts):
                    base = analyze_bib_line(text, index)
                    header_kinds.append(
                        1
                        if base.role == BibRole.HEADING
                        else 2
                        if base.role == BibRole.SUBHEADING
                        else 0
                    )
                    v2 = analyze_bibliography_line_v2(text, index)
                    codes_out.write(
                        json.dumps(
                            {
                                "hard_negative": bool(v2.hard_negative),
                                "codes": list(v2.reason_codes),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                n_docs += 1

    roles = np.concatenate(all_roles).astype(np.uint8)
    kinds = np.asarray(header_kinds, dtype=np.uint8)
    np.save(Path(args.out_dir) / "roles.npy", roles)
    np.save(Path(args.out_dir) / "header_kinds.npy", kinds)
    print(f"{n_docs} documents, {len(kinds)} lines", file=sys.stderr)
    print(f"  roles {roles.shape}, {int(roles.sum())} flagged", file=sys.stderr)
    for index, name in enumerate(ROLE_NAMES):
        print(f"    {name:<32} {int(roles[:, index].sum())}", file=sys.stderr)
    print(f"  header kinds: {int((kinds > 0).sum())} headings/subheadings", file=sys.stderr)


if __name__ == "__main__":
    main()
