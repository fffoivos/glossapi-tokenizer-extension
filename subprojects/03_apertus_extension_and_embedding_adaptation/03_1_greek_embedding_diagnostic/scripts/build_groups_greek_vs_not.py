"""Emit canonical groups file for v2: Greek (strict 1,494) + ¬Greek.

Inputs:
  base_greek_tokens.jsonl (1,494 strict-Greek ids, source of truth)
  Phase A v2 classification (for the classified-subset mask + source-group
  labels used in infiltrators §3.9 / §3.10 source-group breakdowns)

Output:
  geometry/groups_greek_vs_not.json
    {
      "Greek": [int, ...],                 # 1,494 strict
      "not_Greek": [int, ...],             # classified subset minus Greek
      "source_group_of_negreek": {token_id: source_group, ...},
      "all_classified": [int, ...],
      "excluded_buckets": {bucket: count}  # for traceability
    }
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_GREEK = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/"
    "tokenizer_analysis/inspection/base/greek_tokens/base_greek_tokens.jsonl"
)
CLASSIFICATION = Path(
    "/home/foivos/runs/apertus_greek_diagnostic_20260511_v2/token_classification.jsonl"
)
OUT_PATH = Path(
    "/home/foivos/runs/apertus_embedding_init_test_20260512/"
    "geometry/groups_greek_vs_not.json"
)

EXCLUDED_BUCKETS = {"special", "byte_fragment", "whitespace_only", "digits_only"}


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    strict_greek = set()
    with BASE_GREEK.open() as f:
        for line in f:
            strict_greek.add(int(json.loads(line)["id"]))

    classified = []
    source_group: dict[int, str] = {}
    excluded: dict[str, int] = {}
    with CLASSIFICATION.open() as f:
        for line in f:
            r = json.loads(line)
            tid = int(r["id"])
            bucket = r.get("bucket", "")
            if bucket in EXCLUDED_BUCKETS:
                excluded[bucket] = excluded.get(bucket, 0) + 1
                continue
            classified.append(tid)
            groups = r.get("groups") or []
            # store the byte-level source group for non-Greek tokens
            if tid not in strict_greek:
                if groups:
                    source_group[tid] = groups[0]
                else:
                    source_group[tid] = "_unclassified_member"

    classified_set = set(classified)
    not_greek = sorted(classified_set - strict_greek)

    out = {
        "Greek": sorted(strict_greek),
        "not_Greek": not_greek,
        "source_group_of_negreek": source_group,
        "all_classified": sorted(classified_set),
        "excluded_buckets": excluded,
        "n_greek": len(strict_greek),
        "n_not_greek": len(not_greek),
        "n_classified": len(classified_set),
        "source": {
            "strict_greek": str(BASE_GREEK),
            "classification": str(CLASSIFICATION),
        },
    }
    OUT_PATH.write_text(json.dumps(out))
    print(f"wrote {OUT_PATH}")
    print(f"  n_greek={out['n_greek']}, n_not_greek={out['n_not_greek']}, "
          f"n_classified={out['n_classified']}, excluded_buckets={excluded}")


if __name__ == "__main__":
    main()
