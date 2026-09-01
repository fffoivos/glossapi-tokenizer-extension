#!/usr/bin/env python3
"""Freeze one shared, stratified 1,024-token Mini TD pilot inventory.

The pilot must exercise both tokenizer-extension stages.  Sampling only the
highest-frequency modern tokens would make the layer/loss decision largely
irrelevant to the 512 second-stage polytonic rows.  This selector therefore
reserves one quarter of the pilot for eligible polytonic rows and stratifies
both stages by firing-frequency quartile and base-decomposition length.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
from collections import Counter, defaultdict, deque
from pathlib import Path


BASE_VOCAB_SIZE = 131_072
POLYTONIC_START = 148_480
TARGET_VOCAB_SIZE = 148_992
PILOT_SIZE = 1_024
POLYTONIC_QUOTA = 256
SEED = 2_026_080_1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            yield row


def frequency_quartiles(rows: list[dict]) -> dict[int, int]:
    ordered = sorted(rows, key=lambda row: (int(row["extended_firings"]), int(row["new_token_id"])))
    count = len(ordered)
    return {
        int(row["new_token_id"]): min(3, index * 4 // count)
        for index, row in enumerate(ordered)
    }


def length_bucket(row: dict) -> str:
    length = int(row["base_subtoken_len"])
    return "1" if length == 1 else "2" if length == 2 else "3plus"


def select_stage(rows: list[dict], quota: int, seed: int) -> tuple[list[dict], Counter]:
    if len(rows) < quota:
        raise ValueError(f"eligible stage population {len(rows)} is below quota {quota}")
    quartiles = frequency_quartiles(rows)
    strata: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        strata[(quartiles[int(row["new_token_id"])], length_bucket(row))].append(row)
    queues: dict[tuple[int, str], deque[dict]] = {}
    for index, (key, values) in enumerate(sorted(strata.items())):
        rng = random.Random(seed + index)
        rng.shuffle(values)
        queues[key] = deque(values)
    selected: list[dict] = []
    selected_strata: Counter = Counter()
    keys = sorted(queues)
    while len(selected) < quota:
        progressed = False
        for key in keys:
            if queues[key] and len(selected) < quota:
                selected.append(queues[key].popleft())
                selected_strata[f"q{key[0]}_len{key[1]}"] += 1
                progressed = True
        if not progressed:
            raise RuntimeError("stratified round-robin exhausted before quota")
    return selected, selected_strata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = list(iter_jsonl(args.coverage_jsonl))
    by_id = {int(row.get("new_token_id", -1)): row for row in rows}
    if sorted(by_id) != list(range(BASE_VOCAB_SIZE, TARGET_VOCAB_SIZE)) or len(by_id) != len(rows):
        raise ValueError("coverage must contain every added ID exactly once")
    eligible = [
        row
        for row in rows
        if row.get("status") in {"enough_25", "enough_100"}
        and int(row.get("usable_snippets_25", 0)) >= 25
        and int(row.get("extended_firings", 0)) >= 25
        and int(row.get("docs_with_firing", 0)) > 0
        and isinstance(row.get("base_subtoken_ids"), list)
        and bool(row.get("base_subtoken_ids"))
    ]
    modern = [row for row in eligible if int(row["new_token_id"]) < POLYTONIC_START]
    polytonic = [row for row in eligible if int(row["new_token_id"]) >= POLYTONIC_START]
    selected_modern, modern_strata = select_stage(
        modern, PILOT_SIZE - POLYTONIC_QUOTA, args.seed
    )
    selected_poly, poly_strata = select_stage(polytonic, POLYTONIC_QUOTA, args.seed + 10_000)
    selected = selected_modern + selected_poly
    random.Random(args.seed + 20_000).shuffle(selected)
    selected_ids = [int(row["new_token_id"]) for row in selected]
    if len(selected_ids) != PILOT_SIZE or len(set(selected_ids)) != PILOT_SIZE:
        raise AssertionError("pilot selection is not an exact 1,024-ID set")

    args.output_dir.mkdir(parents=True)
    ids_path = args.output_dir / "pilot_token_ids.txt"
    ids_path.write_text("".join(f"{token_id}\n" for token_id in selected_ids), encoding="utf-8")
    payload = {
        "schema_version": "apertus_mini_td_pilot_token_selection_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "coverage_jsonl": str(args.coverage_jsonl.resolve()),
        "coverage_sha256": sha256_file(args.coverage_jsonl),
        "selection_seed": args.seed,
        "selected_count": len(selected_ids),
        "modern_selected": len(selected_modern),
        "polytonic_selected": len(selected_poly),
        "modern_eligible": len(modern),
        "polytonic_eligible": len(polytonic),
        "modern_strata": dict(sorted(modern_strata.items())),
        "polytonic_strata": dict(sorted(poly_strata.items())),
        "eligibility": "status enough_25/enough_100, >=25 accepted snippets/firings, nonempty exact base decomposition",
        "selection": "stage quota then round-robin over within-stage firing quartile x base-decomposition-length stratum",
        "token_ids": selected_ids,
        "token_ids_file": {
            "path": str(ids_path.resolve()),
            "sha256": sha256_file(ids_path),
            "bytes": ids_path.stat().st_size,
        },
    }
    manifest = args.output_dir / "pilot_token_selection.json"
    temporary = Path(str(manifest) + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest)
    print(json.dumps({"ok": True, "manifest": str(manifest), "selected": PILOT_SIZE}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
