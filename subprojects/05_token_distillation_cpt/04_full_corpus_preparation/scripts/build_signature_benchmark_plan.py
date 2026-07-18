#!/usr/bin/env python3
"""Build the immutable signature benchmark plan without a compute allocation.

The benchmark itself uses the pinned runtime on a normal node.  This small
planner only verifies JSON/hash bindings and freezes its input ranks, so it is
safe to run on the CSCS login node before a held benchmark job is released.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile


PLAN_SCHEMA = "agent1_v5_dedup_acceleration_benchmark_plan_v1"
AUDIT_SCHEMA = "agent1_v5_dedup_full_input_audit_v1"
CUTOVER_SCHEMA = "agent1_v5_dedup_acceleration_cutover_v1"
PHASES = ((1, 2, "baseline"), (2, 4, "two_worker"), (4, 8, "four_worker"), (5, 10, "five_worker"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("{}: expected a JSON object".format(path))
    return value


def write_immutable(path, value):
    if os.path.exists(path):
        raise FileExistsError("refusing to overwrite immutable plan: {}".format(path))
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, mode=0o700, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".benchmark-plan-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def phase_ranks(raw, count):
    values = [int(value) for value in raw.split(",") if value]
    if len(values) != count or len(set(values)) != len(values):
        raise ValueError("benchmark phase has an invalid rank count or duplicate")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--full-input-audit", required=True)
    parser.add_argument("--cutover-receipt", required=True)
    parser.add_argument("--combined-manifest", required=True)
    parser.add_argument("--phase-ranks", action="append", required=True, metavar="RANK[,RANK...]")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    audit = read(args.full_input_audit)
    cutover = read(args.cutover_receipt)
    manifest = read(args.combined_manifest)
    manifest_sha = sha256_file(args.combined_manifest)
    if audit.get("schema_version") != AUDIT_SCHEMA or audit.get("status") != "passed":
        raise ValueError("full input audit is not passed")
    if cutover.get("schema_version") != CUTOVER_SCHEMA or cutover.get("status") != "passed":
        raise ValueError("cutover receipt is not passed")
    if audit.get("combined_manifest_sha256") != manifest_sha or cutover.get("combined_manifest_sha256") != manifest_sha:
        raise ValueError("audit/cutover manifest binding drift")
    first = int(cutover.get("first_missing_rank", -1))
    if first < 0:
        raise ValueError("cutover first missing rank is invalid")
    if len(args.phase_ranks) != len(PHASES):
        raise ValueError("exactly four benchmark phase lists are required")
    parsed = [phase_ranks(raw, count) for raw, (_, count, _) in zip(args.phase_ranks, PHASES)]
    flat = [rank for ranks in parsed for rank in ranks]
    if len(flat) != 24 or len(set(flat)) != 24 or min(flat) != first:
        raise ValueError("benchmark must start at first missing rank and contain 24 unique ranks")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("combined manifest files missing")
    by_rank = {int(row.get("rank", -1)): row for row in files}
    for rank in flat:
        row = by_rank.get(rank)
        if row is None:
            raise ValueError("benchmark rank is absent from manifest: {}".format(rank))
        if row.get("origin") != "nanochat_base" or int(row.get("rows", -1)) != 196608:
            raise ValueError("benchmark ranks must be homogeneous full NanoChat shards")
    excluded = []
    for rank in range(first, max(flat) + 1):
        if rank in flat:
            continue
        row = by_rank.get(rank)
        if row is None or (row.get("origin") == "nanochat_base" and int(row.get("rows", -1)) == 196608):
            raise ValueError("benchmark may skip only non-full NanoChat shards")
        excluded.append({"rank": rank, "bytes": row["bytes"], "rows": row["rows"], "sha256": row["sha256"], "reason": "non_full_nanochat_shard"})
    value = {
        "schema_version": PLAN_SCHEMA,
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_root": os.path.realpath(args.run_root),
        "full_input_audit_sha256": sha256_file(args.full_input_audit),
        "cutover_receipt_sha256": sha256_file(args.cutover_receipt),
        "combined_manifest_sha256": manifest_sha,
        "first_missing_rank": first,
        "phases": [{"index": index, "name": name, "workers": workers, "ranks": parsed[index]} for index, (workers, _, name) in enumerate(PHASES)],
        "explicit_nonbenchmark_exclusions": excluded,
        "rank_inventory": [{"rank": by_rank[rank]["rank"], "bytes": by_rank[rank]["bytes"], "rows": by_rank[rank]["rows"], "sha256": by_rank[rank]["sha256"]} for rank in flat],
    }
    write_immutable(args.output, value)
    print(json.dumps({"ok": True, "phases": 4, "ranks": 24, "excluded": len(excluded)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
