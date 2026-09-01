#!/usr/bin/env python3
"""Freeze GreekMMLU items with no policy-defined contaminating corpus match."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


DATASET_REVISION = "6a03aa06b68beb932fb75edff3a34e50b3674649"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def collect_bridge_dropped_ledgers(root: Path) -> tuple[set[str], list[dict], list[dict]]:
    """Read receipt-bound bridge ledgers, including the retained source build format."""
    contaminated: set[str] = set()
    manifests = []
    ledgers = []
    for manifest_path in sorted(root.rglob("*.manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != "full_cpt_megatron_shard_v1"
            or manifest.get("kind") != "training"
            or manifest.get("decontamination", {}).get("status") != "frozen"
        ):
            continue
        dropped = manifest.get("outputs", {}).get("dropped_ledger", {})
        path = Path(dropped.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != int(dropped.get("bytes", -1))
            or sha256(path) != dropped.get("sha256")
        ):
            raise ValueError(f"bridge dropped-ledger receipt drift: {path}")
        rows = 0
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            evidence = row.get("evidence", [])
            if not isinstance(evidence, list):
                raise ValueError(f"invalid evidence at {path}:{line_no}")
            for match in evidence:
                item_id = str(match.get("benchmark_item_id") or "")
                if item_id:
                    contaminated.add(item_id)
        if rows != int(manifest.get("counts", {}).get("contaminated_rows", -1)):
            raise ValueError(f"bridge dropped-ledger row-count drift: {path}")
        manifests.append(file_receipt(manifest_path))
        ledgers.append({**file_receipt(path), "rows": rows})
    return contaminated, manifests, ledgers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, action="append", required=True)
    parser.add_argument("--output-ids", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_ids.exists() or args.output_manifest.exists():
        raise FileExistsError("refusing to overwrite clean-subset artifacts")
    all_ids = []
    for line_no, line in enumerate(args.queries_jsonl.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = str(row.get("example_id") or "")
        if not example_id:
            raise ValueError(f"query line {line_no} has no example_id")
        all_ids.append(example_id)
    if len(all_ids) != 16_632 or len(set(all_ids)) != len(all_ids):
        raise ValueError("GreekMMLU query IDs must contain 16,632 unique examples")

    contaminated: set[str] = set()
    ledger_files = []
    bridge_manifests = []
    for root in args.ledger_root:
        parquet_paths = sorted(root.rglob("*.parquet"))
        if parquet_paths:
            import pyarrow.parquet as pq

            for path in parquet_paths:
                ledger_files.append(file_receipt(path))
                table = pq.read_table(path, columns=["benchmark_matches_json"])
                for raw in table.column("benchmark_matches_json").to_pylist():
                    if not raw:
                        continue
                    evidence = json.loads(raw)
                    for match in evidence:
                        item_id = str(match.get("benchmark_item_id") or "")
                        if item_id:
                            contaminated.add(item_id)
        bridge_ids, manifests, dropped = collect_bridge_dropped_ledgers(root)
        contaminated.update(bridge_ids)
        bridge_manifests.extend(manifests)
        ledger_files.extend(dropped)
    if not ledger_files:
        raise ValueError("no decontamination decision ledgers found")
    unknown = contaminated - set(all_ids)
    if unknown:
        raise ValueError(f"ledger references unknown GreekMMLU IDs: {sorted(unknown)[:5]}")
    clean = sorted(set(all_ids) - contaminated)
    if not clean:
        raise ValueError("clean GreekMMLU subset is empty")
    args.output_ids.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output_ids) + ".partial")
    temporary.write_text("".join(f"{value}\n" for value in clean), encoding="utf-8")
    os.replace(temporary, args.output_ids)
    payload = {
        "schema_version": "apertus_mini_greekmmlu_clean_subset_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_repo_id": "dascim/GreekMMLU",
        "dataset_revision": DATASET_REVISION,
        "dataset_config": "All",
        "dataset_split": "test",
        "queries": {
            "path": str(args.queries_jsonl.resolve()),
            "bytes": args.queries_jsonl.stat().st_size,
            "sha256": sha256(args.queries_jsonl),
        },
        "ledger_files": ledger_files,
        "bridge_manifests": bridge_manifests,
        "ledger_policy": (
            "Exclude every benchmark item named by a policy-defined drop-causing "
            "match in the complete receipt-bound pre-exclusion bridge ledgers. "
            "Answer-only and isolated short n-gram audit evidence is non-contaminating "
            "under the frozen source policy."
        ),
        "full_count": len(all_ids),
        "contaminated_item_count": len(contaminated),
        "clean_count": len(clean),
        "clean_example_ids": {
            "path": str(args.output_ids.resolve()),
            "bytes": args.output_ids.stat().st_size,
            "sha256": sha256(args.output_ids),
        },
        "policy": "exclude every benchmark item with a policy-defined contaminating match in the pre-exclusion corpus",
    }
    args.output_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "clean": len(clean), "contaminated": len(contaminated)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
