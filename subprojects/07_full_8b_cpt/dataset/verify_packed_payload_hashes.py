#!/usr/bin/env python3
"""Re-hash every packed .bin/.idx/.active payload recorded by 512 manifests."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != int(row["bytes"]):
        raise ValueError(f"packed payload path/size drift: {path}")
    observed = sha256_file(path)
    if observed != row["sha256"]:
        raise ValueError(f"packed payload SHA-256 drift: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": observed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    packed = json.loads(args.packed_receipt.read_text(encoding="utf-8"))
    if packed.get("schema_version") != "apertus_packed_sequence_corpus_v1" or packed.get("status") != "completed":
        raise ValueError("packed receipt is not complete")
    payloads: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for expected in packed["packing_task_manifests"]:
        path = Path(expected["manifest_path"])
        observed = sha256_file(path)
        if observed != expected["manifest_sha256"]:
            raise ValueError(f"packed manifest drift: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or int(manifest.get("task_index", -1)) != int(expected["task_index"]):
            raise ValueError(f"packed manifest contract drift: {path}")
        manifests.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": observed})
        for name, receipt in manifest["outputs"].items():
            payloads.append({"kind": name, **receipt})
    if len(manifests) != 512 or len(payloads) != 1536:
        raise ValueError("packed manifest/payload count drift")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        verified = list(executor.map(verify, payloads))
    payload = {
        "schema_version": "apertus_full_8b_packed_payload_integrity_v1",
        "status": "passed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "packed_receipt": {"path": str(args.packed_receipt.resolve()), "bytes": args.packed_receipt.stat().st_size, "sha256": sha256_file(args.packed_receipt)},
        "manifest_count": len(manifests),
        "payload_count": len(verified),
        "payload_bytes": sum(int(row["bytes"]) for row in verified),
        "worker_threads": args.workers,
        "manifests": manifests,
        "payloads": verified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "payloads": len(verified), "bytes": payload["payload_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
