#!/usr/bin/env python3
"""Fail closed unless the executing evaluation tree matches its receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if receipt.get("schema_version") != "native_greek_eval_code_bundle_v1" or receipt.get("status") != "frozen":
        raise ValueError("invalid code-bundle receipt")
    if Path(receipt["root"]).resolve() != args.root.resolve():
        raise ValueError("executing code root differs from receipt")
    measured = []
    for path in sorted(args.root.rglob("*")):
        if path.is_file() and path.resolve() != args.receipt.resolve():
            measured.append(
                {
                    "path": str(path.relative_to(args.root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    tree = hashlib.sha256(
        "\n".join(f"{row['sha256']}  {row['path']}" for row in measured).encode()
    ).hexdigest()
    if measured != receipt.get("files") or tree != receipt.get("tree_sha256"):
        raise ValueError("executing evaluation bundle differs from frozen receipt")
    print(json.dumps({"ok": True, "files": len(measured), "tree_sha256": tree}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
