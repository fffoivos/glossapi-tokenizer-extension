#!/usr/bin/env python3
"""Write the immutable receipt for an already materialized evaluation bundle."""

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
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    receipt_path = root / "bundle_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    files = [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    tree_sha256 = hashlib.sha256(
        "\n".join(f"{row['sha256']}  {row['path']}" for row in files).encode()
    ).hexdigest()
    receipt = {
        "schema_version": "native_greek_eval_code_bundle_v1",
        "status": "frozen",
        "root": str(root),
        "git_commit": args.git_commit,
        "tree_sha256": tree_sha256,
        "files": files,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "files": len(files), "tree_sha256": tree_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
