#!/usr/bin/env python3
"""Freeze the exact nine historical in-loop validation binaries."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, require, sha256_file, write_json_atomic


NAMES = ("hplt", "openarchives", "greek_phd", "english", "de", "ru", "zh", "code", "old_greek")
NEW_GREEK = frozenset(("hplt", "openarchives", "greek_phd"))


def prefix(root: Path, name: str) -> Path:
    stem = f"val_{name}" if name in NEW_GREEK else f"val_forget_{name}"
    return root / f"{stem}_ext_text_document"


def validate_receipt(
    value: dict[str, object],
    root: Path,
    *,
    accepted_code_bundles: set[tuple[str, str]] | None = None,
    verify_payload_hashes: bool = True,
) -> None:
    require(value.get("schema_version") == "apertus_hard_h_to_g_online_validation_v1", "online validation schema drift")
    require(value.get("status") == "frozen", "online validation receipt is not frozen")
    require(Path(str(value.get("root", ""))).resolve() == root.resolve(), "online validation root drift")
    require(value.get("panel_names") == list(NAMES), "online validation panel inventory/order drift")
    require(value.get("new_greek_panel_names") == sorted(NEW_GREEK), "online validation Greek-family drift")
    require(value.get("tokenization") == "historical_148480_extended", "online validation tokenization drift")
    require(value.get("selection_authorized") is False and value.get("historical_comparability_only") is True, "online validation scope drift")
    rows = value.get("files")
    require(isinstance(rows, list) and len(rows) == 2 * len(NAMES), "online validation file inventory drift")
    expected_paths = {
        str(Path(f"{prefix(root, name)}{suffix}").resolve())
        for name in NAMES for suffix in (".bin", ".idx")
    }
    observed_paths: set[str] = set()
    for row in rows:
        require(isinstance(row, dict), "invalid online validation file row")
        path = Path(str(row.get("path", ""))).resolve()
        observed_paths.add(str(path))
        require(path.is_file() and path.stat().st_size == int(row.get("bytes", -1)), f"online validation file size drift: {path}")
        if verify_payload_hashes:
            require(sha256_file(path) == row.get("sha256"), f"online validation file hash drift: {path}")
    require(observed_paths == expected_paths, "online validation file paths drift")
    code = value.get("executing_code_bundle")
    require(isinstance(code, dict), "online validation code-bundle binding missing")
    observed = (str(Path(str(code.get("root", ""))).resolve()), str(code.get("tree_sha256", "")))
    if accepted_code_bundles is None:
        current = executing_code_bundle()
        accepted_code_bundles = {
            (str(Path(str(current["root"])).resolve()), str(current["tree_sha256"]))
        }
    require(observed in accepted_code_bundles, "online validation code-bundle drift")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable online validation receipt exists: {args.output}")
    root = args.root.resolve()
    files = [Path(f"{prefix(root, name)}{suffix}") for name in NAMES for suffix in (".bin", ".idx")]
    require(all(path.is_file() and path.stat().st_size > 0 for path in files), "one or more online validation binaries are missing")
    payload: dict[str, object] = {
        "schema_version": "apertus_hard_h_to_g_online_validation_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(root),
        "panel_names": list(NAMES),
        "new_greek_panel_names": sorted(NEW_GREEK),
        "tokenization": "historical_148480_extended",
        "files": [file_binding(path) for path in files],
        "historical_comparability_only": True,
        "selection_authorized": False,
        "executing_code_bundle": executing_code_bundle(),
    }
    validate_receipt(payload, root)
    write_json_atomic(args.output, payload)
    print(json.dumps({"ok": True, "panels": len(NAMES), "root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
