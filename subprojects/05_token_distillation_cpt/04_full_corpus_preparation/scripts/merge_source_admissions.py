#!/usr/bin/env python3
"""Merge complete pre-clean admissions with the required post-clean re-review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "source_quality_review_admission_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_complete(path: Path, phase: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError(f"{path}: unsupported admission schema")
    if int(value.get("pending_adjudications", 0)):
        raise ValueError(f"{path}: admission still has pending adjudications")
    if value.get("review_phase") != phase:
        raise ValueError(f"{path}: expected review_phase={phase!r}")
    rows = value.get("sources")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: sources must be a list")
    names = [str(row.get("source_dataset") or "") for row in rows]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError(f"{path}: source_dataset values must be non-empty and unique")
    return value


def merge(preclean: dict[str, Any], postclean: dict[str, Any]) -> dict[str, Any]:
    pre_rows = {str(row["source_dataset"]): dict(row) for row in preclean["sources"]}
    post_rows = {str(row["source_dataset"]): dict(row) for row in postclean["sources"]}
    required = {
        name for name, row in pre_rows.items() if row.get("decision") == "include_after_cleaning"
    }
    if set(post_rows) != required:
        missing = sorted(required - set(post_rows))
        unexpected = sorted(set(post_rows) - required)
        raise ValueError(
            "post-clean admission must cover exactly pre-clean include_after_cleaning sources; "
            f"missing={missing}, unexpected={unexpected}"
        )
    merged: list[dict[str, Any]] = []
    for name in sorted(pre_rows):
        pre = pre_rows[name]
        if name not in post_rows:
            merged.append(pre)
            continue
        post = post_rows[name]
        if post.get("decision") == "include_after_cleaning":
            raise ValueError(f"{name}: post-clean decision cannot request another implicit cleaning pass")
        merged.append(
            {
                **post,
                "preclean_decision": pre["decision"],
                "preclean_reasons": pre.get("reasons", []),
                "post_clean_review_required": False,
            }
        )
    return {
        "schema_version": SCHEMA,
        "review_phase": "merged_pre_post_clean",
        "pending_adjudications": 0,
        "sources": merged,
        "provenance": {
            "preclean_unique_documents": preclean.get("unique_documents"),
            "postclean_unique_documents": postclean.get("unique_documents"),
            "postclean_sources": sorted(required),
        },
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable admission: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preclean", type=Path, required=True)
    parser.add_argument("--postclean", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preclean = load_complete(args.preclean, "pre_clean")
    postclean = load_complete(args.postclean, "post_clean")
    result = merge(preclean, postclean)
    result["provenance"].update(
        {
            "preclean": str(args.preclean.resolve()),
            "preclean_sha256": sha256_file(args.preclean),
            "postclean": str(args.postclean.resolve()),
            "postclean_sha256": sha256_file(args.postclean),
        }
    )
    write_json_atomic(args.output, result)
    print(json.dumps({"ok": True, "output": str(args.output), "sources": len(result["sources"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
