#!/usr/bin/env python3
"""Audit every validation document against training and rebuild leaking panels."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer


INDEX_HEADER = b"MMIDIDX\x00\x00"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def verify_binding(value: dict[str, Any]) -> Path:
    path = Path(value["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(value["bytes"])
        or sha256_file(path) != value["sha256"]
    ):
        raise ValueError(f"input binding drift: {path}")
    return path


def write_index(path: Path, lengths: list[int]) -> None:
    values = np.asarray(lengths, dtype=np.int32)
    pointers = np.empty(len(values), dtype=np.int64)
    offset = 0
    for index, length in enumerate(values):
        pointers[index] = offset
        offset += int(length) * 4
    documents = np.arange(len(values) + 1, dtype=np.int64)
    with path.open("wb") as handle:
        handle.write(INDEX_HEADER)
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<B", 4))
        handle.write(struct.pack("<Q", len(values)))
        handle.write(struct.pack("<Q", len(documents)))
        handle.write(values.tobytes(order="C"))
        handle.write(pointers.tobytes(order="C"))
        handle.write(documents.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())


def overlaps_training(text: str, training: np.ndarray) -> tuple[bool, bytes]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = np.asarray([digest], dtype="V32")
    position = int(np.searchsorted(training, value)[0])
    return bool(position < training.size and training[position] == value[0]), digest


def tokenizer_eos(tokenizer_root: Path, tokenizer: Tokenizer) -> int:
    config = read_json(tokenizer_root / "tokenizer_config.json")
    value = config["eos_token"]
    content = value.get("content") if isinstance(value, dict) else value
    token_id = tokenizer.token_to_id(str(content))
    if token_id is None:
        raise ValueError("tokenizer EOS is missing")
    return int(token_id)


def rebuild_panel(
    *, name: str, rows: list[dict[str, Any]], tokenizer: Tokenizer, eos_id: int,
    output_root: Path, source_panel: dict[str, Any], removed_hashes: list[bytes],
) -> tuple[dict[str, Any], dict[str, Any]]:
    panel_root = output_root / name
    panel_root.mkdir(parents=True)
    raw_path = panel_root / f"{name}.training_disjoint.jsonl"
    prefix = panel_root / f"{name}.training_disjoint_text_document"
    bin_path = Path(str(prefix) + ".bin")
    idx_path = Path(str(prefix) + ".idx")
    lengths: list[int] = []
    with raw_path.open("w", encoding="utf-8") as raw, bin_path.open("wb") as binary:
        for row in rows:
            text = str(row["text"])
            ids = tokenizer.encode(text, add_special_tokens=False).ids + [eos_id]
            raw.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            binary.write(np.asarray(ids, dtype=np.int32).tobytes(order="C"))
            lengths.append(len(ids))
        raw.flush(); os.fsync(raw.fileno())
        binary.flush(); os.fsync(binary.fileno())
    write_index(idx_path, lengths)
    tokens = sum(lengths)
    if tokens < 4_194_304:
        raise ValueError(f"cleaned panel {name} has only {tokens} tokens")
    removed_digest = hashlib.sha256(b"".join(sorted(removed_hashes))).hexdigest()
    panel_receipt_path = panel_root / "panel_receipt.json"
    panel_receipt = {
        "schema_version": "apertus_full_8b_training_disjoint_panel_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "name": name,
        "identity": "sha256_exact_utf8_text",
        "source_panel": source_panel,
        "documents": len(lengths),
        "tokens": tokens,
        "removed_training_overlap_documents": len(removed_hashes),
        "removed_content_sha256_set_digest": removed_digest,
        "megatron_prefix": str(prefix.resolve()),
        "raw_jsonl": binding(raw_path, rows=len(lengths)),
        "bin": binding(bin_path),
        "idx": binding(idx_path),
    }
    panel_receipt_path.write_text(
        json.dumps(panel_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    updated = {
        **source_panel,
        "documents": len(lengths),
        "tokens": tokens,
        "megatron_prefix": str(prefix.resolve()),
        "manifest_path": str(panel_receipt_path.resolve()),
        "manifest_sha256": sha256_file(panel_receipt_path),
        "raw_jsonl": panel_receipt["raw_jsonl"],
        "training_exact_content_overlap_documents": 0,
        "training_overlap_remediation": {
            "removed_documents": len(removed_hashes),
            "receipt": binding(panel_receipt_path),
        },
    }
    return updated, panel_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--training-content-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_manifest.exists():
        raise FileExistsError(args.output_dir if args.output_dir.exists() else args.output_manifest)
    args.output_dir.mkdir(parents=True)
    manifest = read_json(args.input_manifest)
    if manifest.get("schema_version") != "apertus_full_8b_validation_manifest_v1":
        raise ValueError("input validation schema drift")
    training_receipt = read_json(args.training_content_receipt)
    training_path = verify_binding(training_receipt["combined"])
    training = np.memmap(training_path, mode="r", dtype="V32")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_root / "tokenizer.json"))
    eos_id = tokenizer_eos(args.tokenizer_root, tokenizer)
    audited = []
    panels = []
    for source_panel in manifest["panels"]:
        name = str(source_panel["name"])
        raw_path = verify_binding(source_panel["raw_jsonl"])
        retained: list[dict[str, Any]] = []
        removed_hashes: list[bytes] = []
        documents = 0
        with raw_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                documents += 1
                overlap, digest = overlaps_training(str(row["text"]), training)
                if overlap:
                    removed_hashes.append(digest)
                else:
                    retained.append(row)
        if documents != int(source_panel["documents"]):
            raise ValueError(f"panel document-count drift: {name}")
        if removed_hashes:
            updated, panel_receipt = rebuild_panel(
                name=name, rows=retained, tokenizer=tokenizer, eos_id=eos_id,
                output_root=args.output_dir, source_panel=source_panel,
                removed_hashes=removed_hashes,
            )
            final_documents = panel_receipt["documents"]
            final_tokens = panel_receipt["tokens"]
        else:
            updated = {**source_panel, "training_exact_content_overlap_documents": 0}
            final_documents = documents
            final_tokens = int(source_panel["tokens"])
        panels.append(updated)
        audited.append({
            "name": name,
            "original_documents": documents,
            "original_training_exact_content_overlap_documents": len(removed_hashes),
            "original_overlap_fraction": len(removed_hashes) / documents,
            "final_documents": final_documents,
            "final_tokens": final_tokens,
            "final_training_exact_content_overlap_documents": 0,
            "rebuilt": bool(removed_hashes),
        })
    if len(panels) != 13 or {row["name"] for row in panels} != {row["name"] for row in manifest["panels"]}:
        raise ValueError("validation panel coverage drift")
    audit_path = args.output_dir / "all_panel_training_content_audit.json"
    audit = {
        "schema_version": "apertus_full_8b_all_panel_training_content_audit_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "identity": "sha256_exact_utf8_text",
        "training_unique_texts": int(training.size),
        "training_content_receipt": binding(args.training_content_receipt),
        "input_validation_manifest": binding(args.input_manifest),
        "panels": audited,
        "all_final_panels_training_disjoint": all(
            row["final_training_exact_content_overlap_documents"] == 0 for row in audited
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = {
        **manifest,
        "status": "frozen",
        "panels": sorted(panels, key=lambda row: row["name"]),
        "training_content_overlap_audit": binding(audit_path),
        "all_panels_training_exact_content_disjoint": True,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output_manifest) + ".partial")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_manifest)
    print(json.dumps({
        "ok": True,
        "panels": len(panels),
        "rebuilt": {row["name"]: row["original_training_exact_content_overlap_documents"] for row in audited if row["rebuilt"]},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
