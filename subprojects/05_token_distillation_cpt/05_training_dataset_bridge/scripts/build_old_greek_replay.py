#!/usr/bin/env python3
"""Rebuild old-Greek replay from pinned Nanochat and Apertus-overlay receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from bridge_common import (
    canonical_sha256,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


def _render(raw: str, scratch_root: Path) -> Path:
    return Path(raw.format(scratch_root=str(scratch_root.resolve()))).resolve()


def _validate_acquired(row: dict[str, Any]) -> Path:
    path = Path(str(row["path"])).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row["bytes"])
        or sha256_file(path) != row["sha256"]
    ):
        raise ValueError(f"acquired old-Greek input drift: {path}")
    return path


def _validate_existing(
    path: Path,
    *,
    acquisition_sha: str,
    config_sha: str,
    implementation_sha: str,
) -> bool:
    if not path.is_file():
        return False
    value = read_json(path)
    if (
        value.get("schema_version") != "full_cpt_old_greek_build_receipt_v1"
        or value.get("status") != "completed"
        or value.get("acquisition_receipt_sha256") != acquisition_sha
        or value.get("config_sha256") != config_sha
        or value.get("implementation_sha256") != implementation_sha
    ):
        raise ValueError("existing old-Greek build receipt has different bindings")
    for row in value.get("outputs", []):
        _validate_acquired(row)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--replace-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    if config.get("schema_version") != "full_cpt_replay_acquisition_config_v1":
        raise ValueError("unsupported replay acquisition config")
    acquisition = read_json(args.acquisition_receipt)
    if (
        acquisition.get("schema_version") != "full_cpt_replay_acquisition_receipt_v1"
        or acquisition.get("status") != "completed"
        or acquisition.get("config_sha256") != sha256_file(args.config.resolve())
    ):
        raise ValueError("old-Greek build requires the matching completed acquisition")
    acquisition_sha = sha256_file(args.acquisition_receipt.resolve())
    config_sha = sha256_file(args.config.resolve())
    implementation_sha = sha256_file(Path(__file__).resolve())
    if _validate_existing(
        args.output_receipt,
        acquisition_sha=acquisition_sha,
        config_sha=config_sha,
        implementation_sha=implementation_sha,
    ):
        print(
            json.dumps(
                {"ok": True, "resumed": True, "output": str(args.output_receipt)}
            )
        )
        return 0

    nanochat_rows = [
        row for row in acquisition["outputs"] if row["role"] == "old_greek_input"
    ]
    overlay_rows = [
        row for row in acquisition["outputs"] if row["role"] == "old_greek_overlay"
    ]
    if not nanochat_rows or len(overlay_rows) != 1:
        raise ValueError("acquisition has an incomplete old-Greek input inventory")
    nanochat_paths = [_validate_acquired(row) for row in nanochat_rows]
    overlay_path = _validate_acquired(overlay_rows[0])

    build = config["old_greek_build"]
    expected_nanochat = sorted(
        path.resolve()
        for path in _render(str(build["nanochat_glob"]), args.scratch_root).parent.glob(
            _render(str(build["nanochat_glob"]), args.scratch_root).name
        )
        if path.is_file()
    )
    if expected_nanochat != sorted(nanochat_paths):
        raise ValueError("live Nanochat inventory differs from acquisition receipt")
    if _render(str(build["overlay"]), args.scratch_root) != overlay_path:
        raise ValueError(
            "configured old-Greek overlay differs from acquisition receipt"
        )
    output = _render(str(build["output"]), args.scratch_root)
    if output.exists() and not args.replace_existing:
        raise ValueError(
            "an unreceipted old-Greek output cannot be trusted; explicit replacement "
            f"is required: {output}"
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    overlay_parquet = pq.ParquetFile(overlay_path)
    required_overlay = {"source_dataset", "source_doc_id"}
    if not required_overlay.issubset(overlay_parquet.schema_arrow.names):
        raise ValueError("Apertus-overlap overlay misses its composite identity fields")
    remaining: set[tuple[str, str]] = set()
    for batch in overlay_parquet.iter_batches(
        columns=["source_dataset", "source_doc_id"], batch_size=1_000_000
    ):
        data = batch.to_pydict()
        remaining.update(
            (str(source), str(document))
            for source, document in zip(
                data["source_dataset"], data["source_doc_id"], strict=True
            )
            if source is not None and document is not None
        )
    if not remaining:
        raise ValueError("Apertus-overlap overlay has no usable composite identities")
    overlay_identities = len(remaining)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.unlink(missing_ok=True)
    schema = pa.schema(
        [
            ("text", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("composite_doc_id", pa.string()),
        ]
    )
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    scanned = kept = characters = 0
    try:
        for path in sorted(nanochat_paths):
            parquet = pq.ParquetFile(path)
            columns = set(parquet.schema_arrow.names)
            text_column = next(
                (value for value in ("text", "content") if value in columns), None
            )
            if text_column is None or not {
                "source_dataset",
                "source_doc_id",
            }.issubset(columns):
                raise ValueError(f"Nanochat old-Greek input schema drift: {path}")
            for batch in parquet.iter_batches(
                columns=["source_dataset", "source_doc_id", text_column],
                batch_size=20_000,
                use_threads=False,
            ):
                data = batch.to_pydict()
                texts: list[str] = []
                sources: list[str] = []
                document_ids: list[str] = []
                composite_ids: list[str] = []
                for source, document, text in zip(
                    data["source_dataset"],
                    data["source_doc_id"],
                    data[text_column],
                    strict=True,
                ):
                    scanned += 1
                    if (
                        source is None
                        or document is None
                        or not isinstance(text, str)
                        or not text
                    ):
                        continue
                    identity = (str(source), str(document))
                    if identity not in remaining:
                        continue
                    remaining.remove(identity)
                    texts.append(text)
                    sources.append(identity[0])
                    document_ids.append(identity[1])
                    composite_ids.append(
                        "oldgreek:"
                        + canonical_sha256(
                            {
                                "contract": "old-greek-composite-identity-v1",
                                "source_dataset": identity[0],
                                "source_doc_id": identity[1],
                            }
                        )
                    )
                    kept += 1
                    characters += len(text)
                if texts:
                    writer.write_table(
                        pa.table(
                            {
                                "text": texts,
                                "source_dataset": sources,
                                "source_doc_id": document_ids,
                                "composite_doc_id": composite_ids,
                            },
                            schema=schema,
                        )
                    )
    finally:
        writer.close()
    if kept <= 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("old-Greek intersection produced no documents")
    os.replace(temporary, output)
    output_receipt = {
        "path": str(output),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "rows": kept,
        "columns": schema.names,
        "role": "old_greek_replay",
    }
    payload = {
        "schema_version": "full_cpt_old_greek_build_receipt_v1",
        "status": "completed",
        "completed_at": utc_now(),
        "config": str(args.config.resolve()),
        "config_sha256": config_sha,
        "acquisition_receipt": str(args.acquisition_receipt.resolve()),
        "acquisition_receipt_sha256": acquisition_sha,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": implementation_sha,
        "identity_contract": {
            "version": "old-greek-composite-identity-v1",
            "columns": ["source_dataset", "source_doc_id"],
            "scope": "global",
        },
        "join": "exact (source_dataset, source_doc_id) intersection",
        "inputs": {
            "nanochat": nanochat_rows,
            "apertus_overlap_overlay": overlay_rows[0],
        },
        "counts": {
            "nanochat_rows_scanned": scanned,
            "overlay_composite_identities": overlay_identities,
            "matched_unique_documents": kept,
            "unmatched_overlay_identities": len(remaining),
            "characters": characters,
        },
        "outputs": [output_receipt],
    }
    write_json_atomic(args.output_receipt.resolve(), payload)
    print(json.dumps({"ok": True, "output": str(output), "rows": kept}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
