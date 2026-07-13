#!/usr/bin/env python3
"""Profile actual acquired fields for Stage 30 after the v4 human gate.

The profiler is deliberately read-only.  It reports source schemas and bounded
value evidence; it does not pick a text/title/author mapping, rewrite text, or
manufacture a six-column envelope.  Those are explicit follow-on decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent1_v4_raw_review import file_binding, read_json_object, write_json_no_replace  # noqa: E402
from full_corpus_io import (  # noqa: E402
    artifact_relative_path,
    artifacts_from_receipt,
    iter_parquet_rows,
    jsonable,
)
from validate_agent1_v4_human_decisions import RECEIPT_SCHEMA  # noqa: E402


PROFILE_SCHEMA = "agent1_v4_field_profile_manifest_v1"
MAX_DISTINCT_VALUES = 100_000
MAX_EXAMPLES = 3
MAX_EXAMPLE_CHARS = 400


def _field_paths(field: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    """Yield visible Arrow leaf paths without exploding list/map value content."""

    import pyarrow as pa

    path = f"{prefix}.{field.name}" if prefix else field.name
    if pa.types.is_struct(field.type):
        for child in field.type:
            yield from _field_paths(child, path)
    else:
        yield path, str(field.type)


def _value_at_path(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _nonblank(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _fingerprint(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bounded_example(value: Any) -> Any:
    value = jsonable(value)
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(rendered) <= MAX_EXAMPLE_CHARS:
        return value
    return rendered[:MAX_EXAMPLE_CHARS] + "…"


def _classify_path(path: str, source: Mapping[str, Any]) -> list[str]:
    top = path.split(".", 1)[0]
    result: list[str] = []
    if top in set(source.get("text_columns", [])):
        result.append("provisional_text")
    if top in set(source.get("alternate_text_columns", []) or []):
        result.append("alternate_text")
    if top in set(source.get("id_columns", [])):
        result.append("stable_id_candidate")
    lowered = path.casefold()
    if any(token in lowered for token in ("title", "titlos", "subject", "headline")):
        result.append("possible_title")
    if any(token in lowered for token in ("author", "creator", "writer", "speaker", "name")):
        result.append("possible_author_or_speaker")
    if any(token in lowered for token in ("url", "link", "handle", "doi")):
        result.append("url_or_locator")
    return result or ["other"]


def _admitted_sources(human_gate_path: Path) -> list[str]:
    receipt = read_json_object(human_gate_path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("status") != "passed":
        raise ValueError("human-gate receipt is not passed")
    rows = receipt.get("admitted_source_ids")
    if not isinstance(rows, list) or any(not isinstance(value, str) or not value for value in rows):
        raise ValueError("human-gate receipt has invalid admitted_source_ids")
    if len(rows) != len(set(rows)):
        raise ValueError("human-gate receipt has duplicate admitted sources")
    return list(rows)


def profile_fields(
    *, sources_path: Path, acquisition_receipt: Path, human_gate_receipt: Path, output: Path
) -> dict[str, object]:
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"field profile output already exists: {output}")
    admitted = _admitted_sources(human_gate_receipt)
    artifacts = artifacts_from_receipt(sources_path, acquisition_receipt, set(admitted)) if admitted else []
    artifact_by_source = {source.source_id: source for source in artifacts}
    if set(artifact_by_source) != set(admitted):
        raise ValueError("admitted source/acquisition closure drift")
    source_reports: list[dict[str, object]] = []
    blocking_issues: list[dict[str, object]] = []
    for source_id in admitted:
        source = artifact_by_source[source_id]
        field_types: dict[str, set[str]] = defaultdict(set)
        counters: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"non_null": 0, "nonblank": 0, "distinct": set(), "distinct_capped": False, "examples": []}
        )
        row_count = 0
        artifact_schemas: list[dict[str, object]] = []
        for artifact in sorted(source.files):
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(artifact)
            paths = list(_field_paths(field) for field in parquet.schema_arrow)
            flattened = [item for group in paths for item in group]
            artifact_schemas.append(
                {
                    "path": artifact_relative_path(source, artifact),
                    "fields": [{"path": path, "arrow_type": arrow_type} for path, arrow_type in flattened],
                }
            )
            for path, arrow_type in flattened:
                field_types[path].add(arrow_type)
            for _, row in iter_parquet_rows(artifact):
                row_count += 1
                for path in field_types:
                    value = _value_at_path(row, path)
                    values = counters[path]
                    if value is not None:
                        values["non_null"] += 1
                    if not _nonblank(value):
                        continue
                    values["nonblank"] += 1
                    if len(values["examples"]) < MAX_EXAMPLES:
                        values["examples"].append(_bounded_example(value))
                    if not values["distinct_capped"]:
                        values["distinct"].add(_fingerprint(value))
                        if len(values["distinct"]) >= MAX_DISTINCT_VALUES:
                            values["distinct_capped"] = True
        fields = []
        for path in sorted(field_types):
            values = counters[path]
            fields.append(
                {
                    "path": path,
                    "arrow_types": sorted(field_types[path]),
                    "classification": _classify_path(path, source.config),
                    "non_null_count": values["non_null"],
                    "nonblank_count": values["nonblank"],
                    "non_null_fraction": values["non_null"] / row_count if row_count else 0.0,
                    "nonblank_fraction": values["nonblank"] / row_count if row_count else 0.0,
                    "distinct_count_capped": len(values["distinct"]),
                    "distinct_count_is_lower_bound": values["distinct_capped"],
                    "examples": values["examples"],
                }
            )
        configured_text = [field for field in fields if "provisional_text" in field["classification"]]
        if not configured_text or not any(int(field["nonblank_count"]) > 0 for field in configured_text):
            blocking_issues.append(
                {
                    "source_id": source_id,
                    "reason": "configured_provisional_text_has_no_nonblank_observed_values",
                    "configured_text_columns": list(source.config.get("text_columns", [])),
                }
            )
        source_reports.append(
            {
                "source_id": source.source_id,
                "repo_id": source.repo_id,
                "revision": source.revision,
                "row_count": row_count,
                "provisional_text_columns": list(source.config.get("text_columns", [])),
                "provisional_id_columns": list(source.config.get("id_columns", [])),
                "artifacts": artifact_schemas,
                "fields": fields,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA,
        "status": "blocked" if blocking_issues else "passed",
        "sources": file_binding(sources_path),
        "acquisition_receipt": file_binding(acquisition_receipt),
        "human_gate_receipt": file_binding(human_gate_receipt),
        "admitted_source_ids": admitted,
        "source_reports": source_reports,
        "blocking_issues": blocking_issues,
        "limits": {"max_distinct_values": MAX_DISTINCT_VALUES, "max_examples": MAX_EXAMPLES, "max_example_chars": MAX_EXAMPLE_CHARS},
    }
    write_json_no_replace(output, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=root / "configs" / "sources.json")
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--human-gate-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = profile_fields(
        sources_path=args.sources,
        acquisition_receipt=args.acquisition_receipt,
        human_gate_receipt=args.human_gate_receipt,
        output=args.output,
    )
    print(json.dumps({"ok": manifest["status"] == "passed", "sources": len(manifest["source_reports"])}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
