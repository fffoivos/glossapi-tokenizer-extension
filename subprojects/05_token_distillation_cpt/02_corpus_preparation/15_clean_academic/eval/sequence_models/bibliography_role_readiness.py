#!/usr/bin/env python3
"""Report whether trusted role labels satisfy the frozen scaling gates."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contract import sha256_file


SOURCES = ("greek_phd", "kallipos", "openarchives")
TRUSTED = frozenset({"AGREED_REVIEW", "ADJUDICATED"})


def _rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping) or row.get("schema_version") != "bibliography-role-overlay-v2":
                raise ValueError(f"{path}:{number}: expected overlay v2 row")
            yield row


def evaluate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    role_by_source: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    role_totals: collections.Counter[str] = collections.Counter()
    boundary_stops = 0
    trusted_rows = 0
    for row in rows:
        source = str(row.get("source", ""))
        # Overlay v2 intentionally does not duplicate source. Recover it from a
        # caller-supplied source map in later all-corpus views; pilot overlays
        # can instead infer source from the selected packet before this call.
        if source not in SOURCES:
            source = str(row.get("_source", ""))
        if row.get("role_status") in TRUSTED:
            trusted_rows += 1
            role = str(row["role"])
            role_totals[role] += 1
            if source in SOURCES:
                role_by_source[source][role] += 1
        if row.get("boundary_status") in TRUSTED and row.get("boundary_flag") in {"SOFT_STOP", "HARD_STOP"}:
            boundary_stops += 1
    gates: list[dict[str, Any]] = []
    for source in SOURCES:
        gates.append(
            {"gate": f"ENTRY_ANCHOR:{source}", "observed": role_by_source[source]["ENTRY_ANCHOR"],
             "required": 500}
        )
    for role in ("CONTINUATION", "FILLER"):
        gates.append({"gate": f"{role}:overall", "observed": role_totals[role], "required": 200})
        for source in SOURCES:
            gates.append(
                {"gate": f"{role}:{source}", "observed": role_by_source[source][role], "required": 30}
            )
    for source in SOURCES:
        headers = role_by_source[source]["HEADER"] + role_by_source[source]["SUBHEADER"]
        gates.append({"gate": f"HEADER_OR_SUBHEADER:{source}", "observed": headers, "required": 50})
    gates.append({"gate": "BOUNDARY_STOPS:overall", "observed": boundary_stops, "required": 100})
    for gate in gates:
        gate["passed"] = int(gate["observed"]) >= int(gate["required"])
        gate["deficit"] = max(0, int(gate["required"]) - int(gate["observed"]))
    return {
        "schema_version": "bibliography-role-readiness-v1",
        "status": "ready" if all(row["passed"] for row in gates) else "more_review_required",
        "trusted_line_count": trusted_rows,
        "role_totals": dict(sorted(role_totals.items())),
        "role_by_source": {source: dict(sorted(role_by_source[source].items())) for source in SOURCES},
        "trusted_boundary_stop_count": boundary_stops,
        "gates": gates,
        "failed_gates": [row["gate"] for row in gates if not row["passed"]],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--packet", required=True, help="blind packet used to recover source by line identity")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    overlay_path, packet_path = Path(args.overlay).resolve(), Path(args.packet).resolve()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    source_by_key = {
        (str(case["document_id"]), str(line["line_id"])): str(case["source"])
        for case in packet["cases"] for line in case["lines"]
    }
    rows = []
    for row in _rows(overlay_path):
        local = dict(row)
        key = (str(local["document_id"]), str(local["line_id"]))
        if key not in source_by_key:
            raise ValueError(f"overlay line absent from packet: {key}")
        local["_source"] = source_by_key[key]
        rows.append(local)
    report = evaluate(rows)
    report["inputs"] = {
        "overlay_sha256": sha256_file(overlay_path), "packet_sha256": sha256_file(packet_path)
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
