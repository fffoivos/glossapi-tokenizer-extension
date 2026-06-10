#!/usr/bin/env python3
"""Reorder only the new-Greek slots of the final CPT JSONL.

The source mix builder interleaves HPLT, openarchives, and replay buckets.  For
the full two-arm CPT run we want the new-Greek subsequence to be cleanly
separable: all HPLT rows first, then all openarchives rows.  Replay must stay
unchanged, so this script preserves every non-new-Greek row at its original line
position and replaces only HPLT/openarchives slots with the ordered new-Greek
queues.

Input rows may either have top-level `source`/`doc_id` (pre-anonymization) or
Datatrove-style `metadata.source` + top-level `id` (post-anonymization).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def source_of(obj: dict[str, Any]) -> str:
    source = obj.get("source")
    if source:
        return str(source)
    metadata = obj.get("metadata")
    if isinstance(metadata, dict) and metadata.get("source"):
        return str(metadata["source"])
    return ""


def id_of(obj: dict[str, Any]) -> str:
    for key in ("doc_id", "id", "doc_key"):
        value = obj.get(key)
        if value is not None:
            return str(value)
    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        for key in ("doc_id", "id", "doc_key"):
            value = metadata.get(key)
            if value is not None:
                return str(value)
    return ""


def update_position_hash(h: "hashlib._Hash", line_no: int, line: str) -> None:
    h.update(str(line_no).encode("ascii"))
    h.update(b"\t")
    h.update(line.encode("utf-8"))
    if not line.endswith("\n"):
        h.update(b"\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--temp-dir", type=Path, default=None)
    ap.add_argument("--hplt-source", default="greek_hplt_70")
    ap.add_argument("--openarchives-source", default="greek_openarchives_30")
    ap.add_argument("--progress-every", type=int, default=1_000_000)
    args = ap.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise SystemExit("input and output must be different paths")
    if not args.input.is_file():
        raise SystemExit(f"input does not exist: {args.input}")

    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    temp_dir = args.temp_dir or args.output.parent / f".{args.output.name}.parts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hplt_tmp = temp_dir / "hplt.jsonl"
    oa_tmp = temp_dir / "openarchives.jsonl"

    t0 = time.time()
    pass1_counts = {
        "total_rows": 0,
        "hplt_rows": 0,
        "openarchives_rows": 0,
        "non_new_greek_rows": 0,
    }
    pass1_bytes = {
        "hplt_bytes": 0,
        "openarchives_bytes": 0,
        "non_new_greek_bytes": 0,
    }
    non_new_input_hash = hashlib.sha256()
    source_counts: dict[str, int] = {}

    print(f"pass1: splitting new-Greek queues from {args.input}", flush=True)
    with args.input.open("r", encoding="utf-8") as src, \
            hplt_tmp.open("w", encoding="utf-8") as hplt_out, \
            oa_tmp.open("w", encoding="utf-8") as oa_out:
        for line_no, line in enumerate(src, start=1):
            obj = json.loads(line)
            source = source_of(obj)
            source_counts[source] = source_counts.get(source, 0) + 1
            pass1_counts["total_rows"] += 1
            if source == args.hplt_source:
                hplt_out.write(line)
                pass1_counts["hplt_rows"] += 1
                pass1_bytes["hplt_bytes"] += len(line.encode("utf-8"))
            elif source == args.openarchives_source:
                oa_out.write(line)
                pass1_counts["openarchives_rows"] += 1
                pass1_bytes["openarchives_bytes"] += len(line.encode("utf-8"))
            else:
                pass1_counts["non_new_greek_rows"] += 1
                pass1_bytes["non_new_greek_bytes"] += len(line.encode("utf-8"))
                update_position_hash(non_new_input_hash, line_no, line)
            if pass1_counts["total_rows"] % args.progress_every == 0:
                print(
                    "  pass1 rows={total_rows:,} hplt={hplt_rows:,} "
                    "openarchives={openarchives_rows:,} non_new={non_new_greek_rows:,}".format(**pass1_counts),
                    flush=True,
                )

    new_greek_slots = pass1_counts["hplt_rows"] + pass1_counts["openarchives_rows"]
    if pass1_counts["hplt_rows"] <= 0 or pass1_counts["openarchives_rows"] <= 0:
        raise SystemExit(f"missing new-Greek source rows: {pass1_counts}")
    print(
        f"pass1 done: total={pass1_counts['total_rows']:,} "
        f"new_greek_slots={new_greek_slots:,} "
        f"hplt={pass1_counts['hplt_rows']:,} "
        f"openarchives={pass1_counts['openarchives_rows']:,}",
        flush=True,
    )

    print(f"pass2: writing slot-preserving ordered stream to {args.output}", flush=True)
    pass2_counts = {
        "total_rows": 0,
        "hplt_rows_written": 0,
        "openarchives_rows_written": 0,
        "non_new_greek_rows_written": 0,
        "new_greek_slots_seen": 0,
    }
    non_new_output_hash = hashlib.sha256()
    ordered_violation_count = 0
    first_openarchives_new_greek_slot = None
    last_hplt_new_greek_slot = None
    hplt_ids = hashlib.sha256()
    openarchives_ids = hashlib.sha256()

    with args.input.open("r", encoding="utf-8") as src, \
            hplt_tmp.open("r", encoding="utf-8") as hplt_in, \
            oa_tmp.open("r", encoding="utf-8") as oa_in, \
            args.output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            obj = json.loads(line)
            source = source_of(obj)
            if source in (args.hplt_source, args.openarchives_source):
                pass2_counts["new_greek_slots_seen"] += 1
                slot_no = pass2_counts["new_greek_slots_seen"]
                if pass2_counts["hplt_rows_written"] < pass1_counts["hplt_rows"]:
                    out_line = hplt_in.readline()
                    out_obj = json.loads(out_line)
                    out_source = source_of(out_obj)
                    if out_source != args.hplt_source:
                        raise SystemExit(f"HPLT temp emitted wrong source at slot {slot_no}: {out_source}")
                    pass2_counts["hplt_rows_written"] += 1
                    last_hplt_new_greek_slot = slot_no
                    hplt_ids.update(id_of(out_obj).encode("utf-8"))
                    hplt_ids.update(b"\n")
                else:
                    out_line = oa_in.readline()
                    out_obj = json.loads(out_line)
                    out_source = source_of(out_obj)
                    if out_source != args.openarchives_source:
                        raise SystemExit(f"openarchives temp emitted wrong source at slot {slot_no}: {out_source}")
                    pass2_counts["openarchives_rows_written"] += 1
                    if first_openarchives_new_greek_slot is None:
                        first_openarchives_new_greek_slot = slot_no
                    openarchives_ids.update(id_of(out_obj).encode("utf-8"))
                    openarchives_ids.update(b"\n")
                if first_openarchives_new_greek_slot is not None and out_source == args.hplt_source:
                    ordered_violation_count += 1
                dst.write(out_line)
            else:
                dst.write(line)
                pass2_counts["non_new_greek_rows_written"] += 1
                update_position_hash(non_new_output_hash, line_no, line)
            pass2_counts["total_rows"] += 1
            if pass2_counts["total_rows"] % args.progress_every == 0:
                print(
                    "  pass2 rows={total_rows:,} hplt_written={hplt_rows_written:,} "
                    "openarchives_written={openarchives_rows_written:,} "
                    "non_new_written={non_new_greek_rows_written:,}".format(**pass2_counts),
                    flush=True,
                )

    # The row counts above are the authoritative queue-exhaustion check.
    if pass2_counts["hplt_rows_written"] != pass1_counts["hplt_rows"]:
        raise SystemExit(f"HPLT row mismatch: {pass2_counts} vs {pass1_counts}")
    if pass2_counts["openarchives_rows_written"] != pass1_counts["openarchives_rows"]:
        raise SystemExit(f"openarchives row mismatch: {pass2_counts} vs {pass1_counts}")
    if pass2_counts["non_new_greek_rows_written"] != pass1_counts["non_new_greek_rows"]:
        raise SystemExit(f"non-new row mismatch: {pass2_counts} vs {pass1_counts}")
    if pass2_counts["total_rows"] != pass1_counts["total_rows"]:
        raise SystemExit(f"total row mismatch: {pass2_counts} vs {pass1_counts}")

    non_new_input_digest = non_new_input_hash.hexdigest()
    non_new_output_digest = non_new_output_hash.hexdigest()
    replay_positions_preserved = non_new_input_digest == non_new_output_digest
    if not replay_positions_preserved:
        raise SystemExit("non-new-Greek position hash changed; replay was not preserved")

    manifest = {
        "schema": "new-greek-slot-order-v1",
        "input": str(args.input),
        "output": str(args.output),
        "temp_dir": str(temp_dir),
        "hplt_source": args.hplt_source,
        "openarchives_source": args.openarchives_source,
        "policy": "preserve all non-new-Greek rows at original line positions; reorder only HPLT/openarchives slots",
        "counts": {**pass1_counts, **pass2_counts},
        "bytes": pass1_bytes,
        "source_counts": dict(sorted(source_counts.items())),
        "new_greek_order": {
            "hplt_first": True,
            "last_hplt_new_greek_slot": last_hplt_new_greek_slot,
            "first_openarchives_new_greek_slot": first_openarchives_new_greek_slot,
            "ordered_violation_count": ordered_violation_count,
        },
        "replay_preservation": {
            "non_new_position_line_hash_input": non_new_input_digest,
            "non_new_position_line_hash_output": non_new_output_digest,
            "non_new_positions_preserved": replay_positions_preserved,
        },
        "id_hashes": {
            "hplt_ids_sha256": hplt_ids.hexdigest(),
            "openarchives_ids_sha256": openarchives_ids.hexdigest(),
        },
        "wall_seconds": time.time() - t0,
        "output_size_bytes": args.output.stat().st_size,
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(args.output.stat().st_mtime)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest: {manifest_path}", flush=True)
    print(json.dumps(manifest["new_greek_order"], indent=2), flush=True)
    print(json.dumps(manifest["replay_preservation"], indent=2), flush=True)

    try:
        hplt_tmp.unlink()
        oa_tmp.unlink()
        temp_dir.rmdir()
    except OSError:
        print(f"warning: temp files left in {temp_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
