#!/usr/bin/env python3
"""Dependency-free closure checker for the guarded serial signature handoff."""

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile


REQUEST_SCHEMA = "agent1_v5_dedup_acceleration_takeover_request_v1"
ARM_SCHEMA = "agent1_v5_dedup_acceleration_takeover_arm_v1"
STOP_SCHEMA = "agent1_v5_dedup_acceleration_sentinel_stop_v1"
QUEUE_SCHEMA = "agent1_v5_dedup_acceleration_sentinel_queue_evidence_v1"
SIGNATURE_SCHEMA = "agent1_v5_minhash_signature_task_receipt_v1"
CUTOVER_SCHEMA = "agent1_v5_dedup_acceleration_cutover_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path, schema):
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != schema or value.get("status") != "passed":
        raise ValueError("required receipt is not passed: {}".format(path))
    return value


def validate_binding(binding, root):
    path = binding.get("path", "")
    path = path if os.path.isabs(path) else os.path.join(root, path)
    path = os.path.realpath(path)
    if not os.path.isfile(path) or os.path.getsize(path) != int(binding.get("bytes", -1)) or sha256_file(path) != binding.get("sha256"):
        raise ValueError("file receipt mismatch: {}".format(path))
    return path


def validate_receipts(run, through_rank):
    receipts = []
    for rank in range(through_rank + 1):
        path = os.path.join(run, "60-dedup", "minhash-signatures", "receipts", "{:06d}.json".format(rank))
        receipt = read(path, SIGNATURE_SCHEMA)
        if int(receipt.get("task_index", -1)) != rank:
            raise ValueError("signature receipt rank drift: {}".format(path))
        outputs = receipt.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 32:
            raise ValueError("signature receipt output closure failed: {}".format(path))
        for binding in outputs:
            validate_binding(binding, run)
        receipts.append((path, receipt))
    return receipts


def write_immutable(path, value):
    if os.path.exists(path):
        raise FileExistsError("refusing to overwrite immutable cutover: {}".format(path))
    parent = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix=".sentinel-cutover-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--arm-receipt", required=True)
    parser.add_argument("--stop-receipt", required=True)
    parser.add_argument("--queue-evidence", required=True)
    parser.add_argument("--combined-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    request = read(args.request, REQUEST_SCHEMA)
    arm = read(args.arm_receipt, ARM_SCHEMA)
    stop = read(args.stop_receipt, STOP_SCHEMA)
    queue = read(args.queue_evidence, QUEUE_SCHEMA)
    run = os.path.realpath(request.get("run_root", ""))
    if not os.path.isdir(run):
        raise ValueError("sentinel request run root is unavailable")
    stop_rank = int(request.get("stop_after_rank", -1))
    if stop_rank < 0 or request.get("combined_manifest_sha256") != sha256_file(args.combined_manifest):
        raise ValueError("sentinel request rank or manifest binding drift")
    request_sha = sha256_file(args.request)
    if arm.get("request_sha256") != request_sha or stop.get("request_sha256") != request_sha:
        raise ValueError("sentinel evidence does not bind request")
    if int(arm.get("stop_after_rank", -1)) != stop_rank or int(stop.get("stopped_after_rank", -1)) != stop_rank:
        raise ValueError("sentinel stop rank drift")
    if int(stop.get("first_missing_rank", -1)) != stop_rank + 1 or stop.get("successor_submitted") is not False:
        raise ValueError("sentinel successor closure drift")
    for item in (arm, stop):
        if item.get("active_helper_sha256") != request.get("guarded_helper_sha256") or item.get("takeover_tool_sha256") != request.get("takeover_tool_sha256"):
            raise ValueError("sentinel helper/tool checksum drift")
    if queue.get("debug_signature_queue_empty") is not True or queue.get("legacy_successor_present") is not False:
        raise ValueError("sentinel queue evidence is incomplete")
    receipt_path = os.path.join(run, "60-dedup", "minhash-signatures", "receipts", "{:06d}.json".format(stop_rank))
    receipt_binding = stop.get("signature_receipt")
    if not isinstance(receipt_binding, dict) or receipt_binding.get("path") != os.path.realpath(receipt_path) or receipt_binding.get("sha256") != sha256_file(receipt_path):
        raise ValueError("sentinel stop receipt binding drift")
    receipts = validate_receipts(run, stop_rank)
    value = {
        "schema_version": CUTOVER_SCHEMA,
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "sentinel",
        "run_root": run,
        "request_sha256": request_sha,
        "arm_receipt_sha256": sha256_file(args.arm_receipt),
        "stop_receipt_sha256": sha256_file(args.stop_receipt),
        "queue_evidence_sha256": sha256_file(args.queue_evidence),
        "final_legacy_rank": stop_rank,
        "first_missing_rank": stop_rank + 1,
        "legacy_receipt_count": len(receipts),
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
    }
    write_immutable(args.output, value)
    print(json.dumps({"ok": True, "first_missing_rank": stop_rank + 1, "method": "sentinel"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
