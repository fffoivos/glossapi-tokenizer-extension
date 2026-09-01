#!/usr/bin/env python3
"""Verify the frozen modern+polytonic tokenizer before publication.

The audit proves that alignment is a property of the actual BPE vocabulary,
not model-side padding: every appended vocabulary entry must be created by
exactly one appended merge, in dependency-safe and ID-sequential order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
FRONT_END_KEYS = (
    "version",
    "truncation",
    "padding",
    "added_tokens",
    "normalizer",
    "pre_tokenizer",
    "post_processor",
    "decoder",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def split_merge(merge: Any) -> tuple[str, str]:
    if isinstance(merge, list) and len(merge) == 2:
        return str(merge[0]), str(merge[1])
    if isinstance(merge, str):
        parts = merge.split(" ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    raise ValueError(f"unsupported merge representation: {merge!r}")


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    evidence: Any,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})


def verify_runtime(
    candidate_dir: Path,
    expected_vocab_size: int,
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        return {
            "status": "skipped",
            "reason": f"transformers unavailable: {exc}",
        }

    tokenizer = AutoTokenizer.from_pretrained(
        str(candidate_dir),
        local_files_only=True,
    )
    runtime_vocab = tokenizer.get_vocab()
    runtime_ids = sorted(runtime_vocab.values())
    probes = [
        "Η ελληνική γλώσσα είναι ζωντανή.",
        "Ἐν ἀρχῇ ἦν ὁ Λόγος.",
        "Ἡ φιλοσοφία τῶν ἀρχαίων Ἑλλήνων.",
        "Greek and English control: 12345.",
    ]
    roundtrips = []
    for text in probes:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        roundtrips.append(
            {
                "text": text,
                "token_count": len(token_ids),
                "exact": decoded == text,
            }
        )

    failures = []
    if tokenizer.vocab_size != expected_vocab_size:
        failures.append(
            f"vocab_size={tokenizer.vocab_size}, expected={expected_vocab_size}"
        )
    if len(tokenizer) != expected_vocab_size:
        failures.append(f"len(tokenizer)={len(tokenizer)}, expected={expected_vocab_size}")
    if runtime_ids != list(range(expected_vocab_size)):
        failures.append("runtime vocabulary IDs are not contiguous")
    if not all(row["exact"] for row in roundtrips):
        failures.append("at least one runtime round trip failed")

    return {
        "status": "passed" if not failures else "failed",
        "class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "len_tokenizer": len(tokenizer),
        "special_token_ids": {
            "unk": tokenizer.unk_token_id,
            "bos": tokenizer.bos_token_id,
            "eos": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token_id,
        },
        "roundtrips": roundtrips,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--expected-vocab-size", type=int, default=148_992)
    parser.add_argument("--expected-added-merges", type=int, default=512)
    parser.add_argument("--alignment", type=int, default=256)
    parser.add_argument("--require-runtime", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_path = args.base_dir / "tokenizer.json"
    candidate_path = args.candidate_dir / "tokenizer.json"
    base = load_json(base_path)
    candidate = load_json(candidate_path)
    base_model = base["model"]
    candidate_model = candidate["model"]
    base_vocab = base_model["vocab"]
    candidate_vocab = candidate_model["vocab"]
    base_merges = base_model["merges"]
    candidate_merges = candidate_model["merges"]
    checks: list[dict[str, Any]] = []

    candidate_ids = sorted(candidate_vocab.values())
    add_check(
        checks,
        "actual_vocab_size",
        len(candidate_vocab) == args.expected_vocab_size,
        len(candidate_vocab),
    )
    add_check(
        checks,
        "actual_vocab_ids_contiguous",
        candidate_ids == list(range(len(candidate_vocab))),
        {
            "min": candidate_ids[0],
            "max": candidate_ids[-1],
            "unique": len(set(candidate_ids)),
        },
    )
    quotient, remainder = divmod(len(candidate_vocab), args.alignment)
    add_check(
        checks,
        "actual_vocab_alignment_without_external_padding",
        remainder == 0,
        {
            "actual_vocab_size": len(candidate_vocab),
            "divisor": args.alignment,
            "quotient": quotient,
            "remainder": remainder,
            "external_padding_tokens": 0 if remainder == 0 else args.alignment - remainder,
        },
    )

    base_by_id = {token_id: token for token, token_id in base_vocab.items()}
    candidate_by_id = {
        token_id: token for token, token_id in candidate_vocab.items()
    }
    base_ids_exact = all(
        candidate_by_id.get(token_id) == token
        for token_id, token in base_by_id.items()
    )
    add_check(
        checks,
        "base_vocab_ids_exact",
        base_ids_exact,
        {"base_vocab_size": len(base_vocab)},
    )
    add_check(
        checks,
        "base_merge_prefix_exact",
        candidate_merges[: len(base_merges)] == base_merges,
        {
            "base_merge_count": len(base_merges),
            "candidate_merge_count": len(candidate_merges),
        },
    )

    added_merges = candidate_merges[len(base_merges) :]
    add_check(
        checks,
        "appended_merge_count",
        len(added_merges) == args.expected_added_merges,
        len(added_merges),
    )

    live = set(base_vocab)
    produced: list[str] = []
    merge_errors: list[str] = []
    for offset, merge in enumerate(added_merges):
        try:
            left, right = split_merge(merge)
        except ValueError as exc:
            merge_errors.append(f"offset {offset}: {exc}")
            continue
        result = left + right
        expected_id = len(base_vocab) + offset
        if left not in live:
            merge_errors.append(f"offset {offset}: left operand not live")
        if right not in live:
            merge_errors.append(f"offset {offset}: right operand not live")
        if result in live:
            merge_errors.append(f"offset {offset}: duplicate merge result")
        if candidate_vocab.get(result) != expected_id:
            merge_errors.append(
                f"offset {offset}: result ID {candidate_vocab.get(result)!r}, "
                f"expected {expected_id}"
            )
        live.add(result)
        produced.append(result)

    new_vocab = {
        token
        for token, token_id in candidate_vocab.items()
        if token_id >= len(base_vocab)
    }
    produced_set = set(produced)
    orphan_new_tokens = sorted(new_vocab - produced_set)
    missing_new_tokens = sorted(produced_set - new_vocab)
    add_check(
        checks,
        "appended_merges_dependency_safe_and_id_sequential",
        not merge_errors,
        {"error_count": len(merge_errors), "errors": merge_errors[:20]},
    )
    add_check(
        checks,
        "no_padded_or_orphan_new_vocab_entries",
        not orphan_new_tokens
        and not missing_new_tokens
        and len(new_vocab) == args.expected_added_merges,
        {
            "new_vocab_entries": len(new_vocab),
            "merge_results": len(produced_set),
            "orphan_count": len(orphan_new_tokens),
            "missing_count": len(missing_new_tokens),
        },
    )

    front_end_differences = {
        key: {"base": base.get(key), "candidate": candidate.get(key)}
        for key in FRONT_END_KEYS
        if base.get(key) != candidate.get(key)
    }
    add_check(
        checks,
        "front_end_contract_exact",
        not front_end_differences,
        {"differing_keys": sorted(front_end_differences)},
    )
    base_model_options = {
        key: value
        for key, value in base_model.items()
        if key not in {"vocab", "merges"}
    }
    candidate_model_options = {
        key: value
        for key, value in candidate_model.items()
        if key not in {"vocab", "merges"}
    }
    add_check(
        checks,
        "bpe_model_options_exact",
        base_model_options == candidate_model_options,
        {
            "base": base_model_options,
            "candidate": candidate_model_options,
        },
    )
    add_check(
        checks,
        "no_new_added_tokens",
        candidate.get("added_tokens") == base.get("added_tokens"),
        {
            "base_count": len(base.get("added_tokens", [])),
            "candidate_count": len(candidate.get("added_tokens", [])),
        },
    )
    add_check(
        checks,
        "tokenizer_padding_configuration_unchanged",
        candidate.get("padding") == base.get("padding"),
        {
            "base": base.get("padding"),
            "candidate": candidate.get("padding"),
        },
    )

    sidecar_hashes = {}
    sidecar_differences = []
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        base_sidecar = args.base_dir / name
        candidate_sidecar = args.candidate_dir / name
        sidecar_hashes[name] = {
            "base": sha256_path(base_sidecar),
            "candidate": sha256_path(candidate_sidecar),
        }
        if base_sidecar.read_bytes() != candidate_sidecar.read_bytes():
            sidecar_differences.append(name)
    add_check(
        checks,
        "hf_sidecars_exact",
        not sidecar_differences,
        {
            "differing_files": sidecar_differences,
            "hashes": sidecar_hashes,
        },
    )

    runtime = verify_runtime(args.candidate_dir, args.expected_vocab_size)
    runtime_ok = runtime["status"] == "passed" or (
        runtime["status"] == "skipped" and not args.require_runtime
    )
    add_check(checks, "transformers_runtime", runtime_ok, runtime)

    failed_checks = [check["name"] for check in checks if not check["passed"]]
    report = {
        "schema_version": "apertus-production-tokenizer-release-audit-v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "passed" if not failed_checks else "failed",
        "base": {
            "path": str(args.base_dir),
            "vocab_size": len(base_vocab),
            "tokenizer_json_sha256": sha256_path(base_path),
        },
        "candidate": {
            "path": str(args.candidate_dir),
            "vocab_size": len(candidate_vocab),
            "tokenizer_json_sha256": sha256_path(candidate_path),
            "files": {
                name: sha256_path(args.candidate_dir / name)
                for name in TOKENIZER_FILES
            },
        },
        "alignment": {
            "divisor": args.alignment,
            "quotient": quotient,
            "remainder": remainder,
            "external_padding_tokens": 0 if remainder == 0 else args.alignment - remainder,
        },
        "checks": checks,
        "failed_checks": failed_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "actual_vocab_size": len(candidate_vocab),
                "alignment": report["alignment"],
                "added_merges": len(added_merges),
                "new_orphan_tokens": len(orphan_new_tokens),
                "failed_checks": failed_checks,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
