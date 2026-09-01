#!/usr/bin/env python3
"""Audit a polytonic cutoff and review byte-fragment tokens.

ByteLevel BPE vocab entries are byte strings, not necessarily standalone
Unicode strings.  A token that begins with a UTF-8 continuation byte can be a
legitimate intermediate merge which completes a code point together with an
adjacent token.  This audit distinguishes those structural pieces from
unresolved byte fragments and verifies the append-only tokenizer contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def bytes_to_unicode() -> dict[int, str]:
    """Return the reversible GPT-2/ByteLevel byte-to-character map."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(codepoint) for codepoint in cs)))


BYTE_DECODER = {char: byte for byte, char in bytes_to_unicode().items()}


def bytelevel_bytes(token: str) -> bytes:
    return bytes(BYTE_DECODER[char] for char in token)


def decode_bytes(raw: bytes) -> str | None:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def escaped_bytes(raw: bytes) -> str:
    return raw.decode("utf-8", errors="backslashreplace")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_merge(merge: Any) -> tuple[str, str] | None:
    if isinstance(merge, list) and len(merge) == 2:
        return str(merge[0]), str(merge[1])
    if isinstance(merge, str):
        parts = merge.split(" ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return None


def load_tokenizer(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_cutoff_contract(
    base: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    base_vocab = base["model"]["vocab"]
    candidate_vocab = candidate["model"]["vocab"]
    base_merges = base["model"]["merges"]
    candidate_merges = candidate["model"]["merges"]
    errors: list[str] = []

    if candidate_merges[: len(base_merges)] != base_merges:
        errors.append("candidate merge table does not preserve the base prefix")

    base_by_id = {token_id: token for token, token_id in base_vocab.items()}
    candidate_by_id = {
        token_id: token for token, token_id in candidate_vocab.items()
    }
    for token_id, token in sorted(base_by_id.items()):
        if candidate_by_id.get(token_id) != token:
            errors.append(f"base vocab mismatch at id {token_id}")
            if len(errors) >= 20:
                break

    live = set(base_vocab)
    added_merges = candidate_merges[len(base_merges) :]
    expected_id = len(base_vocab)
    for offset, merge in enumerate(added_merges):
        parts = split_merge(merge)
        if parts is None:
            errors.append(f"bad merge format at added offset {offset}: {merge!r}")
            continue
        left, right = parts
        result = left + right
        if left not in live or right not in live:
            errors.append(
                f"dangling merge at added offset {offset}: "
                f"{left!r} + {right!r}"
            )
        actual_id = candidate_vocab.get(result)
        if actual_id != expected_id:
            errors.append(
                f"non-sequential result id at added offset {offset}: "
                f"{actual_id!r} != {expected_id}"
            )
        live.add(result)
        expected_id += 1

    expected_vocab_size = len(base_vocab) + len(added_merges)
    if len(candidate_vocab) != expected_vocab_size:
        errors.append(
            f"vocab/merge size mismatch: {len(candidate_vocab)} != "
            f"{expected_vocab_size}"
        )

    return {
        "status": "passed" if not errors else "failed",
        "base_vocab_size": len(base_vocab),
        "candidate_vocab_size": len(candidate_vocab),
        "added_merge_count": len(added_merges),
        "alignment_256": len(candidate_vocab) % 256 == 0,
        "errors": errors,
    }


def build_consumer_graph(
    tokenizer: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    vocab = tokenizer["model"]["vocab"]
    consumers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for merge_index, merge in enumerate(tokenizer["model"]["merges"]):
        parts = split_merge(merge)
        if parts is None:
            continue
        left, right = parts
        result = left + right
        result_id = vocab.get(result)
        edge = {
            "merge_index": merge_index,
            "left": left,
            "right": right,
            "result": result,
            "result_id": result_id,
        }
        consumers[left].append(edge)
        if right != left:
            consumers[right].append(edge)
    return consumers


def valid_descendants(
    token: str,
    consumers: dict[str, list[dict[str, Any]]],
    *,
    max_depth: int,
    limit: int,
) -> list[dict[str, Any]]:
    queue: deque[tuple[str, int, list[int]]] = deque([(token, 0, [])])
    visited = {token}
    resolved: list[dict[str, Any]] = []
    while queue and len(resolved) < limit:
        current, depth, path = queue.popleft()
        if depth >= max_depth:
            continue
        for edge in consumers.get(current, []):
            result = edge["result"]
            if result in visited:
                continue
            visited.add(result)
            next_path = path + [edge["merge_index"]]
            raw = bytelevel_bytes(result)
            decoded = decode_bytes(raw)
            if decoded is not None:
                resolved.append(
                    {
                        "depth": depth + 1,
                        "result_id": edge["result_id"],
                        "decoded": decoded,
                        "raw_bytes_hex": raw.hex(" "),
                        "merge_path": next_path,
                    }
                )
                if len(resolved) >= limit:
                    break
            queue.append((result, depth + 1, next_path))
    return resolved


def load_reviewed_ids(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("source summary must be a JSON list")
    return sorted({int(row["id"]) for row in payload})


def probe_roundtrip(
    tokenizer_path: Path,
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return {
            "status": "skipped",
            "reason": "the optional tokenizers package is not installed",
            "probes": [],
        }

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    probes: list[dict[str, Any]] = []
    failures: list[str] = []
    for review in reviews:
        if not review["in_candidate"]:
            continue
        descendants = review["valid_utf8_descendants"]
        if not descendants:
            continue
        witness = descendants[0]["decoded"]
        encoding = tokenizer.encode(witness, add_special_tokens=False)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        exact = decoded == witness
        if not exact:
            failures.append(
                f"id {review['id']} witness roundtrip mismatch: "
                f"{witness!r} -> {decoded!r}"
            )
        probes.append(
            {
                "reviewed_id": review["id"],
                "witness": witness,
                "encoded_ids": encoding.ids,
                "reviewed_id_emitted": review["id"] in encoding.ids,
                "roundtrip_exact": exact,
            }
        )
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--candidate-tokenizer", type=Path, required=True)
    parser.add_argument(
        "--full-tokenizer",
        type=Path,
        required=True,
        help="Full continuation used to find valid downstream compositions",
    )
    parser.add_argument(
        "--source-summary",
        type=Path,
        required=True,
        help="Existing noisy-token source summary whose ids must be reviewed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--descendant-limit", type=int, default=8)
    args = parser.parse_args()

    base = load_tokenizer(args.base_tokenizer)
    candidate = load_tokenizer(args.candidate_tokenizer)
    full = load_tokenizer(args.full_tokenizer)
    contract = validate_cutoff_contract(base, candidate)
    full_vocab = full["model"]["vocab"]
    full_by_id = {token_id: token for token, token_id in full_vocab.items()}
    candidate_vocab = candidate["model"]["vocab"]
    consumers = build_consumer_graph(full)

    reviews: list[dict[str, Any]] = []
    for token_id in load_reviewed_ids(args.source_summary):
        token = full_by_id.get(token_id)
        if token is None:
            reviews.append(
                {
                    "id": token_id,
                    "decision": "error",
                    "reason": "id absent from full tokenizer",
                }
            )
            continue
        raw = bytelevel_bytes(token)
        standalone = decode_bytes(raw)
        descendants = valid_descendants(
            token,
            consumers,
            max_depth=args.max_depth,
            limit=args.descendant_limit,
        )
        if standalone is not None:
            decision = "review_valid_utf8"
            reason = "token is standalone valid UTF-8; inspect content semantically"
        elif descendants:
            decision = "keep_structural_bytelevel"
            reason = (
                "not standalone UTF-8, but it is a live ByteLevel merge component "
                "with valid UTF-8 downstream compositions"
            )
        else:
            decision = "unresolved_partial"
            reason = (
                "not standalone UTF-8 and no valid downstream composition was "
                "found within the audit depth"
            )
        reviews.append(
            {
                "id": token_id,
                "in_candidate": token_id < len(candidate_vocab),
                "bytelevel_token": token,
                "raw_bytes_hex": raw.hex(" "),
                "standalone_utf8": standalone,
                "escaped": escaped_bytes(raw),
                "starts_with_continuation_byte": bool(
                    raw and 0x80 <= raw[0] <= 0xBF
                ),
                "decision": decision,
                "reason": reason,
                "valid_utf8_descendants": descendants,
            }
        )

    roundtrip = probe_roundtrip(args.candidate_tokenizer, reviews)
    in_candidate = [review for review in reviews if review.get("in_candidate")]
    unresolved = [
        review["id"]
        for review in in_candidate
        if review["decision"] == "unresolved_partial"
    ]
    errors = [
        review["id"] for review in reviews if review["decision"] == "error"
    ]
    status = (
        "passed"
        if contract["status"] == "passed"
        and roundtrip["status"] in {"passed", "skipped"}
        and not unresolved
        and not errors
        else "failed"
    )
    report = {
        "schema_version": "polytonic-cutoff-byte-audit-v1",
        "status": status,
        "base_tokenizer": {
            "path": str(args.base_tokenizer),
            "sha256": sha256_path(args.base_tokenizer),
        },
        "candidate_tokenizer": {
            "path": str(args.candidate_tokenizer),
            "sha256": sha256_path(args.candidate_tokenizer),
        },
        "full_tokenizer": {
            "path": str(args.full_tokenizer),
            "sha256": sha256_path(args.full_tokenizer),
        },
        "contract": contract,
        "reviewed_id_count": len(reviews),
        "reviewed_ids_in_candidate": [
            review["id"] for review in in_candidate
        ],
        "unresolved_ids_in_candidate": unresolved,
        "reviews": reviews,
        "roundtrip": roundtrip,
        "conclusion": (
            "All reviewed non-standalone tokens in this cutoff are structural "
            "ByteLevel merge pieces, not standalone mojibake or corpus text."
            if status == "passed"
            else "The cutoff has unresolved audit failures and must not be frozen."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "candidate_vocab_size": contract["candidate_vocab_size"],
                "reviewed_ids_in_candidate": report[
                    "reviewed_ids_in_candidate"
                ],
                "unresolved_ids_in_candidate": unresolved,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
