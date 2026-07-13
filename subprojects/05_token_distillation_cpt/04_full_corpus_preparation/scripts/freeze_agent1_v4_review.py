#!/usr/bin/env python3
"""Freeze the receipt-bound inputs for Agent 1's isolated v4 raw-review lane.

This is deliberately a metadata-only operation.  It validates the passed
acquisition receipt and pins every review input before any sampled raw document
is materialized or sent to Codex.  It never changes source admission: a source
being cleared for the user-approved exact-raw *review* is not a permission to
train on, redistribute, or clean that source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent1_v4_raw_review import (  # noqa: E402
    ALLOWED_ROUTES,
    GIT_COMMIT_RE,
    HEX_SHA256_RE,
    POLICY_SCHEMA,
    file_binding,
    load_policy,
    load_roster,
    read_json_object,
    sha256_json,
    write_json_no_replace,
)
from full_corpus_io import artifacts_from_receipt  # noqa: E402


FREEZE_SCHEMA = "agent1_v4_raw_review_freeze_receipt_v1"
LICENSE_SCHEMA = "full_cpt_source_license_adjudication_v1"
NANOCHAT_ROSTER_SCHEMA = "nanochat_initial_roster_v1"
GREEKMMLU_ID = "greekmmlu"
EXPECTED_MODEL = "gpt-5.6-terra"


def _require_regular(path: Path, label: str) -> dict[str, object]:
    try:
        return file_binding(path)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{label}: {exc}") from exc


def _require_commit(value: str, label: str) -> str:
    if not GIT_COMMIT_RE.fullmatch(value):
        raise ValueError(f"{label} must be a 40-character lowercase Git commit")
    return value


def _require_seed(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("sampling seed file must be a regular file")
    if path.stat().st_mode & 0o077:
        raise ValueError("sampling seed file must not be group- or world-readable")
    seed = path.read_text(encoding="utf-8").strip()
    if not HEX_SHA256_RE.fullmatch(seed) or len(bytes.fromhex(seed)) != 32:
        raise ValueError("sampling seed file must contain one 32-byte lowercase hex seed")
    return seed, hashlib.sha256(seed.encode("ascii")).hexdigest()


def _license_rows(
    *,
    license_path: Path,
    sources_path: Path,
    selected_sources: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    value = read_json_object(license_path)
    if value.get("schema_version") != LICENSE_SCHEMA:
        raise ValueError("unsupported source-license adjudication schema")
    source_registry = value.get("source_registry")
    if not isinstance(source_registry, Mapping):
        raise ValueError("source-license adjudication lacks source_registry binding")
    sources_binding = _require_regular(sources_path, "sources registry")
    if source_registry.get("sha256") != sources_binding["sha256"]:
        raise ValueError("source-license adjudication is not bound to current sources.json")
    rows = value.get("sources")
    if not isinstance(rows, list):
        raise ValueError("source-license adjudication lacks source rows")
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        source_id = str(row.get("source_id") or "")
        if source_id in selected_sources:
            if source_id in result:
                raise ValueError(f"duplicate source-license row: {source_id}")
            result[source_id] = dict(row)
    missing = sorted(set(selected_sources) - set(result))
    if missing:
        raise ValueError(f"source-license adjudication lacks selected sources: {missing}")
    for source_id, source in selected_sources.items():
        assert hasattr(source, "repo_id")
        row = result[source_id]
        if row.get("repo_id") != source.repo_id or row.get("revision") != source.revision:
            raise ValueError(f"source-license identity drift: {source_id}")
        if row.get("registry_training_eligibility") != source.config.get("training_eligibility"):
            raise ValueError(f"source-license training eligibility drift: {source_id}")
        local_training = row.get("local_training")
        redistribution = row.get("redistribution")
        if not isinstance(local_training, Mapping) or not isinstance(local_training.get("eligible"), bool):
            raise ValueError(f"source-license local_training decision missing: {source_id}")
        if not isinstance(redistribution, Mapping) or not isinstance(redistribution.get("eligible"), bool):
            raise ValueError(f"source-license redistribution decision missing: {source_id}")
    return result


def _greekmmlu_pin(path: Path) -> dict[str, object]:
    value = read_json_object(path)
    rows = value.get("benchmarks")
    if not isinstance(rows, list):
        raise ValueError("GreekMMLU registry has no benchmarks list")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("id") == GREEKMMLU_ID]
    if len(matches) != 1:
        raise ValueError("GreekMMLU registry must contain exactly one greekmmlu row")
    row = dict(matches[0])
    if row.get("source") != "dascim/GreekMMLU" or row.get("config") != "All" or row.get("split") != "test":
        raise ValueError("GreekMMLU registry identity/config/split drift")
    _require_commit(str(row.get("revision") or ""), "GreekMMLU revision")
    return row


def _source_freeze_row(source: object, license_row: Mapping[str, object]) -> dict[str, object]:
    # SourceArtifact is intentionally duck-typed here to keep the function
    # useful for a no-data unit fixture without serializing raw text.
    source_id = str(getattr(source, "source_id"))
    config = getattr(source, "config")
    local_training = license_row["local_training"]
    redistribution = license_row["redistribution"]
    assert isinstance(local_training, Mapping) and isinstance(redistribution, Mapping)
    return {
        "source_id": source_id,
        "repo_id": str(getattr(source, "repo_id")),
        "revision": str(getattr(source, "revision")),
        "role": str(getattr(source, "role")),
        "source_family_id": str(getattr(source, "source_family_id")),
        "provisional_text_columns": list(config.get("text_columns", [])),
        "provisional_id_columns": list(config.get("id_columns", [])),
        "training_eligibility_category": config.get("training_eligibility"),
        "license": {
            "declared_license": license_row.get("declared_license"),
            "local_training_eligible": local_training["eligible"],
            "local_training_status": local_training.get("status"),
            "redistribution_eligible": redistribution["eligible"],
            "redistribution_status": redistribution.get("status"),
        },
        "exact_raw_external_review": {
            "status": "cleared_by_explicit_user_instruction",
            "review_transport": "exact_raw_user_approved",
            "scope": "one isolated Terra review for each of 20 receipt-bound samples",
            "does_not_change_training_or_redistribution_admission": True,
        },
        "files": [dict(binding) for binding in getattr(source, "file_bindings")],
    }


def freeze_review_inputs(
    *,
    sources_path: Path,
    acquisition_receipt: Path,
    roster_path: Path,
    policy_path: Path,
    license_adjudication_path: Path,
    nanochat_roster_path: Path,
    environment_lock_path: Path,
    greekmmlu_registry_path: Path,
    prompt_path: Path,
    response_schema_path: Path,
    sampling_seed_path: Path,
    code_commit: str,
    glossapi_commit: str,
    output: Path,
) -> dict[str, object]:
    """Write a complete, immutable Stage-00 receipt without scanning text."""

    policy = load_policy(policy_path)
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("raw-review policy schema drift")
    _require_commit(code_commit, "code_commit")
    _require_commit(glossapi_commit, "glossapi_commit")
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"freeze receipt already exists: {output}")
    seed, seed_sha256 = _require_seed(sampling_seed_path)
    del seed  # The secret remains in its 0600 sidecar; only its digest is receipted.
    source_ids = list(policy["source_ids"])
    selected = set(source_ids)
    sources = artifacts_from_receipt(sources_path, acquisition_receipt, selected)
    if [source.source_id for source in sources] != sorted(source_ids):
        raise ValueError("acquisition receipt/source-policy closure drift")
    routes = load_roster(roster_path, source_ids)
    if set(routes) != selected or any(route not in ALLOWED_ROUTES for row in routes.values() for route in row.values()):
        raise ValueError("source-route closure drift")
    source_by_id = {source.source_id: source for source in sources}
    license_rows = _license_rows(
        license_path=license_adjudication_path,
        sources_path=sources_path,
        selected_sources=source_by_id,
    )
    nanochat = read_json_object(nanochat_roster_path)
    if nanochat.get("schema_version") != NANOCHAT_ROSTER_SCHEMA:
        raise ValueError("unsupported Nanochat roster schema")
    repository = nanochat.get("repository")
    if not isinstance(repository, Mapping) or repository.get("repo_id") != "fffoivos/glossapi-greek-nanochat-pretraining-dataset":
        raise ValueError("Nanochat roster repository drift")
    _require_commit(str(repository.get("first_data_revision") or ""), "Nanochat first-data revision")
    registry = read_json_object(sources_path)
    overlay = registry.get("apertus_overlap_overlay")
    tokenizer = registry.get("tokenizer")
    if not isinstance(overlay, Mapping) or not isinstance(tokenizer, Mapping):
        raise ValueError("sources registry lacks Apertus overlay or tokenizer pin")
    _require_commit(str(overlay.get("revision") or ""), "Apertus overlay revision")
    _require_commit(str(tokenizer.get("revision") or ""), "tokenizer revision")
    greekmmlu = _greekmmlu_pin(greekmmlu_registry_path)
    receipt: dict[str, object] = {
        "schema_version": FREEZE_SCHEMA,
        "status": "passed",
        "review_scope": {
            "source_ids": source_ids,
            "excluded_source_ids": list(policy["excluded_source_ids"]),
            "documents_per_source": policy["documents_per_source"],
            "logical_review_count": len(source_ids) * int(policy["documents_per_source"]),
            "approval": policy["approval"],
            "approval_sha256": sha256_json(policy["approval"]),
        },
        "review_contract": {
            "transport": policy["review_transport"],
            "model": EXPECTED_MODEL,
            "reasoning_effort": policy["reasoning_effort"],
            "max_attempts_per_document": policy["max_attempts_per_document"],
            "sampling_seed_sha256": seed_sha256,
        },
        "inputs": {
            "sources": _require_regular(sources_path, "sources registry"),
            "acquisition_receipt": _require_regular(acquisition_receipt, "acquisition receipt"),
            "candidate_roster": _require_regular(roster_path, "candidate roster"),
            "scope_policy": _require_regular(policy_path, "scope policy"),
            "source_license_adjudication": _require_regular(license_adjudication_path, "source-license adjudication"),
            "nanochat_initial_roster": _require_regular(nanochat_roster_path, "Nanochat roster"),
            "environment_lock": _require_regular(environment_lock_path, "environment lock"),
            "greekmmlu_registry": _require_regular(greekmmlu_registry_path, "GreekMMLU registry"),
            "prompt": _require_regular(prompt_path, "Terra prompt"),
            "response_schema": _require_regular(response_schema_path, "Terra response schema"),
        },
        "implementation": {
            "code_commit": code_commit,
            "glossapi_commit": glossapi_commit,
            "nanochat": dict(repository),
            "apertus_overlap_overlay": dict(overlay),
            "tokenizer": dict(tokenizer),
            "greekmmlu": greekmmlu,
        },
        "sources": [
            _source_freeze_row(source_by_id[source_id], license_rows[source_id])
            for source_id in source_ids
        ],
        "route_declarations": routes,
    }
    write_json_no_replace(output, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    repo_root = root.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=root / "configs" / "sources.json")
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--roster", type=Path, default=root / "configs" / "agent1_v3_candidate_roster.json")
    parser.add_argument("--policy", type=Path, default=root / "configs" / "agent1_v4_raw_review_policy.json")
    parser.add_argument("--license-adjudication", type=Path, default=root / "configs" / "source_license_adjudication.json")
    parser.add_argument("--nanochat-roster", type=Path, default=root / "configs" / "nanochat_initial_roster.json")
    parser.add_argument("--environment-lock", type=Path, default=root / "requirements-runtime.txt")
    parser.add_argument(
        "--greekmmlu-registry",
        type=Path,
        default=repo_root / "subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/native_greek_benchmark_registry.json",
    )
    parser.add_argument("--prompt", type=Path, default=root / "configs" / "agent1_v4_terra_review_prompt.md")
    parser.add_argument("--response-schema", type=Path, default=root / "schemas" / "agent1_v4_terra_review_response.schema.json")
    parser.add_argument("--sampling-seed-file", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--glossapi-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = freeze_review_inputs(
        sources_path=args.sources,
        acquisition_receipt=args.acquisition_receipt,
        roster_path=args.roster,
        policy_path=args.policy,
        license_adjudication_path=args.license_adjudication,
        nanochat_roster_path=args.nanochat_roster,
        environment_lock_path=args.environment_lock,
        greekmmlu_registry_path=args.greekmmlu_registry,
        prompt_path=args.prompt,
        response_schema_path=args.response_schema,
        sampling_seed_path=args.sampling_seed_file,
        code_commit=args.code_commit,
        glossapi_commit=args.glossapi_commit,
        output=args.output,
    )
    print(json.dumps({"ok": True, "logical_review_count": receipt["review_scope"]["logical_review_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
