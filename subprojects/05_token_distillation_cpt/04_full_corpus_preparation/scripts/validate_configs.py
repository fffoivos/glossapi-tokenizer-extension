#!/usr/bin/env python3
"""Validate Phase-04 source, backlog and cleaning-policy contracts offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HEX40 = set("0123456789abcdef")
ROLES = {
    "additive_candidate",
    "base",
    "base_overlay",
    "overlap_candidate",
    "replacement_audit",
    "replacement_candidate",
}
STRUCTURAL_POLICIES = {
    "apply_after_review",
    "disabled",
    "shadow",
    "shadow_after_segmentation",
}
ELIGIBILITY = {
    "eligible_open",
    "noncommercial_review",
    "per_item_review",
    "policy_review",
}
BACKLOG_DISPOSITIONS = {
    "already_in_base",
    "empty_scaffold",
    "exclude",
    "needs_metadata",
    "optional_additive",
    "overlap_audit",
    "replacement_audit",
    "superseded",
    "unavailable_external",
}
BACKLOG_RELATIONSHIPS = {
    "additive",
    "external_only",
    "mixed_source_overlap",
    "not_in_base",
    "partial_base_overlap",
    "same_source_in_base",
    "same_source_replacement",
    "same_source_resegmented",
    "superseded_by_registered_source",
    "unknown",
}
LICENSE_STATUSES = {
    "noncommercial_review",
    "open_with_attribution",
    "policy_review",
    "unknown",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def require_revision(value: object, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or len(value) != 40 or any(c not in HEX40 for c in value):
        errors.append(f"{label}: revision must be a lowercase 40-hex commit SHA")


def validate_sources(cfg: dict) -> list[str]:
    errors: list[str] = []
    if cfg.get("schema_version") != "full_cpt_sources_v1":
        errors.append("sources: unsupported schema_version")

    base = cfg.get("base", {})
    overlap = cfg.get("apertus_overlap_overlay", {})
    tokenizer = cfg.get("tokenizer", {})
    require_revision(base.get("revision"), "base", errors)
    require_revision(overlap.get("revision"), "apertus_overlap_overlay", errors)
    require_revision(tokenizer.get("revision"), "tokenizer", errors)
    tok_hash = tokenizer.get("tokenizer_json_sha256")
    if not isinstance(tok_hash, str) or len(tok_hash) != 64 or any(c not in HEX40 for c in tok_hash):
        errors.append("tokenizer: tokenizer_json_sha256 must be lowercase 64-hex")
    tokenizer_path = tokenizer.get("tokenizer_json_path")
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        errors.append("tokenizer: tokenizer_json_path required")
    elif tokenizer_path not in tokenizer.get("include_globs", []):
        errors.append("tokenizer: tokenizer_json_path must be selected by include_globs")

    for label, entry in (("base", base), ("apertus_overlap_overlay", overlap)):
        include_globs = entry.get("include_globs")
        if not isinstance(include_globs, list) or not include_globs or not all(
            isinstance(value, str) and value for value in include_globs
        ):
            errors.append(f"{label}: include_globs must be a non-empty string list")

    sources = cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("sources: non-empty list required")
        return errors

    seen_ids: set[str] = set()
    seen_repos: dict[str, list[str]] = {}
    for index, source in enumerate(sources):
        label = f"sources[{index}]"
        if not isinstance(source, dict):
            errors.append(f"{label}: must be an object")
            continue
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"{label}: source_id required")
        elif source_id in seen_ids:
            errors.append(f"{label}: duplicate source_id {source_id!r}")
        else:
            seen_ids.add(source_id)
        repo_id = source.get("repo_id")
        if not isinstance(repo_id, str) or repo_id.count("/") != 1:
            errors.append(f"{label}: repo_id must be namespace/name")
        else:
            seen_repos.setdefault(repo_id, []).append(str(source_id))
        require_revision(source.get("revision"), label, errors)
        if source.get("role") not in ROLES:
            errors.append(f"{label}: invalid role {source.get('role')!r}")
        if source.get("structural_policy") not in STRUCTURAL_POLICIES:
            errors.append(f"{label}: invalid structural_policy {source.get('structural_policy')!r}")
        if source.get("training_eligibility") not in ELIGIBILITY:
            errors.append(f"{label}: invalid training_eligibility {source.get('training_eligibility')!r}")
        for field in ("include_globs", "text_columns", "id_columns"):
            value = source.get(field)
            if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
                errors.append(f"{label}: {field} must be a non-empty string list")
        alternate = source.get("alternate_text_columns")
        if alternate is not None and (
            not isinstance(alternate, list)
            or not alternate
            or not all(isinstance(value, str) and value for value in alternate)
        ):
            errors.append(f"{label}: alternate_text_columns must be a non-empty string list when set")
        if source.get("structural_policy") == "apply_after_review" and not str(
            source.get("cleaning_profile", "")
        ).startswith("academic"):
            errors.append(f"{label}: apply_after_review requires an academic cleaning profile")

    embedded = cfg.get("embedded_structural_routes")
    if not isinstance(embedded, list) or not embedded:
        errors.append("sources: embedded_structural_routes must be a non-empty list")
    else:
        for index, route in enumerate(embedded):
            label = f"embedded_structural_routes[{index}]"
            source_id = route.get("source_id") if isinstance(route, dict) else None
            if not isinstance(source_id, str) or not source_id:
                errors.append(f"{label}: source_id required")
            elif source_id in seen_ids:
                errors.append(f"{label}: source_id collides with top-level source {source_id!r}")
            else:
                seen_ids.add(source_id)
            if route.get("input_scope") != "canonical_mixed":
                errors.append(f"{label}: embedded route must use canonical_mixed")
            if route.get("acquisition_source_id") != "nanochat_base":
                errors.append(f"{label}: embedded route must bind to acquisition source nanochat_base")
            acquisition_globs = route.get("acquisition_include_globs")
            if not isinstance(acquisition_globs, list) or not acquisition_globs or not all(
                isinstance(item, str) and item and not Path(item).is_absolute() and ".." not in Path(item).parts
                for item in acquisition_globs
            ):
                errors.append(
                    f"{label}: acquisition_include_globs must be a non-empty list of safe relative globs"
                )
            if not isinstance(route.get("source_regex"), str) or not route.get("source_regex"):
                errors.append(f"{label}: non-empty source_regex required")
            for field in ("text_columns", "id_columns"):
                value = route.get(field)
                if not isinstance(value, list) or not value or not all(
                    isinstance(item, str) and item for item in value
                ):
                    errors.append(f"{label}: {field} must be a non-empty string list")
            if not isinstance(route.get("source_column"), str) or not route.get("source_column"):
                errors.append(f"{label}: source_column required")
            if route.get("structural_policy") not in STRUCTURAL_POLICIES - {"disabled"}:
                errors.append(f"{label}: invalid structural_policy")

    # A repeated repository is legal only when entries select distinct explicit files.
    for repo_id, ids in seen_repos.items():
        if len(ids) > 1:
            errors.append(f"sources: repo {repo_id!r} is listed more than once ({', '.join(ids)})")
    return errors


def validate_backlog(cfg: dict, sources_cfg: dict | None = None) -> list[str]:
    """Validate metadata-only candidates that must not be acquired yet."""

    errors: list[str] = []
    if cfg.get("schema_version") != "full_cpt_source_backlog_v1":
        errors.append("backlog: unsupported schema_version")
    if not isinstance(cfg.get("audited_at"), str) or not cfg.get("audited_at"):
        errors.append("backlog: audited_at required")

    base = cfg.get("base", {})
    if base.get("repo_id") != "fffoivos/glossapi-greek-nanochat-pretraining-dataset":
        errors.append("backlog: unexpected base repo_id")
    require_revision(base.get("revision"), "backlog.base", errors)
    if sources_cfg is not None and base.get("revision") != sources_cfg.get("base", {}).get("revision"):
        errors.append("backlog: base revision must match sources.json")

    entries = cfg.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("backlog: non-empty entries list required")
        return errors

    acquired_repos = {
        source.get("repo_id")
        for source in (sources_cfg or {}).get("sources", [])
        if isinstance(source, dict)
    }
    seen_ids: set[str] = set()
    seen_repos: set[str] = set()
    for index, entry in enumerate(entries):
        label = f"backlog.entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: must be an object")
            continue

        backlog_id = entry.get("backlog_id")
        if not isinstance(backlog_id, str) or not backlog_id:
            errors.append(f"{label}: backlog_id required")
        elif backlog_id in seen_ids:
            errors.append(f"{label}: duplicate backlog_id {backlog_id!r}")
        else:
            seen_ids.add(backlog_id)

        repo_id = entry.get("repo_id")
        if not isinstance(repo_id, str) or repo_id.count("/") != 1:
            errors.append(f"{label}: repo_id must be namespace/name")
        elif repo_id in seen_repos:
            errors.append(f"{label}: duplicate repo_id {repo_id!r}")
        else:
            seen_repos.add(repo_id)
            if repo_id in acquired_repos:
                errors.append(f"{label}: repo is already present in sources.json")

        require_revision(entry.get("revision"), label, errors)
        if entry.get("acquisition_eligible") is not False:
            errors.append(f"{label}: acquisition_eligible must remain false")
        if entry.get("disposition") not in BACKLOG_DISPOSITIONS:
            errors.append(f"{label}: invalid disposition {entry.get('disposition')!r}")
        if entry.get("relationship_to_base") not in BACKLOG_RELATIONSHIPS:
            errors.append(f"{label}: invalid relationship_to_base")

        license_info = entry.get("license")
        if not isinstance(license_info, dict):
            errors.append(f"{label}: license must be an object")
        else:
            declared = license_info.get("declared")
            if declared is not None and (not isinstance(declared, str) or not declared):
                errors.append(f"{label}: license.declared must be null or a non-empty string")
            if license_info.get("status") not in LICENSE_STATUSES:
                errors.append(f"{label}: invalid license.status")

        route = entry.get("route")
        if not isinstance(route, dict):
            errors.append(f"{label}: route must be an object")
        else:
            if not isinstance(route.get("cleaning_profile"), str) or not route.get("cleaning_profile"):
                errors.append(f"{label}: route.cleaning_profile required")
            if route.get("structural_policy") not in STRUCTURAL_POLICIES:
                errors.append(f"{label}: invalid route.structural_policy")
            for field in ("text_columns", "id_columns"):
                value = route.get(field)
                if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
                    errors.append(f"{label}: route.{field} must be a string list")

        files = entry.get("candidate_files")
        if not isinstance(files, list):
            errors.append(f"{label}: candidate_files must be a list")
        else:
            for file_index, candidate in enumerate(files):
                file_label = f"{label}.candidate_files[{file_index}]"
                if not isinstance(candidate, dict):
                    errors.append(f"{file_label}: must be an object")
                    continue
                if not isinstance(candidate.get("path"), str) or not candidate.get("path"):
                    errors.append(f"{file_label}: path required")
                size = candidate.get("bytes")
                if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
                    errors.append(f"{file_label}: bytes must be null or a non-negative integer")
                schema = candidate.get("schema")
                if not isinstance(schema, list) or not all(
                    isinstance(column, str) and column for column in schema
                ):
                    errors.append(f"{file_label}: schema must be a string list")

        metrics = entry.get("known_metrics")
        if not isinstance(metrics, dict):
            errors.append(f"{label}: known_metrics must be an object")
        else:
            for field in ("repository_rows", "card_rows", "repository_bytes", "reported_tokens"):
                value = metrics.get(field)
                if value is not None and (
                    not isinstance(value, int) or isinstance(value, bool) or value < 0
                ):
                    errors.append(f"{label}: known_metrics.{field} must be null or non-negative")
            tokenizer = metrics.get("tokenizer")
            if metrics.get("reported_tokens") is not None and (
                not isinstance(tokenizer, str) or not tokenizer
            ):
                errors.append(f"{label}: known_metrics.tokenizer required with reported_tokens")
            if files and all(candidate.get("bytes") is not None for candidate in files):
                selected_bytes = sum(int(candidate["bytes"]) for candidate in files)
                if metrics.get("repository_bytes") != selected_bytes:
                    errors.append(
                        f"{label}: known_metrics.repository_bytes must equal candidate file bytes "
                        f"({selected_bytes})"
                    )

        blockers = entry.get("blockers")
        if not isinstance(blockers, list) or not blockers or not all(
            isinstance(blocker, str) and blocker for blocker in blockers
        ):
            errors.append(f"{label}: blockers must be a non-empty string list")
    return errors


def validate_policy(cfg: dict) -> list[str]:
    errors: list[str] = []
    if cfg.get("schema_version") != "full_cpt_cleaning_policy_v1":
        errors.append("policy: unsupported schema_version")
    if cfg.get("status") not in {"audit_only", "approved"}:
        errors.append("policy: status must be audit_only or approved")
    structural = cfg.get("structural", {})
    if structural.get("replacement") != "two_newlines_max":
        errors.append("policy: structural replacement must remain two_newlines_max")
    for kind in ("bibliography", "toc"):
        part = structural.get(kind, {})
        if not isinstance(part.get("enabled_for_materialization"), bool):
            errors.append(f"policy: {kind}.enabled_for_materialization must be boolean")
        floor = part.get("prose_protection_floor")
        if not isinstance(floor, (float, int)) or not 0.99 <= float(floor) <= 1.0:
            errors.append(f"policy: {kind}.prose_protection_floor must be in [0.99, 1.0]")
    if cfg.get("status") != "approved" and any(
        structural.get(kind, {}).get("enabled_for_materialization") for kind in ("bibliography", "toc")
    ):
        errors.append("policy: destructive structural materialization cannot be enabled before approval")
    validation = cfg.get("validation", {})
    gold_hash = validation.get("structural_gold_sha256")
    if gold_hash is not None and (
        not isinstance(gold_hash, str) or len(gold_hash) != 64 or any(c not in HEX40 for c in gold_hash)
    ):
        errors.append("policy: validation.structural_gold_sha256 must be null or lowercase 64-hex")
    if cfg.get("status") == "approved" and gold_hash is None:
        errors.append("policy: approved status requires a pinned structural gold SHA-256")
    if cfg.get("diavgeia", {}).get("academic_structural_cleaner") != "disabled":
        errors.append("policy: Diavgeia academic structural cleaner must be disabled")
    return errors


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=here / "configs" / "sources.json")
    parser.add_argument("--backlog", type=Path, default=here / "configs" / "source_backlog.json")
    parser.add_argument("--policy", type=Path, default=here / "configs" / "cleaning_policy.json")
    parser.add_argument("--json", action="store_true", help="print machine-readable result")
    args = parser.parse_args()

    source_cfg = load_json(args.sources)
    backlog_cfg = load_json(args.backlog)
    policy_cfg = load_json(args.policy)
    errors = (
        validate_sources(source_cfg)
        + validate_backlog(backlog_cfg, source_cfg)
        + validate_policy(policy_cfg)
    )
    result = {
        "ok": not errors,
        "errors": errors,
        "sources": len(source_cfg.get("sources", [])),
        "backlog_entries": len(backlog_cfg.get("entries", [])),
        "sources_sha256": digest(args.sources),
        "backlog_sha256": digest(args.backlog),
        "policy_sha256": digest(args.policy),
        "policy_status": policy_cfg.get("status"),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif errors:
        for error in errors:
            print(f"ERROR: {error}")
    else:
        print(
            f"OK: {result['sources']} sources; backlog={result['backlog_entries']}; "
            f"policy={result['policy_status']}; sources_sha256={result['sources_sha256']}; "
            f"backlog_sha256={result['backlog_sha256']}; policy_sha256={result['policy_sha256']}"
        )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
