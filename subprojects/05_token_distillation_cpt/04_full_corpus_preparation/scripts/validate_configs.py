#!/usr/bin/env python3
"""Validate Phase-04 source, backlog and cleaning-policy contracts offline."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
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
ALIAS_KINDS = {"direct", "hybrid", "replacement"}
ALIAS_CONFIDENCE = {"high", "medium", "low"}
NANOCHAT_FIRST_DATA_REVISION = "500b8bf577e1e70f4902b77edce2cda02a2559cb"
NANOCHAT_EMPTY_INITIAL_REVISION = "1da2730c4e8aa4cd42c3662fbd7c64dc883d7dfe"
NANOCHAT_INITIAL_ROW_COUNTS_SHA256 = (
    "4f55caa8e709777af644958d8deb19d26a7adc64d52d402c837dba5cbc0dac8d"
)
NANOCHAT_INITIAL_SOURCE_COUNT = 18
NANOCHAT_INITIAL_TOTAL_ROWS = 717_265
NANOCHAT_LATER_SOURCE_NAMES = {
    "HPLT/ell_Grek_ge8_no_mt_clean60",
    "OPUS/OpenSubtitles-el-v2018",
}
POST_DECEMBER_CUTOFF = "2026-01-01T00:00:00Z"
POST_DECEMBER_REPOSITORY_COUNT = 25
CURRENT_GLOSSAPI_REPOSITORY_COUNT = 40
REQUIRED_PROVENANCE_FIELDS = {
    "lineage_alias_id",
    "normalized_text_sha256",
    "original_text_sha256",
    "representation_generation",
    "source_artifact_path",
    "source_dataset",
    "source_doc_id",
    "source_family_id",
    "source_repo_id",
    "source_revision",
    "source_row_id",
    "source_text_field",
    "stable_uid",
    "work_key",
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


def is_safe_relative(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not Path(value).is_absolute()
        and ".." not in Path(value).parts
    )


def validate_initial_roster(cfg: dict) -> list[str]:
    """Validate the immutable source-name anchor at Nanochat's first data commit."""

    errors: list[str] = []
    if cfg.get("schema_version") != "nanochat_initial_roster_v1":
        errors.append("nanochat_roster: unsupported schema_version")
    if cfg.get("authority") != "first_data_commit_source_dataset_roster":
        errors.append("nanochat_roster: invalid authority")

    repository = cfg.get("repository")
    if not isinstance(repository, dict):
        errors.append("nanochat_roster: repository must be an object")
        repository = {}
    if repository.get("repo_id") != "fffoivos/glossapi-greek-nanochat-pretraining-dataset":
        errors.append("nanochat_roster: unexpected repository.repo_id")
    require_revision(
        repository.get("empty_initial_revision"), "nanochat_roster.empty_initial_revision", errors
    )
    require_revision(
        repository.get("first_data_revision"), "nanochat_roster.first_data_revision", errors
    )
    if repository.get("empty_initial_revision") != NANOCHAT_EMPTY_INITIAL_REVISION:
        errors.append("nanochat_roster: empty_initial_revision does not match HF history")
    if repository.get("first_data_revision") != NANOCHAT_FIRST_DATA_REVISION:
        errors.append("nanochat_roster: first_data_revision does not match HF history")
    committed_at = repository.get("first_data_committed_at")
    if not isinstance(committed_at, str) or not committed_at.endswith("Z"):
        errors.append("nanochat_roster: first_data_committed_at must be a UTC timestamp")

    row_counts = cfg.get("row_counts")
    if not isinstance(row_counts, dict):
        errors.append("nanochat_roster: row_counts must be an object")
        row_counts = {}
    if not is_safe_relative(row_counts.get("path")):
        errors.append("nanochat_roster: row_counts.path must be a safe relative path")
    row_counts_hash = row_counts.get("sha256")
    if (
        not isinstance(row_counts_hash, str)
        or len(row_counts_hash) != 64
        or any(char not in HEX40 for char in row_counts_hash)
    ):
        errors.append("nanochat_roster: row_counts.sha256 must be lowercase 64-hex")
    elif row_counts_hash != NANOCHAT_INITIAL_ROW_COUNTS_SHA256:
        errors.append("nanochat_roster: row_counts.sha256 does not match the pinned artifact")

    sources = cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("nanochat_roster: sources must be a non-empty list")
        sources = []
    seen_names: set[str] = set()
    seen_artifacts: set[str] = set()
    total_rows = 0
    for index, source in enumerate(sources):
        label = f"nanochat_roster.sources[{index}]"
        if not isinstance(source, dict):
            errors.append(f"{label}: must be an object")
            continue
        source_name = source.get("source_dataset")
        if not isinstance(source_name, str) or not source_name:
            errors.append(f"{label}: source_dataset required")
        elif source_name in seen_names:
            errors.append(f"{label}: duplicate source_dataset {source_name!r}")
        else:
            seen_names.add(source_name)
        rows = source.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            errors.append(f"{label}: rows must be a non-negative integer")
        else:
            total_rows += rows
        file_count = source.get("file_count")
        if not isinstance(file_count, int) or isinstance(file_count, bool) or file_count <= 0:
            errors.append(f"{label}: file_count must be a positive integer")
        artifacts = source.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"{label}: artifacts must be a non-empty list")
            artifacts = []
        elif isinstance(file_count, int) and not isinstance(file_count, bool) and len(artifacts) != file_count:
            errors.append(f"{label}: artifacts count must equal file_count")
        for artifact in artifacts:
            if not is_safe_relative(artifact):
                errors.append(f"{label}: artifact paths must be safe and relative")
            elif artifact in seen_artifacts:
                errors.append(f"{label}: duplicate artifact path {artifact!r}")
            else:
                seen_artifacts.add(artifact)

    if row_counts.get("source_count") != NANOCHAT_INITIAL_SOURCE_COUNT:
        errors.append("nanochat_roster: row_counts.source_count must remain 18")
    if len(sources) != row_counts.get("source_count"):
        errors.append("nanochat_roster: source list length must equal row_counts.source_count")
    if row_counts.get("total_rows") != NANOCHAT_INITIAL_TOTAL_ROWS:
        errors.append("nanochat_roster: row_counts.total_rows must remain 717265")
    if total_rows != row_counts.get("total_rows"):
        errors.append("nanochat_roster: source rows must sum to row_counts.total_rows")

    additions = cfg.get("later_source_name_additions")
    if not isinstance(additions, list):
        errors.append("nanochat_roster: later_source_name_additions must be a list")
        additions = []
    seen_additions: set[str] = set()
    for index, addition in enumerate(additions):
        label = f"nanochat_roster.later_source_name_additions[{index}]"
        if not isinstance(addition, dict):
            errors.append(f"{label}: must be an object")
            continue
        source_name = addition.get("source_dataset")
        if not isinstance(source_name, str) or not source_name:
            errors.append(f"{label}: source_dataset required")
        elif source_name in seen_names:
            errors.append(f"{label}: source_dataset was already present in the initial roster")
        elif source_name in seen_additions:
            errors.append(f"{label}: duplicate later source_dataset {source_name!r}")
        else:
            seen_additions.add(source_name)
        require_revision(addition.get("first_roster_revision"), label, errors)
        timestamp = addition.get("committed_at")
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            errors.append(f"{label}: committed_at must be a UTC timestamp")
        if not is_safe_relative(addition.get("row_counts_path")):
            errors.append(f"{label}: row_counts_path must be a safe relative path")
        for field in ("rows_at_addition", "file_count_at_addition"):
            value = addition.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"{label}: {field} must be a positive integer")
        globs = addition.get("artifact_globs")
        if not isinstance(globs, list) or not globs or not all(is_safe_relative(item) for item in globs):
            errors.append(f"{label}: artifact_globs must be a non-empty safe relative list")
    if seen_additions != NANOCHAT_LATER_SOURCE_NAMES:
        errors.append("nanochat_roster: later source-name additions must be exactly OPUS and HPLT")
    return errors


def validate_lineage_aliases(cfg: dict, roster_cfg: dict) -> list[str]:
    """Validate reviewed name aliases without implying snapshot equivalence."""

    errors: list[str] = []
    if cfg.get("schema_version") != "source_lineage_aliases_v1":
        errors.append("lineage_aliases: unsupported schema_version")
    if not isinstance(cfg.get("reviewed_at"), str) or not cfg.get("reviewed_at"):
        errors.append("lineage_aliases: reviewed_at required")
    if cfg.get("anchor_revision") != roster_cfg.get("repository", {}).get("first_data_revision"):
        errors.append("lineage_aliases: anchor_revision must match the initial roster")
    if cfg.get("authority") != "lineage_only_not_snapshot_equivalence":
        errors.append("lineage_aliases: invalid authority")
    definitions = cfg.get("alias_kind_definitions")
    if not isinstance(definitions, dict) or set(definitions) != ALIAS_KINDS or not all(
        isinstance(value, str) and value for value in definitions.values()
    ):
        errors.append("lineage_aliases: alias_kind_definitions must define direct, replacement and hybrid")

    initial_names = {
        source.get("source_dataset")
        for source in roster_cfg.get("sources", [])
        if isinstance(source, dict) and isinstance(source.get("source_dataset"), str)
    }
    aliases = cfg.get("aliases")
    if not isinstance(aliases, list) or not aliases:
        errors.append("lineage_aliases: aliases must be a non-empty list")
        aliases = []
    seen_ids: set[str] = set()
    mapped_names: set[str] = set()
    seen_kinds: set[str] = set()
    for index, alias in enumerate(aliases):
        label = f"lineage_aliases.aliases[{index}]"
        if not isinstance(alias, dict):
            errors.append(f"{label}: must be an object")
            continue
        alias_id = alias.get("alias_id")
        if not isinstance(alias_id, str) or not alias_id:
            errors.append(f"{label}: alias_id required")
        elif alias_id in seen_ids:
            errors.append(f"{label}: duplicate alias_id {alias_id!r}")
        else:
            seen_ids.add(alias_id)
        repo_id = alias.get("current_repo_id")
        if not isinstance(repo_id, str) or repo_id.count("/") != 1:
            errors.append(f"{label}: current_repo_id must be namespace/name")
        require_revision(alias.get("reviewed_revision"), label, errors)
        source_names = alias.get("initial_source_datasets")
        if not isinstance(source_names, list) or not source_names or not all(
            isinstance(value, str) and value for value in source_names
        ):
            errors.append(f"{label}: initial_source_datasets must be a non-empty string list")
            source_names = []
        elif len(set(source_names)) != len(source_names):
            errors.append(f"{label}: initial_source_datasets contains duplicates")
        for source_name in source_names:
            if source_name not in initial_names:
                errors.append(f"{label}: unknown initial source_dataset {source_name!r}")
            else:
                mapped_names.add(source_name)
        alias_kind = alias.get("alias_kind")
        if alias_kind not in ALIAS_KINDS:
            errors.append(f"{label}: invalid alias_kind {alias_kind!r}")
        else:
            seen_kinds.add(alias_kind)
        if alias_kind in {"direct", "replacement"} and len(source_names) != 1:
            errors.append(f"{label}: direct/replacement aliases must map exactly one initial source")
        if alias.get("confidence") not in ALIAS_CONFIDENCE:
            errors.append(f"{label}: invalid confidence")
        globs = alias.get("artifact_globs")
        if not isinstance(globs, list) or not globs or not all(is_safe_relative(item) for item in globs):
            errors.append(f"{label}: artifact_globs must be a non-empty safe relative list")
        if alias.get("snapshot_equivalence") != "unproven":
            errors.append(f"{label}: snapshot_equivalence must remain unproven")
        if alias.get("requires_document_key_audit") is not True:
            errors.append(f"{label}: requires_document_key_audit must be true")
        evidence = alias.get("evidence")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(item, str) and item for item in evidence
        ):
            errors.append(f"{label}: evidence must be a non-empty string list")
    if not {"replacement", "hybrid"}.issubset(seen_kinds):
        errors.append("lineage_aliases: reviewed aliases must distinguish replacement and hybrid cases")

    unaliased = cfg.get("unaliased_initial_sources")
    if not isinstance(unaliased, list):
        errors.append("lineage_aliases: unaliased_initial_sources must be a list")
        unaliased = []
    unaliased_names: set[str] = set()
    for index, entry in enumerate(unaliased):
        label = f"lineage_aliases.unaliased_initial_sources[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: must be an object")
            continue
        source_name = entry.get("source_dataset")
        if source_name not in initial_names:
            errors.append(f"{label}: unknown initial source_dataset {source_name!r}")
        elif source_name in unaliased_names:
            errors.append(f"{label}: duplicate source_dataset {source_name!r}")
        else:
            unaliased_names.add(source_name)
        if not isinstance(entry.get("reason"), str) or not entry.get("reason"):
            errors.append(f"{label}: reason required")
    overlap = mapped_names & unaliased_names
    if overlap:
        errors.append(f"lineage_aliases: sources cannot be both mapped and unaliased: {sorted(overlap)}")
    missing = initial_names - mapped_names - unaliased_names
    if missing:
        errors.append(f"lineage_aliases: initial sources lack alias review: {sorted(missing)}")
    return errors


def validate_alias_registry_revisions(
    aliases_cfg: dict, sources_cfg: dict, backlog_cfg: dict
) -> list[str]:
    """Keep each reviewed alias pinned to the same current revision as the registry."""

    errors: list[str] = []
    revisions = {
        entry.get("repo_id"): entry.get("revision")
        for entry in [
            *sources_cfg.get("sources", []),
            *backlog_cfg.get("entries", []),
        ]
        if isinstance(entry, dict)
    }
    for index, alias in enumerate(aliases_cfg.get("aliases", [])):
        if not isinstance(alias, dict):
            continue
        repo_id = alias.get("current_repo_id")
        if repo_id not in revisions:
            errors.append(f"lineage_aliases.aliases[{index}]: repo missing from sources/backlog")
        elif alias.get("reviewed_revision") != revisions[repo_id]:
            errors.append(f"lineage_aliases.aliases[{index}]: reviewed revision drift")
    return errors


def nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_post_december_inventory(
    cfg: dict,
    roster_cfg: dict,
    sources_cfg: dict,
    backlog_cfg: dict,
) -> list[str]:
    """Validate the static HF inventory and its source-lineage arithmetic."""

    errors: list[str] = []
    if cfg.get("schema_version") != "post_december_glossapi_inventory_v1":
        errors.append("post_december_inventory: unsupported schema_version")
    if not isinstance(cfg.get("audited_at"), str) or not cfg.get("audited_at"):
        errors.append("post_december_inventory: audited_at required")

    cutoff = cfg.get("cutoff")
    if not isinstance(cutoff, dict):
        errors.append("post_december_inventory: cutoff must be an object")
        cutoff = {}
    if cutoff.get("timestamp") != POST_DECEMBER_CUTOFF:
        errors.append("post_december_inventory: cutoff must remain 2026-01-01T00:00:00Z")

    reference = cfg.get("first_nanochat_reference")
    if not isinstance(reference, dict):
        errors.append("post_december_inventory: first_nanochat_reference must be an object")
        reference = {}
    roster_repository = roster_cfg.get("repository", {})
    roster_counts = roster_cfg.get("row_counts", {})
    if reference.get("repo_id") != roster_repository.get("repo_id"):
        errors.append("post_december_inventory: Nanochat repo_id must match the roster")
    if reference.get("revision") != roster_repository.get("first_data_revision"):
        errors.append("post_december_inventory: Nanochat revision must match the first data commit")
    if reference.get("row_count") != roster_counts.get("total_rows"):
        errors.append("post_december_inventory: Nanochat row count must match the roster")
    if reference.get("source_dataset_count") != roster_counts.get("source_count"):
        errors.append("post_december_inventory: Nanochat source count must match the roster")
    expected_roster = {
        source.get("source_dataset"): source.get("rows")
        for source in roster_cfg.get("sources", [])
        if isinstance(source, dict)
    }
    observed_roster = {
        source.get("source_dataset"): source.get("rows")
        for source in reference.get("source_roster", [])
        if isinstance(source, dict)
    }
    if observed_roster != expected_roster:
        errors.append("post_december_inventory: embedded source roster must match the pinned roster")

    entries = cfg.get("post_cutoff_repositories")
    if not isinstance(entries, list):
        errors.append("post_december_inventory: post_cutoff_repositories must be a list")
        entries = []
    seen_repos: set[str] = set()
    for index, entry in enumerate(entries):
        label = f"post_december_inventory.post_cutoff_repositories[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: must be an object")
            continue
        repo_id = entry.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id.startswith("glossAPI/"):
            errors.append(f"{label}: repo_id must be a glossAPI dataset ID")
        elif repo_id in seen_repos:
            errors.append(f"{label}: duplicate repo_id {repo_id!r}")
        else:
            seen_repos.add(repo_id)
        created_at = entry.get("created_at")
        if not isinstance(created_at, str) or not created_at.endswith("Z"):
            errors.append(f"{label}: created_at must be a UTC timestamp")
        elif created_at < POST_DECEMBER_CUTOFF:
            errors.append(f"{label}: repository predates the post-December cutoff")
        require_revision(entry.get("revision"), label, errors)
        artifact_bytes = entry.get("data_artifact_bytes")
        if not nonnegative_integer(artifact_bytes):
            errors.append(f"{label}: data_artifact_bytes must be non-negative")
            artifact_bytes = 0
        files = entry.get("data_files")
        if not isinstance(files, list):
            errors.append(f"{label}: data_files must be a list")
            files = []
        file_bytes = 0
        for file_index, artifact in enumerate(files):
            file_label = f"{label}.data_files[{file_index}]"
            if not isinstance(artifact, dict):
                errors.append(f"{file_label}: must be an object")
                continue
            if not is_safe_relative(artifact.get("path")):
                errors.append(f"{file_label}: path must be safe and relative")
            value = artifact.get("bytes")
            if not nonnegative_integer(value):
                errors.append(f"{file_label}: bytes must be non-negative")
            else:
                file_bytes += value
        if file_bytes != artifact_bytes:
            errors.append(f"{label}: data_artifact_bytes must equal data_files bytes")
        for field in ("payload_status", "relation_to_first_nanochat", "disposition"):
            if not isinstance(entry.get(field), str) or not entry.get(field):
                errors.append(f"{label}: {field} required")
        initial_name = entry.get("first_nanochat_source_dataset")
        if initial_name is not None and initial_name not in expected_roster:
            errors.append(f"{label}: unknown first_nanochat_source_dataset {initial_name!r}")
        card_tokens = entry.get("card_tokens")
        if not isinstance(card_tokens, dict):
            errors.append(f"{label}: card_tokens must be an object")
        elif card_tokens.get("value") is not None and not nonnegative_integer(
            card_tokens.get("value")
        ):
            errors.append(f"{label}: card_tokens.value must be null or non-negative")

    older = cfg.get("older_repositories_with_material_post_cutoff_changes")
    if not isinstance(older, list):
        errors.append(
            "post_december_inventory: older_repositories_with_material_post_cutoff_changes "
            "must be a list"
        )
        older = []
    older_repos: set[str] = set()
    for index, entry in enumerate(older):
        label = f"post_december_inventory.older_repositories[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: must be an object")
            continue
        repo_id = entry.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id.startswith("glossAPI/"):
            errors.append(f"{label}: repo_id must be a glossAPI dataset ID")
        elif repo_id in older_repos:
            errors.append(f"{label}: duplicate repo_id {repo_id!r}")
        else:
            older_repos.add(repo_id)
        if isinstance(entry.get("created_at"), str) and entry["created_at"] >= POST_DECEMBER_CUTOFF:
            errors.append(f"{label}: repository belongs in post_cutoff_repositories")
        require_revision(entry.get("pre_cutoff_revision"), f"{label}.pre_cutoff_revision", errors)
        require_revision(entry.get("current_revision"), f"{label}.current_revision", errors)
        material_changes = entry.get("material_changes")
        changed_path_count = entry.get("changed_payload_path_count")
        has_explicit_changes = isinstance(material_changes, list) and bool(material_changes)
        has_aggregate_changes = nonnegative_integer(changed_path_count) and changed_path_count > 0
        if not has_explicit_changes and not has_aggregate_changes:
            errors.append(
                f"{label}: require material_changes or a positive changed_payload_path_count"
            )

    selected_revisions = {
        entry.get("repo_id"): entry.get("revision")
        for entry in sources_cfg.get("sources", [])
        if isinstance(entry, dict)
    }
    backlog_revisions = {
        entry.get("repo_id"): entry.get("revision")
        for entry in backlog_cfg.get("entries", [])
        if isinstance(entry, dict)
    }
    registry_revisions = selected_revisions | backlog_revisions
    registry_repos = set(registry_revisions)
    if len(registry_repos) != CURRENT_GLOSSAPI_REPOSITORY_COUNT:
        errors.append("post_december_inventory: sources plus backlog must cover 40 audited repos")
    missing_from_registry = (seen_repos | older_repos) - registry_repos
    if missing_from_registry:
        errors.append(
            "post_december_inventory: audited repos missing from sources/backlog: "
            f"{sorted(missing_from_registry)}"
        )
    for entry in entries:
        if isinstance(entry, dict) and registry_revisions.get(entry.get("repo_id")) != entry.get(
            "revision"
        ):
            errors.append(
                f"post_december_inventory: revision drift for {entry.get('repo_id')!r}"
            )
    for entry in older:
        if isinstance(entry, dict) and registry_revisions.get(entry.get("repo_id")) != entry.get(
            "current_revision"
        ):
            errors.append(
                f"post_december_inventory: current revision drift for {entry.get('repo_id')!r}"
            )

    summary = cfg.get("summary")
    if not isinstance(summary, dict):
        errors.append("post_december_inventory: summary must be an object")
        summary = {}
    new_family = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("first_nanochat_source_dataset") is None
        and "full_text" in str(entry.get("payload_status"))
    ]
    external = [entry for entry in entries if entry.get("payload_status") == "no_hf_data"]
    metadata_only = [
        entry for entry in entries if entry.get("payload_status") == "metadata_only_parquet"
    ]
    empty = [entry for entry in entries if entry.get("payload_status") == "empty_scaffold"]
    same_source = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("first_nanochat_source_dataset") is not None
    ]
    new_family_repos = {entry.get("repo_id") for entry in new_family}
    same_source_repos = {entry.get("repo_id") for entry in same_source}
    nonacquiring_repos = {
        entry.get("repo_id") for entry in [*external, *metadata_only, *empty]
    }
    if new_family_repos - set(selected_revisions):
        errors.append(
            "post_december_inventory: usable new-family repos must be selected for audit: "
            f"{sorted(new_family_repos - set(selected_revisions))}"
        )
    if same_source_repos - set(selected_revisions):
        errors.append(
            "post_december_inventory: post-cutoff replacements must be selected for audit: "
            f"{sorted(same_source_repos - set(selected_revisions))}"
        )
    if nonacquiring_repos & set(selected_revisions):
        errors.append(
            "post_december_inventory: external/metadata/empty repos must remain non-acquiring: "
            f"{sorted(nonacquiring_repos & set(selected_revisions))}"
        )

    def token_value(entry: dict) -> int:
        value = entry.get("card_tokens", {}).get("value")
        return value if nonnegative_integer(value) else 0

    def footer_rows(entry: dict) -> int:
        rows = entry.get("rows", {})
        value = rows.get("footer_sum", rows.get("footer"))
        return value if nonnegative_integer(value) else 0

    expected_summary = {
        "post_cutoff_repository_count": len(entries),
        "post_cutoff_semantic_source_families_absent_from_first_nanochat": sum(
            entry.get("first_nanochat_source_dataset") is None
            for entry in entries
            if isinstance(entry, dict)
        ),
        "post_cutoff_same_source_replacement_repository_count": len(same_source),
        "post_cutoff_new_family_full_text_repository_count": len(new_family),
        "post_cutoff_external_only_repository_count": len(external),
        "post_cutoff_metadata_only_repository_count": len(metadata_only),
        "post_cutoff_empty_scaffold_repository_count": len(empty),
        "post_cutoff_local_data_artifact_bytes_all_statuses": sum(
            int(entry.get("data_artifact_bytes", 0))
            for entry in entries
            if nonnegative_integer(entry.get("data_artifact_bytes"))
        ),
        "new_family_full_text_data_artifact_bytes": sum(
            int(entry.get("data_artifact_bytes", 0)) for entry in new_family
        ),
        "new_family_full_text_footer_rows_raw_sum": sum(footer_rows(entry) for entry in new_family),
        "new_family_card_reported_tokens_arithmetic_sum": sum(
            token_value(entry) for entry in new_family
        ),
        "new_family_card_tokens_explicit_nlpaueb_greekbert_sum": sum(
            token_value(entry)
            for entry in new_family
            if entry.get("card_tokens", {}).get("tokenizer_scope")
            == "nlpaueb/bert-base-greek-uncased-v1"
        ),
        "new_family_card_tokens_unpinned_or_unspecified_sum": sum(
            token_value(entry)
            for entry in new_family
            if entry.get("card_tokens", {}).get("tokenizer_scope")
            != "nlpaueb/bert-base-greek-uncased-v1"
        ),
        "same_source_replacement_card_tokens_where_reported_sum": sum(
            token_value(entry) for entry in same_source
        ),
        "external_only_card_tokens_sum": sum(token_value(entry) for entry in external),
        "all_post_cutoff_card_token_numbers_arithmetic_sum": sum(
            token_value(entry) for entry in entries
        ),
        "older_repository_material_change_count": len(older),
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            errors.append(
                f"post_december_inventory: summary.{field} must equal recomputed value {expected}"
            )
    if len(entries) != POST_DECEMBER_REPOSITORY_COUNT:
        errors.append("post_december_inventory: post-cutoff repository count must remain 25")
    return errors


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
    if base.get("source_column") != "source_dataset":
        errors.append("base: source_column must remain source_dataset")

    provenance = cfg.get("normalized_provenance_contract")
    if not isinstance(provenance, dict):
        errors.append("sources: normalized_provenance_contract must be an object")
        provenance = {}
    if provenance.get("source_dataset_semantics") != (
        "preserve_exact_upstream_source_dataset_value_when_present_otherwise_use_pinned_repo_id"
    ):
        errors.append("sources: source_dataset semantics must preserve the upstream name")
    fields = provenance.get("required_fields")
    if not isinstance(fields, list) or not all(isinstance(field, str) and field for field in fields):
        errors.append("sources: normalized provenance required_fields must be a string list")
    elif set(fields) != REQUIRED_PROVENANCE_FIELDS or len(fields) != len(set(fields)):
        errors.append("sources: normalized provenance fields must match the frozen contract")

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

    seen_ids: set[str] = {"nanochat_base"}
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
        acquisition_kind = source.get("acquisition_kind", "huggingface")
        if acquisition_kind not in {"huggingface", "mozilla_data_collective"}:
            errors.append(f"{label}: invalid acquisition_kind {acquisition_kind!r}")
        if acquisition_kind == "mozilla_data_collective":
            for field in (
                "mdc_dataset_id",
                "mdc_slug",
                "mdc_name",
                "mdc_format",
                "mdc_expected_filename",
            ):
                if not isinstance(source.get(field), str) or not source.get(field):
                    errors.append(f"{label}: {field} required for MDC acquisition")
            expected_bytes = source.get("mdc_expected_bytes")
            if (
                not isinstance(expected_bytes, int)
                or isinstance(expected_bytes, bool)
                or expected_bytes < 1
            ):
                errors.append(f"{label}: mdc_expected_bytes must be a positive integer")
            expected_hash = source.get("mdc_expected_sha256")
            if expected_hash is not None and (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(character not in HEX40 for character in expected_hash)
            ):
                errors.append(
                    f"{label}: mdc_expected_sha256 must be null or lowercase 64-hex"
                )
        if source.get("role") not in ROLES:
            errors.append(f"{label}: invalid role {source.get('role')!r}")
        if source.get("structural_policy") not in STRUCTURAL_POLICIES:
            errors.append(f"{label}: invalid structural_policy {source.get('structural_policy')!r}")
        if source.get("training_eligibility") not in ELIGIBILITY:
            errors.append(f"{label}: invalid training_eligibility {source.get('training_eligibility')!r}")
        for field in ("source_family_id", "content_relation", "merge_policy"):
            if not isinstance(source.get(field), str) or not source.get(field):
                errors.append(f"{label}: {field} required")
        relation = str(source.get("content_relation", ""))
        if source.get("role") == "additive_candidate" and (
            relation.startswith("same_source")
            or relation.startswith("hybrid")
            or "base_overlap" in relation
        ):
            errors.append(f"{label}: overlap/replacement relation cannot use additive_candidate")
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
        required_text = source.get("required_text_columns")
        if required_text is not None:
            if (
                not isinstance(required_text, list)
                or not required_text
                or not all(isinstance(value, str) and value for value in required_text)
            ):
                errors.append(
                    f"{label}: required_text_columns must be a non-empty string list when set"
                )
            elif not set(required_text).issubset(
                set(source.get("text_columns", [])) | set(alternate or [])
            ):
                errors.append(
                    f"{label}: required_text_columns must be a subset of candidate text columns"
                )
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
            else:
                try:
                    re.compile(str(route["source_regex"]))
                except re.error as exc:
                    errors.append(f"{label}: invalid source_regex: {exc}")
            coverage = route.get("coverage_contract")
            if not isinstance(coverage, dict):
                errors.append(f"{label}: coverage_contract must be an object")
            else:
                if not isinstance(coverage.get("expected_source_dataset"), str) or not coverage.get(
                    "expected_source_dataset"
                ):
                    errors.append(
                        f"{label}: coverage_contract.expected_source_dataset required"
                    )
                minimum_rows = coverage.get("minimum_normalized_rows")
                if (
                    not isinstance(minimum_rows, int)
                    or isinstance(minimum_rows, bool)
                    or minimum_rows < 1
                ):
                    errors.append(
                        f"{label}: coverage_contract.minimum_normalized_rows "
                        "must be a positive integer"
                    )
                if coverage.get("enforcement_scope") != "unbounded_normalization":
                    errors.append(
                        f"{label}: coverage_contract.enforcement_scope must be "
                        "unbounded_normalization"
                    )
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


def validate_embedded_route_roster_coverage(
    sources_cfg: dict, roster_cfg: dict
) -> list[str]:
    """Prove tracked embedded routes against the frozen Nanochat roster.

    This static proof catches misspelled artifact globs and source regexes
    before a normalization job. Runtime normalization separately requires a
    positive routed-row count against the actual acquisition receipt.
    """

    errors: list[str] = []
    routes = sources_cfg.get("embedded_structural_routes")
    roster = roster_cfg.get("sources")
    if not isinstance(routes, list) or not isinstance(roster, list):
        return ["embedded route coverage proof requires route and roster lists"]
    roster_rows = {
        str(row.get("source_dataset")): row
        for row in roster
        if isinstance(row, dict) and isinstance(row.get("source_dataset"), str)
    }
    all_artifacts = [
        (source_dataset, str(artifact))
        for source_dataset, row in roster_rows.items()
        for artifact in row.get("artifacts", [])
        if isinstance(artifact, str)
    ]
    base_globs = sources_cfg.get("base", {}).get("include_globs", [])
    for index, route in enumerate(routes):
        label = f"embedded_structural_routes[{index}]"
        if not isinstance(route, dict):
            continue
        coverage = route.get("coverage_contract")
        if not isinstance(coverage, dict):
            continue
        expected = coverage.get("expected_source_dataset")
        if not isinstance(expected, str) or not expected:
            continue
        roster_row = roster_rows.get(expected)
        if roster_row is None:
            errors.append(
                f"{label}: expected source_dataset {expected!r} is absent from the "
                "frozen Nanochat roster"
            )
            continue
        rows = roster_row.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            errors.append(f"{label}: frozen roster has no positive rows for {expected!r}")
        pattern = route.get("source_regex")
        try:
            compiled = re.compile(str(pattern))
        except re.error:
            continue
        regex_matches = sorted(
            name for name in roster_rows if compiled.search(name) is not None
        )
        if regex_matches != [expected]:
            errors.append(
                f"{label}: source_regex must match exactly the expected frozen source "
                f"{expected!r}; matched={regex_matches}"
            )
        globs = route.get("acquisition_include_globs")
        if not isinstance(globs, list) or not globs:
            continue
        target_artifacts = [
            str(artifact)
            for artifact in roster_row.get("artifacts", [])
            if isinstance(artifact, str)
        ]
        missing = [
            artifact
            for artifact in target_artifacts
            if not any(fnmatch.fnmatchcase(artifact, str(pattern)) for pattern in globs)
        ]
        if missing:
            errors.append(
                f"{label}: acquisition globs miss frozen artifacts for {expected!r}: {missing}"
            )
        cross_matches = sorted(
            f"{source_dataset}:{artifact}"
            for source_dataset, artifact in all_artifacts
            if source_dataset != expected
            and any(fnmatch.fnmatchcase(artifact, str(pattern)) for pattern in globs)
        )
        if cross_matches:
            errors.append(
                f"{label}: acquisition globs also match other frozen sources: {cross_matches}"
            )
        base_missing = [
            artifact
            for artifact in target_artifacts
            if not any(
                fnmatch.fnmatchcase(artifact, str(pattern)) for pattern in base_globs
            )
        ]
        if base_missing:
            errors.append(
                f"{label}: expected artifacts are outside base.include_globs: {base_missing}"
            )
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
    parity_hash = validation.get("structural_parity_corpus_sha256")
    if parity_hash is not None and (
        not isinstance(parity_hash, str)
        or len(parity_hash) != 64
        or any(c not in HEX40 for c in parity_hash)
    ):
        errors.append(
            "policy: validation.structural_parity_corpus_sha256 must be null or lowercase 64-hex"
        )
    if validation.get("structural_parity_evidence") != "LLM_silver":
        errors.append("policy: structural parity evidence must be declared as LLM_silver")
    parity_documents = validation.get("required_parity_documents")
    if not isinstance(parity_documents, int) or isinstance(parity_documents, bool) or parity_documents < 1:
        errors.append("policy: required_parity_documents must be a positive integer")
    structural_enabled = any(
        structural.get(kind, {}).get("enabled_for_materialization")
        for kind in ("bibliography", "toc")
    )
    if structural_enabled:
        if validation.get("structural_application_receipt_required") is not True:
            errors.append("policy: enabled structural cleaning requires an application receipt")
        if validation.get("required_model_evidence") != "LLM_silver":
            errors.append("policy: structural model evidence must be declared as LLM_silver")
        if validation.get("required_safety_evidence") != "targeted_manual_false_deletion_audit":
            errors.append("policy: enabled structural cleaning requires targeted manual safety evidence")
    gates = structural.get("application_gates")
    if not isinstance(gates, dict):
        errors.append("policy: structural.application_gates must be an object")
    else:
        reviewed = gates.get("minimum_reviewed_deletions")
        if not isinstance(reviewed, int) or isinstance(reviewed, bool) or reviewed < 1:
            errors.append("policy: minimum_reviewed_deletions must be a positive integer")
        for name in (
            "maximum_running_prose_deletion_rate",
            "minimum_main_text_retention_rate",
            "maximum_catastrophic_document_deletion_rate",
        ):
            value = gates.get(name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= float(value) <= 1:
                errors.append(f"policy: structural.application_gates.{name} must be in [0, 1]")
    if cfg.get("diavgeia", {}).get("academic_structural_cleaner") != "disabled":
        errors.append("policy: Diavgeia academic structural cleaner must be disabled")
    return errors


def validate_source_review_policy(cfg: dict, sources_cfg: dict) -> list[str]:
    errors: list[str] = []
    label = "source_review_policy"
    if cfg.get("schema_version") != "full_cpt_source_review_policy_v1":
        errors.append(f"{label}: unsupported schema_version")
    if not isinstance(cfg.get("seed"), str) or not cfg.get("seed"):
        errors.append(f"{label}: deterministic seed required")
    if cfg.get("grouping_field") != "source_dataset":
        errors.append(f"{label}: grouping_field must remain source_dataset")
    if cfg.get("source_dataset_policy") != (
        "preserve_exact_upstream_value_otherwise_pinned_repo_fallback"
    ):
        errors.append(f"{label}: source_dataset preservation policy drift")
    expected_samples = {
        "default_sample": {
            "total_unique_documents": 100,
            "random": 60,
            "risk": 20,
            "cluster": 20,
        },
        "large_or_heterogeneous_sample": {
            "total_unique_documents": 200,
            "random": 100,
            "risk": 50,
            "cluster": 50,
        },
    }
    for field, expected in expected_samples.items():
        if cfg.get(field) != expected:
            errors.append(f"{label}: {field} must remain {expected}")
    threshold = cfg.get("large_source_min_documents")
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 100:
        errors.append(f"{label}: large_source_min_documents must be an integer >= 100")
    known_source_ids = {
        entry.get("source_id") for entry in sources_cfg.get("sources", []) if isinstance(entry, dict)
    }
    heterogeneous = cfg.get("heterogeneous_source_ids")
    if (
        not isinstance(heterogeneous, list)
        or len(heterogeneous) != len(set(heterogeneous))
        or any(source_id not in known_source_ids for source_id in heterogeneous)
        or "diavgeia" not in heterogeneous
    ):
        errors.append(
            f"{label}: heterogeneous_source_ids must be unique, registered and include diavgeia"
        )
    cluster_fields = cfg.get("cluster_fields_in_priority_order")
    if (
        not isinstance(cluster_fields, list)
        or not cluster_fields
        or not all(isinstance(field, str) and field for field in cluster_fields)
    ):
        errors.append(f"{label}: cluster_fields_in_priority_order must be a non-empty string list")
    fraction = cfg.get("double_review_fraction")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or float(fraction) != 0.1:
        errors.append(f"{label}: double_review_fraction must remain 0.1")
    excerpt = cfg.get("excerpt")
    if not isinstance(excerpt, dict) or any(
        not isinstance(excerpt.get(field), int)
        or isinstance(excerpt.get(field), bool)
        or excerpt.get(field) <= 0
        for field in ("full_text_max_characters", "segment_characters")
    ):
        errors.append(f"{label}: excerpt limits must be positive integers")
    admission = cfg.get("admission")
    if not isinstance(admission, dict):
        errors.append(f"{label}: admission must be an object")
    else:
        expected_fractions = {
            "direct_include_min_usable_fraction": 0.9,
            "cleanable_min_useful_fraction": 0.8,
            "post_clean_min_usable_fraction": 0.9,
            "minimum_novel_token_fraction": 0.05,
        }
        for field, expected in expected_fractions.items():
            if admission.get(field) != expected:
                errors.append(f"{label}: admission.{field} must remain {expected}")
        for field in (
            "low_confidence_requires_adjudication",
            "disagreement_requires_adjudication",
            "safety_or_license_blocker_forces_quarantine",
        ):
            if admission.get(field) is not True:
                errors.append(f"{label}: admission.{field} must remain true")
    return errors


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=here / "configs" / "sources.json")
    parser.add_argument("--backlog", type=Path, default=here / "configs" / "source_backlog.json")
    parser.add_argument("--policy", type=Path, default=here / "configs" / "cleaning_policy.json")
    parser.add_argument(
        "--nanochat-roster",
        type=Path,
        default=here / "configs" / "nanochat_initial_roster.json",
    )
    parser.add_argument(
        "--lineage-aliases",
        type=Path,
        default=here / "configs" / "source_lineage_aliases.json",
    )
    parser.add_argument(
        "--post-december-inventory",
        type=Path,
        default=here / "configs" / "post_december_inventory.json",
    )
    parser.add_argument(
        "--source-review-policy",
        type=Path,
        default=here / "configs" / "source_review_policy.json",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable result")
    args = parser.parse_args()

    source_cfg = load_json(args.sources)
    backlog_cfg = load_json(args.backlog)
    policy_cfg = load_json(args.policy)
    roster_cfg = load_json(args.nanochat_roster)
    aliases_cfg = load_json(args.lineage_aliases)
    inventory_cfg = load_json(args.post_december_inventory)
    source_review_cfg = load_json(args.source_review_policy)
    errors = (
        validate_initial_roster(roster_cfg)
        + validate_lineage_aliases(aliases_cfg, roster_cfg)
        + validate_alias_registry_revisions(aliases_cfg, source_cfg, backlog_cfg)
        + validate_post_december_inventory(
            inventory_cfg,
            roster_cfg,
            source_cfg,
            backlog_cfg,
        )
        + validate_sources(source_cfg)
        + validate_embedded_route_roster_coverage(source_cfg, roster_cfg)
        + validate_backlog(backlog_cfg, source_cfg)
        + validate_policy(policy_cfg)
        + validate_source_review_policy(source_review_cfg, source_cfg)
    )
    result = {
        "ok": not errors,
        "errors": errors,
        "sources": len(source_cfg.get("sources", [])),
        "backlog_entries": len(backlog_cfg.get("entries", [])),
        "nanochat_initial_sources": len(roster_cfg.get("sources", [])),
        "lineage_aliases": len(aliases_cfg.get("aliases", [])),
        "sources_sha256": digest(args.sources),
        "backlog_sha256": digest(args.backlog),
        "policy_sha256": digest(args.policy),
        "nanochat_roster_sha256": digest(args.nanochat_roster),
        "lineage_aliases_sha256": digest(args.lineage_aliases),
        "post_december_inventory_sha256": digest(args.post_december_inventory),
        "source_review_policy_sha256": digest(args.source_review_policy),
        "post_december_repositories": len(inventory_cfg.get("post_cutoff_repositories", [])),
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
            f"nanochat_initial={result['nanochat_initial_sources']}; "
            f"lineage_aliases={result['lineage_aliases']}; "
            f"post_december={result['post_december_repositories']}; "
            f"policy={result['policy_status']}; sources_sha256={result['sources_sha256']}; "
            f"backlog_sha256={result['backlog_sha256']}; policy_sha256={result['policy_sha256']}"
        )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
