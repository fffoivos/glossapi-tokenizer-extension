#!/usr/bin/env python3
"""Fail-closed contracts for controlled bibliography-model evolution.

This module deliberately does not fit a model.  It wraps the existing D1,
signal-TCN, anchored decoder, header, and gap experiments with immutable
identities and receipts so that validation-guided iteration remains auditable.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPEC_SCHEMA = "bibliography-evolution-candidate-v1"
RESULT_SCHEMA = "bibliography-evolution-result-v1"
RECEIPT_SCHEMA = "bibliography-evolution-candidate-receipt-v1"
REGISTRY_SCHEMA = "bibliography-evolution-registry-v1"
LEAKAGE_SCHEMA = "bibliography-evolution-leakage-policy-v1"
BASELINE_SCHEMA = "bibliography-evolution-baseline-lock-v1"

OBJECTIVES = (
    "token_fp",
    "token_fn",
    "spurious_blocks_per_zero_block_document",
    "mean_boundary_error_emitted_lines",
)

# The order is part of every candidate identity.  A component can be disabled,
# but no experiment may silently move it before or after another component.
FIXED_MODULE_ORDER = (
    "d1_line_scorer",
    "signal_tcn_with_deterministic_roles",
    "scope_header_partition",
    "physical_gap_walls",
    "exact_scope_walls",
    "header_prebarriers",
    "anchor_core_formation",
    "internal_gap_connection",
    "boundary_trim",
    "outward_edge_optional",
    "weak_unseeded_optional",
    "bib_header_attachment",
    "whole_component_veto",
    "metrics",
)

CHANGED_COMPONENTS = frozenset(
    {
        "baseline.replay",
        "decoder.anchor_and_expansion_policy",
        "headers.role_controller",
        "decoder.fringe_trim",
        "decoder.gap_connector",
        "decoder.component_veto",
        "decoder.outward_edge",
        "decoder.weak_unseeded",
        "signal.features",
        "signal.architecture",
        "signal.training",
        "composition.pairwise",
    }
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_UNRESOLVED = re.compile(r"\$\{[^}]+\}")
INPUT_PATH_FLAGS = frozenset(
    {
        "--input", "--table-dir", "--validation-table-dir", "--line-oof-dir",
        "--signal-tcn-dir", "--block-oof-dir", "--deterministic-roles-dir",
        "--quality-decisions", "--policy", "--train-recall-block-dir",
        "--validation-line-probability", "--signal-probability", "--line-probability",
        "--scope-mask", "--qualified-documents", "--baseline-prediction",
        "--header-roles", "--left-prediction", "--right-prediction", "--lock",
        "--root", "--replay-prediction", "--authoritative-root",
        "--validation-signal-probability", "--validation-scope-mask",
        "--train-table-dir", "--train-quality-decisions", "--validation-quality-decisions",
        "--validation-policy",
        "--barrier-artifact", "--left-barrier-artifact", "--right-barrier-artifact",
    }
)
OUTPUT_PATH_FLAGS = frozenset({"--output", "--output-dir", "--output-rows", "--output-report"})
PARENT_ARTIFACT_FLAGS = frozenset(
    {
        "--baseline-prediction",
        "--barrier-artifact",
        "--left-prediction",
        "--right-prediction",
        "--left-barrier-artifact",
        "--right-barrier-artifact",
    }
)
EXPECTED_RUNNER_BY_COMPONENT = {
    "baseline.replay": "sequence_models.bibliography_evolution_g0_replay",
    "decoder.anchor_and_expansion_policy": "sequence_models.bibliography_evolution_core_decode",
    "headers.role_controller": "sequence_models.bibliography_evolution_postprocess",
    "decoder.fringe_trim": "sequence_models.bibliography_evolution_postprocess",
    "decoder.gap_connector": "sequence_models.bibliography_evolution_postprocess",
    "decoder.component_veto": "sequence_models.bibliography_evolution_postprocess",
    "decoder.outward_edge": "sequence_models.bibliography_evolution_postprocess",
    "decoder.weak_unseeded": "sequence_models.bibliography_evolution_postprocess",
    "signal.architecture": "sequence_models.bibliography_evolution_signal_pipeline",
    "signal.features": "sequence_models.bibliography_evolution_signal_pipeline",
    "signal.training": "sequence_models.bibliography_evolution_signal_pipeline",
    "composition.pairwise": "sequence_models.bibliography_evolution_composition",
}
ALLOWED_RUNNER_FLAGS = {
    "sequence_models.bibliography_evolution_g0_replay": {
        "--lock", "--authoritative-root", "--validation-table-dir",
        "--validation-signal-probability", "--validation-line-probability",
        "--validation-scope-mask", "--qualified-documents", "--output-dir",
        "--code-commit", "--slurm-job-id",
    },
    "sequence_models.bibliography_evolution_core_decode": {
        "--table-dir", "--signal-probability", "--line-probability", "--scope-mask",
        "--qualified-documents", "--anchor-probability", "--anchors-required",
        "--anchor-window", "--maximum-bridge-gap", "--inside-probability",
        "--adjacent-expansion", "--header-window", "--output-dir", "--code-commit",
        "--slurm-job-id",
    },
    "sequence_models.bibliography_evolution_postprocess": {
        "--table-dir", "--baseline-prediction", "--signal-probability", "--scope-mask",
        "--barrier-artifact", "--header-roles", "--qualified-documents", "--operation",
        "--threshold", "--max-lines", "--output-dir", "--code-commit", "--slurm-job-id",
    },
    "sequence_models.bibliography_evolution_signal_pipeline": {
        "--input", "--train-table-dir", "--line-oof-dir", "--block-oof-dir",
        "--deterministic-roles-dir", "--train-quality-decisions", "--validation-table-dir",
        "--validation-line-probability", "--train-recall-block-dir",
        "--validation-quality-decisions", "--validation-policy", "--output-dir",
        "--hidden-dim", "--dilations", "--dropout", "--epochs", "--seed", "--workers",
        "--cpus", "--code-commit", "--slurm-job-id",
    },
    "sequence_models.bibliography_evolution_composition": {
        "--table-dir", "--left-prediction", "--right-prediction", "--qualified-documents",
        "--left-barrier-artifact", "--right-barrier-artifact", "--operation", "--output-dir",
        "--code-commit", "--slurm-job-id",
    },
}


class ContractError(ValueError):
    """Raised when an evolution artifact fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    """Hash the complete regular-file inventory and contents of a directory.

    A receipt file inside a directory is not an integrity boundary: a runner
    can consume any of its siblings.  Directory inputs are therefore bound to
    every relative file name, byte count, and file digest.  Symlinks and
    special files fail closed so the inventory cannot change underneath the
    resolved root.
    """

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"directory digest requires a real directory: {root}")
    digest = hashlib.sha256(b"bibliography-evolution-directory-tree-v1\0")
    for child in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        relative = child.relative_to(root).as_posix()
        if child.is_symlink():
            raise ContractError(f"directory input contains a symlink: {relative}")
        if child.is_dir():
            continue
        if not child.is_file():
            raise ContractError(f"directory input contains a special file: {relative}")
        digest.update(
            canonical_json_bytes(
                {
                    "path": relative,
                    "bytes": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
        )
    return digest.hexdigest()


def sha256_input_path(path: Path) -> str:
    """Return the contract digest for a regular file or complete directory."""

    raw = Path(path)
    if raw.is_symlink():
        raise ContractError(f"input path is a symlink: {raw}")
    if raw.is_file():
        return sha256_file(raw)
    if raw.is_dir():
        return sha256_directory(raw)
    raise ContractError(f"input path is not a regular file or directory: {raw}")


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json_bytes(value).decode("utf-8"))


def _without_candidate_id(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spec.items() if key != "candidate_id"}


def derive_candidate_id(spec: Mapping[str, Any]) -> str:
    """Derive an identity from parents, full spec, code, and input receipts."""

    payload = _without_candidate_id(spec)
    generation = str(payload.get("generation", "invalid")).lower()
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"{generation}-{digest[:24]}"


def with_candidate_id(spec: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(spec))
    result["candidate_id"] = derive_candidate_id(result)
    validate_candidate_spec(result)
    return result


def _require_sha256(value: Any, field: str) -> str:
    text = str(value)
    if not _HEX64.fullmatch(text):
        raise ContractError(f"{field} must be a lowercase SHA-256")
    return text


def validate_candidate_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ContractError("unsupported candidate schema")
    generation = str(spec.get("generation", ""))
    if generation not in {f"G{index}" for index in range(6)}:
        raise ContractError("generation must be G0 through G5")
    if not str(spec.get("hypothesis", "")).strip():
        raise ContractError("candidate hypothesis is required")
    component = str(spec.get("changed_component", ""))
    if component not in CHANGED_COMPONENTS:
        raise ContractError(f"unsupported changed component: {component!r}")
    changes = spec.get("changes")
    if not isinstance(changes, Mapping) or set(changes) != {component}:
        raise ContractError("changes must contain exactly the declared component")
    family = str(spec.get("parameter_family", ""))
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", family):
        raise ContractError("parameter_family must be one stable snake-case family")
    change = changes[component]
    if not isinstance(change, Mapping) or set(change) != {"parameter_family", "parameters"}:
        raise ContractError(
            "the changed component must declare exactly one parameter_family and parameters"
        )
    if change["parameter_family"] != family:
        raise ContractError("changed component and candidate parameter families differ")
    if not isinstance(change["parameters"], Mapping) or not change["parameters"]:
        raise ContractError("the changed parameter family has no parameters")
    if generation == "G0" and component != "baseline.replay":
        raise ContractError("G0 may only replay the baseline")
    if generation != "G0" and component == "baseline.replay":
        raise ContractError("baseline.replay is reserved for G0")
    parents = spec.get("parent_candidate_ids")
    if not isinstance(parents, list) or any(
        not isinstance(value, str) or not value for value in parents
    ):
        raise ContractError("parent_candidate_ids must be a string list")
    if generation == "G0" and parents:
        raise ContractError("G0 has no candidate parent")
    if generation != "G0" and not parents:
        raise ContractError("descendants require at least one parent")
    if len(parents) != len(set(parents)):
        raise ContractError("parent candidate ids are duplicated")
    if tuple(spec.get("fixed_modules", ())) != FIXED_MODULE_ORDER:
        raise ContractError("fixed module order changed")
    grid = spec.get("sweep_grid")
    point = spec.get("sweep_point")
    if not isinstance(grid, Mapping) or not isinstance(point, Mapping):
        raise ContractError("sweep_grid and sweep_point are required mappings")
    if set(point) != set(grid):
        raise ContractError("sweep point does not instantiate the declared grid")
    for name, values in grid.items():
        if not isinstance(name, str) or not isinstance(values, list) or not values:
            raise ContractError("every sweep dimension requires a non-empty value list")
        if point[name] not in values:
            raise ContractError(f"sweep point {name!r} is outside its grid")
        parameters = change["parameters"]
        if name not in parameters or parameters[name] != point[name]:
            raise ContractError(
                f"sweep dimension {name!r} is not owned by the one changed parameter family"
            )
    seeds = spec.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds
    ):
        raise ContractError("at least one integer seed is required")
    expected = spec.get("expected_direction")
    if not isinstance(expected, Mapping) or not expected:
        raise ContractError("expected_direction is required")
    if not set(expected).issubset(OBJECTIVES):
        raise ContractError("expected_direction names an unknown objective")
    if any(value not in {"decrease", "nonincrease", "tradeoff"} for value in expected.values()):
        raise ContractError("invalid expected objective direction")
    acceptance = spec.get("acceptance_rule")
    if not isinstance(acceptance, Mapping) or not acceptance:
        raise ContractError("acceptance_rule is required")
    code_commit = str(spec.get("code_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ContractError("code_commit must be a full Git commit")
    _require_sha256(spec.get("leakage_policy_sha256"), "leakage_policy_sha256")
    inputs = spec.get("input_receipts")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ContractError("input_receipts are required")
    for name, row in inputs.items():
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise ContractError("invalid input receipt")
        _require_sha256(row.get("sha256"), f"input_receipts.{name}.sha256")
        if not str(row.get("path", "")):
            raise ContractError(f"input_receipts.{name}.path is required")
        if not str(row.get("data_class", "")):
            raise ContractError(f"input_receipts.{name}.data_class is required")
        if not str(row.get("split", "")):
            raise ContractError(f"input_receipts.{name}.split is required")
        if not str(row.get("document_scope", "")):
            raise ContractError(f"input_receipts.{name}.document_scope is required")
    parent_rows = [
        row for row in inputs.values()
        if row.get("data_class") == "parent_candidate_receipt"
    ]
    parent_row_ids = [str(row.get("candidate_id", "")) for row in parent_rows]
    if sorted(parent_row_ids) != sorted(parents):
        raise ContractError(
            "parent_candidate_ids must be backed one-for-one by parent_candidate_receipt inputs"
        )
    if generation == "G5":
        registries = [
            row for row in inputs.values()
            if row.get("data_class") == "development_pareto_registry"
        ]
        if len(registries) != 1:
            raise ContractError("G5 requires exactly one hashed development Pareto registry")
    runner = spec.get("runner")
    if not isinstance(runner, Mapping) or not str(runner.get("module", "")):
        raise ContractError("runner.module is required")
    argv = runner.get("argv")
    if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
        raise ContractError("runner.argv must be a string list")
    module = str(runner["module"])
    if module != EXPECTED_RUNNER_BY_COMPONENT.get(component):
        raise ContractError("runner module is not pinned to the changed component")
    allowed_flags = ALLOWED_RUNNER_FLAGS[module]
    seen_flags: set[str] = set()
    index = 0
    while index < len(argv):
        flag = argv[index]
        if not flag.startswith("--") or flag not in allowed_flags:
            raise ContractError(f"runner contains an unapproved flag/position: {flag!r}")
        if flag in seen_flags:
            raise ContractError(f"runner repeats flag: {flag}")
        seen_flags.add(flag)
        index += 1
        if flag == "--dilations":
            start = index
            while index < len(argv) and not argv[index].startswith("--"):
                index += 1
            if start == index:
                raise ContractError("--dilations requires values")
        else:
            if index >= len(argv) or argv[index].startswith("--"):
                raise ContractError(f"runner flag requires one value: {flag}")
            index += 1
    optional = {"--header-roles"} if module.endswith("postprocess") else set()
    if not (allowed_flags - optional).issubset(seen_flags):
        raise ContractError(
            f"runner misses required flags: {sorted((allowed_flags - optional) - seen_flags)}"
        )
    if module.endswith("postprocess"):
        operation = argv[argv.index("--operation") + 1]
        if (operation == "header_controller") != ("--header-roles" in seen_flags):
            raise ContractError("header_controller alone requires --header-roles")
    unresolved = _UNRESOLVED.search(json.dumps(spec, sort_keys=True))
    if unresolved:
        raise ContractError(f"unresolved template binding: {unresolved.group(0)}")
    expected_id = derive_candidate_id(spec)
    if spec.get("candidate_id") != expected_id:
        raise ContractError("candidate_id does not match canonical spec")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_document_ids(path: Path) -> set[str]:
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("{"):
                row = json.loads(raw)
                value = row.get("document_id")
            else:
                value = raw
            if not isinstance(value, str) or not value:
                raise ContractError(f"invalid document id inventory row in {path}")
            if value in result:
                raise ContractError(f"duplicate document id in {path}: {value}")
            result.add(value)
    return result


def _verified_declared_path(row: Mapping[str, Any], name: str) -> tuple[Path, str]:
    raw_path = Path(str(row["path"])).expanduser()
    if not raw_path.exists() or raw_path.is_symlink():
        raise ContractError(f"{name}: declared input is missing or a symlink")
    path = raw_path.resolve()
    expected = _require_sha256(row["sha256"], f"input_receipts.{name}.sha256")
    if path.is_file():
        if row.get("digest_kind", "file_sha256") != "file_sha256":
            raise ContractError(f"{name}: file input requires digest_kind=file_sha256")
        actual = sha256_file(path)
    elif path.is_dir():
        if row.get("digest_kind") != "recursive_tree_sha256_v1":
            raise ContractError(
                f"{name}: directory input requires digest_kind=recursive_tree_sha256_v1"
            )
        if "hash_path" in row:
            raise ContractError(f"{name}: hash_path cannot bind a directory input")
        actual = sha256_directory(path)
    else:
        raise ContractError(f"{name}: declared input is not a file or directory")
    if actual != expected:
        raise ContractError(f"{name}: declared input SHA-256 does not match bytes")
    return path, actual


def _receipt_artifact_inventory(
    receipt: Mapping[str, Any], candidate_dir: Path
) -> dict[Path, str]:
    """Resolve every artifact owned by a finalized parent receipt."""

    rows: list[Mapping[str, Any]] = []
    if isinstance(receipt.get("all_rows"), Mapping):
        rows.append(receipt["all_rows"])
    for section in ("artifacts", "predictions"):
        value = receipt.get(section, {})
        if not isinstance(value, Mapping):
            raise ContractError(f"parent receipt has invalid {section}")
        rows.extend(row for row in value.values() if isinstance(row, Mapping))
    result: dict[Path, str] = {}
    for row in rows:
        relative = Path(str(row.get("path", "")))
        path = relative if relative.is_absolute() else candidate_dir / relative
        path = path.resolve()
        if not _path_is_within(path, candidate_dir):
            raise ContractError("parent receipt artifact escapes its candidate directory")
        digest = _require_sha256(row.get("sha256"), "parent artifact sha256")
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ContractError(f"parent receipt artifact changed: {path}")
        if path in result and result[path] != digest:
            raise ContractError("parent receipt gives conflicting artifact hashes")
        result[path] = digest
    return result


def verify_parent_lineage(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Verify parent identities and bind every consumed parent artifact."""

    parents = list(spec["parent_candidate_ids"])
    receipt_rows = [
        row for row in spec["input_receipts"].values()
        if row.get("data_class") == "parent_candidate_receipt"
    ]
    if not parents:
        if receipt_rows:
            raise ContractError("G0 cannot declare parent receipts")
        return {"status": "passed", "parents": [], "bound_parent_artifacts": []}
    indexed: dict[str, tuple[Mapping[str, Any], Path, Mapping[str, Any], dict[Path, str]]] = {}
    for row in receipt_rows:
        candidate_id = str(row.get("candidate_id", ""))
        path, _ = _verified_declared_path(row, f"parent_receipt:{candidate_id}")
        if not path.is_file():
            raise ContractError("parent candidate receipt must be a regular file")
        receipt = load_json(path)
        if (
            receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("status") != "passed"
            or receipt.get("candidate_id") != candidate_id
        ):
            raise ContractError(f"parent candidate is not a passed receipt: {candidate_id}")
        # A schema/status-shaped JSON object is not sufficient evidence of a
        # parent.  Revalidate its full finalized result, input lineage, and
        # owned artifacts recursively down to G0.
        parent_verification = verify_finalized_receipt(path)
        if parent_verification.get("candidate_id") != candidate_id:
            raise ContractError(f"parent finalized verification failed: {candidate_id}")
        candidate_dir = path.parent.resolve()
        spec_path = candidate_dir / "spec.json"
        if not spec_path.is_file() or sha256_file(spec_path) != receipt.get("spec_sha256"):
            raise ContractError(f"parent spec changed: {candidate_id}")
        parent_spec = load_json(spec_path)
        validate_candidate_spec(parent_spec)
        if parent_spec.get("candidate_id") != candidate_id:
            raise ContractError(f"parent spec identity mismatch: {candidate_id}")
        if candidate_id in indexed:
            raise ContractError(f"duplicate parent receipt: {candidate_id}")
        indexed[candidate_id] = (
            row,
            path,
            receipt,
            _receipt_artifact_inventory(receipt, candidate_dir),
        )
    if set(indexed) != set(parents):
        raise ContractError("declared parent receipts do not match parent_candidate_ids")

    bound: list[dict[str, Any]] = []
    for name, row in sorted(spec["input_receipts"].items()):
        parent_id = row.get("parent_candidate_id")
        if parent_id is None:
            continue
        parent_id = str(parent_id)
        if parent_id not in indexed:
            raise ContractError(f"{name}: artifact names an undeclared parent")
        if row.get("data_class") == "parent_candidate_receipt":
            continue
        path, digest = _verified_declared_path(row, name)
        if not path.is_file():
            raise ContractError(f"{name}: a parent artifact must be a file")
        inventory = indexed[parent_id][3]
        if inventory.get(path) != digest:
            raise ContractError(f"{name}: artifact is not owned by parent {parent_id}")
        bound.append({"name": name, "candidate_id": parent_id, "path": str(path), "sha256": digest})

    pareto_registry: dict[str, Any] | None = None
    if spec["generation"] == "G5":
        rows = [
            row for row in spec["input_receipts"].values()
            if row.get("data_class") == "development_pareto_registry"
        ]
        if len(rows) != 1:
            raise ContractError("G5 requires one development Pareto registry")
        registry_path, registry_sha = _verified_declared_path(rows[0], "development_pareto_registry")
        registry = load_json(registry_path)
        if registry.get("schema_version") != REGISTRY_SCHEMA:
            raise ContractError("G5 Pareto registry has an unsupported schema")
        if set(parents) - set(registry.get("pareto_candidate_ids", ())):
            raise ContractError("G5 parents are not both in the development Pareto set")
        registry_rows = {row["candidate_id"]: row for row in registry.get("candidates", ())}
        for parent_id in parents:
            row = registry_rows.get(parent_id, {})
            receipt_path = indexed[parent_id][1]
            if (
                not row.get("pareto")
                or not row.get("eligible")
                or Path(str(row.get("receipt_path", ""))).resolve() != receipt_path
                or row.get("receipt_sha256") != sha256_file(receipt_path)
            ):
                raise ContractError(f"G5 registry does not bind parent receipt: {parent_id}")
        pareto_registry = {"path": str(registry_path), "sha256": registry_sha}
    return {
        "status": "passed",
        "parents": [
            {
                "candidate_id": parent_id,
                "receipt_path": str(indexed[parent_id][1]),
                "receipt_sha256": sha256_file(indexed[parent_id][1]),
            }
            for parent_id in parents
        ],
        "bound_parent_artifacts": bound,
        "pareto_registry": pareto_registry,
    }


def _reconcile_runner_paths(
    spec: Mapping[str, Any], declared: Mapping[Path, Sequence[Mapping[str, Any]]]
) -> None:
    argv = list(spec["runner"]["argv"])
    index = 0
    while index < len(argv):
        flag = argv[index]
        if flag in INPUT_PATH_FLAGS | OUTPUT_PATH_FLAGS:
            if index + 1 >= len(argv):
                raise ContractError(f"runner path flag has no value: {flag}")
            value = argv[index + 1]
            if flag in OUTPUT_PATH_FLAGS:
                if not value.startswith("@CANDIDATE_DIR@/"):
                    raise ContractError(f"runner output is outside the candidate directory: {value}")
            else:
                resolved = Path(value).resolve()
                if resolved not in declared:
                    raise ContractError(f"runner reads undeclared input path: {value}")
                if flag in PARENT_ARTIFACT_FLAGS and not any(
                    row.get("parent_candidate_id") in spec["parent_candidate_ids"]
                    for row in declared[resolved]
                ):
                    raise ContractError(
                        f"runner parent artifact is not bound to a declared parent: {value}"
                    )
            index += 2
            continue
        index += 1


def enforce_leakage_barrier(
    spec: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject final-test labels, paths, hashes, identities, and unknown scopes."""

    validate_candidate_spec(spec)
    if policy.get("schema_version") != LEAKAGE_SCHEMA:
        raise ContractError("unsupported leakage policy")
    policy_sha256 = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()
    if spec.get("leakage_policy_sha256") != policy_sha256:
        raise ContractError("candidate identity is bound to a different leakage policy")
    status = policy.get("sealed_test_status")
    if status not in {"not_materialized", "sealed"}:
        raise ContractError("leakage policy has no valid sealed-test status")
    allowed_scopes = set(policy.get("allowed_development_scopes", ()))
    if not allowed_scopes:
        raise ContractError("leakage policy has no approved development scopes")
    forbidden_classes = set(
        policy.get(
            "forbidden_data_classes",
            ("sealed_document", "sealed_label", "sealed_prediction"),
        )
    )
    forbidden_tokens = tuple(
        str(value).lower() for value in policy.get("forbidden_path_tokens", ())
    )
    sealed_roots = [Path(value) for value in policy.get("sealed_roots", ())]
    sealed_hashes = {
        _require_sha256(value, "sealed_artifact_sha256")
        for value in policy.get("sealed_artifact_sha256", ())
    }
    sealed_ids: set[str] = set()
    inventory = policy.get("sealed_document_ids_path")
    if status == "sealed":
        if not inventory:
            raise ContractError("sealed policy lacks its document-id inventory")
        inventory_path = Path(str(inventory))
        expected = _require_sha256(
            policy.get("sealed_document_ids_sha256"),
            "sealed_document_ids_sha256",
        )
        if sha256_file(inventory_path) != expected:
            raise ContractError("sealed document inventory hash changed")
        sealed_ids = _load_document_ids(inventory_path)
        if not sealed_ids:
            raise ContractError("sealed document inventory is empty")
    elif inventory or policy.get("sealed_artifact_sha256") or sealed_roots:
        raise ContractError("not_materialized policy contains future sealed artifacts")

    checked: list[dict[str, Any]] = []
    declared_paths: dict[Path, list[Mapping[str, Any]]] = defaultdict(list)
    for name, row in sorted(spec["input_receipts"].items()):
        data_class = str(row["data_class"])
        split = str(row["split"]).lower()
        scope = str(row["document_scope"])
        path, digest = _verified_declared_path(row, name)
        declared_paths[path].append(row)
        if data_class in forbidden_classes:
            raise ContractError(f"{name}: forbidden data class {data_class!r}")
        if split in {"test", "sealed", "sealed_test", "final_test"}:
            raise ContractError(f"{name}: final-test split is forbidden")
        if scope not in allowed_scopes:
            raise ContractError(f"{name}: unapproved document scope {scope!r}")
        lowered = str(path).lower()
        if any(token and token in lowered for token in forbidden_tokens):
            raise ContractError(f"{name}: path contains a sealed-test token")
        if any(_path_is_within(path, root) for root in sealed_roots):
            raise ContractError(f"{name}: path resolves under a sealed root")
        if digest in sealed_hashes:
            raise ContractError(f"{name}: artifact hash belongs to sealed test")
        contains_labels = bool(row.get("contains_labels", False))
        if contains_labels and split not in {"train", "development", "validation"}:
            raise ContractError(f"{name}: labels are outside an approved dev split")
        row_ids: set[str] = set()
        ids_path = row.get("document_ids_path")
        if ids_path:
            ids_path = Path(str(ids_path))
            ids_hash = _require_sha256(
                row.get("document_ids_sha256"),
                f"input_receipts.{name}.document_ids_sha256",
            )
            if sha256_file(ids_path) != ids_hash:
                raise ContractError(f"{name}: document inventory hash changed")
            row_ids = _load_document_ids(ids_path)
        inline_ids = row.get("document_ids", ())
        if inline_ids:
            if not isinstance(inline_ids, list) or any(
                not isinstance(value, str) for value in inline_ids
            ):
                raise ContractError(f"{name}: invalid inline document ids")
            row_ids.update(inline_ids)
        overlap = row_ids & sealed_ids
        if overlap:
            raise ContractError(
                f"{name}: contains {len(overlap)} sealed document identities"
            )
        checked.append(
            {
                "name": name,
                "path": str(path),
                "sha256": digest,
                "document_scope": scope,
                "document_ids_checked": len(row_ids),
            }
        )
    _reconcile_runner_paths(spec, declared_paths)
    lineage = verify_parent_lineage(spec)
    return {
        "status": "passed",
        "policy_sha256": policy_sha256,
        "sealed_test_status": status,
        "checked_inputs": checked,
        "parent_lineage": lineage,
    }


class CandidateStore:
    """Create and finalize candidate directories without an overwrite path."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def create(
        self, spec: Mapping[str, Any], leakage_policy: Mapping[str, Any]
    ) -> Path:
        validate_candidate_spec(spec)
        leakage = enforce_leakage_barrier(spec, leakage_policy)
        self.root.mkdir(parents=True, exist_ok=True)
        candidate_dir = self.root / str(spec["candidate_id"])
        candidate_dir.mkdir()
        write_json_exclusive(candidate_dir / "spec.json", spec)
        write_json_exclusive(candidate_dir / "leakage.json", leakage)
        write_json_exclusive(
            candidate_dir / "state.json",
            {"schema_version": "bibliography-evolution-state-v1", "state": "created"},
        )
        return candidate_dir

    def finalize(self, candidate_dir: Path, result: Mapping[str, Any]) -> Path:
        candidate_dir = candidate_dir.resolve()
        if not _path_is_within(candidate_dir, self.root):
            raise ContractError("candidate directory escapes the store")
        if (candidate_dir / "receipt.json").exists():
            raise FileExistsError(candidate_dir / "receipt.json")
        spec = load_json(candidate_dir / "spec.json")
        validate_candidate_spec(spec)
        leakage = load_json(candidate_dir / "leakage.json")
        if leakage.get("status") != "passed":
            raise ContractError("candidate did not pass leakage checks")
        checked = {
            str(row.get("name")): row
            for row in leakage.get("checked_inputs", ())
            if isinstance(row, Mapping)
        }
        if set(checked) != set(spec["input_receipts"]):
            raise ContractError("candidate leakage inventory is incomplete at finalization")
        for name, row in sorted(spec["input_receipts"].items()):
            path, digest = _verified_declared_path(row, name)
            if (
                Path(str(checked[name].get("path", ""))).resolve() != path
                or checked[name].get("sha256") != digest
            ):
                raise ContractError(f"candidate input drifted during execution: {name}")
        verify_parent_lineage(spec)
        _validate_result(result, candidate_dir)
        receipt = {
            **json.loads(json.dumps(result)),
            "schema_version": RECEIPT_SCHEMA,
            "candidate_id": spec["candidate_id"],
            "spec_sha256": sha256_file(candidate_dir / "spec.json"),
            "leakage": leakage,
        }
        receipt_path = candidate_dir / "receipt.json"
        write_json_exclusive(receipt_path, receipt)
        # There is deliberately no framework path that reopens these files for
        # writing.  Read-only modes also catch accidental in-place rewrites by
        # ordinary follow-up jobs while leaving the parent store usable.
        for path in candidate_dir.rglob("*"):
            if path.is_file() and not path.is_symlink():
                path.chmod(0o444)
        return receipt_path


def _validate_artifact(
    row: Mapping[str, Any], candidate_dir: Path, field: str
) -> None:
    path = Path(str(row.get("path", "")))
    if not path.is_absolute():
        path = candidate_dir / path
    if not _path_is_within(path, candidate_dir):
        raise ContractError(f"{field}: artifact escapes candidate directory")
    if not path.is_file():
        raise ContractError(f"{field}: artifact is missing")
    expected = _require_sha256(row.get("sha256"), f"{field}.sha256")
    if sha256_file(path) != expected:
        raise ContractError(f"{field}: artifact hash changed")


def _validate_result(result: Mapping[str, Any], candidate_dir: Path) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ContractError("unsupported candidate result schema")
    required = {
        "status",
        "all_rows",
        "artifacts",
        "predictions",
        "metrics",
        "metrics_by_source",
        "paired_deltas",
        "runtime",
        "job",
        "tests",
        "selection",
        "rejection",
    }
    missing = required - set(result)
    if missing:
        raise ContractError(f"candidate result misses fields: {sorted(missing)}")
    if result["status"] not in {"passed", "rejected", "failed"}:
        raise ContractError("invalid candidate status")
    if not isinstance(result["all_rows"], Mapping):
        raise ContractError("all_rows must be a hashed artifact")
    _validate_artifact(result["all_rows"], candidate_dir, "all_rows")
    artifacts = result["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ContractError("artifacts must be a mapping")
    for name, row in artifacts.items():
        if not isinstance(row, Mapping):
            raise ContractError(f"artifacts.{name} is invalid")
        _validate_artifact(row, candidate_dir, f"artifacts.{name}")
    predictions = result["predictions"]
    if not isinstance(predictions, Mapping):
        raise ContractError("predictions must be a mapping")
    for name, row in predictions.items():
        if not isinstance(row, Mapping):
            raise ContractError(f"predictions.{name} is invalid")
        _validate_artifact(row, candidate_dir, f"predictions.{name}")
    metrics = result["metrics"]
    if not isinstance(metrics, Mapping):
        raise ContractError("metrics must be a mapping")
    for objective in OBJECTIVES:
        value = metrics.get(objective)
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise ContractError(f"invalid objective {objective}")
    if int(metrics.get("document_count", -1)) != 268:
        raise ContractError("headline metrics must cover the fixed 268-document dev set")
    by_source = result["metrics_by_source"]
    if not isinstance(by_source, Mapping) or sum(
        int(row.get("document_count", 0)) for row in by_source.values()
    ) != 268:
        raise ContractError("by-source metrics do not reconcile to 268 documents")
    if not isinstance(result["tests"], Mapping) or result["tests"].get("status") != "passed":
        raise ContractError("candidate tests did not pass")
    paired = result["paired_deltas"]
    if (
        not isinstance(paired, Mapping)
        or paired.get("schema_version")
        != "bibliography-evolution-paired-work-bootstrap-v1"
        or int(paired.get("work_count", 0)) <= 0
        or not _HEX64.fullmatch(str(paired.get("candidate_rows_sha256", "")))
        or not _HEX64.fullmatch(str(paired.get("baseline_rows_sha256", "")))
    ):
        raise ContractError("candidate paired-work bootstrap is missing or unbound")
    selection = result["selection"]
    if not isinstance(selection, Mapping) or not isinstance(selection.get("acceptance"), Mapping):
        raise ContractError("candidate acceptance gates were not evaluated")
    if bool(selection.get("eligible_for_pareto")) and not bool(selection["acceptance"].get("passed")):
        raise ContractError("Pareto eligibility bypasses failed machine acceptance gates")


def verify_finalized_receipt(receipt_path: Path) -> dict[str, Any]:
    """Revalidate a candidate and all bytes immediately before selection/freeze."""

    receipt_path = Path(receipt_path).resolve()
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ContractError(f"candidate receipt is missing or a symlink: {receipt_path}")
    receipt = load_json(receipt_path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("status") != "passed":
        raise ContractError("candidate receipt is not a passed finalized receipt")
    candidate_dir = receipt_path.parent
    spec_path = candidate_dir / "spec.json"
    if not spec_path.is_file() or spec_path.is_symlink():
        raise ContractError("candidate spec is missing or a symlink")
    if sha256_file(spec_path) != receipt.get("spec_sha256"):
        raise ContractError("candidate spec bytes changed")
    spec = load_json(spec_path)
    validate_candidate_spec(spec)
    if receipt.get("candidate_id") != spec.get("candidate_id"):
        raise ContractError("candidate receipt/spec identity differs")
    result_view = dict(receipt)
    result_view["schema_version"] = RESULT_SCHEMA
    _validate_result(result_view, candidate_dir)

    leakage = receipt.get("leakage")
    if not isinstance(leakage, Mapping) or leakage.get("status") != "passed":
        raise ContractError("candidate leakage attestation is not passed")
    checked = {
        str(row.get("name")): row
        for row in leakage.get("checked_inputs", ())
        if isinstance(row, Mapping)
    }
    if set(checked) != set(spec["input_receipts"]):
        raise ContractError("leakage attestation input inventory is incomplete")
    verified_inputs = []
    for name, row in sorted(spec["input_receipts"].items()):
        path, digest = _verified_declared_path(row, name)
        prior = checked[name]
        if Path(str(prior.get("path", ""))).resolve() != path or prior.get("sha256") != digest:
            raise ContractError(f"leakage input attestation drifted: {name}")
        if str(row.get("split", "")).lower() in {"test", "sealed_test", "final_test"}:
            raise ContractError(f"candidate input became a final-test split: {name}")
        verified_inputs.append({"name": name, "path": str(path), "sha256": digest})
    lineage = verify_parent_lineage(spec)
    inventory = _receipt_artifact_inventory(receipt, candidate_dir)
    return {
        "status": "passed",
        "candidate_id": spec["candidate_id"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "spec_path": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "input_inventory": verified_inputs,
        "parent_lineage": lineage,
        "artifact_inventory": [
            {"path": str(path), "sha256": digest}
            for path, digest in sorted(inventory.items(), key=lambda row: str(row[0]))
        ],
    }


def _objective_vector(receipt: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = receipt["metrics"]
    values = tuple(float(metrics[name]) for name in OBJECTIVES)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ContractError("candidate has an invalid Pareto objective")
    return values


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    if len(left) != len(right):
        raise ContractError("objective vectors differ in length")
    return all(a <= b for a, b in zip(left, right, strict=True)) and any(
        a < b for a, b in zip(left, right, strict=True)
    )


def build_registry(receipt_paths: Iterable[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted((Path(value).resolve() for value in receipt_paths), key=str):
        receipt = load_json(path)
        if receipt.get("schema_version") != RECEIPT_SCHEMA:
            raise ContractError(f"unsupported receipt: {path}")
        candidate_id = str(receipt.get("candidate_id", ""))
        if candidate_id in seen:
            raise ContractError(f"duplicate candidate receipt: {candidate_id}")
        seen.add(candidate_id)
        spec = load_json(path.parent / "spec.json")
        validate_candidate_spec(spec)
        if spec["candidate_id"] != candidate_id:
            raise ContractError("receipt/spec candidate mismatch")
        reasons: list[str] = []
        if receipt.get("status") != "passed":
            reasons.append(f"status:{receipt.get('status')}")
        if receipt.get("leakage", {}).get("status") != "passed":
            reasons.append("leakage_not_passed")
        if receipt.get("tests", {}).get("status") != "passed":
            reasons.append("tests_not_passed")
        metrics = receipt.get("metrics", {})
        if int(metrics.get("document_count", -1)) != 268:
            reasons.append("wrong_headline_document_count")
        selection = receipt.get("selection", {})
        if not selection.get("eligible_for_pareto", False):
            reasons.append("candidate_selection_rejected")
        vector = None if reasons else _objective_vector(receipt)
        rows.append(
            {
                "candidate_id": candidate_id,
                "generation": spec["generation"],
                "changed_component": spec["changed_component"],
                "parent_candidate_ids": spec["parent_candidate_ids"],
                "receipt_path": str(path),
                "receipt_sha256": sha256_file(path),
                "objective_vector": dict(zip(OBJECTIVES, vector, strict=True)) if vector else None,
                "eligible": not reasons,
                "rejection_reasons": reasons,
                "pareto": False,
                "dominated_by": [],
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    # Collapse exact objective ties deterministically: prefer the earlier
    # generation, then the lexical immutable candidate id.
    tied: dict[tuple[float, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        tied[tuple(row["objective_vector"][name] for name in OBJECTIVES)].append(row)
    survivors: list[dict[str, Any]] = []
    for group in tied.values():
        group.sort(key=lambda row: (int(row["generation"][1:]), row["candidate_id"]))
        survivors.append(group[0])
        for duplicate in group[1:]:
            duplicate["dominated_by"].append(group[0]["candidate_id"])
            duplicate["rejection_reasons"].append("exact_objective_tie")
    for row in survivors:
        vector = tuple(row["objective_vector"][name] for name in OBJECTIVES)
        for other in survivors:
            if other is row:
                continue
            other_vector = tuple(other["objective_vector"][name] for name in OBJECTIVES)
            if dominates(other_vector, vector):
                row["dominated_by"].append(other["candidate_id"])
        row["dominated_by"].sort()
        row["pareto"] = not row["dominated_by"]
    rows.sort(key=lambda row: (int(row["generation"][1:]), row["candidate_id"]))
    pareto_ids = [row["candidate_id"] for row in rows if row["pareto"]]
    return {
        "schema_version": REGISTRY_SCHEMA,
        "objective_order": list(OBJECTIVES),
        "objective_direction": "minimize",
        "headline_document_count": 268,
        "tie_breaker": "earlier_generation_then_candidate_id",
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "pareto_candidate_ids": pareto_ids,
        "candidates": rows,
    }


def _aggregate_work_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    token_fp = sum(float(row["token_fp"]) for row in rows)
    token_fn = sum(float(row["token_fn"]) for row in rows)
    zero_blocks = sum(float(row["spurious_zero_blocks"]) for row in rows)
    zero_docs = sum(float(row["zero_doc_count"]) for row in rows)
    boundary_sum = sum(float(row["boundary_error_sum"]) for row in rows)
    boundary_count = sum(float(row["boundary_match_count"]) for row in rows)
    return (
        token_fp,
        token_fn,
        zero_blocks / zero_docs if zero_docs else 0.0,
        boundary_sum / boundary_count if boundary_count else 0.0,
    )


def paired_work_bootstrap(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = 2000,
    seed: int = 20260718,
) -> dict[str, Any]:
    """Source-stratified paired bootstrap over indivisible work identities."""

    def index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            work_id = str(row.get("work_id", ""))
            if not work_id or work_id in result:
                raise ContractError("work rows need unique non-empty work_id values")
            result[work_id] = row
        return result

    candidate = index(candidate_rows)
    baseline = index(baseline_rows)
    if set(candidate) != set(baseline):
        raise ContractError("paired bootstrap work inventories differ")
    by_source: dict[str, list[str]] = defaultdict(list)
    for work_id in sorted(candidate):
        if candidate[work_id].get("source") != baseline[work_id].get("source"):
            raise ContractError("paired bootstrap source assignments differ")
        by_source[str(candidate[work_id]["source"])].append(work_id)
    if iterations <= 0:
        raise ContractError("bootstrap iterations must be positive")
    rng = random.Random(seed)
    samples = [[] for _ in OBJECTIVES]
    for _ in range(iterations):
        sampled_ids: list[str] = []
        for source in sorted(by_source):
            group = by_source[source]
            sampled_ids.extend(rng.choice(group) for _ in range(len(group)))
        left = _aggregate_work_rows([candidate[value] for value in sampled_ids])
        right = _aggregate_work_rows([baseline[value] for value in sampled_ids])
        for index_value, (a, b) in enumerate(zip(left, right, strict=True)):
            samples[index_value].append(a - b)
    point_left = _aggregate_work_rows(list(candidate.values()))
    point_right = _aggregate_work_rows(list(baseline.values()))

    def percentile(values: Sequence[float], probability: float) -> float:
        ordered = sorted(values)
        position = probability * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "schema_version": "bibliography-evolution-paired-work-bootstrap-v1",
        "iterations": iterations,
        "seed": seed,
        "work_count": len(candidate),
        "source_work_counts": {key: len(value) for key, value in sorted(by_source.items())},
        "deltas_candidate_minus_baseline": {
            name: {
                "point": left - right,
                "ci95": [percentile(samples[index_value], 0.025), percentile(samples[index_value], 0.975)],
                "probability_improved": sum(value < 0 for value in samples[index_value]) / iterations,
            }
            for index_value, (name, left, right) in enumerate(
                zip(OBJECTIVES, point_left, point_right, strict=True)
            )
        },
    }


def verify_g0(
    lock: Mapping[str, Any], *, root: Path, replay_prediction: Path
) -> dict[str, Any]:
    if lock.get("schema_version") != BASELINE_SCHEMA:
        raise ContractError("unsupported baseline lock")
    root = root.resolve()
    if str(root) != str(lock.get("authoritative_root")):
        raise ContractError("baseline root differs from the lock")
    checked_receipts = []
    for relative, expected in sorted(lock["receipt_sha256"].items()):
        path = root / relative
        actual = sha256_file(path)
        if actual != _require_sha256(expected, f"receipt_sha256.{relative}"):
            raise ContractError(f"baseline receipt changed: {relative}")
        checked_receipts.append({"path": relative, "sha256": actual})
    checked_artifacts = []
    for relative, expected in sorted(lock["artifact_sha256"].items()):
        path = root / relative
        actual = sha256_file(path)
        if actual != _require_sha256(expected, f"artifact_sha256.{relative}"):
            raise ContractError(f"baseline artifact changed: {relative}")
        checked_artifacts.append({"path": relative, "sha256": actual})
    expected_prediction = _require_sha256(
        lock["g0_replay"]["prediction_sha256"], "g0_replay.prediction_sha256"
    )
    replay_sha = sha256_file(replay_prediction.resolve())
    if replay_sha != expected_prediction:
        raise ContractError("G0 replay prediction is not byte-identical")
    return {
        "schema_version": "bibliography-evolution-g0-verification-v1",
        "status": "passed_byte_identical",
        "baseline_lock_sha256": hashlib.sha256(canonical_json_bytes(lock)).hexdigest(),
        "checked_receipts": checked_receipts,
        "checked_artifacts": checked_artifacts,
        "replay_prediction": {"path": str(replay_prediction.resolve()), "sha256": replay_sha},
        "decoder_config": lock["decoder_config"],
        "headline_metrics": lock["headline_metrics_268"],
    }


def expand_template(
    template: Mapping[str, Any], bindings: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Expand one predeclared grid into immutable scalar candidate specs."""

    bound = _bind(template, bindings)
    grid = bound.get("sweep_grid", {})
    if not isinstance(grid, Mapping):
        raise ContractError("template sweep_grid must be a mapping")
    names = sorted(grid)
    values = [grid[name] for name in names]
    if any(not isinstance(row, list) or not row for row in values):
        raise ContractError("template sweep dimensions must be non-empty lists")
    result = []
    for combination in itertools.product(*values):
        point = dict(zip(names, combination, strict=True))
        spec = _bind(bound, point)
        spec["sweep_grid"] = grid
        spec["sweep_point"] = point
        spec["fixed_modules"] = list(FIXED_MODULE_ORDER)
        # JSON bindings retain numeric types in the scientific spec, while CLI
        # argv is intentionally a shell-free vector of strings.
        spec["runner"]["argv"] = [str(value) for value in spec["runner"]["argv"]]
        result.append(with_candidate_id(spec))
    return result


def _bind(value: Any, bindings: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        match = re.fullmatch(r"\$\{([^}]+)\}", value)
        if match and match.group(1) in bindings:
            return json.loads(json.dumps(bindings[match.group(1)]))
        result = value
        for key, replacement in bindings.items():
            result = result.replace(f"${{{key}}}", str(replacement))
        return result
    if isinstance(value, list):
        return [_bind(row, bindings) for row in value]
    if isinstance(value, Mapping):
        return {key: _bind(row, bindings) for key, row in value.items()}
    return value
