#!/usr/bin/env python3
"""Prediction-only bridge from frozen evolution candidates to sealed documents.

This module deliberately has no labels argument.  It may run only after the
annotation lane has emitted ``FROZEN.receipt.json``, but it verifies only the
sealed document bytes and the terminal receipt.  It materializes a fresh
unlabelled feature table, applies the frozen line/signal models, and recursively
replays every Pareto candidate from its immutable spec and parent lineage.

The resulting receipt is a proof object, not merely a collection of arrays:
every prediction names its candidate spec/receipt, parents, algorithm and model
artifact hashes.  The final evaluator revalidates this proof before opening the
separate label file.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.metadata
import io
import json
import pickle
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_deterministic_roles import ROLE_NAMES, _analyze_document
from .bibliography_entry_blocks import BlockConfig
from .bibliography_entry_dataset import (
    FEATURE_NAMES,
    HEADER_EXACT,
    HEADER_NONE,
    SUBHEADER_EXACT,
)
from .bibliography_entry_models import PINNED_SKLEARN_VERSION, Table, load_table
from .bibliography_evolution_composition import (
    combine_parent_barriers,
    enforce_combined_barriers,
)
from .bibliography_evolution_contract import (
    ContractError,
    atomic_write_bytes_exclusive,
    canonical_json_bytes,
    load_json,
    sha256_directory,
    sha256_file,
    verify_finalized_receipt,
    write_json_exclusive,
)
from .bibliography_evolution_headers import ROLE_TO_ID
from .bibliography_evolution_postprocess import (
    POSTPROCESS_ORDER,
    REFERENCE_PARAMETERS,
    _postprocess_document,
)
from .bibliography_signal_block_decode import decode_signal_blocks
from .bibliography_signal_tcn import FEATURE_NAMES as SIGNAL_FEATURE_NAMES
from .bibliography_signal_tcn import build_signal_features
from .bibliography_signal_validation import _ensemble_probability
from .bibliography_scope_rules import (
    auxiliary_scope_mask,
    is_exact_non_bibliography_scope_heading,
)
from .bibliography_v2 import extract_bibliography_features
from .deterministic_structure import BibRole, analyze_bib_line


INFERENCE_SCHEMA = "bibliography-evolution-sealed-inference-v1"
TABLE_RECEIPT_SCHEMA = "bibliography-evolution-sealed-feature-table-v1"
DERIVATION_SCHEMA = "bibliography-evolution-sealed-candidate-derivation-v1"
EXPECTED_SOURCES = {"greek_phd": 50, "kallipos": 50, "openarchives": 50}
HEX64 = frozenset("0123456789abcdef")
RUNTIME_CODE_FILES = (
    "bibliography_evolution.py",
    "bibliography_evolution_sealed_inference.py",
    "bibliography_evolution_contract.py",
    "bibliography_evolution_composition.py",
    "bibliography_evolution_headers.py",
    "bibliography_evolution_postprocess.py",
    "bibliography_heading_deployment.py",
    "bibliography_role_experts.py",
    "bibliography_role_features.py",
    "bibliography_entry_blocks.py",
    "bibliography_entry_dataset.py",
    "bibliography_entry_models.py",
    "bibliography_signal_block_decode.py",
    "bibliography_signal_tcn.py",
    "bibliography_signal_validation.py",
    "bibliography_deterministic_roles.py",
    "bibliography_scope_rules.py",
    "bibliography_v2.py",
    "deterministic_structure.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _regular_file(path: Path | str, label: str) -> Path:
    raw = Path(path).expanduser()
    _require(raw.is_file() and not raw.is_symlink(), f"{label} is missing or a symlink")
    return raw.resolve()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def runtime_code_inventory() -> dict[str, Any]:
    """Bind all package sources and replay-critical dependency versions."""

    root = Path(__file__).resolve().parent
    git_root = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "-C", git_root, "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", git_root, "status", "--porcelain", "--untracked-files=all"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    _require(len(commit) == 40 and set(commit) <= HEX64, "runtime Git commit is invalid")
    _require(not dirty.strip(), "sealed runtime Git checkout is not clean")
    files = []
    for path in sorted(root.glob("*.py"), key=lambda value: value.name):
        _require(
            path.is_file() and not path.is_symlink(),
            f"runtime package source is not a regular file: {path.name}",
        )
        files.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    _require(bool(files), "sealed runtime package source inventory is empty")
    environment = {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "numpy": importlib.metadata.version("numpy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "torch": importlib.metadata.version("torch"),
    }
    identity = {"git_commit": commit, "files": files, "environment": environment}
    return {
        "schema_version": "bibliography-evolution-sealed-runtime-code-v2",
        "git_commit": commit,
        "git_clean": True,
        "files": files,
        "environment": environment,
        "inventory_sha256": hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ContractError(f"{path}:{number}: expected an object")
            rows.append(value)
    return rows


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_bytes_exclusive(
        path, b"".join(canonical_json_bytes(row) for row in rows)
    )


def _save_array(path: Path, value: np.ndarray) -> None:
    handle = io.BytesIO()
    np.save(handle, value, allow_pickle=False)
    atomic_write_bytes_exclusive(path, handle.getvalue())


def _save_barriers(path: Path, barriers: Mapping[str, np.ndarray]) -> None:
    handle = io.BytesIO()
    np.savez(
        handle,
        hard_wall=np.asarray(barriers["hard_wall"], dtype=bool),
        upward_stop=np.asarray(barriers["upward_stop"], dtype=bool),
        downward_stop=np.asarray(barriers["downward_stop"], dtype=bool),
    )
    atomic_write_bytes_exclusive(path, handle.getvalue())


def _arg(spec: Mapping[str, Any], flag: str) -> str:
    argv = list(spec["runner"]["argv"])
    _require(argv.count(flag) == 1, f"candidate runner does not contain exactly one {flag}")
    index = argv.index(flag)
    _require(index + 1 < len(argv), f"candidate runner has no value for {flag}")
    return str(argv[index + 1])


def _verified_annotation_inputs(
    documents_path: Path, frozen_path: Path, *, expected_documents_sha256: str,
    expected_frozen_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify prediction-visible annotation bytes without touching labels."""

    documents_path = _regular_file(documents_path, "sealed documents")
    frozen_path = _regular_file(frozen_path, "annotation FROZEN receipt")
    _require(
        sha256_file(documents_path) == expected_documents_sha256,
        "sealed document bytes differ from the frozen Pareto manifest",
    )
    _require(
        sha256_file(frozen_path) == expected_frozen_sha256,
        "annotation FROZEN receipt differs from the frozen Pareto manifest",
    )
    frozen = load_json(frozen_path)
    hashes = frozen.get("sealed_hashes", {})
    _require(
        frozen.get("schema_version") == "bibliography-sealed-freeze-v1"
        and frozen.get("status") == "frozen_prediction_blind_test_set"
        and int(frozen.get("document_count", -1)) == 150
        and frozen.get("source_document_counts") == EXPECTED_SOURCES
        and hashes.get("documents_sha256") == expected_documents_sha256
        and _is_sha256(hashes.get("labels_sha256"))
        and _is_sha256(hashes.get("consensus_receipt_sha256")),
        "annotation FROZEN receipt does not bind the expected sealed set",
    )
    documents = _jsonl(documents_path)
    ids = [str(row.get("document_id", "")) for row in documents]
    counts = collections.Counter(str(row.get("source", "")) for row in documents)
    _require(
        len(documents) == 150
        and len(ids) == len(set(ids))
        and all(_is_sha256(value) for value in ids)
        and dict(counts) == EXPECTED_SOURCES,
        "sealed documents are not 150 unique IDs with exactly 50/source",
    )
    return documents, frozen


def materialize_unlabelled_table(
    documents: Sequence[Mapping[str, Any]], documents_path: Path, output: Path
) -> dict[str, Any]:
    """Build line features while making the absence of labels explicit."""

    output.mkdir()
    document_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    counts_rows: list[list[int]] = []
    headers: list[int] = []
    absolute: list[int] = []
    token_counts: list[int] = []
    char_lengths: list[int] = []
    document_indices: list[int] = []
    cursor = 0
    for document_index, document in enumerate(documents):
        lines = document.get("lines")
        _require(isinstance(lines, list) and bool(lines), "sealed document has no lines")
        local_seen: set[str] = set()
        previous_abs = -1
        for offset, line in enumerate(lines):
            _require(isinstance(line, Mapping), "sealed line is not an object")
            text = line.get("text")
            line_id = str(line.get("line_id", ""))
            abs_idx = line.get("abs_idx")
            _require(
                isinstance(text, str)
                and _is_sha256(line_id)
                and line_id not in local_seen
                and isinstance(abs_idx, int)
                and abs_idx >= 0
                and abs_idx > previous_abs,
                "sealed line text/identity/coordinate is invalid",
            )
            local_seen.add(line_id)
            previous_abs = abs_idx
            feature = extract_bibliography_features(text).as_dict()
            row = [int(feature[name]) for name in FEATURE_NAMES]
            _require(all(value >= 0 for value in row), "negative bibliography feature")
            evidence = analyze_bib_line(text, abs_idx)
            header = HEADER_NONE
            if evidence.role == BibRole.HEADING:
                header = HEADER_EXACT
            elif evidence.role == BibRole.SUBHEADING:
                header = SUBHEADER_EXACT
            counts_rows.append(row)
            headers.append(header)
            absolute.append(abs_idx)
            token_counts.append(int(feature["token_count"]))
            char_lengths.append(len(text))
            document_indices.append(document_index)
            line_rows.append(
                {
                    "row_index": cursor + offset,
                    "document_id": str(document["document_id"]),
                    "line_id": line_id,
                    "abs_idx": abs_idx,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
        length = len(lines)
        document_rows.append(
            {
                "document_id": str(document["document_id"]),
                "work_id": str(document.get("work_id") or document["document_id"]),
                "source": str(document["source"]),
                "split": "sealed_unlabelled",
                "coverage": "complete_present_physical_lines",
                "n_physical_lines": int(document.get("n_physical_lines", length)),
                "line_start": cursor,
                "line_end": cursor + length,
                "line_count": length,
                "block_count": 0,
                "fold": 0,
            }
        )
        cursor += length
    _require(cursor > 0, "sealed set contains no lines")
    arrays = {
        "counts": np.asarray(counts_rows, dtype=np.uint32),
        # These are operational placeholders only.  The final evaluator creates
        # a separate in-memory label view after the fuse has been committed.
        "targets": np.zeros(cursor, dtype=np.int8),
        "original_labels": np.zeros(cursor, dtype=np.uint8),
        "header_kinds": np.asarray(headers, dtype=np.uint8),
        "abs_indices": np.asarray(absolute, dtype=np.uint32),
        "token_counts": np.asarray(token_counts, dtype=np.uint32),
        "char_lengths": np.asarray(char_lengths, dtype=np.uint32),
        "block_indices": np.full(cursor, -1, dtype=np.int32),
        "document_indices": np.asarray(document_indices, dtype=np.uint32),
        "folds": np.zeros(cursor, dtype=np.uint8),
    }
    for name, value in arrays.items():
        _save_array(output / f"{name}.npy", value)
    _write_jsonl_exclusive(output / "documents.jsonl", document_rows)
    _write_jsonl_exclusive(output / "lines.jsonl", line_rows)
    manifest = {
        "schema_version": "bibliography-entry-feature-table-v1",
        "status": "passed_prediction_only_unlabelled_materialization",
        "split": "sealed_unlabelled",
        "input": {"path": str(documents_path.resolve()), "sha256": sha256_file(documents_path)},
        "document_count": len(document_rows),
        "source_document_counts": dict(collections.Counter(row["source"] for row in document_rows)),
        "work_count": len({row["work_id"] for row in document_rows}),
        "line_count": cursor,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "feature_dtype": "uint32",
        "target_semantics": "all-zero operational placeholders; not labels and never used for prediction",
        "n_folds": 1,
    }
    write_json_exclusive(output / "manifest.json", manifest)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    receipt = {**manifest, "schema_version": TABLE_RECEIPT_SCHEMA, "outputs": outputs}
    write_json_exclusive(output / "receipt.json", receipt)
    return receipt


def _table_for_prediction(root: Path) -> Table:
    return load_table(root, expected_split="sealed_unlabelled")


def _materialize_text_signals(
    table: Table, documents: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roles = np.zeros((len(table.targets), len(ROLE_NAMES)), dtype=np.uint8)
    scope = np.zeros(len(table.targets), dtype=bool)
    header_roles = np.zeros(len(table.targets), dtype=np.uint8)
    for document, metadata in zip(documents, table.documents, strict=True):
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        lines = list(document["lines"])
        document_id, local_roles, _counts = _analyze_document(
            (str(document["document_id"]), lines)
        )
        _require(document_id == metadata["document_id"], "role materialization identity drift")
        roles[start:end] = local_roles
        texts = [str(line["text"]) for line in lines]
        scope[start:end] = np.asarray(auxiliary_scope_mask(texts), dtype=bool)
        for offset, text in enumerate(texts):
            evidence = analyze_bib_line(text, int(lines[offset]["abs_idx"]))
            if is_exact_non_bibliography_scope_heading(text):
                header_roles[start + offset] = ROLE_TO_ID["NON_BIB_HEADER"]
            elif evidence.role == BibRole.HEADING:
                header_roles[start + offset] = ROLE_TO_ID["BIB_HEADER"]
            elif evidence.role == BibRole.SUBHEADING:
                header_roles[start + offset] = ROLE_TO_ID["BIB_SUBHEADER"]
    return roles, scope, header_roles


def _artifact_proof(path: Path) -> dict[str, Any]:
    path = _regular_file(path, "model artifact")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _path_is_bound_input(path: Path, spec: Mapping[str, Any]) -> bool:
    path = path.resolve()
    for row in spec["input_receipts"].values():
        root = Path(str(row["path"])).resolve()
        if root == path:
            return True
        if root.is_dir():
            try:
                path.relative_to(root)
                return True
            except ValueError:
                pass
    return False


def _line_probability(
    table: Table, baseline_root: Path, g0_spec: Mapping[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    import sklearn

    _require(
        sklearn.__version__ == PINNED_SKLEARN_VERSION,
        "sealed line inference sklearn version differs from training",
    )
    model_path = baseline_root / "validation_r3" / "final_line_model.pkl"
    _require(_path_is_bound_input(model_path, g0_spec), "final line model is not G0-bound")
    proof = _artifact_proof(model_path)
    with model_path.open("rb") as handle:
        model, transform = pickle.load(handle)
    probability = model.predict_proba(transform.apply(table.counts))[:, 1].astype(np.float32)
    _require(
        probability.shape == (len(table.targets),)
        and np.isfinite(probability).all()
        and np.all((probability >= 0) & (probability <= 1)),
        "frozen D1 model emitted invalid sealed probabilities",
    )
    return probability, [proof]


def _signal_probability(
    table: Table,
    line_probability: np.ndarray,
    roles: np.ndarray,
    model_root: Path,
    *,
    artifact_owner: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    report_path = model_root / "signal_tcn_oof_report.json"
    report_proof = _artifact_proof(report_path)
    report = load_json(report_path)
    architecture = dict(report.get("architecture", {}))
    checkpoint_architecture = {
        "hidden_dim": int(architecture["hidden_dim"]),
        "dilations": [int(value) for value in architecture["dilations"]],
        "dropout": float(architecture["dropout"]),
    }
    model_paths = sorted((model_root / "models").glob("fold*.pt"))
    _require(len(model_paths) == 5, "sealed TCN inference requires exactly five fold models")
    proofs = [report_proof, *(_artifact_proof(path) for path in model_paths)]
    if artifact_owner is not None:
        for proof in proofs:
            _require(
                artifact_owner.get(str(Path(proof["path"]))) == proof["sha256"],
                "candidate TCN inference artifact is not owned by its receipt",
            )
    features = build_signal_features(line_probability, roles, table.header_kinds)
    _require(
        tuple(report.get("feature_names", ())) == SIGNAL_FEATURE_NAMES,
        "signal feature contract differs from the frozen model",
    )
    probability = _ensemble_probability(
        table,
        features,
        model_paths,
        checkpoint_architecture,
        central_width=int(architecture["central_width"]),
        context=int(architecture["context_lines"]),
        batch_size=16,
    )
    return probability, proofs, architecture


@dataclass(frozen=True)
class CandidateNode:
    candidate_id: str
    receipt_path: Path
    receipt: Mapping[str, Any]
    spec: Mapping[str, Any]
    verification: Mapping[str, Any]


def load_candidate_graph(manifest: Mapping[str, Any]) -> dict[str, CandidateNode]:
    """Recursively verify Pareto candidates and every non-Pareto ancestor."""

    nodes: dict[str, CandidateNode] = {}
    visiting: set[str] = set()
    runtime_commit = str(
        manifest.get("sealed_inference_runtime_code", {}).get("git_commit", "")
    )
    _require(
        len(runtime_commit) == 40 and set(runtime_commit) <= HEX64,
        "frozen manifest has no valid runtime Git commit",
    )

    def visit(receipt_path: Path, expected_id: str) -> None:
        receipt_path = _regular_file(receipt_path, "candidate lineage receipt")
        _require(expected_id not in visiting, "candidate lineage contains a cycle")
        if expected_id in nodes:
            _require(nodes[expected_id].receipt_path == receipt_path, "candidate ID has two receipts")
            return
        visiting.add(expected_id)
        verification = verify_finalized_receipt(receipt_path)
        receipt = load_json(receipt_path)
        spec = load_json(receipt_path.parent / "spec.json")
        _require(
            verification["candidate_id"] == expected_id
            and receipt.get("candidate_id") == expected_id
            and spec.get("candidate_id") == expected_id
            and spec.get("code_commit") == runtime_commit,
            "candidate lineage identity mismatch",
        )
        nodes[expected_id] = CandidateNode(
            expected_id, receipt_path, receipt, spec, verification
        )
        parent_rows = {
            str(row.get("candidate_id")): Path(str(row["path"]))
            for row in spec["input_receipts"].values()
            if row.get("data_class") == "parent_candidate_receipt"
        }
        _require(
            set(parent_rows) == set(spec["parent_candidate_ids"]),
            "candidate parent receipts do not match its declared lineage",
        )
        for parent_id in spec["parent_candidate_ids"]:
            visit(parent_rows[parent_id], parent_id)
        visiting.remove(expected_id)

    indexed = {str(row["candidate_id"]): row for row in manifest["candidates"]}
    for candidate_id in manifest["candidate_ids"]:
        row = indexed[candidate_id]
        visit(Path(str(row["receipt_path"])), candidate_id)
    return nodes


def _topological_candidate_ids(nodes: Mapping[str, CandidateNode]) -> list[str]:
    ordered: list[str] = []
    completed: set[str] = set()

    def visit(candidate_id: str) -> None:
        if candidate_id in completed:
            return
        for parent_id in nodes[candidate_id].spec["parent_candidate_ids"]:
            _require(parent_id in nodes, "candidate graph omits a declared parent")
            visit(parent_id)
        completed.add(candidate_id)
        ordered.append(candidate_id)

    for candidate_id in sorted(nodes):
        visit(candidate_id)
    return ordered


def _barriers(scope: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "hard_wall": np.asarray(scope, dtype=bool).copy(),
        "upward_stop": np.zeros(len(scope), dtype=bool),
        "downward_stop": np.zeros(len(scope), dtype=bool),
    }


def _runner_block_config(spec: Mapping[str, Any]) -> BlockConfig:
    return BlockConfig(
        anchor_probability=float(_arg(spec, "--anchor-probability")),
        seed_length_limit=1,
        anchors_required=int(_arg(spec, "--anchors-required")),
        anchor_window=int(_arg(spec, "--anchor-window")),
        maximum_bridge_gap=int(_arg(spec, "--maximum-bridge-gap")),
        inside_probability=float(_arg(spec, "--inside-probability")),
        adjacent_expansion=int(_arg(spec, "--adjacent-expansion")),
        header_window=int(_arg(spec, "--header-window")),
    )


def _mapping_block_config(value: Mapping[str, Any]) -> BlockConfig:
    """Discard descriptive lock fields that are not BlockConfig arguments."""

    return BlockConfig(
        anchor_probability=float(value["anchor_probability"]),
        seed_length_limit=int(value.get("seed_length_limit", 1)),
        anchors_required=int(value["anchors_required"]),
        anchor_window=int(value["anchor_window"]),
        maximum_bridge_gap=int(value["maximum_bridge_gap"]),
        inside_probability=float(value["inside_probability"]),
        adjacent_expansion=int(value["adjacent_expansion"]),
        header_window=int(value.get("header_window", 2)),
    )


def _decode(
    table: Table,
    signal: np.ndarray,
    line: np.ndarray,
    scope: np.ndarray,
    config: BlockConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prediction, _ = decode_signal_blocks(
        table,
        signal,
        line,
        scope,
        config,
        qualified_documents=set(range(len(table.documents))),
        apply_veto=True,
    )
    return prediction, _barriers(scope)


def _postprocess(
    table: Table,
    parent_prediction: np.ndarray,
    parent_barriers: Mapping[str, np.ndarray],
    signal: np.ndarray,
    header_roles: np.ndarray,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    operation = _arg(spec, "--operation")
    threshold = float(_arg(spec, "--threshold"))
    max_lines = int(_arg(spec, "--max-lines"))
    barriers = {name: np.asarray(value, dtype=bool).copy() for name, value in parent_barriers.items()}
    if operation == "header_controller":
        from .bibliography_evolution_headers import predecoder_walls

        upward, downward = predecoder_walls(header_roles)
        barriers["upward_stop"] |= upward
        barriers["downward_stop"] |= downward
        barriers["hard_wall"] |= downward
        stages = (operation,)
    else:
        stages = POSTPROCESS_ORDER
    current = parent_prediction.astype(bool, copy=True)
    for stage in stages:
        parameters = dict(REFERENCE_PARAMETERS.get(stage, {}))
        if stage == operation:
            parameters = {"threshold": threshold, "max_lines": max_lines}
        next_prediction = np.zeros(len(current), dtype=bool)
        for document in table.documents:
            start, end = int(document["line_start"]), int(document["line_end"])
            next_prediction[start:end] = _postprocess_document(
                current[start:end],
                signal[start:end],
                table.abs_indices[start:end],
                barriers["hard_wall"][start:end],
                barriers["upward_stop"][start:end],
                barriers["downward_stop"][start:end],
                header_roles[start:end],
                operation=stage,
                threshold=float(parameters["threshold"]),
                max_lines=int(parameters["max_lines"]),
            )
        current = next_prediction
    return current, barriers


def _owned_artifacts(node: CandidateNode) -> dict[str, str]:
    return {
        str(Path(row["path"]).resolve()): str(row["sha256"])
        for row in node.verification["artifact_inventory"]
    }


def _composition_parent_ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    """Return the exact parents bound to G5's left and right runner inputs."""

    parents = set(str(value) for value in spec["parent_candidate_ids"])

    def owner(*flags: str) -> str:
        observed: list[str] = []
        for flag in flags:
            path = Path(_arg(spec, flag)).resolve()
            owners = {
                str(row.get("parent_candidate_id", ""))
                for row in spec["input_receipts"].values()
                if Path(str(row.get("path", ""))).resolve() == path
                and row.get("parent_candidate_id") is not None
            }
            _require(
                len(owners) == 1 and next(iter(owners)) in parents,
                f"{flag} is not bound to exactly one declared parent",
            )
            observed.append(next(iter(owners)))
        _require(len(set(observed)) == 1, "prediction/barrier orientation differs")
        return observed[0]

    left = owner("--left-prediction", "--left-barrier-artifact")
    right = owner("--right-prediction", "--right-barrier-artifact")
    _require(
        left != right and len(parents) == 2 and {left, right} == parents,
        "G5 left/right artifacts do not bind the exact two parents",
    )
    return left, right


def run_inference(
    manifest: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    table_root: Path,
    output_root: Path,
    *,
    candidate_graph: Mapping[str, CandidateNode] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = _table_for_prediction(table_root)
    roles, scope, header_roles = _materialize_text_signals(table, documents)
    nodes = dict(candidate_graph) if candidate_graph is not None else load_candidate_graph(manifest)
    g0_nodes = [node for node in nodes.values() if node.spec["generation"] == "G0"]
    _require(len(g0_nodes) == 1, "sealed frontier must have exactly one G0 ancestor")
    g0 = g0_nodes[0]
    baseline_root = Path(_arg(g0.spec, "--authoritative-root")).resolve()
    lock_path = Path(_arg(g0.spec, "--lock")).resolve()
    _require(
        _path_is_bound_input(lock_path, g0.spec)
        and _path_is_bound_input(baseline_root, g0.spec),
        "G0 inference foundation is not bound by its candidate spec",
    )
    lock = load_json(lock_path)
    _require(
        lock.get("schema_version") == "bibliography-evolution-baseline-lock-v1"
        and Path(str(lock.get("authoritative_root"))).resolve() == baseline_root,
        "G0 baseline lock/root contract differs",
    )
    line, line_models = _line_probability(table, baseline_root, g0.spec)
    base_signal, base_signal_models, _ = _signal_probability(
        table, line, roles, baseline_root / "signal_tcn_r1"
    )
    for proof in base_signal_models:
        _require(
            _path_is_bound_input(Path(proof["path"]), g0.spec),
            "baseline signal model is not G0-bound",
        )
    common = {
        "line_probability_sha256": None,
        "line_model_artifacts": line_models,
        "baseline_signal_model_artifacts": base_signal_models,
        "baseline_lock": _artifact_proof(lock_path),
    }
    prediction_dir = output_root / "predictions"
    prediction_dir.mkdir()
    computed: dict[str, tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]] = {}

    def compute(candidate_id: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        if candidate_id in computed:
            return computed[candidate_id]
        node = nodes[candidate_id]
        spec = node.spec
        parents = [compute(parent_id) for parent_id in spec["parent_candidate_ids"]]
        generation = str(spec["generation"])
        component = str(spec["changed_component"])
        model_proofs: list[dict[str, Any]] = []
        composition_orientation: dict[str, str] | None = None
        if generation == "G0":
            prediction, barrier = _decode(
                table, base_signal, line, scope, _mapping_block_config(lock["decoder_config"])
            )
        elif generation == "G1":
            _require(len(parents) == 1, "G1 requires one parent")
            prediction, barrier = _decode(table, base_signal, line, scope, _runner_block_config(spec))
        elif generation in {"G2", "G3"}:
            _require(len(parents) == 1, f"{generation} requires one parent")
            local_header_roles = header_roles
            if generation == "G2" and component == "headers.role_controller":
                parameters = spec["changes"]["headers.role_controller"]["parameters"]
                backend = str(parameters.get("backend"))
                if backend == "learned_argmax":
                    from .bibliography_heading_deployment import (
                        deployment_proofs,
                        predict_documents,
                    )

                    deployment_root = node.receipt_path.parent / "backend" / "heading_deployment"
                    local_header_roles, _heading_probability, _heading_candidates = predict_documents(
                        deployment_root,
                        table,
                        documents,
                        line,
                        assignment_threshold=float(
                            parameters.get("heading_assignment_threshold", 0.5)
                        ),
                    )
                    model_proofs = deployment_proofs(deployment_root)
                    owned = _owned_artifacts(node)
                    for proof in model_proofs:
                        _require(
                            owned.get(str(Path(proof["path"]))) == proof["sha256"],
                            "learned G2 heading artifact is not candidate-owned",
                        )
                else:
                    _require(backend == "deterministic", "unsupported G2 heading backend")
            prediction, barrier = _postprocess(
                table, parents[0][0], parents[0][1], base_signal, local_header_roles, spec
            )
        elif generation == "G4":
            _require(len(parents) == 1 and component.startswith("signal."), "G4 lineage invalid")
            backend = node.receipt_path.parent / "backend"
            candidate_signal, model_proofs, _ = _signal_probability(
                table,
                line,
                roles,
                backend / "train",
                artifact_owner=_owned_artifacts(node),
            )
            report_path = backend / "validation" / "signal_validation_report.json"
            _require(
                _owned_artifacts(node).get(str(report_path.resolve())) == sha256_file(report_path),
                "G4 decoder report is not candidate-owned",
            )
            report = load_json(report_path)
            config = report["candidates"]["recall_first_anchored"]["train_oof_frozen_row"]["config"]
            prediction, barrier = _decode(
                table, candidate_signal, line, scope, BlockConfig(**config)
            )
            model_proofs.append(_artifact_proof(report_path))
        elif generation == "G5":
            _require(len(parents) == 2, "G5 requires two parents")
            left_id, right_id = _composition_parent_ids(spec)
            parent_by_id = {
                parent_id: value
                for parent_id, value in zip(
                    spec["parent_candidate_ids"], parents, strict=True
                )
            }
            left, right = parent_by_id[left_id], parent_by_id[right_id]
            combined = combine_parent_barriers(
                left[1], right[1], (len(table.targets),)
            )
            operation = _arg(spec, "--operation")
            if operation == "union":
                prediction = left[0] | right[0]
            elif operation == "intersection":
                prediction = left[0] & right[0]
            elif operation == "left_minus_right":
                prediction = left[0] & ~right[0]
            else:
                raise ContractError(f"unsupported frozen G5 operation: {operation}")
            prediction = enforce_combined_barriers(prediction, combined)
            barrier = combined
            composition_orientation = {
                "left_parent_id": left_id,
                "right_parent_id": right_id,
            }
        else:
            raise ContractError(f"unsupported candidate generation: {generation}")
        _require(
            prediction.shape == (len(table.targets),)
            and prediction.dtype == np.bool_
            and all(np.asarray(barrier[name]).shape == prediction.shape for name in barrier),
            "candidate inference output shape/dtype is invalid",
        )
        proof = {
            "schema_version": DERIVATION_SCHEMA,
            "candidate_id": candidate_id,
            "generation": generation,
            "changed_component": component,
            "candidate_receipt": {
                "path": str(node.receipt_path),
                "sha256": sha256_file(node.receipt_path),
            },
            "candidate_spec": {
                "path": str(node.receipt_path.parent / "spec.json"),
                "sha256": sha256_file(node.receipt_path.parent / "spec.json"),
            },
            "parent_candidate_ids": list(spec["parent_candidate_ids"]),
            "algorithm": {"module": str(spec["runner"]["module"]), "sweep_point": spec["sweep_point"]},
            "model_artifacts": model_proofs,
            "composition_orientation": composition_orientation,
        }
        computed[candidate_id] = prediction, barrier, proof
        return computed[candidate_id]

    for candidate_id in manifest["candidate_ids"]:
        compute(candidate_id)

    lineage_rows: list[dict[str, Any]] = []
    for candidate_id in _topological_candidate_ids(nodes):
        prediction, barrier, proof = computed[candidate_id]
        candidate_root = prediction_dir / candidate_id
        candidate_root.mkdir()
        prediction_path = candidate_root / "prediction.npy"
        barrier_path = candidate_root / "combined_barriers.npz"
        _save_array(prediction_path, prediction)
        _save_barriers(barrier_path, barrier)
        proof["prediction_sha256"] = sha256_file(prediction_path)
        proof["parent_prediction_sha256"] = [
            computed[parent_id][2]["prediction_sha256"]
            for parent_id in proof["parent_candidate_ids"]
        ]
        proof["barrier_sha256"] = sha256_file(barrier_path)
        proof["line_count"] = len(prediction)
        proof_path = candidate_root / "derivation.json"
        write_json_exclusive(proof_path, proof)
        lineage_rows.append(
            {
                "candidate_id": candidate_id,
                "frontier": candidate_id in set(manifest["candidate_ids"]),
                "prediction": _artifact_proof(prediction_path),
                "barriers": _artifact_proof(barrier_path),
                "derivation": _artifact_proof(proof_path),
            }
        )
    indexed_lineage = {row["candidate_id"]: row for row in lineage_rows}
    rows = [indexed_lineage[candidate_id] for candidate_id in manifest["candidate_ids"]]
    common["derivation_graph"] = lineage_rows
    common["line_probability_sha256"] = hashlib.sha256(
        np.ascontiguousarray(line).tobytes()
    ).hexdigest()
    return rows, common


def prepare_inference(
    manifest_path: Path,
    documents_path: Path,
    frozen_path: Path,
    output_root: Path,
) -> Path:
    """Create immutable candidate predictions without opening labels."""

    from .bibliography_evolution import _verify_frozen_manifest_fresh

    manifest_path = _regular_file(manifest_path, "frozen Pareto manifest")
    manifest = _verify_frozen_manifest_fresh(manifest_path)
    documents, frozen = _verified_annotation_inputs(
        documents_path,
        frozen_path,
        expected_documents_sha256=str(manifest["sealed_documents"]["sha256"]),
        expected_frozen_sha256=str(manifest["sealed_freeze_receipt"]["sha256"]),
    )
    _require(
        output_root.is_absolute() and not output_root.exists() and not output_root.is_symlink(),
        "sealed inference output must be a new absolute path",
    )
    output_root.mkdir(parents=True)
    table_root = output_root / "feature_table"
    table_receipt = materialize_unlabelled_table(documents, documents_path, table_root)
    candidates, foundation = run_inference(manifest, documents, table_root, output_root)
    _require(
        [row["candidate_id"] for row in candidates] == manifest["candidate_ids"],
        "sealed inference candidate cardinality/order differs from the frontier",
    )
    receipt = {
        "schema_version": INFERENCE_SCHEMA,
        "status": "passed_predictions_before_label_access",
        "labels_opened": False,
        "frozen_manifest_id": manifest["frozen_manifest_id"],
        "frozen_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "annotation_frozen": {"path": str(frozen_path.resolve()), "sha256": sha256_file(frozen_path)},
        "sealed_documents": {"path": str(documents_path.resolve()), "sha256": sha256_file(documents_path)},
        "annotation_label_sha256_bound_but_not_opened": frozen["sealed_hashes"]["labels_sha256"],
        "source_document_counts": EXPECTED_SOURCES,
        "document_count": 150,
        "line_count": int(table_receipt["line_count"]),
        "feature_table": {
            "path": str(table_root.resolve()),
            "sha256": sha256_directory(table_root),
            "receipt_sha256": sha256_file(table_root / "receipt.json"),
        },
        "candidate_ids": list(manifest["candidate_ids"]),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "foundation": foundation,
        "runtime_code": runtime_code_inventory(),
        "bridge_code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }
    receipt_path = output_root / "receipt.json"
    write_json_exclusive(receipt_path, receipt)
    for path in output_root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o440)
    return receipt_path


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def verify_inference_receipt(
    manifest: Mapping[str, Any], receipt_path: Path
) -> dict[str, Any]:
    """Revalidate all prediction-visible bytes before the label fuse.

    This function still does not accept or discover a label path.
    """

    receipt_path = _regular_file(receipt_path, "sealed inference receipt")
    receipt = load_json(receipt_path)
    inference_root = receipt_path.parent
    candidate_ids = list(manifest["candidate_ids"])
    _require(
        receipt.get("schema_version") == INFERENCE_SCHEMA
        and receipt.get("status") == "passed_predictions_before_label_access"
        and receipt.get("labels_opened") is False
        and receipt.get("frozen_manifest_id") == manifest["frozen_manifest_id"]
        and receipt.get("candidate_ids") == candidate_ids
        and int(receipt.get("candidate_count", -1)) == len(candidate_ids)
        and int(receipt.get("document_count", -1)) == 150
        and receipt.get("source_document_counts") == EXPECTED_SOURCES,
        "sealed inference receipt does not match the frozen frontier",
    )
    bridge = receipt.get("bridge_code", {})
    _require(
        bridge.get("sha256") == sha256_file(Path(__file__)),
        "sealed inference bridge code differs from the prediction receipt",
    )
    _require(
        receipt.get("runtime_code") == manifest.get("sealed_inference_runtime_code")
        == runtime_code_inventory(),
        "sealed inference runtime source inventory differs from the frozen manifest",
    )
    documents_row = receipt.get("sealed_documents", {})
    _require(
        documents_row.get("sha256") == manifest["sealed_documents"]["sha256"],
        "sealed inference used different document bytes",
    )
    documents_path = _regular_file(
        str(documents_row.get("path", "")), "sealed inference document input"
    )
    _require(
        sha256_file(documents_path) == documents_row["sha256"],
        "sealed inference document input drifted",
    )
    feature_row = receipt.get("feature_table", {})
    raw_table_root = Path(str(feature_row.get("path", ""))).expanduser()
    _require(not raw_table_root.is_symlink(), "sealed inference feature table is a symlink")
    table_root = raw_table_root.resolve()
    _require(
        _within(table_root, inference_root)
        and table_root.is_dir()
        and sha256_directory(table_root) == feature_row.get("sha256")
        and sha256_file(table_root / "receipt.json") == feature_row.get("receipt_sha256"),
        "sealed inference feature table drifted",
    )
    table = _table_for_prediction(table_root)
    _require(
        len(table.documents) == 150
        and len(table.targets) == int(receipt.get("line_count", -1))
        and dict(collections.Counter(str(row["source"]) for row in table.documents))
        == EXPECTED_SOURCES,
        "sealed feature table cardinality/source balance differs",
    )
    documents = _jsonl(documents_path)
    table_lines = _jsonl(table_root / "lines.jsonl")
    expected_lines = [
        (
            str(document["document_id"]),
            str(line["line_id"]),
            int(line["abs_idx"]),
            hashlib.sha256(str(line["text"]).encode("utf-8")).hexdigest(),
        )
        for document in documents
        for line in document["lines"]
    ]
    observed_lines = [
        (
            str(row["document_id"]),
            str(row["line_id"]),
            int(row["abs_idx"]),
            str(row["text_sha256"]),
        )
        for row in table_lines
    ]
    _require(observed_lines == expected_lines, "feature table line inventory differs from sealed text")

    nodes = load_candidate_graph(manifest)
    lineage = receipt.get("foundation", {}).get("derivation_graph")
    _require(isinstance(lineage, list) and bool(lineage), "inference derivation graph is missing")
    lineage_by_id = {str(row.get("candidate_id")): row for row in lineage if isinstance(row, Mapping)}
    _require(
        len(lineage_by_id) == len(lineage)
        and set(candidate_ids) <= set(lineage_by_id)
        and set(lineage_by_id) == set(nodes),
        "inference derivation graph does not equal the candidate ancestry",
    )
    prediction_hashes: dict[str, str] = {}
    g0_nodes = [node for node in nodes.values() if node.spec["generation"] == "G0"]
    _require(len(g0_nodes) == 1, "inference proof does not have exactly one G0")
    g0 = g0_nodes[0]
    foundation = receipt.get("foundation", {})
    baseline_lock = foundation.get("baseline_lock", {})
    foundation_artifacts = [
        *foundation.get("line_model_artifacts", ()),
        *foundation.get("baseline_signal_model_artifacts", ()),
        baseline_lock,
    ]
    _require(bool(foundation_artifacts), "inference foundation artifact proof is missing")
    for artifact in foundation_artifacts:
        _require(isinstance(artifact, Mapping), "foundation artifact proof is malformed")
        path = _regular_file(
            str(artifact.get("path", "")), "inference foundation artifact"
        )
        _require(
            sha256_file(path) == artifact.get("sha256")
            and _path_is_bound_input(path, g0.spec),
            "inference foundation artifact is not byte-exact and G0-bound",
        )
    for candidate_id in _topological_candidate_ids(nodes):
        row = lineage_by_id[candidate_id]
        prediction_path = _regular_file(
            str(row.get("prediction", {}).get("path", "")), "candidate prediction"
        )
        barrier_path = _regular_file(
            str(row.get("barriers", {}).get("path", "")), "candidate barriers"
        )
        derivation_path = _regular_file(
            str(row.get("derivation", {}).get("path", "")), "candidate derivation"
        )
        _require(
            all(_within(path, inference_root)
                for path in (prediction_path, barrier_path, derivation_path)),
            "candidate inference artifact escapes or is missing",
        )
        for name, path in (
            ("prediction", prediction_path), ("barriers", barrier_path),
            ("derivation", derivation_path),
        ):
            _require(
                sha256_file(path) == row[name].get("sha256"),
                f"candidate {name} artifact hash drifted",
            )
        prediction = np.load(prediction_path, allow_pickle=False)
        barrier = np.load(barrier_path, allow_pickle=False)
        _require(
            prediction.shape == (len(table.targets),)
            and prediction.dtype == np.bool_
            and all(name in barrier and barrier[name].shape == prediction.shape
                    and barrier[name].dtype == np.bool_
                    for name in ("hard_wall", "upward_stop", "downward_stop")),
            "candidate prediction/barrier shape or dtype is invalid",
        )
        proof = load_json(derivation_path)
        node = nodes[candidate_id]
        spec = node.spec
        expected_parents = list(spec["parent_candidate_ids"])
        orientation_ids = (
            _composition_parent_ids(spec) if spec["generation"] == "G5" else None
        )
        expected_orientation = (
            {
                "left_parent_id": orientation_ids[0],
                "right_parent_id": orientation_ids[1],
            }
            if orientation_ids is not None
            else None
        )
        _require(
            proof.get("schema_version") == DERIVATION_SCHEMA
            and proof.get("candidate_id") == candidate_id
            and proof.get("generation") == spec["generation"]
            and proof.get("changed_component") == spec["changed_component"]
            and proof.get("parent_candidate_ids") == expected_parents
            and proof.get("parent_prediction_sha256")
            == [prediction_hashes[parent] for parent in expected_parents]
            and proof.get("prediction_sha256") == sha256_file(prediction_path)
            and proof.get("barrier_sha256") == sha256_file(barrier_path)
            and proof.get("candidate_receipt", {}).get("sha256") == sha256_file(node.receipt_path)
            and proof.get("candidate_spec", {}).get("sha256")
            == sha256_file(node.receipt_path.parent / "spec.json")
            and proof.get("algorithm")
            == {"module": spec["runner"]["module"], "sweep_point": spec["sweep_point"]}
            and proof.get("composition_orientation") == expected_orientation,
            "candidate derivation proof does not follow its frozen spec/parents",
        )
        model_artifacts = proof.get("model_artifacts")
        _require(isinstance(model_artifacts, list), "candidate model proof is malformed")
        owned = _owned_artifacts(node)
        for artifact in model_artifacts:
            path = _regular_file(
                str(artifact.get("path", "")), "candidate-specific model artifact"
            )
            _require(
                sha256_file(path) == artifact.get("sha256")
                and owned.get(str(path)) == artifact.get("sha256"),
                "candidate-specific model artifact is not receipt-owned",
            )
        learned_g2 = (
            spec["generation"] == "G2"
            and spec["changed_component"] == "headers.role_controller"
            and spec["changes"]["headers.role_controller"]["parameters"].get("backend")
            == "learned_argmax"
        )
        if spec["generation"] == "G4" or learned_g2:
            _require(
                bool(model_artifacts),
                "learned candidate prediction lacks candidate-owned model proof",
            )
        else:
            _require(
                not model_artifacts,
                "model-free candidate unexpectedly substitutes a model",
            )
        prediction_hashes[candidate_id] = sha256_file(prediction_path)

    frontier_rows = receipt.get("candidates")
    _require(
        isinstance(frontier_rows, list)
        and [row.get("candidate_id") for row in frontier_rows] == candidate_ids
        and all(
            row.get("prediction", {}).get("sha256") == prediction_hashes[row["candidate_id"]]
            and lineage_by_id[row["candidate_id"]] == row
            for row in frontier_rows
        ),
        "frontier prediction inventory is not the exact derivation subset",
    )

    # A self-declared derivation receipt is not proof by itself. Rebuild the
    # feature table directly from the frozen text, replay every candidate from
    # its model/spec bytes, and compare every ancestor output before labels are
    # even discoverable by the caller of this function.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="bibliography-sealed-replay-") as temporary:
        temporary_root = Path(temporary)
        rebuilt_table_root = temporary_root / "feature_table"
        materialize_unlabelled_table(documents, documents_path, rebuilt_table_root)
        _require(
            sha256_directory(rebuilt_table_root) == feature_row.get("sha256"),
            "sealed feature table is not a deterministic derivation of frozen text",
        )
        replay_root = temporary_root / "candidate_replay"
        replay_root.mkdir()
        _frontier_replay, replay_foundation = run_inference(
            manifest,
            documents,
            rebuilt_table_root,
            replay_root,
            candidate_graph=nodes,
        )
        for key in (
            "line_probability_sha256",
            "line_model_artifacts",
            "baseline_signal_model_artifacts",
            "baseline_lock",
        ):
            _require(
                foundation.get(key) == replay_foundation.get(key),
                f"sealed inference foundation replay differs: {key}",
            )
        replay_lineage = {
            str(row["candidate_id"]): row
            for row in replay_foundation["derivation_graph"]
        }
        _require(
            set(replay_lineage) == set(lineage_by_id),
            "replayed candidate ancestry differs from the prediction receipt",
        )
        for candidate_id in sorted(lineage_by_id):
            observed = lineage_by_id[candidate_id]
            replayed = replay_lineage[candidate_id]
            observed_prediction = np.load(
                Path(str(observed["prediction"]["path"])), allow_pickle=False
            )
            replayed_prediction = np.load(
                Path(str(replayed["prediction"]["path"])), allow_pickle=False
            )
            _require(
                np.array_equal(observed_prediction, replayed_prediction),
                f"candidate prediction is not reproduced by its frozen derivation: {candidate_id}",
            )
            with (
                np.load(
                    Path(str(observed["barriers"]["path"])), allow_pickle=False
                ) as observed_barriers,
                np.load(
                    Path(str(replayed["barriers"]["path"])), allow_pickle=False
                ) as replayed_barriers,
            ):
                _require(
                    all(
                        np.array_equal(observed_barriers[name], replayed_barriers[name])
                        for name in ("hard_wall", "upward_stop", "downward_stop")
                    ),
                    f"candidate barriers are not reproduced by their frozen derivation: {candidate_id}",
                )
    return {
        "status": "passed_prediction_only_candidate_derivation_and_replay_preflight",
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "feature_table": str(table_root),
        "feature_table_sha256": str(feature_row["sha256"]),
        "feature_table_receipt_sha256": str(feature_row["receipt_sha256"]),
        "line_count": len(table.targets),
        "candidate_ids": candidate_ids,
        "prediction_paths": {
            candidate_id: str(Path(lineage_by_id[candidate_id]["prediction"]["path"]).resolve())
            for candidate_id in candidate_ids
        },
        "prediction_sha256": {candidate_id: prediction_hashes[candidate_id] for candidate_id in candidate_ids},
        "independent_replay": True,
    }
