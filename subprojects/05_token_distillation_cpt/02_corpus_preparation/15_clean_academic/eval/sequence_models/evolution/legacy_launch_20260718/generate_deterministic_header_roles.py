#!/usr/bin/env python3
"""Generate deployable deterministic bibliography heading roles from text only.

The frozen validation arrays are equivalence references, never prediction
inputs.  The emitted vector is recomputed from ``text`` and ``abs_idx`` with
the audited deterministic analyzers, then required to match both historical
text-derived reference arrays byte-for-byte.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sequence_models.bibliography_deterministic_roles import (
    ROLE_NAMES as NEGATIVE_ROLE_NAMES,
    _analyze_document,
)
from sequence_models.deterministic_structure import BibRole, analyze_bib_line


SCHEMA_VERSION = "bibliography-deterministic-header-role-lineage-v1"
EXPECTED_COMMIT = "931a56d119b9ab44e79c23fa82a16cd2edf0c4b7"
HEADER_ROLES = ("NONE", "BIB_HEADER", "BIB_SUBHEADER", "NON_BIB_HEADER")
EXPECTED_LINE_COUNT = 259_067
EXPECTED_DOCUMENT_COUNT = 274
EXPECTED_HEADER_KINDS_SHA = "c8a22608d4060b9ee03b2c4c2286765759d594a39dc94f1285714c2d5cabb0ad"
EXPECTED_HEADER_RECEIPT_SHA = "689ecce5c7e9ede12feba6df26b0286d883f1b7e42422558fa3df605c0d6ccfd"
EXPECTED_NEGATIVE_ROLES_SHA = "fa220e5b3f41bbc75c8200731acd6cedd1fd99dab3d9594445f04d9f1592603c"
EXPECTED_NEGATIVE_RECEIPT_SHA = "444e4ad8a422d81c583dcfe01fd4c62bf3a08b4bab6931e771e0980ea0e365ee"
EXPECTED_RAW_INPUT_SHA = "f18ef6bf3061d932ae0aaeb2349392a2e590f2778e3205cf7cbcb5c79dffa7c0"
EXPECTED_REFERENCE_CODE = {
    (
        "e1463dbf29562ba8784438e945b349445977b6a8",
        "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/"
        "eval/sequence_models/bibliography_entry_dataset.py",
    ): "6a40d4f16c767e385fef1ab1fde64a43ebe88844959e77078f201daee04137a5",
    (
        "b49a359346a833cbc12c9c28eec186dd933a522b",
        "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/"
        "eval/sequence_models/bibliography_signal_validation.py",
    ): "9258b47a5f128f12e8cf1e692e079b217670d48fac04ab11a34998b0572fac9d",
    (
        "b49a359346a833cbc12c9c28eec186dd933a522b",
        "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/"
        "eval/sequence_models/bibliography_deterministic_roles.py",
    ): "b49532d224816a6150128bd9e05468484ee2f7ebfc26c17554bb2104a5d0168f",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def analyze_document(
    task: tuple[str, list[Mapping[str, Any]]]
) -> tuple[str, np.ndarray, np.ndarray, dict[str, int]]:
    """Return text-only header roles plus the complete deterministic negatives."""

    document_id, lines = task
    header_roles = np.zeros(len(lines), dtype=np.uint8)
    for offset, line in enumerate(lines):
        text = line.get("text")
        abs_idx = line.get("abs_idx")
        if not isinstance(text, str) or not isinstance(abs_idx, int) or abs_idx < 0:
            raise ValueError(f"{document_id}: malformed unlabeled line {offset}")
        evidence = analyze_bib_line(text, abs_idx)
        if evidence.role == BibRole.HEADING:
            header_roles[offset] = 1
        elif evidence.role == BibRole.SUBHEADING:
            header_roles[offset] = 2
    returned_id, negative_roles, negative_counts = _analyze_document(
        (document_id, lines)
    )
    if returned_id != document_id:
        raise ValueError("deterministic negative-role worker changed document id")
    return document_id, header_roles, negative_roles, negative_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path, required=True)
    parser.add_argument("--validation-table-dir", type=Path, required=True)
    parser.add_argument("--reference-negative-roles", type=Path, required=True)
    parser.add_argument("--reference-negative-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("role generation must run on a Slurm compute node")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    repo = args.repo_root.resolve()
    if git(repo, "rev-parse", "HEAD").decode().strip() != EXPECTED_COMMIT:
        raise RuntimeError("deployed checkout is not the audited commit")
    if git(repo, "status", "--porcelain", "--untracked-files=all").strip():
        raise RuntimeError("deployed checkout is dirty")

    reference_code = []
    for (commit, path), expected_sha in EXPECTED_REFERENCE_CODE.items():
        content = git(repo, "show", f"{commit}:{path}")
        if sha256_bytes(content) != expected_sha:
            raise RuntimeError(f"reference source blob drift: {commit}:{path}")
        reference_code.append(
            {"commit": commit, "path": path, "sha256": expected_sha}
        )

    source = args.validation_source.resolve()
    table_dir = args.validation_table_dir.resolve()
    documents_path = table_dir / "documents.jsonl"
    abs_indices_path = table_dir / "abs_indices.npy"
    header_reference_path = table_dir / "header_kinds.npy"
    header_receipt_path = table_dir / "receipt.json"
    negative_reference_path = args.reference_negative_roles.resolve()
    negative_receipt_path = args.reference_negative_receipt.resolve()
    required_files = (
        source,
        documents_path,
        abs_indices_path,
        header_reference_path,
        header_receipt_path,
        negative_reference_path,
        negative_receipt_path,
    )
    if any(not path.is_file() or path.is_symlink() for path in required_files):
        raise RuntimeError("a deterministic role input is missing or a symlink")
    expected_hashes = {
        source: EXPECTED_RAW_INPUT_SHA,
        header_reference_path: EXPECTED_HEADER_KINDS_SHA,
        header_receipt_path: EXPECTED_HEADER_RECEIPT_SHA,
        negative_reference_path: EXPECTED_NEGATIVE_ROLES_SHA,
        negative_receipt_path: EXPECTED_NEGATIVE_RECEIPT_SHA,
    }
    for path, expected_sha in expected_hashes.items():
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"deterministic role input drift: {path}")

    header_receipt = json.loads(header_receipt_path.read_text(encoding="utf-8"))
    negative_receipt = json.loads(negative_receipt_path.read_text(encoding="utf-8"))
    if (
        header_receipt.get("code_commit")
        != "e1463dbf29562ba8784438e945b349445977b6a8"
        or header_receipt.get("line_count") != EXPECTED_LINE_COUNT
        or header_receipt.get("document_count") != EXPECTED_DOCUMENT_COUNT
        or header_receipt.get("outputs", {}).get("header_kinds.npy", {}).get("sha256")
        != EXPECTED_HEADER_KINDS_SHA
    ):
        raise RuntimeError("header-kinds receipt contract changed")
    if (
        negative_receipt.get("code_commit")
        != "b49a359346a833cbc12c9c28eec186dd933a522b"
        or negative_receipt.get("outputs", {}).get("validation_roles.npy", {}).get("sha256")
        != EXPECTED_NEGATIVE_ROLES_SHA
        or negative_receipt.get("role_counts", {}).get("exact_negative_scope_heading")
        != 196
    ):
        raise RuntimeError("negative-role receipt contract changed")
    if tuple(NEGATIVE_ROLE_NAMES) != (
        "figure_caption",
        "table_or_equation",
        "exact_negative_scope_heading",
        "generic_markdown_heading",
        "footnote",
        "running_or_enumerated_prose",
        "legal_procedure",
        "other_explicit_negative",
    ):
        raise RuntimeError("deterministic negative-role order changed")

    documents = []
    with documents_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(json.loads(line))
    if len(documents) != EXPECTED_DOCUMENT_COUNT:
        raise RuntimeError("validation document inventory changed")
    expected = {
        str(row["document_id"]): (
            position,
            int(row["line_start"]),
            int(row["line_end"]),
        )
        for position, row in enumerate(documents)
    }
    if len(expected) != EXPECTED_DOCUMENT_COUNT:
        raise RuntimeError("duplicate validation document ids")
    absolute = np.load(abs_indices_path, mmap_mode="r", allow_pickle=False)
    if absolute.shape != (EXPECTED_LINE_COUNT,):
        raise RuntimeError("validation absolute-index vector changed")

    # Only validation rows are deserialized.  Prediction code below reads
    # document_id, text and abs_idx; it never reads label, target, or split
    # annotations from an individual line.
    source_rows: dict[str, list[Mapping[str, Any]]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if '"split": "validation"' not in raw:
                continue
            row = json.loads(raw)
            document_id = str(row.get("document_id"))
            if document_id not in expected:
                continue
            if document_id in source_rows:
                raise RuntimeError("duplicate validation source document")
            raw_lines = row.get("lines")
            if not isinstance(raw_lines, list):
                raise RuntimeError(f"{document_id}: missing source lines")
            # Drop every field except the deployable inference inputs.
            source_rows[document_id] = [
                {"text": line.get("text"), "abs_idx": line.get("abs_idx")}
                for line in raw_lines
            ]
    if set(source_rows) != set(expected):
        raise RuntimeError("source/table validation inventory mismatch")

    unlabeled_digest = hashlib.sha256()
    tasks = []
    for document in documents:
        document_id = str(document["document_id"])
        _, start, end = expected[document_id]
        lines = source_rows[document_id]
        if len(lines) != end - start:
            raise RuntimeError(f"{document_id}: line-count mismatch")
        line_abs = np.asarray([line["abs_idx"] for line in lines], dtype=np.uint32)
        if not np.array_equal(line_abs, absolute[start:end]):
            raise RuntimeError(f"{document_id}: absolute-index alignment failure")
        canonical = json.dumps(
            {"document_id": document_id, "lines": lines},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        unlabeled_digest.update(len(canonical).to_bytes(8, "big"))
        unlabeled_digest.update(canonical)
        tasks.append((document_id, lines))

    generated_headers = np.zeros(EXPECTED_LINE_COUNT, dtype=np.uint8)
    generated_negative = np.zeros(
        (EXPECTED_LINE_COUNT, len(NEGATIVE_ROLE_NAMES)), dtype=np.uint8
    )
    worker_counts = {name: 0 for name in NEGATIVE_ROLE_NAMES}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for document_id, headers, negatives, counts in executor.map(
            analyze_document, tasks, chunksize=1
        ):
            _, start, end = expected[document_id]
            if headers.shape != (end - start,) or negatives.shape != (
                end - start,
                len(NEGATIVE_ROLE_NAMES),
            ):
                raise RuntimeError(f"{document_id}: worker output misaligned")
            generated_headers[start:end] = headers
            generated_negative[start:end] = negatives
            for name, count in counts.items():
                worker_counts[name] += int(count)

    # Equivalence references are opened only after text-only inference is
    # complete.  They gate reproducibility and never supply a predicted role.
    reference_headers = np.load(header_reference_path, mmap_mode="r", allow_pickle=False)
    reference_negative = np.load(
        negative_reference_path, mmap_mode="r", allow_pickle=False
    )
    if not np.array_equal(generated_headers, reference_headers):
        raise RuntimeError("text-only heading regeneration differs from frozen reference")
    if not np.array_equal(generated_negative, reference_negative):
        raise RuntimeError("text-only negative-role regeneration differs from frozen reference")
    actual_negative_counts = {
        name: int(generated_negative[:, index].sum())
        for index, name in enumerate(NEGATIVE_ROLE_NAMES)
    }
    if actual_negative_counts != {
        name: int(negative_receipt["role_counts"][name]) for name in NEGATIVE_ROLE_NAMES
    } or worker_counts != actual_negative_counts:
        raise RuntimeError("deterministic negative-role counts differ from receipt")

    non_bib = generated_negative[
        :, NEGATIVE_ROLE_NAMES.index("exact_negative_scope_heading")
    ].astype(bool)
    conflict = (generated_headers > 0) & non_bib
    if np.any(conflict):
        raise RuntimeError("bibliography and non-bibliography deterministic headings conflict")
    roles = generated_headers.copy()
    roles[non_bib] = HEADER_ROLES.index("NON_BIB_HEADER")
    role_counts = {
        name: int(np.count_nonzero(roles == index))
        for index, name in enumerate(HEADER_ROLES)
    }
    if role_counts != {
        "NONE": 258672,
        "BIB_HEADER": 144,
        "BIB_SUBHEADER": 55,
        "NON_BIB_HEADER": 196,
    }:
        raise RuntimeError(f"unexpected deterministic header-role counts: {role_counts}")

    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    try:
        roles_path = partial / "deterministic_header_roles.npy"
        with roles_path.open("xb") as handle:
            np.save(handle, roles, allow_pickle=False)
        generator_copy = partial / "generate_deterministic_header_roles.py"
        shutil.copyfile(Path(__file__).resolve(), generator_copy)
        policy = {
            "schema_version": "bibliography-deterministic-header-role-policy-v1",
            "roles": {name: index for index, name in enumerate(HEADER_ROLES)},
            "bib_header": "analyze_bib_line(text, abs_idx) == BibRole.HEADING",
            "bib_subheader": "analyze_bib_line(text, abs_idx) == BibRole.SUBHEADING",
            "non_bib_header": (
                "bibliography_deterministic_roles._analyze_document(text) emits "
                "exact_negative_scope_heading"
            ),
            "conflict_policy": "fail_closed_no_role_vector_emitted",
            "prediction_fields": ["document_id", "lines[].text", "lines[].abs_idx"],
            "forbidden_prediction_fields": ["lines[].label", "targets", "original_labels"],
        }
        policy_path = partial / "generation_policy.json"
        write_json_new(policy_path, policy)
        current_sources = {}
        for name in (
            "deterministic_structure.py",
            "bibliography_v2.py",
            "bibliography_deterministic_roles.py",
            "bibliography_scope_rules.py",
        ):
            path = (
                repo
                / "subprojects/05_token_distillation_cpt/02_corpus_preparation/"
                "15_clean_academic/eval/sequence_models"
                / name
            )
            current_sources[name] = {"path": str(path), "sha256": sha256_file(path)}
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed_text_only_generation_byte_identical_reference_gates",
            "code_commit": EXPECTED_COMMIT,
            "slurm_job_id": job_id,
            "line_count": EXPECTED_LINE_COUNT,
            "document_count": EXPECTED_DOCUMENT_COUNT,
            "role_encoding": {name: index for index, name in enumerate(HEADER_ROLES)},
            "role_counts": role_counts,
            "semantic_conflict_count": 0,
            "unlabeled_inventory_sha256": unlabeled_digest.hexdigest(),
            "inference_contract": {
                "uses_text": True,
                "uses_absolute_line_index": True,
                "uses_labels": False,
                "uses_targets": False,
                "uses_original_labels": False,
                "other_splits_deserialized": False,
                "reference_arrays_used_for_prediction": False,
                "reference_arrays_opened_after_prediction_for_equivalence_gate_only": True,
                "reproducible_on_unlabeled_sealed_documents": True,
            },
            "source_inputs": {
                "validation_source": {"path": str(source), "sha256": EXPECTED_RAW_INPUT_SHA},
                "documents": {"path": str(documents_path), "sha256": sha256_file(documents_path)},
                "abs_indices": {"path": str(abs_indices_path), "sha256": sha256_file(abs_indices_path)},
                "header_reference": {"path": str(header_reference_path), "sha256": EXPECTED_HEADER_KINDS_SHA},
                "header_reference_receipt": {"path": str(header_receipt_path), "sha256": EXPECTED_HEADER_RECEIPT_SHA},
                "negative_reference": {"path": str(negative_reference_path), "sha256": EXPECTED_NEGATIVE_ROLES_SHA},
                "negative_reference_receipt": {"path": str(negative_receipt_path), "sha256": EXPECTED_NEGATIVE_RECEIPT_SHA},
            },
            "equivalence_gates": {
                "generated_bib_roles_equal_header_kinds_reference": True,
                "generated_negative_roles_equal_validation_roles_reference": True,
                "reference_role_counts_match_receipts": True,
            },
            "reference_generation_code": reference_code,
            "current_generation_code": current_sources,
            "outputs": {
                "deterministic_header_roles.npy": {
                    "bytes": roles_path.stat().st_size,
                    "sha256": sha256_file(roles_path),
                },
                "generation_policy.json": {
                    "bytes": policy_path.stat().st_size,
                    "sha256": sha256_file(policy_path),
                },
                "generate_deterministic_header_roles.py": {
                    "bytes": generator_copy.stat().st_size,
                    "sha256": sha256_file(generator_copy),
                },
            },
        }
        write_json_new(partial / "receipt.json", receipt)
        os.replace(partial, output)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o440)
    output.chmod(0o550)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output_dir": str(output),
                "roles_sha256": receipt["outputs"]["deterministic_header_roles.npy"]["sha256"],
                "receipt_sha256": sha256_file(output / "receipt.json"),
                "role_counts": role_counts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
