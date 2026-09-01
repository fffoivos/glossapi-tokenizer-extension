#!/usr/bin/env python3
"""Adopt existing H-to-G streams into one portable canonical data manifest.

This is a metadata-only adapter. It hard-links the already full-hashed indexed
payloads and copies no token data. Canonical verifiers remain the authority for
prepared-dataset and ordered-manifest acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from contract_utils import executing_code_bundle, file_binding

TOKENIZER_REPO = "fffoivos/apertus-tokenizer-extension"
TOKENIZER_REVISION = "fcd33ec09fb7d86bc072b3a4b3e890efa6473b66"
TOKENIZER_SHA256 = "358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394"
DATASET_REPO = "fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2"
DATASET_REVISION = "987b8955fcd395c6219e39df9e64715457f69065"
SEQUENCE_LENGTH = 4096
SUITES = ["greekmmlu_public_v1", "native_greek_suite_v1", "validation_panels_v1"]


def training_phases() -> list[dict[str, Any]]:
    """Declare the full frozen horizon without claiming a Phase-3 selection yet.

    The prepared-data manifest binds source pools and mix semantics.  The later
    Phase-3 authority still owns the exact unseen-document selection, cache and
    checkpoint gate; including the pool-level phase here keeps one canonical
    campaign identity across the already-approved 0..3694 trajectory.
    """

    replay_weights = {
        "openarchives": 0.79,
        "foreign_replay": 0.20,
        "old_greek_replay": 0.01,
    }
    return [
        {
            "id": "phase_1_hplt",
            "dataset_weights": {
                "hplt": 0.79,
                "foreign_replay": 0.20,
                "old_greek_replay": 0.01,
            },
        },
        {
            "id": "phase_2_openarchives",
            "dataset_weights": dict(replay_weights),
        },
        {
            "id": "phase_3_unseen_openarchives",
            "dataset_weights": dict(replay_weights),
        },
    ]


def portable_manifest_gate(
    gate: dict[str, Any], manifest_path: Path, root: Path
) -> dict[str, Any]:
    """Replace a verifier's transient absolute path with its portable binding."""

    result = json.loads(json.dumps(gate))
    binding = result.get("manifest")
    require(isinstance(binding, dict), "canonical manifest gate binding missing")
    require(
        int(binding.get("bytes", -1)) == manifest_path.stat().st_size
        and str(binding.get("sha256", "")) == sha256_file(manifest_path),
        "canonical manifest gate binding drift",
    )
    binding["path"] = str(manifest_path.resolve().relative_to(root.resolve()))
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def link_file(source: Path, destination: Path) -> None:
    source = source.resolve()
    require(source.is_file() and not source.is_symlink(), f"source missing or symlinked: {source}")
    require(not destination.exists(), f"immutable link exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def relative_binding(path: Path, root: Path, *, known_sha256: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    root = root.resolve()
    relative = path.relative_to(root)
    return {
        "path": str(relative),
        "bytes": path.stat().st_size,
        "sha256": known_sha256 or sha256_file(path),
    }


def run(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def tokenized_token_count(tokenized: dict[str, Any]) -> int:
    """Read the frozen Megatron index token count without renaming its field."""
    index = tokenized.get("index")
    require(isinstance(index, dict), "tokenized receipt index missing")
    require("tokens_including_eod" in index, "tokenized receipt token count missing")
    return int(index["tokens_including_eod"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--canonical-code-receipt", type=Path, required=True)
    parser.add_argument("--hf-inventory", type=Path, required=True)
    parser.add_argument("--reader-smoke", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    require(os.environ.get("SLURM_JOB_PARTITION") == "debug", "adoption must run on debug")
    require(int(os.environ.get("SLURM_NNODES", "0")) == 1, "adoption requires one debug node")
    stage = args.stage_root.resolve()
    canonical = args.canonical_root.resolve()
    final_root = args.output_root.resolve()
    require(stage.is_dir(), "stage root missing")
    require(canonical.is_dir(), "canonical efficiency root missing")
    require(not final_root.exists(), f"immutable adoption root exists: {final_root}")

    canonical_bundle = read_json(args.canonical_code_receipt)
    require(
        canonical_bundle.get("schema_version")
        == "apertus_mini_immutable_code_bundle_v1"
        and canonical_bundle.get("status") == "frozen"
        and canonical_bundle.get("kind") == "efficiency"
        and Path(str(canonical_bundle.get("root", ""))).resolve() == canonical
        and isinstance(canonical_bundle.get("tree_sha256"), str)
        and len(canonical_bundle["tree_sha256"]) == 64,
        "canonical efficiency bundle receipt drift",
    )

    prepared_verifier = canonical / "skills/prepare-apertus-experiment/scripts/verify_prepared_dataset.py"
    manifest_verifier = canonical / "skills/prepare-apertus-experiment/scripts/verify_training_data_manifest.py"
    require(prepared_verifier.is_file() and manifest_verifier.is_file(), "canonical verifiers missing")

    temporary = final_root.parent / f".{final_root.name}.partial-{os.environ.get('SLURM_JOB_ID', 'local')}"
    require(not temporary.exists(), f"temporary adoption root exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        evidence_dir = temporary / "evidence"
        payload_dir = temporary / "payload"
        evidence_dir.mkdir()
        payload_dir.mkdir()

        receipt_sources = {
            "hf_inventory.json": args.hf_inventory,
            "replay_source_inventory.json": stage / "receipts/replay_source_inventory.json",
            "source_views.json": stage / "receipts/source_views.json",
            "hplt_prepared_stream.json": stage / "receipts/hplt_prepared_stream.json",
            "openarchives_prepared_stream.json": stage / "receipts/openarchives_prepared_stream.json",
            "replay_native_scan.json": stage / "receipts/native_suite_replay_scan_post_filter.json",
            "replay_greekmmlu_scan.json": stage / "receipts/greekmmlu_scan_replay_selected_post.json",
            "replay_validation_exclusion.json": stage / "receipts/replay_validation_exclusion.json",
            "hplt_stage_b.json": stage / "receipts/hplt_stage_b.json",
            "openarchives_stage_b.json": stage / "receipts/openarchives_stage_b.json",
            "replay_stage_b.json": stage / "receipts/replay_selected_stage_b.json",
            "reader_smoke.json": args.reader_smoke,
        }
        for name, source in receipt_sources.items():
            link_file(source, evidence_dir / name)

        inventory = read_json(evidence_dir / "hf_inventory.json")
        require(
            inventory.get("schema_version") == "apertus_hf_dataset_inventory_v1"
            and inventory.get("status") == "completed"
            and inventory.get("repo_id") == DATASET_REPO
            and inventory.get("revision") == DATASET_REVISION,
            "canonical HF inventory drift",
        )
        smoke = read_json(evidence_dir / "reader_smoke.json")
        require(
            smoke.get("schema_version") == "apertus_hard_h_to_g_profile_benchmark_v1"
            and smoke.get("status") == "passed"
            and smoke.get("scale") == "8b",
            "exact-profile reader evidence drift",
        )

        datasets = {
            "hplt": {
                "role": "main",
                "tokenized": stage / "receipts/tokenized_hplt.json",
                "anonymization": "hplt_stage_b.json",
                "decontamination": {
                    "greekmmlu_public_v1": "hplt_prepared_stream.json",
                    "native_greek_suite_v1": "source_views.json",
                    "validation_panels_v1": "source_views.json",
                },
                "metadata": [
                    "source_dataset", "source_doc_id", "source_metadata_json", "release_shard", "release_row_index"
                ],
                "hf": True,
            },
            "openarchives": {
                "role": "main",
                "tokenized": stage / "receipts/tokenized_openarchives.json",
                "anonymization": "openarchives_stage_b.json",
                "decontamination": {
                    "greekmmlu_public_v1": "openarchives_prepared_stream.json",
                    "native_greek_suite_v1": "source_views.json",
                    "validation_panels_v1": "source_views.json",
                },
                "metadata": [
                    "source_dataset", "source_doc_id", "source_metadata_json", "release_shard", "release_row_index"
                ],
                "hf": True,
            },
            "foreign_replay": {
                "role": "replay",
                "tokenized": stage / "receipts/tokenized_foreign.json",
                "anonymization": "replay_stage_b.json",
                "decontamination": {
                    "greekmmlu_public_v1": "replay_greekmmlu_scan.json",
                    "native_greek_suite_v1": "replay_native_scan.json",
                    "validation_panels_v1": "replay_validation_exclusion.json",
                },
                "metadata": ["source_dataset", "source_doc_id", "adapter_source", "adapter_row_index"],
                "hf": False,
            },
            "old_greek_replay": {
                "role": "replay",
                "tokenized": stage / "receipts/tokenized_old_greek.json",
                "anonymization": "replay_stage_b.json",
                "decontamination": {
                    "greekmmlu_public_v1": "replay_greekmmlu_scan.json",
                    "native_greek_suite_v1": "replay_native_scan.json",
                    "validation_panels_v1": "replay_validation_exclusion.json",
                },
                "metadata": ["source_dataset", "source_doc_id", "adapter_source", "adapter_row_index"],
                "hf": False,
            },
        }

        gate_rows: list[dict[str, Any]] = []
        replay_inventory_sha = sha256_file(evidence_dir / "replay_source_inventory.json")
        for dataset_id, spec in datasets.items():
            tokenized_source = Path(spec["tokenized"])
            tokenized = read_json(tokenized_source)
            require(
                tokenized.get("schema_version") == "apertus_hard_h_to_g_tokenized_stream_v1"
                and tokenized.get("status") == "frozen",
                f"{dataset_id}: tokenized receipt drift",
            )
            tokenized_copy = evidence_dir / f"tokenized_{dataset_id}.json"
            link_file(tokenized_source, tokenized_copy)

            payload_files = []
            for suffix in ("bin", "idx"):
                bound = tokenized["files"][suffix]
                source = Path(str(bound["path"]))
                require(source.stat().st_size == int(bound["bytes"]), f"{dataset_id}: payload size drift")
                destination = payload_dir / f"{dataset_id}.{suffix}"
                link_file(source, destination)
                payload_files.append(relative_binding(destination, temporary, known_sha256=str(bound["sha256"])))

            packed_receipt_path = temporary / f"packed_{dataset_id}.json"
            write_json(
                packed_receipt_path,
                {
                    "schema_version": "apertus_hard_h_to_g_adopted_packed_corpus_v1",
                    "status": "completed",
                    "dataset_id": dataset_id,
                    "source_tokenized_receipt": relative_binding(tokenized_copy, temporary),
                    "documents": int(tokenized["index"]["documents"]),
                    "tokens": tokenized_token_count(tokenized),
                    "payload_files": payload_files,
                    "adoption": {
                        "payload_copy_performed": False,
                        "payload_rehashed": False,
                        "payload_hardlinks_created": True,
                    },
                },
            )

            requirements = {
                "schema_version": "apertus_dataset_requirements_v1",
                "status": "accepted",
                "dataset_id": dataset_id,
                "anonymization": True,
                "deduplication": False,
                "decontamination_suites": SUITES,
            }
            requirements_path = temporary / f"requirements_{dataset_id}.json"
            write_json(requirements_path, requirements)
            requirement_value = {key: requirements[key] for key in ("anonymization", "deduplication", "decontamination_suites")}

            transformations = {
                "anonymization": relative_binding(evidence_dir / str(spec["anonymization"]), temporary),
                "decontamination": {
                    suite: relative_binding(evidence_dir / receipt_name, temporary)
                    for suite, receipt_name in spec["decontamination"].items()
                },
            }
            prepared = {
                "schema_version": "apertus_prepared_hf_dataset_v1" if spec["hf"] else "apertus_prepared_dataset_v2",
                "status": "completed",
                "dataset_id": dataset_id,
                "payload_root": ".",
                "source_inventory": relative_binding(
                    evidence_dir / ("hf_inventory.json" if spec["hf"] else "replay_source_inventory.json"),
                    temporary,
                ),
                "schema": {
                    "text_column": "text",
                    "document_id_column": "source_doc_id",
                    "metadata_columns": spec["metadata"],
                    "row_count": int(tokenized["index"]["documents"]),
                },
                "requirements": requirement_value,
                "transformations": transformations,
                "tokenizer": {
                    "repo_id": TOKENIZER_REPO,
                    "revision": TOKENIZER_REVISION,
                    "tokenizer_sha256": TOKENIZER_SHA256,
                    "sequence_length": SEQUENCE_LENGTH,
                },
                "packed_corpus_receipt": relative_binding(packed_receipt_path, temporary),
                "reader_smoke_receipt": relative_binding(evidence_dir / "reader_smoke.json", temporary),
                "full_hash_verification": {
                    "status": "passed",
                    "manifest_sha256": sha256_file(tokenized_copy),
                },
                "prepared_on": {
                    "target": "debug",
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "adopted_existing_payload": True,
                    "payload_rehashed": False,
                },
                "payload_files": payload_files,
            }
            if spec["hf"]:
                prepared["hf_snapshot"] = {"repo_id": DATASET_REPO, "revision": DATASET_REVISION}
            else:
                prepared["source_snapshot"] = {
                    "kind": "receipted_composite",
                    "label": dataset_id,
                    "identity_sha256": replay_inventory_sha,
                }

            prepared_path = temporary / f"prepared_{dataset_id}.json"
            gate_path = temporary / f"prepared_gate_{dataset_id}.json"
            write_json(prepared_path, prepared)
            run([
                sys.executable,
                str(prepared_verifier),
                "--receipt", str(prepared_path),
                "--requirements", str(requirements_path),
                "--output", str(gate_path),
            ])
            gate = read_json(gate_path)
            require(gate.get("status") == "passed", f"{dataset_id}: canonical prepared gate failed")
            gate_rows.append({
                "id": dataset_id,
                "role": spec["role"],
                "requirements_sha256": gate["declared_requirements_sha256"],
                "prepared_gate": relative_binding(gate_path, temporary),
            })

        manifest = {
            "schema_version": "apertus_training_data_manifest_v1",
            "status": "completed",
            "manifest_id": "hard-h2g-full-horizon-v2",
            "root": ".",
            "tokenization_recipe": {
                "tokenizer_repo": TOKENIZER_REPO,
                "tokenizer_revision": TOKENIZER_REVISION,
                "tokenizer_sha256": TOKENIZER_SHA256,
                "text_column": "text",
                "normalization": "none",
                "eos_policy": "append_eod",
                "sequence_length": SEQUENCE_LENGTH,
                "packing_policy": "megatron_gptdataset_randomized_cross_document_reset_masks",
            },
            "datasets": gate_rows,
            "phases": training_phases(),
        }
        manifest["identity_sha256"] = digest(manifest)
        manifest_path = temporary / "training_data_manifest.json"
        raw_manifest_gate_path = temporary / ".training_data_manifest_gate.raw.json"
        manifest_gate_path = temporary / "training_data_manifest_gate.json"
        write_json(manifest_path, manifest)
        run([
            sys.executable,
            str(manifest_verifier),
            "--manifest", str(manifest_path),
            "--output", str(raw_manifest_gate_path),
        ])
        manifest_gate = portable_manifest_gate(
            read_json(raw_manifest_gate_path), manifest_path, temporary
        )
        require(
            manifest_gate.get("status") == "passed"
            and manifest_gate.get("identity_sha256") == manifest["identity_sha256"],
            "canonical training-data manifest gate failed",
        )
        write_json(manifest_gate_path, manifest_gate)
        raw_manifest_gate_path.unlink()
        write_json(temporary / "adoption_receipt.json", {
            "schema_version": "apertus_hard_h_to_g_canonical_data_adoption_v1",
            "status": "passed",
            "canonical_root": str(canonical),
            "canonical_code_bundle": {
                "root": str(canonical),
                "tree_sha256": canonical_bundle["tree_sha256"],
                "receipt": file_binding(args.canonical_code_receipt),
            },
            "executing_code_bundle": executing_code_bundle(),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_scope": "full_0_to_3694_horizon",
            "phase_3_exact_selection_requires_later_authority": True,
            "datasets": [row["id"] for row in gate_rows],
            "payload_copy_performed": False,
            "payload_hardlinks_created": True,
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "nodes": int(os.environ.get("SLURM_NNODES", "0")),
            },
        })
        os.rename(temporary, final_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps({
        "status": "passed",
        "output_root": str(final_root),
        "manifest": str(final_root / "training_data_manifest.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
