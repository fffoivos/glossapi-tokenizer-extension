from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


WORK_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = WORK_ROOT / "nanochat_glossapi_en_vs_el" / "prepare_glossapi_greek_experiment_data.py"
PYTHON_BIN = WORK_ROOT / ".venv" / "bin" / "python"

if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from glossapi_corpus_cli import pipeline


def load_builder_module():
    spec = importlib.util.spec_from_file_location("nanochat_greek_builder_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_dataset_row_groups(repo_root: Path, rows: list[dict[str, object]]) -> None:
    data_root = repo_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    for source_dataset in sorted({str(row["source_dataset"]) for row in rows}):
        shard_rows = [
            {key: value for key, value in row.items() if key != "source_dataset"}
            for row in rows
            if str(row["source_dataset"]) == source_dataset
        ]
        pq.write_table(pa.Table.from_pylist(shard_rows), data_root / f"{source_dataset.replace('/', '__')}.parquet")


def write_builder_bundle(
    repo_root: Path,
    *,
    run_id: str,
    rows: list[dict[str, object]],
    strict_groups: dict[str, tuple[str | None, int]] | None = None,
    relaxed_groups: dict[str, tuple[str | None, int]] | None = None,
    near_pairs: list[dict[str, object]] | None = None,
    family_membership: list[dict[str, object]] | None = None,
    candidate_score_floor: float = 0.85,
) -> Path:
    bundle_root = repo_root / "dedup_metadata" / run_id / "builder_metadata"
    bundle_root.mkdir(parents=True, exist_ok=True)
    strict_groups = strict_groups or {}
    relaxed_groups = relaxed_groups or {}
    near_pairs = near_pairs or []
    family_membership = family_membership or []

    doc_rows: list[dict[str, object]] = []
    for row in rows:
        source_dataset = str(row["source_dataset"])
        source_doc_id = str(row["source_doc_id"])
        doc_key = pipeline.stable_doc_key(source_dataset, source_doc_id)
        strict_hash, strict_size = strict_groups.get(source_doc_id, (f"strict:{source_doc_id}", 1))
        relaxed_hash, relaxed_size = relaxed_groups.get(source_doc_id, (None, 0))
        len_greek = int(row.get("len_greek", len(str(row["text"]))))
        representative_score = float(
            row.get(
                "representative_score",
                max(0.0, len_greek * (1.0 - (float(row.get("greek_badness_score", 0.0) or 0.0) / 10.0))),
            )
        )
        doc_rows.append(
            {
                "doc_key": doc_key,
                "source_dataset": source_dataset,
                "source_doc_id": source_doc_id,
                "strict_exact_group_hash": strict_hash,
                "strict_exact_group_size": strict_size,
                "strict_exact_kept_doc_key": doc_key,
                "relaxed_exact_group_hash": relaxed_hash,
                "relaxed_exact_group_size": relaxed_size,
                "relaxed_exact_kept_doc_key": doc_key if relaxed_hash else None,
                "needs_ocr": bool(row.get("needs_ocr", False)),
                "greek_badness_score": row.get("greek_badness_score"),
                "len_greek": len_greek,
                "mojibake_badness_score": row.get("mojibake_badness_score"),
                "ocr_success": bool(row.get("ocr_success", True)),
                "text_length_for_selection": len(str(row["text"])),
                "representative_score": representative_score,
                "representative_score_version": "representative_score_v1_len_greek_badness10",
                "has_title": bool(row.get("title")),
                "has_author": bool(row.get("author")),
            }
        )
    pq.write_table(pa.Table.from_pylist(doc_rows), bundle_root / "doc_dedup_metadata.parquet")
    if family_membership:
        pq.write_table(pa.Table.from_pylist(family_membership), bundle_root / "dedup_family_membership.parquet")
    pq.write_table(pa.Table.from_pylist(near_pairs), bundle_root / "near_candidate_pairs.parquet")
    files = {
        "doc_metadata": "doc_dedup_metadata.parquet",
        "near_candidate_pairs": "near_candidate_pairs.parquet",
    }
    if family_membership:
        files["family_membership"] = "dedup_family_membership.parquet"
    (bundle_root / "manifest.json").write_text(
        json.dumps(
            {
                "builder_metadata_version": "builder_metadata_v2",
                "candidate_score_floor": candidate_score_floor,
                "builder_default_threshold": candidate_score_floor,
                "builder_exact_stage": "strict_and_relaxed",
                "near_cluster_mode": "representative_validation",
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (repo_root / "dedup_metadata" / "latest.json").write_text(
        json.dumps(
            {
                "latest_run_id": run_id,
                "path": f"dedup_metadata/{run_id}",
                "builder_metadata_root": f"dedup_metadata/{run_id}/builder_metadata",
                "code_root": f"dedup_metadata/{run_id}/code",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle_root


def test_resolve_input_layout_detects_repo_and_data_roots(tmp_path: Path) -> None:
    module = load_builder_module()
    repo_root = tmp_path / "repo"
    rows = [
        {
            "source_dataset": "alpha",
            "source_doc_id": "a1",
            "text": "Δείγμα",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
        }
    ]
    write_dataset_row_groups(repo_root, rows)
    write_builder_bundle(repo_root, run_id="run1", rows=rows)

    repo_layout = module.resolve_input_layout(repo_root)
    data_layout = module.resolve_input_layout(repo_root / "data")

    assert repo_layout.repo_root == repo_root.resolve()
    assert repo_layout.data_root == (repo_root / "data").resolve()
    assert repo_layout.latest_dedup_pointer_path == (repo_root / "dedup_metadata" / "latest.json").resolve()
    assert repo_layout.has_chars_column is False

    assert data_layout.repo_root == repo_root.resolve()
    assert data_layout.data_root == (repo_root / "data").resolve()
    assert data_layout.latest_dedup_pointer_path == (repo_root / "dedup_metadata" / "latest.json").resolve()
    assert data_layout.has_chars_column is False

    dedup_root, code_root = module.resolve_dedup_metadata_root(repo_layout, explicit_root=None)
    assert dedup_root == (repo_root / "dedup_metadata" / "run1" / "builder_metadata").resolve()
    assert code_root == (repo_root / "dedup_metadata" / "run1" / "code").resolve()


def test_dedup_annotations_match_pipeline_reference_on_bundle_metadata(tmp_path: Path) -> None:
    module = load_builder_module()
    repo_root = tmp_path / "repo"
    rows = [
        {
            "source_dataset": "alpha",
            "source_doc_id": "a1",
            "text": "κοινό alpha",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "ocr_success": True,
            "title": "Α",
        },
        {
            "source_dataset": "beta",
            "source_doc_id": "b1",
            "text": "κοινό beta",
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "ocr_success": True,
            "title": "Β",
        },
        {
            "source_dataset": "alpha",
            "source_doc_id": "a2",
            "text": "ενδο alpha χαμηλής ποιότητας",
            "greek_badness_score": 3.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "ocr_success": True,
            "title": "ΑΑ",
        },
        {
            "source_dataset": "alpha",
            "source_doc_id": "a3",
            "text": "ενδο alpha καλύτερο",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "ocr_success": True,
            "title": "ΑΑΑ",
        },
    ]
    bundle_root = write_builder_bundle(
        repo_root,
        run_id="run1",
        rows=rows,
        strict_groups={
            "a2": ("strict-alpha", 2),
            "a3": ("strict-alpha", 2),
        },
        family_membership=[
            {
                "doc_key": pipeline.stable_doc_key("alpha", "a1"),
                "source_dataset": "alpha",
                "source_doc_id": "a1",
                "family_id": "family-shared",
                "family_size": 2,
                "family_source_count": 2,
                "family_mixed_source": True,
                "canonical_kept_doc_key": pipeline.stable_doc_key("alpha", "a1"),
                "canonical_decision": "keep",
                "canonical_decision_stage": "near_duplicate",
                "dedup_run_id": "run1",
                "selection_version": "selection_v2",
                "representative_score_version": "representative_score_v1_len_greek_badness10",
            },
            {
                "doc_key": pipeline.stable_doc_key("beta", "b1"),
                "source_dataset": "beta",
                "source_doc_id": "b1",
                "family_id": "family-shared",
                "family_size": 2,
                "family_source_count": 2,
                "family_mixed_source": True,
                "canonical_kept_doc_key": pipeline.stable_doc_key("alpha", "a1"),
                "canonical_decision": "drop",
                "canonical_decision_stage": "near_duplicate",
                "dedup_run_id": "run1",
                "selection_version": "selection_v2",
                "representative_score_version": "representative_score_v1_len_greek_badness10",
            },
            {
                "doc_key": pipeline.stable_doc_key("alpha", "a2"),
                "source_dataset": "alpha",
                "source_doc_id": "a2",
                "family_id": "family-alpha",
                "family_size": 2,
                "family_source_count": 1,
                "family_mixed_source": False,
                "canonical_kept_doc_key": pipeline.stable_doc_key("alpha", "a3"),
                "canonical_decision": "drop",
                "canonical_decision_stage": "strict_exact",
                "dedup_run_id": "run1",
                "selection_version": "selection_v2",
                "representative_score_version": "representative_score_v1_len_greek_badness10",
            },
            {
                "doc_key": pipeline.stable_doc_key("alpha", "a3"),
                "source_dataset": "alpha",
                "source_doc_id": "a3",
                "family_id": "family-alpha",
                "family_size": 2,
                "family_source_count": 1,
                "family_mixed_source": False,
                "canonical_kept_doc_key": pipeline.stable_doc_key("alpha", "a3"),
                "canonical_decision": "keep",
                "canonical_decision_stage": "strict_exact",
                "dedup_run_id": "run1",
                "selection_version": "selection_v2",
                "representative_score_version": "representative_score_v1_len_greek_badness10",
            },
        ],
        near_pairs=[
            {
                "doc_key_left": pipeline.stable_doc_key("alpha", "a1"),
                "source_dataset_left": "alpha",
                "doc_key_right": pipeline.stable_doc_key("beta", "b1"),
                "source_dataset_right": "beta",
                "estimated_jaccard": 0.91,
            }
        ],
    )

    frame = pd.DataFrame(
        [
            {
                "source_dataset": str(row["source_dataset"]),
                "source_doc_id": str(row["source_doc_id"]),
                "chars": len(str(row["text"])),
                "stable_key": str(row["source_doc_id"]),
            }
            for row in rows
        ]
    )

    annotations, summary = module.build_dedup_annotations(
        frame,
        dedup_metadata_root=bundle_root,
        dedup_action="annotate",
        dedup_exact_stage="strict_and_relaxed",
        dedup_similarity_threshold=0.85,
        dedup_inter_dataset_policy="share_aware",
        dedup_source_weights_path=None,
    )
    reference_frame, reference_summary = pipeline.apply_builder_dedup(
        frame,
        dedup_metadata_root=bundle_root,
        dedup_action="annotate",
        dedup_exact_stage="strict_and_relaxed",
        dedup_similarity_threshold=0.85,
        dedup_inter_dataset_policy="share_aware",
        dedup_source_weights_path=None,
    )

    annotation_columns = [
        "source_dataset",
        "source_doc_id",
        "dedup_family_id",
        "dedup_family_size",
        "dedup_pool_key",
        "dedup_pool_source_count",
        "dedup_is_shared_pool",
        "dedup_representative_doc_key",
        "dedup_family_role",
        "dedup_similarity_threshold",
        "dedup_candidate_score_floor",
    ]
    actual = annotations[annotation_columns].sort_values(["source_dataset", "source_doc_id"]).reset_index(drop=True)
    expected = reference_frame[annotation_columns].sort_values(["source_dataset", "source_doc_id"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected)
    assert summary == reference_summary


def test_dedup_annotations_prefer_successful_ocr_when_ocr_is_required(tmp_path: Path) -> None:
    module = load_builder_module()
    repo_root = tmp_path / "repo"
    rows = [
        {
            "source_dataset": "alpha",
            "source_doc_id": "a1",
            "text": "ocr variant alpha",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": True,
            "ocr_success": False,
            "title": "Α",
        },
        {
            "source_dataset": "beta",
            "source_doc_id": "b1",
            "text": "ocr variant beta",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": True,
            "ocr_success": True,
            "title": "Β",
        },
    ]
    bundle_root = write_builder_bundle(
        repo_root,
        run_id="run1",
        rows=rows,
        strict_groups={
            "a1": ("strict-ocr", 2),
            "b1": ("strict-ocr", 2),
        },
        family_membership=[
            {
                "doc_key": pipeline.stable_doc_key("alpha", "a1"),
                "source_dataset": "alpha",
                "source_doc_id": "a1",
                "family_id": "family-ocr",
                "family_size": 2,
                "family_source_count": 2,
                "family_mixed_source": True,
                "canonical_kept_doc_key": pipeline.stable_doc_key("beta", "b1"),
                "canonical_decision": "drop",
                "canonical_decision_stage": "strict_exact",
                "dedup_run_id": "run1",
                "selection_version": "selection_v2",
                "representative_score_version": "representative_score_v1_len_greek_badness10",
            },
            {
                "doc_key": pipeline.stable_doc_key("beta", "b1"),
                "source_dataset": "beta",
                "source_doc_id": "b1",
                "family_id": "family-ocr",
                "family_size": 2,
                "family_source_count": 2,
                "family_mixed_source": True,
                "canonical_kept_doc_key": pipeline.stable_doc_key("beta", "b1"),
                "canonical_decision": "keep",
                "canonical_decision_stage": "strict_exact",
                "dedup_run_id": "run1",
                "selection_version": "selection_v2",
                "representative_score_version": "representative_score_v1_len_greek_badness10",
            },
        ],
        near_pairs=[
            {
                "doc_key_left": pipeline.stable_doc_key("alpha", "a1"),
                "source_dataset_left": "alpha",
                "doc_key_right": pipeline.stable_doc_key("beta", "b1"),
                "source_dataset_right": "beta",
                "estimated_jaccard": 0.10,
            }
        ],
    )

    frame = pd.DataFrame(
        [
            {
                "source_dataset": str(row["source_dataset"]),
                "source_doc_id": str(row["source_doc_id"]),
                "chars": len(str(row["text"])),
                "stable_key": str(row["source_doc_id"]),
            }
            for row in rows
        ]
    )

    annotations, _ = module.build_dedup_annotations(
        frame,
        dedup_metadata_root=bundle_root,
        dedup_action="annotate",
        dedup_exact_stage="strict_only",
        dedup_similarity_threshold=0.85,
        dedup_inter_dataset_policy="quality_first",
        dedup_source_weights_path=None,
    )

    representatives = annotations.loc[annotations["dedup_family_role"] == "representative", "source_doc_id"].tolist()
    assert representatives == ["b1"]


def test_main_smoke_applies_quality_and_dedup_filters(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    build_root = tmp_path / "build"
    rows = [
        {
            "source_dataset": "alpha",
            "source_doc_id": "a1",
            "text": "κείμενο alpha 1",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
        },
        {
            "source_dataset": "alpha",
            "source_doc_id": "a2",
            "text": "κείμενο alpha 2",
            "greek_badness_score": 12.0,
            "mojibake_badness_score": 0.0,
        },
        {
            "source_dataset": "beta",
            "source_doc_id": "b1",
            "text": "κείμενο beta 1",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.2,
        },
        {
            "source_dataset": "beta",
            "source_doc_id": "b2",
            "text": "κείμενο beta 2",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
        },
        {
            "source_dataset": "gamma",
            "source_doc_id": "g1",
            "text": "κοινό gamma 1",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
        },
        {
            "source_dataset": "delta",
            "source_doc_id": "d1",
            "text": "κοινό delta 1",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
        },
        {
            "source_dataset": "epsilon",
            "source_doc_id": "e1",
            "text": "ocr failed epsilon 1",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": True,
            "ocr_success": False,
        },
        {
            "source_dataset": "zeta",
            "source_doc_id": "z1",
            "text": "ocr success zeta 1",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": True,
            "ocr_success": True,
        },
    ]
    write_dataset_row_groups(repo_root, rows)
    write_builder_bundle(
        repo_root,
        run_id="run1",
        rows=rows,
        near_pairs=[
            {
                "doc_key_left": pipeline.stable_doc_key("gamma", "g1"),
                "source_dataset_left": "gamma",
                "doc_key_right": pipeline.stable_doc_key("delta", "d1"),
                "source_dataset_right": "delta",
                "estimated_jaccard": 0.90,
            }
        ],
    )

    subprocess.run(
        [
            str(PYTHON_BIN),
            str(BUILDER_PATH),
            "--input-root",
            str(repo_root),
            "--output-root",
            str(build_root),
            "--threads",
            "2",
            "--badness-lt",
            "10",
            "--mojibake-lte",
            "0.1",
            "--train-chars",
            "1000",
            "--val-chars",
            "100",
            "--test-chars",
            "100",
            "--chunk-markdown",
            "--fully-include-below-share",
            "0.05",
            "--dedup-action",
            "drop_intra_and_inter",
            "--dedup-similarity-threshold",
            "0.85",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(WORK_ROOT)},
    )

    summary = json.loads((build_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["badness_lt"] == 10.0
    assert summary["mojibake_lte"] == 0.1
    assert summary["filtered_rows_before_dedup"] == 5
    assert summary["dedup_summary"]["rows_after"] == 4
    assert summary["dedup_metadata_root"] == str((repo_root / "dedup_metadata" / "run1" / "builder_metadata").resolve())

    total_manifest_rows = 0
    for split in ("train", "val", "test"):
        manifest = pd.read_csv(build_root / "manifests" / f"{split}_manifest.csv")
        total_manifest_rows += int(len(manifest))
        assert "e1" not in set(manifest["source_doc_id"])
    assert total_manifest_rows == 4

    pool_summary = pd.read_csv(build_root / "split_pool_summary.csv")
    assert "shared:delta+gamma" in set(pool_summary["allocation_group"])
