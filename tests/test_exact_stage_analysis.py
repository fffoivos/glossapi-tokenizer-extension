from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from glossapi_corpus_cli import text_dedup

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPTS_ROOT = REPO_ROOT / "analysis" / "dedup" / "text_publish" / "scripts"


def load_analysis_module():
    path = ANALYSIS_SCRIPTS_ROOT / "analyze_exact_stage_run.py"
    spec = importlib.util.spec_from_file_location("analyze_exact_stage_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_policy_module():
    path = ANALYSIS_SCRIPTS_ROOT / "analyze_exact_policy_implications.py"
    spec = importlib.util.spec_from_file_location("analyze_exact_policy_implications", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oa_sample_module():
    path = ANALYSIS_SCRIPTS_ROOT / "sample_openarchives_internal_exact_groups.py"
    spec = importlib.util.spec_from_file_location("sample_openarchives_internal_exact_groups", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oa_patterns_module():
    path = ANALYSIS_SCRIPTS_ROOT / "analyze_openarchives_internal_collection_patterns.py"
    spec = importlib.util.spec_from_file_location("analyze_openarchives_internal_collection_patterns", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oa_meaningful_module():
    path = ANALYSIS_SCRIPTS_ROOT / "analyze_openarchives_meaningful_exact_groups.py"
    spec = importlib.util.spec_from_file_location("analyze_openarchives_meaningful_exact_groups", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oa_suspicious_module():
    path = ANALYSIS_SCRIPTS_ROOT / "analyze_openarchives_suspicious_exact_groups.py"
    spec = importlib.util.spec_from_file_location("analyze_openarchives_suspicious_exact_groups", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oa_origin_audit_module():
    path = ANALYSIS_SCRIPTS_ROOT / "analyze_openarchives_suspicious_origin_audit.py"
    spec = importlib.util.spec_from_file_location("analyze_openarchives_suspicious_origin_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oa_llm_review_module():
    path = ANALYSIS_SCRIPTS_ROOT / "integrate_openarchives_llm_semantic_review.py"
    spec = importlib.util.spec_from_file_location("integrate_openarchives_llm_semantic_review", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oa_semantic_pipeline_module():
    path = ANALYSIS_SCRIPTS_ROOT / "prepare_openarchives_semantic_resolution_pipeline.py"
    spec = importlib.util.spec_from_file_location("prepare_openarchives_semantic_resolution_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oa_high_level_review_module():
    path = ANALYSIS_SCRIPTS_ROOT / "integrate_openarchives_high_level_content_review.py"
    spec = importlib.util.spec_from_file_location("integrate_openarchives_high_level_content_review", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oa_codex_bundle_module():
    path = ANALYSIS_SCRIPTS_ROOT / "materialize_openarchives_codex_content_review_bundle.py"
    spec = importlib.util.spec_from_file_location("materialize_openarchives_codex_content_review_bundle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oa_text_shape_research_module():
    path = ANALYSIS_SCRIPTS_ROOT / "analyze_openarchives_text_shape_research.py"
    spec = importlib.util.spec_from_file_location("analyze_openarchives_text_shape_research", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_snapshot(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, row_group_size=2)


def test_exact_stage_analysis_compares_openarchives_and_phd(tmp_path: Path) -> None:
    analysis_mod = load_analysis_module()
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    rows = [
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-pergamos-1",
            "text": "Το ίδιο κείμενο Pergamos",
            "title": "Pergamos OA",
            "author": "Author OA",
            "source_metadata_json": json.dumps({"collection_slug": "pergamos"}, ensure_ascii=False),
            "greek_badness_score": 5.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "Apothetirio_Pergamos",
            "source_doc_id": "pergamos-1",
            "text": "Το ίδιο κείμενο Pergamos",
            "title": "Pergamos source",
            "author": "Author source",
            "source_metadata_json": "{}",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-phd-1",
            "text": "Το ίδιο κείμενο PhD",
            "title": "PhD OA",
            "author": "OA Author",
            "source_metadata_json": json.dumps({"collection_slug": "phdtheses"}, ensure_ascii=False),
            "greek_badness_score": 4.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "greek_phd",
            "source_doc_id": "phd-1",
            "text": "Το ίδιο κείμενο PhD",
            "title": "PhD source",
            "author": "PhD Author",
            "source_metadata_json": json.dumps({"handle_url": "http://example/phd-1"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-unique-1",
            "text": "Μοναδικό κείμενο",
            "title": "Unique OA",
            "author": "U",
            "source_metadata_json": json.dumps({"collection_slug": "Pandemos"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_snapshot(input_root / "openarchives.gr.part-00000.parquet", [row for row in rows if row["source_dataset"] == "openarchives.gr"])
    write_snapshot(input_root / "Apothetirio_Pergamos.parquet", [row for row in rows if row["source_dataset"] == "Apothetirio_Pergamos"])
    write_snapshot(input_root / "greek_phd.part-00000.parquet", [row for row in rows if row["source_dataset"] == "greek_phd"])

    text_dedup.run_exact_dedup(input_root=input_root, state_root=state_root, run_root=run_root)

    oa_summary = tmp_path / "oa_summary.json"
    oa_summary.write_text(
        json.dumps(
            {
                "drop_rows_total": 20,
                "origins": {
                    "kallipos": {"drop_rows": 5},
                    "pergamos": {"drop_rows": 15},
                },
            },
            ensure_ascii=False,
        )
    )
    phd_summary = tmp_path / "phd_summary.json"
    phd_summary.write_text(json.dumps({"matched_oa_outside_phdtheses": 7}, ensure_ascii=False))
    recovery_summary = tmp_path / "recovery_summary.json"
    recovery_summary.write_text(
        json.dumps(
            {
                "exact_title_author_rows": 3,
                "total_rows_recovered_if_exact_title_unique_fallback_added": 4,
            },
            ensure_ascii=False,
        )
    )
    review_summary = tmp_path / "review_summary.json"
    review_summary.write_text(
        json.dumps(
            {
                "after_reduced_title_unique": 5,
                "after_all_tested_layers": 6,
            },
            ensure_ascii=False,
        )
    )

    analysis_mod.DEFAULT_OA_DROP_LIST_SUMMARY = oa_summary
    analysis_mod.DEFAULT_PHD_CROSS_COLLECTION_SUMMARY = phd_summary
    analysis_mod.DEFAULT_PERGAMOS_PHD_RECOVERY_SUMMARY = recovery_summary
    analysis_mod.DEFAULT_PERGAMOS_PHD_REVIEW_SUMMARY = review_summary

    output_dir = tmp_path / "analysis-out"
    summary = analysis_mod.analyze_exact_run(run_root=run_root, output_dir=output_dir)
    assert Path(summary["report_path"]).exists()

    metadata = json.loads((output_dir / "metadata_comparison.json").read_text())
    assert metadata["exact_published"]["openarchives"]["pergamos_to_pergamos_source_drops"] == 1
    assert metadata["exact_published"]["phd"]["oa_phdtheses_to_greek_phd_drops"] == 1

    source_summary_csv = (output_dir / "source_summary.csv").read_text()
    assert "openarchives.gr" in source_summary_csv
    assert "Apothetirio_Pergamos" in source_summary_csv

    oa_collection_csv = (output_dir / "openarchives_collection_summary.csv").read_text()
    assert "pergamos" in oa_collection_csv
    assert "phdtheses" in oa_collection_csv


def test_policy_analysis_builds_pair_recommendations(tmp_path: Path) -> None:
    analysis_mod = load_analysis_module()
    policy_mod = load_policy_module()
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    rows = [
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-1",
            "text": "Κοινό κείμενο",
            "title": "OA",
            "author": "A",
            "source_metadata_json": json.dumps({"collection_slug": "Pandemos"}, ensure_ascii=False),
            "greek_badness_score": 5.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "greek_phd",
            "source_doc_id": "phd-1",
            "text": "Κοινό κείμενο",
            "title": "PhD",
            "author": "B",
            "source_metadata_json": json.dumps({"handle_url": "http://example/phd-1"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_snapshot(input_root / "openarchives.gr.part-00000.parquet", [rows[0]])
    write_snapshot(input_root / "greek_phd.part-00000.parquet", [rows[1]])

    text_dedup.run_exact_dedup(input_root=input_root, state_root=state_root, run_root=run_root)

    oa_summary = tmp_path / "oa_summary.json"
    oa_summary.write_text(json.dumps({"drop_rows_total": 1, "origins": {"kallipos": {"drop_rows": 0}, "pergamos": {"drop_rows": 0}}}, ensure_ascii=False))
    phd_summary = tmp_path / "phd_summary.json"
    phd_summary.write_text(json.dumps({"matched_oa_outside_phdtheses": 1}, ensure_ascii=False))
    recovery_summary = tmp_path / "recovery_summary.json"
    recovery_summary.write_text(json.dumps({"exact_title_author_rows": 1, "total_rows_recovered_if_exact_title_unique_fallback_added": 1}, ensure_ascii=False))
    review_summary = tmp_path / "review_summary.json"
    review_summary.write_text(json.dumps({"after_reduced_title_unique": 1, "after_all_tested_layers": 1}, ensure_ascii=False))

    analysis_mod.DEFAULT_OA_DROP_LIST_SUMMARY = oa_summary
    analysis_mod.DEFAULT_PHD_CROSS_COLLECTION_SUMMARY = phd_summary
    analysis_mod.DEFAULT_PERGAMOS_PHD_RECOVERY_SUMMARY = recovery_summary
    analysis_mod.DEFAULT_PERGAMOS_PHD_REVIEW_SUMMARY = review_summary
    analysis_mod.analyze_exact_run(run_root=run_root, output_dir=run_root / "analysis")

    payload = policy_mod.analyze_policy(run_root=run_root)
    assert Path(payload["policy_report_path"]).exists()
    pair_summary = (run_root / "analysis" / "policy_pair_summary.csv").read_text()
    assert "openarchives.gr" in pair_summary
    assert "greek_phd" in pair_summary
    report = (run_root / "analysis" / "policy_report.md").read_text()
    assert "greek_phd" in report


def test_policy_analysis_excludes_placeholder_only_cross_source_groups(tmp_path: Path) -> None:
    analysis_mod = load_analysis_module()
    policy_mod = load_policy_module()
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    rows = [
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-placeholder",
            "text": "<!-- image -->",
            "title": "OA placeholder",
            "author": "A",
            "source_metadata_json": json.dumps({"collection_slug": "Pandemos"}, ensure_ascii=False),
            "greek_badness_score": 10.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "opengov.gr-diaboyleuseis",
            "source_doc_id": "opengov-placeholder",
            "text": "<!-- image -->",
            "title": "Consultation placeholder",
            "author": "B",
            "source_metadata_json": "{}",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-real",
            "text": "Κοινό πραγματικό κείμενο",
            "title": "OA real",
            "author": "C",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT"}, ensure_ascii=False),
            "greek_badness_score": 4.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "greek_phd",
            "source_doc_id": "phd-real",
            "text": "Κοινό πραγματικό κείμενο",
            "title": "PhD real",
            "author": "D",
            "source_metadata_json": json.dumps({"handle_url": "http://example/phd-real"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_snapshot(input_root / "openarchives.gr.part-00000.parquet", [rows[0], rows[2]])
    write_snapshot(input_root / "opengov.gr-diaboyleuseis.parquet", [rows[1]])
    write_snapshot(input_root / "greek_phd.part-00000.parquet", [rows[3]])

    text_dedup.run_exact_dedup(input_root=input_root, state_root=state_root, run_root=run_root)

    oa_summary = tmp_path / "oa_summary.json"
    oa_summary.write_text(json.dumps({"drop_rows_total": 1, "origins": {"kallipos": {"drop_rows": 0}, "pergamos": {"drop_rows": 0}}}, ensure_ascii=False))
    phd_summary = tmp_path / "phd_summary.json"
    phd_summary.write_text(json.dumps({"matched_oa_outside_phdtheses": 1}, ensure_ascii=False))
    recovery_summary = tmp_path / "recovery_summary.json"
    recovery_summary.write_text(json.dumps({"exact_title_author_rows": 1, "total_rows_recovered_if_exact_title_unique_fallback_added": 1}, ensure_ascii=False))
    review_summary = tmp_path / "review_summary.json"
    review_summary.write_text(json.dumps({"after_reduced_title_unique": 1, "after_all_tested_layers": 1}, ensure_ascii=False))

    analysis_mod.DEFAULT_OA_DROP_LIST_SUMMARY = oa_summary
    analysis_mod.DEFAULT_PHD_CROSS_COLLECTION_SUMMARY = phd_summary
    analysis_mod.DEFAULT_PERGAMOS_PHD_RECOVERY_SUMMARY = recovery_summary
    analysis_mod.DEFAULT_PERGAMOS_PHD_REVIEW_SUMMARY = review_summary
    analysis_mod.analyze_exact_run(run_root=run_root, output_dir=run_root / "analysis")

    policy_mod.analyze_policy(run_root=run_root)

    pair_summary = pd.read_csv(run_root / "analysis" / "policy_pair_summary.csv")
    low_info = pd.read_csv(run_root / "analysis" / "policy_low_information_groups.csv")

    opengov_pair = pair_summary[
        (pair_summary["source_a"] == "openarchives.gr")
        & (pair_summary["source_b"] == "opengov.gr-diaboyleuseis")
    ].iloc[0]
    assert int(opengov_pair["total_rows_in_shared_groups"]) == 0
    assert int(opengov_pair["excluded_low_information_groups"]) == 1

    phd_pair = pair_summary[
        (
            (pair_summary["source_a"] == "greek_phd")
            & (pair_summary["source_b"] == "openarchives.gr")
        )
        | (
            (pair_summary["source_a"] == "openarchives.gr")
            & (pair_summary["source_b"] == "greek_phd")
        )
    ]
    assert not phd_pair.empty
    assert int(phd_pair.iloc[0]["total_rows_in_shared_groups"]) > 0

    assert "image_placeholder_only" in set(low_info["low_information_reason"])


def test_oa_internal_exact_sampler_emits_group_and_member_artifacts(tmp_path: Path) -> None:
    oa_sample_mod = load_oa_sample_module()
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()

    rows = [
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-1",
            "text": "Το ίδιο κείμενο",
            "title": "Τίτλος Α",
            "author": "Συγγραφέας Α",
            "source_metadata_json": json.dumps({"collection_slug": "Pandemos", "id": "x1"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-2",
            "text": "Το ίδιο κείμενο",
            "title": "Τίτλος Α",
            "author": "Συγγραφέας Α",
            "source_metadata_json": json.dumps({"collection_slug": "Pandemos", "id": "x2"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-3",
            "text": "<!-- image -->",
            "title": "Τίτλος Β",
            "author": "Συγγραφέας Β",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT", "id": "x3"}, ensure_ascii=False),
            "greek_badness_score": 3.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-4",
            "text": "<!-- image -->",
            "title": "Τίτλος Γ",
            "author": "Συγγραφέας Γ",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT", "id": "x4"}, ensure_ascii=False),
            "greek_badness_score": 4.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_snapshot(input_root / "openarchives.gr.part-00000.parquet", rows)

    text_dedup.run_exact_dedup(input_root=input_root, state_root=state_root, run_root=run_root)

    payload = oa_sample_mod.analyze_oa_internal_exact_sample(
        run_root=run_root,
        sample_size=2,
        seed=7,
    )

    assert Path(payload["oa_internal_exact_sample_groups_path"]).exists()
    assert Path(payload["oa_internal_exact_sample_members_path"]).exists()

    groups = pd.read_csv(payload["oa_internal_exact_sample_groups_path"])
    members = pd.read_csv(payload["oa_internal_exact_sample_members_path"])
    assert len(groups) == 2
    assert set(members["exact_strict_hash"]) == set(groups["group_hash"])
    assert "literal_duplicate_same_metadata" in set(groups["derived_reason"]) or "image_placeholder_only" in set(groups["derived_reason"])
    assert "oa_collection_slug" in members.columns


def test_oa_internal_collection_patterns_split_meaningful_from_placeholder(tmp_path: Path) -> None:
    patterns_mod = load_oa_patterns_module()
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()

    rows = [
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-a1",
            "text": "Κοινό αληθινό κείμενο",
            "title": "Τίτλος Α1",
            "author": "Συγγραφέας Α1",
            "source_metadata_json": json.dumps({"collection_slug": "Pandemos"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-a2",
            "text": "Κοινό αληθινό κείμενο",
            "title": "Τίτλος Α2",
            "author": "Συγγραφέας Α2",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-b1",
            "text": "<!-- image -->",
            "title": "Τίτλος Β1",
            "author": "Συγγραφέας Β1",
            "source_metadata_json": json.dumps({"collection_slug": "Pandemos"}, ensure_ascii=False),
            "greek_badness_score": 3.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-b2",
            "text": "<!-- image -->",
            "title": "Τίτλος Β2",
            "author": "Συγγραφέας Β2",
            "source_metadata_json": json.dumps({"collection_slug": "ntua"}, ensure_ascii=False),
            "greek_badness_score": 4.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_snapshot(input_root / "openarchives.gr.part-00000.parquet", rows)
    text_dedup.run_exact_dedup(input_root=input_root, state_root=state_root, run_root=run_root)

    payload = patterns_mod.analyze_openarchives_internal_patterns(run_root=run_root)
    cross = pd.read_csv(payload["oa_internal_cross_collection_pairs_meaningful_path"])
    reasons = pd.read_csv(payload["oa_internal_reason_summary_path"])

    assert ((cross["source_a"] == "IKEE_AUT") & (cross["source_b"] == "Pandemos")).any()
    assert "image_placeholder_only" in set(reasons["reason_bucket"])


def test_oa_meaningful_exact_analysis_splits_true_duplicate_from_metadata_conflict(tmp_path: Path) -> None:
    meaningful_mod = load_oa_meaningful_module()
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()

    rows = [
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-t1",
            "text": "Ακριβώς το ίδιο κείμενο α",
            "title": "Τίτλος Α",
            "author": "Συγγραφέας Α",
            "source_metadata_json": json.dumps({"collection_slug": "Pandemos"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-t2",
            "text": "Ακριβώς το ίδιο κείμενο α",
            "title": "Τίτλος Α",
            "author": "Συγγραφέας Α",
            "source_metadata_json": json.dumps({"collection_slug": "Pandemos"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-v1",
            "text": "Ακριβώς το ίδιο κείμενο β",
            "title": "Τίτλος Β",
            "author": "Συγγραφέας Β1",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-v2",
            "text": "Ακριβώς το ίδιο κείμενο β",
            "title": "Τίτλος Β",
            "author": "Συγγραφέας Β2",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-m1",
            "text": "Ακριβώς το ίδιο κείμενο γ",
            "title": "Τίτλος Γ1",
            "author": "Συγγραφέας Γ1",
            "source_metadata_json": json.dumps({"collection_slug": "ntua"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-m2",
            "text": "Ακριβώς το ίδιο κείμενο γ",
            "title": "Τίτλος Γ2",
            "author": "Συγγραφέας Γ2",
            "source_metadata_json": json.dumps({"collection_slug": "ntua"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-n1",
            "text": "<!-- image -->",
            "title": "Τίτλος Ν1",
            "author": "Συγγραφέας Ν1",
            "source_metadata_json": json.dumps({"collection_slug": "Dione"}, ensure_ascii=False),
            "greek_badness_score": 3.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-n2",
            "text": "<!-- image -->",
            "title": "Τίτλος Ν2",
            "author": "Συγγραφέας Ν2",
            "source_metadata_json": json.dumps({"collection_slug": "Dione"}, ensure_ascii=False),
            "greek_badness_score": 4.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_snapshot(input_root / "openarchives.gr.part-00000.parquet", rows)
    text_dedup.run_exact_dedup(input_root=input_root, state_root=state_root, run_root=run_root)

    payload = meaningful_mod.analyze_openarchives_meaningful_exact_groups(run_root=run_root)
    bucket_summary = pd.read_csv(payload["oa_meaningful_exact_bucket_summary_path"])
    details = pd.read_csv(payload["oa_meaningful_exact_group_details_path"])

    assert set(bucket_summary["analysis_bucket"]) == {
        "likely_true_duplicate",
        "likely_same_work_metadata_variation",
        "likely_text_metadata_attachment_problem",
    }
    assert "image_placeholder_only" not in set(details["derived_reason"])
    assert "literal_duplicate_same_metadata" in set(details["derived_reason"])
    assert "same_text_title_author_variation" in set(details["derived_reason"])
    assert "same_text_metadata_variation" in set(details["derived_reason"])


def test_oa_suspicious_exact_analysis_splits_notice_and_markup_patterns(tmp_path: Path) -> None:
    meaningful_mod = load_oa_meaningful_module()
    suspicious_mod = load_oa_suspicious_module()
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()

    rows = [
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-a1",
            "text": "Ο συγγραφέας δεν επιτρέπει την πρόσβαση στο πλήρες κείμενο της διατριβής του. Η έντυπη μορφή της διατριβής είναι διαθέσιμη.",
            "title": "Τίτλος Α1",
            "author": "Συγγραφέας Α1",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-a2",
            "text": "Ο συγγραφέας δεν επιτρέπει την πρόσβαση στο πλήρες κείμενο της διατριβής του. Η έντυπη μορφή της διατριβής είναι διαθέσιμη.",
            "title": "Τίτλος Α2",
            "author": "Συγγραφέας Α2",
            "source_metadata_json": json.dumps({"collection_slug": "Pandemos"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-b1",
            "text": "<!-- image -->\n\nΣΠΟΥΔΑΣΤΕΣ: Α\n\nΕΠΙΒΛΕΠΟΝΤΕΣ: Β",
            "title": "Τίτλος Β1",
            "author": "Συγγραφέας Β1",
            "source_metadata_json": json.dumps({"collection_slug": "elocus"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-b2",
            "text": "<!-- image -->\n\nΣΠΟΥΔΑΣΤΕΣ: Α\n\nΕΠΙΒΛΕΠΟΝΤΕΣ: Β",
            "title": "Τίτλος Β2",
            "author": "Συγγραφέας Β2",
            "source_metadata_json": json.dumps({"collection_slug": "elocus"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_snapshot(input_root / "openarchives.gr.part-00000.parquet", rows)
    text_dedup.run_exact_dedup(input_root=input_root, state_root=state_root, run_root=run_root)
    meaningful_mod.analyze_openarchives_meaningful_exact_groups(run_root=run_root)
    payload = suspicious_mod.analyze_openarchives_suspicious_exact_groups(run_root=run_root)
    summary = pd.read_csv(payload["oa_suspicious_exact_subreason_summary_path"])

    assert "author_denies_access_notice" in set(summary["suspicious_subreason"])
    assert "image_markup_mixed_text" in set(summary["suspicious_subreason"])


def test_oa_origin_audit_marks_true_file_and_no_true_file_groups(tmp_path: Path) -> None:
    meaningful_mod = load_oa_meaningful_module()
    suspicious_mod = load_oa_suspicious_module()
    origin_mod = load_oa_origin_audit_module()
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()

    rows = [
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-match-1",
            "text": "## Mesh Networks in Telemedicine\n\nΧρήστος Μαυρίδης\n\nThis thesis studies mesh networks in telemedicine applications.",
            "title": 'Δίκτυα πλέγματος "Mesh Networks" σε εφαρμογές τηλεϊατρικής',
            "author": "Μαυρίδης, Χρήστος",
            "source_metadata_json": json.dumps({"collection_slug": "ntua"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-match-2",
            "text": "## Mesh Networks in Telemedicine\n\nΧρήστος Μαυρίδης\n\nThis thesis studies mesh networks in telemedicine applications.",
            "title": "Σχέδιο προσαρμογής στην κλιματική αλλαγή για το δήμο Μεταμόρφωσης",
            "author": "Γκουτζιούπα, Ευαγγελία",
            "source_metadata_json": json.dumps({"collection_slug": "ntua"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-nomatch-1",
            "text": "Β. Granic'. L'actc de fondation d'un monastere ... Byzantine studies review text.",
            "title": "Τεχνολογία φόρτισης ηλεκτρικών οχημάτων",
            "author": "Ξενάκης, Σταύρος",
            "source_metadata_json": json.dumps({"collection_slug": "ntua"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-nomatch-2",
            "text": "Β. Granic'. L'actc de fondation d'un monastere ... Byzantine studies review text.",
            "title": "Ανάπτυξη μεθόδου επιβεβαίωσης γεωγραφικής προέλευσης δειγμάτων φακής",
            "author": "Κανλής, Γεώργιος",
            "source_metadata_json": json.dumps({"collection_slug": "ntua"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-notice-1",
            "text": "Ο συγγραφέας δεν επιτρέπει την πρόσβαση στο πλήρες κείμενο της διατριβής του. Η έντυπη μορφή της διατριβής είναι διαθέσιμη.",
            "title": "Τίτλος Ν1",
            "author": "Συγγραφέας Ν1",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "oa-notice-2",
            "text": "Ο συγγραφέας δεν επιτρέπει την πρόσβαση στο πλήρες κείμενο της διατριβής του. Η έντυπη μορφή της διατριβής είναι διαθέσιμη.",
            "title": "Τίτλος Ν2",
            "author": "Συγγραφέας Ν2",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_snapshot(input_root / "openarchives.gr.part-00000.parquet", rows)
    text_dedup.run_exact_dedup(input_root=input_root, state_root=state_root, run_root=run_root)
    meaningful_mod.analyze_openarchives_meaningful_exact_groups(run_root=run_root)
    suspicious_mod.analyze_openarchives_suspicious_exact_groups(run_root=run_root)
    payload = origin_mod.analyze_openarchives_suspicious_origin_audit(run_root=run_root)

    group_audit = pd.read_csv(payload["oa_suspicious_origin_group_audit_path"])
    row_markers = pd.read_csv(payload["oa_suspicious_origin_row_markers_path"])

    assert "likely_true_file_in_group" in set(group_audit["origin_resolution"])
    assert "no_plausible_true_file_in_group" in set(group_audit["origin_resolution"])
    assert "generic_notice_group" in set(group_audit["origin_resolution"])

    matched_group = group_audit[group_audit["origin_resolution"] == "likely_true_file_in_group"].iloc[0]
    assert matched_group["best_candidate_title"] == 'Δίκτυα πλέγματος "Mesh Networks" σε εφαρμογές τηλεϊατρικής'

    true_row = row_markers[row_markers["row_issue_marker"] == "candidate_true_file_for_shared_text"]
    assert not true_row.empty
    assert true_row.iloc[0]["title"] == 'Δίκτυα πλέγματος "Mesh Networks" σε εφαρμογές τηλεϊατρικής'

    no_true_rows = row_markers[row_markers["row_issue_marker"] == "no_true_file_detected_in_group"]
    assert not no_true_rows.empty


def test_oa_llm_semantic_review_integration_tracks_heuristic_mismatches(tmp_path: Path) -> None:
    llm_mod = load_oa_llm_review_module()
    run_root = tmp_path / "run"
    analysis_dir = run_root / "analysis"
    packet_dir = analysis_dir / "oa_llm_review_packets"
    packet_dir.mkdir(parents=True)

    packet = {
        "packet_id": 1,
        "group_hash": "group-alpha",
        "group_size": 3,
        "collections_in_group": "IKEE_AUT:3",
        "suspicious_subreason": "other_content_collision",
        "shared_text": "Ο συγγραφέας επιτρέπει την πρόσβαση στο πλήρες κείμενο της μεταπτυχιακής του από 1/5/2020",
        "members": [
            {
                "source_doc_id": "doc-a",
                "oa_collection_slug": "IKEE_AUT",
                "title": "Σωστός τίτλος",
                "author": "Σωστός Συγγραφέας",
            },
            {
                "source_doc_id": "doc-b",
                "oa_collection_slug": "IKEE_AUT",
                "title": "Άσχετος τίτλος",
                "author": "Άλλος",
            },
            {
                "source_doc_id": "doc-c",
                "oa_collection_slug": "IKEE_AUT",
                "title": "Άλλος τίτλος",
                "author": "Άλλος",
            },
        ],
    }
    packet_path = packet_dir / "01_group-alpha.json"
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2))
    (packet_dir / "index.json").write_text(
        json.dumps(
            [
                {
                    "packet_id": 1,
                    "group_hash": "group-alpha",
                    "group_size": 3,
                    "collections_in_group": "IKEE_AUT:3",
                    "representative_title": "Σωστός τίτλος",
                    "packet_path": str(packet_path),
                }
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    (analysis_dir / "oa_suspicious_origin_group_audit.csv").write_text(
        "\n".join(
            [
                "group_hash,group_size,suspicious_subreason,origin_resolution,best_candidate_source_doc_id,best_candidate_title",
                "group-alpha,3,other_content_collision,no_plausible_true_file_in_group,doc-a,Σωστός τίτλος",
            ]
        )
        + "\n"
    )
    review_input = analysis_dir / "review_input.json"
    review_input.write_text(
        json.dumps(
            {
                "review_batch_id": "batch_demo",
                "reviews": [
                    {
                        "packet_id": 1,
                        "reviewer": "Pauli",
                        "llm_owner_resolution": "best_candidate_in_group_owner",
                        "llm_shared_text_type": "repository_notice_leakage",
                        "confidence": "high",
                        "best_candidate_row_index": 1,
                        "reasoning_summary": "This is clearly notice leakage belonging to the thesis row.",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    payload = llm_mod.integrate_openarchives_llm_semantic_review(run_root=run_root, review_input=review_input)
    review_frame = pd.read_csv(payload["review_csv_path"])
    summary_frame = pd.read_csv(payload["summary_csv_path"])

    assert review_frame.iloc[0]["llm_best_candidate_source_doc_id"] == "doc-a"
    assert bool(review_frame.iloc[0]["heuristic_agrees_owner_resolution"]) is False
    assert summary_frame.iloc[0]["heuristic_resolution_mismatches"] == 1


def test_oa_semantic_pipeline_gates_notices_and_packetizes_substantive_groups(tmp_path: Path) -> None:
    meaningful_mod = load_oa_meaningful_module()
    suspicious_mod = load_oa_suspicious_module()
    semantic_mod = load_oa_semantic_pipeline_module()
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()

    shared_notice = "Ο συγγραφέας επιτρέπει την πρόσβαση στο πλήρες κείμενο της μεταπτυχιακής του από 1/5/2020"
    shared_article = (
        "## ΤΟ ΦΑΙΝΟΜΕΝΟ ΤΗΣ ΑΡΙΣΤΕΡΟΣΤΡΟΦΗΣ ΚΕΦΑΛΗΣ ΣΤΗ ΣΥΝΘΕΣΗ ΤΗΣ ΚΑΤΩΙΤΑΛΙΚΗΣ\n\n"
        "ΜΑΡΙΟΣ ΑΝΔΡΕΟΥ\n\n## Abstract\n\nThe purpose of this paper is to discuss historical review material.\n\n"
        "## 1. Εισαγωγή\n\nΚείμενο με ουσιαστικό περιεχόμενο και βιβλιογραφικές αναφορές.\n\n"
        "## 2. Βιβλιογραφία\n\nAndriotis 1939.\n"
    )
    rows = [
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "notice-1",
            "text": shared_notice,
            "title": "Τίτλος Ν1",
            "author": "Συγγραφέας Ν1",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT", "Περιγραφή": "notice thesis"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "notice-2",
            "text": shared_notice,
            "title": "Τίτλος Ν2",
            "author": "Συγγραφέας Ν2",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT", "Περιγραφή": "notice thesis 2"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "article-1",
            "text": shared_article,
            "title": "Άσχετος τίτλος Α1",
            "author": "Συγγραφέας Α1",
            "source_metadata_json": json.dumps({"collection_slug": "ntua", "Περιγραφή": "mismatch one"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "article-2",
            "text": shared_article,
            "title": "Άσχετος τίτλος Α2",
            "author": "Συγγραφέας Α2",
            "source_metadata_json": json.dumps({"collection_slug": "ntua", "Περιγραφή": "mismatch two"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "article-3",
            "text": shared_article,
            "title": "Άσχετος τίτλος Α3",
            "author": "Συγγραφέας Α3",
            "source_metadata_json": json.dumps({"collection_slug": "ntua", "Περιγραφή": "mismatch three"}, ensure_ascii=False),
            "greek_badness_score": 3.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_snapshot(input_root / "openarchives.gr.part-00000.parquet", rows)
    text_dedup.run_exact_dedup(input_root=input_root, state_root=state_root, run_root=run_root)
    meaningful_mod.analyze_openarchives_meaningful_exact_groups(run_root=run_root)
    suspicious_mod.analyze_openarchives_suspicious_exact_groups(run_root=run_root)
    payload = semantic_mod.prepare_openarchives_semantic_resolution_pipeline(run_root=run_root, max_rows_per_packet=2)

    profiles = pd.read_csv(payload["oa_semantic_resolution_group_profiles_path"])
    packet_index = pd.read_csv(payload["oa_gemini_resolution_packet_index_csv_path"])

    assert "clear_notice_like_output" in set(profiles["deterministic_resolution_status"])
    assert "needs_semantic_review" in set(profiles["deterministic_resolution_status"])
    notice_row = profiles[profiles["deterministic_resolution_status"] == "clear_notice_like_output"].iloc[0]
    assert notice_row["semantic_review_path"] == "skip_clear_deterministic_output"
    substantive_row = profiles[profiles["deterministic_resolution_status"] == "needs_semantic_review"].iloc[0]
    assert substantive_row["semantic_review_path"] == "split_group_batches"
    assert substantive_row["chars_plain_body"] > 100
    assert len(packet_index) == 2


def test_notice_bucket_requires_notice_like_shape() -> None:
    semantic_mod = load_oa_semantic_pipeline_module()

    short_notice_text = "Ο συγγραφέας επιτρέπει την πρόσβαση στο πλήρες κείμενο της εργασίας μετά την 2027-01-01."
    short_notice_profile = semantic_mod.build_shared_text_shape_metrics(short_notice_text)
    assert (
        semantic_mod.classify_deterministic_resolution_status(
            short_notice_text,
            "embargo_release_notice",
            short_notice_profile,
        )
        == "clear_notice_like_output"
    )

    long_structured_text = (
        "Ο συγγραφέας επιτρέπει την πρόσβαση στο πλήρες κείμενο της εργασίας μετά την 2027-01-01.\n\n"
        + "\n\n".join(
            [
                f"## Κεφάλαιο {index}\n"
                + ("Αυτό είναι αναλυτικό σώμα κειμένου με πραγματικό περιεχόμενο. " * 80)
                for index in range(1, 6)
            ]
        )
    )
    long_structured_profile = semantic_mod.build_shared_text_shape_metrics(long_structured_text)
    assert long_structured_profile["chars_plain_body"] > 1_000
    assert long_structured_profile["header_count"] >= 5
    assert (
        semantic_mod.classify_deterministic_resolution_status(
            long_structured_text,
            "other_content_collision",
            long_structured_profile,
        )
        == "needs_semantic_review"
    )


def test_oa_high_level_content_review_integration_writes_summary(tmp_path: Path) -> None:
    review_mod = load_oa_high_level_review_module()
    run_root = tmp_path / "run"
    analysis_dir = run_root / "analysis"
    packet_dir = analysis_dir / "oa_llm_review_packets"
    packet_dir.mkdir(parents=True)

    packet = {
        "packet_id": 1,
        "group_hash": "group-alpha",
        "group_size": 3,
        "collections_in_group": "ntua:3",
        "shared_text": "## Abstract\n\nΚείμενο",
        "members": [],
    }
    packet_path = packet_dir / "01_group-alpha.json"
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2))
    (packet_dir / "index.json").write_text(
        json.dumps(
            [
                {
                    "packet_id": 1,
                    "group_hash": "group-alpha",
                    "group_size": 3,
                    "collections_in_group": "ntua:3",
                    "representative_title": "Άλφα",
                    "packet_path": str(packet_path),
                }
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    review_input = analysis_dir / "content_review_input.json"
    review_input.write_text(
        json.dumps(
            {
                "review_batch_id": "batch_content_demo",
                "reviews": [
                    {
                        "packet_id": 1,
                        "reviewer": "Galileo",
                        "high_level_content_type": "full_thesis_or_article",
                        "is_substantive_content": True,
                        "confidence": "high",
                        "reasoning_summary": "Abstract plus structured body.",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    payload = review_mod.integrate_openarchives_high_level_content_review(run_root=run_root, review_input=review_input)
    review_frame = pd.read_csv(payload["review_csv_path"])
    summary_frame = pd.read_csv(payload["summary_csv_path"])

    assert review_frame.iloc[0]["high_level_content_type"] == "full_thesis_or_article"
    assert bool(review_frame.iloc[0]["is_substantive_content"]) is True
    assert summary_frame.iloc[0]["group_count"] == 1


def test_oa_codex_content_review_bundle_materializes_runner(tmp_path: Path) -> None:
    bundle_mod = load_oa_codex_bundle_module()
    run_root = tmp_path / "run"
    packet_dir = run_root / "analysis" / "oa_gemini_resolution_packets"
    packet_dir.mkdir(parents=True)

    packet = {
        "packet_family_id": "groupalpha",
        "group_hash": "group-alpha",
        "group_size": 2,
        "packet_mode": "single_batch",
        "batch_index": 1,
        "batch_count": 1,
        "collections_in_group": "ntua:2",
        "suspicious_subreason": "other_content_collision",
        "deterministic_resolution_status": "needs_semantic_review",
        "content_size_hint": "article_sized",
        "shared_text_profile": {"text_chars": 1200, "header_count": 3},
        "shared_text": "## Abstract\n\nΚείμενο με ουσιαστικό περιεχόμενο.",
        "rows": [
            {"row_index_in_batch": 1, "row_index_in_group": 1, "source_doc_id": "doc-a", "title": "Α", "author": "Α", "source_metadata": {}, "semantic_metadata": {}, "metadata_preview": "{}"},
            {"row_index_in_batch": 2, "row_index_in_group": 2, "source_doc_id": "doc-b", "title": "Β", "author": "Β", "source_metadata": {}, "semantic_metadata": {}, "metadata_preview": "{}"},
        ],
    }
    packet_path = packet_dir / "0001_groupalpha_b01.json"
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2))
    (packet_dir / "index.json").write_text(
        json.dumps(
            [
                {
                    "packet_id": 1,
                    "packet_path": str(packet_path),
                    "packet_mode": "single_batch",
                    "batch_index": 1,
                    "batch_count": 1,
                    "group_hash": "group-alpha",
                    "group_size": 2,
                    "rows_in_packet": 2,
                    "deterministic_resolution_status": "needs_semantic_review",
                    "content_size_hint": "article_sized",
                    "representative_title": "Α",
                }
            ],
            ensure_ascii=False,
            indent=2,
        )
    )

    payload = bundle_mod.materialize_openarchives_codex_content_review_bundle(run_root=run_root, bundle_name="bundle_demo", overwrite=True)
    runner_path = Path(payload["runner_path"])
    jobs_jsonl_path = Path(payload["jobs_jsonl_path"])
    assert runner_path.exists()
    assert jobs_jsonl_path.exists()
    assert "codex exec" in runner_path.read_text(encoding="utf-8")


def test_oa_text_shape_research_writes_bucket_summary(tmp_path: Path) -> None:
    meaningful_mod = load_oa_meaningful_module()
    suspicious_mod = load_oa_suspicious_module()
    semantic_mod = load_oa_semantic_pipeline_module()
    research_mod = load_oa_text_shape_research_module()
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()

    rows = [
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "n1",
            "text": "Ο συγγραφέας επιτρέπει την πρόσβαση στο πλήρες κείμενο της μεταπτυχιακής του από 1/5/2020",
            "title": "Τίτλος Ν1",
            "author": "Συγγραφέας Ν1",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "n2",
            "text": "Ο συγγραφέας επιτρέπει την πρόσβαση στο πλήρες κείμενο της μεταπτυχιακής του από 1/5/2020",
            "title": "Τίτλος Ν2",
            "author": "Συγγραφέας Ν2",
            "source_metadata_json": json.dumps({"collection_slug": "IKEE_AUT"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "a1",
            "text": "## Abstract\n\nΚείμενο ουσιαστικού περιεχομένου.\n\n| A | B |\n|---|---|\n| 1 | 2 |",
            "title": "Άσχετο Α1",
            "author": "Α1",
            "source_metadata_json": json.dumps({"collection_slug": "ntua"}, ensure_ascii=False),
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "openarchives.gr",
            "source_doc_id": "a2",
            "text": "## Abstract\n\nΚείμενο ουσιαστικού περιεχομένου.\n\n| A | B |\n|---|---|\n| 1 | 2 |",
            "title": "Άσχετο Α2",
            "author": "Α2",
            "source_metadata_json": json.dumps({"collection_slug": "ntua"}, ensure_ascii=False),
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_snapshot(input_root / "openarchives.gr.part-00000.parquet", rows)
    text_dedup.run_exact_dedup(input_root=input_root, state_root=state_root, run_root=run_root)
    meaningful_mod.analyze_openarchives_meaningful_exact_groups(run_root=run_root)
    suspicious_mod.analyze_openarchives_suspicious_exact_groups(run_root=run_root)
    semantic_mod.prepare_openarchives_semantic_resolution_pipeline(run_root=run_root, max_rows_per_packet=8)
    payload = research_mod.analyze_openarchives_text_shape_research(run_root=run_root)

    bucket_summary = pd.read_csv(payload["oa_text_shape_summary_by_bucket_path"])
    assert "clear_notice_like_output" in set(bucket_summary["deterministic_resolution_status"])
