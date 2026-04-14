from __future__ import annotations

import json
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from glossapi_corpus_cli import cli
from glossapi_corpus_cli import text_dedup


def write_test_snapshot(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, row_group_size=2)


def make_test_row(source_dataset: str, source_doc_id: str, text: str) -> dict[str, object]:
    return {
        "source_dataset": source_dataset,
        "source_doc_id": source_doc_id,
        "text": text,
        "title": f"title-{source_doc_id}",
        "author": f"author-{source_doc_id}",
        "greek_badness_score": 0.0,
        "mojibake_badness_score": 0.0,
        "needs_ocr": False,
        "is_empty": False,
        "ocr_success": True,
        "is_historical_or_polytonic": False,
    }


def reference_minhash_signature(shingle_hashes: list[int], *, num_perm: int) -> np.ndarray:
    a_values, b_values = text_dedup.permutation_params(num_perm)
    signature: list[int] = []
    for perm_index in range(num_perm):
        a_value = int(a_values[perm_index])
        b_value = int(b_values[perm_index])
        best = text_dedup.MINHASH_MODULUS
        for shingle_hash in shingle_hashes:
            candidate = ((a_value * (int(shingle_hash) % text_dedup.MINHASH_MODULUS)) + b_value) % text_dedup.MINHASH_MODULUS
            if candidate < best:
                best = candidate
        signature.append(best)
    return np.array(signature, dtype=np.uint64)


def build_signature_metadata_lookup(
    rows: dict[str, dict[str, object]],
) -> text_dedup.SignatureMetadataLookup:
    ordered_doc_keys = list(rows)
    metadata_values = np.array(
        [
            (
                int(rows[doc_key]["token_count"]),
                int(rows[doc_key]["char_count"]),
                int(rows[doc_key].get("near_text_chars", rows[doc_key]["char_count"])),
                int(rows[doc_key].get("shingle_count", 0)),
                int(rows[doc_key].get("shingle_size", 0)),
                str(rows[doc_key].get("shingle_mode", "token")),
            )
            for doc_key in ordered_doc_keys
        ],
        dtype=text_dedup.signature_metadata_dtype(),
    )
    return text_dedup.SignatureMetadataLookup(
        row_by_doc_key={doc_key: index for index, doc_key in enumerate(ordered_doc_keys)},
        values=metadata_values,
    )


def test_prepare_test_dedup_run_materializes_head_sample_and_launcher(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    experiment_root = tmp_path / "experiment"
    write_test_snapshot(
        input_root / "a.parquet",
        [
            make_test_row("alpha", "a-1", "κείμενο α1"),
            make_test_row("alpha", "a-2", "κείμενο α2"),
            make_test_row("alpha", "a-3", "κείμενο α3"),
            make_test_row("alpha", "a-4", "κείμενο α4"),
        ],
    )
    write_test_snapshot(
        input_root / "nested" / "b.parquet",
        [
            make_test_row("beta", "b-1", "κείμενο β1"),
            make_test_row("beta", "b-2", "κείμενο β2"),
        ],
    )

    payload = text_dedup.prepare_test_dedup_run(
        experiment_root=experiment_root,
        input_root=input_root,
        rows_per_file=3,
        max_workers=3,
        greek_diacritic_policy="strip",
        exact_only=True,
    )

    sampled_a = pq.read_table(experiment_root / "input" / "a.parquet").to_pylist()
    sampled_b = pq.read_table(experiment_root / "input" / "nested" / "b.parquet").to_pylist()
    assert [row["source_doc_id"] for row in sampled_a] == ["a-1", "a-2", "a-3"]
    assert [row["source_doc_id"] for row in sampled_b] == ["b-1", "b-2"]
    assert payload["sampling"]["prepared_rows"] == 5
    assert payload["sampling"]["selected_file_count"] == 2

    launch_script_path = Path(payload["launch_script_path"])
    script_text = launch_script_path.read_text()
    assert launch_script_path.stat().st_mode & 0o111
    assert "--exact-only" in script_text
    assert "--greek-diacritic-policy strip" in script_text
    assert str(experiment_root / "input") in script_text
    assert str(experiment_root / "run_current") in script_text

    summary_payload = json.loads(Path(payload["summary_path"]).read_text())
    assert summary_payload["run_command"] == payload["run_command"]
    assert len(summary_payload["sampled_files"]) == 2
    assert summary_payload["sampled_files"][0]["relative_path"] == "a.parquet"
    assert summary_payload["sampled_files"][1]["relative_path"] == "nested/b.parquet"


def test_prepare_test_run_cli_emits_json_payload(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    experiment_root = tmp_path / "prepared"
    write_test_snapshot(
        input_root / "demo.parquet",
        [
            make_test_row("demo", "doc-1", "κείμενο 1"),
            make_test_row("demo", "doc-2", "κείμενο 2"),
        ],
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "dedup-text",
            "prepare-test-run",
            "--experiment-root",
            str(experiment_root),
            "--input-root",
            str(input_root),
            "--rows-per-file",
            "1",
            "--max-workers",
            "2",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["sampling"]["prepared_rows"] == 1
    assert Path(payload["launch_script_path"]).exists()
    assert "glossapi-corpus dedup-text run" in Path(payload["launch_script_path"]).read_text()


def read_exact_survivor_rows(run_root: Path) -> list[dict[str, object]]:
    manifest_path = run_root / "stage_01_exact" / "exact_survivor_manifest.parquet"
    rows: list[dict[str, object]] = []
    for manifest_row in pq.read_table(manifest_path).to_pylist():
        rows.extend(pq.read_table(Path(str(manifest_row["shard_path"]))).to_pylist())
    return rows


def test_relaxed_normalization_preserves_polytonic_and_monotonic_distinction_by_default() -> None:
    polytonic = text_dedup.normalize_exact_relaxed_text(text_dedup.normalize_exact_strict_text("Ἄνθρωπος"))
    monotonic = text_dedup.normalize_exact_relaxed_text(text_dedup.normalize_exact_strict_text("Άνθρωπος"))
    assert polytonic != monotonic


def test_relaxed_normalization_preserves_polytonic_and_accentless_distinction_by_default() -> None:
    polytonic = text_dedup.normalize_exact_relaxed_text(text_dedup.normalize_exact_strict_text("Ἄνθρωπος"))
    accentless = text_dedup.normalize_exact_relaxed_text(text_dedup.normalize_exact_strict_text("Ανθρωπος"))
    assert polytonic != accentless


def test_relaxed_normalization_strip_mode_collapses_polytonic_and_monotonic_forms() -> None:
    polytonic = text_dedup.normalize_exact_relaxed_text(
        text_dedup.normalize_exact_strict_text("Ἄνθρωπος"),
        greek_diacritic_policy="strip",
    )
    monotonic = text_dedup.normalize_exact_relaxed_text(
        text_dedup.normalize_exact_strict_text("Άνθρωπος"),
        greek_diacritic_policy="strip",
    )
    assert polytonic == monotonic


def test_relaxed_normalization_strip_mode_collapses_polytonic_and_accentless_forms() -> None:
    polytonic = text_dedup.normalize_exact_relaxed_text(
        text_dedup.normalize_exact_strict_text("Ἄνθρωπος"),
        greek_diacritic_policy="strip",
    )
    accentless = text_dedup.normalize_exact_relaxed_text(
        text_dedup.normalize_exact_strict_text("Ανθρωπος"),
        greek_diacritic_policy="strip",
    )
    assert polytonic == accentless


def test_relaxed_normalization_collapses_unicode_punctuation_and_separators() -> None:
    normalized = text_dedup.normalize_exact_relaxed_text(
        text_dedup.normalize_exact_strict_text("«Καλημέρα»\u00a0κόσμε·"),
    )
    assert normalized == "καλημέρα κόσμε"


def test_relaxed_normalization_removes_soft_hyphen() -> None:
    normalized = text_dedup.normalize_exact_relaxed_text(
        text_dedup.normalize_exact_strict_text("παρά\u00adδειγμα"),
    )
    assert normalized == "παράδειγμα"


def test_near_normalization_dehyphenates_line_wrap_forms() -> None:
    wrapped = text_dedup.normalize_near_text("παρά-\nδειγμα")
    plain = text_dedup.normalize_near_text("παράδειγμα")
    relaxed_wrapped = text_dedup.normalize_exact_relaxed_text(
        text_dedup.normalize_exact_strict_text("παρά-\nδειγμα"),
    )

    assert wrapped == plain
    assert relaxed_wrapped != plain


def test_unicode_word_tokens_follow_regex_word_token_spec() -> None:
    decomposed = unicodedata.normalize("NFD", "Λόγος")
    tokens = text_dedup.unicode_word_tokens(f"«{decomposed}»\u00a0καὶ 123")
    assert tokens == [decomposed, "καὶ", "123"]


def test_minhash_signature_matches_reference_modular_hash_family() -> None:
    shingle_hashes = [0, 1, 2, (1 << 61) - 2, (1 << 63) + 12345, (1 << 64) - 1]
    signature = text_dedup.minhash_signature(shingle_hashes, num_perm=8)
    expected = reference_minhash_signature(shingle_hashes, num_perm=8)
    assert np.array_equal(signature, expected)


def test_exact_dedup_run_strict_and_relaxed(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    rows = [
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-1",
            "text": "Καλημέρα   κόσμε",
            "title": "Α",
            "author": "X",
            "greek_badness_score": 3.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-2",
            "text": "Καλημέρα κόσμε",
            "title": "Α",
            "author": "X",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-3",
            "text": "ΚΑΛΗΜΈΡΑ - ΚΌΣΜΕ",
            "title": "Α",
            "author": "X",
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-4",
            "text": "Ἄνθρωπος",
            "title": "Β",
            "author": "Y",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": True,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-5",
            "text": "Άνθρωπος",
            "title": "Γ",
            "author": "Y",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-6",
            "text": "Ανθρωπος",
            "title": "Δ",
            "author": "Y",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_test_snapshot(input_root / "demo.parquet", rows)
    summary = text_dedup.run_exact_dedup(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        max_workers=2,
    )
    assert summary["total_rows"] == 6
    assert summary["strict"]["dropped_rows"] == 1
    assert summary["relaxed"]["dropped_rows"] == 1
    assert summary["kept_after_exact_rows"] == 4
    docs_exact = pq.read_table(run_root / "stage_01_exact" / "docs_exact.parquet").to_pylist()
    kept_rows = [row for row in docs_exact if row["kept_after_exact"]]
    kept_ids = {row["source_doc_id"] for row in kept_rows}
    assert kept_ids == {"doc-2", "doc-4", "doc-5", "doc-6"}


def test_exact_dedup_run_strip_mode_collapses_polytonic_monotonic_and_accentless_variants(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    rows = [
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-1",
            "text": "Καλημέρα   κόσμε",
            "title": "Α",
            "author": "X",
            "greek_badness_score": 3.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-2",
            "text": "Καλημέρα κόσμε",
            "title": "Α",
            "author": "X",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-3",
            "text": "ΚΑΛΗΜΈΡΑ - ΚΌΣΜΕ",
            "title": "Α",
            "author": "X",
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-4",
            "text": "Ἄνθρωπος",
            "title": "Β",
            "author": "Y",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": True,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-5",
            "text": "Άνθρωπος",
            "title": "Γ",
            "author": "Y",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-6",
            "text": "Ανθρωπος",
            "title": "Δ",
            "author": "Y",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_test_snapshot(input_root / "demo.parquet", rows)
    summary = text_dedup.run_exact_dedup(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        max_workers=2,
        greek_diacritic_policy="strip",
    )
    assert summary["total_rows"] == 6
    assert summary["strict"]["dropped_rows"] == 1
    assert summary["relaxed"]["dropped_rows"] == 3
    assert summary["kept_after_exact_rows"] == 2
    kept_rows = [
        row
        for row in pq.read_table(run_root / "stage_01_exact" / "docs_exact.parquet").to_pylist()
        if row["kept_after_exact"]
    ]
    kept_ids = {row["source_doc_id"] for row in kept_rows}
    assert kept_ids == {"doc-2", "doc-4"}


def test_exact_dedup_reuses_cache_for_unchanged_docs(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root_1 = tmp_path / "run-1"
    run_root_2 = tmp_path / "run-2"
    input_root.mkdir()
    rows = [
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-1",
            "text": "Καλημέρα κόσμε",
            "title": "Α",
            "author": "X",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-2",
            "text": "Άνθρωπος",
            "title": "Β",
            "author": "Y",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_test_snapshot(input_root / "demo.parquet", rows)
    first = text_dedup.run_exact_dedup(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root_1,
        max_workers=2,
    )
    second = text_dedup.run_exact_dedup(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root_2,
        max_workers=2,
    )
    assert first["reused_exact_rows"] == 0
    assert second["reused_exact_rows"] == 2


def test_exact_dedup_export_handles_late_all_null_string_columns(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    rows = [
        {
            "source_dataset": "demo",
            "source_doc_id": f"doc-{idx:04d}",
            "text": f"Κείμενο {idx}",
            "title": None if idx >= 2048 else f"Τίτλος {idx}",
            "author": None if idx >= 2048 else f"Συγγραφέας {idx}",
            "greek_badness_score": float(idx % 3),
            "mojibake_badness_score": None,
            "needs_ocr": None,
            "is_empty": False,
            "ocr_success": None,
            "is_historical_or_polytonic": False,
        }
        for idx in range(4096)
    ]
    write_test_snapshot(input_root / "demo.parquet", rows)
    summary = text_dedup.run_exact_dedup(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        max_workers=1,
    )
    assert summary["total_rows"] == 4096
    docs_exact = pq.read_table(run_root / "stage_01_exact" / "docs_exact.parquet")
    assert docs_exact.num_rows == 4096
    assert docs_exact.schema.field("title").type == pa.string()
    assert docs_exact.schema.field("author").type == pa.string()


def test_exact_dedup_validates_required_input_columns(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    table = pa.Table.from_pylist(
        [
            {
                "source_dataset": "demo",
                "source_doc_id": "doc-1",
                "text": "Καλημέρα",
            }
        ]
    )
    pq.write_table(table, input_root / "broken.parquet")
    try:
        text_dedup.run_exact_dedup(
            input_root=input_root,
            state_root=state_root,
            run_root=run_root,
            max_workers=1,
        )
    except ValueError as exc:
        assert "schema validation failed" in str(exc)
        assert "title" in str(exc)
    else:
        raise AssertionError("expected schema validation failure for missing required columns")


def test_discover_input_files_recurses_into_nested_snapshot_dirs(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    nested_root = input_root / "nested" / "source"
    nested_root.mkdir(parents=True)
    write_test_snapshot(
        nested_root / "demo.parquet",
        [
            {
                "source_dataset": "demo",
                "source_doc_id": "doc-1",
                "text": "Καλημέρα κόσμε",
                "title": "Α",
                "author": "X",
                "greek_badness_score": 1.0,
                "mojibake_badness_score": 0.0,
                "needs_ocr": False,
                "is_empty": False,
                "ocr_success": True,
                "is_historical_or_polytonic": False,
            }
        ],
    )
    files = text_dedup.discover_input_files(input_root)
    assert len(files) == 1
    assert files[0].path == nested_root / "demo.parquet"
    assert files[0].fingerprint


def test_resume_rejects_changed_input_snapshot_under_same_run_root(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    snapshot_path = input_root / "demo.parquet"
    write_test_snapshot(
        snapshot_path,
        [
            {
                "source_dataset": "demo",
                "source_doc_id": "doc-1",
                "text": "Καλημέρα κόσμε",
                "title": "Α",
                "author": "X",
                "greek_badness_score": 1.0,
                "mojibake_badness_score": 0.0,
                "needs_ocr": False,
                "is_empty": False,
                "ocr_success": True,
                "is_historical_or_polytonic": False,
            }
        ],
    )
    text_dedup.run_exact_dedup(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        max_workers=1,
    )

    write_test_snapshot(
        snapshot_path,
        [
            {
                "source_dataset": "demo",
                "source_doc_id": "doc-1",
                "text": "Καλημέρα τροποποιημένε κόσμε",
                "title": "Α",
                "author": "X",
                "greek_badness_score": 1.0,
                "mojibake_badness_score": 0.0,
                "needs_ocr": False,
                "is_empty": False,
                "ocr_success": True,
                "is_historical_or_polytonic": False,
            }
        ],
    )

    with pytest.raises(ValueError, match="input snapshot changed"):
        text_dedup.run_exact_dedup(
            input_root=input_root,
            state_root=state_root,
            run_root=run_root,
            resume=True,
            max_workers=1,
        )


def test_resume_rejects_large_component_threshold_drift(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    write_test_snapshot(
        input_root / "demo.parquet",
        [
            {
                "source_dataset": "demo",
                "source_doc_id": "doc-1",
                "text": " ".join(f"λέξη{i}" for i in range(40)),
                "title": "Α",
                "author": "X",
                "greek_badness_score": 1.0,
                "mojibake_badness_score": 0.0,
                "needs_ocr": False,
                "is_empty": False,
                "ocr_success": True,
                "is_historical_or_polytonic": False,
            }
        ],
    )
    text_dedup.run_dedup_pipeline(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        exact_only=True,
        large_component_threshold=10,
        max_workers=1,
    )

    with pytest.raises(ValueError, match="different config hash"):
        text_dedup.run_dedup_pipeline(
            input_root=input_root,
            state_root=state_root,
            run_root=run_root,
            resume=True,
            exact_only=True,
            large_component_threshold=11,
            max_workers=1,
        )


def test_resume_reuses_completed_exact_stage_even_if_progress_marker_was_overwritten(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    write_test_snapshot(
        input_root / "demo.parquet",
        [
            make_test_row("demo", "doc-1", "Καλημέρα κόσμε"),
            make_test_row("demo", "doc-2", "Καλημέρα κόσμε"),
            make_test_row("demo", "doc-3", "Άνθρωπος"),
        ],
    )
    first = text_dedup.run_exact_dedup(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        max_workers=1,
    )
    inventory_path = run_root / "run_docs_inventory.parquet"
    docs_exact_path = run_root / "stage_01_exact" / "docs_exact.parquet"
    summary_path = run_root / "stage_01_exact" / "summary.json"
    original_mtimes = {
        inventory_path: inventory_path.stat().st_mtime_ns,
        docs_exact_path: docs_exact_path.stat().st_mtime_ns,
        summary_path: summary_path.stat().st_mtime_ns,
    }
    text_dedup.write_json_atomic(
        run_root / "progress" / "stage_01_exact.json",
        {
            "run_id": run_root.name,
            "stage": "stage_01_exact",
            "status": "running",
            "total_chunks": 2,
            "completed_chunks": 2,
        },
    )

    second = text_dedup.run_exact_dedup(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        resume=True,
        max_workers=1,
    )

    assert first["total_rows"] == second["total_rows"] == 3
    for path, mtime_ns in original_mtimes.items():
        assert path.stat().st_mtime_ns == mtime_ns


def test_full_pipeline_resume_reuses_completed_stage_outputs(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    near_tokens = [f"λέξη{i}" for i in range(60)]
    near_tokens_variant = list(near_tokens)
    near_tokens_variant[20] = "παραλλαγή"
    rows = [
        make_test_row("demo", "exact-a", "Καλημέρα κόσμε"),
        make_test_row("demo", "exact-b", "Καλημέρα κόσμε"),
        make_test_row("demo", "near-a", " ".join(near_tokens)),
        make_test_row("demo", "near-b", " ".join(near_tokens_variant)),
    ]
    write_test_snapshot(input_root / "demo.parquet", rows)
    text_dedup.run_dedup_pipeline(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        max_workers=1,
        minhash_threshold=0.80,
        num_perm=128,
        bands=32,
        rows_per_band=4,
        shingle_mode="token",
        shingle_size=2,
    )

    tracked_paths = [
        run_root / "run_docs_inventory.parquet",
        run_root / "stage_02_near" / "signatures.parquet",
        run_root / "stage_02_near" / "candidate_pairs.parquet",
        run_root / "stage_02_near" / "near_clusters.parquet",
        run_root / "final" / "run_summary.json",
        run_root / "builder_metadata" / "manifest.json",
    ]
    original_mtimes = {path: path.stat().st_mtime_ns for path in tracked_paths}
    text_dedup.write_json_atomic(
        run_root / "progress" / "stage_01_exact.json",
        {
            "run_id": run_root.name,
            "stage": "stage_01_exact",
            "status": "running",
            "total_chunks": 2,
            "completed_chunks": 2,
        },
    )

    resumed = text_dedup.run_dedup_pipeline(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        resume=True,
        max_workers=1,
        minhash_threshold=0.80,
        num_perm=128,
        bands=32,
        rows_per_band=4,
        shingle_mode="token",
        shingle_size=2,
    )

    assert resumed["final"]["decision_rows"] == 4
    for path, mtime_ns in original_mtimes.items():
        assert path.stat().st_mtime_ns == mtime_ns


def test_char_shingle_mode_does_not_require_token_threshold() -> None:
    near_text = "abcdefghijklmnopqrstuvwxyz"
    shingle_hashes, token_count, char_count = text_dedup.shingle_hashes_from_text(
        near_text=near_text,
        shingle_mode="char",
        shingle_size=7,
    )
    assert token_count == 1
    assert char_count == len(near_text)
    assert len(shingle_hashes) > 0


def test_candidate_band_chunk_subdivides_oversized_buckets(tmp_path: Path) -> None:
    stage_root = tmp_path / "stage_02_near"
    band_path = stage_root / "shards" / "lsh_buckets" / "band_00" / "demo.parquet"
    band_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "doc_key": "doc-a",
                    "band_index": 0,
                    "bucket_hash": "same-bucket",
                    "token_count": 80,
                    "char_count": 400,
                    "shingle_mode": "token",
                },
                {
                    "doc_key": "doc-b",
                    "band_index": 0,
                    "bucket_hash": "same-bucket",
                    "token_count": 80,
                    "char_count": 400,
                    "shingle_mode": "token",
                },
                {
                    "doc_key": "doc-c",
                    "band_index": 0,
                    "bucket_hash": "same-bucket",
                    "token_count": 80,
                    "char_count": 400,
                    "shingle_mode": "token",
                },
            ]
        ),
        band_path,
    )
    signature_map = {
        "doc-a": np.array([1, 2, 3, 4], dtype=np.uint64),
        "doc-b": np.array([1, 2, 3, 4], dtype=np.uint64),
        "doc-c": np.array([1, 2, 3, 4], dtype=np.uint64),
    }
    signature_meta = build_signature_metadata_lookup(
        {
        doc_key: {
            "token_count": 80,
            "char_count": 400,
            "shingle_mode": "token",
        }
        for doc_key in signature_map
        }
    )

    result = text_dedup.build_candidate_band_chunk(
        stage_root=stage_root,
        band_index=0,
        bands=1,
        rows_per_band=4,
        minhash_threshold=0.80,
        signature_map=signature_map,
        signature_meta=signature_meta,
        max_bucket_size=2,
    )

    candidate_rows = []
    for path in sorted((stage_root / "shards" / "candidate_pairs" / "band_00").glob("*.parquet")):
        candidate_rows.extend(pq.read_table(path).to_pylist())
    assert result["row_count"] == 2
    assert result["oversized_bucket_count"] == 1
    assert result["oversized_bucket_member_rows"] == 3
    assert result["fallback_chunked_bucket_count"] == 1
    assert result["fallback_chunked_member_rows"] == 3
    assert {(row["doc_key_left"], row["doc_key_right"]) for row in candidate_rows} == {
        ("doc-a", "doc-b"),
        ("doc-b", "doc-c"),
    }


def test_full_dedup_pipeline_runs_stage_2_only_on_exact_survivors(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    near_tokens = [f"λέξη{i}" for i in range(120)]
    near_tokens_variant = list(near_tokens)
    near_tokens_variant[40] = "παραλλαγή"
    relaxed_tokens = [f"άλλο{i}" for i in range(30)]
    rows = [
        {
            "source_dataset": "demo",
            "source_doc_id": "relaxed-1",
            "text": "ΑΥΤΌ είναι - " + " ".join(relaxed_tokens),
            "title": "Τίτλος",
            "author": "Α",
            "greek_badness_score": 0.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "relaxed-2",
            "text": "αυτό είναι " + " ".join(relaxed_tokens),
            "title": "Τίτλος",
            "author": "Α",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "near-1",
            "text": " ".join(near_tokens),
            "title": "Κείμενο",
            "author": "Συγγραφέας",
            "greek_badness_score": 0.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "near-2",
            "text": " ".join(near_tokens_variant),
            "title": "Κείμενο",
            "author": "Συγγραφέας",
            "greek_badness_score": 0.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "short",
            "text": " ".join(f"βραχύ{i}" for i in range(10)),
            "title": "Σύντομο",
            "author": "Β",
            "greek_badness_score": 0.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    for row in rows:
        row["len_greek"] = len(str(row["text"]))
    write_test_snapshot(input_root / "nested" / "demo.parquet", rows)
    payload = text_dedup.run_dedup_pipeline(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        max_workers=2,
        shingle_mode="token",
        shingle_size=5,
    )
    assert payload["exact"]["relaxed"]["dropped_rows"] == 1
    assert Path(payload["near"]["candidates"]["candidate_pairs_db_path"]).exists()
    exact_survivors = read_exact_survivor_rows(run_root)
    exact_survivor_ids = {row["source_doc_id"] for row in exact_survivors}
    assert exact_survivor_ids == {"relaxed-1", "near-1", "near-2", "short"}
    signatures = pq.read_table(run_root / "stage_02_near" / "signatures.parquet").to_pylist()
    signature_ids = {row["source_doc_id"] for row in signatures}
    assert "relaxed-1" in signature_ids
    assert "relaxed-2" not in signature_ids
    assert "short" not in signature_ids
    decisions = pq.read_table(run_root / "final" / "dedup_decisions.parquet").to_pylist()
    decisions_by_id = {row["source_doc_id"]: row for row in decisions}
    near_decisions = {doc_id: decisions_by_id[doc_id]["decision"] for doc_id in ("near-1", "near-2")}
    assert sorted(near_decisions.values()) == ["drop", "keep"]
    dropped_near_id = next(doc_id for doc_id, decision in near_decisions.items() if decision == "drop")
    assert decisions_by_id[dropped_near_id]["decision_stage"] == "near_duplicate"
    assert decisions_by_id["short"]["decision"] == "keep"
    assert decisions_by_id["short"]["decision_stage"] == "kept_after_exact"
    export_payload = text_dedup.export_dedup_run(state_root=state_root, run_root=run_root)
    assert export_payload["final"]["decision_rows"] == len(decisions)
    builder_manifest = json.loads((run_root / "builder_metadata" / "manifest.json").read_text())
    assert builder_manifest["builder_default_threshold"] == text_dedup.DEFAULT_NEAR_THRESHOLD
    assert builder_manifest["files"]["family_membership"] == "dedup_family_membership.parquet"
    assert builder_manifest["representative_score_version"] == text_dedup.REPRESENTATIVE_SCORE_VERSION
    builder_doc_rows = pq.read_table(run_root / "builder_metadata" / "doc_dedup_metadata.parquet").to_pylist()
    builder_doc_rows_by_id = {row["source_doc_id"]: row for row in builder_doc_rows}
    assert builder_doc_rows_by_id["near-1"]["near_candidate_count"] == 1
    assert builder_doc_rows_by_id["near-2"]["near_best_match_source_dataset"] == "demo"
    assert builder_doc_rows_by_id["near-1"]["len_greek"] == len(rows[2]["text"])
    assert builder_doc_rows_by_id["near-1"]["representative_score_version"] == text_dedup.REPRESENTATIVE_SCORE_VERSION
    family_rows = pq.read_table(run_root / "builder_metadata" / "dedup_family_membership.parquet").to_pylist()
    family_rows_by_id = {row["source_doc_id"]: row for row in family_rows}
    assert len(family_rows) == len(decisions)
    assert family_rows_by_id["near-1"]["representative_score_version"] == text_dedup.REPRESENTATIVE_SCORE_VERSION
    assert family_rows_by_id[dropped_near_id]["canonical_kept_doc_key"] == decisions_by_id[dropped_near_id]["kept_doc_key"]

    shutil.rmtree(input_root)
    shutil.rmtree(run_root / "stage_02_near")
    conn = text_dedup.connect_db(state_root)
    try:
        with conn:
            conn.execute(
                "DELETE FROM run_stage_chunks WHERE run_id = ? AND stage = ?",
                (run_root.name, text_dedup.NEAR_SIGNATURE_STAGE),
            )
            conn.execute(
                "DELETE FROM stage_progress WHERE run_id = ? AND stage = ?",
                (run_root.name, text_dedup.NEAR_SIGNATURE_STAGE),
            )
        rerun_summary = text_dedup._run_near_signature_stage(
            conn,
            run_id=run_root.name,
            run_root=run_root,
            survivor_manifest_path=run_root / "stage_01_exact" / "exact_survivor_manifest.parquet",
            state_root=state_root,
            config=json.loads((run_root / "run_config.json").read_text())["config"],
            greek_diacritic_policy=text_dedup.DEFAULT_GREEK_DIACRITIC_POLICY,
            max_workers=2,
            num_perm=text_dedup.DEFAULT_NUM_PERM,
            bands=text_dedup.DEFAULT_BANDS,
            rows_per_band=text_dedup.DEFAULT_ROWS_PER_BAND,
            shingle_mode="token",
            shingle_size=5,
        )
    finally:
        conn.close()
    rerun_signatures = pq.read_table(run_root / "stage_02_near" / "signatures.parquet").to_pylist()
    rerun_signature_ids = {row["source_doc_id"] for row in rerun_signatures}
    assert rerun_summary["exact_survivor_rows"] == 4
    assert rerun_signature_ids == signature_ids


def test_near_signature_stage_emits_shared_signature_matrix_artifacts(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    rows = [
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-1",
            "text": " ".join(f"λέξη{i}" for i in range(60)),
            "title": "Α",
            "author": "X",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-2",
            "text": " ".join(f"λέξη{i}" for i in range(59)) + " αλλαγή",
            "title": "Β",
            "author": "X",
            "greek_badness_score": 0.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_test_snapshot(input_root / "demo.parquet", rows)
    payload = text_dedup.run_dedup_pipeline(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        max_workers=1,
        shingle_mode="token",
        shingle_size=5,
    )

    signatures_path = run_root / "stage_02_near" / "signatures.parquet"
    matrix_path = text_dedup.signature_matrix_path(signatures_path)
    doc_keys_path = text_dedup.signature_doc_keys_path(signatures_path)
    assert Path(payload["near"]["signatures"]["signature_matrix_path"]) == matrix_path
    assert Path(payload["near"]["signatures"]["signature_doc_keys_path"]) == doc_keys_path
    assert Path(payload["near"]["signatures"]["signature_metadata_path"]) == text_dedup.signature_metadata_path(signatures_path)
    assert matrix_path.exists()
    assert doc_keys_path.exists()
    assert text_dedup.signature_metadata_path(signatures_path).exists()
    signature_lookup, signature_meta = text_dedup.load_signature_index(signatures_path)
    assert len(signature_lookup) == 2
    assert len(signature_meta) == 2
    assert signature_lookup.matrix.shape == (2, text_dedup.DEFAULT_NUM_PERM)
    assert set(signature_meta.row_by_doc_key) == {row["doc_key"] for row in pq.read_table(signatures_path).to_pylist()}


def test_full_dedup_pipeline_reuses_unchanged_stage2_state_across_runs(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root_1 = tmp_path / "run-1"
    run_root_2 = tmp_path / "run-2"
    input_root.mkdir()
    base_tokens = [f"λέξη{i}" for i in range(100)]
    variant_tokens = list(base_tokens)
    variant_tokens[30] = "παραλλαγή"
    rows = [
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-a",
            "text": " ".join(base_tokens),
            "title": "Α",
            "author": "X",
            "greek_badness_score": 0.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-b",
            "text": " ".join(variant_tokens),
            "title": "Β",
            "author": "X",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "doc-c",
            "text": " ".join(f"άλλο{i}" for i in range(80)),
            "title": "Γ",
            "author": "Y",
            "greek_badness_score": 0.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_test_snapshot(input_root / "demo.parquet", rows)
    first = text_dedup.run_dedup_pipeline(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root_1,
        max_workers=1,
        shingle_mode="token",
        shingle_size=5,
    )
    second = text_dedup.run_dedup_pipeline(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root_2,
        max_workers=1,
        shingle_mode="token",
        shingle_size=5,
    )
    assert second["near"]["signatures"]["reused_signature_chunks"] > 0
    assert second["near"]["candidates"]["reused_bucket_count"] > 0
    assert second["near"]["candidates"]["touched_doc_count"] == 0
    assert second["near"]["clusters"]["reused_cluster_count"] > 0
    assert second["final"]["decision_rows"] == first["final"]["decision_rows"]


def test_final_decisions_resolve_transitive_kept_doc_key_lineage(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    base_tokens = [f"λέξη{i}" for i in range(80)]
    near_tokens = list(base_tokens)
    near_tokens[20] = "παραλλαγή"
    rows = [
        {
            "source_dataset": "demo",
            "source_doc_id": "exact-a",
            "text": " ".join(base_tokens),
            "title": "Α",
            "author": "X",
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "exact-b",
            "text": " ".join(base_tokens),
            "title": "Β",
            "author": "X",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "near-c",
            "text": " ".join(near_tokens),
            "title": "Γ",
            "author": "X",
            "greek_badness_score": 0.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_test_snapshot(input_root / "demo.parquet", rows)
    text_dedup.run_dedup_pipeline(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        max_workers=1,
        minhash_threshold=0.80,
        num_perm=1024,
        bands=128,
        rows_per_band=8,
        shingle_mode="token",
        shingle_size=2,
    )

    decisions = pq.read_table(run_root / "final" / "dedup_decisions.parquet").to_pylist()
    by_id = {row["source_doc_id"]: row for row in decisions}
    final_kept_doc_key = str(by_id["near-c"]["doc_key"])
    assert by_id["exact-a"]["decision_stage"] == text_dedup.STRICT_STAGE
    assert by_id["exact-b"]["decision_stage"] == "near_duplicate"
    assert by_id["exact-a"]["kept_doc_key"] == final_kept_doc_key
    assert by_id["exact-b"]["kept_doc_key"] == final_kept_doc_key
    assert by_id["near-c"]["kept_doc_key"] == final_kept_doc_key


def test_near_cluster_stage_can_resume_from_partial_component_chunks(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    rows = []
    for idx in range(6):
        rows.append(
            {
                "source_dataset": "demo",
                "source_doc_id": f"doc-{idx}",
                "text": " ".join(f"διαφορετικη{idx}_{tok}" for tok in range(30)),
                "title": f"Τίτλος {idx}",
                "author": f"Συγγραφέας {idx}",
                "greek_badness_score": float(idx),
                "mojibake_badness_score": 0.0,
                "needs_ocr": False,
                "is_empty": False,
                "ocr_success": True,
                "is_historical_or_polytonic": False,
            }
        )
    write_test_snapshot(input_root / "demo.parquet", rows)

    original_build_cluster_chunks = text_dedup.build_near_cluster_chunk_specs
    monkeypatch.setattr(
        text_dedup,
        "build_near_cluster_chunk_specs",
        lambda components: original_build_cluster_chunks(components, target_doc_count=2, max_components=1),
    )

    payload = text_dedup.run_dedup_pipeline(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        max_workers=2,
        minhash_threshold=0.99,
        shingle_mode="token",
        shingle_size=5,
    )
    assert payload["near"]["clusters"]["cluster_rows"] == 6
    conn = text_dedup.connect_db(state_root)
    try:
        cluster_chunks = list(
            conn.execute(
                """
                SELECT chunk_key, artifact_path
                FROM run_stage_chunks
                WHERE run_id = ? AND stage = ?
                ORDER BY chunk_key
                """,
                (run_root.name, text_dedup.NEAR_CLUSTER_STAGE),
            )
        )
        assert len(cluster_chunks) > 1
        broken_chunk = cluster_chunks[0]
        broken_artifact_path = Path(str(broken_chunk["artifact_path"]))
        broken_summary_path = (
            run_root
            / "stage_02_near"
            / "shards"
            / "cluster_summaries"
            / broken_artifact_path.name
        )
        broken_artifact_path.unlink()
        broken_summary_path.unlink()
        (run_root / "stage_02_near" / "near_clusters.parquet").unlink()
        (run_root / "stage_02_near" / "cluster_summary.parquet").unlink()
        (run_root / "stage_02_near" / "near_drop_list.parquet").unlink()
        with conn:
            conn.execute(
                """
                UPDATE run_stage_chunks
                SET status = 'pending', processed_at = NULL, artifact_path = NULL, row_count = NULL
                WHERE run_id = ? AND stage = ? AND chunk_key = ?
                """,
                (run_root.name, text_dedup.NEAR_CLUSTER_STAGE, str(broken_chunk["chunk_key"])),
            )
            conn.execute(
                "DELETE FROM stage_progress WHERE run_id = ? AND stage = ?",
                (run_root.name, text_dedup.NEAR_CLUSTER_STAGE),
            )
        rerun_summary = text_dedup._run_near_cluster_stage(
            conn,
            run_id=run_root.name,
            run_root=run_root,
            state_root=state_root,
            config=json.loads((run_root / "run_config.json").read_text())["config"],
            minhash_threshold=0.99,
            large_component_threshold=text_dedup.DEFAULT_LARGE_COMPONENT_THRESHOLD,
        )
    finally:
        conn.close()
    assert rerun_summary["cluster_rows"] == 6
    assert rerun_summary["cluster_count"] == 6
    assert rerun_summary["candidate_pair_rows"] == 0
    stage_progress = json.loads(
        (run_root / "progress" / f"{text_dedup.NEAR_CLUSTER_STAGE}.json").read_text()
    )
    assert stage_progress["status"] == "completed"
    assert stage_progress["total_chunks"] > 1
    near_clusters = pq.read_table(run_root / "stage_02_near" / "near_clusters.parquet").to_pylist()
    assert len(near_clusters) == 6
    assert all(row["accepted_reason"] == "singleton" for row in near_clusters)


def test_near_candidate_worker_cap_env_override(monkeypatch) -> None:
    monkeypatch.setenv("GLOSSAPI_NEAR_CANDIDATE_MAX_WORKERS", "16")
    assert text_dedup.near_candidate_worker_cap() == 16
    assert text_dedup.effective_worker_count(
        min(32, text_dedup.near_candidate_worker_cap()),
        32,
    ) == 16


def test_resolve_near_component_splits_weak_member_after_representative_validation() -> None:
    component = {"doc-a", "doc-b", "doc-c", "doc-d"}
    adjacency = {
        "doc-a": {"doc-b", "doc-c"},
        "doc-b": {"doc-a", "doc-c"},
        "doc-c": {"doc-a", "doc-b", "doc-d"},
        "doc-d": {"doc-c"},
    }
    signature_map = {
        "doc-a": np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint64),
        "doc-b": np.array([1, 2, 3, 4, 5, 6, 7, 80], dtype=np.uint64),
        "doc-c": np.array([1, 2, 3, 4, 5, 6, 70, 80], dtype=np.uint64),
        "doc-d": np.array([1, 2, 3, 4, 50, 60, 70, 80], dtype=np.uint64),
    }
    signature_meta = build_signature_metadata_lookup(
        {
        doc_key: {
            "token_count": 80,
            "char_count": 400,
            "near_text_chars": 400,
            "shingle_mode": "token",
        }
        for doc_key in component
        }
    )
    doc_meta = {
        "doc-a": {
            "source_dataset": "demo",
            "source_doc_id": "doc-a",
            "needs_ocr": 0,
            "greek_badness_score": 0.0,
            "mojibake_badness_score": 0.0,
            "ocr_success": 1,
            "title": "A",
            "author": "Author",
        },
        "doc-b": {
            "source_dataset": "demo",
            "source_doc_id": "doc-b",
            "needs_ocr": 0,
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "ocr_success": 1,
            "title": "B",
            "author": "Author",
        },
        "doc-c": {
            "source_dataset": "demo",
            "source_doc_id": "doc-c",
            "needs_ocr": 0,
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "ocr_success": 1,
            "title": "C",
            "author": "Author",
        },
        "doc-d": {
            "source_dataset": "demo",
            "source_doc_id": "doc-d",
            "needs_ocr": 0,
            "greek_badness_score": 3.0,
            "mojibake_badness_score": 0.0,
            "ocr_success": 1,
            "title": "D",
            "author": "Author",
        },
    }

    cluster_rows, summary_rows = text_dedup._resolve_near_component(
        component,
        adjacency=adjacency,
        signature_map=signature_map,
        signature_meta=signature_meta,
        doc_meta=doc_meta,
        minhash_threshold=0.75,
        large_component_threshold=10,
    )

    by_doc = {row["member_doc_key"]: row for row in cluster_rows}
    assert {row["member_doc_key"] for row in cluster_rows} == component
    assert by_doc["doc-a"]["kept_doc_key"] == "doc-a"
    assert by_doc["doc-b"]["kept_doc_key"] == "doc-a"
    assert by_doc["doc-c"]["kept_doc_key"] == "doc-a"
    assert by_doc["doc-d"]["kept_doc_key"] == "doc-d"
    assert by_doc["doc-a"]["estimated_jaccard"] == 1.0
    assert by_doc["doc-b"]["estimated_jaccard"] == 0.875
    assert by_doc["doc-c"]["estimated_jaccard"] == 0.75
    assert by_doc["doc-d"]["estimated_jaccard"] == 1.0
    assert by_doc["doc-a"]["accepted_reason"] == "representative"
    assert by_doc["doc-b"]["accepted_reason"] == "representative_validation"
    assert by_doc["doc-c"]["accepted_reason"] == "representative_validation"
    assert by_doc["doc-d"]["accepted_reason"] == "singleton"
    assert by_doc["doc-a"]["cluster_size"] == 3
    assert by_doc["doc-b"]["cluster_size"] == 3
    assert by_doc["doc-c"]["cluster_size"] == 3
    assert by_doc["doc-d"]["cluster_size"] == 1
    assert by_doc["doc-a"]["component_size"] == 4
    assert by_doc["doc-d"]["component_size"] == 1
    assert by_doc["doc-b"]["dropped"] is True
    assert by_doc["doc-c"]["dropped"] is True
    assert by_doc["doc-d"]["dropped"] is False
    assert sorted(row["cluster_size"] for row in summary_rows) == [1, 3]
    assert sorted(row["dropped_count"] for row in summary_rows) == [0, 2]


def test_full_dedup_pipeline_splits_weak_member_from_multi_doc_near_component(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    input_root.mkdir()
    base_tokens = [f"tok{i}" for i in range(80)]
    near_b_tokens = list(base_tokens)
    near_b_tokens[5] = "alt-b"
    near_c_tokens = list(base_tokens)
    near_c_tokens[10] = "alt-c10"
    near_c_tokens[20] = "alt-c20"
    near_c_tokens[30] = "alt-c30"
    near_d_tokens = list(base_tokens)
    near_d_tokens[10] = "alt-c10"
    near_d_tokens[20] = "alt-c20"
    near_d_tokens[30] = "alt-c30"
    near_d_tokens[40] = "alt-d40"
    near_d_tokens[50] = "alt-d50"
    near_d_tokens[60] = "alt-d60"
    relaxed_tokens = [f"dup{i}" for i in range(30)]
    rows = [
        {
            "source_dataset": "demo",
            "source_doc_id": "relaxed-1",
            "text": "ΑΥΤΌ είναι - " + " ".join(relaxed_tokens),
            "title": "Τίτλος",
            "author": "Α",
            "greek_badness_score": 0.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "relaxed-2",
            "text": "αυτό είναι " + " ".join(relaxed_tokens),
            "title": "Τίτλος",
            "author": "Α",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "near-a",
            "text": " ".join(base_tokens),
            "title": "Κείμενο",
            "author": "Συγγραφέας",
            "greek_badness_score": 0.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "near-b",
            "text": " ".join(near_b_tokens),
            "title": "Κείμενο",
            "author": "Συγγραφέας",
            "greek_badness_score": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "near-c",
            "text": " ".join(near_c_tokens),
            "title": "Κείμενο",
            "author": "Συγγραφέας",
            "greek_badness_score": 2.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "near-d",
            "text": " ".join(near_d_tokens),
            "title": "Κείμενο",
            "author": "Συγγραφέας",
            "greek_badness_score": 3.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
        {
            "source_dataset": "demo",
            "source_doc_id": "short",
            "text": " ".join(f"βραχύ{i}" for i in range(10)),
            "title": "Σύντομο",
            "author": "Β",
            "greek_badness_score": 0.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "ocr_success": True,
            "is_historical_or_polytonic": False,
        },
    ]
    write_test_snapshot(input_root / "nested" / "demo.parquet", rows)

    payload = text_dedup.run_dedup_pipeline(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        max_workers=2,
        minhash_threshold=0.80,
        num_perm=1024,
        bands=128,
        rows_per_band=8,
        shingle_mode="token",
        shingle_size=2,
    )

    assert payload["exact"]["relaxed"]["dropped_rows"] == 1
    candidate_pairs = pq.read_table(run_root / "stage_02_near" / "candidate_pairs.parquet").to_pylist()
    candidate_doc_pairs = {
        tuple(sorted((row["doc_key_left"], row["doc_key_right"])))
        for row in candidate_pairs
    }
    signatures = pq.read_table(run_root / "stage_02_near" / "signatures.parquet").to_pylist()
    signature_ids = {row["source_doc_id"] for row in signatures}
    assert signature_ids == {"near-a", "near-b", "near-c", "near-d", "relaxed-1"}
    assert any(row["estimated_jaccard"] >= 0.80 for row in candidate_pairs)

    near_clusters = pq.read_table(run_root / "stage_02_near" / "near_clusters.parquet").to_pylist()
    near_clusters_by_id = {row["member_source_doc_id"]: row for row in near_clusters}
    assert near_clusters_by_id["near-a"]["kept_doc_key"] == near_clusters_by_id["near-a"]["member_doc_key"]
    assert near_clusters_by_id["near-a"]["cluster_size"] == 3
    assert near_clusters_by_id["near-b"]["cluster_size"] == 3
    assert near_clusters_by_id["near-c"]["cluster_size"] == 3
    assert near_clusters_by_id["near-d"]["cluster_size"] == 1
    assert near_clusters_by_id["near-b"]["dropped"] is True
    assert near_clusters_by_id["near-c"]["dropped"] is True
    assert near_clusters_by_id["near-d"]["dropped"] is False
    assert near_clusters_by_id["near-b"]["accepted_reason"] == "representative_validation"
    assert near_clusters_by_id["near-c"]["accepted_reason"] == "representative_validation"
    assert near_clusters_by_id["near-d"]["accepted_reason"] == "singleton"
    assert near_clusters_by_id["near-a"]["component_size"] == 4

    decisions = pq.read_table(run_root / "final" / "dedup_decisions.parquet").to_pylist()
    decisions_by_id = {row["source_doc_id"]: row for row in decisions}
    assert decisions_by_id["near-a"]["decision"] == "keep"
    assert decisions_by_id["near-a"]["decision_stage"] == "kept_after_near"
    assert decisions_by_id["near-b"]["decision"] == "drop"
    assert decisions_by_id["near-c"]["decision"] == "drop"
    assert decisions_by_id["near-b"]["decision_stage"] == "near_duplicate"
    assert decisions_by_id["near-c"]["decision_stage"] == "near_duplicate"
    assert decisions_by_id["near-d"]["decision"] == "keep"
    assert decisions_by_id["near-d"]["decision_stage"] == "kept_after_near"
    assert decisions_by_id["relaxed-2"]["decision_stage"] == text_dedup.RELAXED_STAGE
    assert decisions_by_id["short"]["decision_stage"] == "kept_after_exact"
    assert candidate_doc_pairs
