from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TOKENIZER_REPO_ROOT = Path(__file__).resolve().parents[1]
GLOSSAPI_WORK_ROOT = Path("/home/foivos/data/glossapi_work")
HF_RELEASE_ROOT = GLOSSAPI_WORK_ROOT / "hf_release_publish"

sys.path.insert(0, str(TOKENIZER_REPO_ROOT / "subprojects" / "01_hplt_filtering" / "scripts"))
sys.path.insert(0, str(TOKENIZER_REPO_ROOT / "subprojects" / "01_1_corpus_dedup" / "scripts"))
sys.path.insert(0, str(TOKENIZER_REPO_ROOT / "ops" / "upload"))
sys.path.insert(0, str(TOKENIZER_REPO_ROOT))

from glossapi_corpus_cli import pipeline  # noqa: E402
import build_hplt_hf_slice  # noqa: E402
import integrate_hplt_slice_into_working_release  # noqa: E402
import launch_hf_uploader_handoff  # noqa: E402
import prepare_hf_uploader_handoff  # noqa: E402
import publish_dedup_overlay_into_working_release  # noqa: E402


def _write_canonical_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame = pipeline.finalize_frame(frame)
    table = pa.Table.from_pandas(
        frame[pipeline.CANONICAL_COLUMNS],
        schema=pipeline.CANONICAL_ARROW_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, path, compression="zstd")


def _base_row(
    *,
    source_dataset: str,
    source_doc_id: str,
    text: str,
    needs_ocr: bool = False,
) -> dict[str, object]:
    return {
        "source_dataset": source_dataset,
        "source_doc_id": source_doc_id,
        "text": text,
        "title": None,
        "author": None,
        "source_metadata_json": None,
        "is_historical_or_polytonic": False,
        "contains_math": False,
        "contains_latex": False,
        "greek_percentage": None,
        "latin_percentage": None,
        "polytonic_ratio": None,
        "table_ratio": None,
        "greek_badness_score": None,
        "mojibake_badness_score": None,
        "needs_ocr": needs_ocr,
        "is_empty": False,
        "filter": "keep",
        "ocr_success": None,
        "quality_method": None,
        "reevaluated_at": None,
    }


def _write_hplt_zst(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_jsonl = path.with_suffix("")
    raw_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    subprocess.run(
        ["zstd", "-q", "-f", str(raw_jsonl), "-o", str(path)],
        check=True,
    )
    raw_jsonl.unlink()


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_hplt_slice_contract_smoke(tmp_path: Path) -> None:
    shard_dir = tmp_path / "raw"
    shard_path = shard_dir / "10_1.jsonl.zst"
    _write_hplt_zst(
        shard_path,
        [
            {
                "id": "keep-1",
                "text": "Αυτό είναι ένα πραγματικό μικρό ελληνικό κείμενο για δοκιμή.",
                "u": "https://example.test/keep-1",
                "c": "text/html",
                "crawl_id": "crawl-a",
                "lang": "el",
                "prob": 0.99,
                "cluster_size": 1,
                "filter": "keep",
                "web-register": {"IN": 0.91, "en": 0.77},
            },
            {
                "id": "mt-1",
                "text": "This is translated junk that should not survive.",
                "u": "https://example.test/mt-1",
                "c": "text/html",
                "crawl_id": "crawl-a",
                "lang": "en",
                "prob": 0.55,
                "cluster_size": 1,
                "filter": "keep",
                "web-register": {"MT": 0.99},
            },
            {
                "id": "empty-1",
                "text": "   ",
                "u": "https://example.test/empty-1",
                "c": "text/html",
                "crawl_id": "crawl-a",
                "lang": "el",
                "prob": 0.99,
                "cluster_size": 1,
                "filter": "keep",
                "web-register": {"NA": 0.55},
            },
        ],
    )

    release_root = tmp_path / "release"
    data_root = release_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    result = build_hplt_hf_slice.process_shard(
        "10_1.jsonl.zst",
        base_url=shard_dir.as_uri(),
        dataset_name="HPLT/ell_Grek_ge8_no_mt_clean60",
        quality_min=8,
        exclude_main_registers={"MT"},
        require_filter=None,
        batch_size=8,
        rows_per_part=100,
        data_root=data_root,
        target_schema_path=tmp_path / "missing-schema.parquet",
        quality_mode="score_only",
        greek_badness_max=1000.0,
        clean_num_threads=1,
        max_docs=None,
        max_chars=None,
        log_every_rows=0,
    )

    assert result.rows_seen == 3
    assert result.rows_skipped_mt == 1
    assert result.rows_skipped_empty == 1
    assert result.rows_written == 1
    assert len(result.part_files) == 1

    frame = pd.read_parquet(result.part_files[0])
    assert frame.columns.tolist() == pipeline.CANONICAL_COLUMNS
    assert frame["source_dataset"].tolist() == ["HPLT/ell_Grek_ge8_no_mt_clean60"]
    assert frame["source_doc_id"].tolist() == ["hplt::10_1.jsonl.zst::keep-1"]
    metadata = json.loads(frame.loc[0, "source_metadata_json"])
    assert metadata["quality_bin"] == 10
    assert metadata["url"] == "https://example.test/keep-1"


def test_integrate_hplt_slice_refreshes_working_release(tmp_path: Path, monkeypatch) -> None:
    working_root = tmp_path / "working_release"
    hplt_root = tmp_path / "hplt_release"

    _write_canonical_rows(
        working_root / "data" / "old_hplt.parquet",
        [_base_row(source_dataset="HPLT/ell_Grek_ge8_no_mt", source_doc_id="old-1", text="παλιό hplt")],
    )
    _write_canonical_rows(
        working_root / "data" / "other.parquet",
        [_base_row(source_dataset="alpha", source_doc_id="alpha-1", text="άλλο κείμενο")],
    )
    _write_canonical_rows(
        hplt_root / "data" / "new_hplt.parquet",
        [_base_row(source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60", source_doc_id="new-1", text="νέο hplt")],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "integrate_hplt_slice_into_working_release.py",
            "--working-release-root",
            str(working_root),
            "--hplt-release-root",
            str(hplt_root),
        ],
    )
    integrate_hplt_slice_into_working_release.main()

    remaining = sorted((working_root / "data").glob("*.parquet"))
    datasets = sorted({pq.read_table(path, columns=["source_dataset"]).column(0)[0].as_py() for path in remaining})
    assert datasets == ["HPLT/ell_Grek_ge8_no_mt_clean60", "alpha"]
    assert not any(path.name == "old_hplt.parquet" for path in remaining)
    assert (working_root / "row_counts.csv").exists()
    summary = json.loads((working_root / "hplt_integration_summary.json").read_text(encoding="utf-8"))
    assert summary["new_dataset_name"] == "HPLT/ell_Grek_ge8_no_mt_clean60"
    assert summary["removed_file_count"] == 1
    assert summary["copied_file_count"] == 1
    assert summary["validation_summary_csv"] is None
    assert summary["prepare_manifest_json"] is None


def test_publish_dedup_overlay_contract(tmp_path: Path, monkeypatch) -> None:
    working_root = tmp_path / "working_release"
    state_root = tmp_path / "state"
    run_root = state_root / "runs" / "run-001"
    builder_metadata_root = run_root / "builder_metadata"
    final_root = run_root / "final"
    code_root = tmp_path / "code_root"

    builder_metadata_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)
    (builder_metadata_root / "manifest.json").write_text('{"schema_version":"builder_metadata_v2"}\n', encoding="utf-8")
    (final_root / "run_summary.json").write_text('{"ok":true}\n', encoding="utf-8")
    (state_root / "latest_success.json").parent.mkdir(parents=True, exist_ok=True)
    (state_root / "latest_success.json").write_text(
        json.dumps({"run_id": "run-001", "run_root": str(run_root)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    (code_root / "glossapi_corpus_cli").mkdir(parents=True, exist_ok=True)
    (code_root / "tests").mkdir(parents=True, exist_ok=True)
    for relative_path in [
        "glossapi_corpus_cli/cli.py",
        "glossapi_corpus_cli/pipeline.py",
        "glossapi_corpus_cli/text_dedup.py",
        "tests/test_pipeline.py",
        "tests/test_text_dedup.py",
    ]:
        target = code_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# smoke\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_dedup_overlay_into_working_release.py",
            "--working-release-root",
            str(working_root),
            "--state-root",
            str(state_root),
            "--code-root",
            str(code_root),
            "--published-at",
            "2026-04-13",
        ],
    )
    publish_dedup_overlay_into_working_release.main()

    latest = json.loads((working_root / "dedup_metadata" / "latest.json").read_text(encoding="utf-8"))
    assert latest["latest_run_id"] == "run-001"
    assert latest["builder_metadata_root"] == "dedup_metadata/run-001/builder_metadata"
    assert (working_root / "dedup_metadata" / "run-001" / "builder_metadata" / "manifest.json").exists()
    assert (working_root / "dedup_metadata" / "run-001" / "publish_summary.json").exists()


def test_prepare_uploader_handoff_contract(tmp_path: Path, monkeypatch) -> None:
    working_root = tmp_path / "working_release"
    handoff_root = tmp_path / "handoff"

    _write_canonical_rows(
        working_root / "data" / "alpha.parquet",
        [_base_row(source_dataset="alpha", source_doc_id="alpha-1", text="άλλο κείμενο")],
    )
    _write_canonical_rows(
        working_root / "data" / "hplt.parquet",
        [
            _base_row(
                source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60",
                source_doc_id="hplt-1",
                text="ελληνικό κείμενο hplt",
            )
        ],
    )
    (working_root / "dedup_metadata").mkdir(parents=True, exist_ok=True)
    (working_root / "dedup_metadata" / "run-001" / "builder_metadata").mkdir(parents=True, exist_ok=True)
    (working_root / "dedup_metadata" / "latest.json").write_text(
        json.dumps(
            {
                "latest_run_id": "run-001",
                "builder_metadata_root": "dedup_metadata/run-001/builder_metadata",
                "path": "dedup_metadata/run-001",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (working_root / "row_counts.csv").write_text("source_dataset,row_count\nalpha,1\n", encoding="utf-8")
    (working_root / "hplt_integration_summary.json").write_text(
        json.dumps({"new_dataset_name": "HPLT/ell_Grek_ge8_no_mt_clean60"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_hf_uploader_handoff.py",
            "--working-release-root",
            str(working_root),
            "--handoff-root",
            str(handoff_root),
            "--repo-id",
            "fffoivos/glossapi-greek-nanochat-pretraining-dataset",
            "--public",
        ],
    )
    prepare_hf_uploader_handoff.main()

    manifest = json.loads((handoff_root / "uploader_handoff.json").read_text(encoding="utf-8"))
    assert manifest["repo_id"] == "fffoivos/glossapi-greek-nanochat-pretraining-dataset"
    assert manifest["visibility"] == "public"
    assert manifest["contracts"]["hplt_dataset_name"] == "HPLT/ell_Grek_ge8_no_mt_clean60"
    assert manifest["contracts"]["hplt_file_count"] == 1
    assert manifest["contracts"]["dedup_latest_run_id"] == "run-001"
    assert "publish_hf_release.py" in manifest["upload"]["remote_publish_command"]
    assert "data" in manifest["sync_paths"]
    assert "dedup_metadata" in manifest["sync_paths"]


def test_prepare_uploader_handoff_source_only_contract(tmp_path: Path, monkeypatch) -> None:
    working_root = tmp_path / "working_release"
    handoff_root = tmp_path / "handoff"

    _write_canonical_rows(
        working_root / "data" / "hplt.parquet",
        [
            _base_row(
                source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60",
                source_doc_id="hplt-1",
                text="ελληνικό κείμενο hplt",
            )
        ],
    )
    (working_root / "README.md").write_text("# dataset\n", encoding="utf-8")
    (working_root / "hplt_integration_summary.json").write_text(
        json.dumps({"new_dataset_name": "HPLT/ell_Grek_ge8_no_mt_clean60"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_hf_uploader_handoff.py",
            "--working-release-root",
            str(working_root),
            "--handoff-root",
            str(handoff_root),
            "--repo-id",
            "fffoivos/glossapi-greek-nanochat-pretraining-dataset",
            "--public",
            "--source-only",
        ],
    )
    prepare_hf_uploader_handoff.main()

    manifest = json.loads((handoff_root / "uploader_handoff.json").read_text(encoding="utf-8"))
    assert manifest["scope"] == "source_only"
    assert manifest["contracts"]["hplt_dataset_name"] == "HPLT/ell_Grek_ge8_no_mt_clean60"
    assert manifest["contracts"]["dedup_latest_run_id"] is None
    assert "data" in manifest["sync_paths"]
    assert "README.md" in manifest["sync_paths"]
    assert "dedup_metadata" not in manifest["sync_paths"]


def test_wait_for_prepare_uploader_handoff_shell(tmp_path: Path) -> None:
    working_root = tmp_path / "working_release"
    state_root = tmp_path / "state"
    handoff_root = tmp_path / "handoff"

    _write_canonical_rows(
        working_root / "data" / "hplt.parquet",
        [
            _base_row(
                source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60",
                source_doc_id="hplt-1",
                text="ελληνικό κείμενο hplt",
            )
        ],
    )
    (working_root / "dedup_metadata").mkdir(parents=True, exist_ok=True)
    (working_root / "dedup_metadata" / "run-002" / "builder_metadata").mkdir(parents=True, exist_ok=True)
    (working_root / "dedup_metadata" / "latest.json").write_text(
        json.dumps(
            {
                "latest_run_id": "run-002",
                "builder_metadata_root": "dedup_metadata/run-002/builder_metadata",
                "path": "dedup_metadata/run-002",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (state_root / "latest_success.json").parent.mkdir(parents=True, exist_ok=True)
    (state_root / "latest_success.json").write_text(
        json.dumps({"run_id": "run-002", "run_root": str(state_root / "runs" / "run-002")}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (working_root / "hplt_integration_summary.json").write_text(
        json.dumps({"new_dataset_name": "HPLT/ell_Grek_ge8_no_mt_clean60"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    script = TOKENIZER_REPO_ROOT / "ops" / "upload" / "wait_for_dedup_overlay_and_prepare_handoff.sh"
    subprocess.run(["bash", str(script), str(working_root), str(state_root), str(handoff_root)], check=True)
    assert (handoff_root / "uploader_handoff.json").exists()
    assert (handoff_root / "handoff_summary.json").exists()


def test_wait_for_prepare_source_only_uploader_handoff_shell(tmp_path: Path) -> None:
    working_root = tmp_path / "working_release"
    handoff_root = tmp_path / "handoff"

    _write_canonical_rows(
        working_root / "data" / "hplt.parquet",
        [
            _base_row(
                source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60",
                source_doc_id="hplt-1",
                text="ελληνικό κείμενο hplt",
            )
        ],
    )
    (working_root / "README.md").write_text("# dataset\n", encoding="utf-8")
    (working_root / "hplt_integration_summary.json").write_text(
        json.dumps({"new_dataset_name": "HPLT/ell_Grek_ge8_no_mt_clean60"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    script = TOKENIZER_REPO_ROOT / "ops" / "upload" / "wait_for_hplt_and_prepare_source_only_handoff.sh"
    env = os.environ.copy()
    env["TOKENIZER_PIPELINE_PYTHON_BIN"] = str(GLOSSAPI_WORK_ROOT / ".venv" / "bin" / "python")
    subprocess.run(["bash", str(script), str(working_root), str(handoff_root)], check=True, env=env)

    manifest = json.loads((handoff_root / "uploader_handoff.json").read_text(encoding="utf-8"))
    assert manifest["scope"] == "source_only"
    assert manifest["contracts"]["dedup_latest_run_id"] is None
    assert "data" in manifest["sync_paths"]
    assert "dedup_metadata" not in manifest["sync_paths"]


def test_wait_for_build_tokenizer_mixes_shell_parallel_and_resumable(tmp_path: Path) -> None:
    working_root = tmp_path / "working_release"
    state_root = tmp_path / "state"
    mix_root = tmp_path / "mixes"
    builder_root = working_root / "dedup_metadata" / "run-004" / "builder_metadata"

    builder_root.mkdir(parents=True, exist_ok=True)
    (builder_root / "manifest.json").write_text(
        json.dumps(
            {
                "builder_metadata_version": "builder_metadata_v2",
                "files": {
                    "doc_metadata": "doc_dedup_metadata.parquet",
                    "family_membership": "dedup_family_membership.parquet",
                    "near_candidate_pairs": "near_candidate_pairs.parquet",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pq.write_table(pa.table({"doc_key": pa.array([], type=pa.string())}), builder_root / "doc_dedup_metadata.parquet")
    pq.write_table(pa.table({"doc_key": pa.array([], type=pa.string())}), builder_root / "dedup_family_membership.parquet")
    pq.write_table(pa.table({"left_doc_key": pa.array([], type=pa.string())}), builder_root / "near_candidate_pairs.parquet")
    (working_root / "dedup_metadata" / "latest.json").write_text(
        json.dumps(
            {
                "latest_run_id": "run-004",
                "builder_metadata_root": "dedup_metadata/run-004/builder_metadata",
                "path": "dedup_metadata/run-004",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (state_root / "latest_success.json").parent.mkdir(parents=True, exist_ok=True)
    (state_root / "latest_success.json").write_text(
        json.dumps({"run_id": "run-004", "run_root": str(state_root / "runs" / "run-004")}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (working_root / "hplt_integration_summary.json").write_text(
        json.dumps({"new_dataset_name": "HPLT/ell_Grek_ge8_no_mt_clean60"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fake_python = tmp_path / "fake_python.py"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import annotations",
                "import json",
                "import os",
                "import sys",
                "import time",
                "from pathlib import Path",
                f"REAL_PYTHON = {sys.executable!r}",
                "if len(sys.argv) > 1 and sys.argv[1] == '-':",
                "    os.execv(REAL_PYTHON, [REAL_PYTHON, *sys.argv[1:]])",
                "if len(sys.argv) > 3 and sys.argv[1:4] == ['-m', 'glossapi_corpus_cli.cli', 'mix-prepare-selected-input']:",
                "    selected_input_path = None",
                "    args = sys.argv[4:]",
                "    idx = 0",
                "    while idx < len(args):",
                "        arg = args[idx]",
                "        if arg == '--selected-input-path':",
                "            selected_input_path = Path(args[idx + 1])",
                "            idx += 2",
                "            continue",
                "        idx += 1",
                "    if selected_input_path is None:",
                "        raise SystemExit('missing selected input path')",
                "    selected_input_path.parent.mkdir(parents=True, exist_ok=True)",
                "    time.sleep(1.2)",
                "    selected_input_path.write_text('fake selected input\\n', encoding='utf-8')",
                "    (selected_input_path.parent / 'prelude_done.json').write_text(",
                "        json.dumps({'prepared_at': time.time()}, ensure_ascii=False, indent=2) + '\\n',",
                "        encoding='utf-8',",
                "    )",
                "    print(json.dumps({'selected_input': {'path': str(selected_input_path), 'rows': 1, 'chars': 1}}, ensure_ascii=False))",
                "    raise SystemExit(0)",
                "if len(sys.argv) > 3 and sys.argv[1:4] == ['-m', 'glossapi_corpus_cli.cli', 'mix-build-from-selected-input']:",
                "    selected_input_path = None",
                "    output_path = None",
                "    config_path = None",
                "    args = sys.argv[4:]",
                "    idx = 0",
                "    while idx < len(args):",
                "        arg = args[idx]",
                "        if arg == '--selected-input-path':",
                "            selected_input_path = Path(args[idx + 1])",
                "            idx += 2",
                "            continue",
                "        if arg == '--mix-output-path':",
                "            output_path = Path(args[idx + 1])",
                "            idx += 2",
                "            continue",
                "        if arg == '--source-mix-config-path':",
                "            config_path = Path(args[idx + 1])",
                "            idx += 2",
                "            continue",
                "        idx += 1",
                "    if selected_input_path is None or output_path is None or config_path is None:",
                "        raise SystemExit('missing mix args')",
                "    if not selected_input_path.exists():",
                "        raise SystemExit('selected input missing')",
                "    output_path.parent.mkdir(parents=True, exist_ok=True)",
                "    (output_path.parent / 'runner_start.json').write_text(",
                "        json.dumps({'started_at': time.time(), 'config_path': str(config_path), 'selected_input_path': str(selected_input_path)}, ensure_ascii=False, indent=2) + '\\n',",
                "        encoding='utf-8',",
                "    )",
                "    time.sleep(1.2)",
                "    output_path.write_text('fake parquet placeholder\\n', encoding='utf-8')",
                "    output_path.with_suffix('.summary.csv').write_text(",
                "        'source_dataset,rows,estimated_tokens\\nalpha,1,1\\n',",
                "        encoding='utf-8',",
                "    )",
                "    (output_path.parent / 'runner_end.json').write_text(",
                "        json.dumps({'finished_at': time.time()}, ensure_ascii=False, indent=2) + '\\n',",
                "        encoding='utf-8',",
                "    )",
                "    print(json.dumps({'rows_kept': 1}, ensure_ascii=False))",
                "    raise SystemExit(0)",
                "os.execv(REAL_PYTHON, [REAL_PYTHON, *sys.argv[1:]])",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    script = TOKENIZER_REPO_ROOT / "subprojects" / "01_2_training_dataset_mix" / "scripts" / "wait_for_dedup_overlay_and_build_tokenizer_mixes.sh"
    env = os.environ.copy()
    env["TOKENIZER_PIPELINE_PYTHON_BIN"] = str(fake_python)
    env["TOKENIZER_MIX_MAX_JOBS"] = "2"

    started = time.monotonic()
    subprocess.run(["bash", str(script), str(working_root), str(state_root), str(mix_root)], check=True, env=env)
    elapsed = time.monotonic() - started

    glossapi_status = json.loads((mix_root / "glossapi_only" / "build_status.json").read_text(encoding="utf-8"))
    mixed_status = json.loads((mix_root / "glossapi_plus_hplt_70_30" / "build_status.json").read_text(encoding="utf-8"))
    prelude_status = json.loads((mix_root / "_shared" / "prepare_status.json").read_text(encoding="utf-8"))
    prelude_summary = json.loads((mix_root / "_shared" / "selected_input_summary.json").read_text(encoding="utf-8"))
    glossapi_start = json.loads((mix_root / "glossapi_only" / "runner_start.json").read_text(encoding="utf-8"))
    mixed_start = json.loads((mix_root / "glossapi_plus_hplt_70_30" / "runner_start.json").read_text(encoding="utf-8"))
    prelude_done = json.loads((mix_root / "_shared" / "prelude_done.json").read_text(encoding="utf-8"))

    assert elapsed < 3.3
    assert prelude_status["state"] == "completed"
    assert prelude_summary["selected_input"]["path"].endswith("selected_input.parquet")
    assert glossapi_status["state"] == "completed"
    assert mixed_status["state"] == "completed"
    assert glossapi_start["started_at"] >= prelude_done["prepared_at"]
    assert mixed_start["started_at"] >= prelude_done["prepared_at"]
    assert abs(float(glossapi_start["started_at"]) - float(mixed_start["started_at"])) < 0.8
    assert (mix_root / "glossapi_only" / "mix.parquet").exists()
    assert (mix_root / "glossapi_plus_hplt_70_30" / "mix.parquet").exists()

    subprocess.run(["bash", str(script), str(working_root), str(state_root), str(mix_root)], check=True, env=env)
    glossapi_status_rerun = json.loads((mix_root / "glossapi_only" / "build_status.json").read_text(encoding="utf-8"))
    mixed_status_rerun = json.loads((mix_root / "glossapi_plus_hplt_70_30" / "build_status.json").read_text(encoding="utf-8"))
    prelude_status_rerun = json.loads((mix_root / "_shared" / "prepare_status.json").read_text(encoding="utf-8"))
    assert glossapi_status_rerun["note"] == "skipped_existing_output"
    assert mixed_status_rerun["note"] == "skipped_existing_output"
    assert prelude_status_rerun["note"] == "skipped_existing_selected_input"


def test_wait_for_tokenizer_mixes_launches_each_training_when_its_mix_is_ready(tmp_path: Path) -> None:
    mix_root = tmp_path / "mixes"
    training_root = tmp_path / "training"
    glossapi_mix = mix_root / "glossapi_only" / "mix.parquet"
    mixed_mix = mix_root / "glossapi_plus_hplt_70_30" / "mix.parquet"
    glossapi_build_status = mix_root / "glossapi_only" / "build_status.json"
    mixed_build_status = mix_root / "glossapi_plus_hplt_70_30" / "build_status.json"
    glossapi_mix.parent.mkdir(parents=True, exist_ok=True)
    mixed_mix.parent.mkdir(parents=True, exist_ok=True)
    glossapi_mix.write_text("ready\n", encoding="utf-8")
    glossapi_build_status.write_text(
        json.dumps(
            {
                "mix_name": "glossapi_only",
                "state": "completed",
                "mix_output_path": str(glossapi_mix),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    mixed_mix.write_text("", encoding="utf-8")
    mixed_build_status.write_text(
        json.dumps(
            {
                "mix_name": "glossapi_plus_hplt_70_30",
                "state": "running",
                "mix_output_path": str(mixed_mix),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fake_python = tmp_path / "fake_train_python.py"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import annotations",
                "import json",
                "import os",
                "import sys",
                "import time",
                "from pathlib import Path",
                f"REAL_PYTHON = {sys.executable!r}",
                "if len(sys.argv) > 1 and sys.argv[1].endswith('train_discovery_tokenizer.py'):",
                "    output_dir = None",
                "    run_name = None",
                "    input_glob = None",
                "    args = sys.argv[2:]",
                "    idx = 0",
                "    while idx < len(args):",
                "        arg = args[idx]",
                "        if arg == '--output-dir':",
                "            output_dir = Path(args[idx + 1])",
                "            idx += 2",
                "            continue",
                "        if arg == '--name':",
                "            run_name = args[idx + 1]",
                "            idx += 2",
                "            continue",
                "        if arg == '--input-glob':",
                "            input_glob = args[idx + 1]",
                "            idx += 2",
                "            continue",
                "        idx += 1",
                "    if output_dir is None or run_name is None or input_glob is None:",
                "        raise SystemExit('missing training args')",
                "    output_dir.mkdir(parents=True, exist_ok=True)",
                "    (output_dir / 'runner_start.json').write_text(",
                "        json.dumps({'started_at': time.time(), 'run_name': run_name, 'input_glob': input_glob}, ensure_ascii=False, indent=2) + '\\n',",
                "        encoding='utf-8',",
                "    )",
                "    time.sleep(0.4)",
                "    (output_dir / 'training_summary.json').write_text(",
                "        json.dumps({'ok': True, 'run_name': run_name}, ensure_ascii=False, indent=2) + '\\n',",
                "        encoding='utf-8',",
                "    )",
                "    raise SystemExit(0)",
                "os.execv(REAL_PYTHON, [REAL_PYTHON, *sys.argv[1:]])",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    delayed_mix_creator = tmp_path / "delayed_mixed_mix.sh"
    delayed_mix_creator.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "sleep 1",
                f"printf 'ready\\n' > {mixed_mix}",
                f"python3 - <<'PY'\nimport json\nfrom pathlib import Path\nPath({str(mixed_build_status)!r}).write_text(json.dumps({{'mix_name': 'glossapi_plus_hplt_70_30', 'state': 'completed', 'mix_output_path': {str(mixed_mix)!r}}}, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')\nPY",
                f"python3 - <<'PY'\nimport json, time\nfrom pathlib import Path\nPath({str((mix_root / 'glossapi_plus_hplt_70_30' / 'mix_created.json'))!r}).write_text(json.dumps({{'created_at': time.time()}}, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')\nPY",
                "",
            ]
        ),
        encoding="utf-8",
    )
    delayed_mix_creator.chmod(0o755)
    delayed_proc = subprocess.Popen(["bash", str(delayed_mix_creator)])

    script = TOKENIZER_REPO_ROOT / "subprojects" / "02_1_tokenizer_experiments" / "scripts" / "wait_for_tokenizer_mixes_and_launch_training.sh"
    env = os.environ.copy()
    env["TOKENIZER_TRAINING_PYTHON_BIN"] = str(fake_python)
    env["TOKENIZER_TRAINING_LAUNCH_MODE"] = "inline"
    env["TOKENIZER_TRAINING_INSTALL_DEPS"] = "0"
    env["TOKENIZER_TRAINING_WAIT_INTERVAL_SECONDS"] = "1"
    env["TOKENIZER_TRAINING_GLOSSAPI_NAME"] = "glossapi_only_test"
    env["TOKENIZER_TRAINING_MIXED_NAME"] = "glossapi_plus_hplt_70_30_test"

    subprocess.run(["bash", str(script), str(mix_root), str(training_root)], check=True, env=env)
    delayed_proc.wait(timeout=5)

    glossapi_start = json.loads((training_root / "glossapi_only_test" / "runner_start.json").read_text(encoding="utf-8"))
    mixed_start = json.loads((training_root / "glossapi_plus_hplt_70_30_test" / "runner_start.json").read_text(encoding="utf-8"))
    mixed_mix_created = json.loads((mix_root / "glossapi_plus_hplt_70_30" / "mix_created.json").read_text(encoding="utf-8"))
    glossapi_status = json.loads((training_root / "glossapi_only_test.launch_status.json").read_text(encoding="utf-8"))
    mixed_status = json.loads((training_root / "glossapi_plus_hplt_70_30_test.launch_status.json").read_text(encoding="utf-8"))

    assert glossapi_status["state"] == "completed"
    assert mixed_status["state"] == "completed"
    assert glossapi_start["started_at"] < mixed_mix_created["created_at"]
    assert mixed_start["started_at"] >= mixed_mix_created["created_at"]
    assert (training_root / "glossapi_only_test" / "training_summary.json").exists()
    assert (training_root / "glossapi_plus_hplt_70_30_test" / "training_summary.json").exists()


def test_launch_uploader_handoff_local_stage(tmp_path: Path, monkeypatch) -> None:
    working_root = tmp_path / "working_release"
    handoff_root = tmp_path / "handoff"
    local_stage_root = tmp_path / "local_uploader"

    _write_canonical_rows(
        working_root / "data" / "alpha.parquet",
        [_base_row(source_dataset="alpha", source_doc_id="alpha-1", text="άλλο κείμενο")],
    )
    _write_canonical_rows(
        working_root / "data" / "hplt.parquet",
        [
            _base_row(
                source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60",
                source_doc_id="hplt-1",
                text="ελληνικό κείμενο hplt",
            )
        ],
    )
    (working_root / "dedup_metadata").mkdir(parents=True, exist_ok=True)
    (working_root / "dedup_metadata" / "run-003" / "builder_metadata").mkdir(parents=True, exist_ok=True)
    (working_root / "dedup_metadata" / "latest.json").write_text(
        json.dumps(
            {
                "latest_run_id": "run-003",
                "builder_metadata_root": "dedup_metadata/run-003/builder_metadata",
                "path": "dedup_metadata/run-003",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (working_root / "hplt_integration_summary.json").write_text(
        json.dumps({"new_dataset_name": "HPLT/ell_Grek_ge8_no_mt_clean60"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (working_root / "README.md").write_text("# dataset\n", encoding="utf-8")
    (working_root / "notes.tmp").write_text("do not stage me\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_hf_uploader_handoff.py",
            "--working-release-root",
            str(working_root),
            "--handoff-root",
            str(handoff_root),
            "--repo-id",
            "fffoivos/glossapi-greek-nanochat-pretraining-dataset",
            "--public",
        ],
    )
    prepare_hf_uploader_handoff.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_hf_uploader_handoff.py",
            "--handoff-json",
            str(handoff_root / "uploader_handoff.json"),
            "--local-stage-root",
            str(local_stage_root),
            "--skip-launch",
        ],
    )
    launch_hf_uploader_handoff.main()

    launch_summary = json.loads((handoff_root / "launch_summary.json").read_text(encoding="utf-8"))
    staged_root = Path(str(launch_summary["staged_release_root"]))
    assert staged_root.exists()
    assert (staged_root / "data" / "hplt.parquet").exists()
    assert (staged_root / "dedup_metadata" / "latest.json").exists()
    assert (staged_root / "README.md").exists()
    assert not (staged_root / "notes.tmp").exists()
    assert launch_summary["sync_executed"] is True
    assert launch_summary["launch_executed"] is False


def test_launch_uploader_handoff_local_stage_source_only(tmp_path: Path, monkeypatch) -> None:
    working_root = tmp_path / "working_release"
    handoff_root = tmp_path / "handoff"
    local_stage_root = tmp_path / "local_uploader"

    _write_canonical_rows(
        working_root / "data" / "hplt.parquet",
        [
            _base_row(
                source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60",
                source_doc_id="hplt-1",
                text="ελληνικό κείμενο hplt",
            )
        ],
    )
    (working_root / "README.md").write_text("# dataset\n", encoding="utf-8")
    (working_root / "dedup_metadata").mkdir(parents=True, exist_ok=True)
    (working_root / "dedup_metadata" / "latest.json").write_text("{}\n", encoding="utf-8")
    (working_root / "hplt_integration_summary.json").write_text(
        json.dumps({"new_dataset_name": "HPLT/ell_Grek_ge8_no_mt_clean60"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_hf_uploader_handoff.py",
            "--working-release-root",
            str(working_root),
            "--handoff-root",
            str(handoff_root),
            "--repo-id",
            "fffoivos/glossapi-greek-nanochat-pretraining-dataset",
            "--public",
            "--source-only",
        ],
    )
    prepare_hf_uploader_handoff.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_hf_uploader_handoff.py",
            "--handoff-json",
            str(handoff_root / "uploader_handoff.json"),
            "--local-stage-root",
            str(local_stage_root),
            "--skip-launch",
        ],
    )
    launch_hf_uploader_handoff.main()

    launch_summary = json.loads((handoff_root / "launch_summary.json").read_text(encoding="utf-8"))
    staged_root = Path(str(launch_summary["staged_release_root"]))
    assert staged_root.exists()
    assert (staged_root / "data" / "hplt.parquet").exists()
    assert (staged_root / "README.md").exists()
    assert not (staged_root / "dedup_metadata").exists()


def test_wait_for_uploader_handoff_launch_shell(tmp_path: Path, monkeypatch) -> None:
    working_root = tmp_path / "working_release"
    handoff_root = tmp_path / "handoff"
    local_stage_root = tmp_path / "local_uploader"

    _write_canonical_rows(
        working_root / "data" / "alpha.parquet",
        [_base_row(source_dataset="alpha", source_doc_id="alpha-1", text="άλλο κείμενο")],
    )
    _write_canonical_rows(
        working_root / "data" / "hplt.parquet",
        [
            _base_row(
                source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60",
                source_doc_id="hplt-1",
                text="ελληνικό κείμενο hplt",
            )
        ],
    )
    (working_root / "dedup_metadata").mkdir(parents=True, exist_ok=True)
    (working_root / "dedup_metadata" / "run-004" / "builder_metadata").mkdir(parents=True, exist_ok=True)
    (working_root / "dedup_metadata" / "latest.json").write_text(
        json.dumps(
            {
                "latest_run_id": "run-004",
                "builder_metadata_root": "dedup_metadata/run-004/builder_metadata",
                "path": "dedup_metadata/run-004",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (working_root / "hplt_integration_summary.json").write_text(
        json.dumps({"new_dataset_name": "HPLT/ell_Grek_ge8_no_mt_clean60"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_hf_uploader_handoff.py",
            "--working-release-root",
            str(working_root),
            "--handoff-root",
            str(handoff_root),
            "--repo-id",
            "fffoivos/glossapi-greek-nanochat-pretraining-dataset",
            "--public",
        ],
    )
    prepare_hf_uploader_handoff.main()

    script = TOKENIZER_REPO_ROOT / "ops" / "upload" / "wait_for_uploader_handoff_and_launch.sh"
    env = os.environ.copy()
    env["UPLOAD_LOCAL_STAGE_ROOT"] = str(local_stage_root)
    env["UPLOAD_SKIP_LAUNCH"] = "1"
    env["TOKENIZER_PIPELINE_PYTHON_BIN"] = str(GLOSSAPI_WORK_ROOT / ".venv" / "bin" / "python")
    subprocess.run(["bash", str(script), str(handoff_root)], check=True, env=env)

    launch_summary = json.loads((handoff_root / "launch_summary.json").read_text(encoding="utf-8"))
    staged_root = Path(str(launch_summary["staged_release_root"]))
    assert staged_root.exists()
    assert (staged_root / "data" / "hplt.parquet").exists()


def test_publish_hf_release_uses_upload_large_folder(monkeypatch, tmp_path: Path, capsys) -> None:
    module = _load_module_from_path("publish_hf_release_under_test", TOKENIZER_REPO_ROOT / "publish_hf_release.py")
    calls: dict[str, object] = {}

    class FakeHfApi:
        def __init__(self, *, token: str) -> None:
            calls["token"] = token

        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs

        def upload_large_folder(self, **kwargs):
            calls["upload_large_folder"] = kwargs

    monkeypatch.setattr(module, "HfApi", FakeHfApi)
    monkeypatch.setattr(module, "get_token", lambda: "hf_test_token")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_hf_release.py",
            "--release-root",
            str(tmp_path),
            "--repo-id",
            "fffoivos/glossapi-greek-nanochat-pretraining-dataset",
            "--public",
            "--num-workers",
            "7",
            "--print-report-every",
            "30",
        ],
    )

    module.main()
    captured = capsys.readouterr()
    assert "https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset" in captured.out
    assert calls["create_repo"] == {
        "repo_id": "fffoivos/glossapi-greek-nanochat-pretraining-dataset",
        "repo_type": "dataset",
        "private": False,
        "exist_ok": True,
    }
    assert calls["upload_large_folder"] == {
        "repo_id": "fffoivos/glossapi-greek-nanochat-pretraining-dataset",
        "repo_type": "dataset",
        "folder_path": tmp_path,
        "num_workers": 7,
        "print_report": True,
        "print_report_every": 30,
    }
