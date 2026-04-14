#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
GLOSSAPI_WORK_ROOT = Path("/home/foivos/data/glossapi_work")
HF_RELEASE_ROOT = GLOSSAPI_WORK_ROOT / "hf_release_publish"

sys.path.insert(0, str(REPO_ROOT / "subprojects" / "01_hplt_filtering" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "subprojects" / "01_1_corpus_dedup" / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from glossapi_corpus_cli import pipeline  # noqa: E402
import build_hplt_hf_slice  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny real-document end-to-end smoke test for the tokenizer corpus pipeline.")
    parser.add_argument("--source-release-root", type=Path, required=True)
    parser.add_argument("--raw-hplt-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--corpus-python", required=True)
    parser.add_argument("--tokenizer-python", required=True)
    parser.add_argument("--base-tokenizer", default="swiss-ai/Apertus-8B-2509")
    parser.add_argument("--new-dataset-name", default="HPLT/ell_Grek_ge8_no_mt_clean60")
    parser.add_argument("--old-dataset-name", default="HPLT/ell_Grek_ge8_no_mt")
    parser.add_argument("--hplt-doc-target", type=int, default=40)
    parser.add_argument("--vocab-size", type=int, default=1024)
    return parser.parse_args()


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)


def write_canonical_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame = pipeline.finalize_frame(frame)
    table = pa.Table.from_pandas(
        frame[pipeline.CANONICAL_COLUMNS],
        schema=pipeline.CANONICAL_ARROW_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, path, compression="zstd")


def sample_release_rows(
    source_release_root: Path,
    *,
    old_dataset_name: str,
) -> list[dict[str, Any]]:
    data_root = source_release_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data dir: {data_root}")

    oa_true: dict[str, Any] | None = None
    oa_false: dict[str, Any] | None = None
    other_rows: list[dict[str, Any]] = []
    seen_other_datasets: set[str] = set()
    target_other_rows = 8

    def iter_rows(paths: list[Path], *, batch_size: int = 256) -> list[dict[str, Any]]:
        for path in paths:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    yield row

    openarchives_paths = sorted(data_root.glob("openarchives.gr*.parquet"))
    for row in iter_rows(openarchives_paths):
        if bool(row.get("needs_ocr")) and oa_true is None:
            oa_true = row
        if not bool(row.get("needs_ocr")) and oa_false is None:
            oa_false = row
        if oa_true and oa_false:
            break

    hplt_pattern = f"{old_dataset_name.replace('/', '__')}*.parquet"
    hplt_paths = sorted(data_root.glob(hplt_pattern))
    excluded = {path.name for path in openarchives_paths + hplt_paths}
    for path in sorted(data_root.glob("*.parquet")):
        if path.name in excluded:
            continue
        for row in iter_rows([path], batch_size=1):
            dataset = str(row["source_dataset"])
            if dataset in seen_other_datasets:
                continue
            other_rows.append(row)
            seen_other_datasets.add(dataset)
            break
        if len(other_rows) >= target_other_rows:
            break

    if oa_true and oa_false and len(other_rows) >= target_other_rows:
        return [oa_true, oa_false, *other_rows[:target_other_rows]]

    missing = []
    if oa_true is None:
        missing.append("openarchives.gr needs_ocr=true row")
    if oa_false is None:
        missing.append("openarchives.gr needs_ocr=false row")
    if len(other_rows) < target_other_rows:
        missing.append(f"{target_other_rows} non-HPLT source rows")
    raise RuntimeError(f"Could not assemble real-doc sample from release: {missing}")


def sample_real_hplt_rows(raw_hplt_root: Path, target_docs: int) -> list[dict[str, Any]]:
    shard_path = raw_hplt_root / "10_1.jsonl.zst"
    if not shard_path.exists():
        candidates = sorted(raw_hplt_root.glob("*.jsonl.zst"))
        if not candidates:
            raise FileNotFoundError(f"No HPLT shards found under {raw_hplt_root}")
        shard_path = candidates[0]

    rows: list[dict[str, Any]] = []
    for raw_row in build_hplt_hf_slice.stream_hplt_rows(shard_path.as_uri()):
        row = build_hplt_hf_slice.build_base_row(
            raw_row,
            dataset_name="HPLT/ell_Grek_ge8_no_mt_clean60",
            shard=shard_path.name,
            quality_bin=build_hplt_hf_slice.shard_quality_bin(shard_path.name),
        )
        if row is None:
            continue
        if row.pop("_top_main_code", None) == "MT":
            continue
        rows.append(raw_row)
        if len(rows) >= target_docs:
            return rows
    raise RuntimeError(f"Could not sample {target_docs} acceptable real HPLT rows from {shard_path}")


def write_hplt_shard(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_jsonl = path.with_suffix("")
    raw_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    run(["zstd", "-q", "-f", str(raw_jsonl), "-o", str(path)])
    raw_jsonl.unlink()


def assert_mix_contracts(glossapi_mix_path: Path, mixed_mix_path: Path, hplt_dataset_name: str) -> dict[str, Any]:
    glossapi = pd.read_parquet(glossapi_mix_path)
    mixed = pd.read_parquet(mixed_mix_path)

    if any((glossapi["source_dataset"] == "openarchives.gr") & (glossapi["needs_ocr"].fillna(False).astype(bool))):
        raise AssertionError("GlossAPI-only mix still contains openarchives.gr rows with needs_ocr=true")
    if set(glossapi["source_dataset"]).intersection({hplt_dataset_name}):
        raise AssertionError("GlossAPI-only mix unexpectedly contains HPLT rows")

    hplt_chars = int(mixed.loc[mixed["source_dataset"] == hplt_dataset_name, "text"].str.len().sum())
    non_hplt_chars = int(mixed.loc[mixed["source_dataset"] != hplt_dataset_name, "text"].str.len().sum())
    total = hplt_chars + non_hplt_chars
    if total <= 0:
        raise AssertionError("Mixed tokenizer corpus is empty")
    ratio = hplt_chars / total
    if abs(ratio - 0.30) > 0.02:
        raise AssertionError(f"HPLT ratio drifted: expected ~0.30, got {ratio:.4f}")
    return {
        "glossapi_only_rows": int(len(glossapi)),
        "mixed_rows": int(len(mixed)),
        "mixed_hplt_chars": hplt_chars,
        "mixed_non_hplt_chars": non_hplt_chars,
        "mixed_hplt_ratio": ratio,
    }


def assert_tokenizer_contract(base_tokenizer_name: str, trained_dir: Path) -> dict[str, Any]:
    base = AutoTokenizer.from_pretrained(base_tokenizer_name, use_fast=True)
    base_json = json.loads(base.backend_tokenizer.to_str())
    trained_json = json.loads((trained_dir / "tokenizer.json").read_text(encoding="utf-8"))

    for key in ["normalizer", "pre_tokenizer", "decoder", "post_processor"]:
        if base_json.get(key) != trained_json.get(key):
            raise AssertionError(f"Tokenizer front-end mismatch for {key}")
    if trained_json.get("model", {}).get("type") != "BPE":
        raise AssertionError("Trained tokenizer model is not BPE")

    summary = json.loads((trained_dir / "training_summary.json").read_text(encoding="utf-8"))
    return {
        "training_summary": str(trained_dir / "training_summary.json"),
        "vocab_size_actual": int(summary["vocab_size_actual"]),
    }


def main() -> None:
    args = parse_args()

    work_root = args.work_root.resolve()
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    source_release_root = args.source_release_root.resolve()
    raw_hplt_root = args.raw_hplt_root.resolve()
    corpus_python = args.corpus_python
    tokenizer_python = args.tokenizer_python

    tiny_release_root = work_root / "tiny_source_release"
    tiny_working_release_root = work_root / "tiny_working_release"
    tiny_hplt_release_root = work_root / "tiny_hplt_release"
    tiny_raw_hplt_root = work_root / "tiny_raw_hplt"
    state_root = work_root / "dedup_state"
    mix_root = work_root / "mixes"
    training_root = work_root / "training"
    handoff_root = work_root / "upload_handoff"
    local_uploader_root = work_root / "local_uploader"
    summary_path = work_root / "smoke_summary.json"

    tiny_release_root.mkdir(parents=True, exist_ok=True)
    (tiny_release_root / "data").mkdir(exist_ok=True)
    print("stage=sample_release_rows", flush=True)
    real_rows = sample_release_rows(source_release_root, old_dataset_name=args.old_dataset_name)
    sampled_release_datasets: dict[str, int] = {}
    for idx, row in enumerate(real_rows):
        dataset = str(row["source_dataset"])
        sampled_release_datasets[dataset] = sampled_release_datasets.get(dataset, 0) + 1
        write_canonical_rows(tiny_release_root / "data" / f"sample_{idx:03d}.parquet", [row])

    shutil.copytree(tiny_release_root, tiny_working_release_root)

    print("stage=sample_real_hplt_rows", flush=True)
    sampled_hplt_rows = sample_real_hplt_rows(raw_hplt_root, args.hplt_doc_target)
    old_hplt_row = build_hplt_hf_slice.build_base_row(
        sampled_hplt_rows[0],
        dataset_name=args.old_dataset_name,
        shard="10_1.jsonl.zst",
        quality_bin=10,
    )
    if old_hplt_row is None:
        raise RuntimeError("Could not synthesize a tiny old-HPLT row from sampled real HPLT data")
    old_hplt_row.pop("_top_main_code", None)
    write_canonical_rows(tiny_release_root / "data" / "sample_old_hplt.parquet", [old_hplt_row])
    sampled_release_datasets[args.old_dataset_name] = 1
    write_hplt_shard(tiny_raw_hplt_root / "10_1.jsonl.zst", sampled_hplt_rows)

    print("stage=build_hplt_slice", flush=True)
    build_cmd = [
        str(corpus_python),
        str(REPO_ROOT / "subprojects" / "01_hplt_filtering" / "scripts" / "build_hplt_hf_slice.py"),
        "--release-root",
        str(tiny_hplt_release_root),
        "--dataset-name",
        args.new_dataset_name,
        "--hplt-base-url",
        tiny_raw_hplt_root.as_uri(),
        "--only-shards",
        "10_1.jsonl.zst",
        "--quality-min",
        "8",
        "--workers",
        "1",
        "--batch-size",
        "8",
        "--rows-per-part",
        "20",
        "--quality-mode",
        "corpus_clean",
        "--greek-badness-max",
        "60",
        "--clean-num-threads",
        "2",
        "--summary-json",
        str(tiny_hplt_release_root / "hplt_clean60_summary.json"),
        "--no-upload",
    ]
    run(build_cmd)

    print("stage=integrate_hplt_slice", flush=True)
    integrate_wait_cmd = [
        "bash",
        str(REPO_ROOT / "subprojects" / "01_hplt_filtering" / "scripts" / "wait_for_hplt_and_integrate.sh"),
        str(tiny_working_release_root),
        str(tiny_hplt_release_root),
    ]
    run(integrate_wait_cmd)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["GLOSSAPI_WORK_ROOT"] = str(GLOSSAPI_WORK_ROOT)
    print("stage=dedup", flush=True)
    dedup_cmd = [
        "bash",
        str(REPO_ROOT / "subprojects" / "01_1_corpus_dedup" / "scripts" / "wait_for_hplt_integration_and_run_dedup.sh"),
        str(tiny_working_release_root),
        str(state_root),
    ]
    run(dedup_cmd, env=env)

    print("stage=publish_overlay", flush=True)
    overlay_cmd = [
        "bash",
        str(REPO_ROOT / "subprojects" / "01_1_corpus_dedup" / "scripts" / "wait_for_dedup_and_publish_overlay.sh"),
        str(tiny_working_release_root),
        str(state_root),
    ]
    run(overlay_cmd)

    print("stage=prepare_upload_handoff", flush=True)
    handoff_cmd = [
        "bash",
        str(REPO_ROOT / "ops" / "upload" / "wait_for_dedup_overlay_and_prepare_handoff.sh"),
        str(tiny_working_release_root),
        str(state_root),
        str(handoff_root),
    ]
    run(handoff_cmd)

    print("stage=local_stage_upload_handoff", flush=True)
    upload_env = dict(os.environ)
    upload_env["UPLOAD_LOCAL_STAGE_ROOT"] = str(local_uploader_root)
    upload_env["UPLOAD_SKIP_LAUNCH"] = "1"
    upload_env["TOKENIZER_PIPELINE_PYTHON_BIN"] = corpus_python
    local_stage_cmd = [
        "bash",
        str(REPO_ROOT / "ops" / "upload" / "wait_for_uploader_handoff_and_launch.sh"),
        str(handoff_root),
    ]
    run(local_stage_cmd, env=upload_env)
    launch_summary = json.loads((handoff_root / "launch_summary.json").read_text(encoding="utf-8"))
    staged_release_root = Path(str(launch_summary["staged_release_root"]))
    if not (staged_release_root / "data").exists():
        raise AssertionError(f"Local uploader stage is missing data/: {staged_release_root}")
    if not (staged_release_root / "dedup_metadata" / "latest.json").exists():
        raise AssertionError(f"Local uploader stage is missing dedup_metadata/latest.json: {staged_release_root}")

    latest = json.loads((tiny_working_release_root / "dedup_metadata" / "latest.json").read_text(encoding="utf-8"))
    dedup_metadata_root = (tiny_working_release_root / latest["builder_metadata_root"]).resolve()
    mix_root.mkdir(parents=True, exist_ok=True)
    glossapi_only_mix = mix_root / "glossapi_only" / "mix.parquet"
    mixed_mix = mix_root / "glossapi_plus_hplt_70_30" / "mix.parquet"
    print("stage=build_mixes", flush=True)
    mix_cmd = [
        "bash",
        str(REPO_ROOT / "subprojects" / "01_2_training_dataset_mix" / "scripts" / "wait_for_dedup_overlay_and_build_tokenizer_mixes.sh"),
        str(tiny_working_release_root),
        str(state_root),
        str(mix_root),
    ]
    run(mix_cmd, env=env)

    mix_contracts = assert_mix_contracts(glossapi_only_mix, mixed_mix, args.new_dataset_name)

    print("stage=train_tokenizers", flush=True)
    training_root.mkdir(parents=True, exist_ok=True)
    glossapi_train_dir = training_root / "glossapi_only_1k"
    mixed_train_dir = training_root / "glossapi_plus_hplt_70_30_1k"
    train_env = dict(os.environ)
    train_env["TOKENIZER_TRAINING_LAUNCH_MODE"] = "inline"
    train_env["TOKENIZER_TRAINING_PYTHON_BIN"] = tokenizer_python
    train_env["TOKENIZER_TRAINING_INSTALL_DEPS"] = "0"
    train_env["TOKENIZER_TRAINING_BASE_TOKENIZER"] = args.base_tokenizer
    train_env["TOKENIZER_TRAINING_VOCAB_SIZE"] = str(args.vocab_size)
    train_env["TOKENIZER_TRAINING_GLOSSAPI_NAME"] = "glossapi_only_1k"
    train_env["TOKENIZER_TRAINING_MIXED_NAME"] = "glossapi_plus_hplt_70_30_1k"
    train_env["TOKENIZER_TRAINING_RAYON_THREADS"] = "2"
    train_cmd = [
        "bash",
        str(REPO_ROOT / "subprojects" / "02_1_tokenizer_experiments" / "scripts" / "wait_for_tokenizer_mixes_and_launch_training.sh"),
        str(mix_root),
        str(training_root),
    ]
    run(train_cmd, env=train_env)

    tokenizer_contracts = {
        "glossapi_only": assert_tokenizer_contract(args.base_tokenizer, glossapi_train_dir),
        "glossapi_plus_hplt_70_30": assert_tokenizer_contract(args.base_tokenizer, mixed_train_dir),
    }

    summary = {
        "work_root": str(work_root),
        "sampled_release_datasets": sampled_release_datasets,
        "sampled_hplt_rows": len(sampled_hplt_rows),
        "hplt_summary": str(tiny_hplt_release_root / "hplt_clean60_summary.json"),
        "integration_summary": str(tiny_working_release_root / "hplt_integration_summary.json"),
        "dedup_latest_success": str(state_root / "latest_success.json"),
        "dedup_latest_json": str(tiny_working_release_root / "dedup_metadata" / "latest.json"),
        "uploader_handoff_summary": str(handoff_root / "handoff_summary.json"),
        "uploader_launch_summary": str(handoff_root / "launch_summary.json"),
        "local_uploader_root": str(local_uploader_root),
        "glossapi_only_mix": str(glossapi_only_mix),
        "glossapi_plus_hplt_70_30_mix": str(mixed_mix),
        "mix_contracts": mix_contracts,
        "tokenizer_contracts": tokenizer_contracts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
