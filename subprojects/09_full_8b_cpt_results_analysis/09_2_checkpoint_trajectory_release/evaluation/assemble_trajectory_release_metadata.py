#!/usr/bin/env python3
"""Assemble branch-specific, receipt-bound Hugging Face release metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


TOKENS_PER_UPDATE = 4_194_304
REPOSITORY_ID = "fffoivos/apertus-8b-greek-cpt"
PUBLIC_TRAIN_REPOSITORY_ID = "fffoivos/apertus-8b-greek-cpt-modern-greek-train"
PRIVATE_FULL_MIX_REPOSITORY_ID = "fffoivos/apertus-8b-greek-cpt-d0-full-mix"
PARENT = {"repo_id": "swiss-ai/Apertus-8B-2509", "revision": "3162c99675aa588097cecd4a24b9aa1f712af477"}
TOKENIZER = {"repo_id": "fffoivos/apertus-tokenizer-extension", "sha256": "acf4d5c6a8aa0cc64c9c781c203fd6dbb4581b0124cca0a76a9e10322fd81092"}
NATIVE_COUNTS = {
    "asep_mcqa": 1180, "demosqa": 599, "gpcr": 194, "medical_mcqa": 419,
    "oyxoy_metaphor": 2042, "oyxoy_nli": 5244, "oyxoy_wic": 54217,
    "oyxoy_wsd_definition": 9999,
}
CHECKPOINTS = [
    (400, 0, "step400-tokens2B"), (1192, 0, "step1192-tokens5B"),
    (2384, 0, "step2384-tokens10B"), (3576, 0, "step3576-tokens15B"),
    (4768, 0, "step4768-tokens20B"), (5960, 0, "step5960-tokens25B"),
    (7152, 0, "step7152-tokens30B"), (8344, 0, "step8344-tokens35B"),
    (9536, 0, "step9536-tokens40B"), (10728, 0, "step10728-tokens45B"),
    (11920, 0, "step11920-tokens50B"), (13112, 0, "step13112-tokens55B"),
    (14304, 0, "step14304-tokens60B"), (14627, 0, "step14627-tokens61B"),
    (15496, 1, "step15496-tokens65B"), (16688, 0, "step16688-tokens70B"),
    (17880, 0, "step17880-tokens75B"), (18284, 0, "main"),
]
PEAK_ITERATIONS = {7152, 8344, 9536, 10728, 11920}
FINAL_ITERATION = 18284


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def label(iteration: int) -> str:
    return f"iter_{iteration:07d}"


def parse_metrics(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("subject") != "__all__":
                continue
            benchmark = row["benchmark"]
            if benchmark.endswith("_exact_set"):
                continue
            value: dict[str, Any] = {"n": int(row["n"])}
            for key in ("accuracy", "choice_nll", "correct_answer_bpb", "balanced_accuracy", "binary_macro_f1"):
                value[key] = float(row[key]) if row.get(key) else None
            result[benchmark] = value
    require(set(result) == set(NATIVE_COUNTS), f"native benchmark set drift in {path}")
    for benchmark, count in NATIVE_COUNTS.items():
        require(result[benchmark]["n"] == count, f"native count drift for {benchmark}")
    return result


def existing_peak_metrics(path: Path) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
    value = read_json(path)
    require(value.get("status") == "completed", "peak result is not completed")
    rows: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in value.get("table", []):
        iteration = int(row["iteration"])
        if iteration not in PEAK_ITERATIONS:
            continue
        metrics = row["benchmarks"]
        require(set(metrics) >= set(NATIVE_COUNTS), f"peak benchmark coverage drift: {iteration}")
        for benchmark, count in NATIVE_COUNTS.items():
            require(int(metrics[benchmark]["n"]) == count, f"peak native count drift: {iteration}/{benchmark}")
        source = row["artifacts"]["clean_subset_metrics"]
        artifact = Path(source["path"])
        require(artifact.is_file() and sha256(artifact) == source["sha256"], f"peak metrics hash drift: {iteration}")
        rows[iteration] = (metrics, {"path": str(artifact), "sha256": source["sha256"], "method": row["provenance"]})
    require(set(rows) == PEAK_ITERATIONS, "peak checkpoint scope drift")
    return rows


def remaining_metrics(root: Path) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
    receipt = read_json(root / "matrix_receipt.json")
    require(receipt.get("status") == "completed", "remaining matrix is not completed")
    rows: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for checkpoint in receipt.get("checkpoint_receipts", []):
        item = read_json(Path(checkpoint["path"]))
        require(item.get("status") == "completed", "remaining aggregate is incomplete")
        current_label = str(item["model"])
        iteration = int(current_label.split("_")[-1])
        metrics_path = Path(item["artifacts"]["metrics"]["path"])
        require(sha256(metrics_path) == item["artifacts"]["metrics"]["sha256"], "remaining metrics hash drift")
        rows[iteration] = (parse_metrics(metrics_path), {"path": str(metrics_path), "sha256": sha256(metrics_path), "method": "direct_frozen_clean_subset"})
    expected = {iteration for iteration, _, _ in CHECKPOINTS} - PEAK_ITERATIONS - {FINAL_ITERATION}
    require(set(rows) == expected, "remaining matrix checkpoint scope drift")
    return rows


def greekmmlu(run_root: Path, iteration: int, attempt: int) -> tuple[dict[str, Any], dict[str, Any]]:
    root = run_root / "checkpoint_evaluations" / label(iteration) / f"attempt_{attempt}"
    path = root / "exact_checkpoint_native_greekmmlu_receipt.json"
    export_receipt = root / "export" / "checkpoint_eval_export_receipt.json"
    value = read_json(path)
    require(value.get("status") == "completed", f"GreekMMLU receipt incomplete: {iteration}")
    checkpoint = value["checkpoint"]
    require(int(checkpoint["iteration"]) == iteration, f"GreekMMLU iteration drift: {iteration}")
    require(Path(checkpoint["export_receipt_path"]).resolve() == export_receipt.resolve(), f"GreekMMLU export binding drift: {iteration}")
    require(checkpoint["export_receipt_sha256"] == sha256(export_receipt), f"GreekMMLU export hash drift: {iteration}")
    metric = value["metrics"]
    clean = metric["decontaminated"]
    require(int(metric["n"]) == 16632 and int(clean["n"]) == 16159, f"GreekMMLU population drift: {iteration}")
    return {
        "full": {key: metric[key] for key in ("n", "accuracy", "choice_nll", "correct_answer_bpb")},
        "decontaminated": clean,
        "evaluator": value["evaluator"],
        "clean_subset_manifest": value["clean_subset_manifest"],
    }, {"path": str(path), "sha256": sha256(path)}


def format_number(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.2f}%" if percent else f"{value:.4f}"


def frozen_dataset_upload(path: Path, *, repo_id: str, private: bool) -> dict[str, Any]:
    """Bind release cards to the exact companion-dataset commit, not a moving main."""
    value = read_json(path)
    require(value.get("schema_version") == "apertus_full8_frozen_dataset_hf_upload_v1", "dataset upload receipt schema drift")
    require(value.get("status") == "completed", "dataset upload receipt is incomplete")
    require(value.get("repo_id") == repo_id, "dataset upload repository drift")
    require(bool(value.get("private")) is private, "dataset upload visibility drift")
    revision = str(value.get("revision", ""))
    require(len(revision) == 40 and value.get("training_must_pin_revision") == revision, "dataset upload revision is not immutable")
    return {"repo_id": repo_id, "revision": revision, "private": private, "receipt": {"path": str(path.resolve()), "sha256": sha256(path)}}


def card(row: dict[str, Any], *, trajectory: list[dict[str, Any]], datasets: dict[str, dict[str, Any]]) -> str:
    checkpoint = row["checkpoint"]
    greek = row["greekmmlu"]["decontaminated"]
    lines = [
        "---", "language: el", "license: apache-2.0", "library_name: transformers", "pipeline_tag: text-generation", "---", "",
        f"# Apertus 8B Greek CPT — {checkpoint['branch']}", "",
        "This branch is one immutable checkpoint from the same continued-pretraining trajectory. "
        "It is not an instruction-tuned model.", "",
        "## Checkpoint", "",
        f"- Update: `{checkpoint['iteration']}`", f"- Token slots consumed: `{checkpoint['token_slots']:,}` ({checkpoint['token_slots'] / 1e9:.3f}B)",
        f"- Parent: [`{PARENT['repo_id']}`](https://huggingface.co/{PARENT['repo_id']}/tree/{PARENT['revision']}) at `{PARENT['revision']}`",
        "- Geometry: 8B, 148,992-token extended vocabulary, untied input/output embeddings, RoPE θ=500,000, context 4,096.",
        "- CPT mixture: stationary 79% Modern Greek, 20% foreign-language replay, 1% Old-Greek replay.", "",
        "## Evaluation", "",
        "Primary values below use the decontaminated question population. `evaluation/metrics.json`, "
        "`evaluation/population.json`, and `evaluation/provenance.json` bind every displayed value to its source receipt.", "",
        "| Benchmark | Questions used | Accuracy | Choice NLL | Correct-answer BPB |", "| --- | ---: | ---: | ---: | ---: |",
        f"| GreekMMLU (decontaminated) | {greek['n']:,} | {format_number(greek['accuracy'], percent=True)} | {format_number(greek['choice_nll'])} | {format_number(greek['correct_answer_bpb'])} |",
    ]
    for benchmark, metric in sorted(row["native_greek_suite"]["benchmarks"].items()):
        lines.append(f"| {benchmark} | {metric['n']:,} | {format_number(metric['accuracy'], percent=True)} | {format_number(metric['choice_nll'])} | {format_number(metric['correct_answer_bpb'])} |")
    public_train, private_mix = datasets["public_modern_greek_train"], datasets["private_d0_full_mix"]
    lines += ["", "## Frozen training-data provenance", "", f"- Exact public Modern-Greek train-only snapshot: [`{public_train['repo_id']}`](https://huggingface.co/datasets/{public_train['repo_id']}/tree/{public_train['revision']}) at `{public_train['revision']}`.", f"- Exact packed 79/20/1 D0 mixture: [`{private_mix['repo_id']}`](https://huggingface.co/datasets/{private_mix['repo_id']}/tree/{private_mix['revision']}) at `{private_mix['revision']}` (private because replay redistribution is restricted).", "- These companion snapshots document the already-trained content only; neither changes the checkpoint’s text order, masking, tokenizer, or weights.", "", "## Which questions were used?", "", "The native Greek suite starts with 83,970 scored rows. It excludes 10,076 only when the evaluation identity has a strong two-surface match to the CPT corpus, leaving 73,894 rows. The public audit includes the excluded IDs and matched document/line evidence: [exclusion list](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2/blob/main/benchmark_contamination/native_greek_suite_v1/recommended_excluded_example_ids.jsonl) and [match table](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2/blob/main/benchmark_contamination/native_greek_suite_v1/qa_document_line_matches.parquet). GreekMMLU uses its separate frozen 16,159-question decontaminated subset.", ""]
    if checkpoint["branch"] == "main":
        lines += ["## GreekMMLU trajectory", "", "| Branch | Token slots | Clean accuracy | Clean choice NLL |", "| --- | ---: | ---: | ---: |"]
        for item in trajectory:
            g = item["greekmmlu"]["decontaminated"]
            c = item["checkpoint"]
            lines.append(f"| {c['branch']} | {c['token_slots'] / 1e9:.3f}B | {format_number(g['accuracy'], percent=True)} | {format_number(g['choice_nll'])} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--remaining-root", type=Path, required=True)
    parser.add_argument("--peak-results", type=Path, required=True)
    parser.add_argument("--final-filtered-metrics", type=Path, required=True)
    parser.add_argument("--final-filtered-receipt", type=Path, required=True)
    parser.add_argument("--public-train-upload-receipt", type=Path, required=True)
    parser.add_argument("--private-full-mix-upload-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing to overwrite release metadata")
    peak = existing_peak_metrics(args.peak_results)
    remaining = remaining_metrics(args.remaining_root)
    final_receipt = read_json(args.final_filtered_receipt)
    require(final_receipt.get("status") == "completed", "final filtered evidence is incomplete")
    datasets = {
        "public_modern_greek_train": frozen_dataset_upload(args.public_train_upload_receipt, repo_id=PUBLIC_TRAIN_REPOSITORY_ID, private=False),
        "private_d0_full_mix": frozen_dataset_upload(args.private_full_mix_upload_receipt, repo_id=PRIVATE_FULL_MIX_REPOSITORY_ID, private=True),
    }
    final_native = parse_metrics(args.final_filtered_metrics)
    trajectory: list[dict[str, Any]] = []
    for iteration, attempt, branch in CHECKPOINTS:
        if iteration in PEAK_ITERATIONS:
            native, native_source = peak[iteration]
        elif iteration == FINAL_ITERATION:
            native = final_native
            native_source = {"path": str(args.final_filtered_metrics), "sha256": sha256(args.final_filtered_metrics), "method": "strict_filtered_existing_predictions", "receipt": {"path": str(args.final_filtered_receipt), "sha256": sha256(args.final_filtered_receipt)}}
        else:
            native, native_source = remaining[iteration]
        greek, greek_source = greekmmlu(args.run_root, iteration, attempt)
        trajectory.append({
            "checkpoint": {"label": label(iteration), "iteration": iteration, "attempt": attempt, "branch": branch, "token_slots": iteration * TOKENS_PER_UPDATE},
            "greekmmlu": greek,
            "native_greek_suite": {"scoring": {"dtype": "float32", "candidate_batch_size": 1, "example_batch_size": 16, "mode": "legacy"}, "benchmarks": native},
            "provenance": {"greekmmlu_receipt": greek_source, "native_metrics": native_source},
        })
    args.output_dir.mkdir(parents=True)
    population = {"schema_version": "apertus_full8_release_question_population_v1", "status": "passed", "native_greek": {"source_scored_rows": 83970, "excluded_strong_matches": 10076, "retained_rows": 73894, "counts": NATIVE_COUNTS, "training_dataset": "fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2", "training_dataset_revision": "987b8955fcd395c6219e39df9e64715457f69065", "exclusions_sha256": "7a8559461b15a308f599faf0ff25cd16c07be0a597078864f779af7f2f1fdd32", "audit_receipt_sha256": "78273fcc9e45e2e8d54c72d4b765fe9d43af2b29a5602c59b1aab8cee4849feb"}, "greekmmlu": {"full_rows": 16632, "decontaminated_rows": 16159}}
    index = {"schema_version": "apertus_full8_checkpoint_trajectory_release_index_v1", "status": "ready_for_private_release", "repository_id": REPOSITORY_ID, "parent": PARENT, "tokenizer": TOKENIZER, "datasets": datasets, "checkpoints": trajectory, "population": population}
    for row in trajectory:
        branch_root = args.output_dir / "branches" / row["checkpoint"]["branch"]
        (branch_root / "evaluation").mkdir(parents=True)
        (branch_root / "README.md").write_text(card(row, trajectory=trajectory, datasets=datasets), encoding="utf-8")
        write_json(branch_root / "evaluation" / "metrics.json", row)
        write_json(branch_root / "evaluation" / "population.json", population)
        write_json(branch_root / "evaluation" / "provenance.json", {"schema_version": "apertus_full8_checkpoint_release_provenance_v1", "status": "passed", "checkpoint": row["checkpoint"], "datasets": datasets, "provenance": row["provenance"]})
    write_json(args.output_dir / "release_index.json", index)
    print(json.dumps({"ok": True, "branches": len(trajectory), "output": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
