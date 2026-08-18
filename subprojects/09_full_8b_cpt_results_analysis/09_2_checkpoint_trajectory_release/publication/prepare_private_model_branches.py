#!/usr/bin/env python3
"""Create receipt-bound *private* HF release contracts for the 8B trajectory.

This is deliberately a small experiment adapter.  It does not upload models,
convert checkpoints, or calculate scores.  The canonical checkpoint publisher
in ``apertus-cscs-efficiency`` executes the resulting contracts on an Xfer
node.  Keeping contract creation separate makes the data identity and the
branch-to-export binding inspectable before a large upload starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


TOKENS_PER_UPDATE = 4_194_304
PARENT = {"repo_id": "swiss-ai/Apertus-8B-2509", "revision": "3162c99675aa588097cecd4a24b9aa1f712af477"}
TOKENIZER = {"repo_id": "fffoivos/apertus-tokenizer-extension", "revision": "fcd33ec09fb7d86bc072b3a4b3e890efa6473b66"}
GEOMETRY = {
    "vocab_size": 148_992,
    "hidden_size": 4_096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rope_theta": 500_000.0,
    "max_position_embeddings": 4_096,
}
CHECKPOINTS = (
    (400, 0, "step400-tokens2B"), (1192, 0, "step1192-tokens5B"),
    (2384, 0, "step2384-tokens10B"), (3576, 0, "step3576-tokens15B"),
    (4768, 0, "step4768-tokens20B"), (5960, 0, "step5960-tokens25B"),
    (7152, 0, "step7152-tokens30B"), (8344, 0, "step8344-tokens35B"),
    (9536, 0, "step9536-tokens40B"), (10728, 0, "step10728-tokens45B"),
    (11920, 0, "step11920-tokens50B"), (13112, 0, "step13112-tokens55B"),
    (14304, 0, "step14304-tokens60B"), (14627, 0, "step14627-tokens61B"),
    (15496, 1, "step15496-tokens65B"), (16688, 0, "step16688-tokens70B"),
    (17880, 0, "step17880-tokens75B"), (18284, 0, "main"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def label(iteration: int) -> str:
    return f"iter_{iteration:07d}"


def bounded_file(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing evidence: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_export(path: Path, iteration: int) -> tuple[dict[str, Any], Path, str]:
    export = read_json(path)
    require(export.get("schema_version") == "native_greekmmlu_exact_checkpoint_export_v1", "export schema drift")
    require(export.get("status") == "completed", "export is not completed")
    require(export.get("model_scale") == "8B", "export scale drift")
    require(int(export.get("source", {}).get("iteration", -1)) == iteration, "export iteration drift")
    hf = export.get("hf_export")
    require(isinstance(hf, dict), "export lacks HF binding")
    hf_root = Path(str(hf.get("path", ""))).resolve()
    require(hf_root.is_dir(), f"export HF directory is missing: {hf_root}")
    config = read_json(hf_root / "config.json")
    for key, expected in GEOMETRY.items():
        require(config.get(key) == expected, f"{path}: converted geometry drift: {key}")
    require(config.get("tie_word_embeddings") is False, "converted export unexpectedly ties embeddings")
    tokenizer_sha = str(hf.get("tokenizer_json_sha256", ""))
    require(len(tokenizer_sha) == 64, "export tokenizer hash is invalid")
    require(sha256_file(hf_root / "tokenizer.json") == tokenizer_sha, "exported tokenizer hash drift")
    files = hf.get("files")
    require(isinstance(files, list) and files, "export file inventory is empty")
    for row in files:
        relative = Path(str(row.get("relative_path", "")))
        require(not relative.is_absolute() and ".." not in relative.parts, "nonportable export inventory path")
        file_path = hf_root / relative
        require(file_path.is_file(), f"export file missing: {relative}")
        require(file_path.stat().st_size == int(row["bytes"]), f"export file size drift: {relative}")
        require(sha256_file(file_path) == row["sha256"], f"export file hash drift: {relative}")
    return export, hf_root, tokenizer_sha


def validate_greekmmlu(path: Path, iteration: int, export_path: Path) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema_version") == "exact_checkpoint_native_greekmmlu_receipt_v1", "GreekMMLU schema drift")
    require(value.get("status") == "completed", "GreekMMLU is not completed")
    checkpoint = value.get("checkpoint")
    require(isinstance(checkpoint, dict) and int(checkpoint.get("iteration", -1)) == iteration, "GreekMMLU iteration drift")
    require(Path(str(checkpoint.get("export_receipt_path", ""))).resolve() == export_path.resolve(), "GreekMMLU export path drift")
    require(checkpoint.get("export_receipt_sha256") == sha256_file(export_path), "GreekMMLU export hash drift")
    clean = value.get("metrics", {}).get("decontaminated")
    require(isinstance(clean, dict) and int(clean.get("n", 0)) == 16_159, "GreekMMLU clean population drift")
    return value


def card(*, branch: str, iteration: int, greekmmlu: dict[str, Any]) -> str:
    clean = greekmmlu["metrics"]["decontaminated"]
    tokens = iteration * TOKENS_PER_UPDATE
    return "\n".join((
        "---", "language: el", "license: apache-2.0", "library_name: transformers", "pipeline_tag: text-generation", "---", "",
        f"# Apertus 8B Greek CPT — {branch}", "",
        "This is one immutable checkpoint from a continued-pretraining trajectory. It is a base model, not instruction tuned.", "",
        "## Private staging status", "",
        "This branch is private staging while the full, predeclared native-Greek checkpoint matrix completes. The weights and conversion evidence are frozen; a later metadata-only release pass will add the complete score matrix before any public promotion.", "",
        "## Checkpoint", "",
        f"- Update: `{iteration}`", f"- Token slots consumed: `{tokens:,}` ({tokens / 1e9:.3f}B)",
        f"- Parent: [`{PARENT['repo_id']}`](https://huggingface.co/{PARENT['repo_id']}/tree/{PARENT['revision']}) at `{PARENT['revision']}`",
        "- Geometry: 8B; 148,992-token extended vocabulary; untied input/output embeddings; RoPE θ=500,000; context 4,096.",
        "- Training mix: stationary 79% Modern Greek, 20% foreign-language replay, 1% Old-Greek replay.", "",
        "## Frozen GreekMMLU point", "",
        f"- Decontaminated GreekMMLU: {int(clean['n']):,} questions; accuracy `{100 * float(clean['accuracy']):.2f}%`; choice NLL `{float(clean['choice_nll']):.4f}`; correct-answer BPB `{float(clean['correct_answer_bpb']):.4f}`.",
        "- The frozen 16,159-question subset is held constant across this trajectory. The branch provenance receipt contains the evaluator and source-receipt bindings.", "",
        "## Training-data provenance", "",
        "- Public Modern-Greek train-only snapshot: [`fffoivos/apertus-8b-greek-cpt-modern-greek-train`](https://huggingface.co/datasets/fffoivos/apertus-8b-greek-cpt-modern-greek-train) (revision added after its immutable upload completes).",
        "- Exact packed D0 79/20/1 mixture, including constrained replay, is held in the private companion dataset `fffoivos/apertus-8b-greek-cpt-d0-full-mix` and is not a redistribution grant for replay sources.",
        "- Neither checkpoint publication nor dataset packaging changes the trained text, token sequence order, masking, tokenizer, or model weights.", "",
    ))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    require(not output.exists(), f"refusing to overwrite immutable output: {output}")
    require(args.model_repo == "fffoivos/apertus-8b-greek-cpt", "unexpected model repository")
    init = read_json(args.initialization_receipt)
    require(init.get("schema_version") == "production_polytonic_td_init_verification_v1", "initialization schema drift")
    require(init.get("status") == "passed", "initialization evidence not passed")
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for iteration, attempt, branch in CHECKPOINTS:
        root = args.run_root / "checkpoint_evaluations" / label(iteration) / f"attempt_{attempt}"
        export_path = root / "export" / "checkpoint_eval_export_receipt.json"
        export, hf_root, tokenizer_sha = validate_export(export_path, iteration)
        greek_path = root / "exact_checkpoint_native_greekmmlu_receipt.json"
        greek = validate_greekmmlu(greek_path, iteration, export_path)
        card_path = output / "cards" / f"{branch}.md"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(card(branch=branch, iteration=iteration, greekmmlu=greek), encoding="utf-8")
        receipt_dir = output / "receipts" / branch
        release_root = output / "staged" / branch
        contract = {
            "schema_version": "apertus_model_checkpoint_release_contract_v1",
            "scale": "8b",
            "sources": {"hf_root": str(hf_root), "model_card": str(card_path), "evidence": [str(args.initialization_receipt.resolve()), str(export_path.resolve()), str(greek_path.resolve())]},
            "parent": PARENT,
            "tokenizer": {**TOKENIZER, "sha256": tokenizer_sha},
            "geometry": GEOMETRY,
            "repository": {"repo_id": args.model_repo, "private": True, "revision": branch, "workers": args.workers},
            "release_root": str(release_root),
            "receipts": {"stage": str(receipt_dir / "stage.json"), "freeze": str(receipt_dir / "freeze.json"), "upload": str(receipt_dir / "upload.json"), "inspection": str(receipt_dir / "inspection.json")},
            "required_evidence_schemas": ["production_polytonic_td_init_verification_v1", "native_greekmmlu_exact_checkpoint_export_v1", "exact_checkpoint_native_greekmmlu_receipt_v1"],
        }
        contract_path = output / "contracts" / f"{branch}.json"
        write_json(contract_path, contract)
        rows.append({"branch": branch, "iteration": iteration, "attempt": attempt, "token_slots": iteration * TOKENS_PER_UPDATE, "export": bounded_file(export_path), "greekmmlu": bounded_file(greek_path), "contract": {"path": str(contract_path.resolve()), "sha256": sha256_file(contract_path)}, "export_tree_manifest_sha256": export["hf_export"]["tree_manifest_sha256"], "exported_tokenizer_json_sha256": tokenizer_sha})
    result = {"schema_version": "apertus_full8_private_branch_release_plan_v1", "status": "ready_for_private_xfer_release", "model_repo": args.model_repo, "visibility": "private", "initialization": bounded_file(args.initialization_receipt), "branches": rows, "public_promotion": {"allowed": False, "requires": ["complete_18_checkpoint_native_greek_matrix", "final_metadata_assembly", "separate_explicit_visibility_decision"]}}
    write_json(output / "private_release_plan.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--initialization-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-repo", default="fffoivos/apertus-8b-greek-cpt")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    require(1 <= args.workers <= 32, "workers must be in [1,32]")
    result = prepare(args)
    print(json.dumps({"ok": True, "branches": len(result["branches"]), "output": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
