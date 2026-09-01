#!/usr/bin/env python3
"""Evaluate a deterministic intrinsic Token-Distillation objective probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

from contract_utils import file_binding, require, sha256_file, write_json_atomic


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selected_probe_ids(token_ids: list[int], *, salt: str, count: int) -> list[int]:
    require(0 < count <= len(token_ids), "invalid objective-probe token count")
    ranked = sorted(
        token_ids,
        key=lambda token_id: (
            hashlib.sha256(f"{salt}\0{token_id}".encode()).digest(),
            token_id,
        ),
    )
    return ranked[:count]


def probe_identity(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("reference", "td"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--td-wrapper", type=Path, required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--student-tokenizer", type=Path, required=True)
    parser.add_argument("--coverage-jsonl", type=Path, required=True)
    parser.add_argument("--snippets-jsonl", type=Path, required=True)
    parser.add_argument("--token-ids-file", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable objective receipt exists: {args.output}")
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    require(
        policy.get("schema_version") == "apertus_1p5b_td_acceptance_policy_v2",
        "TD acceptance policy schema drift",
    )
    probe = policy.get("objective_probe")
    require(isinstance(probe, dict), "TD objective-probe policy missing")
    td = load_module(args.td_wrapper.resolve(), "_apertus_h2g_td_wrapper")
    selected = td.read_token_ids(args.token_ids_file)
    probe_ids = selected_probe_ids(
        selected,
        salt=str(probe["salt"]),
        count=int(probe["token_groups"]),
    )
    coverage = td.load_coverage(args.coverage_jsonl, set(probe_ids))

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, trust_remote_code=True)
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student_tokenizer, trust_remote_code=True
    )
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
    grouped, phrases, trained_ids = None, None, None
    (
        trained_ids,
        grouped,
        phrases,
        _snippet_stats,
        skipped,
    ) = td.load_grouped_snippets(
        snippets_jsonl=args.snippets_jsonl,
        selected_ids=probe_ids,
        coverage_rows=coverage,
        base_tokenizer=base_tokenizer,
        snippets_per_token=int(probe["snippets_per_group"]),
        min_accepted_snippets_per_token=int(probe["snippets_per_group"]),
        seed=int(policy["scientific_constants_unchanged"]["seed"]),
    )
    require(not skipped and trained_ids == probe_ids, "objective probe lacks frozen snippets")
    phrase_to_id = {tuple(phrase): token_id for token_id, phrase in zip(trained_ids, phrases)}
    td.load_train_embeddings()
    train_loop = sys.modules["_apertus_td_vendor.train_loop"]
    transformed = train_loop.transform_input_token_format(
        grouped,
        phrase_to_id,
        student_tokenizer.pad_token_id,
        assigned_new_phrases=phrases,
    )
    loader = DataLoader(
        train_loop.TextDataset(transformed),
        batch_size=int(probe["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda rows: train_loop.collate_fn(
            rows, pad_id=student_tokenizer.pad_token_id
        ),
    )
    identity_rows = [
        {
            "token_id": token_id,
            "base_phrase": list(phrase),
            "snippets": group,
        }
        for token_id, phrase, group in zip(trained_ids, phrases, grouped)
    ]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()
    require(model.config.tie_word_embeddings is False, "objective-probe model is tied")
    hidden_sse = 0.0
    hidden_count = 0
    ce_sum = 0.0
    ce_count = 0
    target_layer = int(probe["target_layer_hidden_state_index"])
    pad_id = int(student_tokenizer.pad_token_id)
    with torch.inference_mode():
        for batch in loader:
            merged = batch["merged_seq"].to("cuda")
            original = batch["original_seq"].to("cuda")
            alignment = batch["unmerged_to_merged_mask"].to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student = model(merged, output_hidden_states=True, use_cache=False)
                teacher = model(original, output_hidden_states=True, use_cache=False)
            student_hidden = student.hidden_states[target_layer].float()[merged != pad_id]
            teacher_hidden = teacher.hidden_states[target_layer].float()[alignment == 1]
            require(
                student_hidden.shape == teacher_hidden.shape,
                "objective-probe hidden alignment drift",
            )
            delta = student_hidden - teacher_hidden
            hidden_sse += float(torch.sum(delta * delta).item())
            hidden_count += int(delta.numel())
            logits = student.logits[:, :-1].float()
            targets = merged[:, 1:]
            ce_sum += float(
                torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=pad_id,
                    reduction="sum",
                ).item()
            )
            ce_count += int((targets != pad_id).sum().item())
    hidden_mse = hidden_sse / hidden_count
    output_ce = ce_sum / ce_count
    require(
        hidden_count > 0
        and ce_count > 0
        and math.isfinite(hidden_mse)
        and math.isfinite(output_ce),
        "objective probe produced invalid losses",
    )
    receipt = {
        "schema_version": "apertus_1p5b_td_objective_probe_v1",
        "status": "completed",
        "role": args.role,
        "policy": file_binding(args.policy),
        "model_root": str(args.model.resolve()),
        "model_config": file_binding(args.model / "config.json"),
        "inputs": {
            "td_wrapper": file_binding(args.td_wrapper),
            "coverage_jsonl": file_binding(args.coverage_jsonl),
            "snippets_jsonl": file_binding(args.snippets_jsonl),
            "token_ids_file": file_binding(args.token_ids_file),
        },
        "selection": {
            "salt": probe["salt"],
            "token_groups": len(trained_ids),
            "snippets_per_group": int(probe["snippets_per_group"]),
            "selected_token_ids": trained_ids,
            "probe_identity_sha256": probe_identity(identity_rows),
        },
        "metrics": {
            "hidden_mse": hidden_mse,
            "hidden_scalar_count": hidden_count,
            "output_ce": output_ce,
            "output_target_count": ce_count,
        },
        "model_tokenizer_sha256": sha256_file(args.model / "tokenizer.json"),
    }
    write_json_atomic(args.output, receipt)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
