#!/usr/bin/env python3
"""Verify the 1.5B layer-6 TD initialization and prospective intrinsic gates."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import (
    executing_code_bundle,
    file_binding,
    require,
    require_file_binding,
    require_receipt,
    require_relative_inventory,
    sha256_file,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--parent-materialization-receipt", type=Path, required=True)
    parser.add_argument("--retok-reference", type=Path, required=True)
    parser.add_argument("--retok-reference-receipt", type=Path, required=True)
    parser.add_argument("--td-model", type=Path, required=True)
    parser.add_argument("--td-manifest", type=Path, required=True)
    parser.add_argument("--td-training-inputs-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-receipt", type=Path, required=True)
    parser.add_argument("--acceptance-policy", type=Path, required=True)
    parser.add_argument("--policy-authorization", type=Path, required=True)
    parser.add_argument("--reference-objective-probe", type=Path, required=True)
    parser.add_argument("--td-objective-probe", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable TD verification exists: {args.output}")
    parent_receipt = require_receipt(
        args.parent_materialization_receipt,
        schemas={"apertus_pinned_hf_model_materialization_v1"},
    )
    require(Path(str(parent_receipt.get("output_root", ""))).resolve() == args.parent_model.resolve(), "parent materialization root drift")
    require_relative_inventory(root=args.parent_model, rows=parent_receipt.get("files"))
    reference_receipt = require_receipt(
        args.retok_reference_receipt,
        schemas={"apertus_retok_reference_init_v1"},
    )
    require(Path(str(reference_receipt.get("output_root", ""))).resolve() == args.retok_reference.resolve(), "ReTok reference root drift")
    require_relative_inventory(root=args.retok_reference, rows=reference_receipt.get("output_files"))
    td_inputs = require_receipt(
        args.td_training_inputs_receipt,
        schemas={"apertus_td_training_inputs_v1"},
    )
    tokenizer_receipt = require_receipt(
        args.tokenizer_receipt,
        schemas={"apertus_historical_tokenizer_148480_v1"},
    )
    require_file_binding(td_inputs["coverage_jsonl"])
    require_file_binding(td_inputs["snippets_jsonl"])
    require_file_binding(td_inputs["token_ids"])
    frozen_tokenizer_path = require_file_binding(tokenizer_receipt["files"]["tokenizer.json"])
    require(
        sha256_file(args.td_model / "tokenizer.json") == sha256_file(frozen_tokenizer_path),
        "TD model tokenizer differs from frozen tokenizer receipt",
    )
    import torch
    from transformers import AutoModelForCausalLM

    parent = AutoModelForCausalLM.from_pretrained(args.parent_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    reference = AutoModelForCausalLM.from_pretrained(args.retok_reference, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    td = AutoModelForCausalLM.from_pretrained(args.td_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    require(parent.config.hidden_size == reference.config.hidden_size == td.config.hidden_size == 2048, "1.5B hidden-size drift")
    require(parent.config.num_hidden_layers == reference.config.num_hidden_layers == td.config.num_hidden_layers == 16, "1.5B layer-count drift")
    require(parent.config.tie_word_embeddings is False and reference.config.tie_word_embeddings is False and td.config.tie_word_embeddings is False, "embeddings became tied")

    parent_params = dict(parent.named_parameters())
    ref_params = dict(reference.named_parameters())
    td_params = dict(td.named_parameters())
    require(set(parent_params) == set(ref_params) == set(td_params), "parameter inventory drift")
    parent_input_name = next(name for name, value in parent_params.items() if value is parent.get_input_embeddings().weight)
    parent_output_name = next(name for name, value in parent_params.items() if value is parent.get_output_embeddings().weight)
    ref_input_name = next(name for name, value in ref_params.items() if value is reference.get_input_embeddings().weight)
    ref_output_name = next(name for name, value in ref_params.items() if value is reference.get_output_embeddings().weight)
    td_input_name = next(name for name, value in td_params.items() if value is td.get_input_embeddings().weight)
    td_output_name = next(name for name, value in td_params.items() if value is td.get_output_embeddings().weight)
    require((parent_input_name, parent_output_name) == (ref_input_name, ref_output_name) == (td_input_name, td_output_name), "embedding parameter-name drift")
    embedding_names = {parent_input_name, parent_output_name}
    changed_non_embedding = []
    for name in sorted(parent_params):
        if name in embedding_names:
            continue
        if not torch.equal(parent_params[name].detach().cpu(), ref_params[name].detach().cpu()) or not torch.equal(ref_params[name].detach().cpu(), td_params[name].detach().cpu()):
            changed_non_embedding.append(name)
    require(not changed_non_embedding, f"non-embedding tensors changed: {changed_non_embedding[:10]}")

    manifest = json.loads(args.td_manifest.read_text(encoding="utf-8"))
    require(manifest.get("target_layer") == 6, "1.5B TD target-layer drift")
    require(manifest.get("completed") is True, "TD training did not complete")
    require(manifest.get("epochs") == 1 and float(manifest.get("learning_rate")) == 1e-4, "TD optimization recipe drift")
    require(manifest.get("batch_size") == 8 and manifest.get("seed") == 20260523, "TD batch/seed drift")
    require(manifest.get("dtype") == "bfloat16" and manifest.get("device") == "cuda", "TD execution dtype/device drift")
    require(manifest.get("snippets_per_token") == 25 and manifest.get("min_accepted_snippets_per_token") == 25, "TD snippet policy drift")
    requested = manifest.get("selected_token_ids")
    trained = manifest.get("trained_token_ids")
    skipped = manifest.get("skipped_tokens")
    require(isinstance(requested, list) and len(requested) == 17_392 and len(set(requested)) == len(requested), "requested token ledger drift")
    require(isinstance(trained, list) and isinstance(skipped, dict), "trained/skipped token ledger missing")
    require(all(isinstance(value, int) for value in trained) and len(trained) == len(set(trained)), "trained token ledger drift")
    trained_set = set(trained)
    skipped_set = {int(value) for value in skipped}
    require(trained_set.isdisjoint(skipped_set), "trained/skipped token overlap")
    require(trained_set | skipped_set == set(requested), "trained/skipped token accounting does not close")
    trained_fraction = len(trained) / len(requested)
    require(trained_fraction >= 0.99, "trained-token fraction below 0.99")
    snippet_stats = manifest.get("snippet_stats", {})
    require(isinstance(snippet_stats, dict), "snippet statistics missing")
    for token_id in trained:
        row = snippet_stats.get(str(token_id), snippet_stats.get(token_id))
        require(isinstance(row, dict) and int(row.get("accepted_snippets", -1)) >= 25, f"trained token lacks 25 accepted snippets: {token_id}")

    policy = json.loads(args.acceptance_policy.read_text(encoding="utf-8"))
    require(
        policy.get("schema_version") == "apertus_1p5b_td_acceptance_policy_v2"
        and policy.get("status") == "proposal_pending_owner_approval",
        "1.5B TD acceptance policy drift",
    )
    authorization = require_receipt(
        args.policy_authorization,
        schemas={"apertus_1p5b_td_policy_authorization_v1"},
    )
    require(
        authorization.get("policy") == file_binding(args.acceptance_policy)
        and authorization.get("approved_by") == "user"
        and authorization.get("approval_predates_new_td_job") is True,
        "1.5B TD policy is not prospectively owner-authorized",
    )
    reference_probe = require_receipt(
        args.reference_objective_probe,
        schemas={"apertus_1p5b_td_objective_probe_v1"},
        statuses={"completed"},
    )
    td_probe = require_receipt(
        args.td_objective_probe,
        schemas={"apertus_1p5b_td_objective_probe_v1"},
        statuses={"completed"},
    )
    require(
        reference_probe.get("role") == "reference"
        and td_probe.get("role") == "td"
        and reference_probe.get("policy") == td_probe.get("policy") == file_binding(args.acceptance_policy),
        "TD objective-probe role/policy drift",
    )
    reference_selection = reference_probe.get("selection")
    td_selection = td_probe.get("selection")
    require(
        isinstance(reference_selection, dict)
        and reference_selection == td_selection,
        "TD objective-probe selection drift",
    )
    require(
        Path(str(reference_probe.get("model_root", ""))).resolve() == args.retok_reference.resolve()
        and Path(str(td_probe.get("model_root", ""))).resolve() == args.td_model.resolve(),
        "TD objective-probe model root drift",
    )
    objective_policy = policy["objective_probe"]
    initial_metrics = reference_probe.get("metrics")
    final_metrics = td_probe.get("metrics")
    require(isinstance(initial_metrics, dict) and isinstance(final_metrics, dict), "TD objective metrics missing")
    initial_hidden = float(initial_metrics.get("hidden_mse", float("nan")))
    final_hidden = float(final_metrics.get("hidden_mse", float("nan")))
    initial_ce = float(initial_metrics.get("output_ce", float("nan")))
    final_ce = float(final_metrics.get("output_ce", float("nan")))
    import math

    require(all(math.isfinite(value) for value in (initial_hidden, final_hidden, initial_ce, final_ce)), "TD objective metrics are non-finite")
    require(
        final_hidden <= initial_hidden * float(objective_policy["hidden_mse_maximum_final_over_initial"]),
        "TD hidden-state objective did not improve enough",
    )
    require(
        final_ce <= initial_ce + float(objective_policy["output_ce_maximum_absolute_regression"]),
        "TD output-embedding CE objective regressed",
    )
    row_policy = policy["architecture_local_row_safety"]
    matrix_pairs = (
        ("model.embed_tokens.weight", parent.get_input_embeddings().weight, reference.get_input_embeddings().weight, td.get_input_embeddings().weight),
        ("lm_head.weight", parent.get_output_embeddings().weight, reference.get_output_embeddings().weight, td.get_output_embeddings().weight),
    )
    matrix_results = {}
    for key, parent_weight, ref_weight, td_weight in matrix_pairs:
        parent_cpu = parent_weight.detach().cpu()
        ref_cpu = ref_weight.detach().cpu()
        td_cpu = td_weight.detach().cpu()
        require(tuple(parent_cpu.shape) == (131_072, 2048), f"parent embedding shape drift: {key}")
        require(tuple(ref_cpu.shape) == tuple(td_cpu.shape) == (148_480, 2048), f"extended embedding shape drift: {key}")
        require(torch.equal(parent_cpu, ref_cpu[:131_072]) and torch.equal(parent_cpu, td_cpu[:131_072]), f"base rows changed: {key}")
        require(all(torch.equal(ref_cpu[token_id], td_cpu[token_id]) for token_id in skipped_set), f"skipped fallback rows changed: {key}")
        require(all(not torch.equal(ref_cpu[token_id], td_cpu[token_id]) for token_id in trained_set), f"trained row failed to move: {key}")
        ref_added = ref_cpu[trained].float()
        td_added = td_cpu[trained].float()
        ref_norms = torch.linalg.vector_norm(ref_added, dim=1)
        td_norms = torch.linalg.vector_norm(td_added, dim=1)
        require(
            torch.isfinite(ref_added).all().item()
            and torch.isfinite(td_added).all().item()
            and bool((ref_norms > 0).all().item())
            and bool((td_norms > 0).all().item()),
            f"invalid trained appended rows: {key}",
        )
        ratios = td_norms / ref_norms
        absolute_low, absolute_high = map(float, row_policy["trained_row_ratio_absolute_interval"])
        core_low, core_high = map(float, row_policy["trained_row_ratio_core_interval"])
        median_low, median_high = map(float, row_policy["trained_row_ratio_median_interval"])
        inside_fraction = float(((ratios >= core_low) & (ratios <= core_high)).float().mean().item())
        ratio_median = float(ratios.median().item())
        require(bool(((ratios >= absolute_low) & (ratios <= absolute_high)).all().item()), f"architecture-local absolute row-norm safety failed: {key}")
        require(inside_fraction >= float(row_policy["minimum_fraction_in_core_interval"]), f"architecture-local core row-norm fraction failed: {key}")
        require(median_low <= ratio_median <= median_high, f"architecture-local median row-norm safety failed: {key}")
        matrix_results[key] = {
            "base_rows_byte_exact": True,
            "all_appended_rows_finite_nonzero": True,
            "trained_rows_changed": len(trained_set),
            "skipped_merge_chain_fallback_rows_exact": len(skipped_set),
            "architecture_local_reference": "same 1.5B ReTok row",
            "inside_core_ratio_interval_fraction": inside_fraction,
            "required_minimum_inside_core_fraction": float(row_policy["minimum_fraction_in_core_interval"]),
            "trained_row_ratio_median": ratio_median,
            "required_absolute_interval": [absolute_low, absolute_high],
            "required_core_interval": [core_low, core_high],
            "required_median_interval": [median_low, median_high],
        }
    tokenizer_path = args.td_model / "tokenizer.json"
    require(sha256_file(tokenizer_path) == args.tokenizer_sha256, "TD tokenizer bytes drift")
    td_model_files = [
        {"path": str(path.relative_to(args.td_model)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(value for value in args.td_model.rglob("*") if value.is_file())
    ]
    require(bool(td_model_files), "TD model output inventory is empty")
    receipt = {
        "schema_version": "apertus_1p5b_td_initialization_verification_v2",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_layer": 6,
        "target_layer_hidden_state_index": 6,
        "trained_token_count": len(trained),
        "skipped_token_count": len(skipped_set),
        "trained_token_fraction": trained_fraction,
        "base_rows_byte_exact": True,
        "non_embedding_tensors_byte_exact": True,
        "tie_word_embeddings": False,
        "matrix_results": matrix_results,
        "objective_probe_results": {
            "hidden_mse_initial": initial_hidden,
            "hidden_mse_final": final_hidden,
            "hidden_mse_final_over_initial": final_hidden / initial_hidden,
            "output_ce_initial": initial_ce,
            "output_ce_final": final_ce,
            "output_ce_delta": final_ce - initial_ce,
            "selection": reference_selection,
        },
        "td_manifest": file_binding(args.td_manifest),
        "parent_materialization_receipt": file_binding(args.parent_materialization_receipt),
        "retok_reference_receipt": file_binding(args.retok_reference_receipt),
        "td_training_inputs_receipt": file_binding(args.td_training_inputs_receipt),
        "tokenizer_receipt": file_binding(args.tokenizer_receipt),
        "acceptance_policy": file_binding(args.acceptance_policy),
        "policy_authorization": file_binding(args.policy_authorization),
        "reference_objective_probe": file_binding(args.reference_objective_probe),
        "td_objective_probe": file_binding(args.td_objective_probe),
        "parent_config": file_binding(args.parent_model / "config.json"),
        "reference_config": file_binding(args.retok_reference / "config.json"),
        "td_config": file_binding(args.td_model / "config.json"),
        "tokenizer": file_binding(tokenizer_path),
        "td_model_root": str(args.td_model.resolve()),
        "td_model_files": td_model_files,
        "executing_code_bundle": executing_code_bundle(),
    }
    write_json_atomic(args.output, receipt)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
