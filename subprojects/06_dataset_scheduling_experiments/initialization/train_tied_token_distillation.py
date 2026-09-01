#!/usr/bin/env python3
"""Run canonical Token Distillation on the Mini model's shared embedding table.

This is the tied-head counterpart of the existing Apertus-8B adapter.  It uses
the vendored upstream lower-level training loop so fixed BPE IDs are retained,
trains only selected new rows, and intentionally disables the separate
output-embedding CE step because input and output are one tensor.  The small
initialization bakeoff can compare canonical MSE-only Token Distillation with
the upstream MSE + automatically weighted next-token-prediction safeguard for
tied embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tokenizer_geometry import derive_added_token_base_ids


BASE_VOCAB_SIZE = 131_072
TARGET_VOCAB_SIZE = 148_992
PAD_TOKEN_ID = 10
UPSTREAM_COMMIT = "35702b5809599ecd68b7845eca27a0d7b7cec0da"
ADAPTER_SHA256 = "c9d417bf8f28aaa9dc079c05641bdeae84b86df039b0ce3066f9e96955104462"
TRAIN_LOOP_SHA256 = "aa7128f8025ddde091460bdda2b65e927c98d5683b29d5bb2c573c7717263430"
UTILS_SHA256 = "82f96caa6e2e527e7d3881ee72e65c13a257e42b6a20fdc67d673dab06641672"


def load_existing_adapter(adapter_path: Path):
    vendor = adapter_path.parent / "external" / "token-distillation" / "token_distillation"
    expected = {
        adapter_path: ADAPTER_SHA256,
        vendor / "train_loop.py": TRAIN_LOOP_SHA256,
        vendor / "utils.py": UTILS_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"canonical Token Distillation dependency drift: {path}")
    spec = importlib.util.spec_from_file_location("_apertus_8b_td_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load existing adapter: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def norm_summary(values) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {
        "min": values.min().item(),
        "p50": values.quantile(0.50).item(),
        "p95": values.quantile(0.95).item(),
        "p99": values.quantile(0.99).item(),
        "p999": values.quantile(0.999).item(),
        "max": values.max().item(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retok-model", type=Path, required=True)
    parser.add_argument("--canonical-adapter", type=Path, required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--target-tokenizer", type=Path, required=True)
    parser.add_argument("--coverage-jsonl", type=Path, required=True)
    parser.add_argument("--snippets-jsonl", type=Path, required=True)
    parser.add_argument("--token-ids-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    parser.add_argument("--target-layer", type=int, required=True)
    parser.add_argument(
        "--loss-profile",
        choices=("mse", "mse_ce_auto"),
        default="mse_ce_auto",
        help=(
            "mse is canonical hidden-state TD; mse_ce_auto adds the canonical "
            "automatically weighted NTP term through the same tied table"
        ),
    )
    parser.add_argument("--snippets-per-token", type=int, default=25)
    parser.add_argument("--min-accepted-snippets-per-token", type=int)
    parser.add_argument("--min-trained-token-fraction", type=float)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-to-base-p999-ratio", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.target_layer not in {-1, 7}:
        raise SystemExit("initialization pilot is restricted to target layers 7 and -1")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    if args.snippets_per_token <= 0:
        raise SystemExit("--snippets-per-token must be positive")
    if args.min_accepted_snippets_per_token is None:
        args.min_accepted_snippets_per_token = args.snippets_per_token
    if args.min_accepted_snippets_per_token <= 0:
        raise SystemExit("--min-accepted-snippets-per-token must be positive")
    if args.min_trained_token_fraction is None:
        args.min_trained_token_fraction = 1.0 if args.scope == "pilot" else 0.90
    if not 0.0 < args.min_trained_token_fraction <= 1.0:
        raise SystemExit("--min-trained-token-fraction must be in (0, 1]")
    loss_methods = ["MSE-on-hiddens"]
    if args.loss_profile == "mse_ce_auto":
        loss_methods.append("CE-auto-weighted")

    adapter = load_existing_adapter(args.canonical_adapter)
    selected_ids = adapter.read_token_ids(args.token_ids_file)
    if len(set(selected_ids)) != len(selected_ids):
        raise SystemExit("token ID list contains duplicates")
    if any(token_id < BASE_VOCAB_SIZE or token_id >= TARGET_VOCAB_SIZE for token_id in selected_ids):
        raise SystemExit("token IDs must be inside 131072..148991")
    if args.scope == "full" and selected_ids != list(range(BASE_VOCAB_SIZE, TARGET_VOCAB_SIZE)):
        raise SystemExit("full scope requires the complete ordered range 131072..148991")
    if args.scope == "pilot" and len(selected_ids) > 1024:
        raise SystemExit("pilot scope is capped at 1024 selected token IDs")
    coverage = adapter.load_coverage(args.coverage_jsonl, set(selected_ids))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_tokenizer, local_files_only=True
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        args.target_tokenizer, local_files_only=True
    )
    if len(target_tokenizer) != TARGET_VOCAB_SIZE:
        raise SystemExit("target tokenizer vocabulary size drift")
    if target_tokenizer.pad_token != "<pad>" or target_tokenizer.pad_token_id != PAD_TOKEN_ID:
        raise SystemExit("target tokenizer must use <pad> at ID 10")
    exact_base_ids = derive_added_token_base_ids(
        args.target_tokenizer / "tokenizer.json",
        base_vocab_size=BASE_VOCAB_SIZE,
        target_vocab_size=TARGET_VOCAB_SIZE,
    )
    for token_id in selected_ids:
        expected = exact_base_ids[token_id]
        observed = [int(value) for value in coverage[token_id]["base_subtoken_ids"]]
        if observed != expected:
            raise SystemExit(
                f"token {token_id}: stale/incompatible coverage decomposition; "
                f"observed={observed} expected={expected}"
            )
    trained_ids, grouped, phrases, snippet_stats, skipped = adapter.load_grouped_snippets(
        snippets_jsonl=args.snippets_jsonl,
        selected_ids=selected_ids,
        coverage_rows=coverage,
        base_tokenizer=base_tokenizer,
        snippets_per_token=args.snippets_per_token,
        min_accepted_snippets_per_token=args.min_accepted_snippets_per_token,
        seed=args.seed,
    )
    trained_fraction = len(trained_ids) / len(selected_ids)
    if trained_fraction < args.min_trained_token_fraction:
        raise SystemExit(
            f"trained token fraction {trained_fraction:.6f} is below "
            f"{args.min_trained_token_fraction:.6f}"
        )
    phrase_to_id = {tuple(phrase): token_id for phrase, token_id in zip(phrases, trained_ids)}
    if len(phrase_to_id) != len(trained_ids):
        raise SystemExit("duplicate base-token decomposition across selected new tokens")
    trained_set = set(trained_ids)
    preserve_ids = [
        token_id for token_id in range(TARGET_VOCAB_SIZE) if token_id not in trained_set
    ]

    args.output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "apertus_mini_tied_token_distillation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if args.dry_run else "running",
        "scope": args.scope,
        "retok_model": str(args.retok_model.resolve()),
        "canonical_adapter": str(args.canonical_adapter.resolve()),
        "canonical_adapter_sha256": sha256_file(args.canonical_adapter),
        "base_tokenizer": str(args.base_tokenizer.resolve()),
        "target_tokenizer": str(args.target_tokenizer.resolve()),
        "coverage_jsonl": str(args.coverage_jsonl.resolve()),
        "coverage_jsonl_sha256": sha256_file(args.coverage_jsonl),
        "snippets_jsonl": str(args.snippets_jsonl.resolve()),
        "token_ids_file": str(args.token_ids_file.resolve()),
        "token_ids_file_sha256": sha256_file(args.token_ids_file),
        "target_tokenizer_json_sha256": sha256_file(
            args.target_tokenizer / "tokenizer.json"
        ),
        "retok_manifest_sha256": sha256_file(
            args.retok_model / "tied_retok_manifest.json"
        ),
        "requested_token_count": len(selected_ids),
        "trained_token_count": len(trained_ids),
        "trained_token_fraction": trained_fraction,
        "skipped_tokens": skipped,
        "snippet_stats": snippet_stats,
        "target_layer": args.target_layer,
        "snippets_per_token": args.snippets_per_token,
        "min_accepted_snippets_per_token": args.min_accepted_snippets_per_token,
        "min_trained_token_fraction_required": args.min_trained_token_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_profile": args.loss_profile,
        "loss_methods": loss_methods,
        "learn_output_with_ce": False,
        "tied_embedding_policy": (
            "one shared table; no separate output-only update; any CE-auto-weighted "
            "term backpropagates through the shared input/output table"
        ),
        "canonical_token_distillation_commit": UPSTREAM_COMMIT,
        "preservation_policy": (
            "all non-trained rows, including low-coverage requested rows, are "
            "gradient-zeroed and exact-checked; they retain FVT initialization"
        ),
        "max_new_to_base_p999_ratio": args.max_new_to_base_p999_ratio,
    }
    manifest_path = args.output_dir / "tied_td_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    import torch

    train_embeddings = adapter.load_train_embeddings()
    model = AutoModelForCausalLM.from_pretrained(
        args.retok_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(args.device)
    if not bool(getattr(model.config, "tie_word_embeddings", False)):
        raise SystemExit("expected tie_word_embeddings=true")
    if model.config.pad_token_id != PAD_TOKEN_ID:
        raise SystemExit("ReTok model must reconcile pad_token_id to 10")
    if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise SystemExit("input/output embeddings do not share storage before TD")

    model = train_embeddings(
        model=model,
        tokenized_texts=grouped,
        new_phrase_to_new_id=phrase_to_id,
        assigned_new_phrases=phrases,
        tokenizer=target_tokenizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss_methods=loss_methods,
        preserve_original_embeddings=True,
        seed=args.seed,
        original_token_ids=preserve_ids,
        target_layer=args.target_layer,
        mixed_precision=True,
        learn_output_with_ce=False,
    )
    model.tie_weights()
    weight = model.get_input_embeddings().weight
    if weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise SystemExit("input/output embeddings do not share storage after TD")
    if not torch.isfinite(weight).all():
        raise SystemExit("non-finite tied embeddings after TD")
    norms = weight.detach().float().norm(dim=1)
    base_norms = norm_summary(norms[:BASE_VOCAB_SIZE])
    new_norms = norm_summary(norms[BASE_VOCAB_SIZE:])
    ratio = new_norms["max"] / base_norms["p999"]
    if ratio > args.max_new_to_base_p999_ratio:
        raise SystemExit(
            f"tied-embedding norm-collapse gate failed: max_new/base_p999={ratio:.6f}"
        )

    model.save_pretrained(args.output_dir, safe_serialization=True)
    target_tokenizer.save_pretrained(args.output_dir)
    adapter.copy_tokenizer_files(args.target_tokenizer, args.output_dir)
    overlay_manifest = args.target_tokenizer / "overlay_manifest.json"
    if overlay_manifest.is_file():
        import shutil

        shutil.copy2(overlay_manifest, args.output_dir / overlay_manifest.name)
    manifest.update(
        {
            "status": "completed",
            "input_output_share_storage": True,
            "base_norms": base_norms,
            "new_norms": new_norms,
            "max_new_to_base_p999_ratio_observed": ratio,
            "norm_collapse_gate_passed": True,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
