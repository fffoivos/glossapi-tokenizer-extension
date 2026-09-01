#!/usr/bin/env python3
"""Run bounded ReTok + Token Distillation training from prepass snippets.

This is an Apertus-specific wrapper around the vendored implementation. It
avoids the high-level upstream entry point because that path appends tokens with
`add_tokens(...)`; our student tokenizer is already merge-extended with fixed
IDs. The wrapper loads the exact ReTok HF checkpoint, groups real prepass
snippets by selected new token ID, and calls the lower-level `train_embeddings`
loop with an explicit base-phrase -> new-ID mapping.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Iterable


REPO_DIR = Path(__file__).resolve().parent
VENDORED = REPO_DIR / "external" / "token-distillation"
if str(VENDORED) not in sys.path:
    sys.path.insert(0, str(VENDORED))


def load_train_embeddings():
    """Load upstream train_loop.py without executing upstream __init__.py.

    The package __init__ imports the high-level HF-dataset entry point, which
    pulls in optional dependencies we do not need for this local-snippet path.
    Loading train_loop as a private package preserves its relative `.utils`
    import while keeping the adapter dependency surface small.
    """
    package_name = "_apertus_td_vendor"
    package_dir = VENDORED / "token_distillation"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package

    for module_name in ("utils", "train_loop"):
        full_name = f"{package_name}.{module_name}"
        if full_name in sys.modules:
            continue
        module_path = package_dir / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(full_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load vendored module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}.train_loop"].train_embeddings


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: expected JSON object")
            yield row


def read_token_ids(path: Path) -> list[int]:
    ids: list[int] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line))
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_no}: expected integer token ID") from exc
    if not ids:
        raise SystemExit(f"empty token ID list: {path}")
    return ids


def contains_subsequence(seq: list[int], needle: tuple[int, ...]) -> bool:
    if not needle or len(needle) > len(seq):
        return False
    n = len(needle)
    return any(tuple(seq[i : i + n]) == needle for i in range(0, len(seq) - n + 1))


def load_coverage(path: Path, selected_ids: set[int]) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for row in iter_jsonl(path):
        token_id = row.get("new_token_id")
        if not isinstance(token_id, int) or token_id not in selected_ids:
            continue
        base_ids = row.get("base_subtoken_ids")
        if not isinstance(base_ids, list) or not base_ids or not all(isinstance(x, int) for x in base_ids):
            raise SystemExit(f"token {token_id}: missing/invalid base_subtoken_ids")
        if row.get("status") == "mismatch":
            raise SystemExit(f"token {token_id}: refusing TD because coverage status=mismatch")
        rows[token_id] = row
    missing = sorted(selected_ids - set(rows))
    if missing:
        raise SystemExit(f"selected IDs missing from coverage: {missing[:10]} (n={len(missing)})")
    return rows


def load_grouped_snippets(
    snippets_jsonl: Path,
    selected_ids: list[int],
    coverage_rows: dict[int, dict[str, object]],
    base_tokenizer,
    snippets_per_token: int,
    min_accepted_snippets_per_token: int,
    seed: int,
) -> tuple[list[int], list[list[list[int]]], list[list[int]], dict[int, dict[str, int]], dict[int, dict[str, object]]]:
    by_token: dict[int, list[dict[str, object]]] = {token_id: [] for token_id in selected_ids}
    selected_set = set(selected_ids)
    for row in iter_jsonl(snippets_jsonl):
        token_id = row.get("new_token_id")
        if isinstance(token_id, int) and token_id in selected_set:
            by_token[token_id].append(row)

    rng = random.Random(seed)
    trained_token_ids: list[int] = []
    grouped_texts: list[list[list[int]]] = []
    assigned_phrases: list[list[int]] = []
    snippet_stats: dict[int, dict[str, int]] = {}
    skipped_tokens: dict[int, dict[str, object]] = {}

    for token_id in selected_ids:
        phrase = list(coverage_rows[token_id]["base_subtoken_ids"])  # validated in load_coverage
        phrase_key = tuple(phrase)
        candidates = list(by_token[token_id])
        rng.shuffle(candidates)

        accepted: list[list[int]] = []
        rejected_no_phrase = 0
        for snippet in candidates:
            text = snippet.get("snippet_text")
            if not isinstance(text, str) or not text:
                continue
            tokenized = base_tokenizer.encode(text, add_special_tokens=False)
            if not contains_subsequence(tokenized, phrase_key):
                rejected_no_phrase += 1
                continue
            accepted.append(tokenized)
            if len(accepted) >= snippets_per_token:
                break

        if len(accepted) < min_accepted_snippets_per_token:
            skipped_tokens[token_id] = {
                "reason": "insufficient_snippets_with_base_phrase",
                "token_string": coverage_rows[token_id].get("token_string"),
                "raw_token": coverage_rows[token_id].get("raw_token"),
                "base_subtoken_ids": phrase,
                "candidate_snippets": len(candidates),
                "accepted_snippets": len(accepted),
                "rejected_no_phrase": rejected_no_phrase,
                "required_accepted_snippets": min_accepted_snippets_per_token,
                "coverage_status": coverage_rows[token_id].get("status"),
            }
            continue

        trained_token_ids.append(token_id)
        grouped_texts.append(accepted)
        assigned_phrases.append(phrase)
        snippet_stats[token_id] = {
            "candidate_snippets": len(candidates),
            "accepted_snippets": len(accepted),
            "rejected_no_phrase": rejected_no_phrase,
        }

    return trained_token_ids, grouped_texts, assigned_phrases, snippet_stats, skipped_tokens


def copy_tokenizer_files(tokenizer_dir: Path, output_dir: Path) -> None:
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ]:
        src = tokenizer_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--student-tokenizer", type=Path, required=True)
    parser.add_argument("--coverage-jsonl", type=Path, required=True)
    parser.add_argument("--snippets-jsonl", type=Path, required=True)
    parser.add_argument("--token-ids-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-vocab-size", type=int, default=131_072)
    parser.add_argument("--new-id-start", type=int, default=131_072)
    parser.add_argument("--new-id-end", type=int, default=148_480)
    parser.add_argument("--snippets-per-token", type=int, default=25)
    parser.add_argument("--min-accepted-snippets-per-token", type=int, default=None)
    parser.add_argument("--min-trained-token-fraction", type=float, default=0.90)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--target-layer", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-selected-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="build data/manifest but do not load or train the model")
    args = parser.parse_args()

    if args.snippets_per_token <= 0:
        raise SystemExit("--snippets-per-token must be positive")
    if args.min_accepted_snippets_per_token is None:
        args.min_accepted_snippets_per_token = args.snippets_per_token
    if args.min_accepted_snippets_per_token <= 0:
        raise SystemExit("--min-accepted-snippets-per-token must be positive")
    if not (0.0 < args.min_trained_token_fraction <= 1.0):
        raise SystemExit("--min-trained-token-fraction must be in (0, 1]")
    if args.new_id_start != args.base_vocab_size:
        raise SystemExit("this wrapper assumes new-id-start == base-vocab-size")

    t0 = time.time()
    selected_ids = read_token_ids(args.token_ids_file)
    if args.max_selected_tokens is not None:
        selected_ids = selected_ids[: args.max_selected_tokens]
    selected_set = set(selected_ids)
    if len(selected_set) != len(selected_ids):
        raise SystemExit("token ID list contains duplicates")
    out_of_range = [x for x in selected_ids if x < args.new_id_start or x >= args.new_id_end]
    if out_of_range:
        raise SystemExit(f"selected IDs outside new range: {out_of_range[:10]}")

    coverage_rows = load_coverage(args.coverage_jsonl, selected_set)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    train_embeddings = load_train_embeddings()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    print(f"loading tokenizers: base={args.base_tokenizer} student={args.student_tokenizer}")
    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer, trust_remote_code=True)
    student_tok = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)
    if len(student_tok) != args.new_id_end:
        raise SystemExit(f"student tokenizer length {len(student_tok)} != expected {args.new_id_end}")

    trained_token_ids, grouped_texts, assigned_phrases, snippet_stats, skipped_tokens = load_grouped_snippets(
        snippets_jsonl=args.snippets_jsonl,
        selected_ids=selected_ids,
        coverage_rows=coverage_rows,
        base_tokenizer=base_tok,
        snippets_per_token=args.snippets_per_token,
        min_accepted_snippets_per_token=args.min_accepted_snippets_per_token,
        seed=args.seed,
    )

    phrase_to_new_id: dict[tuple[int, ...], int] = {}
    for token_id, phrase in zip(trained_token_ids, assigned_phrases):
        key = tuple(phrase)
        if key in phrase_to_new_id:
            raise SystemExit(f"duplicate base phrase {key} for tokens {phrase_to_new_id[key]} and {token_id}")
        phrase_to_new_id[key] = token_id

    trained_set = set(trained_token_ids)
    preserve_ids = [i for i in range(args.new_id_end) if i not in trained_set]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "student_model": str(args.student_model),
        "base_tokenizer": str(args.base_tokenizer),
        "student_tokenizer": str(args.student_tokenizer),
        "coverage_jsonl": str(args.coverage_jsonl),
        "snippets_jsonl": str(args.snippets_jsonl),
        "token_ids_file": str(args.token_ids_file),
        "output_dir": str(args.output_dir),
        "base_vocab_size": args.base_vocab_size,
        "new_id_range": [args.new_id_start, args.new_id_end],
        "requested_token_count": len(selected_ids),
        "trained_token_count": len(trained_token_ids),
        "skipped_token_count": len(skipped_tokens),
        "trained_token_fraction": len(trained_token_ids) / max(len(selected_ids), 1),
        "selected_token_ids": selected_ids,
        "trained_token_ids": trained_token_ids,
        "skipped_tokens": skipped_tokens,
        "snippets_per_token": args.snippets_per_token,
        "min_accepted_snippets_per_token": args.min_accepted_snippets_per_token,
        "min_trained_token_fraction": args.min_trained_token_fraction,
        "snippet_stats": snippet_stats,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "target_layer": args.target_layer,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": args.device,
        "dry_run": args.dry_run,
        "vendored_token_distillation_commit": "35702b5809599ecd68b7845eca27a0d7b7cec0da",
        "preservation_policy": "all IDs except selected new-token IDs are gradient-zeroed and exact-checked by train_embeddings",
    }
    manifest_path = args.output_dir / "retok_td_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")
    if skipped_tokens:
        print(f"skipped {len(skipped_tokens)} tokens without enough stable base-phrase snippets")
    if manifest["trained_token_fraction"] < args.min_trained_token_fraction:
        raise SystemExit(
            "trained token fraction {got:.3f} is below required {want:.3f}; "
            "inspect retok_td_manifest.json skipped_tokens".format(
                got=manifest["trained_token_fraction"],
                want=args.min_trained_token_fraction,
            )
        )

    if args.dry_run:
        print("dry-run complete; model was not loaded")
        return 0

    print(f"loading student model: {args.student_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    if bool(getattr(model.config, "tie_word_embeddings", False)):
        raise SystemExit("Apertus path expects untied embeddings; got tie_word_embeddings=True")

    model = train_embeddings(
        model=model,
        tokenized_texts=grouped_texts,
        new_phrase_to_new_id=phrase_to_new_id,
        assigned_new_phrases=assigned_phrases,
        tokenizer=student_tok,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss_methods=["MSE-on-hiddens"],
        preserve_original_embeddings=True,
        seed=args.seed,
        original_token_ids=preserve_ids,
        target_layer=args.target_layer,
        mixed_precision=args.dtype in {"bfloat16", "float16"},
        learn_output_with_ce=True,
    )

    print(f"saving TD checkpoint: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    student_tok.save_pretrained(args.output_dir)
    copy_tokenizer_files(args.student_tokenizer, args.output_dir)

    manifest["completed"] = True
    manifest["elapsed_seconds"] = time.time() - t0
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done in {manifest['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
