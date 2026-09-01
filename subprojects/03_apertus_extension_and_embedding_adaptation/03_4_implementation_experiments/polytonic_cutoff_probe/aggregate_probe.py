#!/usr/bin/env python3
"""Apply the precommitted +512/+1024 tokenizer selection rule."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_json_path(path: Path) -> Path:
    """Accept either a tokenizer directory or its tokenizer.json file."""
    return path / "tokenizer.json" if path.is_dir() else path


def load_metrics(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    global_metrics = data.get("global")
    truncation = data.get("truncation")
    if not isinstance(global_metrics, dict):
        raise SystemExit(f"{path}: missing global metrics")
    if not isinstance(truncation, dict):
        raise SystemExit(f"{path}: missing truncation report")
    if truncation.get("n_docs_truncated") != 0:
        raise SystemExit(f"{path}: refusing candidate-specific truncated evaluation")
    bpb = global_metrics.get("bpb_bits_per_byte")
    if not isinstance(bpb, (float, int)) or bpb <= 0:
        raise SystemExit(f"{path}: missing positive BPB")
    return data


def extract(path: Path) -> dict[str, object]:
    data = load_metrics(path)
    metrics = data["global"]
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "vocab_size": data.get("tokenizer_vocab_size"),
        "n_docs": metrics.get("n_docs"),
        "n_bytes": metrics.get("n_bytes"),
        "n_tokens": metrics.get("n_tokens"),
        "bpb": metrics["bpb_bits_per_byte"],
        "tokens_per_word": metrics.get("tokens_per_word"),
        "chars_per_token": metrics.get("chars_per_token"),
        "strr": (data.get("strr") or {}).get("rate"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ancient", type=Path, required=True)
    parser.add_argument("--baseline-modern", type=Path, required=True)
    parser.add_argument("--cutoff-512-ancient", type=Path, required=True)
    parser.add_argument("--cutoff-512-modern", type=Path, required=True)
    parser.add_argument("--cutoff-1024-ancient", type=Path, required=True)
    parser.add_argument("--cutoff-1024-modern", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-512",
        type=Path,
        required=True,
        help="Candidate directory or explicit tokenizer.json path",
    )
    parser.add_argument(
        "--tokenizer-1024",
        type=Path,
        required=True,
        help="Candidate directory or explicit tokenizer.json path",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--modern-max-regression", type=float, default=0.005)
    parser.add_argument(
        "--min-ancient-relative-gain-for-1024", type=float, default=0.01
    )
    args = parser.parse_args()

    paths = {
        "baseline": {
            "ancient": args.baseline_ancient,
            "modern": args.baseline_modern,
        },
        "cutoff_512": {
            "ancient": args.cutoff_512_ancient,
            "modern": args.cutoff_512_modern,
        },
        "cutoff_1024": {
            "ancient": args.cutoff_1024_ancient,
            "modern": args.cutoff_1024_modern,
        },
    }
    arms = {
        arm: {register: extract(path) for register, path in registers.items()}
        for arm, registers in paths.items()
    }

    base_modern = float(arms["baseline"]["modern"]["bpb"])
    b512_modern = float(arms["cutoff_512"]["modern"]["bpb"])
    b1024_modern = float(arms["cutoff_1024"]["modern"]["bpb"])
    b512_ancient = float(arms["cutoff_512"]["ancient"]["bpb"])
    b1024_ancient = float(arms["cutoff_1024"]["ancient"]["bpb"])

    ratio_512_modern = b512_modern / base_modern
    ratio_1024_modern = b1024_modern / base_modern
    gain_1024_vs_512 = 1.0 - (b1024_ancient / b512_ancient)
    pass_512 = ratio_512_modern <= 1.0 + args.modern_max_regression
    pass_1024 = ratio_1024_modern <= 1.0 + args.modern_max_regression

    selected = None
    reason = None
    if pass_1024 and pass_512 and (
        gain_1024_vs_512 >= args.min_ancient_relative_gain_for_1024
    ):
        selected = "cutoff_1024"
        reason = "1024 passed the modern guard and beat 512 ancient BPB by the required margin"
    elif pass_512:
        selected = "cutoff_512"
        reason = "512 passed the modern guard; 1024 did not earn the precommitted ancient-BPB margin"
    else:
        reason = "512 failed the modern-regression guard; no production tokenizer selected"

    tokenizers = {
        "cutoff_512": {
            "path": str(args.tokenizer_512),
            "sha256": sha256_path(tokenizer_json_path(args.tokenizer_512)),
        },
        "cutoff_1024": {
            "path": str(args.tokenizer_1024),
            "sha256": sha256_path(tokenizer_json_path(args.tokenizer_1024)),
        },
    }
    report = {
        "schema_version": "polytonic-cutoff-model-probe-v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "passed" if selected else "failed",
        "selected": selected,
        "reason": reason,
        "precommitted_rule": {
            "modern_bpb_max_relative_regression": args.modern_max_regression,
            "choose_1024_only_if_ancient_bpb_relative_gain_over_512_at_least": (
                args.min_ancient_relative_gain_for_1024
            ),
            "fallback": "512 if it passes the modern guard; otherwise select none",
            "truncation": "zero truncated documents required in every arm",
        },
        "derived": {
            "cutoff_512_modern_bpb_ratio_to_baseline": ratio_512_modern,
            "cutoff_1024_modern_bpb_ratio_to_baseline": ratio_1024_modern,
            "cutoff_1024_ancient_bpb_relative_gain_over_512": gain_1024_vs_512,
            "cutoff_512_passes_modern_guard": pass_512,
            "cutoff_1024_passes_modern_guard": pass_1024,
        },
        "arms": arms,
        "tokenizers": tokenizers,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if selected else 3


if __name__ == "__main__":
    raise SystemExit(main())
