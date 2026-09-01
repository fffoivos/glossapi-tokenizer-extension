#!/usr/bin/env python3
"""Emit per-document NLL/BPB rows for one frozen JSONL validation panel.

The metric is deliberately document-local: a BOS token (or EOS fallback) is
prepended, no context crosses document boundaries, and only targets belonging
to the UTF-8 document text are scored. Long documents are split into
non-overlapping target windows while retaining the immediately preceding token
as context for the first target in each window.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-targets-per-forward", type=int, default=4096)
    parser.add_argument("--added-token-start", type=int, default=131072)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("refusing to overwrite per-document validation outputs")
    if args.max_targets_per_forward <= 0:
        raise ValueError("max targets per forward must be positive")

    process_started_at = dt.datetime.now(dt.timezone.utc)
    process_started = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    scoring_started = time.perf_counter()
    context_id = tokenizer.bos_token_id
    if context_id is None:
        context_id = tokenizer.eos_token_id
    if context_id is None:
        raise ValueError("tokenizer has neither BOS nor EOS for document-local scoring")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    documents = 0
    total_targets = 0
    total_bytes = 0
    total_nll = 0.0
    total_base_targets = 0
    total_base_nll = 0.0
    total_added_targets = 0
    total_added_nll = 0.0
    with args.input.open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as output:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row["text"])
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            stream = [int(context_id), *map(int, token_ids)]
            doc_nll = 0.0
            doc_base_nll = 0.0
            doc_added_nll = 0.0
            doc_base_targets = 0
            doc_added_targets = 0
            with torch.inference_mode():
                for start in range(0, len(token_ids), args.max_targets_per_forward):
                    end = min(len(token_ids), start + args.max_targets_per_forward)
                    inputs = torch.tensor(
                        stream[start:end], device=args.device, dtype=torch.long
                    ).unsqueeze(0)
                    labels = torch.tensor(
                        stream[start + 1 : end + 1], device=args.device, dtype=torch.long
                    )
                    logits = model(input_ids=inputs, use_cache=False).logits[0]
                    losses = F.cross_entropy(logits.float(), labels, reduction="none")
                    base = labels < args.added_token_start
                    added = ~base
                    doc_nll += float(losses.sum().item())
                    doc_base_nll += float(losses[base].sum().item())
                    doc_added_nll += float(losses[added].sum().item())
                    doc_base_targets += int(base.sum().item())
                    doc_added_targets += int(added.sum().item())
            utf8_bytes = len(text.encode("utf-8"))
            targets = len(token_ids)
            doc_id = row.get("doc_id") or row.get("cluster_id") or f"line:{line_number}"
            result = {
                "doc_id": str(doc_id),
                "cluster_id": row.get("cluster_id"),
                "source_dataset": row.get("source_dataset") or row.get("source_id"),
                "nll_numerator_nats": doc_nll,
                "target_tokens": targets,
                "utf8_bytes": utf8_bytes,
                "mean_nll": doc_nll / targets,
                "bpb": doc_nll / (math.log(2) * utf8_bytes) if utf8_bytes else None,
                "base_target_nll_numerator_nats": doc_base_nll,
                "base_target_tokens": doc_base_targets,
                "added_target_nll_numerator_nats": doc_added_nll,
                "added_target_tokens": doc_added_targets,
            }
            output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            documents += 1
            total_targets += targets
            total_bytes += utf8_bytes
            total_nll += doc_nll
            total_base_targets += doc_base_targets
            total_base_nll += doc_base_nll
            total_added_targets += doc_added_targets
            total_added_nll += doc_added_nll
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, args.output)
    scoring_seconds = time.perf_counter() - scoring_started
    elapsed_seconds = time.perf_counter() - process_started
    receipt = {
        "schema_version": "apertus_per_document_validation_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_at": process_started_at.isoformat(),
        "elapsed_seconds_total": elapsed_seconds,
        "scoring_seconds": scoring_seconds,
        "target_tokens_per_scoring_second": total_targets / scoring_seconds,
        "metric_contract": "document_local_bos_context_no_cross_document_context_text_targets_only",
        "model": args.model,
        "tokenizer": args.tokenizer,
        "dtype": args.dtype,
        "input": {
            "path": str(args.input.resolve()),
            "bytes": args.input.stat().st_size,
            "sha256": sha256_file(args.input),
        },
        "output": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
            "rows": documents,
        },
        "aggregate": {
            "documents": documents,
            "target_tokens": total_targets,
            "utf8_bytes": total_bytes,
            "nll_numerator_nats": total_nll,
            "mean_nll": total_nll / total_targets,
            "bpb": total_nll / (math.log(2) * total_bytes),
            "base_target_tokens": total_base_targets,
            "base_target_nll_numerator_nats": total_base_nll,
            "added_target_tokens": total_added_targets,
            "added_target_nll_numerator_nats": total_added_nll,
        },
    }
    tmp_receipt = Path(str(args.receipt) + ".partial")
    tmp_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_receipt, args.receipt)
    print(json.dumps({"ok": True, "documents": documents, "tokens": total_targets}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
