#!/usr/bin/env python3
"""Build train/eval splits for the Ancient/Polytonic Greek tokenizer arm.

The split is deterministic and text-hash based: duplicate exact texts are
assigned to the same split, so exact train/eval leakage cannot occur unless
the text normalization policy changes.

This script intentionally keeps per-row text inspection light. The corpus is
only ~19k rows but ~515M characters; full Unicode per-character analysis here
would turn split construction into a training-sized job. We reuse the upstream
orthography columns (`polytonic_ratio`, `greek_percentage`, `len_greek`) for
stratification and compute only cheap hygiene fields directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata as ud
from pathlib import Path

import pandas as pd

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_for_hash(text: str) -> str:
    return " ".join(ud.normalize("NFC", text or "").split())


def text_metrics(text: str) -> dict[str, int | float | bool]:
    text = text or ""
    chars = len(text)
    control_chars = len(CONTROL_RE.findall(text))
    return {
        "text_chars": chars,
        "utf8_bytes": len(text.encode("utf-8")),
        "control_char_count": control_chars,
        "control_char_ratio": (control_chars / chars) if chars else 0.0,
        "looks_like_rtf": text.lstrip().startswith("{\\rtf") or "\\fonttbl" in text[:10_000],
    }


def split_for_hash(hex_digest: str, val_pct: int, test_pct: int) -> str:
    bucket = int(hex_digest[:16], 16) % 10_000
    if bucket < test_pct * 100:
        return "test"
    if bucket < (test_pct + val_pct) * 100:
        return "val"
    return "train"


def sample_balanced(df: pd.DataFrame, per_source: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    parts = []
    for source in sorted(df["source_dataset"].dropna().unique().tolist()):
        sub = df[df["source_dataset"] == source].sort_values("split_hash")
        parts.append(sub.head(per_source))
    return pd.concat(parts, ignore_index=True) if parts else df.head(0)


def write_frame(df: pd.DataFrame, path: Path, row_group_size: int) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", row_group_size=row_group_size)
    source_counts = (
        df.groupby("source_dataset", dropna=False)
        .size()
        .reset_index(name="len")
        .sort_values("source_dataset")
        .to_dict(orient="records")
        if len(df)
        else []
    )
    return {
        "path": str(path),
        "rows": int(len(df)),
        "text_chars": int(df["text_chars"].sum()) if len(df) else 0,
        "utf8_bytes": int(df["utf8_bytes"].sum()) if len(df) else 0,
        "source_counts": source_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-pct", type=int, default=10)
    parser.add_argument("--test-pct", type=int, default=10)
    parser.add_argument("--balanced-per-source", type=int, default=250)
    parser.add_argument("--row-group-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-control-ratio", type=float, default=0.02)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input_parquet)
    if "text" not in df.columns:
        raise SystemExit("input parquet must contain a text column")
    if "source_dataset" not in df.columns:
        raise SystemExit("input parquet must contain source_dataset")

    metric_rows = [text_metrics(t) for t in df["text"].tolist()]
    metrics = pd.DataFrame(metric_rows)
    hashes = [stable_hash(normalize_for_hash(t)) for t in df["text"].tolist()]
    split_hashes = [stable_hash(h + f":{args.seed}") for h in hashes]
    splits = [split_for_hash(h, args.val_pct, args.test_pct) for h in split_hashes]

    for col in metrics.columns:
        df[col] = metrics[col].values
    df["distinctive_polytonic_word_ratio_signal"] = pd.to_numeric(
        df["polytonic_ratio"] if "polytonic_ratio" in df.columns else 0.0,
        errors="coerce",
    ).fillna(0.0)
    if "len_greek" in df.columns:
        greek_den = pd.to_numeric(df["len_greek"], errors="coerce").fillna(0.0)
        df["distinctive_polytonic_char_ratio_signal"] = (
            pd.to_numeric(df["polytonic_ratio"], errors="coerce").fillna(0.0)
        )
        df["greek_char_count_signal"] = greek_den
    else:
        df["distinctive_polytonic_char_ratio_signal"] = df["distinctive_polytonic_word_ratio_signal"]
        df["greek_char_count_signal"] = 0
    df["text_hash_nfc_ws"] = hashes
    df["split_hash"] = split_hashes
    df["split"] = splits

    before = len(df)
    if "is_empty" not in df.columns:
        df["is_empty"] = False
    df = df[
        df["text"].notna()
        & (df["text_chars"] >= args.min_chars)
        & (~df["is_empty"].fillna(False).astype(bool))
        & (~df["looks_like_rtf"].astype(bool))
        & (df["control_char_ratio"] <= args.max_control_ratio)
    ].copy()
    after = len(df)

    # Ensure duplicate normalized texts cannot cross split boundaries.
    leakage = df.groupby("text_hash_nfc_ws")["split"].nunique()
    leakage = leakage[leakage > 1]
    if len(leakage):
        raise RuntimeError(f"{len(leakage)} normalized text hashes span multiple splits")

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    val_bal = sample_balanced(val, args.balanced_per_source, args.seed)
    test_bal = sample_balanced(test, args.balanced_per_source, args.seed)
    high_poly = test[test["distinctive_polytonic_word_ratio_signal"] >= 0.75].copy()
    underaccented = test[test["distinctive_polytonic_word_ratio_signal"] < 0.10].copy()

    outputs: dict[str, object] = {
        "poly_train": write_frame(train, args.output_dir / "poly_train.parquet", args.row_group_size),
        "poly_val": write_frame(val, args.output_dir / "poly_val.parquet", args.row_group_size),
        "poly_test": write_frame(test, args.output_dir / "poly_test.parquet", args.row_group_size),
        "poly_val_balanced": write_frame(val_bal, args.output_dir / "poly_val_balanced.parquet", args.row_group_size),
        "poly_test_balanced": write_frame(test_bal, args.output_dir / "poly_test_balanced.parquet", args.row_group_size),
        "poly_high_diacritic_test": write_frame(high_poly, args.output_dir / "poly_high_diacritic_test.parquet", args.row_group_size),
        "poly_underaccented_test": write_frame(underaccented, args.output_dir / "poly_underaccented_test.parquet", args.row_group_size),
    }

    by_source = {}
    source_dir = args.output_dir / "by_source"
    for source in sorted(test["source_dataset"].dropna().unique().tolist()):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_") or "source"
        by_source[f"poly_{safe}_test"] = write_frame(
            test[test["source_dataset"] == source].copy(),
            source_dir / f"{safe}_test.parquet",
            args.row_group_size,
        )
    outputs["by_source"] = by_source

    manifest = {
        "input_parquet": str(args.input_parquet),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "split_policy": {
            "hash_normalization": "NFC + whitespace collapse",
            "val_pct": args.val_pct,
            "test_pct": args.test_pct,
            "train_pct": 100 - args.val_pct - args.test_pct,
            "balanced_per_source": args.balanced_per_source,
            "row_group_size": args.row_group_size,
        },
        "hygiene": {
            "rows_before": before,
            "rows_after": after,
            "rows_dropped": before - after,
            "min_chars": args.min_chars,
            "max_control_ratio": args.max_control_ratio,
            "drop_rtf_like": True,
        },
        "outputs": outputs,
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
