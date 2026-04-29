#!/usr/bin/env python3
"""Wave-3 tokenizer-cleaning validation orchestrator.

This run intentionally reuses an existing builder-facing dedup metadata
bundle. It does not run dedup. The chain is:

  selected input with dedup overlay -> three mixes -> text splits ->
  fresh GlossAPI tokenizer + fresh HPLT tokenizer + continuous 70/30 tokenizer.

It is resumable by output markers, so the same script can drive both a small
real-doc integration run and the production wave-3 run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(os.environ.get("TOKENIZER_REPO_ROOT", "/home/foivos/Projects/glossapi-tokenizer-extension"))
GLOSSAPI_WORK_ROOT = os.environ.get("GLOSSAPI_WORK_ROOT", "/home/foivos/data/glossapi_work")
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)
APERTUS_BASE_VOCAB = 131072
BASE_TOKENIZER_DIR = os.environ.get(
    "APERTUS_BASE_TOKENIZER_DIR",
    f"{GLOSSAPI_WORK_ROOT}/tokenizer_base_snapshots/apertus_8b_2509_20260415",
)

MIX_CONFIGS = {
    "glossapi_only": PROJECT_ROOT / "subprojects/01_2_training_dataset_mix/examples/glossapi_only_all_non_hplt.json",
    "hplt_only": PROJECT_ROOT / "subprojects/01_2_training_dataset_mix/examples/hplt_only.json",
    "glossapi_plus_hplt_70_30": PROJECT_ROOT / "subprojects/01_2_training_dataset_mix/examples/glossapi_plus_hplt_70_30.json",
}


@dataclass
class Phase:
    name: str
    output_marker: Path
    fn: Callable[[], int]

    def is_done(self) -> bool:
        if not self.output_marker.exists():
            return False
        if self.output_marker.is_file():
            return self.output_marker.stat().st_size > 0
        if self.output_marker.is_dir():
            try:
                return any(self.output_marker.iterdir())
            except OSError:
                return False
        return True


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_cmd(cmd: list[str], log_path: Path, env_extra: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    repo_root = str(PROJECT_ROOT)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_root + (":" + existing_pp if existing_pp else "")
    if env_extra:
        env.update(env_extra)
    print(f"[{now_iso()}] cmd: {' '.join(str(c) for c in cmd)}", flush=True)
    with open(log_path, "ab") as fh:
        fh.write(f"\n[{now_iso()}] CMD: {' '.join(str(c) for c in cmd)}\n".encode("utf-8"))
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=repo_root)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wave-3 mix/split/tokenizer orchestrator using existing dedup metadata.")
    p.add_argument("--input-root", type=Path, required=True, help="Clean canonical corpus root, usually containing data/*.parquet.")
    p.add_argument("--run-root", type=Path, required=True, help="Run output root.")
    p.add_argument("--dedup-metadata-root", type=Path, required=True, help="Existing builder_metadata bundle. Dedup is not rerun.")
    p.add_argument("--train-chars", type=int, default=100_000_000_000)
    p.add_argument("--val-chars", type=int, default=50_000_000)
    p.add_argument("--test-chars", type=int, default=50_000_000)
    p.add_argument("--seed-salt", default="wave3_20260428")
    p.add_argument("--max-workers", type=int, default=64)
    p.add_argument("--row-group-size", type=int, default=2048)
    p.add_argument("--vocab-discovery", type=int, default=50000)
    p.add_argument("--target-extension-units", type=int, default=25600)
    p.add_argument("--skip-identity-check", action="store_true", help="Forwarded only to continuous BPE training.")
    p.add_argument(
        "--mix-build-mode",
        choices=["shared", "single-step"],
        default="shared",
        help="shared writes one dedup-applied selected input; single-step rebuilds each mix directly to reduce disk.",
    )
    p.add_argument(
        "--delete-mixes-after-split",
        action="store_true",
        help="Delete each large mix.parquet after its text splits are exported.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    input_root = args.input_root.resolve()
    dedup_metadata_root = args.dedup_metadata_root.resolve()
    if not dedup_metadata_root.exists():
        raise FileNotFoundError(f"dedup metadata root does not exist: {dedup_metadata_root}")
    if not (dedup_metadata_root / "manifest.json").exists():
        raise FileNotFoundError(f"dedup metadata manifest missing: {dedup_metadata_root / 'manifest.json'}")

    log_dir = run_root / "logs"
    shared = run_root / "shared"
    mixes_root = run_root / "mixes"
    splits_root = run_root / "splits"
    tokenizers_root = run_root / "tokenizers"
    for path in (log_dir, shared, mixes_root, splits_root, tokenizers_root):
        path.mkdir(parents=True, exist_ok=True)

    selected_input = shared / "selected_input.parquet"
    orchestrator_log = log_dir / "orchestrator.log"

    def phase_prepare_selected_input() -> int:
        cmd = [
            PYTHON,
            "-m",
            "glossapi_corpus_cli.cli",
            "mix-prepare-selected-input",
            "--output-root",
            str(input_root),
            "--selected-input-path",
            str(selected_input),
            "--dedup-metadata-root",
            str(dedup_metadata_root),
            "--dedup-action",
            "drop_intra_and_inter",
        ]
        return run_cmd(cmd, log_dir / "prepare_selected_input.log", {"GLOSSAPI_WORK_ROOT": GLOSSAPI_WORK_ROOT})

    def phase_mix(mix_name: str) -> Callable[[], int]:
        def _go() -> int:
            mix_path = mixes_root / mix_name / "mix.parquet"
            if args.mix_build_mode == "single-step":
                cmd = [
                    PYTHON,
                    "-m",
                    "glossapi_corpus_cli.cli",
                    "mix",
                    "--output-root",
                    str(input_root),
                    "--mix-output-path",
                    str(mix_path),
                    "--source-mix-config-path",
                    str(MIX_CONFIGS[mix_name]),
                    "--dedup-metadata-root",
                    str(dedup_metadata_root),
                    "--dedup-action",
                    "drop_intra_and_inter",
                ]
            else:
                cmd = [
                    PYTHON,
                    "-m",
                    "glossapi_corpus_cli.cli",
                    "mix-build-from-selected-input",
                    "--selected-input-path",
                    str(selected_input),
                    "--mix-output-path",
                    str(mix_path),
                    "--source-mix-config-path",
                    str(MIX_CONFIGS[mix_name]),
                    "--standard-split-filters",
                ]
            rc = run_cmd(cmd, log_dir / f"mix_{mix_name}.log", {"GLOSSAPI_WORK_ROOT": GLOSSAPI_WORK_ROOT})
            if rc == 0 and args.delete_mixes_after_split:
                marker = mixes_root / mix_name / "mix.done.json"
                marker.write_text(json.dumps({"completed_at": now_iso(), "mix_path": str(mix_path)}, indent=2) + "\n", encoding="utf-8")
            return rc

        return _go

    def phase_export_splits(mix_name: str) -> Callable[[], int]:
        def _go() -> int:
            cmd = [
                PYTHON,
                str(PROJECT_ROOT / "subprojects/01_2_training_dataset_mix/scripts/export_text_budgeted_splits.py"),
                "--input-root",
                str(mixes_root / mix_name),
                "--output-root",
                str(splits_root / mix_name),
                "--threads",
                str(args.max_workers),
                "--train-chars",
                str(args.train_chars),
                "--val-chars",
                str(args.val_chars),
                "--test-chars",
                str(args.test_chars),
                "--row-group-size",
                str(args.row_group_size),
                "--seed-salt",
                args.seed_salt + "_" + mix_name,
            ]
            rc = run_cmd(cmd, log_dir / f"export_splits_{mix_name}.log", {"GLOSSAPI_WORK_ROOT": GLOSSAPI_WORK_ROOT})
            if rc == 0 and args.delete_mixes_after_split:
                mix_path = mixes_root / mix_name / "mix.parquet"
                if mix_path.exists():
                    mix_path.unlink()
                (mixes_root / mix_name / "mix.deleted_after_split.json").write_text(
                    json.dumps({"deleted_at": now_iso(), "mix_path": str(mix_path)}, indent=2) + "\n",
                    encoding="utf-8",
                )
            return rc

        return _go

    def phase_train_discovery(mix_name: str, arm_name: str) -> Callable[[], int]:
        def _go() -> int:
            cmd = [
                PYTHON,
                str(PROJECT_ROOT / "subprojects/02_1_tokenizer_experiments/scripts/train_discovery_tokenizer.py"),
                "--input-glob",
                str(splits_root / mix_name / "exports" / "train.parquet"),
                "--output-dir",
                str(tokenizers_root / arm_name),
                "--vocab-size",
                str(args.vocab_discovery),
                "--name",
                arm_name,
            ]
            return run_cmd(cmd, log_dir / f"train_{arm_name}.log", {"GLOSSAPI_WORK_ROOT": GLOSSAPI_WORK_ROOT})

        return _go

    def phase_train_continuous() -> int:
        arm_name = "C1_glossapi_plus_hplt_70_30"
        target_vocab_size = APERTUS_BASE_VOCAB + int(args.target_extension_units)
        cmd = [
            PYTHON,
            str(PROJECT_ROOT / "subprojects/02_1_tokenizer_experiments/scripts/train_continuous_bpe_tokenizer.py"),
            "--base-tokenizer-dir",
            BASE_TOKENIZER_DIR,
            "--input-glob",
            str(splits_root / "glossapi_plus_hplt_70_30" / "exports" / "train.parquet"),
            "--output-dir",
            str(tokenizers_root / arm_name),
            "--target-vocab-size",
            str(target_vocab_size),
            "--num-workers",
            str(args.max_workers),
            "--row-group-chunk-size",
            "1",
            "--name",
            arm_name,
        ]
        if args.skip_identity_check:
            cmd.append("--skip-identity-check")
        return run_cmd(cmd, log_dir / f"train_{arm_name}.log", {"GLOSSAPI_WORK_ROOT": GLOSSAPI_WORK_ROOT})

    def phase_done() -> int:
        marker = run_root / "all_done.json"
        payload = {
            "completed_at": now_iso(),
            "run_root": str(run_root),
            "input_root": str(input_root),
            "dedup_metadata_root": str(dedup_metadata_root),
            "selected_input": str(selected_input),
            "mixes": {name: str(mixes_root / name / "mix.parquet") for name in MIX_CONFIGS},
            "splits": {name: str(splits_root / name / "exports") for name in MIX_CONFIGS},
            "tokenizers": {
                "F1_glossapi_only": str(tokenizers_root / "F1_glossapi_only"),
                "F2_hplt_only": str(tokenizers_root / "F2_hplt_only"),
                "C1_glossapi_plus_hplt_70_30": str(tokenizers_root / "C1_glossapi_plus_hplt_70_30"),
            },
            "train_chars": args.train_chars,
            "val_chars": args.val_chars,
            "test_chars": args.test_chars,
            "row_group_size": args.row_group_size,
            "mix_build_mode": args.mix_build_mode,
            "delete_mixes_after_split": args.delete_mixes_after_split,
        }
        marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0

    phases: list[Phase] = []
    if args.mix_build_mode == "shared":
        phases.append(Phase("prepare_selected_input", selected_input, phase_prepare_selected_input))
    for mix_name in MIX_CONFIGS:
        mix_marker = (
            mixes_root / mix_name / "mix.done.json"
            if args.delete_mixes_after_split
            else mixes_root / mix_name / "mix.parquet"
        )
        phases.append(Phase(f"mix_{mix_name}", mix_marker, phase_mix(mix_name)))
        phases.append(
            Phase(
                f"export_splits_{mix_name}",
                splits_root / mix_name / "exports" / "train.parquet",
                phase_export_splits(mix_name),
            )
        )
    phases.extend(
        [
            Phase(
                "train_F1_glossapi_only",
                tokenizers_root / "F1_glossapi_only" / "tokenizer.json",
                phase_train_discovery("glossapi_only", "F1_glossapi_only"),
            ),
            Phase(
                "train_F2_hplt_only",
                tokenizers_root / "F2_hplt_only" / "tokenizer.json",
                phase_train_discovery("hplt_only", "F2_hplt_only"),
            ),
            Phase(
                "train_C1_glossapi_plus_hplt_70_30",
                tokenizers_root / "C1_glossapi_plus_hplt_70_30" / "tokenizer" / "tokenizer.json",
                phase_train_continuous,
            ),
            Phase("done", run_root / "all_done.json", phase_done),
        ]
    )

    started = time.time()
    print(f"[{now_iso()}] wave3 orchestrator START run_root={run_root}", flush=True)
    with open(orchestrator_log, "a", encoding="utf-8") as fh:
        fh.write(f"\n[{now_iso()}] START dedup_metadata_root={dedup_metadata_root}\n")

    for phase in phases:
        if phase.is_done():
            msg = f"[{now_iso()}] SKIP {phase.name} (marker exists: {phase.output_marker})"
            print(msg, flush=True)
            with open(orchestrator_log, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
            continue
        msg = f"[{now_iso()}] RUN {phase.name}"
        print(msg, flush=True)
        with open(orchestrator_log, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
        rc = phase.fn()
        if rc != 0:
            err = f"[{now_iso()}] FAIL {phase.name} rc={rc}; see logs in {log_dir}"
            print(err, flush=True)
            with open(orchestrator_log, "a", encoding="utf-8") as fh:
                fh.write(err + "\n")
            return rc
        ok = f"[{now_iso()}] DONE {phase.name}"
        print(ok, flush=True)
        with open(orchestrator_log, "a", encoding="utf-8") as fh:
            fh.write(ok + "\n")

    elapsed = time.time() - started
    msg = f"[{now_iso()}] ALL DONE in {elapsed:.0f}s"
    print(msg, flush=True)
    with open(orchestrator_log, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
