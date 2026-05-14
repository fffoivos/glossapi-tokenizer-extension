#!/usr/bin/env python3
"""Wave-2 production pipeline orchestrator.

Drives the chain: re-clean → dedup → mix-prepare → mix-build × 2 →
export-splits × 2 → train tokenizer × 4. Each phase is resumable —
re-running this script picks up at the first phase whose output
marker is missing or stale.

Phase markers:

  re-clean       canonical/data/_done.marker       (touched at end)
  dedup          dedup_run/progress/stage_02_*.json status=completed
                 (or stage_01 if --exact-only)
  mix-prepare    shared/selected_input.parquet     (non-empty file)
  mix-build      mixes/<name>/mix.parquet          (non-empty file)
  export-splits  splits/<name>/exports/train.parquet (non-empty)
  train          tokenizers/<arm>/tokenizer.json   (non-empty)
  done           all_done.json                     (final summary)

Designed for two invocation modes:
- Production: --mode prod --input-root /…/canonical --run-root /…/wave2_20260426
- Smoke:      --mode smoke --input-root /…/smoke/canonical --run-root /…/smoke

Pass --skip-dedup for a faster smoke (mix-prepare runs with
--dedup-action ignore).

Usage:
  python3 wave2_orchestrate.py \
    --input-root /home/foivos/data/glossapi_work_wave2_20260426/canonical \
    --run-root /home/foivos/runs/wave2_20260426 \
    --mode prod
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional


PROJECT_ROOT = Path(os.environ.get(
    "TOKENIZER_REPO_ROOT", "/home/foivos/Projects/glossapi-tokenizer-extension"
))
GLOSSAPI_WORK_ROOT = os.environ.get("GLOSSAPI_WORK_ROOT", "/home/foivos/data/glossapi_work")
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)
APERTUS_BASE_VOCAB = 131072
BASE_TOKENIZER_DIR = os.environ.get(
    "APERTUS_BASE_TOKENIZER_DIR",
    f"{GLOSSAPI_WORK_ROOT}/tokenizer_base_snapshots/apertus_8b_2509_20260415",
)

MIX_CONFIGS = {
    "glossapi_only": PROJECT_ROOT / "subprojects/01_2_training_dataset_mix/examples/glossapi_only_all_non_hplt.json",
    "glossapi_plus_hplt_70_30": PROJECT_ROOT / "subprojects/01_2_training_dataset_mix/examples/glossapi_plus_hplt_70_30.json",
}


@dataclass
class Phase:
    name: str
    output_marker: Path
    fn: Callable
    skip_when_smoke: bool = False
    skip_when_skip_dedup: bool = False

    def is_done(self) -> bool:
        if not self.output_marker.exists():
            return False
        # Treat zero-byte / empty-dir as not done.
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


def run_cmd(cmd: List[str], log_path: Path, env_extra: Optional[dict] = None) -> int:
    """Run a subprocess with output appended to log_path. Return exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Make sure subprocesses can import `glossapi_corpus_cli` regardless of
    # working directory — prepend the tokenizer-extension repo root to
    # PYTHONPATH.
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True,
                   help="Path containing canonical/data/*.parquet (post-reclean).")
    p.add_argument("--run-root", type=Path, required=True,
                   help="Where to put per-phase outputs (dedup_state, dedup_run, shared, mixes, splits, tokenizers).")
    p.add_argument("--mode", choices=["prod", "smoke"], default="prod")
    p.add_argument("--skip-dedup", action="store_true",
                   help="Smoke shortcut: run mix-prepare with --dedup-action ignore and skip the dedup phase entirely.")
    p.add_argument("--vocab-discovery", type=int, default=50000)
    p.add_argument("--target-extension-units", type=int, default=25600,
                   help="Continuous BPE: number of new units to add over the Apertus base.")
    p.add_argument("--train-chars", type=int, default=None,
                   help="Approx training char budget (mix-export). Defaults: prod=full, smoke=20M.")
    p.add_argument("--val-chars", type=int, default=None)
    p.add_argument("--test-chars", type=int, default=None)
    p.add_argument("--seed-salt", default="wave2_20260426")
    p.add_argument("--max-workers", type=int, default=64)
    p.add_argument("--row-group-size", type=int, default=2048,
                   help="Rows per split-export parquet row group; smaller groups improve continuous-BPE parallelism.")
    args = p.parse_args()

    # Defaults for budget by mode.
    if args.train_chars is None:
        args.train_chars = 100_000_000_000 if args.mode == "prod" else 20_000_000
    if args.val_chars is None:
        args.val_chars = 50_000_000 if args.mode == "prod" else 200_000
    if args.test_chars is None:
        args.test_chars = 50_000_000 if args.mode == "prod" else 200_000

    run_root: Path = args.run_root
    input_root: Path = args.input_root
    run_root.mkdir(parents=True, exist_ok=True)

    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    orchestrator_log = log_dir / "orchestrator.log"

    state_root = run_root / "dedup_state"
    dedup_run = run_root / "dedup_run"
    dedup_metadata = run_root / "dedup_metadata"
    shared = run_root / "shared"
    mixes_root = run_root / "mixes"
    splits_root = run_root / "splits"
    tokenizers_root = run_root / "tokenizers"
    for d in (state_root, dedup_run, dedup_metadata, shared, mixes_root, splits_root, tokenizers_root):
        d.mkdir(parents=True, exist_ok=True)

    # ----- phase functions -----

    def phase_dedup() -> int:
        """Stage 1 + Stage 2 dedup. Resumable via --resume."""
        cmd = [
            PYTHON, "-m", "glossapi_corpus_cli.cli", "dedup-text", "run",
            "--input-root", str(input_root),
            "--state-root", str(state_root),
            "--run-root", str(dedup_run),
            "--max-workers", str(args.max_workers),
            "--greek-diacritic-policy", "preserve",
        ]
        # If state.sqlite already exists, add --resume so dedup picks up where killed.
        if (state_root / "state.sqlite").exists():
            cmd.append("--resume")
        return run_cmd(cmd, log_dir / "dedup.log", env_extra={"GLOSSAPI_WORK_ROOT": GLOSSAPI_WORK_ROOT})

    def phase_dedup_export_metadata() -> int:
        # The dedup CLI's export-builder-metadata writes the bundle into
        # <run_root>/builder_metadata/. We mirror it (via symlink) at
        # `dedup_metadata/` so the rest of the pipeline can consume it
        # at a stable path, and we explicitly invoke the CLI to ensure
        # the bundle is finalized.
        cmd = [
            PYTHON, "-m", "glossapi_corpus_cli.cli", "dedup-text", "export-builder-metadata",
            "--state-root", str(state_root),
            "--run-root", str(dedup_run),
        ]
        rc = run_cmd(cmd, log_dir / "dedup_export.log", env_extra={"GLOSSAPI_WORK_ROOT": GLOSSAPI_WORK_ROOT})
        if rc != 0:
            return rc
        bundle_src = dedup_run / "builder_metadata"
        if not bundle_src.exists():
            return 1
        # Mirror at dedup_metadata/ for downstream mix: replace empty dir
        # with a symlink (or copy if symlink not appropriate). Tolerate
        # pre-existing equivalent symlink on resume.
        if dedup_metadata.is_symlink() or dedup_metadata.exists():
            try:
                if dedup_metadata.is_symlink():
                    if dedup_metadata.resolve() == bundle_src.resolve():
                        return 0
                    dedup_metadata.unlink()
                elif dedup_metadata.is_dir() and not any(dedup_metadata.iterdir()):
                    dedup_metadata.rmdir()
            except OSError:
                pass
        dedup_metadata.symlink_to(bundle_src, target_is_directory=True)
        return 0

    def phase_mix(mix_name: str) -> Callable[[], int]:
        """Single-step `mix` command — does filter + dedup-apply + per-config
        output in one shot. The split into mix-prepare-selected-input +
        mix-build-from-selected-input lives only on local working trees
        (not in the codex branch); use the committed `mix` until that
        landing actually merges."""
        cfg = MIX_CONFIGS[mix_name]
        def _go() -> int:
            cmd = [
                PYTHON, "-m", "glossapi_corpus_cli.cli", "mix",
                "--output-root", str(input_root),
                "--mix-output-path", str(mixes_root / mix_name / "mix.parquet"),
                "--source-mix-config-path", str(cfg),
            ]
            if args.skip_dedup:
                cmd += ["--dedup-action", "ignore"]
            else:
                cmd += [
                    "--dedup-metadata-root", str(dedup_metadata),
                    "--dedup-action", "drop_intra_and_inter",
                ]
            return run_cmd(cmd, log_dir / f"mix_{mix_name}.log",
                           env_extra={"GLOSSAPI_WORK_ROOT": GLOSSAPI_WORK_ROOT})
        return _go

    def phase_export_splits(mix_name: str) -> Callable[[], int]:
        def _go() -> int:
            mix_dir = mixes_root / mix_name
            export_root = splits_root / mix_name
            cmd = [
                PYTHON,
                str(PROJECT_ROOT / "subprojects/01_2_training_dataset_mix/scripts/export_text_budgeted_splits.py"),
                "--input-root", str(mix_dir),
                "--output-root", str(export_root),
                "--threads", str(args.max_workers),
                "--train-chars", str(args.train_chars),
                "--val-chars", str(args.val_chars),
                "--test-chars", str(args.test_chars),
                "--row-group-size", str(args.row_group_size),
                "--seed-salt", args.seed_salt + "_" + mix_name,
            ]
            return run_cmd(cmd, log_dir / f"export_splits_{mix_name}.log",
                           env_extra={"GLOSSAPI_WORK_ROOT": GLOSSAPI_WORK_ROOT})
        return _go

    def phase_train_discovery(mix_name: str) -> Callable[[], int]:
        """F1 (glossapi-only) or F2 (glossapi+hplt) — fresh BPE discovery."""
        def _go() -> int:
            arm_name = "F1_glossapi_only" if mix_name == "glossapi_only" else "F2_glossapi_plus_hplt_70_30"
            export_train = splits_root / mix_name / "exports" / "train.parquet"
            out_dir = tokenizers_root / arm_name
            cmd = [
                PYTHON,
                str(PROJECT_ROOT / "subprojects/02_1_tokenizer_experiments/scripts/train_discovery_tokenizer.py"),
                "--input-glob", str(export_train),
                "--output-dir", str(out_dir),
                "--vocab-size", str(args.vocab_discovery),
                "--name", arm_name,
            ]
            return run_cmd(cmd, log_dir / f"train_{arm_name}.log",
                           env_extra={"GLOSSAPI_WORK_ROOT": GLOSSAPI_WORK_ROOT})
        return _go

    def phase_train_continuous(mix_name: str) -> Callable[[], int]:
        """C1 / C2 — continuous BPE from Apertus base."""
        def _go() -> int:
            arm_name = "C1_glossapi_only" if mix_name == "glossapi_only" else "C2_glossapi_plus_hplt_70_30"
            export_train = splits_root / mix_name / "exports" / "train.parquet"
            out_dir = tokenizers_root / arm_name
            target_vocab_size = APERTUS_BASE_VOCAB + int(args.target_extension_units)
            cmd = [
                PYTHON,
                str(PROJECT_ROOT / "subprojects/02_1_tokenizer_experiments/scripts/train_continuous_bpe_tokenizer.py"),
                "--base-tokenizer-dir", BASE_TOKENIZER_DIR,
                "--input-glob", str(export_train),
                "--output-dir", str(out_dir),
                "--target-vocab-size", str(target_vocab_size),
                "--num-workers", str(args.max_workers),
                "--row-group-chunk-size", "1",
                "--name", arm_name,
            ]
            return run_cmd(cmd, log_dir / f"train_{arm_name}.log",
                           env_extra={"GLOSSAPI_WORK_ROOT": GLOSSAPI_WORK_ROOT})
        return _go

    def phase_done() -> int:
        marker = run_root / "all_done.json"
        payload = {
            "completed_at": now_iso(),
            "mode": args.mode,
            "run_root": str(run_root),
            "input_root": str(input_root),
            "dedup_run": str(dedup_run),
            "selected_input": str(shared / "selected_input.parquet"),
            "mixes": {name: str(mixes_root / name / "mix.parquet") for name in MIX_CONFIGS},
            "splits": {name: str(splits_root / name / "exports") for name in MIX_CONFIGS},
            "tokenizers": {
                "F1_glossapi_only": str(tokenizers_root / "F1_glossapi_only"),
                "F2_glossapi_plus_hplt_70_30": str(tokenizers_root / "F2_glossapi_plus_hplt_70_30"),
                "C1_glossapi_only": str(tokenizers_root / "C1_glossapi_only"),
                "C2_glossapi_plus_hplt_70_30": str(tokenizers_root / "C2_glossapi_plus_hplt_70_30"),
            },
        }
        marker.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0

    # ----- phase list -----

    phases: List[Phase] = []

    # Dedup. Marker = stage_01 OR stage_02 finished. Skip entirely if --skip-dedup.
    dedup_marker = dedup_run / "progress" / "_dedup_complete.marker"
    if not args.skip_dedup:
        phases.append(Phase("dedup", dedup_marker, lambda: _wrap_dedup(phase_dedup, dedup_marker)))
        phases.append(Phase("dedup_export", dedup_metadata / "manifest.json", phase_dedup_export_metadata))

    for mix_name in MIX_CONFIGS:
        phases.append(Phase(
            f"mix_{mix_name}",
            mixes_root / mix_name / "mix.parquet",
            phase_mix(mix_name),
        ))
        phases.append(Phase(
            f"export_splits_{mix_name}",
            splits_root / mix_name / "exports" / "train.parquet",
            phase_export_splits(mix_name),
        ))

    # Tokenizer trainings.
    phases.append(Phase(
        "train_F1_glossapi_only",
        tokenizers_root / "F1_glossapi_only" / "tokenizer.json",
        phase_train_discovery("glossapi_only"),
    ))
    phases.append(Phase(
        "train_F2_glossapi_plus_hplt",
        tokenizers_root / "F2_glossapi_plus_hplt_70_30" / "tokenizer.json",
        phase_train_discovery("glossapi_plus_hplt_70_30"),
    ))
    phases.append(Phase(
        "train_C1_glossapi_only",
        tokenizers_root / "C1_glossapi_only" / "tokenizer" / "tokenizer.json",
        phase_train_continuous("glossapi_only"),
    ))
    phases.append(Phase(
        "train_C2_glossapi_plus_hplt",
        tokenizers_root / "C2_glossapi_plus_hplt_70_30" / "tokenizer" / "tokenizer.json",
        phase_train_continuous("glossapi_plus_hplt_70_30"),
    ))
    phases.append(Phase("done", run_root / "all_done.json", phase_done))

    # ----- run -----

    started = time.time()
    print(f"[{now_iso()}] orchestrator START run_root={run_root} mode={args.mode}", flush=True)
    with open(orchestrator_log, "a", encoding="utf-8") as fh:
        fh.write(f"\n[{now_iso()}] START mode={args.mode} skip_dedup={args.skip_dedup}\n")

    for ph in phases:
        if ph.is_done():
            msg = f"[{now_iso()}] SKIP {ph.name} (marker exists: {ph.output_marker})"
            print(msg, flush=True)
            with open(orchestrator_log, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
            continue
        msg = f"[{now_iso()}] RUN {ph.name}"
        print(msg, flush=True)
        with open(orchestrator_log, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
        rc = ph.fn()
        if rc != 0:
            err = f"[{now_iso()}] FAIL {ph.name} rc={rc}; see {log_dir / (ph.name + '.log')}"
            print(err, flush=True)
            with open(orchestrator_log, "a", encoding="utf-8") as fh:
                fh.write(err + "\n")
            return rc
        ok = f"[{now_iso()}] DONE {ph.name}"
        print(ok, flush=True)
        with open(orchestrator_log, "a", encoding="utf-8") as fh:
            fh.write(ok + "\n")

    elapsed = time.time() - started
    msg = f"[{now_iso()}] ALL DONE in {elapsed:.0f}s"
    print(msg, flush=True)
    with open(orchestrator_log, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    return 0


def _wrap_dedup(dedup_fn: Callable[[], int], marker: Path) -> int:
    """Run dedup. If the underlying CLI reports stage_01 or stage_02 status=completed,
    touch the marker so subsequent runs skip dedup."""
    rc = dedup_fn()
    if rc == 0:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"completed_at": now_iso()}, indent=2) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
