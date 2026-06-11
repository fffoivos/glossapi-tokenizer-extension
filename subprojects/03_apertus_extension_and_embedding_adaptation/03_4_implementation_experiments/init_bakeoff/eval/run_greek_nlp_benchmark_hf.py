#!/usr/bin/env python3
"""Run greek-nlp/benchmark tasks against HF-format Apertus checkpoints.

The upstream benchmark runner is Ollama-oriented. This script keeps the
upstream task definitions, prompts, dataset loaders, and metrics unchanged,
but replaces the generation backend with a small Transformers backend that can
load our local HF-format checkpoints on Clariden.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ALL_TASKS = "all"
NO_CAP_PROFILE = "none"
REASONABLE_CAP_PROFILE = "reasonable"

FULL_TEST_TASK_CAPS = {
    REASONABLE_CAP_PROFILE: {
        "legal_classification": 500,
        "ner": 500,
        "summarization": 300,
    }
}

FULL_TEST_TASK_OPTIONS = {
    REASONABLE_CAP_PROFILE: {
        "machine_translation": {
            "target_lang_limits": {"eng": 500, "fas": 500, "jpn": 500},
            "full_corpus_target_langs": ["fas", "jpn"],
        },
    }
}


@dataclass
class ModelSpec:
    label: str
    path: str


@dataclass
class GenerationResult:
    response: str
    latency_seconds: float


class TransformersBackend:
    def __init__(
        self,
        model_specs: list[ModelSpec],
        *,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        max_input_tokens: int = 3072,
    ) -> None:
        self.model_specs = {spec.label: spec.path for spec in model_specs}
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.max_input_tokens = max_input_tokens
        self._loaded_label: str | None = None
        self._tokenizer = None
        self._model = None

    def _torch_dtype(self):
        import torch

        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"Unsupported dtype: {self.dtype}")

    def _unload(self) -> None:
        if self._model is not None:
            del self._model
        if self._tokenizer is not None:
            del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded_label = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load(self, label: str) -> None:
        if self._loaded_label == label:
            return
        if label not in self.model_specs:
            raise KeyError(f"Unknown model label: {label}")
        self._unload()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.model_specs[label]
        print(f"[hf-backend] loading {label}: {model_path}", flush=True)
        started = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self._torch_dtype(),
            trust_remote_code=self.trust_remote_code,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if getattr(self._tokenizer, "pad_token_id", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()
        self._loaded_label = label
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[hf-backend] loaded {label} in {time.perf_counter() - started:.1f}s", flush=True)

    @staticmethod
    def _compose_prompt(system_prompt: str, prompt: str) -> str:
        return (
            f"{system_prompt.strip()}\n\n"
            f"User:\n{prompt.strip()}\n\n"
            "Assistant:\n"
        )

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        num_predict: int,
        timeout_seconds: int,
    ) -> GenerationResult:
        del timeout_seconds
        self._load(model)
        assert self._tokenizer is not None
        assert self._model is not None

        import torch

        full_prompt = self._compose_prompt(system_prompt, prompt)
        encoded = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
        generate_kwargs = {
            "max_new_tokens": num_predict,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
        }
        generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}

        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(**encoded, **generate_kwargs)
        input_len = int(encoded["input_ids"].shape[-1])
        new_ids = output_ids[0][input_len:]
        response = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return GenerationResult(response=response, latency_seconds=time.perf_counter() - started)


def parse_model_specs(values: list[str]) -> list[ModelSpec]:
    specs = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Model must be LABEL=PATH, got: {value}")
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise SystemExit(f"Model must be LABEL=PATH, got: {value}")
        specs.append(ModelSpec(label=label, path=path))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True, help="Path to cloned greek-nlp/benchmark repo.")
    parser.add_argument(
        "--task",
        default=ALL_TASKS,
        help="Task to run: all, or one of the upstream benchmark tasks.",
    )
    parser.add_argument("--models", nargs="+", required=True, help="One or more LABEL=HF_MODEL_OR_PATH specs.")
    parser.add_argument("--sample-size", type=int, default=100, help="Examples per task; 0 means full selected split.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeated sampled runs; >1 requires sample-size > 0.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task-cap-profile",
        choices=[NO_CAP_PROFILE, REASONABLE_CAP_PROFILE],
        default=NO_CAP_PROFILE,
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument(
        "--task-num-predict-overrides",
        default="",
        help="Comma-separated TASK=NUM overrides for generation caps, e.g. intent_classification=16,legal_classification=16.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    return parser.parse_args()


def _print_summary(summary: pd.DataFrame) -> None:
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}" if not math.isnan(x) else "nan"))


def _selected_tasks(task_name: str, available_tasks: list[str]) -> list[str]:
    if task_name == ALL_TASKS:
        return available_tasks
    if task_name not in available_tasks:
        raise SystemExit(f"Unknown task '{task_name}'. Available: all, {', '.join(available_tasks)}")
    return [task_name]


def _task_data_limit(task_name: str, cap_profile: str) -> int | None:
    return FULL_TEST_TASK_CAPS.get(cap_profile, {}).get(task_name)


def _task_specific_options(task_name: str, cap_profile: str) -> dict[str, object]:
    return dict(FULL_TEST_TASK_OPTIONS.get(cap_profile, {}).get(task_name, {}))


def _parse_task_num_predict_overrides(value: str) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Expected TASK=NUM in --task-num-predict-overrides, got: {item}")
        task_name, raw_num = item.split("=", 1)
        task_name = task_name.strip()
        if not task_name:
            raise SystemExit(f"Empty task name in --task-num-predict-overrides item: {item}")
        try:
            num = int(raw_num)
        except ValueError as exc:
            raise SystemExit(f"Invalid num_predict override in item '{item}': {raw_num}") from exc
        if num <= 0:
            raise SystemExit(f"num_predict override must be > 0 in item: {item}")
        overrides[task_name] = num
    return overrides


def _aggregate_repeated_summaries(summary_by_repeat: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeated_summary = pd.concat(summary_by_repeat, ignore_index=True)
    key_columns = [column for column in ["task", "model", "target_lang"] if column in repeated_summary.columns]
    numeric_columns = [
        column
        for column in repeated_summary.columns
        if column not in key_columns + ["repeat"] and pd.api.types.is_numeric_dtype(repeated_summary[column])
    ]
    aggregated = repeated_summary[key_columns].drop_duplicates().sort_values(key_columns).reset_index(drop=True)
    grouped = repeated_summary.groupby(key_columns, sort=False)
    for column in numeric_columns:
        aggregated[f"{column}_mean"] = grouped[column].mean().to_numpy()
        aggregated[f"{column}_sem"] = grouped[column].sem(ddof=1).fillna(0.0).to_numpy()
    return aggregated, repeated_summary


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    sys.path.insert(0, str(benchmark_root))

    from benchmark_suite import GenerationConfig, list_tasks, run_task, save_run_outputs

    model_specs = parse_model_specs(args.models)
    backend = TransformersBackend(
        model_specs,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
    )
    task_num_predict_overrides = _parse_task_num_predict_overrides(args.task_num_predict_overrides)
    sample_size = None if args.sample_size <= 0 else args.sample_size
    if args.repeats <= 0:
        raise SystemExit("--repeats must be >= 1")
    if args.repeats > 1 and sample_size is None:
        raise SystemExit("--repeats > 1 requires --sample-size > 0")

    selected_tasks = _selected_tasks(args.task, list_tasks())
    unknown_override_tasks = sorted(set(task_num_predict_overrides) - set(selected_tasks))
    if unknown_override_tasks:
        print(
            "Ignoring num_predict overrides outside this task selection: "
            + ", ".join(unknown_override_tasks),
            flush=True,
        )
        task_num_predict_overrides = {
            key: value for key, value in task_num_predict_overrides.items() if key in selected_tasks
        }
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output_dir)
    metadata = {
        "benchmark_root": str(benchmark_root),
        "benchmark_git_commit": _git_commit(benchmark_root),
        "work_dir": str(args.output_dir),
        "models": [spec.__dict__ for spec in model_specs],
        "tasks": selected_tasks,
        "sample_size": args.sample_size,
        "repeats": args.repeats,
        "random_state": args.random_state,
        "task_cap_profile": args.task_cap_profile,
        "temperature": args.temperature,
        "num_predict": args.num_predict,
        "task_num_predict_overrides": task_num_predict_overrides,
        "dtype": args.dtype,
        "max_input_tokens": args.max_input_tokens,
        "generated_utc": _utc_now(),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    combined_summaries: list[pd.DataFrame] = []
    combined_predictions: list[pd.DataFrame] = []
    data_csv = benchmark_root / "data.csv"

    print("Tasks:", selected_tasks, flush=True)
    print("Models:", [spec.label for spec in model_specs], flush=True)
    print("Output:", args.output_dir, flush=True)

    for task_name in selected_tasks:
        print(f"\n=== Running task: {task_name} ===", flush=True)
        task_data_limit = _task_data_limit(task_name, args.task_cap_profile)
        task_options = _task_specific_options(task_name, args.task_cap_profile)
        task_num_predict = task_num_predict_overrides.get(task_name, args.num_predict)
        config = GenerationConfig(
            temperature=args.temperature,
            num_predict=task_num_predict,
            timeout_seconds=args.timeout_seconds,
        )
        print(f"num_predict: {task_num_predict}", flush=True)
        if args.repeats == 1:
            summary, raw = run_task(
                task_name=task_name,
                models=[spec.label for spec in model_specs],
                sample_size=sample_size,
                data_limit=task_data_limit,
                random_state=args.random_state,
                data_csv=data_csv,
                config=config,
                backend=backend,
                task_options=task_options,
            )
            save_run_outputs(summary, raw, args.output_dir, task_name)
            combined_summaries.append(summary.assign(task_name=task_name))
            combined_predictions.append(raw.assign(task_name=task_name))
            _print_summary(summary)
        else:
            repeat_summaries: list[pd.DataFrame] = []
            repeat_predictions: list[pd.DataFrame] = []
            for repeat_index in range(args.repeats):
                repeat_number = repeat_index + 1
                repeat_seed = args.random_state + repeat_index
                repeat_output_dir = args.output_dir / task_name / f"repeat_{repeat_number:02d}"
                print(f"Running repeat {repeat_number}/{args.repeats} with seed {repeat_seed}", flush=True)
                summary, raw = run_task(
                    task_name=task_name,
                    models=[spec.label for spec in model_specs],
                    sample_size=sample_size,
                    data_limit=task_data_limit,
                    random_state=repeat_seed,
                    data_csv=data_csv,
                    config=config,
                    backend=backend,
                    task_options=task_options,
                )
                save_run_outputs(summary, raw, repeat_output_dir, task_name)
                summary = summary.copy()
                raw = raw.copy()
                summary["repeat"] = repeat_number
                raw["repeat"] = repeat_number
                repeat_summaries.append(summary)
                repeat_predictions.append(raw)

            aggregated_summary, repeated_summary = _aggregate_repeated_summaries(repeat_summaries)
            repeated_predictions = pd.concat(repeat_predictions, ignore_index=True)
            task_output_dir = args.output_dir / task_name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            aggregated_summary.to_csv(task_output_dir / f"{task_name}_summary_with_sem.csv", index=False)
            repeated_summary.to_csv(task_output_dir / f"{task_name}_repeat_summaries.csv", index=False)
            repeated_predictions.to_csv(task_output_dir / f"{task_name}_repeat_predictions.csv", index=False)
            combined_summaries.append(aggregated_summary.assign(task_name=task_name))
            combined_predictions.append(repeated_predictions.assign(task_name=task_name))
            _print_summary(aggregated_summary)

    if len(combined_summaries) > 1:
        combined_summary = pd.concat(combined_summaries, ignore_index=True)
        combined_summary.to_csv(
            args.output_dir / ("all_tasks_summary_with_sem.csv" if args.repeats > 1 else "all_tasks_summary.csv"),
            index=False,
        )
    if len(combined_predictions) > 1:
        combined_predictions_df = pd.concat(combined_predictions, ignore_index=True)
        combined_predictions_df.to_csv(
            args.output_dir / ("all_tasks_repeat_predictions.csv" if args.repeats > 1 else "all_tasks_predictions.csv"),
            index=False,
        )


def _git_commit(path: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
