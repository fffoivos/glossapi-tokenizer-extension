from __future__ import annotations

import json
from pathlib import Path

import typer

from glossapi_corpus_cli import pipeline
from glossapi_corpus_cli import text_dedup


app = typer.Typer(add_completion=False, no_args_is_help=True)
dedup_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(dedup_app, name="dedup-text")


@app.command("build")
def build_command(
    output_root: Path = typer.Option(pipeline.DEFAULT_OUTPUT_ROOT, help="Output root for canonical parquet files"),
    include_external: bool = typer.Option(True, help="Include the selected external HF corpora"),
    score_external_quality: bool = typer.Option(False, help="Run glossapi_rs_noise on the external HF corpora during build"),
    force_download_external: bool = typer.Option(False, help="Redownload external HF files even if they already exist locally"),
    workers: int = typer.Option(pipeline.default_build_workers(), min=1, help="Parallel dataset workers"),
) -> None:
    results = pipeline.build_canonical_corpus(
        output_root=output_root,
        include_external=include_external,
        score_external_quality=score_external_quality,
        force_download_external=force_download_external,
        workers=workers,
    )
    payload = [{"source_dataset": item.dataset_name, "path": str(item.path), "row_count": item.row_count} for item in results]
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("validate")
def validate_command(
    output_root: Path = typer.Option(pipeline.DEFAULT_OUTPUT_ROOT, help="Canonical dataset root"),
) -> None:
    typer.echo(json.dumps(pipeline.validate_canonical_corpus(output_root), ensure_ascii=False, indent=2))


@app.command("mix")
def mix_command(
    output_root: Path = typer.Option(pipeline.DEFAULT_OUTPUT_ROOT, help="Canonical dataset root"),
    mix_output_path: Path = typer.Option(..., help="Parquet path for the filtered mix"),
    include_sources: list[str] = typer.Option(None, help="Explicit source_dataset whitelist"),
    exclude_sources: list[str] = typer.Option(None, help="Explicit source_dataset blacklist"),
    exclude_needs_ocr_sources: list[str] = typer.Option(
        None,
        help="Drop rows with needs_ocr=true for these source_dataset values before mix export",
    ),
    quality_preset: str = typer.Option("none", help="none|modern_strict|modern_relaxed|historical_tolerant"),
    historical_mode: str = typer.Option("include", help="include|exclude|only"),
    math_mode: str = typer.Option("include", help="include|exclude|only"),
    latex_mode: str = typer.Option("include", help="include|exclude|only"),
    dedup_metadata_root: Path | None = typer.Option(None, help="Builder-facing dedup metadata bundle root"),
    dedup_action: str = typer.Option("ignore", help="ignore|annotate|drop_intra|drop_intra_and_inter"),
    dedup_exact_stage: str = typer.Option("strict_and_relaxed", help="strict_only|strict_and_relaxed"),
    dedup_similarity_threshold: float | None = typer.Option(None, help="Builder-time near-dup threshold; must be >= exported pair floor"),
    dedup_inter_dataset_policy: str = typer.Option("share_aware", help="quality_first|share_aware"),
    dedup_source_weights_path: Path | None = typer.Option(None, help="Optional JSON mapping source_dataset to positive weight"),
    source_mix_config_path: Path | None = typer.Option(
        None,
        help="Optional JSON config describing grouped/per-dataset source mix fractions after filtering and dedup",
    ),
) -> None:
    payload = pipeline.build_mix_export(
        output_root=output_root,
        mix_output_path=mix_output_path,
        include_sources=include_sources or None,
        exclude_sources=exclude_sources or None,
        exclude_needs_ocr_sources=exclude_needs_ocr_sources or None,
        quality_preset=quality_preset,
        historical_mode=historical_mode,
        math_mode=math_mode,
        latex_mode=latex_mode,
        dedup_metadata_root=dedup_metadata_root,
        dedup_action=dedup_action,
        dedup_exact_stage=dedup_exact_stage,
        dedup_similarity_threshold=dedup_similarity_threshold,
        dedup_inter_dataset_policy=dedup_inter_dataset_policy,
        dedup_source_weights_path=dedup_source_weights_path,
        source_mix_config_path=source_mix_config_path,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("mix-prepare-selected-input")
def mix_prepare_selected_input_command(
    output_root: Path = typer.Option(pipeline.DEFAULT_OUTPUT_ROOT, help="Canonical dataset root"),
    selected_input_path: Path = typer.Option(..., help="Parquet path for the shared filtered+deduped selected input"),
    include_sources: list[str] = typer.Option(None, help="Explicit source_dataset whitelist"),
    exclude_sources: list[str] = typer.Option(None, help="Explicit source_dataset blacklist"),
    exclude_needs_ocr_sources: list[str] = typer.Option(
        None,
        help="Drop rows with needs_ocr=true for these source_dataset values before mix export",
    ),
    quality_preset: str = typer.Option("none", help="none|modern_strict|modern_relaxed|historical_tolerant"),
    historical_mode: str = typer.Option("include", help="include|exclude|only"),
    math_mode: str = typer.Option("include", help="include|exclude|only"),
    latex_mode: str = typer.Option("include", help="include|exclude|only"),
    dedup_metadata_root: Path | None = typer.Option(None, help="Builder-facing dedup metadata bundle root"),
    dedup_action: str = typer.Option("ignore", help="ignore|annotate|drop_intra|drop_intra_and_inter"),
    dedup_exact_stage: str = typer.Option("strict_and_relaxed", help="strict_only|strict_and_relaxed"),
    dedup_similarity_threshold: float | None = typer.Option(None, help="Builder-time near-dup threshold; must be >= exported pair floor"),
    dedup_inter_dataset_policy: str = typer.Option("share_aware", help="quality_first|share_aware"),
    dedup_source_weights_path: Path | None = typer.Option(None, help="Optional JSON mapping source_dataset to positive weight"),
) -> None:
    payload = pipeline.materialize_streaming_mix_selected_input(
        output_root=output_root,
        destination=selected_input_path,
        include_sources=include_sources or None,
        exclude_sources=exclude_sources or None,
        exclude_needs_ocr_sources=exclude_needs_ocr_sources or None,
        quality_preset=quality_preset,
        historical_mode=historical_mode,
        math_mode=math_mode,
        latex_mode=latex_mode,
        dedup_metadata_root=dedup_metadata_root,
        dedup_action=dedup_action,
        dedup_exact_stage=dedup_exact_stage,
        dedup_similarity_threshold=dedup_similarity_threshold,
        dedup_inter_dataset_policy=dedup_inter_dataset_policy,
        dedup_source_weights_path=dedup_source_weights_path,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("mix-build-from-selected-input")
def mix_build_from_selected_input_command(
    selected_input_path: Path = typer.Option(..., help="Shared filtered+deduped selected input parquet"),
    mix_output_path: Path = typer.Option(..., help="Parquet path for the final mix output"),
    source_mix_config_path: Path | None = typer.Option(
        None,
        help="Optional JSON config describing grouped/per-dataset source mix fractions after filtering and dedup",
    ),
    apply_standard_split_filters: bool = typer.Option(
        False,
        "--standard-split-filters",
        help="Apply the production split badness/OCR filters before resolving source-mix fractions",
    ),
    badness_lt: float = typer.Option(
        pipeline.DEFAULT_STANDARD_BADNESS_LT,
        help="When --standard-split-filters is enabled, drop greek_badness_score >= this threshold",
    ),
    mojibake_lte: float = typer.Option(
        pipeline.DEFAULT_STANDARD_MOJIBAKE_LTE,
        help="When --standard-split-filters is enabled, drop mojibake_badness_score > this threshold",
    ),
    allow_missing_badness_scores: bool = typer.Option(
        False,
        help="When --standard-split-filters is enabled, keep rows with missing score values",
    ),
) -> None:
    payload = pipeline.build_mix_output_from_selected_input(
        selected_input_path=selected_input_path,
        mix_output_path=mix_output_path,
        source_mix_config_path=source_mix_config_path,
        apply_standard_split_filters=apply_standard_split_filters,
        badness_lt=badness_lt,
        mojibake_lte=mojibake_lte,
        allow_missing_badness_scores=allow_missing_badness_scores,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("nanochat")
def nanochat_command(
    output_root: Path = typer.Option(pipeline.DEFAULT_OUTPUT_ROOT, help="Canonical dataset root"),
    export_root: Path = typer.Option(..., help="Directory for text-only nanochat shards"),
    nanochat_depth: int = typer.Option(..., help="nanochat --depth value"),
    target_param_data_ratio: float = typer.Option(10.5, help="Current nanochat default from scripts/base_train.py"),
    target_tokens: int | None = typer.Option(None, help="Explicit token budget override"),
    include_sources: list[str] = typer.Option(None, help="Explicit source_dataset whitelist"),
    exclude_sources: list[str] = typer.Option(None, help="Explicit source_dataset blacklist"),
    exclude_needs_ocr_sources: list[str] = typer.Option(
        None,
        help="Drop rows with needs_ocr=true for these source_dataset values before shard export",
    ),
    quality_preset: str = typer.Option("none", help="none|modern_strict|modern_relaxed|historical_tolerant"),
    historical_mode: str = typer.Option("include", help="include|exclude|only"),
    math_mode: str = typer.Option("include", help="include|exclude|only"),
    latex_mode: str = typer.Option("include", help="include|exclude|only"),
    dedup_metadata_root: Path | None = typer.Option(None, help="Builder-facing dedup metadata bundle root"),
    dedup_action: str = typer.Option("ignore", help="ignore|annotate|drop_intra|drop_intra_and_inter"),
    dedup_exact_stage: str = typer.Option("strict_and_relaxed", help="strict_only|strict_and_relaxed"),
    dedup_similarity_threshold: float | None = typer.Option(None, help="Builder-time near-dup threshold; must be >= exported pair floor"),
    dedup_inter_dataset_policy: str = typer.Option("share_aware", help="quality_first|share_aware"),
    dedup_source_weights_path: Path | None = typer.Option(None, help="Optional JSON mapping source_dataset to positive weight"),
    dedup_pool_full_include_threshold: float = typer.Option(
        pipeline.DEFAULT_DEDUP_POOL_FULL_INCLUDE_THRESHOLD,
        min=0.0,
        max=1.0,
        help="Fully include pools at or below this corpus-share threshold before proportional allocation",
    ),
    source_mix_config_path: Path | None = typer.Option(
        None,
        help="Optional JSON config describing grouped/per-dataset source mix fractions before shard export",
    ),
    shard_target_tokens: int = typer.Option(2_000_000, help="Approximate token target per shard"),
    row_group_rows: int = typer.Option(2048, min=1, help="Rows per parquet row group for nanochat shards"),
) -> None:
    payload = pipeline.export_nanochat_shards(
        output_root=output_root,
        export_root=export_root,
        nanochat_depth=nanochat_depth,
        target_param_data_ratio=target_param_data_ratio,
        target_tokens=target_tokens,
        include_sources=include_sources or None,
        exclude_sources=exclude_sources or None,
        exclude_needs_ocr_sources=exclude_needs_ocr_sources or None,
        quality_preset=quality_preset,
        historical_mode=historical_mode,
        math_mode=math_mode,
        latex_mode=latex_mode,
        dedup_metadata_root=dedup_metadata_root,
        dedup_action=dedup_action,
        dedup_exact_stage=dedup_exact_stage,
        dedup_similarity_threshold=dedup_similarity_threshold,
        dedup_inter_dataset_policy=dedup_inter_dataset_policy,
        dedup_source_weights_path=dedup_source_weights_path,
        dedup_pool_full_include_threshold=dedup_pool_full_include_threshold,
        source_mix_config_path=source_mix_config_path,
        shard_target_tokens=shard_target_tokens,
        row_group_rows=row_group_rows,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("estimate")
def estimate_command(
    nanochat_depth: int = typer.Option(..., help="nanochat --depth value"),
    target_param_data_ratio: float = typer.Option(10.5, help="Current nanochat default ratio"),
    target_tokens: int | None = typer.Option(None, help="Explicit token-budget override"),
) -> None:
    payload = {
        "nanochat_depth": nanochat_depth,
        "estimated_model_dim": pipeline.nanochat_model_dim(nanochat_depth),
        "estimated_scaling_params": pipeline.estimate_nanochat_scaling_params(nanochat_depth),
        "target_tokens": pipeline.target_nanochat_tokens(
            nanochat_depth,
            target_param_data_ratio=target_param_data_ratio,
            target_tokens=target_tokens,
        ),
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@dedup_app.command("run")
def dedup_run_command(
    input_root: Path = typer.Option(text_dedup.DEFAULT_INPUT_ROOT, help="Published snapshot root containing parquet files"),
    state_root: Path = typer.Option(text_dedup.DEFAULT_STATE_ROOT, help="Persistent dedup state root"),
    run_root: Path | None = typer.Option(None, help="Specific run directory; defaults to a timestamped path under analysis/dedup/text_publish/runs"),
    resume: bool = typer.Option(False, help="Resume an existing run_root with the same config"),
    max_workers: int = typer.Option(text_dedup.DEFAULT_RUN_MAX_WORKERS, min=1, help="Worker budget for exact-stage chunk compute and Stage 2 independent chunks"),
    greek_diacritic_policy: str = typer.Option(text_dedup.DEFAULT_GREEK_DIACRITIC_POLICY, help="Greek dedup policy: preserve|strip"),
    exact_only: bool = typer.Option(False, help="Stop after Stage 1 exact dedup and skip Stage 2 near-dup work"),
    minhash_threshold: float = typer.Option(text_dedup.DEFAULT_NEAR_THRESHOLD, help="Estimated signature Jaccard threshold for near-dup acceptance"),
    num_perm: int = typer.Option(text_dedup.DEFAULT_NUM_PERM, min=1, help="MinHash permutation count"),
    bands: int = typer.Option(text_dedup.DEFAULT_BANDS, min=1, help="LSH band count"),
    rows_per_band: int = typer.Option(text_dedup.DEFAULT_ROWS_PER_BAND, min=1, help="Rows per LSH band"),
    shingle_mode: str = typer.Option(text_dedup.DEFAULT_SHINGLE_MODE, help="Near-dup shingle mode: token|char"),
    shingle_size: int = typer.Option(text_dedup.DEFAULT_SHINGLE_SIZE, min=2, help="Near-dup shingle size"),
    max_bucket_size: int = typer.Option(text_dedup.DEFAULT_MAX_BUCKET_SIZE, min=2, help="Skip oversized LSH buckets above this member count"),
) -> None:
    payload = text_dedup.run_dedup_pipeline(
        input_root=input_root,
        state_root=state_root,
        run_root=run_root,
        resume=resume,
        max_workers=max_workers,
        greek_diacritic_policy=greek_diacritic_policy,
        exact_only=exact_only,
        minhash_threshold=minhash_threshold,
        num_perm=num_perm,
        bands=bands,
        rows_per_band=rows_per_band,
        shingle_mode=shingle_mode,
        shingle_size=shingle_size,
        max_bucket_size=max_bucket_size,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@dedup_app.command("prepare-test-run")
def dedup_prepare_test_run_command(
    experiment_root: Path = typer.Option(..., help="Root for the prepared test input, state, and launch script"),
    input_root: Path = typer.Option(text_dedup.DEFAULT_INPUT_ROOT, help="Published snapshot root containing parquet files"),
    rows_per_file: int = typer.Option(
        text_dedup.DEFAULT_TEST_ROWS_PER_FILE,
        min=1,
        help="Take at most this many head rows from each selected parquet file",
    ),
    max_files: int | None = typer.Option(
        None,
        min=1,
        help="Optional cap on sampled parquet files in sorted path order",
    ),
    max_workers: int = typer.Option(text_dedup.DEFAULT_RUN_MAX_WORKERS, min=1, help="Worker budget for the future dedup run"),
    greek_diacritic_policy: str = typer.Option(text_dedup.DEFAULT_GREEK_DIACRITIC_POLICY, help="Greek dedup policy: preserve|strip"),
    exact_only: bool = typer.Option(False, help="Prepare a launcher that stops after Stage 1 exact dedup"),
    minhash_threshold: float = typer.Option(text_dedup.DEFAULT_NEAR_THRESHOLD, help="Estimated signature Jaccard threshold for near-dup acceptance"),
    num_perm: int = typer.Option(text_dedup.DEFAULT_NUM_PERM, min=1, help="MinHash permutation count"),
    bands: int = typer.Option(text_dedup.DEFAULT_BANDS, min=1, help="LSH band count"),
    rows_per_band: int = typer.Option(text_dedup.DEFAULT_ROWS_PER_BAND, min=1, help="Rows per LSH band"),
    shingle_mode: str = typer.Option(text_dedup.DEFAULT_SHINGLE_MODE, help="Near-dup shingle mode: token|char"),
    shingle_size: int = typer.Option(text_dedup.DEFAULT_SHINGLE_SIZE, min=2, help="Near-dup shingle size"),
    max_bucket_size: int = typer.Option(text_dedup.DEFAULT_MAX_BUCKET_SIZE, min=2, help="Skip oversized LSH buckets above this member count"),
) -> None:
    payload = text_dedup.prepare_test_dedup_run(
        experiment_root=experiment_root,
        input_root=input_root,
        rows_per_file=rows_per_file,
        max_files=max_files,
        max_workers=max_workers,
        greek_diacritic_policy=greek_diacritic_policy,
        exact_only=exact_only,
        minhash_threshold=minhash_threshold,
        num_perm=num_perm,
        bands=bands,
        rows_per_band=rows_per_band,
        shingle_mode=shingle_mode,
        shingle_size=shingle_size,
        max_bucket_size=max_bucket_size,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@dedup_app.command("status")
def dedup_status_command(
    state_root: Path = typer.Option(text_dedup.DEFAULT_STATE_ROOT, help="Persistent dedup state root"),
    run_root: Path | None = typer.Option(None, help="Specific run directory; defaults to the latest successful run"),
) -> None:
    payload = text_dedup.dedup_status(state_root=state_root, run_root=run_root)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@dedup_app.command("export")
def dedup_export_command(
    state_root: Path = typer.Option(text_dedup.DEFAULT_STATE_ROOT, help="Persistent dedup state root"),
    run_root: Path | None = typer.Option(None, help="Specific run directory; defaults to the latest successful run"),
) -> None:
    payload = text_dedup.export_dedup_run(state_root=state_root, run_root=run_root)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@dedup_app.command("export-builder-metadata")
def dedup_export_builder_metadata_command(
    state_root: Path = typer.Option(text_dedup.DEFAULT_STATE_ROOT, help="Persistent dedup state root"),
    run_root: Path | None = typer.Option(None, help="Specific run directory; defaults to the latest successful run"),
) -> None:
    payload = text_dedup.export_builder_metadata_run(state_root=state_root, run_root=run_root)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
