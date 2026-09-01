#!/usr/bin/env python3
"""Publish the completed benchmark metadata and ordered branch aliases.

This is deliberately metadata-only. It makes small README/index commits to
the existing private model branches, creates ordered refs to those commits,
verifies the model-file inventory was unchanged, and only then removes the
old unprefixed aliases. It never uploads or transforms model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SUBPROJECT = HERE.parents[1]
DATA = SUBPROJECT / "presentations/FULL8_ALL_CHECKPOINT_NATIVE_BENCHMARKS_20260819.data.json"
REPO = "fffoivos/apertus-8b-greek-cpt"


@dataclass(frozen=True)
class Ref:
    ordinal: int
    iteration: int
    old: str
    new: str


REFS = (
    Ref(0, 400, "step400-tokens2B", "00-step400-tokens2B"),
    Ref(1, 1192, "step1192-tokens5B", "01-step1192-tokens5B"),
    Ref(2, 2384, "step2384-tokens10B", "02-step2384-tokens10B"),
    Ref(3, 3576, "step3576-tokens15B", "03-step3576-tokens15B"),
    Ref(4, 4768, "step4768-tokens20B", "04-step4768-tokens20B"),
    Ref(5, 5960, "step5960-tokens25B", "05-step5960-tokens25B"),
    Ref(6, 7152, "step7152-tokens30B", "06-step7152-tokens30B"),
    Ref(7, 8344, "step8344-tokens35B", "07-step8344-tokens35B"),
    Ref(8, 9536, "step9536-tokens40B", "08-step9536-tokens40B"),
    Ref(9, 10728, "step10728-tokens45B", "09-step10728-tokens45B"),
    Ref(10, 11920, "step11920-tokens50B", "10-step11920-tokens50B"),
    Ref(11, 13112, "step13112-tokens55B", "11-step13112-tokens55B"),
    Ref(12, 14304, "step14304-tokens60B", "12-step14304-tokens60B"),
    Ref(13, 14627, "step14627-tokens61B", "13-step14627-tokens61B"),
    Ref(14, 15496, "step15496-tokens65B", "14-step15496-tokens65B"),
    Ref(15, 16688, "step16688-tokens70B", "15-step16688-tokens70B"),
    Ref(16, 17880, "step17880-tokens75B", "16-step17880-tokens75B"),
    Ref(17, 18284, "main", "17-step18284-tokens77B"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def branch_map(api: Any) -> dict[str, str]:
    return {str(ref.name): str(ref.target_commit) for ref in api.list_repo_refs(REPO, repo_type="model").branches}


def point_map(data: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    native = {int(point["iteration"]): point for benchmark in data["benchmarks"] for point in benchmark["points"]}
    # Native points are needed grouped by benchmark, rather than flattened.
    by_iteration: dict[int, dict[str, Any]] = {int(c["iteration"]): {} for c in data["checkpoints"]}
    for benchmark in data["benchmarks"]:
        for point in benchmark["points"]:
            by_iteration[int(point["iteration"])][benchmark["id"]] = point
    greek = {int(point["iteration"]): point for point in data["greekmmlu"]}
    require(len(by_iteration) == 19 and len(greek) == 19 and native, "unexpected report-data checkpoint coverage")
    return by_iteration, greek


def score_table(points: dict[str, dict[str, Any]]) -> str:
    order = ("asep_mcqa", "demosqa", "gpcr", "medical_mcqa", "oyxoy_metaphor", "oyxoy_nli", "oyxoy_wic", "oyxoy_wsd_definition")
    labels = {"asep_mcqa": "ASEP MCQA", "demosqa": "DemosQA", "gpcr": "GPCR", "medical_mcqa": "Medical MCQA", "oyxoy_metaphor": "OYXOY metaphor", "oyxoy_nli": "OYXOY NLI", "oyxoy_wic": "OYXOY WiC", "oyxoy_wsd_definition": "OYXOY WSD"}
    rows = ["| Benchmark | Strict n | Accuracy | Choice NLL |", "| --- | ---: | ---: | ---: |"]
    for ident in order:
        point = points[ident]
        nll = point["choice_nll"]
        rows.append(f"| {labels[ident]} | {int(point['n']):,} | {100 * float(point['accuracy']):.2f}% | {'—' if nll is None else f'{float(nll):.4f}'} |")
    return "\n".join(rows)


def card(ref: Ref, native: dict[str, dict[str, Any]], greek: dict[str, Any]) -> str:
    tokens = ref.iteration * 4_194_304
    phase = "cooldown start" if ref.iteration == 14627 else "cooldown" if ref.iteration > 14627 else "plateau"
    return "\n".join((
        "---", "language: el", "license: apache-2.0", "library_name: transformers", "pipeline_tag: text-generation", "---", "",
        f"# Apertus 8B Greek CPT — {ref.new}", "",
        "One immutable continued-pretraining checkpoint. This is a base model, not instruction tuned.", "",
        "## Checkpoint", "",
        f"- Ordered release position: `{ref.ordinal:02d}`", f"- Update: `{ref.iteration:,}`", f"- Token slots consumed: `{tokens:,}` ({tokens / 1e9:.3f}B)",
        f"- Training phase: `{phase}`", "- Geometry: 8B; 148,992-token extended vocabulary; untied input/output embeddings; RoPE θ=500,000; context 4,096.",
        "- Training mix: stationary 79% Modern Greek, 20% foreign-language replay, 1% Old-Greek replay.", "",
        "## Frozen GreekMMLU", "",
        f"- Decontaminated GreekMMLU: {int(greek['n']):,} questions; accuracy `{100 * float(greek['accuracy']):.2f}%`; choice NLL `{float(greek['choice_nll']):.4f}`.", "",
        "## Native-Greek benchmark matrix", "",
        "All 19 saved checkpoints were evaluated with the same frozen Greek prompts and FP32 zero-shot candidate-likelihood scorer. Scores below use the strict post-hoc contamination-filtered subsets: 73,894 retained examples of 83,970 scored examples. The filter removes only traceable strong two-surface matches; it does not change the trained model or corpus.", "",
        score_table(native), "",
        f"The root [`CHECKPOINTS.md`](https://huggingface.co/{REPO}/blob/main/CHECKPOINTS.md) and [`checkpoint-index.json`](https://huggingface.co/{REPO}/raw/main/checkpoint-index.json) provide the complete ordered trajectory, question populations and evaluation contract.", "",
        "## Frozen provenance", "",
        "- Parent: [`swiss-ai/Apertus-8B-2509`](https://huggingface.co/swiss-ai/Apertus-8B-2509/tree/3162c99675aa588097cecd4a24b9aa1f712af477).",
        "- Extended tokenizer: [`fffoivos/apertus-tokenizer-extension`](https://huggingface.co/fffoivos/apertus-tokenizer-extension/tree/fcd33ec09fb7d86bc072b3a4b3e890efa6473b66).",
        "- Public Modern-Greek train-only snapshot: [`fffoivos/apertus-8b-greek-cpt-modern-greek-train`](https://huggingface.co/datasets/fffoivos/apertus-8b-greek-cpt-modern-greek-train/tree/99f15b6eb554416b17cc0e5ae3a1b594055f85e1).",
        "- Exact packed 79/20/1 mixture is documented in the private companion dataset `fffoivos/apertus-8b-greek-cpt-d0-full-mix`; that does not authorize replay-source redistribution.",
        "- This metadata-only release does not alter model weights, tokenizer, training text, masking or token order.", "",
    ))


def index_markdown(entries: list[dict[str, Any]]) -> str:
    lines = ["# Apertus 8B Greek CPT — checkpoint index", "", "This ordered release contains the 18 converted CPT exports. The separately evaluated initialization anchor is not re-uploaded here; it is the pre-CPT reference in the matrix.", "", "All benchmark scores use the same strict post-hoc contamination-filtered populations: GreekMMLU `n=16,159`; native-Greek suite `n=73,894` across eight benchmark views. The scorer is FP32 zero-shot length-normalized candidate continuation likelihood.", "", "| Release | Branch | Update | Token slots | GreekMMLU accuracy | GreekMMLU NLL |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for entry in entries:
        lines.append(f"| {entry['release_ordinal']:02d} | [`{entry['revision']}`](https://huggingface.co/{REPO}/tree/{entry['revision']}) | {entry['iteration']:,} | {entry['token_slots']:,} | {100 * entry['greekmmlu']['accuracy']:.2f}% | {entry['greekmmlu']['choice_nll']:.4f} |")
    lines.extend(("", "`main` is retained as the terminal-checkpoint alias and points to the same commit as `17-step18284-tokens77B`."))
    return "\n".join(lines) + "\n"


def model_inventory(api: Any, revision: str) -> dict[str, tuple[int, str | None]]:
    info = api.model_info(REPO, revision=revision, files_metadata=True)
    result = {}
    for item in info.siblings:
        if item.rfilename in {"README.md", "CHECKPOINTS.md", "checkpoint-index.json"}:
            continue
        lfs = getattr(item, "lfs", None)
        oid = lfs.get("oid") if isinstance(lfs, dict) else getattr(lfs, "oid", None)
        result[str(item.rfilename)] = (int(item.size or 0), str(oid) if oid else None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform Hub metadata commits/ref changes")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("HF_TOKEN"), "HF_TOKEN must be supplied per command")
    from huggingface_hub import HfApi, hf_hub_download

    data = json.loads(DATA.read_text())
    native, greek = point_map(data)
    api = HfApi(token=os.environ["HF_TOKEN"])
    before = branch_map(api)
    expected_old = {ref.old for ref in REFS}
    require(expected_old <= set(before), f"missing expected live refs: {sorted(expected_old - set(before))}")
    require(not {ref.new for ref in REFS} & set(before), "ordered refs already exist; refusing ambiguous partial release")
    baseline = {ref.old: {"commit": before[ref.old], "model_inventory": model_inventory(api, ref.old)} for ref in REFS}
    cards = {ref.old: card(ref, native[ref.iteration], greek[ref.iteration]) for ref in REFS}
    entries = [{"release_ordinal": ref.ordinal, "revision": ref.new, "iteration": ref.iteration,
                "token_slots": ref.iteration * 4_194_304, "tokens_b": ref.iteration * 4_194_304 / 1e9,
                "phase": "cooldown_start" if ref.iteration == 14627 else "cooldown" if ref.iteration > 14627 else "plateau",
                "greekmmlu": greek[ref.iteration], "native_greek": native[ref.iteration]} for ref in REFS]
    index = {"schema_version": "apertus_checkpoint_index_v1", "status": "complete_private_metadata_release", "model_repo": REPO,
             "ordering": "zero_padded_release_ordinal_ascending", "default_revision": "08-step9536-tokens40B",
             "evaluation": {"greekmmlu_clean_n": 16159, "native_suite_retained_examples": 73894,
                            "native_suite_source_examples": 83970, "native_suite_excluded_examples": 10076,
                            "scorer": "FP32 zero-shot length-normalized candidate continuation likelihood"},
             "checkpoints": entries,
             "note": "The initialization anchor is evaluated in the 19-point matrix but is not a converted export branch in this repository."}
    plan = {"schema_version": "apertus_ordered_hf_checkpoint_metadata_release_v1", "repo": REPO, "apply": args.apply,
            "old_refs": baseline, "ordered_refs": [{"old": ref.old, "new": ref.new, "iteration": ref.iteration} for ref in REFS],
            "metadata_files": ["README.md", "CHECKPOINTS.md", "checkpoint-index.json"]}
    if not args.apply:
        write_json(args.receipt, {**plan, "status": "dry_run_passed"})
        print(json.dumps({"ok": True, "status": "dry_run_passed", "refs": len(REFS), "receipt": str(args.receipt)}, sort_keys=True))
        return 0

    for ref in REFS:
        response = api.upload_file(path_or_fileobj=cards[ref.old].encode(), path_in_repo="README.md", repo_id=REPO,
                                   repo_type="model", revision=ref.old, commit_message=f"Document completed benchmark matrix for {ref.new}",
                                   parent_commit=before[ref.old])
        require(str(response.oid), f"missing Hub commit for {ref.old}")
    # main receives the ordered directory and machine-readable matrix alongside its card.
    main_after_card = branch_map(api)["main"]
    api.upload_file(path_or_fileobj=index_markdown(entries).encode(), path_in_repo="CHECKPOINTS.md", repo_id=REPO,
                    repo_type="model", revision="main", commit_message="Add ordered checkpoint directory", parent_commit=main_after_card)
    main_after_index = branch_map(api)["main"]
    api.upload_file(path_or_fileobj=json.dumps(index, indent=2, sort_keys=True).encode() + b"\n", path_in_repo="checkpoint-index.json", repo_id=REPO,
                    repo_type="model", revision="main", commit_message="Add machine-readable checkpoint index", parent_commit=main_after_index)
    after_cards = branch_map(api)
    for ref in REFS:
        require(model_inventory(api, ref.old) == baseline[ref.old]["model_inventory"], f"model inventory drift on {ref.old}")
        api.create_branch(REPO, branch=ref.new, revision=after_cards[ref.old], repo_type="model", exist_ok=False)
    after = branch_map(api)
    verified: list[dict[str, Any]] = []
    for ref in REFS:
        require(after[ref.new] == after[ref.old], f"ordered ref target drift: {ref.new}")
        local = Path(hf_hub_download(repo_id=REPO, repo_type="model", revision=ref.new, filename="README.md", token=os.environ["HF_TOKEN"]))
        observed = local.read_bytes()
        expected = cards[ref.old].encode()
        require(observed == expected, f"README content drift: {ref.new}")
        verified.append({"old": ref.old, "new": ref.new, "commit": after[ref.new], "readme_sha256": sha256_bytes(observed)})
    main_index = Path(hf_hub_download(repo_id=REPO, repo_type="model", revision="main", filename="checkpoint-index.json", token=os.environ["HF_TOKEN"]))
    require(json.loads(main_index.read_text()) == index, "checkpoint index content drift")
    for ref in REFS:
        if ref.old != "main":
            api.delete_branch(REPO, branch=ref.old, repo_type="model")
    final_refs = branch_map(api)
    require({ref.new for ref in REFS} <= set(final_refs) and "main" in final_refs, "final ordered ref coverage drift")
    receipt = {**plan, "status": "completed", "verified": verified, "final_refs": final_refs,
               "main_index_sha256": sha256_bytes(main_index.read_bytes())}
    write_json(args.receipt, receipt)
    print(json.dumps({"ok": True, "status": "completed", "ordered_refs": len(REFS), "receipt": str(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
