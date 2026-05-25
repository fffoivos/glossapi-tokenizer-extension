# Apertus Extension Human Naming Plan

Date: 2026-05-25.

Status: implemented in
`release/apertus-tokenizer-extension/` on 2026-05-25.

Purpose: replace the current unreadable release names with short names a human
can recognize quickly.

## The Real Issue

The current release has paths like:

```text
checkpoints/td-layer11-cpt-3p5b-iter834/
```

That name mixes the essence with technical subtext:

- essence: Token Distillation checkpoint after 3.5B tokens;
- useful but secondary: layer 11;
- obvious context: CPT;
- implementation detail: iter 834;
- hidden fact: the folder does not contain weights.

The public name should be:

```text
TokenDistil-3.5B
```

The README/manifest should say:

- target layer: 11;
- exact iteration: 834;
- format: HF or Megatron;
- source run tag;
- tokenizer;
- training data;
- verification status.

## Naming Rule

Public names should contain only:

```text
<method-or-artifact>-<training-point-or-size>
```

Examples:

- `TokenDistil-3.5B`;
- `TokenDistil-2B`;
- `TokenDistil-Init`;
- `Vanilla-3.5B`;
- `ReTok-3.5B`;
- `ModernGreek-148k`;
- `ModernGreek-Polytonic-154k`;
- `CPT-7B-mix`;
- `3.5B-comparison`.

Technical details go in `README.md` and `manifest.json`.

## Essence Versus Subtext

| Put in the public name | Put in README/manifest |
|---|---|
| `TokenDistil` | layer 11, final-layer pilot, 25 snippets, TD source repo |
| `Vanilla` | base tokenizer, no new rows |
| `ReTok` | subpiece averaging, norm matching |
| `Init`, `0.5B`, `1B`, `2B`, `3.5B` | exact iteration number, global step, Slurm job, timestamp |
| `ModernGreek-148k` | 17,408 tokens, exact vocab size 148,480, SHA-256 |
| `ModernGreek-Polytonic-154k` | 17,408 + 5,120 tokens, exact vocab size 153,600 |
| `CPT-7B-mix` | 70/24/4/2 recipe, source datasets, base-tokenized 9.83B-token count |
| `3.5B-comparison` | task list, checkpoint ids, eval job ids |

Avoid in public names unless needed to distinguish two visible siblings:

- `layer11`;
- `cpt`;
- `iter834`;
- `r17`;
- `tp2`;
- `megatron`;
- timestamps;
- Slurm job ids.

## Weights Rule

`checkpoints/` should contain actual model weights.

If a folder only points to Clariden, it is not a checkpoint folder. It should be
under:

```text
locations/
```

or:

```text
remote/
```

Recommended:

```text
locations/
```

because it reads plainly: "where are the artifacts?"

## Proposed Release Layout

```text
README.md
ARTIFACTS.md
manifest.json
checksums.sha256

tokenizers/
  ModernGreek-148k/
  ModernGreek-Polytonic-154k/

checkpoints/
  TokenDistil-3.5B/
  TokenDistil-2B/
  TokenDistil-Init/
  Vanilla-3.5B/
  ReTok-3.5B/

locations/
  TokenDistil-3.5B.md
  TokenDistil-2B.md
  TokenDistil-Init.md
  Vanilla-3.5B.md
  ReTok-3.5B.md
  CPT-7B-mix.md

datasets/
  CPT-7B-mix/

results/
  3.5B-comparison/
  2B-bakeoff/
  V4-baseline/

source-code/
  README.md
  manifest.json

provenance/
  tokenizer-selection/
  dataset-build/
  token-distillation/
  conversion-roundtrip/
  evals/

archive/
  legacy-layout.md
```

Important: if weights are not uploaded yet, `checkpoints/` should either be
absent or contain only `README.md` saying "weights not uploaded yet." It should
not contain pointer-only artifact folders.

## Old Versus New

### Tokenizers

| Current path | Better path |
|---|---|
| `tokenizer/modern-greek-17408/` | `tokenizers/ModernGreek-148k/` |
| `tokenizer/polytonic-plus-5120/` | `tokenizers/ModernGreek-Polytonic-154k/` |

Why:

- `ModernGreek` is the visible thing.
- `148k` is easier to read than `17408`, while the exact `17,408` added-token
  number stays in the README.
- `Polytonic` tells the human what the second tokenizer is.

### Checkpoints With Weights

These names are for folders that actually contain model files.

| Current or missing path | Better path |
|---|---|
| missing selected HF weights | `checkpoints/TokenDistil-3.5B/` |
| missing TD 2B HF weights, if uploaded | `checkpoints/TokenDistil-2B/` |
| missing TD init HF weights, if uploaded | `checkpoints/TokenDistil-Init/` |
| missing Vanilla 3.5B HF weights, if uploaded | `checkpoints/Vanilla-3.5B/` |
| missing ReTok 3.5B HF weights, if uploaded | `checkpoints/ReTok-3.5B/` |

Expected contents for a loadable HF checkpoint:

```text
config.json
generation_config.json
model-00001-of-0000N.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
README.md
manifest.json
```

The selected weights source to upload is:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_td_layer11/iter_0000834_hf
```

### Remote Locations, Not Weights

These names are for pointer files only.

| Current path | Better path |
|---|---|
| `checkpoints/td-layer11-cpt-3p5b-iter834/` | `locations/TokenDistil-3.5B.md` |
| `checkpoints/td-layer11-init-r17-tp2/` | `locations/TokenDistil-Init.md` |
| `checkpoints/baselines/vanilla-3p5b-iter834/` | `locations/Vanilla-3.5B.md` |
| `checkpoints/baselines/retok-3p5b-iter834/` | `locations/ReTok-3.5B.md` |

Each location file should have:

```text
Human name: TokenDistil-3.5B
Weights status: not uploaded here / uploaded at checkpoints/TokenDistil-3.5B
Clariden Megatron checkpoint: ...
Clariden HF eval copy: ...
Tokenizer: ModernGreek-148k
Training data: CPT-7B-mix
Technical notes: layer 11, iter 834, TP=2, R17-patched
```

### Dataset

| Current path | Better path |
|---|---|
| `training-data/cpt-7b-mix/` | `datasets/CPT-7B-mix/` |
| `training-data/cpt-7b-mix/source_graph.json` | `datasets/CPT-7B-mix/source-graph.json` |

Why:

- `datasets/` is shorter and normal.
- `CPT-7B-mix` is the human-level artifact.
- The fact that it is a recipe/pointer rather than full dataset payload should
  be stated in `datasets/CPT-7B-mix/README.md`.

### Results

| Current path | Better path |
|---|---|
| `results/SUMMARY.md` | `results/3.5B-comparison/README.md` |
| `results/benchmark_table.csv` | `results/3.5B-comparison/benchmark-table.csv` |
| `results/continuation_3p5b_summary.json` | `results/3.5B-comparison/summary.json` |
| `results/plots/*` | `results/3.5B-comparison/plots/*` |

Optional result folders:

```text
results/2B-bakeoff/
results/TD-layer-pilot/
results/V4-baseline/
```

### Source Code

| Current path | Better path |
|---|---|
| `code-links/README.md` | `source-code/README.md` |
| `code-links/github_subproject_manifest.json` | `source-code/manifest.json` |

The release should say:

```text
The runnable scripts live on GitHub:
https://github.com/fffoivos/glossapi-tokenizer-extension/tree/main/subprojects/03_apertus_extension_and_embedding_adaptation
```

No broad `subprojects/` mirror should be center stage on HF.

### Root Files

| Current file | Better file |
|---|---|
| `MANIFEST.json` | `manifest.json` |
| `ARTIFACT_GRAPH.md` | `ARTIFACTS.md` |
| `SHA256SUMS` | `checksums.sha256` |
| `README.md` | `README.md` |

## Method Names

Use these exact visible method names:

| Method | Meaning |
|---|---|
| `Vanilla` | original Apertus tokenizer/control |
| `ReTok` | retokenization/subpiece-mean init |
| `TokenDistil` | Token Distillation init/refinement |
| `Centroid` | centroid init; archive unless needed for comparison |

Do not use `td` in public names. Use `TokenDistil`.

## Training Point Names

Use these exact visible training points:

| Name | Meaning |
|---|---|
| `Init` | initialized but not CPT-trained |
| `0.5B` | roughly 0.5B tokens |
| `1B` | roughly 1B tokens |
| `2B` | roughly 2B tokens |
| `3.5B` | roughly 3.5B tokens |
| `15B` | future production-scale run |

Exact iterations go in manifests:

| Human name | Exact metadata |
|---|---|
| `TokenDistil-3.5B` | `iter=834`, `target_layer=11`, `run_tag=continuation_3p5b_20260524T143012Z_td_layer11` |
| `TokenDistil-2B` | `iter=476`, `target_layer=11`, `run_tag=td_full25_layer11_2b_20260523T165038Z` |
| `Vanilla-3.5B` | `iter=834`, `run_tag=continuation_3p5b_20260524T143012Z_vanilla` |
| `ReTok-3.5B` | `iter=834`, `run_tag=continuation_3p5b_20260524T143012Z_retok` |

## Recommended Immediate Fix

1. Rename the public folders using the tables above.

2. Move pointer-only entries out of `checkpoints/` into `locations/`.

3. Add:

   ```text
   checkpoints/README.md
   ```

   saying whether checkpoint weights are uploaded.

4. Upload actual selected HF weights into:

   ```text
   checkpoints/TokenDistil-3.5B/
   ```

   If weights cannot be uploaded yet, leave `checkpoints/TokenDistil-3.5B/`
   absent and keep only `locations/TokenDistil-3.5B.md`.

5. Update root README main table to say:

   | Artifact | Status |
   |---|---|
   | `TokenDistil-3.5B` | weights uploaded / location only |
   | `ModernGreek-148k` | tokenizer uploaded |
   | `CPT-7B-mix` | recipe and location |
   | `3.5B-comparison` | results uploaded |

7. Upload the v2 layout additively to HF.

8. After review, move the current bad names under:

   ```text
   archive/legacy-layout-20260525/
   ```

## Bottom Line

The release should read like this:

```text
Tokenizers:
  ModernGreek-148k
  ModernGreek-Polytonic-154k

Checkpoints:
  TokenDistil-3.5B
  TokenDistil-2B
  TokenDistil-Init
  Vanilla-3.5B
  ReTok-3.5B

Dataset:
  CPT-7B-mix

Results:
  3.5B-comparison
```

Everything else is detail, and belongs in READMEs, manifests, or provenance.
