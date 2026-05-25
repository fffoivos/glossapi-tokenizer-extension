# Apertus Release Upload Verification

Date: 2026-05-25.

Release repo:

```text
https://huggingface.co/fffoivos/apertus-tokenizer-extension
```

## What Changed

The public Hugging Face repo was reorganized around explicit top-level artifact
names:

```text
greek-extension-tokenizer/
cpt-training-dataset/
experiment-checkpoints/
benchmark-evals/
supporting-material/
```

The old generic top-level folders were removed:

```text
selected-tokenizer/
dataset/
checkpoints/
evals/
supporting/
```

## Checkpoint Weights Uploaded

All public experiment checkpoint folders now contain HF-format weights:

| Folder | Shards | Approx size |
|---|---:|---:|
| `experiment-checkpoints/TokenDistil-Init/` | 4 | 15.28 GiB |
| `experiment-checkpoints/TokenDistil-2B/` | 4 | 15.28 GiB |
| `experiment-checkpoints/TokenDistil-3.5B/` | 4 | 15.28 GiB |
| `experiment-checkpoints/Vanilla-2B/` | 4 | 15.02 GiB |
| `experiment-checkpoints/Vanilla-3.5B/` | 4 | 15.02 GiB |
| `experiment-checkpoints/ReTok-2B/` | 4 | 15.28 GiB |
| `experiment-checkpoints/ReTok-3.5B/` | 4 | 15.28 GiB |
| `experiment-checkpoints/Centroid-2B/` | 4 | 15.28 GiB |

## Upload Run

Final upload job:

```text
Clariden Slurm job: 2382635
partition: xfer
state: COMPLETED
elapsed: 00:07:21
exit code: 0:0
log: /users/fffoivos/apertus_hf_upload_checkpoints_20260525_2382635.log
```

The reusable upload script is committed at:

```text
subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/release_upload/upload_release_checkpoints_to_hf_from_clariden.sh
```

## Notes

A first detached login-node uploader was started while checking the HF token
path. It eventually uploaded the three TokenDistil folders, but the final
controlled run was moved to the non-GPU `xfer` partition. The xfer job skipped
unchanged TokenDistil payloads and uploaded the remaining Vanilla, ReTok, and
Centroid checkpoints.

A tiny Clariden upload-smoke file was used to verify auth from the cluster and
then removed from Hugging Face.

## Verification

Independent Hugging Face API verification passed after upload:

- top-level folders exactly matched the explicit release layout;
- every checkpoint folder had `README.md`, `manifest.json`, model config,
  tokenizer files, `model.safetensors.index.json`, and four safetensor shards;
- non-init bakeoff folders also had `bakeoff_conversion_metadata.json`;
- no old generic top-level folders remained.
