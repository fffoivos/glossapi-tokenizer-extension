#!/usr/bin/env python3
"""Patch Swiss-AI Megatron GPTDataset to support physical-order curriculum runs.

Megatron's GPTDataset normally shuffles both the document index and the sample
index. A physically ordered JSONL/bin artifact is therefore not enough for a
curriculum. This idempotent patch adds an environment-controlled path:

    MEGATRON_GPT_DATASET_NO_SHUFFLE=1

When set, GPTDataset builds document_index in source order and shuffle_index as
identity. The launcher only enables this through CURRICULUM_ORDER_MODE, so
exact reproduction runs can set CURRICULUM_ORDER_MODE=randomized.
"""
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "MEGATRON_GPT_DATASET_NO_SHUFFLE"


def patch_text(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False

    old = """            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            # Build the document index
            document_index = _build_document_index(
                self.indices, num_epochs, numpy_random_state, separate_final_epoch
            )
"""
    new = """            numpy_random_state = numpy.random.RandomState(self.config.random_seed)
            no_shuffle = os.environ.get("MEGATRON_GPT_DATASET_NO_SHUFFLE", "0") == "1"
            if no_shuffle:
                log_single_rank(
                    logger,
                    logging.INFO,
                    "GPTDataset physical-order mode active: document_index is sequential and shuffle_index is identity",
                )

            # Build the document index
            document_index = _build_document_index(
                self.indices,
                num_epochs,
                numpy_random_state,
                separate_final_epoch,
                shuffle=not no_shuffle,
            )
"""
    if old not in text:
        raise SystemExit("could not find document_index build block; patch target may have drifted")
    text = text.replace(old, new, 1)

    old = """            if separate_final_epoch:
                shuffle_index = _build_shuffle_index(
                    num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
                )
            else:
                shuffle_index = _build_shuffle_index(
                    sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
                )
"""
    new = """            if separate_final_epoch:
                shuffle_index = _build_shuffle_index(
                    num_samples_sans_final_epoch,
                    sample_index.shape[0] - 1,
                    numpy_random_state,
                    shuffle=not no_shuffle,
                )
            else:
                shuffle_index = _build_shuffle_index(
                    sample_index.shape[0] - 1,
                    sample_index.shape[0] - 1,
                    numpy_random_state,
                    shuffle=not no_shuffle,
                )
"""
    if old not in text:
        raise SystemExit("could not find shuffle_index build block; patch target may have drifted")
    text = text.replace(old, new, 1)

    old = """def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
) -> numpy.ndarray:
"""
    new = """def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
    shuffle: bool = True,
) -> numpy.ndarray:
"""
    if old not in text:
        raise SystemExit("could not find _build_document_index signature")
    text = text.replace(old, new, 1)

    old = "        numpy_random_state.shuffle(document_index)\n        return document_index"
    new = "        if shuffle:\n            numpy_random_state.shuffle(document_index)\n        return document_index"
    if old not in text:
        raise SystemExit("could not find document_index shuffle call")
    text = text.replace(old, new, 1)

    old = (
        "    doc_idx_first = _build_document_index(documents, num_epochs - 1, numpy_random_state, False)\n"
        "    doc_idx_last = _build_document_index(documents, 1, numpy_random_state, False)\n"
    )
    new = (
        "    doc_idx_first = _build_document_index(documents, num_epochs - 1, numpy_random_state, False, shuffle=shuffle)\n"
        "    doc_idx_last = _build_document_index(documents, 1, numpy_random_state, False, shuffle=shuffle)\n"
    )
    if old not in text:
        raise SystemExit("could not find recursive document_index calls")
    text = text.replace(old, new, 1)

    old = """def _build_shuffle_index(
    num_samples: int, total_size: int, numpy_random_state: numpy.random.RandomState
) -> numpy.ndarray:
"""
    new = """def _build_shuffle_index(
    num_samples: int,
    total_size: int,
    numpy_random_state: numpy.random.RandomState,
    shuffle: bool = True,
) -> numpy.ndarray:
"""
    if old not in text:
        raise SystemExit("could not find _build_shuffle_index signature")
    text = text.replace(old, new, 1)

    old = """    if total_size >= (numpy.iinfo(numpy.uint32).max - 1):
        dtype_ = numpy.int64

    shuffle_idx_first = numpy.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
"""
    new = """    if total_size >= (numpy.iinfo(numpy.uint32).max - 1):
        dtype_ = numpy.int64

    if not shuffle:
        return numpy.arange(start=0, stop=total_size, step=1, dtype=dtype_)

    shuffle_idx_first = numpy.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
"""
    if old not in text:
        raise SystemExit("could not find shuffle_index dtype block")
    text = text.replace(old, new, 1)

    if MARKER not in text:
        raise SystemExit("internal error: marker missing after patch")
    return text, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--megatron-dir", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="Only verify that the patch is present.")
    args = ap.parse_args()

    path = args.megatron_dir / "megatron/core/datasets/gpt_dataset.py"
    if not path.is_file():
        raise SystemExit(f"missing GPTDataset file: {path}")
    text = path.read_text()
    if args.check:
        if MARKER not in text:
            raise SystemExit(f"missing {MARKER} in {path}")
        print(f"OK no-shuffle patch present: {path}")
        return 0
    patched, changed = patch_text(text)
    if changed:
        backup = path.with_suffix(path.suffix + ".pre_curriculum_patch")
        if not backup.exists():
            backup.write_text(text)
        path.write_text(patched)
        print(f"patched {path}")
        print(f"backup {backup}")
    else:
        print(f"already patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
