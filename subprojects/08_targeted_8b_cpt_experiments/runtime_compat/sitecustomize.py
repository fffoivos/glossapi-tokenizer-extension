"""Preserve pinned SwissAI Megatron checkpoint semantics under PyTorch 2.9.1."""

import dataclasses
import os
import pickle
from pathlib import Path

import numpy as np

if not hasattr(np, "product"):
    np.product = np.prod


def _install_dcp_metadata_preservation() -> None:
    try:
        from torch.distributed.checkpoint import FileSystemWriter
    except ImportError:
        return

    original = FileSystemWriter.finish
    if getattr(original, "_apertus_preserves_dynamic_metadata", False):
        return

    def finish_preserving_dynamic_metadata(self, metadata, results):
        declared = {field.name for field in dataclasses.fields(metadata)}
        dynamic = {
            key: value for key, value in vars(metadata).items() if key not in declared
        }
        original(self, metadata, results)
        if not dynamic:
            return

        if not getattr(self, "use_collectives", True) and self.rank is not None:
            metadata_path = Path(self._get_metadata_path(self.rank))
        else:
            metadata_path = Path(self._get_metadata_path())
        with metadata_path.open("rb") as stream:
            completed = pickle.load(stream)
        for key, value in dynamic.items():
            setattr(completed, key, value)

        temporary = metadata_path.with_name(metadata_path.name + ".apertus.tmp")
        with temporary.open("wb") as stream:
            pickle.dump(completed, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, metadata_path)

    finish_preserving_dynamic_metadata._apertus_preserves_dynamic_metadata = True
    FileSystemWriter.finish = finish_preserving_dynamic_metadata


_install_dcp_metadata_preservation()
