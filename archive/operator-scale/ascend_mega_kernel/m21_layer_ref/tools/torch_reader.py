"""Torch-facing wrapper over the verbatim upstream shard reader.

Why this file exists instead of editing the reader: `tools/safetensors_reader.py`
in this package is a **byte-identical copy** of the repo's
`tools/weights/safetensors_reader.py` (M8, agent-weights). Keeping it verbatim
means (a) the "diff vs upstream" table required by docs/17 §3.4 is trivially
exact ("identical, sha256 pinned"), and (b) upstream fixes to the
partially-downloaded-shard handling keep flowing in by re-copying the file.

The M39 harness needs torch tensors rather than numpy views, so the conversion
lives here, as a subclass, with nothing else changed:

    numpy view (bf16 as uint16 bit patterns)  ->  torch view (torch.bfloat16)

The bf16 conversion is a **bit reinterpretation** (`torch.view(torch.bfloat16)`),
not a cast, so no rounding is introduced anywhere in the load path.

`selfcheck.py` section J asserts the sha256 of the copied reader, and also flags
it if the upstream file has moved on.
"""

from __future__ import annotations

import warnings

import numpy as np

from safetensors_reader import ShardReader, TensorInfo  # noqa: F401  (re-export)

_NP_TO_TORCH = {
    "uint8": "uint8",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
}


def _from_numpy(arr: np.ndarray):
    """torch view of a read-only mmap view.

    The array is not writable, which torch warns about. The guarantee is really
    at the OS level: the reader maps shards with ``mmap.ACCESS_READ``, so an
    in-place write would fail rather than corrupt the checkpoint. M39 only ever
    reads these tensors (`@`, `.to()`, `.float()`); nothing is written back.
    """
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable"
        )
        return torch.from_numpy(np.ascontiguousarray(arr))


class TorchShardReader(ShardReader):
    """`ShardReader` + torch tensor loading (bf16 kept exact)."""

    def load_torch(self, name: str, *slices, device: str = "cpu"):
        """Zero-copy torch tensor; BF16 is a bit reinterpretation of the bytes."""
        import torch

        arr = self.load(name, *slices)
        t = _from_numpy(arr)
        if self.info(name).is_bf16:
            return t.view(torch.bfloat16).to(device)
        return t.to(getattr(torch, _NP_TO_TORCH[arr.dtype.name])).to(device)

    def load_torch_f32(self, name: str, *slices, device: str = "cpu"):
        """Flip any float/BF16 tensor to fp32 (BF16 via exact upcast)."""
        import torch

        arr = self.load_f32(name, *slices)
        return _from_numpy(arr).to(torch.float32).to(device)

    def file_size(self, name: str) -> int:
        return self.info(name).nbytes
