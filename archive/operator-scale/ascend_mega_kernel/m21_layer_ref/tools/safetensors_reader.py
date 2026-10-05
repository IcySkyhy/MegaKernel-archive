"""Minimal pure-numpy memmap reader for HF safetensors shards.

No torch, no safetensors package, no index json required: a ``ShardReader``
scans a directory for ``*.safetensors`` files, parses each file's
(8-byte LE header length + JSON header) and builds a
``tensor name -> (file, dtype, shape, data offsets)`` index directly from
the shard headers. Works on partially downloaded model directories: a
shard that is still growing is truncated to the tensors whose byte ranges
have already landed, so slices from complete shards stay readable while
the rest of the download continues.

File format (https://huggingface.co/docs/safetensors):
    [8 bytes: uint64 LE header length N][N bytes: UTF-8 JSON header]
    [payload: raw little-endian tensor bytes]

The JSON header maps each tensor name to
``{"dtype": "BF16"|"F16"|"F32"|..., "shape": [...], "data_offsets": [beg, end]}``
where the offsets are relative to the start of the payload.

numpy has no native bfloat16, so BF16 tensors are viewed as uint16 bit
patterns; use ``bf16_bits_to_f32`` (same RNE-free upcast as
tools/golden/moe_block_ref.py) to convert to float32. F16/F32/etc. map to
native numpy dtypes.

Reads go through ``np.memmap``/buffer views, so slicing rows out of a
multi-GB tensor (e.g. 8 experts out of a [512, ...] MoE weight) touches
only the requested pages.

Usage:
    from safetensors_reader import ShardReader
    rdr = ShardReader("/path/to/model-dir")
    rdr.names()                       # all indexed tensor names
    info = rdr.info("model...mlp.experts.gate_up_proj")
    w = rdr.load("model...mlp.experts.gate_up_proj", rows=slice(0, 8))
    w32 = rdr.load_f32("model...mlp.gate.weight")   # bf16 upcast to f32

Pure numpy (2.5.1). Python 3.12.
"""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# safetensors dtype name -> numpy dtype (BF16 handled separately as bits)
_ST_TO_NP = {
    "BOOL": np.bool_,
    "U8": np.uint8,
    "I8": np.int8,
    "I16": np.int16,
    "U16": np.uint16,
    "F16": np.float16,
    "I32": np.int32,
    "U32": np.uint32,
    "F32": np.float32,
    "I64": np.int64,
    "U64": np.uint64,
    "F64": np.float64,
}

BYTES_PER_ELEMENT = {
    "BF16": 2,
    **{name: np.dtype(dt).itemsize for name, dt in _ST_TO_NP.items()},
}

_HEADER_LEN_FMT = struct.Struct("<Q")


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """uint16 bf16 bit patterns -> float32 (exact upcast, no rounding)."""
    b = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return b.view(np.float32)


@dataclass(frozen=True)
class TensorInfo:
    """Location and layout of one tensor inside one shard."""

    name: str
    file: Path
    st_dtype: str
    shape: tuple[int, ...]
    data_begin: int   # absolute file offset of first payload byte
    data_end: int     # absolute file offset one past last payload byte

    @property
    def nbytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n * BYTES_PER_ELEMENT[self.st_dtype]

    @property
    def is_bf16(self) -> bool:
        return self.st_dtype == "BF16"


@dataclass
class _Shard:
    path: Path
    mm: mmap.mmap
    payload_base: int          # absolute offset of payload start
    available: int             # payload bytes readable (file may be incomplete)
    tensors: dict[str, dict]   # name -> raw header entry


class SafetensorsFile:
    """One parsed shard. Tensors exposed as zero-copy views on the mmap."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            raw_len = f.read(_HEADER_LEN_FMT.size)
            if len(raw_len) != _HEADER_LEN_FMT.size:
                raise ValueError(f"{self.path}: truncated header length")
            (header_len,) = _HEADER_LEN_FMT.unpack(raw_len)
            header = f.read(header_len)
            if len(header) != header_len:
                raise ValueError(f"{self.path}: truncated json header")
            self._header = json.loads(header)
            payload_base = _HEADER_LEN_FMT.size + header_len
            fsize = self.path.stat().st_size
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self.payload_base = payload_base
        # payload bytes actually backed by the file (download may be partial)
        self.available = max(0, fsize - payload_base)
        self._tensors = {
            name: meta for name, meta in self._header.items()
            if name != "__metadata__"
        }

    def complete(self, name: str) -> bool:
        """True if the tensor's full byte range is inside the file."""
        meta = self._tensors[name]
        return meta["data_offsets"][1] <= self.available

    def tensor_names(self, complete_only: bool = False) -> list[str]:
        if not complete_only:
            return sorted(self._tensors)
        return sorted(n for n in self._tensors if self.complete(n))

    def info(self, name: str) -> TensorInfo:
        meta = self._tensors[name]
        beg, end = meta["data_offsets"]
        return TensorInfo(
            name=name,
            file=self.path,
            st_dtype=meta["dtype"],
            shape=tuple(meta["shape"]),
            data_begin=self.payload_base + beg,
            data_end=self.payload_base + end,
        )

    def _view(self, info: TensorInfo) -> np.ndarray:
        count = info.data_end - info.data_begin
        buf = np.ndarray(
            count, dtype=np.uint8,
            buffer=self.mm, offset=info.data_begin,
        )
        if info.is_bf16:
            return buf.view(np.uint16).reshape(info.shape)
        np_dt = np.dtype(_ST_TO_NP[info.st_dtype]).newbyteorder("<")
        return buf.view(np_dt).reshape(info.shape)

    def load(self, name: str, *slices) -> np.ndarray:
        """Zero-copy view of ``name``, optionally subscripted per dimension.

        ``slices`` are numpy-style indices applied left to right, e.g.
        ``load("...gate_up_proj", slice(0, 8))`` on a [512, 1280, 1280]
        tensor returns a [8, 1280, 1280] view. No slice = full tensor.
        Int indexing drops the dimension, exactly like ndarray indexing.
        """
        info = self.info(name)
        if not self.complete(name):
            raise ValueError(
                f"{self.path}: tensor {name!r} is beyond the current end of "
                f"this (still downloading?) shard"
            )
        arr = self._view(info)
        if slices:
            arr = arr[slices]
        return arr

    def load_f32(self, name: str, *slices) -> np.ndarray:
        """Like :meth:`load`, but BF16/F16 are upcast to float32 values."""
        arr = self.load(name, *slices)
        if arr.dtype == np.uint16:  # bf16 bit patterns
            return bf16_bits_to_f32(arr)
        if arr.dtype == np.float16:
            return arr.astype(np.float32)
        return arr

    def close(self) -> None:
        self.mm.close()


class ShardReader:
    """Index over every ``*.safetensors`` shard in a model directory."""

    def __init__(self, model_dir: Path | str):
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(self.model_dir)
        shards = sorted(self.model_dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(
                f"no *.safetensors shards under {self.model_dir}"
            )
        self._shards: dict[Path, SafetensorsFile] = {}
        self._index: dict[str, tuple[SafetensorsFile, bool]] = {}
        self.incomplete: dict[str, list[str]] = {}
        for path in shards:
            st = SafetensorsFile(path)
            self._shards[path] = st
            for name in st.tensor_names():
                if name in self._index:
                    raise ValueError(
                        f"tensor {name!r} present in both "
                        f"{self._index[name][0].path} and {path}"
                    )
                self._index[name] = (st, st.complete(name))
            missing = [n for n in st.tensor_names() if not st.complete(n)]
            if missing:
                self.incomplete[str(path)] = missing

    # -- introspection ------------------------------------------------------
    def names(self, complete_only: bool = False) -> list[str]:
        """All tensor names (complete ones only if requested)."""
        return sorted(
            n for n, (_, ok) in self._index.items() if ok or not complete_only
        )

    def info(self, name: str) -> TensorInfo:
        st, ok = self._index[name]
        return st.info(name)

    def is_complete(self, name: str) -> bool:
        return self._index[name][1]

    def find(self, prefix: str, complete_only: bool = True) -> list[str]:
        """Names starting with ``prefix`` (e.g. a layer's mlp scope)."""
        return [n for n in self.names(complete_only) if n.startswith(prefix)]

    # -- reads ---------------------------------------------------------------
    def load(self, name: str, *slices) -> np.ndarray:
        """Zero-copy (sub)tensor view; bf16 comes back as uint16 bits."""
        st, _ = self._index[name]
        return st.load(name, *slices)

    def load_f32(self, name: str, *slices) -> np.ndarray:
        """Like :meth:`load`, upcasting BF16/F16 payloads to float32."""
        st, _ = self._index[name]
        return st.load_f32(name, *slices)

    def close(self) -> None:
        for st in self._shards.values():
            st.close()

    def __enter__(self) -> "ShardReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
