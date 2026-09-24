#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared plumbing for the kernel-bundle STs: the .pyptokb TLV reader and the 4-symbol C ABI binding.

Deliberately free of top-level ``pypto`` / ``torch`` imports so the pytest driver can read a bundle's
segment table without dragging the whole frontend into the collector process.
Mirrors ``framework/src/machine/runtime/bundle/kernel_bundle_format.h`` and
``framework/include/tile_fwk_bundle/pypto_bundle_api.h``.
"""
import ctypes
import os
import struct

BUNDLE_MAGIC = b"PYPTOKB\0"
BUNDLE_VERSION = 1

# TlvType (kernel_bundle_format.h). A segment that exists but is empty is still written as a 0-length TLV,
# which is what distinguishes "absent" from "present but empty" -- the value-dependent ops rely on that.
TLV_AICORE_KERNEL = 1
TLV_AICPU_SO = 2
TLV_DEV_PROGRAM = 3
TLV_CTRL_FLOW_CACHE = 4
TLV_SYMBOL_META = 5

TLV_NAMES = {
    TLV_AICORE_KERNEL: "AicoreKernel",
    TLV_AICPU_SO: "AicpuSo",
    TLV_DEV_PROGRAM: "DevProgram",
    TLV_CTRL_FLOW_CACHE: "CtrlFlowCache",
    TLV_SYMBOL_META: "SymbolMeta",
}

_HEADER_SIZE = 64
_TLV_SIZE = 32


class BundleTlv:
    """One parsed TLV entry."""

    def __init__(self, type_id: int, offset: int, length: int, crc32: int):
        self.type_id = type_id
        self.offset = offset
        self.length = length
        self.crc32 = crc32

    @property
    def name(self) -> str:
        return TLV_NAMES.get(self.type_id, f"Unknown({self.type_id})")

    def __repr__(self) -> str:
        return f"<Tlv {self.name} offset={self.offset} length={self.length}>"


def read_bundle_tlvs(path: str) -> dict:
    """Parse the .pyptokb header + TLV table. Returns {type_id: BundleTlv}.

    Validates the container envelope only (magic / version / totalSize); the per-segment CRCs are the
    loader's job and are covered by the fact that the consume step actually loads the file.
    """
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        header = f.read(_HEADER_SIZE)
        if len(header) != _HEADER_SIZE:
            raise ValueError(f"{path}: truncated bundle header ({len(header)}B)")
        magic = header[:8]
        version, _flags, tlv_count, _header_size = struct.unpack_from("<IIII", header, 8)
        total_size, = struct.unpack_from("<Q", header, 24)
        if magic != BUNDLE_MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}, expected {BUNDLE_MAGIC!r}")
        if version != BUNDLE_VERSION:
            raise ValueError(f"{path}: bundle version {version}, expected {BUNDLE_VERSION}")
        if total_size != file_size:
            raise ValueError(f"{path}: header totalSize={total_size} but file is {file_size}B")

        tlvs = {}
        for _ in range(tlv_count):
            raw = f.read(_TLV_SIZE)
            if len(raw) != _TLV_SIZE:
                raise ValueError(f"{path}: truncated TLV table")
            type_id, _tlv_flags = struct.unpack_from("<II", raw, 0)
            offset, length = struct.unpack_from("<QQ", raw, 8)
            crc32, = struct.unpack_from("<I", raw, 24)
            if offset + length > file_size:
                raise ValueError(f"{path}: TLV {type_id} runs past EOF ({offset}+{length} > {file_size})")
            tlvs[type_id] = BundleTlv(type_id, offset, length, crc32)
    return tlvs


def read_symbol_meta(path: str) -> str:
    """Return the SymbolMeta segment as text ('' when the segment is absent or empty)."""
    tlv = read_bundle_tlvs(path).get(TLV_SYMBOL_META)
    if tlv is None or tlv.length == 0:
        return ""
    with open(path, "rb") as f:
        f.seek(tlv.offset)
        return f.read(tlv.length).decode("utf-8", "replace")


class PyptoTensorDesc(ctypes.Structure):
    """pypto_bundle_api.h PyptoTensorDesc."""

    _fields_ = [
        ("addr", ctypes.c_void_p),
        ("dataType", ctypes.c_int32),
        ("rank", ctypes.c_int32),
        ("shape", ctypes.c_int64 * 8),
    ]


def dtype_enum(torch_dtype) -> int:
    """Map a torch dtype onto the internal DataType enum value (tilefwk/data_type.h).

    Resolved through the pypto enum rather than hard-coded, so a reordering of DATA_TYPE_ALL cannot
    silently desync the ST from the runtime.
    """
    import torch

    import pypto

    mapping = {
        torch.float32: pypto.DT_FP32,
        torch.float16: pypto.DT_FP16,
        torch.bfloat16: pypto.DT_BF16,
        torch.int32: pypto.DT_INT32,
        torch.int64: pypto.DT_INT64,
        torch.int8: pypto.DT_INT8,
        torch.uint8: pypto.DT_UINT8,
        torch.bool: pypto.DT_BOOL,
    }
    if torch_dtype not in mapping:
        raise ValueError(f"no DataType mapping for {torch_dtype}")
    return int(mapping[torch_dtype])


def make_desc(tensor) -> PyptoTensorDesc:
    """Build a PyptoTensorDesc from a device torch tensor."""
    desc = PyptoTensorDesc()
    desc.addr = ctypes.c_void_p(tensor.data_ptr())
    desc.dataType = dtype_enum(tensor.dtype)
    desc.rank = tensor.dim()
    if tensor.dim() > len(desc.shape):
        raise ValueError(f"rank {tensor.dim()} exceeds the {len(desc.shape)}-dim PyptoTensorDesc")
    for i in range(tensor.dim()):
        desc.shape[i] = tensor.shape[i]
    return desc


def make_desc_array(tensors):
    """Build the unified operand array (inputs then outputs, nOut folded in) the ABI expects."""
    return (PyptoTensorDesc * len(tensors))(*[make_desc(t) for t in tensors])


def bundle_so_path() -> str:
    """Locate libtile_fwk_bundle.so. PYPTO_BUNDLE_SO overrides it (e.g. to point at the standalone form)."""
    override = os.environ.get("PYPTO_BUNDLE_SO")
    if override:
        return override
    import pypto

    return os.path.join(os.path.dirname(os.path.abspath(pypto.__file__)), "lib", "libtile_fwk_bundle.so")


# DT_NEEDED of the plain libtile_fwk_bundle.so. Production builds skip RPATH, so dlopen-by-abspath
# does not search the sibling directory; map these first (same order as python/pypto/_loader.py).
_BUNDLE_NEEDED = (
    "libc_sec.so",
    "libtile_fwk_utils.so",
    "libtile_fwk_adapter.so",
    "libtile_fwk_cann_host_runtime.so",
    "libtile_fwk_platform.so",
    "libtile_fwk_interface.so",
    "libtile_fwk_codegen.so",
    "libtile_fwk_compiler.so",
    "libtile_fwk_runtime.so",
)


def _preload_bundle_needed(so_path: str) -> None:
    so_dir = os.path.dirname(os.path.abspath(so_path))
    for name in _BUNDLE_NEEDED:
        path = os.path.join(so_dir, name)
        if os.path.isfile(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def load_bundle_lib():
    """dlopen the bundle ABI and declare its 4 exported symbols."""
    so_path = bundle_so_path()
    _preload_bundle_needed(so_path)
    lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    desc_p = ctypes.POINTER(PyptoTensorDesc)

    lib.PyptoWorkspace.restype = ctypes.c_uint64
    lib.PyptoWorkspace.argtypes = [ctypes.c_char_p, desc_p, ctypes.c_uint32]
    lib.PyptoLaunch.restype = ctypes.c_int
    lib.PyptoLaunch.argtypes = [
        ctypes.c_char_p, desc_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ]
    lib.PyptoWorkspaceFromMemory.restype = ctypes.c_uint64
    lib.PyptoWorkspaceFromMemory.argtypes = [ctypes.c_void_p, ctypes.c_uint64, desc_p, ctypes.c_uint32]
    lib.PyptoLaunchFromMemory.restype = ctypes.c_int
    lib.PyptoLaunchFromMemory.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, desc_p, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ]
    return lib


class BundleClient:
    """Thin facade over the two-call ABI, selecting the path or the in-memory variant."""

    def __init__(self, bundle_path: str, api: str = "path"):
        if api not in ("path", "memory"):
            raise ValueError(f"api must be 'path' or 'memory', got {api!r}")
        self.api = api
        self.lib = load_bundle_lib()
        self._path_c = bundle_path.encode()
        self._blob = None
        self._blob_ptr = None
        self._blob_len = 0
        if api == "memory":
            # The client already holds the image in its own buffer; the ABI never touches the disk path.
            with open(bundle_path, "rb") as f:
                self._blob = f.read()
            self._blob_ptr = ctypes.cast(ctypes.c_char_p(self._blob), ctypes.c_void_p)
            self._blob_len = len(self._blob)

    def workspace(self, descs) -> int:
        """Query the workspace byte count for these shapes. Pure host: does not init the device."""
        n = len(descs)
        if self.api == "path":
            return self.lib.PyptoWorkspace(self._path_c, descs, n)
        return self.lib.PyptoWorkspaceFromMemory(self._blob_ptr, self._blob_len, descs, n)

    def launch(self, descs, workspace_ptr, stream=None, sync=1) -> int:
        n = len(descs)
        if self.api == "path":
            return self.lib.PyptoLaunch(self._path_c, descs, n, workspace_ptr, stream, sync)
        return self.lib.PyptoLaunchFromMemory(
            self._blob_ptr, self._blob_len, descs, n, workspace_ptr, stream, sync)
