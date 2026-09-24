#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use this
# file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
# IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
# full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exception dump callback management for PyPTO Pro.

The C++ shim (``exception_dump_callback.cpp``) is compiled into
``libtile_fwk_interface.so`` at build time.  This module loads the
``extern "C"`` symbols from that library and provides a Python API to
register the callback and cache tensor info before each kernel launch.

On AICORE error, CANN asynchronously invokes the callback which writes
dump files (tensor data + ``_host.o``) to
``ASCEND_WORK_PATH/extra-info/data-dump/<device_id>/``.

The callback also automatically compiles the kernel with ``-g`` via bisheng
and places the debug ``.o`` in the dump directory, so that ``msnpureport``
can resolve source line numbers.  No manual intervention is needed — the
compile command is pre-constructed by :func:`set_dump_info` (called before
every kernel launch) and cached in the C++ shim for the callback to execute.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
import shlex
import shutil

import torch

from pypto_pro import DataType

_PL_DTYPE_TO_ACL_DTYPE: dict[str, int] = {
    str(DataType.FP32): 0,    # ACL_FLOAT
    str(DataType.FP16): 1,    # ACL_FLOAT16
    str(DataType.INT8): 2,    # ACL_INT8
    str(DataType.INT32): 3,   # ACL_INT32
    str(DataType.UINT8): 4,   # ACL_UINT8
    str(DataType.INT16): 6,   # ACL_INT16
    str(DataType.UINT16): 7,  # ACL_UINT16
    str(DataType.UINT32): 8,  # ACL_UINT32
    str(DataType.INT64): 9,   # ACL_INT64
    str(DataType.UINT64): 10, # ACL_UINT64
    str(DataType.BOOL): 12,   # ACL_BOOL
    str(DataType.BF16): 27,   # ACL_BF16
    str(DataType.FP8E5M2): 29,    # ACL_FP8_E5M2
    str(DataType.FP8E4M3FN): 30,  # ACL_FP8_E4M3FN
}

_MAX_DIMS = 8

_lib: ctypes.CDLL | None = None
_registered: bool = False
_cached_build_dir: str = ""
_cached_kernel_name: str = ""
_cached_has_cube: bool = False
_cached_has_vector: bool = True


def _ensure_lib() -> ctypes.CDLL:
    """Load libtile_fwk_interface.so and resolve the extern "C" symbols."""
    global _lib
    if _lib is not None:
        return _lib

    import pypto

    lib_dir = Path(pypto.__file__).parent.resolve() / "lib"
    so_path = lib_dir / "libtile_fwk_interface.so"
    if not so_path.exists():
        raise RuntimeError(f"libtile_fwk_interface.so not found at {so_path}")

    _lib = ctypes.CDLL(str(so_path))

    _lib.pro_register_exception_dump_callback.restype = ctypes.c_int32
    _lib.pro_register_exception_dump_callback.argtypes = []

    _lib.pro_set_dump_info.restype = None
    _lib.pro_set_dump_info.argtypes = [
        ctypes.c_char_p,                   # kernelName
        ctypes.c_int32,                    # numTensors
        ctypes.POINTER(ctypes.c_int32),    # types
        ctypes.POINTER(ctypes.c_size_t),   # tensorSizes
        ctypes.POINTER(ctypes.c_int32),    # dataTypes
        ctypes.POINTER(ctypes.c_void_p),   # tensorAddrs
        ctypes.POINTER(ctypes.c_int64),    # flatShapes
        ctypes.POINTER(ctypes.c_int32),    # shapeCounts
        ctypes.c_int32,                    # maxDims
    ]

    _lib.pro_clear_dump_info.restype = None
    _lib.pro_clear_dump_info.argtypes = []

    _lib.pro_set_debug_cmd.restype = None
    _lib.pro_set_debug_cmd.argtypes = [ctypes.c_char_p]
    return _lib


def register_callback() -> bool:
    """Register the exception dump callback with CANN. Returns True on success."""
    global _registered
    if _registered:
        return True
    try:
        lib = _ensure_lib()
    except RuntimeError as exc:
        logging.warning("Exception dump callback unavailable: %s", exc)
        return False
    ret = lib.pro_register_exception_dump_callback()
    if ret == 0:
        _registered = True
        logging.debug("Exception dump callback registered successfully")
    else:
        logging.warning("Failed to register exception dump callback (ret=%d)", ret)
    return _registered


def _build_debug_compile_cmd(build_dir: str, kernel_name: str,
                             has_cube: bool, has_vector: bool) -> str:
    """Pre-construct the bisheng -g compile command for the C++ callback to execute.

    Returns a shell command string that:
      1. mkdir -p the dump directory
      2. compiles kernel.cpp with -g via bisheng
      3. outputs <dump_dir>/<kernel_name>_debug.o

    Returns empty string if any prerequisite is missing (bisheng not found,
    kernel.cpp missing, env vars unset, etc.) so the callback skips compilation.
    """
    if not build_dir:
        return ""
    kernel_cpp = os.path.join(build_dir, "kernel.cpp")
    if not os.path.isfile(kernel_cpp):
        return ""

    work_path = os.environ.get("ASCEND_WORK_PATH", "")
    if not work_path:
        return ""

    device_id = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    dump_dir = os.path.join(work_path, "extra-info", "data-dump", device_id)
    debug_o_path = os.path.join(dump_dir, f"{kernel_name}_debug.o")

    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_home:
        return ""

    toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME", ascend_home)

    bisheng = shutil.which("bisheng")
    if bisheng is None:
        return ""

    from pypto_pro.runtime.compile_config import get_jit_compile_config
    from pypto_pro.runtime.jit import _build_llvm_args, _normalize_arch

    arch = _normalize_arch(os.environ.get("PYPTOPRO_JIT_ARCH", "a5"))
    cfg = get_jit_compile_config()

    npu_arch = cfg._resolve_npu_arch(arch, has_cube=has_cube, has_vec=has_vector)
    mem_arch = cfg._resolve_memory_arch_flag(arch)
    variables = {
        "toolkit_home": toolkit_home,
        "mem_arch": mem_arch,
        "npu_arch": npu_arch,
    }
    common = cfg._format_values(cfg.common_flags, variables)
    arch_flags = cfg._format_values(cfg.arch_flags, variables)
    include_flags = cfg.runtime_include_flags(ascend_home)
    llvm_args = _build_llvm_args(arch)

    cmd_parts = [
        shlex.quote(bisheng),
        *[shlex.quote(f) for f in arch_flags],
        *[shlex.quote(f) for f in common],
        "-g",
        "-xcce",
        *[shlex.quote(f) for f in include_flags],
        *[shlex.quote(f) for f in llvm_args],
        "-c",
        shlex.quote(kernel_cpp),
        "-o",
        shlex.quote(debug_o_path),
    ]
    return f"mkdir -p {shlex.quote(dump_dir)} && {' '.join(cmd_parts)}"


def set_dump_info(kernel_name: str, args: tuple, param_specs: list, build_dir: str = "",
                  has_cube: bool = False, has_vector: bool = True) -> None:
    """Cache tensor info from runtime args and param_specs before kernel launch."""
    global _cached_build_dir, _cached_kernel_name, _cached_has_cube, _cached_has_vector
    if build_dir:
        _cached_build_dir = build_dir
    _cached_kernel_name = kernel_name
    _cached_has_cube = has_cube
    _cached_has_vector = has_vector

    if not _cached_build_dir or not os.path.isdir(_cached_build_dir):
        logging.warning("exception dump skipped: not a jit scenario (build_dir unavailable)")
        return

    if not register_callback():
        logging.warning("register_callback failed, exception dump will not be triggered")
        return

    types: list[int] = []
    sizes: list[int] = []
    dtypes: list[int] = []
    addrs: list[int] = []
    flat_shapes: list[int] = []
    shape_counts: list[int] = []

    for arg, spec in zip(args, param_specs):
        kind_name = spec.kind.name
        if kind_name in ("TENSOR", "PTR"):
            if arg is None:
                continue
            if not isinstance(arg, torch.Tensor):
                continue
            types.append(0)  # INPUT
            sizes.append(arg.numel() * arg.element_size())
            acl_dtype = _PL_DTYPE_TO_ACL_DTYPE.get(str(spec.dtype), 0)
            dtypes.append(acl_dtype)
            addrs.append(arg.data_ptr())
            shape = list(arg.shape)
            shape_counts.append(len(shape))
            padded = shape + [0] * (_MAX_DIMS - len(shape))
            flat_shapes.extend(padded[:_MAX_DIMS])
        elif kind_name == "TILING":
            if arg is None or not isinstance(arg, torch.Tensor):
                continue
            types.append(2)  # WORKSPACE
            sizes.append(arg.numel() * arg.element_size())
            dtypes.append(4)  # ACL_UINT8
            addrs.append(arg.data_ptr())
            shape_counts.append(0)
            flat_shapes.extend([0] * _MAX_DIMS)

    num_tensors = len(types)

    lib = _ensure_lib()

    debug_cmd = _build_debug_compile_cmd(
        _cached_build_dir, _cached_kernel_name, _cached_has_cube, _cached_has_vector
    )
    lib.pro_set_debug_cmd(debug_cmd.encode("utf-8"))

    if num_tensors == 0:
        return

    c_types = (ctypes.c_int32 * num_tensors)(*types)
    c_sizes = (ctypes.c_size_t * num_tensors)(*sizes)
    c_dtypes = (ctypes.c_int32 * num_tensors)(*dtypes)
    c_addrs = (ctypes.c_void_p * num_tensors)(*addrs)
    c_shapes = (ctypes.c_int64 * (num_tensors * _MAX_DIMS))(*flat_shapes)
    c_counts = (ctypes.c_int32 * num_tensors)(*shape_counts)

    lib.pro_set_dump_info(
        kernel_name.encode("utf-8"),
        num_tensors,
        c_types,
        c_sizes,
        c_dtypes,
        c_addrs,
        c_shapes,
        c_counts,
        _MAX_DIMS,
    )


def clear_dump_info() -> None:
    """Clear cached tensor info."""
    global _lib
    if _lib is not None:
        _lib.pro_clear_dump_info()
