#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Delivery smoke test for the SELF-CONTAINED kernel-bundle library (libtile_fwk_bundle_standalone.so).

Stands in for a downstream that only consumes .pyptokb files: it loads ONE .so by absolute path, next to ONE
.pyptokb, and proves nothing else from the pypto install tree is needed.

    python3 standalone_delivery_smoke.py <standalone.so> <bundle.pyptokb>

Driven by ``test_kernel_bundle.py::test_bundle_standalone_delivery``, which assembles the delivery directory
first; runnable by hand too (see bundle/E2E_MANUAL.md section 5).

TWO RULES THIS FILE MUST KEEP -- both are constraints, not style:

1. **Never import pypto, and never reuse st/bundle/bundle_abi.py.** The standalone library statically absorbs
   the whole pypto closure, so loading it into a process that also holds the regular pypto stack gives every
   Meyers singleton on that closure (Platform::Instance, DeviceRunner::Get, ConfigManager::Instance,
   OpInfoManager, plus the bundle's own registry and dev-cache) two independent copies -- silent state
   divergence, not a crash. See bundle/CMakeLists.txt and BUNDLE_FLOW.md section 7.2. That is why the
   PyptoTensorDesc / ABI boilerplate below is duplicated from bundle_abi.py rather than imported: bundle_abi
   resolves the DataType enum through ``import pypto``. Do not DRY these two together.

2. **The .so needs a configs/ directory beside it.** GetPyptoLibPath() (utils/file_utils.cpp) uses dladdr to
   find the directory of its OWN .so, and ConfigManager plus the platform ini parsing read configs/ relative
   to it. Missing it reports ErrCode F29002 / FeError::INVALID_FILE. This is a data-file dependency; static
   linking cannot absorb it.
"""
import ctypes
import os
import sys

import torch
import torch_npu  # noqa: F401  -- registers the npu backend


class PyptoTensorDesc(ctypes.Structure):
    """pypto_bundle_api.h PyptoTensorDesc. Duplicated on purpose -- see rule 1 in the module docstring."""

    _fields_ = [
        ("addr", ctypes.c_void_p),
        ("dataType", ctypes.c_int32),
        ("rank", ctypes.c_int32),
        ("shape", ctypes.c_int64 * 8),
    ]


# DataType enum index for DT_FP32, i.e. the position of DT_FP32 in DATA_TYPE_ALL (tilefwk/data_type.h):
# DT_INT4=0 INT8=1 INT16=2 INT32=3 INT64=4 FP8=5 FP16=6 FP32=7 BF16=8 ...
# Hard-coded because rule 1 forbids importing pypto to resolve it; overridable if the enum is ever reordered.
# Today the device compute path reads only address/shape from DevTensorData, but the exception dump and the
# profiling path do read dataType (device_exception_dump.cpp, aicore_dump.h, host_prof.cpp) and size tensors
# with BitsOf(dataType)/8 -- a wrong value there silently corrupts the dump rather than the result.
DT_FP32 = int(os.environ.get("PYPTO_ST_DT_FP32", "7"))

# The op the reference bundle was packed with: examples/00_hello_world style elementwise add.
ADD_SHAPE = (1, 4, 1, 64)


def make_desc(tensor, dtype_val: int) -> PyptoTensorDesc:
    desc = PyptoTensorDesc()
    desc.addr = ctypes.c_void_p(tensor.data_ptr())
    desc.dataType = dtype_val
    desc.rank = tensor.dim()
    for i in range(tensor.dim()):
        desc.shape[i] = tensor.shape[i]
    return desc


def _exit(code: int):
    # os._exit skips the static destructors (device teardown races torch_npu), but it also skips flushing
    # stdout, which is block-buffered whenever output is redirected. Flush explicitly or the log is lost.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def assert_self_contained(so_path):
    """Fail if any pypto shared object got mapped: that would mean the .so was not actually self-contained.

    This is the property the whole flavour exists for, and it is invisible in the numeric result -- a build
    that regressed to pulling in libtile_fwk_runtime.so would still compute the right answer here.

    The library under test is excluded by name: it is itself called libtile_fwk_bundle_standalone.so, so it
    matches the very prefix this check scans for. Everything else starting with those prefixes is a leak.
    """
    with open("/proc/self/maps", "r") as f:
        mapped = f.read()
    under_test = os.path.basename(so_path)
    leaked = sorted({
        os.path.basename(line.split()[-1])
        for line in mapped.splitlines()
        if line.split()
        and os.path.basename(line.split()[-1]).startswith(("libtile_fwk", "libpypto_ctrl"))
        and os.path.basename(line.split()[-1]) != under_test
    })
    if leaked:
        print(f"[ST] FAILED: pypto libraries pulled into the process: {leaked}")
        _exit(1)
    print("[ST] self-containment: no pypto shared objects mapped besides the library under test")


def main():
    if len(sys.argv) < 3:
        print(f"usage: {os.path.basename(sys.argv[0])} <standalone.so> <bundle.pyptokb>")
        _exit(2)
    so_path, bundle_path = sys.argv[1], sys.argv[2]
    dev_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(dev_id)

    configs_dir = os.path.join(os.path.dirname(os.path.abspath(so_path)), "configs")
    if not os.path.isdir(configs_dir):
        print(f"[ST] FAILED: no configs/ beside {so_path} -- see rule 2 in this file's docstring")
        _exit(1)

    lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    lib.PyptoWorkspace.restype = ctypes.c_uint64
    lib.PyptoWorkspace.argtypes = [ctypes.c_char_p, ctypes.POINTER(PyptoTensorDesc), ctypes.c_uint32]
    lib.PyptoLaunch.restype = ctypes.c_int
    lib.PyptoLaunch.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(PyptoTensorDesc), ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ]
    print(f"[ST] loaded {so_path}")
    assert_self_contained(so_path)

    x = torch.rand(ADD_SHAPE, dtype=torch.float32, device=f"npu:{dev_id}")
    y = torch.rand(ADD_SHAPE, dtype=torch.float32, device=f"npu:{dev_id}")
    out = torch.empty(ADD_SHAPE, dtype=torch.float32, device=f"npu:{dev_id}")
    tensors = (PyptoTensorDesc * 3)(make_desc(x, DT_FP32), make_desc(y, DT_FP32), make_desc(out, DT_FP32))

    path_c = bundle_path.encode()
    ws_size = lib.PyptoWorkspace(path_c, tensors, 3)
    print(f"[ST] workspace size = {ws_size}")
    ws_keep = None
    ws_ptr = None
    if ws_size > 0:
        ws_keep = torch.empty(ws_size, dtype=torch.uint8, device=f"npu:{dev_id}")
        ws_ptr = ctypes.c_void_p(ws_keep.data_ptr())

    rc = lib.PyptoLaunch(path_c, tensors, 3, ws_ptr, None, 1)
    if rc != 0:
        print(f"[ST] FAILED: PyptoLaunch rc={rc}")
        _exit(1)
    torch.npu.synchronize()
    del ws_keep

    diff = (out - (x + y)).abs().max().item()
    print(f"[ST] max diff = {diff:.6f}")
    if diff > 1e-5:
        print("[ST] FAILED: numeric mismatch")
        _exit(1)
    print("✓ self-contained kernel-bundle delivery ST passed (one .so + one .pyptokb, no pypto install)")
    _exit(0)


if __name__ == "__main__":
    main()
