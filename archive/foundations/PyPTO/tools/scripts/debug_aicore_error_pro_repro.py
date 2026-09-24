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

r"""PyPTO Pro AICORE Error 离线复现工具。

从异常 dump 的 extra-info/data-dump 目录中读取 tensor 数据和编译产物，
构建单算子复现脚本并执行，验证 AICORE error 是否重现。

用法示例
--------
python debug_aicore_error_pro_repro.py -p <work_dir> [-d <device_id>]
"""

import argparse
import logging
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")


_STR_TO_TORCH = {
    "float16": "torch.float16", "float32": "torch.float32", "float64": "torch.float64",
    "int8": "torch.int8", "int16": "torch.int16", "int32": "torch.int32", "int64": "torch.int64",
    "uint8": "torch.uint8", "uint16": "torch.int16", "uint32": "torch.int32", "uint64": "torch.int64",
    "bfloat16": "torch.bfloat16", "bool": "torch.bool",
    "fp8e4m3fn": "torch.float8_e4m3fn", "fp8e5m2": "torch.float8_e5m2",
}

_STR_TO_NP = {
    "float16": "np.float16", "float32": "np.float32", "float64": "np.float64",
    "int8": "np.int8", "int16": "np.int16", "int32": "np.int32", "int64": "np.int64",
    "uint8": "np.uint8", "uint16": "np.int16", "uint32": "np.int32", "uint64": "np.int64",
    "bfloat16": "np.int16", "bool": "np.bool_",
    "fp8e4m3fn": "np.uint8", "fp8e5m2": "np.uint8",
}

_ACL_TO_STR_DTYPE = {
    0: "float32", 1: "float16", 2: "int8", 3: "int32", 4: "uint8",
    6: "int16", 7: "uint16", 8: "uint32", 9: "int64", 10: "uint64",
    11: "float64", 12: "bool", 27: "bfloat16",
    29: "fp8e5m2", 30: "fp8e4m3fn",
}

_DTYPE_TO_SIZE = {
    "float16": 2, "float32": 4, "float64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8,
    "bfloat16": 2, "bool": 1,
    "fp8e4m3fn": 1, "fp8e5m2": 1,
}

_DT_NAME_TO_STR = {
    "FLOAT": "float32", "FLOAT16": "float16", "FLOAT64": "float64",
    "INT8": "int8", "INT16": "int16", "INT32": "int32", "INT64": "int64",
    "UINT8": "uint8", "UINT16": "uint16", "UINT32": "uint32", "UINT64": "uint64",
    "BOOL": "bool", "BFLOAT16": "bfloat16",
    "FP8E5M2": "fp8e5m2", "FP8E4M3FN": "fp8e4m3fn",
}

_SIZE_TO_DTYPE = {1: "uint8", 2: "float16", 4: "float32", 8: "float64"}


class TensorInfo:
    __slots__ = ("addr", "size", "dtype_str", "shape", "io_type", "index")

    def __init__(self, addr: int, size: int, dtype_str: str, shape: tuple, io_type: str, index: int):
        self.addr = addr
        self.size = size
        self.dtype_str = dtype_str
        self.shape = shape
        self.io_type = io_type
        self.index = index


def _read_varint(data: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return result, pos


def _parse_protobuf_header(data: bytes) -> Dict[int, Dict]:
    """Parse the protobuf binary header of a dump file.

    The header starts with an 8-byte uint64 total_size, followed by repeated
    protobuf messages (field 4, length-delimited), each describing one tensor:
      field 1 (varint): dataType (ACL dtype + 1)
      field 3 (length-delimited): shape wrapper containing packed int64 dims
      field 5 (varint): tensorSize in bytes
      field 10 (varint): tensor index (0 omitted as protobuf default)

    Returns a dict mapping tensor index -> {dtype_acl, shape, size}.
    """
    if len(data) < 8:
        return {}

    pos = 8

    text_start = data.find(b"[Dump]")
    if text_start == -1:
        text_start = len(data)

    result: Dict[int, Dict] = {}
    while pos < text_start:
        try:
            tag, pos = _read_varint(data, pos)
            wire_type = tag & 0x7
            if wire_type != 2:
                break
            length, pos = _read_varint(data, pos)
            if pos + length > text_start:
                break
            entry = data[pos:pos + length]
            pos += length

            info: Dict = {"dtype_acl": None, "shape": (), "size": 0, "index": 0}
            ep = 0
            while ep < len(entry):
                ft, ep = _read_varint(entry, ep)
                fn, wt = ft >> 3, ft & 0x7
                if wt == 0:
                    v, ep = _read_varint(entry, ep)
                    if fn == 1:
                        info["dtype_acl"] = v - 1
                    elif fn == 5:
                        info["size"] = v
                    elif fn == 10:
                        info["index"] = v
                elif wt == 2:
                    ln, ep = _read_varint(entry, ep)
                    raw = entry[ep:ep + ln]
                    ep += ln
                    if fn == 3:
                        sp = 0
                        dims: List[int] = []
                        while sp < len(raw):
                            st, sp = _read_varint(raw, sp)
                            swt = st & 0x7
                            if swt == 2:
                                sl, sp = _read_varint(raw, sp)
                                sraw = raw[sp:sp + sl]
                                sp += sl
                                pp = 0
                                while pp < len(sraw):
                                    dv, pp = _read_varint(sraw, pp)
                                    dims.append(dv)
                            elif swt == 0:
                                sv, sp = _read_varint(raw, sp)
                                dims.append(sv)
                        info["shape"] = tuple(dims)
                else:
                    break
            result[info["index"]] = info
        except (IndexError, ValueError):
            break

    return result


def _get_device_id(cli_device_id: Optional[str] = None) -> int:
    if cli_device_id is not None:
        return int(cli_device_id)
    return int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))


def _find_data_dump_dir(work_dir: str) -> Tuple[str, int]:
    """在 work_dir/extra-info/data-dump/ 下找到包含 dump 文件的 device 目录。"""
    base = os.path.join(work_dir, "extra-info", "data-dump")
    if not os.path.isdir(base):
        raise RuntimeError(f"data-dump directory not found: {base}")
    for dev_dir in sorted(os.listdir(base)):
        full = os.path.join(base, dev_dir)
        if os.path.isdir(full) and os.listdir(full):
            return full, int(dev_dir)
    raise RuntimeError(f"No dump files found under {base}")


def _parse_dump_file(dump_file: str) -> Tuple[str, List[TensorInfo]]:
    """从 dump 文件中解析 kernel name 和 tensor 信息。

    dump 文件格式:
    - 8 字节 uint64 total_size
    - protobuf 二进制头: 每个 tensor 的 dataType / shape / size / index
    - 文本元数据: [Dump][Exception] 行，记录 type / index / shape / dtype / address / size
    - 纯 tensor 数据（按 index 顺序连续排列）
    """
    with open(dump_file, "rb") as f:
        content = f.read()

    text = content.decode("utf-8", errors="replace")

    tensors: List[TensorInfo] = []
    for m in re.finditer(
        r"type=(\w+);\s*index=(\d+);\s*shape=\[([\d,]*)\];\s*format=\w+;\s*dtype=DT_(\w+);\s*address=(0x[0-9a-fA-F]+);\s*size=(\d+)\s*bytes",
        text
    ):
        io_type = m.group(1)
        index = int(m.group(2))
        shape_str = m.group(3)
        shape = tuple(int(x) for x in shape_str.split(",")) if shape_str else ()
        dtype_prefix = m.group(4)
        addr = int(m.group(5), 16)
        size = int(m.group(6))

        dtype_str = _DT_NAME_TO_STR.get(dtype_prefix)
        if dtype_str is None:
            if shape:
                total_elements = 1
                for s in shape:
                    total_elements *= s
                if total_elements > 0:
                    elem_size = size // total_elements
                    dtype_str = _SIZE_TO_DTYPE.get(elem_size, "float16")
                else:
                    dtype_str = "uint8"
            else:
                dtype_str = "uint8"
            logging.warning(f"  tensor[{index}] dtype DT_{dtype_prefix} unknown, inferred {dtype_str}")
        else:
            if shape:
                total_elements = 1
                for s in shape:
                    total_elements *= s
                if total_elements > 0:
                    expected_elem_size = size // total_elements
                    actual_elem_size = _DTYPE_TO_SIZE.get(dtype_str, 0)
                    if actual_elem_size != expected_elem_size:
                        inferred = _SIZE_TO_DTYPE.get(expected_elem_size, dtype_str)
                        logging.warning(
                            f"  tensor[{index}] dtype {dtype_str} ({actual_elem_size}B/elem) "
                            f"mismatch with size/shape ({expected_elem_size}B/elem), using {inferred}"
                        )
                        dtype_str = inferred

        tensors.append(TensorInfo(addr, size, dtype_str, shape, io_type, index))
        logging.info(f"  tensor[{index}] {io_type} addr=0x{addr:x} size={size} dtype={dtype_str} shape={shape}")

    basename = os.path.basename(dump_file)
    parts = basename.split(".")
    kernel_name = parts[0] if parts else "unknown"

    return kernel_name, tensors


def _find_call_kernel_so(work_dir: str) -> Optional[str]:
    """在 work_dir 及相关路径下查找 call_kernel.so。"""
    search_dirs = [
        os.path.join(work_dir, "build"),
        os.path.join(os.path.dirname(os.path.abspath(work_dir)), "build"),
        os.path.join(os.getcwd(), "build"),
    ]
    ascend_work_path = os.environ.get("ASCEND_WORK_PATH", "")
    if ascend_work_path:
        search_dirs.append(os.path.join(ascend_work_path, "build"))
    for build_dir in search_dirs:
        if not os.path.isdir(build_dir):
            continue
        try:
            result = subprocess.run(
                ["find", build_dir, "-name", "call_kernel.so"],
                capture_output=True, text=True, timeout=30,
            )
        except Exception:
            continue
        for line in result.stdout.strip().splitlines():
            if line.strip():
                return line.strip()
    return None


def codegen_test_script(
    call_kernel_so: str,
    kernel_name: str,
    dump_file: str,
    tensors: List[TensorInfo],
    device_id: int,
    output_path: str,
    block_dim: int = 1,
) -> None:
    """生成单算子复现脚本。"""
    with open(dump_file, "rb") as f:
        all_data = f.read()

    dump_text_start = all_data.find(b"[Dump]")
    if dump_text_start == -1:
        text_end = 0
    else:
        search_region = all_data[dump_text_start:dump_text_start + 4096]
        last_bytes = search_region.rfind(b"bytes\n")
        if last_bytes != -1:
            text_end = dump_text_start + last_bytes + len(b"bytes\n")
        else:
            text_end = 0

    tensor_data_list = []
    offset = text_end
    for t in sorted(tensors, key=lambda x: x.index):
        tensor_data = all_data[offset:offset + t.size]
        tensor_data_list.append(tensor_data)
        offset += t.size

    bin_dir = os.path.dirname(output_path)
    load_lines: List[str] = []
    tensor_vars: List[str] = []

    for t, data in zip(sorted(tensors, key=lambda x: x.index), tensor_data_list):
        bin_path = os.path.join(bin_dir, f"tensor_{t.io_type}_{t.index}.bin")
        with open(bin_path, "wb") as f:
            f.write(data)

        var = f"t_{t.io_type}_{t.index}"
        torch_dtype = _STR_TO_TORCH.get(t.dtype_str, "torch.float16")
        np_dtype = _STR_TO_NP.get(t.dtype_str, "np.float16")
        shape_repr = repr(t.shape) if t.shape else "(-1,)"
        load_lines.append(f"{var}_np = np.fromfile(r'{bin_path}', dtype={np_dtype}).reshape({shape_repr})")
        load_lines.append(f"{var} = torch.tensor({var}_np, device=device).view({torch_dtype})")
        tensor_vars.append(var)

    lines = [
        "#!/usr/bin/env python3",
        "# Auto-generated by debug_aicore_error_pro_repro.py",
        f"# Kernel: {kernel_name}",
        f"# call_kernel.so: {call_kernel_so}",
        f"# Device: {device_id}",
        f"# block_dim: {block_dim}",
        "",
        "import ctypes",
        "import os",
        "import sys",
        "import numpy as np",
        "import torch",
        "import torch_npu  # noqa: F401",
        "",
        f"device = torch.device(f'npu:{device_id}')",
        "torch.npu.set_device(device)",
        "",
        f"_SO = ctypes.CDLL(r'{call_kernel_so}')",
        "",
    ]
    lines.extend(load_lines)
    lines += [
        "",
        f"_args = [{', '.join(tensor_vars)}]",
        "_ctypes_args = [ctypes.c_void_p(t.data_ptr()) for t in _args]",
        "_stream = torch.npu.current_stream()",
        f"_block_dim = {block_dim}",
        "",
        "print(f'Launching kernel with block_dim={_block_dim}')",
        "_SO.call_kernel(_block_dim, getattr(_stream, '_as_parameter_'), *_ctypes_args)",
        "try:",
        "    torch.npu.synchronize()",
        "    print('PASS: No AICORE error — execution succeeded')",
        "    sys.exit(0)",
        "except Exception as e:",
        "    print(f'FAIL: AICORE error reproduced: {e}')",
        "    sys.exit(1)",
        "",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logging.info(f"Test script generated: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="PyPTO Pro AICORE Error Offline Reproduction Tool")
    parser.add_argument("-p", type=str, required=True, help="Path to ASCEND_WORK_PATH work directory")
    parser.add_argument("-d", type=str, default=None, help="Device ID")
    parser.add_argument("-out", type=str, default=None,
                        help="Output directory (default: <work_dir>/aicore_error_debug)")
    parser.add_argument("-t", type=int, default=600, help="Timeout (seconds)")
    return parser.parse_args()


def main():
    args = parse_args()
    device_id = _get_device_id(args.d)
    logging.info(f"device_id = {device_id}")

    if args.out is None:
        args.out = os.path.join(os.path.abspath(args.p), "aicore_error_debug")
    os.makedirs(args.out, exist_ok=True)
    logging.info(f"output dir = {args.out}")

    dump_dir, dump_dev_id = _find_data_dump_dir(args.p)
    if args.d is None:
        device_id = dump_dev_id
        logging.info(f"device_id (from dump) = {device_id}")
    logging.info(f"dump dir = {dump_dir}")

    dump_file = None
    for fname in os.listdir(dump_dir):
        fpath = os.path.join(dump_dir, fname)
        if os.path.isfile(fpath) and not fname.endswith("_host.o") and not fname.endswith("_debug.o"):
            dump_file = fpath
            break
    if dump_file is None:
        logging.error("No dump data file found")
        sys.exit(1)
    logging.info(f"dump file = {dump_file}")

    kernel_name, tensors = _parse_dump_file(dump_file)
    if not tensors:
        logging.error("No tensors parsed from dump file")
        sys.exit(1)
    logging.info(f"kernel_name = {kernel_name}, {len(tensors)} tensors")

    call_kernel_so = _find_call_kernel_so(args.p)
    if not call_kernel_so:
        logging.error("call_kernel.so not found in build directory")
        sys.exit(1)
    logging.info(f"call_kernel.so = {call_kernel_so}")

    script_path = os.path.join(args.out, "test_single_op.py")
    codegen_test_script(call_kernel_so, kernel_name, dump_file, tensors, device_id, script_path, block_dim=1)

    logging.info("========== Executing Single-Operator Test ==========")
    try:
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=args.t)
    except subprocess.TimeoutExpired:
        logging.error(f"Test script timed out ({args.t}s)")
        sys.exit(1)

    output = result.stdout or ""
    if result.stderr:
        output += "\n[stderr]\n" + result.stderr
    print(output)

    if result.returncode == 0:
        logging.info("PASS: No AICORE error — execution succeeded")
    else:
        logging.info("FAIL: AICORE error reproduced")

    report_path = os.path.join(args.out, "reproduction_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Kernel: {kernel_name}\n")
        f.write(f"Device: {device_id}\n")
        f.write(f"call_kernel.so: {call_kernel_so}\n")
        f.write(f"dump file: {dump_file}\n")
        f.write(f"Tensors: {len(tensors)}\n")
        for t in tensors:
            f.write(f"  {t.io_type}[{t.index}] addr=0x{t.addr:x} size={t.size} dtype={t.dtype_str} shape={t.shape}\n")
        f.write(f"\nExecution result (exit code={result.returncode}):\n")
        f.write(output)
    logging.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
