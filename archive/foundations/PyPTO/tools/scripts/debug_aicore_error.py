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
r"""
基于 .pyptokb 离线二进制包的单算子复现工具。

自动串联: msaicerr 解析 → info.txt 解析 → plog 补充 → 生成 test_single_op.py → 执行诊断。

用法示例
--------
python debug_aicore_error.py -p /path/to/report -out /tmp/out

python debug_aicore_error.py -p /path/to/report -out /tmp/out -d 0

python debug_aicore_error.py -p /path/to/report -out /tmp/out -t 1200
"""

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# -------------------------------------------------------------------
# 模块级: pypto 安装根路径（Phase A 中初始化）
# -------------------------------------------------------------------

_PYTO_ROOT: Optional[str] = None

# -------------------------------------------------------------------
# 日志输出（与 msaicerr 格式一致），追加到 -out 目录的 debug_info.txt
# -------------------------------------------------------------------

_DEBUG_LOG_PATH: Optional[str] = None  # _init_debug_log() 后指向 msaicerr_out 下的 debug_info.txt
_BUNDLED_KERNEL_PATH: Optional[str] = None  # _enrich_from_plog() 后指向 .pyptokb 路径
_BUNDLED_KERNEL_PATH_UNDEF: Optional[str] = None  # _enrich_from_plog() 后指向 *_nosubfunc.pyptokb 路径
_EARLY_LOGS: List[str] = []             # _init_debug_log() 之前暂存日志


def _print_log(level: str, msg: str) -> None:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getpid()
    line = current_time + " (" + str(pid) + ") - [" + level + "] " + msg
    print(line)
    sys.stdout.flush()
    if _DEBUG_LOG_PATH is not None:
        try:
            with open(_DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass
    else:
        _EARLY_LOGS.append(line)


def _init_debug_log(msaicerr_out: str) -> None:
    """找到 -out 目录下的 debug_info.txt，后续日志直接追加到该文件。"""
    global _DEBUG_LOG_PATH
    target = os.path.join(msaicerr_out, "debug_info.txt")
    _DEBUG_LOG_PATH = os.path.abspath(target)
    # 将 init 前的早期日志刷入
    for line in _EARLY_LOGS:
        try:
            with open(_DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass
    _EARLY_LOGS.clear()


# -------------------------------------------------------------------
# dtype 映射表
# -------------------------------------------------------------------

_STR_TO_NP: Dict[str, str] = {
    "float16":  "np.float16",
    "float32":  "np.float32",
    "float64":  "np.float64",
    "int8":     "np.int8",
    "int16":    "np.int16",
    "int32":    "np.int32",
    "int64":    "np.int64",
    "uint8":    "np.uint8",
    "bfloat16": "np.int16",   # numpy 不支持 bfloat16，用 int16 先读
    "bool":     "np.bool_",
    # fp8: numpy 不支持，用 uint8 读原始字节后 view 为对应 torch 类型
    "float8_e4m3fn": "np.uint8",
    "fp8e4m3":       "np.uint8",
    "float8_e5m2":   "np.uint8",
    "fp8e5m2":       "np.uint8",
    "fp8":           "np.uint8",
    "float8":        "np.uint8",
}

_STR_TO_TORCH: Dict[str, str] = {
    "float16":  "torch.float16",
    "float32":  "torch.float32",
    "float64":  "torch.float64",
    "int8":     "torch.int8",
    "int16":    "torch.int16",
    "int32":    "torch.int32",
    "int64":    "torch.int64",
    "uint8":    "torch.uint8",
    "bfloat16": "torch.bfloat16",
    "bool":     "torch.bool",
    "float8_e4m3fn": "torch.float8_e4m3fn",
    "fp8e4m3":       "torch.float8_e4m3fn",
    "float8_e5m2":   "torch.float8_e5m2",
    "fp8e5m2":       "torch.float8_e5m2",
    "fp8":           "torch.float8_e4m3fn",
    "float8":        "torch.float8_e4m3fn",
}


# -------------------------------------------------------------------
# 获取 CANN 目录
# -------------------------------------------------------------------

def get_ascend_home() -> str:
    """通过环境变量 ASCEND_HOME_PATH 获取 CANN 包目录。"""
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_home:
        raise RuntimeError(
            "ASCEND_HOME_PATH env variable not set, please source set_env.sh and retry"
        )
    if not os.path.isdir(ascend_home):
        raise RuntimeError(f"ASCEND_HOME_PATH directory does not exist: {ascend_home}")
    return ascend_home


# -------------------------------------------------------------------
# 调用 msaicerr.py 解析
# -------------------------------------------------------------------

def run_msaicerr(report_path: str, output_path: str, device_id: int,
                 ascend_home: str) -> str:
    """
    调用 CANN 包下的 msaicerr.py 解析 AIC Error 报告。

    返回 msaicerr 输出目录（即 info_<timestamp> 目录）。
    """
    msaicerr_script = os.path.join(ascend_home, "tools", "msaicerr", "msaicerr.py")
    if not os.path.isfile(msaicerr_script):
        raise RuntimeError(f"msaicerr.py not found: {msaicerr_script}")

    # 先列出将要输出的目录，用于事后定位 info_<timestamp>
    before_items = set()
    if os.path.isdir(output_path):
        for item in os.listdir(output_path):
            before_items.add(item)

    cmd = [
        sys.executable,
        msaicerr_script,
        "-p", report_path,
        "-out", output_path,
        "-dev", str(device_id),
    ]
    _print_log("INFO", f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Phase B: msaicerr.py parsing timed out (600s), check log size or run manually")

    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)

    if result.returncode != 0:
        _print_log("WARNING", f"msaicerr.py return code: {result.returncode} (continuing to locate info.txt)")

    # 定位 info_<timestamp> 目录
    after_items = set()
    if os.path.isdir(output_path):
        for item in os.listdir(output_path):
            after_items.add(item)

    new_items = after_items - before_items
    for item in new_items:
        full = os.path.join(output_path, item)
        if os.path.isdir(full) and item.startswith("info_"):
            _print_log("INFO", f"msaicerr output directory: {full}")
            return full

    # 备选：在 output_path 下找最新创建的 info_* 目录
    best = None
    best_mtime = 0
    for item in os.listdir(output_path):
        full = os.path.join(output_path, item)
        if os.path.isdir(full) and item.startswith("info_"):
            mtime = os.path.getmtime(full)
            if mtime > best_mtime:
                best_mtime = mtime
                best = full
    if best:
        _print_log("INFO", f"msaicerr output directory (mtime): {best}")
        return best

    raise RuntimeError(
        f"Cannot find info_<timestamp> directory from msaicerr output in {output_path}"
    )


def find_info_txt(msaicerr_out_dir: str) -> str:
    """在 msaicerr 输出目录下查找 info.txt。"""
    for root, dirs, files in os.walk(msaicerr_out_dir):
        for fname in files:
            if fname == "info.txt":
                return os.path.join(root, fname)
    raise RuntimeError(f"Cannot find info.txt in {msaicerr_out_dir}")


def find_debug_info_txt_path(msaicerr_out: str) -> str:
    """返回 msaicerr 输出目录下 debug_info.txt 的路径（不检查是否存在）。"""
    return os.path.join(msaicerr_out, "debug_info.txt")


# -------------------------------------------------------------------
# 工作目录结构校验 & 环境检查
# -------------------------------------------------------------------

def _get_app_plog_dir(work_dir: str) -> str:
    """返回工作目录下应用层 PYPTO plog 目录路径。"""
    return os.path.join(work_dir, "log", "debug", "plog")


def _validate_work_dir(work_dir: str) -> None:
    """检查 -p 目录结构，缺失必要子目录时报错中断。"""
    required_dirs = {
        "log/debug/plog": _get_app_plog_dir(work_dir),
        "extra-info/data-dump": os.path.join(work_dir, "extra-info", "data-dump"),
        "pypto": os.path.join(work_dir, "pypto"),
    }
    missing = []
    for label, path in required_dirs.items():
        if not os.path.isdir(path):
            missing.append(f"  {label}: {path}")
    if missing:
        raise RuntimeError(
            "Work directory missing these subdirectories:\n" + "\n".join(missing)
        )


def _get_device_id(cli_device_id: Optional[str] = None) -> int:
    """获取 device id。优先级: -d 参数 > TILE_FWK_DEVICE_ID 环境变量。"""
    if cli_device_id is not None:
        try:
            return int(cli_device_id)
        except ValueError:
            raise RuntimeError(f"-d argument value invalid: {cli_device_id}")

    val = os.environ.get("TILE_FWK_DEVICE_ID", "")
    if not val:
        raise RuntimeError("-d argument not specified and TILE_FWK_DEVICE_ID env variable not set, "
                           "please specify -d or set the env variable and retry")
    try:
        return int(val)
    except ValueError:
        raise RuntimeError(f"TILE_FWK_DEVICE_ID value invalid: {val}")


def _check_log_level() -> None:
    """检查 ASCEND_GLOBAL_LOG_LEVEL，非 ERROR 级别时提示可能较慢。"""
    level = os.environ.get("ASCEND_GLOBAL_LOG_LEVEL", "3")
    if level != "3":
        _print_log("WARNING",
            f"ASCEND_GLOBAL_LOG_LEVEL({level}) not set to ERROR(3), single-operator test be slow at current log level, "
            "recommend export ASCEND_GLOBAL_LOG_LEVEL=3")


# -------------------------------------------------------------------
# 解析 info.txt
# -------------------------------------------------------------------

class TensorInfo:
    """单个 dump tensor 的元信息。"""
    __slots__ = ("path", "shape", "dtype", "io_type", "index")

    def __init__(self, path: str, shape: tuple, dtype: str, io_type: str, index: int):
        self.path = path
        self.shape = shape
        self.dtype = dtype      # 原始 dtype 字符串，如 "float16"
        self.io_type = io_type  # "input" / "output" / "workspace"
        self.index = index


def _extract_from_sections(sections: Dict[str, str]) -> Tuple[str, List[TensorInfo]]:
    """从 sections dict 中提取 kernel 名和 tensor 列表。"""
    # --- 提取 kernel name (section 1) ---
    sec1_key = _find_section_key(sections, "1. Basic information")
    if sec1_key is None:
        raise RuntimeError("section 1 (Basic information) not found in info.txt")
    sec1_content = sections[sec1_key]

    kernel_name_match = re.search(r"kernel name\s+:\s*(.+)", sec1_content)
    if not kernel_name_match:
        raise RuntimeError("'kernel name' field not found in section 1")
    kernel_name = kernel_name_match.group(1).strip()

    # 从 "PyPTO_xxx_0_mix_aic" 提取 PyPTO_ 和 _0_mix_aic 之间的部分
    func_match = re.match(r"PyPTO_(.+?)_\d+_mix_aic", kernel_name)
    if func_match:
        kernel_func_name = func_match.group(1)
    else:
        if kernel_name.startswith("PyPTO_"):
            kernel_func_name = kernel_name[len("PyPTO_"):]
        else:
            kernel_func_name = kernel_name

    _print_log("INFO", f"kernel name = {kernel_name}")
    _print_log("INFO", f"Inferred pypto function name = {kernel_func_name}")

    # --- 提取 section 5: Operator Dump File Parsing ---
    sec5_key = _find_section_key(sections, "5. Operator Dump File Parsing")
    if sec5_key is None:
        raise RuntimeError("section 5 (Operator Dump File Parsing) not found in info.txt")
    sec5_content = sections[sec5_key]

    tensor_pattern = re.compile(
        r"shape:\s*\(([^)]*)\)\s+size:\s*\d+\s+dtype:\s*(\S+)\s*\n"
        r"(.+?)\n",
    )
    raw_matches = list(tensor_pattern.finditer(sec5_content))

    tensors: List[TensorInfo] = []
    for i, m in enumerate(raw_matches):
        shape_str = m.group(1)
        dtype_str = m.group(2)
        file_path = m.group(3).strip()

        if shape_str.strip():
            shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
        else:
            shape = ()

        basename = os.path.basename(file_path)
        parts = basename.split(".")
        io_type = "unknown"
        index = 0
        if len(parts) >= 5:
            io_field = parts[-4]
            if io_field.lower() in ("input", "output", "workspace"):
                io_type = io_field.lower()
            try:
                index = int(parts[-3])
            except ValueError:
                pass

        tensors.append(TensorInfo(
            path=file_path, shape=shape, dtype=dtype_str,
            io_type=io_type, index=index,
        ))
        _print_log("DEBUG", f"  tensor[{i}] {io_type}[{index}] shape={shape} dtype={dtype_str} "
                   f"file={basename}")

    order = {"input": 0, "output": 1, "workspace": 2}
    tensors.sort(key=lambda t: (order.get(t.io_type, 9), t.index))

    _print_log("INFO", f"Parsed {len(tensors)} tensors "
               f"({sum(1 for t in tensors if t.io_type == 'input')} input, "
               f"{sum(1 for t in tensors if t.io_type == 'output')} output, "
               f"{sum(1 for t in tensors if t.io_type == 'workspace')} workspace)")

    return kernel_func_name, tensors


# -------------------------------------------------------------------
# Bundle 模式：用 .pyptokb 离线二进制做单算子复现
# -------------------------------------------------------------------

# DataType enum (tilefwk/data_type.h: DATA_TYPE_ALL)
# DT_INT4=0, INT8=1, INT16=2, INT32=3, INT64=4, FP8=5, FP16=6, FP32=7, BF16=8, BOOL=9, UINT8=10
_STR_TO_DT_ENUM: Dict[str, int] = {
    "int8":     1,
    "int16":    2,
    "int32":    3,
    "int64":    4,
    "fp8":      5,
    "float8":   5,
    "float16":  6,
    "float32":  7,
    "bfloat16": 8,
    "bool":     9,
    "uint8":    10,
    "float8_e4m3fn": 17,
    "fp8e4m3":       17,
    "float8_e5m2":   18,
    "fp8e5m2":       18,
}


# 需要做 view 的 dtype（numpy 无原生支持）：bfloat16 用 int16，fp8 系列用 uint8
_NEEDS_VIEW_DTYPES = frozenset({
    "bfloat16",
    "float8_e4m3fn", "fp8e4m3", "float8_e5m2", "fp8e5m2", "fp8", "float8",
})

_BUNDLE_SO_NAMES = ("libtile_fwk_bundle.so",)


def find_bundle_so() -> Optional[str]:
    """自动查找 libtile_fwk_bundle.so。"""
    # 1. 环境变量
    env_so = os.environ.get("PYPTO_BUNDLE_SO", "")
    if env_so and os.path.isfile(env_so):
        return env_so

    # 2. 从 _PYTO_ROOT 推断
    if _PYTO_ROOT:
        so_dir = os.path.join(_PYTO_ROOT, "lib")
        for name in _BUNDLE_SO_NAMES:
            cand = os.path.join(so_dir, name)
            if os.path.isfile(cand):
                return cand

    # 3. LD_LIBRARY_PATH 搜索
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not d:
            continue
        for name in _BUNDLE_SO_NAMES:
            cand = os.path.join(d, name)
            if os.path.isfile(cand):
                return cand

    return None


def _bundle_tensor_load_code(tensors: List[TensorInfo]) -> Tuple[List[str], List[str]]:
    """
    生成 bundle 模式的 tensor 加载 + PyptoTensorDesc 构造代码。

    返回 (load_and_desc_lines, tensor_desc_var_names)
    每个 tensor_desc_var 是 PyptoTensorDesc 的变量名。
    """
    lines: List[str] = []
    desc_vars: List[str] = []

    for t in tensors:
        var_name = f"t_{t.io_type}_{t.index}"
        desc_var = f"d_{t.io_type}_{t.index}"
        dt_enum = _STR_TO_DT_ENUM.get(t.dtype, _STR_TO_DT_ENUM["float32"])

        if t.path.endswith(".npy"):
            lines.append(f"{var_name}_np = np.load(r'{t.path}')")
            lines.append(f"{var_name} = torch.tensor({var_name}_np, device=device)")
        elif t.path.endswith(".bin"):
            np_dtype = _STR_TO_NP.get(t.dtype, _STR_TO_NP["float16"])
            torch_dtype = _STR_TO_TORCH.get(t.dtype, _STR_TO_TORCH["float16"])
            shape_repr = repr(t.shape)

            if t.dtype in _NEEDS_VIEW_DTYPES:
                lines.append(f"# {t.dtype}: numpy 无原生支持，用 {np_dtype} 加载后 view 为 {torch_dtype}")
                lines.append(f"{var_name}_np = np.fromfile(r'{t.path}', dtype={np_dtype}).reshape({shape_repr})")
                lines.append(f"{var_name} = torch.tensor({var_name}_np, device=device).view({torch_dtype})")
            else:
                lines.append(f"{var_name}_np = np.fromfile(r'{t.path}', dtype={np_dtype}).reshape({shape_repr})")
                lines.append(f"{var_name} = torch.tensor({var_name}_np, device=device)")

        # 构造 PyptoTensorDesc
        lines.append(f"{desc_var} = PyptoTensorDesc()")
        lines.append(f"{desc_var}.addr = ctypes.c_void_p({var_name}.data_ptr())")
        lines.append(f"{desc_var}.dataType = {dt_enum}  # {t.dtype}")
        lines.append(f"{desc_var}.rank = {len(t.shape)}")
        for i, s in enumerate(t.shape):
            lines.append(f"{desc_var}.shape[{i}] = {s}")
        lines.append("")

        desc_vars.append(desc_var)

    return lines, desc_vars


def codegen_bundle_script(
    bundle_path: str,
    kernel_func_name: str,
    tensors: List[TensorInfo],
    device_id: int,
    output_path: str,
    bundle_so: str,
) -> None:
    """生成基于 .pyptokb 离线二进制的单算子复现脚本。"""

    load_lines, desc_vars = _bundle_tensor_load_code(tensors)

    lines = [
        "#!/usr/bin/env python3",
        "# coding: utf-8",
        "# Auto-generated by pypto_aicerr_repro.py (bundle mode)",
        f"# Kernel: {kernel_func_name}",
        f"# Bundled kernel:  {bundle_path}",
        f"# Device:  {device_id}",
        "#",
        "",
        "import ctypes",
        "import os",
        "import sys",
        "import numpy as np",
        "import torch",
        "import torch_npu  # noqa: F401",
        "",
        "# ---- PyptoTensorDesc (kernel_bundle_format.h / pypto_bundle_api.h) ----",
        "",
        "class PyptoTensorDesc(ctypes.Structure):",
        '    _fields_ = [',
        '        ("addr",     ctypes.c_void_p),',
        '        ("dataType", ctypes.c_int32),',
        '        ("rank",     ctypes.c_int32),',
        '        ("shape",    ctypes.c_int64 * 8),',
        "    ]",
        "",
        "# ---- Load libtile_fwk_bundle.so ----",
        "",
        f'_BUNDLE_SO = os.environ.get("PYPTO_BUNDLE_SO", r"{bundle_so}")',
        "# Production builds skip RPATH, so dlopen-by-abspath does not search the sibling dir.",
        "# Preload DT_NEEDED of the plain bundle .so first (same order as python/pypto/_loader.py).",
        "_BUNDLE_DIR = os.path.dirname(os.path.abspath(_BUNDLE_SO))",
        "for _dep in (",
        '    "libc_sec.so",',
        '    "libtile_fwk_utils.so",',
        '    "libtile_fwk_adapter.so",',
        '    "libtile_fwk_cann_host_runtime.so",',
        '    "libtile_fwk_platform.so",',
        '    "libtile_fwk_interface.so",',
        '    "libtile_fwk_codegen.so",',
        '    "libtile_fwk_compiler.so",',
        '    "libtile_fwk_runtime.so",',
        "):",
        "    _p = os.path.join(_BUNDLE_DIR, _dep)",
        "    if os.path.isfile(_p):",
        "        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)",
        "_BUNDLE_LIB = ctypes.CDLL(_BUNDLE_SO, mode=ctypes.RTLD_GLOBAL)",
        "",
        "_desc_p = ctypes.POINTER(PyptoTensorDesc)",
        "_BUNDLE_LIB.PyptoWorkspace.restype = ctypes.c_uint64",
        "_BUNDLE_LIB.PyptoWorkspace.argtypes = [ctypes.c_char_p, _desc_p, ctypes.c_uint32]",
        "_BUNDLE_LIB.PyptoLaunch.restype = ctypes.c_int",
        "_BUNDLE_LIB.PyptoLaunch.argtypes = [",
        "    ctypes.c_char_p, _desc_p, ctypes.c_uint32,",
        "    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,",
        "]",
        "",
        "# ---- Device ----",
        f"device = torch.device(f'npu:{device_id}')",
        "torch.npu.set_device(device)",
        "",
        "# ---- 加载 dump tensor ----",
        "",
    ]
    lines.extend(load_lines)

    lines += [
        "# ---- 构造 PyptoTensorDesc 数组 ----",
        "",
        f"_descs = (PyptoTensorDesc * {len(desc_vars)})({', '.join(desc_vars)})",
        "",
        "# ---- 调用 bundle ----",
        "",
        f'_bundle_path = os.environ.get("PYPTO_BUNDLE_PATH", r"{bundle_path}").encode()',
        "",
        f"print('kernel: {kernel_func_name}  (bundle mode)')",
        "",
        "ws_size = _BUNDLE_LIB.PyptoWorkspace(_bundle_path, _descs, len(_descs))",
        "",
        "_ws_keep = None",
        "_ws_ptr = None",
        "if ws_size > 0:",
        "    _ws_keep = torch.empty(ws_size, dtype=torch.uint8, device=device)",
        "    _ws_ptr = ctypes.c_void_p(_ws_keep.data_ptr())",
        "",
        "rc = _BUNDLE_LIB.PyptoLaunch(_bundle_path, _descs, len(_descs), _ws_ptr, None, 1)",
        "if rc != 0:",
        "    sys.exit(1)",
        "torch.npu.synchronize()",
        'print("Bundle execution completed.")',
        'print("sync done.")',
        "",
    ]

    content = "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(content)

    _print_log("INFO", f"Test script: {output_path}")
    _print_log("INFO", f"Bundled kernel: {bundle_path}")


# -------------------------------------------------------------------
# info.txt 读写工具
# -------------------------------------------------------------------

_INFO_SECTION_PATTERN = re.compile(
    r'^\*{3,}\d+\.\s+.+?\*{3,}$', re.MULTILINE
)
_TARGET_SECTION_KEYWORD = "6. Execution Result of the Single-Operator Test Case"



def _parse_info_txt_to_sections(info_txt_path: str):
    """
    将 info.txt 解析为 (preamble, sections_dict, header_order)。

    preamble:    第一个 section header 之前的内容（根因结论 + 空行）
    sections:    {header_line: content}  例如
                 {"********************6. ... ***********************": "执行结果内容"}
    header_order: 保持原始顺序的 header 列表
    """
    with open(info_txt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    matches = list(_INFO_SECTION_PATTERN.finditer(content))
    if not matches:
        return content, {}, []

    preamble = content[:matches[0].start()]
    sections: Dict[str, str] = {}
    header_order: List[str] = []

    for i, m in enumerate(matches):
        header = m.group().strip()
        header_order.append(header)
        start = m.end() + 1  # +1 跳过 header 后的换行符
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections[header] = content[start:end].rstrip()

    return preamble, sections, header_order


def _find_section_key(sections: Dict[str, str], keyword: str) -> Optional[str]:
    """在 sections 中查找匹配 keyword 的 header key。"""
    for header in sections:
        if keyword in header:
            return header
    return None


def _find_section6_key(sections: Dict[str, str]) -> Optional[str]:
    """在 sections 中查找第 6 段 Single-Operator Test Case 的 header key。"""
    return _find_section_key(sections, _TARGET_SECTION_KEYWORD)


def _rewrite_info_txt(info_txt_path: str, preamble: str,
                      sections: Dict[str, str], header_order: List[str]):
    """用更新后的 sections 重写 info.txt。"""
    lines = [preamble.rstrip()]
    for header in header_order:
        lines.append("")
        lines.append(header)
        lines.append(sections.get(header, ""))
    lines.append("")  # 文件末尾换行
    with open(info_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        _print_log("INFO", f"Updated {info_txt_path}")



def _enrich_from_plog(msaicerr_out: str, sections: Dict[str, str],
                       work_dir: str = ""):
    """
    补充信息到 info.txt：coreType / bundledKernelPath / fixedPC / llvm-symbolizer。
    """
    global _BUNDLED_KERNEL_PATH, _BUNDLED_KERNEL_PATH_UNDEF
    # 1. 从 section 1 提取 core id
    sec1_key = _find_section_key(sections, "1. Basic information")
    if sec1_key is None:
        _print_log("WARNING", "Section 1 (Basic information) not found, skipping plog enrichment")
        return
    sec1_content = sections[sec1_key]

    core_id_match = re.search(r'core\s+id\s*:\s*(\d+)', sec1_content, re.IGNORECASE)
    if not core_id_match:
        _print_log("WARNING", "core id not found in section 1, skipping plog enrichment")
        return
    core_id = core_id_match.group(1)
    _print_log("INFO", f"section 1 core id = {core_id}")

    # 2. find -name 'PyPTO*0_mix_aic.pyptokb' 获取 bundled kernel 路径
    report_dir = work_dir if work_dir and os.path.isdir(work_dir) else ""
    bundled_kernel_path = _find_bundled_kernel(report_dir) if report_dir else None
    _BUNDLED_KERNEL_PATH = bundled_kernel_path

    # 2b. find -name 'PyPTO*0_mix_aic_nosubfunc.pyptokb' 获取 undef bundled kernel 路径
    bundled_kernel_path_undef = (
        _find_undef_bundled_kernel(report_dir) if report_dir else None
    )
    _BUNDLED_KERNEL_PATH_UNDEF = bundled_kernel_path_undef

    plog_dir = os.path.join(msaicerr_out, "collection", "plog")
    if not os.path.isdir(plog_dir):
        _print_log("WARNING", f"plog directory does not exist: {plog_dir}")
        if bundled_kernel_path is None:
            return

    # 3. grep 'error info:' 取时间戳最早的内容，从中判断 coreType
    core_type = _parse_core_type_from_earliest_error_info(plog_dir)

    # 4. grep kernel_symbol_locator.cpp 获取 fixedPC 信息（同时匹配 core id + core type）
    pc_match = _parse_fixed_pc_from_plog(plog_dir, core_id, core_type)

    if core_type is None and bundled_kernel_path is None and pc_match is None:
        return

    # 5. 追加 coreType 和 bundled kernel 路径到 section 1
    extra_lines: List[str] = []
    if core_type is not None:
        extra_lines.append(f"core type          : {core_type}")
    if bundled_kernel_path is not None:
        extra_lines.append(f"bundled kernel     : {bundled_kernel_path}")
    if bundled_kernel_path_undef is not None:
        extra_lines.append(f"bundled kernel undef: {bundled_kernel_path_undef}")
    if extra_lines:
        sections[sec1_key] = sec1_content.rstrip() + "\n" + "\n".join(extra_lines)

    # 6. 追加 fixed PC 信息到 section 3，并用 llvm-symbolizer 解析符号
    if pc_match is not None:
        sec3_key = _find_section_key(sections, "3. Operator Error Line Number")
        if sec3_key is None:
            _print_log("WARNING", "section 3 (Operator Error Line Number) not found")
        else:
            sec3_content = sections[sec3_key]
            kernel_file = _extract_kernel_file(sec1_content, msaicerr_out)
            extra = (
                f"\ncore id            : {core_id}"
                f"\ncore type          : {core_type}"
                f"\nfixedStartPC       : {pc_match['fixedStartPC']}"
                f"\nfixedCurrentPC     : {pc_match['fixedCurrentPC']}"
                f"\nfixedPCOffset      : {pc_match['fixedPCOffset']}"
            )
            symbol_info = _run_llvm_symbolizer(kernel_file, pc_match['fixedPCOffset'])
            if symbol_info:
                extra += f"\n{symbol_info}"
            sections[sec3_key] = sec3_content.rstrip() + extra

    # 7. [MSAICERR-WORKAROUND] 多 extend info 场景修正：第一条 errcode 全 0 时，重新匹配到第一条非全 0 的 extend info
    #    msaicerr 修复后可移除本调用
    _fix_multi_extend_info(plog_dir, sections, sec1_content, msaicerr_out)


def _find_bundled_kernel(report_dir: str) -> Optional[str]:
    """从 -p 目录下 find -name 'PyPTO*0_mix_aic.pyptokb' 获取 .pyptokb 路径（排除 _nosubfunc 后缀）。"""
    try:
        result = subprocess.run(
            ["find", report_dir, "-name", "PyPTO*0_mix_aic.pyptokb",
             "!", "-name", "*_nosubfunc.pyptokb"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        _print_log("WARNING", "Phase C: find bundled kernel timed out (30s)")
        return None
    except Exception as e:
        _print_log("WARNING", f"find bundled kernel failed: {e}")
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        _print_log("WARNING", "PyPTO*0_mix_aic.pyptokb not found under -p directory")
        return None
    if len(lines) > 1:
        raise RuntimeError(
            f"Found {len(lines)} PyPTO*0_mix_aic.pyptokb files, "
            "multiple AIC errors in one -p directory is not supported:\n"
            + "\n".join(f"  {p}" for p in lines)
        )

    path = lines[0]
    _print_log("INFO", f"Found bundled kernel: {path}")
    return path


def _find_undef_bundled_kernel(report_dir: str) -> Optional[str]:
    """从 -p 目录下 find -name 'PyPTO*0_mix_aic_nosubfunc.pyptokb' 获取 *_nosubfunc.pyptokb 路径。"""
    try:
        result = subprocess.run(
            ["find", report_dir, "-name", "PyPTO*0_mix_aic_nosubfunc.pyptokb"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        _print_log("WARNING", "Phase C: find undef bundled kernel timed out (30s)")
        return None
    except Exception as e:
        _print_log("WARNING", f"find undef bundled kernel failed: {e}")
        return None

    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        _print_log("WARNING", "PyPTO*0_mix_aic_nosubfunc.pyptokb not found under -p directory")
        return None
    if len(lines) > 1:
        raise RuntimeError(
            f"Found {len(lines)} PyPTO*0_mix_aic_nosubfunc.pyptokb files, "
            "multiple AIC errors in one -p directory is not supported:\n"
            + "\n".join(f"  {p}" for p in lines)
        )

    path = lines[0]
    _print_log("INFO", f"Found undef bundled kernel: {path}")
    return path


def _parse_core_type_from_earliest_error_info(plog_dir: str) -> Optional[str]:
    """
    参考 msaicerr 逻辑：grep 'error info:' 取时间戳最早的那一条，
    根据异常类型判断 coreType：
      - "aicore error"（A2/A3/A5）     → 0
      - "aivector error"（A2/A3）/ "aivec error"（A5）  → 1
    """
    try:
        result = subprocess.run(
            ["grep", "-rnE", "error info:", plog_dir],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        _print_log("WARNING", "Phase C: grep 'error info:' timed out (30s)")
        return None
    except Exception as e:
        _print_log("WARNING", f"grep 'error info:' failed: {e}")
        return None

    if result.returncode != 0 or not result.stdout.strip():
        _print_log("WARNING", "'error info:' not found in plog")
        return None

    # 提取时间戳，参考 msaicerr 直接用字符串排序
    # （YYYY-MM-DD-HH:MM:SS.xxxx 格式天然保证字典序即时间序，无需转 datetime）
    _ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}\.\d+\.\d+)')
    ts_lines = []
    for line in result.stdout.strip().split("\n"):
        m = _ts_pattern.search(line)
        if m:
            ts_lines.append((m.group(1), line))

    if not ts_lines:
        _print_log("WARNING", "No valid timestamp found in 'error info:' lines")
        return None

    ts_lines.sort(key=lambda x: (x[0] is None, x[0]))
    earliest_line = ts_lines[0][1]
    _print_log("INFO", f"Earliest error info: {earliest_line.strip()[:200]}...")


    if "aicore error" in earliest_line.lower():
        _print_log("INFO", "coreType determined as 0 (aicore error)")
        return "0"
    elif "aivector error" in earliest_line.lower() or "aivec error" in earliest_line.lower():
        _print_log("INFO", "coreType determined as 1 (aivector/aivec error)")
        return "1"
    else:
        _print_log("WARNING", "aicore/aivector/aivec error not recognized in earliest error info, coreType unknown")
        return None


# [MSAICERR-WORKAROUND] 多 extend info 场景修正，msaicerr 修复后可整体移除本函数
def _parse_extend_info_records(plog_dir: str) -> List[Dict[str, str]]:
    """grep 'The extend info: errcode:'，解析每条 extend info 的 errcode/core_id/core_type。

    每条记录: {errcode, core_id, core_type}，按 grep 输出顺序（即日志时间顺序）排列。
    core_type: aicore→"0", aivector/aivec→"1"（与 kernel_symbol_locator.cpp 的 coreType 一致）。
    兼容 A2/A3（aicore/aivector error）与 A5（aicore/aivec error）。
    """
    try:
        result = subprocess.run(
            ["grep", "-rnE", "The extend info: errcode:", plog_dir],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        _print_log("WARNING", "grep 'The extend info: errcode:' timed out (30s)")
        return []
    except Exception as e:
        _print_log("WARNING", f"grep 'The extend info: errcode:' failed: {e}")
        return []

    if result.returncode != 0 or not result.stdout.strip():
        _print_log("WARNING", "'The extend info: errcode:' not found in plog")
        return []

    records: List[Dict[str, str]] = []
    for line in result.stdout.strip().split("\n"):
        core_id_m = re.search(r"core id is (\d+)", line)
        # 兼容 A2/A3（aicore/aivector error）与 A5（aicore/aivec error）
        core_type_m = re.search(r"(aicore|aivec(?:tor)?) error", line)
        errcode_m = re.search(r"errcode:(\([^)]*\))", line)
        if not (core_id_m and core_type_m and errcode_m):
            continue
        core_type = "0" if core_type_m.group(1) == "aicore" else "1"
        records.append({
            "core_id": core_id_m.group(1),
            "core_type": core_type,
            "errcode": errcode_m.group(1),
        })
    return records


# [MSAICERR-WORKAROUND] 多 extend info 场景修正，msaicerr 修复后可整体移除本函数
def _errcode_is_all_zero(errcode: str) -> bool:
    """判断 errcode（一元组/三元组，如 "(0)" 或 "(0, 0x800, 0)"）是否所有数值都为 0。"""
    nums = re.findall(r"0x[0-9a-fA-F]+|\d+", errcode)
    if not nums:
        return False
    return all(int(n, 16) == 0 for n in nums)


def _parse_fixed_pc_from_plog(plog_dir: str, core_id: str, core_type: Optional[str]) -> Optional[Dict[str, str]]:
    """
    grep kernel_symbol_locator.cpp，只解析含 "Error PC information" 的行，
    同时匹配 core id + core type，返回 fixedStartPC / fixedCurrentPC / fixedPCOffset。
    """
    try:
        result = subprocess.run(
            ["grep", "-rn", "kernel_symbol_locator.cpp", plog_dir],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        _print_log("WARNING", "Phase C: grep kernel_symbol_locator.cpp timed out (30s)")
        return None
    except Exception as e:
        _print_log("WARNING", f"grep kernel_symbol_locator.cpp failed: {e}")
        return None

    if result.returncode != 0 or not result.stdout.strip():
        _print_log("WARNING", "kernel_symbol_locator.cpp not found in plog")
        return None

    _pc_pattern = re.compile(
        r'coreId=(\d+),\s*coreType=(\d+).*?'
        r'fixedStartPC=(0x[0-9a-fA-F]+).*?'
        r'fixedCurrentPC=(0x[0-9a-fA-F]+).*?'
        r'fixedPCOffset=(0x[0-9a-fA-F]+)',
    )
    parsed = []
    for line in result.stdout.strip().split("\n"):
        if "Error PC information" not in line:
            continue
        m = _pc_pattern.search(line)
        if not m:
            continue
        parsed.append({
            "coreId": m.group(1),
            "coreType": m.group(2),
            "fixedStartPC": m.group(3),
            "fixedCurrentPC": m.group(4),
            "fixedPCOffset": m.group(5),
        })

    if not parsed:
        _print_log("WARNING", "No kernel_symbol_locator line with 'Error PC information' found in plog")
        return None

    # 优先匹配 coreId + coreType，其次仅 coreId
    match = None
    if core_type is not None:
        for p in parsed:
            if p["coreId"] == core_id and p["coreType"] == core_type:
                match = p
                break
    if match is None:
        for p in parsed:
            if p["coreId"] == core_id:
                match = p
                break
    if match is None:
        match = parsed[0]
        _print_log("WARNING", f"No match found for coreId={core_id}, coreType={core_type}"
                   f" ({len(parsed)} total), using coreId={match['coreId']},"
                   f" coreType={match['coreType']}")


    _print_log("INFO", f"Extracted fixed PC from plog: "
              f"fixedStartPC={match['fixedStartPC']}, "
              f"fixedCurrentPC={match['fixedCurrentPC']}, "
              f"fixedPCOffset={match['fixedPCOffset']}")
    return {
        "fixedStartPC": match["fixedStartPC"],
        "fixedCurrentPC": match["fixedCurrentPC"],
        "fixedPCOffset": match["fixedPCOffset"],
    }


def _extract_kernel_file(sec1_content: str, msaicerr_out: str) -> Optional[str]:
    """从 section 1 中提取 kernel file 路径（`.o` 文件）。"""
    m = re.search(r'kernel\s+file\s*:\s*(\S+)', sec1_content, re.IGNORECASE)
    if not m:
        _print_log("WARNING", "kernel file path not found in section 1")
        return None
    kernel_file = m.group(1)
    # 如果路径不包含 collection/compile，尝试拼接
    if not os.path.isabs(kernel_file) and "collection/compile" not in kernel_file:
        candidates = []
        compile_dir = os.path.join(msaicerr_out, "collection", "compile")
        if os.path.isdir(compile_dir):
            for fname in os.listdir(compile_dir):
                if fname.endswith(".o") and kernel_file in fname:
                    candidates.append(os.path.join(compile_dir, fname))
        if candidates:
            kernel_file = candidates[0]
    if not os.path.isfile(kernel_file):
        _print_log("WARNING", f"kernel file does not exist: {kernel_file}")
        return None
    return kernel_file


def _run_llvm_symbolizer(kernel_file: Optional[str], pc_offset: str) -> Optional[str]:
    """执行 llvm-symbolizer --obj=<kernel_file> <pc_offset>，返回解析后的符号行。"""
    if not kernel_file:
        return None
    try:
        result = subprocess.run(
            ["llvm-symbolizer", f"--obj={kernel_file}", pc_offset],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        _print_log("WARNING", "Phase C: llvm-symbolizer symbol resolution timed out (10s)")
        return None
    except Exception as e:
        _print_log("WARNING", f"llvm-symbolizer execution failed: {e}")
        return None

    if result.returncode != 0 or not result.stdout.strip():
        _print_log("WARNING", f"llvm-symbolizer no output (rc={result.returncode})")
        return None

    # 直接追加：指令 + 原始输出
    cmd_str = f"llvm-symbolizer --obj={kernel_file} {pc_offset}"
    # 去掉末尾多余空行，保留原始内容中的换行
    output = result.stdout.rstrip('\n')
    symbol_str = cmd_str + "\n" + output
    _print_log("INFO", f"llvm-symbolizer result:\n{output}")
    return symbol_str


# [MSAICERR-WORKAROUND] 多 extend info 场景修正，msaicerr 修复后可整体移除本函数
def _fix_multi_extend_info(plog_dir: str, sections: Dict[str, str],
                           sec1_content: str, msaicerr_out: str) -> None:
    """
    修正 msaicerr section 2/3 在多 extend info 场景下的错误匹配。

    规则：
    - 只匹配到一条 extend info：保持现有逻辑，不处理
    - 多条 extend info，但第一条 errcode 不全为 0：保持现有逻辑，不处理
    - 多条 extend info，且第一条 errcode 全为 0：
      找到第一条 errcode 不全为 0 的 extend info，
      在 section 2 后追加重新匹配的 AIC_ERROR，
      在 section 3 后追加重新匹配的 fixedStartPC/fixedCurrentPC/fixedPCOffset，
      并用新 core id + core type 重新跑 llvm-symbolizer 追加符号信息。
    """
    records = _parse_extend_info_records(plog_dir)
    if len(records) <= 1:
        _print_log("INFO", "Only one extend info, no re-match needed")
        return

    if not _errcode_is_all_zero(records[0]["errcode"]):
        _print_log("INFO", f"First extend info errcode {records[0]['errcode']} is not all-zero, keep existing logic")
        return

    # 找第一条 errcode 不全为 0 的 extend info
    target = None
    for rec in records:
        if not _errcode_is_all_zero(rec["errcode"]):
            target = rec
            break
    if target is None:
        _print_log("WARNING", "All extend info errcodes are all-zero, no re-match target")
        return

    _print_log("INFO", f"First extend info errcode {records[0]['errcode']} is all-zero, "
              f"re-match to core id {target['core_id']}, core type {target['core_type']}, "
              f"errcode {target['errcode']}")

    # 1. section 2 后追加重新匹配的 AIC_ERROR
    sec2_key = _find_section_key(sections, "2. AI Core DFX Register")
    if sec2_key is None:
        _print_log("WARNING", "section 2 (AI Core DFX Register) not found, skip AIC_ERROR re-match")
    else:
        sections[sec2_key] = sections[sec2_key].rstrip() + \
            f"\n(re-matched) AIC_ERROR : {target['errcode']}"

    # 2. 用新 core id + core type 重新匹配 fixedPC
    pc_match = _parse_fixed_pc_from_plog(plog_dir, target["core_id"], target["core_type"])
    if pc_match is None:
        _print_log("WARNING", "re-matched fixedPC not found, skip section 3 append")
        return

    # 3. section 3 后追加重新匹配的 fixedPC + llvm-symbolizer
    sec3_key = _find_section_key(sections, "3. Operator Error Line Number")
    if sec3_key is None:
        _print_log("WARNING", "section 3 (Operator Error Line Number) not found, skip fixedPC append")
        return

    kernel_file = _extract_kernel_file(sec1_content, msaicerr_out)
    extra = (
        f"\n(re-matched) core id           : {target['core_id']}"
        f"\n(re-matched) core type         : {target['core_type']}"
        f"\n(re-matched) fixedStartPC       : {pc_match['fixedStartPC']}"
        f"\n(re-matched) fixedCurrentPC     : {pc_match['fixedCurrentPC']}"
        f"\n(re-matched) fixedPCOffset      : {pc_match['fixedPCOffset']}"
    )
    symbol_info = _run_llvm_symbolizer(kernel_file, pc_match['fixedPCOffset'])
    if symbol_info:
        extra += f"\n{symbol_info}"
    sections[sec3_key] = sections[sec3_key].rstrip() + extra


def _detect_python() -> str:
    """检测可用的 Python 解释器，确保能 import pypto 和 import torch_npu。"""
    candidates = [sys.executable]
    if os.path.basename(sys.executable) != "python3":
        candidates.append("python3")
    candidates.append("python")
    # 去重，避免同一解释器重复探测
    candidates = list(dict.fromkeys([c for c in candidates if c]))

    for exe in candidates:
        if not exe:
            continue
        try:
            result = subprocess.run(
                [exe, "-c", "import pypto; import torch_npu"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            _print_log("WARNING", f"Phase A: {exe} import check timed out (30s), trying next interpreter")
            continue
        except Exception:
            continue
        if result.returncode == 0:
            _print_log("INFO", f"Using Python: {exe}")
            return exe

    raise RuntimeError("No usable Python interpreter found (must be able to import pypto and torch_npu), "
                       "check the runtime environment")


def _execute_test_script(
    script_path: str,
    python_exe: str,
    timeout: int = 600,
    bundle_path: str = "",
    section: int = 0,
    out_dir: str = "",
) -> Tuple[bool, str]:
    """
    统一执行测试脚本，返回 (passed: bool, output: str)。
    - python_exe: Python 解释器路径
    - bundle_path: 可选，设置 PYPTO_BUNDLE_PATH 环境变量，让脚本用指定 bundle 执行
    - section: 可选，执行前将 ASCEND_WORK_PATH 修改为 {out_dir}/section{section}
    - out_dir: 可选，debug_info.txt 同级目录（即 msaicerr 输出目录），与 section 拼接作为 ASCEND_WORK_PATH 基准
    """
    env = os.environ.copy()

    if section:
        new_work_path = os.path.join(out_dir, f"tmp/section{section}") if out_dir else f"section{section}"
        env["ASCEND_WORK_PATH"] = new_work_path
        _print_log("INFO", f"Section {section}: ASCEND_WORK_PATH={new_work_path}")

    if bundle_path:
        env["PYPTO_BUNDLE_PATH"] = bundle_path

    # Ensure pypto/lib is on LD_LIBRARY_PATH so libtile_fwk_runtime.so can be found
    if _PYTO_ROOT:
        pypto_lib = os.path.join(_PYTO_ROOT, "lib")
        if os.path.isdir(pypto_lib):
            ld_existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join([pypto_lib, ld_existing]) if ld_existing else pypto_lib

    cmd = [python_exe, script_path]
    _print_log("INFO", f"Executing command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"[ERROR] {os.path.basename(script_path)} execution timed out ({timeout}s)"
    except Exception as e:
        return False, f"[ERROR] Script execution exception: {e}"

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += "\n[stderr]\n" + result.stderr
    return (result.returncode == 0), output


def run_test_script_and_update_info(script_path: str, device_id: int,
                                    python_exe: str,
                                    sections: Dict[str, str],
                                    timeout: int = 600,
                                    bundle_path: str = "",
                                    out_dir: str = "") -> bool:
    """执行测试脚本，将结果覆盖 sections dict 中的 section 6。返回 True 表示通过。"""
    test_passed, output = _execute_test_script(script_path, python_exe, timeout, bundle_path,
                                               section=6, out_dir=out_dir)

    # Update section 6 in sections dict
    sec6_key = _find_section6_key(sections)
    if sec6_key:
        sections[sec6_key] = output
    else:
        _print_log("WARNING", "section 6 not found in info.txt, skipping")

    return test_passed


# -------------------------------------------------------------------
# Section 7: 排除核内同步问题
# -------------------------------------------------------------------

_SECTION7_TITLE = "7. Inter-Core Synchronization Diagnosis"


def _find_msnpureport() -> str:
    """定位 msnpureport 工具。

    优先级:
    1. 环境变量 MSNPUREPORT_PATH（显式指定）
    2. ASCEND_HOME_PATH 父目录下 driver/tools/msnpureport（标准昇腾安装布局）
    3. 默认 /usr/local/Ascend/driver/tools/msnpureport（兜底）
    """
    env_path = os.environ.get("MSNPUREPORT_PATH", "")
    if env_path and os.path.isfile(env_path):
        return env_path

    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if ascend_home:
        ascend_root = os.path.dirname(os.path.abspath(ascend_home))
        cand = os.path.join(ascend_root, "driver", "tools", "msnpureport")
        if os.path.isfile(cand):
            return cand

    return "/usr/local/Ascend/driver/tools/msnpureport"


def _is_docker_env():
    """判断当前是否为 docker 环境（通过 /.dockerenv 或 /proc/1/cgroup 检测）。"""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            content = f.read()
            if "docker" in content or "kubepods" in content:
                return True
    except (IOError, PermissionError):
        pass
    return False


def _msnpureport_set_singlecommit(enable: bool, device_id: int, is_docker: bool = False):
    """Enable/disable singlecommit mode. Returns (success, cmd_str, output)."""
    val = "1" if enable else "0"
    cmd = [_find_msnpureport(), "config", "--set", "--singlecommit", val, "-d", str(device_id)]
    if is_docker:
        cmd.append("--docker")
    cmd_str = " ".join(cmd)
    action = "enable" if enable else "restore"
    _print_log("INFO", f"{action} singlecommit: {cmd_str}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if result.stderr:
            output += "\n" + result.stderr.strip()
        success = (result.returncode == 0)
    except subprocess.TimeoutExpired:
        output = f"[ERROR] msnpureport singlecommit config timed out (30s): {cmd_str}"
        success = False
    except Exception as e:
        output = f"[ERROR] {e}"
        success = False
    return success, cmd_str, output


def run_section7_intercore_sync(script_path: str, device_id: int,
                                 sections: Dict[str, str], header_order: List[str],
                                 python_exe: str,
                                 timeout: int = 600,
                                 bundle_path: str = "",
                                 out_dir: str = ""):
    """
    Phase E: msnpureport singlecommit=1 排除核内同步问题。
    """
    _print_log("INFO", "========== Section 7: Inter-Core Sync Diagnosis ==========")

    output_lines: List[str] = []

    is_docker = _is_docker_env()
    _print_log("INFO", f"Docker environment: {is_docker}")

    success, cmd_str, msn_output = _msnpureport_set_singlecommit(True, device_id, is_docker)
    output_lines.append(f"msnpureport enable single-step: {cmd_str}\n{msn_output}")
    output_lines.append("")
    if not success:
        _print_log("WARNING", f"msnpureport enable singlecommit failed: {msn_output}")

    test_passed, test_output = _execute_test_script(script_path, python_exe, timeout, bundle_path,
                                                    section=7, out_dir=out_dir)

    output_lines.append(f"Re-execute (singlecommit=1):\n{test_output}")
    output_lines.append("")

    success2, cmd_str2, msn_output2 = _msnpureport_set_singlecommit(False, device_id, is_docker)
    output_lines.append(f"msnpureport restore: {cmd_str2}\n{msn_output2}")
    output_lines.append("")

    if test_passed:
        conclusion = "Conclusion: PASS in single-step mode → Inter-core sync issue"
    else:
        conclusion = "Conclusion: Still FAIL in single-step mode → Not an inter-core sync issue"
    output_lines.append(conclusion)
    _print_log("INFO", conclusion)

    _upsert_section7(sections, header_order, "\n".join(output_lines))
    return test_passed


def _upsert_section_into_dict(sections: Dict[str, str], header_order: List[str],
                              section_title: str, content: str):
    """将 content 写入 sections dict 中匹配 section_title 的段；不存在则追加。"""
    for header in list(sections.keys()):
        if section_title in header:
            sections[header] = content
            return
    # 不存在则新建
    new_header = f"********************{section_title}***********************"
    new_key = new_header.strip()
    header_order.append(new_key)
    sections[new_key] = content


def _upsert_section7(sections: Dict[str, str], header_order: List[str], content: str):
    """将内容写入 sections dict 的 section 7 段。"""
    _upsert_section_into_dict(sections, header_order, _SECTION7_TITLE, content)


# -------------------------------------------------------------------
# Section 8: 排查框架 vs 算子 CCE 问题
# -------------------------------------------------------------------

_SECTION8_TITLE = "8. Framework vs Operator (CCE) Root Cause Analysis"


def run_section8_framework_vs_cce(script_path: str, device_id: int,
                                   sections: Dict[str, str], header_order: List[str],
                                   python_exe: str,
                                   timeout: int = 600,
                                   bundle_path_undef: str = "",
                                   out_dir: str = ""):
    """
    Phase F: 用 *_nosubfunc.pyptokb 重新执行，判断 sub-func 是否为根因。
    """
    _print_log("INFO", "========== Section 8: Framework vs Operator CCE Diagnosis ==========")

    output_lines: List[str] = []

    if not bundle_path_undef or not os.path.isfile(bundle_path_undef):
        _print_log("WARNING", "_nosubfunc.pyptokb not found, skipping Section 8")
        output_lines.append("[ERROR] *_nosubfunc.pyptokb not found")
        _upsert_section8(sections, header_order, "\n".join(output_lines))
        return

    output_lines.append(f"Nosubfunc bundled kernel: {bundle_path_undef}")
    output_lines.append("")
    _print_log("INFO", f"Nosubfunc bundled kernel: {bundle_path_undef}")

    test_passed, test_output = _execute_test_script(script_path, python_exe, timeout, bundle_path_undef,
                                                    section=8, out_dir=out_dir)

    output_lines.append(f"Re-execute (nosubfunc):\n{test_output}")
    output_lines.append("")

    if test_passed:
        conclusion = "Conclusion: PASS with nosubfunc bundled kernel → Sub-func (CCE) issue"
    else:
        conclusion = "Conclusion: Still FAIL with nosubfunc bundled kernel → Framework issue (not sub-func)"
    output_lines.append(conclusion)
    _print_log("INFO", conclusion)

    _upsert_section8(sections, header_order, "\n".join(output_lines))


def _upsert_section8(sections: Dict[str, str], header_order: List[str], content: str):
    """将内容写入 sections dict 的 section 8 段。"""
    _upsert_section_into_dict(sections, header_order, _SECTION8_TITLE, content)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug AICore Error",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python debug_aicore_error.py -p /path/to/report -out /tmp/out\n"
            "  python debug_aicore_error.py -p /path/to/report -out /tmp/out -d 0\n"
            "  python debug_aicore_error.py -p /path/to/report -out /tmp/out -t 1200\n"
        ),
    )
    parser.add_argument(
        "-p", type=str, required=True,
        help="Path to AIC error debug info",
    )
    parser.add_argument(
        "-d", type=str, default=None,
        help="Device ID (optional, defaults to TILE_FWK_DEVICE_ID env variable)",
    )
    parser.add_argument(
        "-out", type=str, required=True,
        help="Output directory for the debug report",
    )
    parser.add_argument(
        "-t", type=int, default=600,
        help="Timeout threshold (seconds) for single-operator reproduction test script execution, default 600",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ============================================================
    # Phase A: 预处理
    # ============================================================

    # 获取 device id（-d 参数 > TILE_FWK_DEVICE_ID 环境变量）
    device_id = _get_device_id(args.d)
    _print_log("INFO", f"device_id = {device_id}")

    # 检查 -p 目录结构
    _validate_work_dir(args.p)

    # 获取 ASCEND_HOME_PATH
    ascend_home = get_ascend_home()
    _print_log("INFO", f"ASCEND_HOME_PATH = {ascend_home}")

    # 检查日志级别
    _check_log_level()

    # 检测 Python 解释器
    python_exe = _detect_python()

    # 获取 pypto 安装路径
    global _PYTO_ROOT
    try:
        import pypto
        _PYTO_ROOT = os.path.dirname(os.path.abspath(pypto.__file__))
        _print_log("INFO", f"pypto root = {_PYTO_ROOT}")
    except ImportError:
        raise RuntimeError("cannot import pypto, please ensure pypto is installed")

    # 校验 libtile_fwk_bundle.so
    bundle_so = find_bundle_so()
    if not bundle_so:
        _print_log("ERROR", "libtile_fwk_bundle.so not found, please specify via PYPTO_BUNDLE_SO env variable")
        sys.exit(1)

    # ============================================================
    # Phase B: 调用 msaicerr.py 解析 (→ info.txt section 1~5)
    # ============================================================

    msaicerr_out = run_msaicerr(args.p, args.out, device_id, ascend_home)
    _init_debug_log(msaicerr_out)

    # ============================================================
    # Phase C: 信息补全 (→ info.txt section 1, 3)
    # ============================================================

    info_txt = find_info_txt(msaicerr_out)
    _print_log("INFO", f"info.txt: {info_txt}")
    debug_info_txt = find_debug_info_txt_path(msaicerr_out)

    preamble, sections, header_order = _parse_info_txt_to_sections(info_txt)

    kernel_func_name, tensors = _extract_from_sections(sections)

    if not tensors:
        _print_log("ERROR", "No dump tensors parsed, cannot generate reproduction script")
        sys.exit(1)

    _enrich_from_plog(msaicerr_out, sections, args.p)

    bundle_path = _BUNDLED_KERNEL_PATH
    if not bundle_path:
        _print_log("ERROR", ".pyptokb file not found")
        sys.exit(1)

    # ============================================================
    # Phase D: Single-operator Test (→ info.txt section 6)
    # ============================================================

    _print_log("INFO", "========== Section 6: Single-Operator Test ==========")

    output_script = os.path.join(msaicerr_out, "test_single_op.py")
    codegen_bundle_script(
        bundle_path, kernel_func_name, tensors, device_id,
        output_script, bundle_so,
    )

    section6_passed = run_test_script_and_update_info(
        output_script, device_id, python_exe, sections,
        timeout=args.t, bundle_path=bundle_path, out_dir=msaicerr_out,
    )

    # ============================================================
    # Phase E: Inter-core sync diagnosis (→ info.txt section 7, only if Phase D fails)
    # ============================================================

    if section6_passed:
        _print_log("INFO", "Section 6 passed, skipping Section 7 and Section 8")
    else:
        is_sync_issue = run_section7_intercore_sync(
            output_script, device_id,
            sections, header_order,
            python_exe, timeout=args.t,
            bundle_path=bundle_path, out_dir=msaicerr_out,
        )

        # ============================================================
        # Phase F: Framework vs CCE root cause analysis (→ info.txt section 8, only if Phase E rules out sync)
        # ============================================================

        if not is_sync_issue:
            run_section8_framework_vs_cce(
                output_script, device_id,
                sections, header_order,
                python_exe, timeout=args.t,
                bundle_path_undef=_BUNDLED_KERNEL_PATH_UNDEF,
                out_dir=msaicerr_out,
            )
        else:
            _print_log("INFO", "Inter-core sync issue identified, skipping Section 8")

    # ============================================================
    # Phase G: Finalize
    # ============================================================

    _rewrite_info_txt(info_txt, preamble, sections, header_order)
    _print_log("INFO", f"debug_info.txt: {debug_info_txt}")
    _print_log("INFO", "All phases completed, please check " + info_txt)


if __name__ == "__main__":
    main()
