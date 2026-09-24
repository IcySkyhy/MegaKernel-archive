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

"""``pypto_compile_op`` — a drop-in replacement for the real ``asc_op_compiler.compile_op`` *leaf*.

The whole surrounding asc_opc flow is reused verbatim (op-store lookup, ``SingleOpCompile`` with its
``build_config`` + ``op_context``, ``SingleOpPostCompile`` writing ``supportInfo`` / json post-process).
The generated per-op wrapper (``ascendc_impl_build.py``) builds the exact same ``options`` / ``OpInfo`` as
the template wrapper and, instead of ``compile_op(src_cpp, ...)``, calls **this** with the PyPTO DSL ``.py``
as ``cce_file``.

The *only* PyPTO-specific difference is the kernel source: the template kernel is one
``template<TEMPLATE_PARAMS> .cpp`` compiled N times; PyPTO folds each tilingkey to a constant and codegens
a distinct concrete ``kernel.cpp`` per key. So this function does per-tilingkey codegen and compiles each
concrete key, while **reusing the real ``asc_op_compile_base`` backend leaves** for everything else:

- ``compile_pre_process`` / ``_add_op_compile_options_by_customized_json`` / ``_update_compile_option``
  (+ ``CompileOptionTuple``) — identical option preprocessing + ``global_var_storage`` / bisheng-path bootstrap;
- ``gen_compile_cmd_v220`` — the bisheng compile command (replaces the hand-ported flag list);
- ``get_ktype_section_variable`` — the ``FunLevelMixCoreType`` meta section;
- ``fatbin_objs`` — the fat ``.o`` link;
- ``CommonUtility.get_kernel_meta_dir`` / ``get_distinct_filename_tag`` — the real flat ``kernel_meta`` layout.

Artifacts land flat in ``kernel_meta`` exactly like the real compiler: ``<kernel>_mix_aic_<tk>.o`` /
``<kernel>_mix_aiv_<tk>.o`` per (tilingkey, core), one fat ``<kernel>.o`` and ``<kernel>.json``. Per-key
codegen output (``kernel.cpp`` + the per-key wrapper ``.cpp``) is kept on disk for debugging (not cleaned).
The json path is recorded via ``op_context.add_build_res("json_file_path", ...)`` so the reused
``SingleOpCompile`` picks it up and ``SingleOpPostCompile`` appends ``supportInfo``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import time

from asc_op_compile_base.asc_op_compiler.ascendc_common_utility import CommonUtility, CompileInfo
from asc_op_compile_base.asc_op_compiler.ascendc_compile_base import (
    compile_multi_tilingkey,
    compile_pre_process,
    fatbin_objs,
    link_relocatable,
)
from asc_op_compile_base.asc_op_compiler.ascendc_compile_dfx import (
    DFXArgInfo,
    DFXParamType,
    DFXPointType,
    DFXSectionGenerator,
)
from asc_op_compile_base.asc_op_compiler.ascendc_compile_gen_json import (
    _generate_final_json,
)
from asc_op_compile_base.asc_op_compiler.ascendc_compile_utils import check_if_gen_placehoder
from asc_op_compile_base.asc_op_compiler.ascendc_compile_v220 import (
    gen_compile_cmd_v220,
    get_ktype_section_variable,
)
from asc_op_compile_base.asc_op_compiler.ascendc_constants import (
    CORE_TYPE_CUBE,
    CORE_TYPE_MIX,
    CORE_TYPE_VEC,
    MIX_CORE_MACRO,
    TILING_KEY_MACRO,
    CompileOptionTuple,
)
from asc_op_compile_base.asc_op_compiler.compile_op import (
    _add_op_compile_options_by_customized_json,
    _json_post_process,
    _update_compile_option,
    handle_compile_options,
    handle_sk_codegen_options,
)
from asc_op_compile_base.asc_op_compiler.get_op_tiling import TilingInfo, get_tiling_info_by_tiling
from asc_op_compile_base.asc_op_compiler.global_storage import global_var_storage
from asc_op_compile_base.asc_op_compiler.kernel_info_infer import KernelInfoInfer
from asc_op_compile_base.common.context import op_context

# --- reused asc_op_compile_base backend leaves -------------------------------------------------------
from asc_op_compile_base.common.utils import log as logger

from pypto_pro import DataType

# (AscendC core channel, kernel symbol/meta suffix, compile-make suffix)
_MIX_CORE_COMPILE_TARGETS = (
    (CORE_TYPE_CUBE, "mix_aic", "aic"),
    (CORE_TYPE_VEC, "mix_aiv", "aiv"),
)
_ORIG_DTYPE_TO_PYPTO = {
    "dt_bool": DataType.BOOL,
    "dt_int8": DataType.INT8,
    "dt_int16": DataType.INT16,
    "dt_int32": DataType.INT32,
    "dt_int64": DataType.INT64,
    "dt_uint8": DataType.UINT8,
    "dt_uint16": DataType.UINT16,
    "dt_uint32": DataType.UINT32,
    "dt_uint64": DataType.UINT64,
    "dt_float16": DataType.FP16,
    "dt_float": DataType.FP32,
    "dt_bf16": DataType.BF16,
    "dt_float8_e4m3fn": DataType.FP8E4M3FN,
    "dt_float8_e5m2": DataType.FP8E5M2,
    "dt_float8_e8m0": DataType.FP8E8M0,
    "dt_fp4_e2m1": DataType.FP4E2M1,
    "dt_fp4_e1m2": DataType.FP4E1M2,
    "dt_hifloat8": DataType.HF8,
}

_ORIG_DTYPE_MACRO_PREFIX = "-DORIG_DTYPE_"


def _load_kernel(op_path: str, main_func: str | None):
    """Import the PyPTO kernel module and locate the target ``_TileJitKernel``."""
    import importlib.util

    from pypto_pro.runtime.jit import _TileJitKernel

    spec = importlib.util.spec_from_file_location("_pypto_opc_kernel_mod", op_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load kernel module from {op_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if main_func:
        kernel = getattr(mod, main_func, None)
        if not isinstance(kernel, _TileJitKernel):
            raise RuntimeError(f"main_func '{main_func}' is not a @pl.jit kernel in {op_path}")
        return kernel
    kernels = [v for v in vars(mod).values() if isinstance(v, _TileJitKernel)]
    if len(kernels) != 1:
        raise RuntimeError(f"main_func not given and module has {len(kernels)} jit kernels; specify main_func")
    return kernels[0]


def _write_tilingkey_header(schema, cg, output_dir: str) -> None:
    """Generate ``<TilingKey>_tilingkey.h`` for binary delivery."""
    t_start = time.perf_counter()
    op_name = cg.kernel_name
    fields = schema._fields
    valid_combos = schema.enumerate_valid()

    header = '#include "ascendc/host_api/tiling/template_argument.h"\n\n'

    for bw in sorted({f.bits for f in fields}):
        header += f"#define ASCENDC_TPL_{bw}_BW {bw}\n"
    header += "\n"

    header += f"ASCENDC_TPL_ARGS_DECL({op_name},\n"
    for field in fields:
        values_str = ", ".join(str(v) for v in field.values)
        comment = f"// bit:{field.offset + field.bits - 1}-{field.offset}"
        header += f"    {comment}\n"
        header += (
            f"    ASCENDC_TPL_UINT_DECL({field.name}, ASCENDC_TPL_{field.bits}_BW, "
            f"ASCENDC_TPL_UI_LIST, {values_str}),\n"
        )
    header += ");\n\n"

    header += "ASCENDC_TPL_SEL(\n"
    for i, combo in enumerate(valid_combos):
        comma = "," if i < len(valid_combos) - 1 else ""
        header += "    ASCENDC_TPL_ARGS_SEL(\n"
        for j, field in enumerate(fields):
            jcomma = "," if j < len(fields) - 1 else ""
            header += f"        ASCENDC_TPL_UINT_SEL({field.name}, ASCENDC_TPL_UI_LIST, {combo[j]}){jcomma}\n"
        header += f"    ){comma}\n"
    header += ");\n"

    tilingkey_path = Path(output_dir) / f"{schema.cls_name}_tilingkey.h"
    tilingkey_path.write_text(header, encoding="utf-8")

    total_combos = 1
    for field in fields:
        total_combos *= len(field.values)
    logger.info(
        "tilingkey header '%s': %d valid / %d total combos | %.3fs",
        tilingkey_path.name,
        len(valid_combos),
        total_combos,
        time.perf_counter() - t_start,
    )


def _signature_parts(entry_params):
    decls, names = [], []
    for typ, name, is_ptr in entry_params:
        decls.append(f"__gm__ {typ}* {name}" if is_ptr else f"{typ} {name}")
        names.append(name)
    return ", ".join(decls), names


# Markers of device-side print usage in the generated kernel source. Any hit means the impl
# needs the no-prefix AscendC wrapper: TPRINT/dump helpers live in pypto_tprint.h, and pl.printf
# lowers to plain cce::printf calls that are rewritten to the same wrapper.
_TPRINT_MARKERS = ("TPRINT(", "cce::printf(", "__pypto_dtype_name", "__pypto_print_val")


def _needs_tprint_support(content: str) -> bool:
    return any(marker in content for marker in _TPRINT_MARKERS)


def _write_ascendc_tprint_header(kernel_meta_dir: str) -> None:
    """Write a no-prefix AscendC dump copy of pypto_tprint.h next to the impl source.

    The shared header (included by default from codegen) prints via cce::printf
    (jit-mode compatible), but the OPC toolchain swallows cce::printf output and
    AscendC::printf prepends a "[AIV Block x/y]" dumphead. The local copy remaps
    __PYPTO_PRINTF to a wrapper around __asc_aicore::scalar_printf_impl with an
    empty dumphead; it is included by a quoted name so the impl resolves it ahead
    of the packaged header.
    """
    jit_macro_def = (
        "#ifndef __PYPTO_PRINTF\n"
        "#if defined(PTOAS_ENABLE_CCE_PRINT) && PTOAS_ENABLE_CCE_PRINT\n"
        "#define __PYPTO_PRINTF(...) cce::printf(__VA_ARGS__)\n"
        "#else\n"
        "#define __PYPTO_PRINTF(...) ((void)0)\n"
        "#endif\n"
        "#endif\n"
    )
    opc_macro_def = (
        "#ifndef __PYPTO_PRINTF\n"
        "template <class... Args>\n"
        "__aicore__ inline void __pypto_aicore_printf(const __gm__ char* fmt, Args&&... args)\n"
        "{\n"
        "    __asc_aicore::enable_asc_diagnostics();\n"
        "    __asc_aicore::scalar_printf_impl(__asc_aicore::DumpType::DUMP_SCALAR, fmt, \"\", args...);\n"
        "}\n"
        "#define __PYPTO_PRINTF(...) __pypto_aicore_printf(__VA_ARGS__)\n"
        "#endif\n"
    )
    src_header = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "include", "pypto_tprint.h")
    tprint_content = open(src_header, "r", encoding="utf-8").read()
    tprint_content = tprint_content.replace(jit_macro_def, opc_macro_def)
    dst_header = os.path.join(kernel_meta_dir, "pypto_tprint.h")
    if not os.path.exists(dst_header) or open(dst_header, "r", encoding="utf-8").read() != tprint_content:
        with open(dst_header, "w", encoding="utf-8") as f:
            f.write(tprint_content)


def _gen_infer_cpp(cg, tilingkey_header: str, kernel_cpp: str) -> str:
    sig, names = _signature_parts(cg.entry_params)
    ws_idx = len(names) - 2 if len(names) >= 2 else None
    inner = list(names)
    ws_lines = ""
    if ws_idx is not None:
        ws = names[ws_idx]
        ws_lines = (
            f"    AscendC::SetSysWorkspaceForce({ws});\n    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace({ws});\n"
        )
        inner[ws_idx] = "usrWorkspace"
    # The infer source is only compiled to extract tiling metadata, so the dump
    # helpers are replaced with no-op stubs and the header include is dropped.
    # This keeps the infer source self-contained (no pypto_tprint.h dependency).
    infer_kernel_cpp = kernel_cpp.replace(
        "#include <pypto_tprint.h>\n", ""
    )
    tprint_stubs = (
        "#define TPRINT(x)\n"
        "template <typename T>\n"
        "__aicore__ inline const __gm__ char* __pypto_dtype_name() { return \"\"; }\n"
        "template <typename V>\n"
        "__aicore__ inline void __pypto_print_val(V) {}\n"
    )
    return (
        '#include "kernel_operator.h"\n'
        f'#include "{tilingkey_header}"\n'
        f"{tprint_stubs}"
        f"{infer_kernel_cpp}\n"
        f'extern "C" __global__ AICORE void {cg.kernel_name}({sig})\n'
        "{\n"
        f"    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);\n"
        f"{ws_lines}"
        f"    {cg.kernel_name}_impl({', '.join(inner)});\n"
        "}\n"
    )


def generate_binary_headers(kernel) -> str:
    """Generate the tiling-data, tilingkey and infer source required by binary delivery."""
    from pypto_pro.runtime.jit import _codegen, _setup_arch_env, _TileJitKernel

    if not isinstance(kernel, _TileJitKernel):
        raise TypeError("generate_binary_headers() expects a @pl.jit kernel")
    if kernel.tilingkey_schema is None:
        raise ValueError(
            f"generate_binary_headers() requires a tiling_key schema, but kernel "
            f"'{kernel.__name__}' has none; binary delivery needs a tilingkey header"
        )

    arch = _setup_arch_env(kernel.arch)
    schema = kernel.tilingkey_schema
    valid_combos = schema.enumerate_valid()
    if not valid_combos:
        raise ValueError(f"tiling_key schema '{schema.cls_name}' has no valid tilingkey combination")
    concrete_key = dict(zip(schema.field_names(), valid_combos[0]))
    packed = schema.pack(concrete_key)
    datatype_schema = kernel.datatype_schema
    datatype_consts = None
    if datatype_schema is not None:
        datatype_consts = {var_name: DataType.FP16 for var_name in set(datatype_schema.values())}

    cg = _codegen(
        kernel.to_kernel_def(concrete_key, datatype_consts),
        arch,
        clean_up=False,
        tilingkey_packed=packed,
    )
    if cg is None:
        raise RuntimeError(f"Failed to generate code for kernel '{kernel.__name__}'")

    binary_dir = os.path.join(os.path.dirname(cg.build_dir), "binary")
    Path(binary_dir).mkdir(parents=True, exist_ok=True)

    tiling_headers = list(Path(cg.build_dir).glob("*_tiling.h"))
    if not tiling_headers:
        raise RuntimeError(
            f"kernel '{kernel.__name__}' produced no tiling struct header; binary delivery "
            f"requires a tiling-class parameter so a '*_tiling.h' is generated"
        )
    for header in tiling_headers:
        shutil.copy(str(header), os.path.join(binary_dir, header.name))

    _write_tilingkey_header(schema, cg, output_dir=binary_dir)
    tilingkey_header = f"{schema.cls_name}_tilingkey.h"
    kernel_cpp = Path(cg.build_dir, "kernel.cpp").read_text(encoding="utf-8")
    infer_cpp = _gen_infer_cpp(cg, tilingkey_header, kernel_cpp)
    _write(os.path.join(binary_dir, f"{cg.kernel_name}_pypto_infer.cpp"), infer_cpp)
    return binary_dir


def _op_info_get(op_info, name, default=None):
    """OpInfo is an object (attrs) in the real flow; tolerate a dict too."""
    if isinstance(op_info, dict):
        return op_info.get(name, default)
    return getattr(op_info, name, default)


def _orig_dtype_to_pypto(dtype_name: str, param_name: str) -> DataType:
    if not isinstance(dtype_name, str):
        raise RuntimeError(f"param '{param_name}' dtype must be a string, got {type(dtype_name).__name__}")
    key = dtype_name.lower()
    dtype = _ORIG_DTYPE_TO_PYPTO.get(key)
    if dtype is None:
        raise RuntimeError(f"param '{param_name}' dtype '{dtype_name}' is not supported by PyPTO datatype")
    return dtype


def _extract_datatype_key_from_compile_options(compile_options, datatype_schema) -> dict | None:
    """Build the launch dtype dict from AscendC wrapper's -DORIG_DTYPE_<PARAM>=<dtype> options."""
    if datatype_schema is None:
        return None
    wanted = set(datatype_schema)
    by_name = {}
    for option in compile_options or []:
        if not isinstance(option, str) or not option.startswith(_ORIG_DTYPE_MACRO_PREFIX):
            continue
        macro = option[len(_ORIG_DTYPE_MACRO_PREFIX):]
        if "=" not in macro:
            continue
        param_macro, dtype_name = macro.split("=", 1)
        param_name = param_macro.lower()
        if param_name in wanted and param_name not in by_name:
            by_name[param_name] = _orig_dtype_to_pypto(dtype_name, param_name)
    missing = sorted(wanted - set(by_name))
    if missing:
        raise RuntimeError(f"compile_options is missing ORIG_DTYPE macros for datatype params: {missing}")
    return by_name


def _setup_options(op_info, compile_options, op_compile_option, extend_options):
    """Mirror ``compile_op`` opening (compile_op.py:1378-1397): build the CompileOptionTuple and run the
    real option preprocessing. ``compile_pre_process`` also bootstraps the bisheng path + ``global_var_storage``
    (it internally calls ``CommonUtility.get_ascendc_compiler_path``). Runs inside the reused ``build_config``."""
    global_var_storage.global_storage_reset()
    opt = CompileOptionTuple([] if compile_options is None else list(compile_options), [])
    impl_mode = _op_info_get(op_info, "impl_mode")
    if CommonUtility.is_c310() and isinstance(impl_mode, str) and impl_mode != "":
        impl_mode_def = f"-D{impl_mode.upper()}_"
        if impl_mode_def not in opt.compile_options:
            opt.compile_options.append(impl_mode_def)
    _add_op_compile_options_by_customized_json(op_compile_option, opt)
    opt.compile_options = compile_pre_process(op_info, opt.compile_options)
    _update_compile_option(_op_info_get(op_info, "kernel_name"), opt.compile_options, extend_options)
    opt.compile_options.append("-DASCENDC_TPL_KERNEL")
    opt.compile_options.append("--cce-enable-print")
    return opt


def _core_arch(core_type: int) -> str:
    chip_version = CommonUtility.get_chip_version()
    suffix = "cube" if core_type == CORE_TYPE_CUBE else "vec"
    return f"dav-{chip_version}-{suffix}"


def _normalize_dfx_op_info(op_info):
    if _op_info_get(op_info, "mc2_ctx") is not None:
        return op_info
    if hasattr(op_info, "_replace"):
        return op_info._replace(mc2_ctx=[])
    if isinstance(op_info, dict):
        copied = dict(op_info)
        copied["mc2_ctx"] = []
        return copied
    return op_info


def _prepare_dfx(op_info, tiling_info: TilingInfo, compile_info: CompileInfo) -> object:
    dfx_op_info = _normalize_dfx_op_info(op_info)
    dfx_generator = DFXSectionGenerator()
    dfx_generator.dfx_info_reset(dfx_op_info)
    if not dfx_generator.is_support:
        return dfx_op_info

    inputs = _op_info_get(dfx_op_info, "inputs") or []
    outputs = _op_info_get(dfx_op_info, "outputs") or []
    mc2_ctx = _op_info_get(dfx_op_info, "mc2_ctx") or []
    needs_ffts = compile_info.code_channel == CORE_TYPE_MIX and not CommonUtility.is_c310()
    needs_ffts = needs_ffts and not CommonUtility.is_m510()
    if needs_ffts:
        dfx_generator.insert_param(DFXArgInfo("ffts", DFXParamType.FFTS))
    for ctx_name in mc2_ctx:
        dfx_generator.insert_param(DFXArgInfo(ctx_name, DFXParamType.MC2CTX))
    for input_info in inputs:
        if input_info is not None:
            dfx_generator.insert_param(DFXArgInfo(input_info["param_name"], DFXParamType.INPUT))
    for output_info in outputs:
        if output_info is not None:
            dfx_generator.insert_param(DFXArgInfo(output_info["param_name"], DFXParamType.OUTPUT))
    output_shape_depend = _op_info_get(dfx_op_info, "output_shape_depend_on_compute") or []
    if len(output_shape_depend) > 0:
        dfx_generator.insert_param(DFXArgInfo("shape_tensor", DFXParamType.SHAPE_TENSOR))
        for index in output_shape_depend:
            parameter = dfx_generator.get_param(outputs[index]["param_name"])
            parameter.point_type = DFXPointType.LEVEL_1_FOR_SHAPE_TENSOR
        if tiling_info.static_shape_flag:
            dfx_generator.set_size_of_dfx_info("shape_tensor", len(output_shape_depend) * 8 * 8)
    if not tiling_info.static_shape_flag or tiling_info.static_workspace_size >= 0:
        dfx_generator.insert_param(DFXArgInfo("workspace", DFXParamType.WORKSPACE))
    if not tiling_info.static_shape_flag:
        dfx_generator.insert_param(DFXArgInfo("tiling", DFXParamType.TILING))
    dfx_generator.generate_dfx_binary(compile_info, dfx_op_info, tiling_info)
    return dfx_op_info


def _build_compile_info(cce_file, kernel_name, origin_func_name, op_info, infered_info_from_ifile, compile_log_path):
    compile_info = CompileInfo()
    compile_info.src_file = cce_file
    compile_info.dst_file = os.path.join(CommonUtility.get_kernel_meta_dir(), f"{kernel_name}.o")
    compile_info.kernel_name = kernel_name
    compile_info.origin_func_name = origin_func_name
    compile_info.op_type = _op_info_get(op_info, "op_type")
    compile_info.code_channel = infered_info_from_ifile.code_channel
    compile_info.tiling_key_list = infered_info_from_ifile.tiling_key_list
    compile_info.tiling_key_group_map = infered_info_from_ifile.tiling_key_group_map
    compile_info.compile_log_path = compile_log_path
    compile_info.hard_sync = infered_info_from_ifile.hard_sync
    compile_info.enable_deterministic = infered_info_from_ifile.enable_deterministic
    compile_info.tiling_key_deterministic = infered_info_from_ifile.tiling_key_deterministic
    compile_info.tiling_key_kernel_type = infered_info_from_ifile.tiling_key_kernel_type
    compile_info.raw_tiling_key_kernel_type = copy.deepcopy(compile_info.tiling_key_kernel_type)
    compile_info.no_set_kernel_type = infered_info_from_ifile.no_set_kernel_type
    compile_info.default_kernel_type = infered_info_from_ifile.default_kernel_type
    compile_info.dump_info = {"dump_type": "", "dump_size": 1024}
    compile_info.template_tiling_info = infered_info_from_ifile.template_tiling_info
    compile_info.tiling_key_struct_map = infered_info_from_ifile.tiling_key_struct_map
    compile_info.register_tiling_struct = infered_info_from_ifile.register_tiling_struct
    compile_info.tpl_tiling_struct = infered_info_from_ifile.tpl_tiling_struct
    handle_sk_codegen_options(compile_info, infered_info_from_ifile)
    return compile_info


def _gen_binary_meta_sections(kernel_name):
    binsec = '".ascend.meta"'
    return (
        f"static const struct BinaryMetaVersion {kernel_name}_kernel_metainfo_version_section "
        f"__attribute__ ((used, section ({binsec}))) = {{{{B_TYPE_BIN_VERSION_INFO, sizeof(unsigned int)}}, 0x01}};\n"
        f"static const struct BinaryMetaDebug {kernel_name}_kernel_metainfo_debug_section "
        f"__attribute__ ((used, section ({binsec}))) = {{{{B_TYPE_DEBUG_INFO, 8}}, 0, 0}};\n"
        f"static const struct BinaryMetaDynamicParam {kernel_name}_kernel_metainfo_dynamicparam_section "
        f"__attribute__ ((used, section ({binsec}))) = {{{{B_TYPE_DYNAMIC_PARAM, 4}}, 0, 0}};\n"
        f"static const struct BinaryMetaOptionalParam {kernel_name}_kernel_metainfo_optionalparam_section "
        f"__attribute__ ((used, section ({binsec}))) = {{{{B_TYPE_OPTIONAL_PARAM, 4}}, 1, 1}};\n"
    )


def _gen_meta_sections(kernel_name, packed, tiling_info: TilingInfo, compile_info: CompileInfo):
    """Per-key ``.ascend.meta`` sections, using AscendC's ktype and DFX generators."""
    out = []
    dfx_generator = DFXSectionGenerator()
    dfx_generator.gen_dfx_struct_flag = False
    old_sub_core_type = compile_info.sub_core_type
    for core_type, channel, _short in _MIX_CORE_COMPILE_TARGETS:
        section_kernel = f"{kernel_name}_{packed}_{channel}"
        out.append(
            get_ktype_section_variable(
                f"{section_kernel}_section",
                section_kernel,
                compile_info.tiling_key_kernel_type[str(packed)],
            )
        )
        compile_info.sub_core_type = core_type
        out.append(dfx_generator.generate_dfx_section(str(packed), tiling_info, section_kernel, compile_info))
    compile_info.sub_core_type = old_sub_core_type
    out.append(_gen_binary_meta_sections(kernel_name))
    return "".join(out)


def _gen_key_src(kernel_cpp_path, origin_func, impl_name, entry_params, kernel_name, packed, tiling_info, compile_info):
    """The per-key src ``.cpp``: ``kernel_operator.h`` + the codegen'd ``kernel.cpp`` + a concrete
    ``__global__`` entry forwarding to ``<impl>`` (workspace offset like AscendC's gen_kernel_fun) + the
    per-key meta sections. Compiled once with ``-DTILING_KEY_VAR=<packed>``."""
    sig, names = _signature_parts(entry_params)
    _ws_idx = len(names) - 2 if len(names) >= 2 else None
    inner = list(names)
    return (
        '#include "kernel_operator.h"\n'
        f'#include "{kernel_cpp_path}"\n'
        f"__aicore__ inline __attribute__((always_inline)) void ascendc_auto_gen_{origin_func}_kernel({sig})\n"
        "{\n"
        f"    {impl_name}({', '.join(inner)});\n"
        "}\n"
        f'extern "C" __global__ AICORE void auto_gen_{origin_func}_kernel({sig})\n'
        "{\n"
        f"    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);\n"
        f"    ascendc_auto_gen_{origin_func}_kernel({', '.join(names)});\n"
        "}\n" + _gen_meta_sections(kernel_name, packed, tiling_info, compile_info)
    )


def _decode_tiling_key(schema, packed: int):
    if schema is None:
        return None
    concrete = {}
    for field in schema._fields:
        value_index = (packed >> field.offset) & ((1 << field.bits) - 1)
        if value_index >= len(field.values):
            raise RuntimeError(
                f"tilingkey field '{field.name}' index {value_index} is out of range for values {list(field.values)}"
            )
        concrete[field.name] = field.values[value_index]
    return concrete


def _find_infer_cpp(cce_file: str, origin_func_name: str) -> str:
    cce_dir = Path(cce_file).resolve().parent
    direct = cce_dir / f"{origin_func_name}_pypto_infer.cpp"
    if direct.is_file():
        return str(direct)
    matches = sorted(cce_dir.glob("*_pypto_infer.cpp"))
    if len(matches) == 1:
        return str(matches[0])
    raise RuntimeError(f"cannot find PyPTO infer cpp next to {cce_file}: expected {direct}")


def _filter_tiling_keys(tiling_key_list, extend_options, ctx, kernel_name):
    if "customized_tiling_key_list" in extend_options:
        context_tiling_key = extend_options.get("customized_tiling_key_list")
    else:
        context_tiling_key = ctx.get_addition("tiling_key") if ctx is not None else None
    if not context_tiling_key:
        return list(tiling_key_list)
    wanted = {str(int(k)) for k in context_tiling_key}
    filtered = [tiling_key for tiling_key in tiling_key_list if str(int(tiling_key)) in wanted]
    if not filtered:
        raise RuntimeError(f"None of the given tiling keys are supported by {kernel_name}: {context_tiling_key}")
    return filtered


def pypto_compile_op(
    cce_file,
    origin_func_name,
    op_info,
    compile_options=None,
    code_channel=-1,
    op_compile_option="{}",
    extend_options=None,
):
    """PyPTO leaf replacing ``asc_op_compiler.compile_op``. Signature-compatible; ``cce_file`` is the PyPTO
    DSL ``.py``. Writes the flat ``kernel_meta`` artifacts + ``<kernel>.o``/``.json`` and records the json
    path into the op_context for the reused post-compile step. Called inside the reused ``build_config`` +
    ``op_context`` of ``SingleOpCompile``.
    """
    from pypto_pro.runtime.jit import (
        _codegen,
        _setup_arch_env,
        _validate_datatype_key,
    )

    extend_options = extend_options or {}
    kernel_name = _op_info_get(op_info, "kernel_name")
    arch = _pypto_arch()
    _setup_arch_env(arch)
    opt = _setup_options(op_info, compile_options, op_compile_option, extend_options)

    kernel = _load_kernel(cce_file, origin_func_name)
    schema = getattr(kernel, "_tilingkey_schema", None)
    datatype_schema = getattr(kernel, "_datatype_schema", None)
    dtype_key = _extract_datatype_key_from_compile_options(compile_options, datatype_schema)
    dtype_consts = _validate_datatype_key(datatype_schema, dtype_key)

    ctx = op_context.get_context()

    kernel_meta_dir = CommonUtility.get_kernel_meta_dir()
    distinct_tag = CommonUtility.get_distinct_filename_tag()
    compile_log_path = None
    if global_var_storage.get_variable("ascendc_compile_debug_config"):
        compile_log_path = os.path.join(kernel_meta_dir, kernel_name + distinct_tag + ".log")
    infer_cpp = _find_infer_cpp(cce_file, origin_func_name)
    infer_i = os.path.join(kernel_meta_dir, kernel_name + ".i")
    logger.info("pypto_compile_op: infer cpp=%s -> %s", infer_cpp, infer_i)
    infered_info = KernelInfoInfer.get_tiling_key_list_and_simple_infer_code_channel(
        op_info,
        infer_cpp,
        infer_i,
        opt,
        compile_log_path,
        origin_func_name,
    )
    is_const_propagation = "-DFORCE_TILING_CONST_PROPAGATION" in opt.compile_options
    global_var_storage.set_variable("ascendc_tiling_const_propagation", is_const_propagation)
    tiling_info = get_tiling_info_by_tiling(
        op_info,
        infered_info,
        extend_options.get("valueDepend"),
        origin_func_name,
    )
    tiling_data_file_path = os.path.join(kernel_meta_dir, kernel_name + distinct_tag + "_tiling_data.h")
    tiling_info.save_file(tiling_data_file_path)
    global_var_storage.set_variable("ascendc_is_static_op", tiling_info.static_shape_flag)
    tiling_keys = _filter_tiling_keys(infered_info.tiling_key_list, extend_options, ctx, kernel_name)
    compile_info = _build_compile_info(cce_file, kernel_name, origin_func_name, op_info, infered_info, compile_log_path)
    compile_info.tiling_key_list = tiling_keys
    logger.info(
        "pypto_compile_op: op=%s kernel_name=%s arch=%s keys=%d dtype=%s -> %s",
        _op_info_get(op_info, "op_type"),
        kernel_name,
        arch,
        len(tiling_keys),
        dtype_key or "none",
        kernel_meta_dir,
    )

    op_info = _prepare_dfx(op_info, tiling_info, compile_info)
    obj_files: list[str] = []
    cmds_by_core: dict[str, list] = {ch: [] for _, ch, _short in _MIX_CORE_COMPILE_TARGETS}
    compile_options_ready = False
    for tiling_key in tiling_keys:
        packed = int(tiling_key)
        concrete = _decode_tiling_key(schema, packed)

        # Codegen into a per-key subdir under this kernel's (unique) kernel_meta dir. asc_opc forks one
        # process per dtype-combo, all sharing cwd — the default ./build/<prog_name>__<arch>/tk_<packed>
        # location is keyed only on prog.name, so concurrent processes would clobber each other's
        # kernel.cpp (yielding empty/half-written copies). kernel_meta_dir is unique per kernel_name.
        cg_dir = os.path.join(kernel_meta_dir, f"_cg_tk_{packed}")
        cg = _codegen(
            kernel.to_kernel_def(concrete, dtype_consts),
            arch,
            clean_up=False,
            tilingkey_packed=packed,
            out_dir=cg_dir,
        )
        if cg is None:
            raise RuntimeError(f"pypto codegen failed for tilingkey {packed}")
        origin_func = cg.kernel_name
        impl_name = f"{origin_func}_impl"
        if not compile_options_ready:
            workspace_idx = len(cg.entry_params) - 2 if len(cg.entry_params) >= 2 else 0
            handle_compile_options(compile_info, opt, tiling_info, workspace_idx)
            compile_options_ready = True

        # Co-locate the codegen'd kernel.cpp (+ its tiling headers) into kernel_meta under a per-key name,
        # so the per-key src #includes it by a bare relative name (no absolute path; bisheng resolves a
        # quoted include against the including file's own dir). Tiling headers are dtype/key-agnostic
        # (same content across keys) — copying per key is idempotent.
        impl_cpp_name = f"{kernel_name}_{packed}_impl.cpp"
        impl_src = os.path.join(cg.build_dir, "kernel.cpp")
        impl_dst = os.path.join(kernel_meta_dir, impl_cpp_name)
        impl_content = open(impl_src, "r", encoding="utf-8").read()
        needs_tprint_support = _needs_tprint_support(impl_content)
        impl_content = impl_content.replace("cce::printf", "__pypto_aicore_printf")
        if needs_tprint_support:
            # Binary-mode TPRINT helpers live in the shared pypto_tprint.h header, which uses
            # cce::printf for jit-mode compatibility. The OPC toolchain has no working
            # cce::printf output, so drop a no-prefix AscendC dump copy next to the impl and
            # resolve it by a quoted local name (the include itself is emitted by codegen).
            _write_ascendc_tprint_header(kernel_meta_dir)
            impl_content = impl_content.replace(
                '#include <pypto_tprint.h>\n', '#include "pypto_tprint.h"\n'
            )
        with open(impl_dst, "w", encoding="utf-8") as f:
            f.write(impl_content)
        for hdr in (f for f in os.listdir(cg.build_dir) if f.endswith(".h")):
            shutil.copyfile(os.path.join(cg.build_dir, hdr), os.path.join(kernel_meta_dir, hdr))

        src = _gen_key_src(
            impl_cpp_name, origin_func, impl_name, cg.entry_params, kernel_name, packed, tiling_info, compile_info
        )
        src_path = os.path.join(kernel_meta_dir, f"{kernel_name}_{packed}_kernel.cpp")
        _write(src_path, src)

        for core_type, channel, _short in _MIX_CORE_COMPILE_TARGETS:
            dst_o = os.path.join(kernel_meta_dir, f"{kernel_name}_{channel}_{packed}.o")
            sub_arch = _core_arch(core_type)
            cmd = gen_compile_cmd_v220(src_path, dst_o, opt, sub_arch, "")  # kernel.cpp self-includes its tiling.h
            cmd += [f"-I{os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'include')}"]
            cmd += [f"-D{TILING_KEY_MACRO}={packed}UL"]
            cmd += [f"-D{MIX_CORE_MACRO}=1"]
            if CommonUtility.is_c310():
                cmd += ["-D__ASCENDC_ENABLE_VEC_TAIL_TILING_COPY__"]
            cmd += [f"-Dauto_gen_{origin_func}_kernel={kernel_name}_{packed}_{channel}"]
            cmd += [f"-D{impl_name}={impl_name}_{packed}"]
            cmds_by_core[channel].append(cmd)
            obj_files.append(dst_o)

    # Per-key compile via the real backend: compile_multi_tilingkey writes <op>_tmp_<core>_<pid>.mk (one
    # target per tilingkey, +a -E precompile line under dump_cce) and runs `make -j`, exactly as
    # compile_op's compile_kernel_and_meta does. One .mk per core (aic/aiv), mirroring the golden layout.
    for _core_type, channel, short in _MIX_CORE_COMPILE_TARGETS:
        compile_multi_tilingkey(tiling_keys, cmds_by_core[channel], f"{kernel_name}_tmp_{short}", compile_log_path)
    # compile_multi_tilingkey swallows make failures unless build-log is enabled; verify every obj landed.
    missing = [o for o in obj_files if not os.path.exists(o)]
    if missing:
        raise RuntimeError(f"pypto compile failed (make); missing {len(missing)} objs e.g. {missing[:3]}")

    dst_o = os.path.join(kernel_meta_dir, f"{kernel_name}.o")
    fatbin_objs(obj_files, dst_o, compile_info.is_debug, compile_log_path)
    link_relocatable(dst_o)

    json_path = os.path.join(kernel_meta_dir, f"{kernel_name}.json")
    _generate_final_json(compile_info, tiling_info)
    _json_post_process(
        compile_info,
        op_info,
        tiling_info,
        check_if_gen_placehoder(op_info, True),
        check_if_gen_placehoder(op_info, False),
        compile_log_path,
    )
    ctx.add_build_res("json_file_path", os.path.abspath(json_path))
    logger.info("pypto_compile_op done: %s (%d tilingkeys)", json_path, len(tiling_keys))


def _pypto_arch() -> str:
    """SOC -> pypto codegen arch. ops-transformer sets PYPTO_JIT_ARCH at configure time; default a5."""
    return os.environ.get("PYPTO_JIT_ARCH") or "a5"


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
