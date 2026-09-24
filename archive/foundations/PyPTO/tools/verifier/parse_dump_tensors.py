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
""" """

import argparse
import copy
from itertools import groupby
import json
import logging
import multiprocessing
import os
import struct
import time
from typing import Dict

import ml_dtypes
import numpy as np
import pandas as pd
from tensor_diff import IsCloseConfig, compare_tensors_result_dict
import torch

# ===================== 核心配置（需和C/C++端一致）=====================
DEV_SHAPE_DIM_MAX = 6  # 替换为实际值
BYTE_ORDER = "<"  # 小端：< ；大端：> ；本机字节序：=
RESULT_FILE = ""

# 单个字段的字节数定义（无对齐，纯原始字节）
FIELD_SIZES = {"uint8_t": 1, "uint16_t": 2, "uint32_t": 4, "int32_t": 4, "int64_t": 8, "uint64_t": 8}


logging.basicConfig(
    level=logging.DEBUG,  # 日志级别：DEBUG < INFO < WARNING < ERROR < CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式（含时间、级别、内容）
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("app.log", encoding="utf-8"),  # 输出到文件（持久化）
    ],
)


_data_type_full_mapping = {
    0: ("INT4", ml_dtypes.int4),
    1: ("INT8", np.int8),
    2: ("INT16", np.int16),
    3: ("INT32", np.int32),
    4: ("INT64", np.int64),
    5: ("FP8", ml_dtypes.float8_e4m3fn),
    6: ("FP16", np.float16),
    7: ("FP32", np.float32),
    8: ("BF16", ml_dtypes.bfloat16),
    9: ("HF4", None),  # 暂不支持解析
    10: ("HF8", None),  # 暂不支持解析
    11: ("UINT8", np.uint8),
    12: ("UINT16", np.uint16),
    13: ("UINT32", np.uint32),
    14: ("UINT64", np.uint64),
    15: ("BOOL", np.bool_),
    16: ("DOUBLE", np.float64),
    17: ("BOTTOM", None),
}


def _get_data_type(data_type: int):
    """Convert data type numeric value to readable string"""
    return _data_type_full_mapping.get(data_type, f"UNKNOWN({data_type})")


def _get_dtype_from_str(dtype_str: str):
    str_to_dtype = {v[0]: v[1] for k, v in _data_type_full_mapping.items()}
    dtype_str = dtype_str.strip().upper()
    return str_to_dtype.get(dtype_str, None)


def _get_compare_config(dtype):
    """
    根据数据类型返回合适的对比配置

    Args:
        dtype: numpy dtype 对象，可能为 None

    Returns:
        IsCloseConfig: 对比配置对象，或 None（表示不支持的类型）
    """
    if dtype is None:
        return None

    # 整型数据：精确匹配
    if np.issubdtype(dtype, np.integer):
        return IsCloseConfig(rtol=0, atol=0, calc_dtype=torch.float64, is_detail=True)

    # FP32/FP64：标准容差
    if dtype in [np.float32, np.float64]:
        return IsCloseConfig(rtol=1e-3, atol=1e-3, calc_dtype=torch.float64, is_detail=True)

    # FP16/BF16/FP8 等低精度浮点：放宽容差
    return IsCloseConfig(rtol=1e-2, atol=1e-2, calc_dtype=torch.float64, is_detail=True)


class VerifyRes:
    def __init__(self):
        self.verify_codegen_op_info_list = None
        self.verify_tensorgraph_op_info_list = None
        self.verify_path = ""

    @staticmethod
    def parse_loop_info(loop_info_str):
        """
        解析 LOOP_INFO 字符串为 dict

        Args:
            loop_info_str: 's_idx=0@b_idx=0@loop_idx_0=0'

        Returns:
            {'s_idx': 0, 'b_idx': 0, 'loop_idx_0': 0}
        """
        if not loop_info_str:
            return {}

        result = {}
        pairs = loop_info_str.split('@')
        for pair in pairs:
            if '=' in pair:
                name, value = pair.split('=', 1)
                try:
                    result[name] = int(value)
                except ValueError:
                    result[name] = value

        return result

    @staticmethod
    def match_loop_info(tensor_loop_dict, verify_loop_dict):
        """
        只比较原始 loop_idx_*。
        interpreter LOOP_INFO 常带 s_idx/b_idx；dump 可能带 CF 余数循环名。
        """
        if not tensor_loop_dict or not verify_loop_dict:
            return False

        verify_loops = {name: value for name, value in verify_loop_dict.items() if "loop_idx_" in name}
        if not verify_loops:
            return False

        for name, value in verify_loops.items():
            if tensor_loop_dict.get(name) != value:
                return False

        return True

    @staticmethod
    def loop_dict_to_str(loop_dict):
        """
        将 loop info dict 转换为字符串格式

        Args:
            loop_dict: {'s_idx': 0, 'b_idx': 0, 'loop_idx_0': 0}

        Returns:
            's_idx=0@b_idx=0@loop_idx_0=0'
        """
        if not loop_dict:
            return ""
        return "@".join(f"{k}={v}" for k, v in loop_dict.items())

    @staticmethod
    def _compare_codegen_tensors(tensor_infos, tensor_infos_new):
        for i, tensor_info in enumerate(tensor_infos_new):
            dump_tshape = tensor_info.get("B>:validshape")
            verify_tensor_info = tensor_info["verify_dup_tensor"]

            # 复制 B>loopVarInfos 和 LOOP_INFO（已转换为字符串格式）
            if "B>loopVarInfos" in tensor_info:
                tensor_infos[i]["B>loopVarInfos"] = tensor_info["B>loopVarInfos"]
            if "LOOP_INFO" in tensor_info:
                tensor_infos[i]["LOOP_INFO"] = tensor_info["LOOP_INFO"]

            if not verify_tensor_info:
                tensor_infos[i]["AB>RESULT"] = "SKIP"
                tensor_infos[i]["AB>RESULT_REASON"] = "verify file not exist"
                continue

            verify_tshape = tensor_info["valid_shape"]
            tensor_infos[i]["A>PHASE_NAME"] = tensor_info["PHASE_NAME"]
            tensor_infos[i]["A>:validshape"] = verify_tshape
            tensor_infos[i]["A>:datatype"] = tensor_info["A>:datatype"]
            tensor_infos[i]["A>FILENAME"] = tensor_info["verify_dup_tensor"]
            tensor_infos[i]["PATH_FUNC:hash"] = tensor_info.get("PATH_FUNC:hash")
            tensor_infos[i]["A>:rawmagic"] = tensor_info.get("A>:rawmagic")
            tensor_infos[i]["A>:format"] = tensor_info.get("A>:format")
            tensor_infos[i]["A>:symbol"] = tensor_info.get("A>:symbol")
            tensor_infos[i]["A>:magic"] = tensor_info.get("A>:magic")
            tensor_infos[i]["A>:offset"] = tensor_info.get("A>:offset")
            tensor_infos[i]["A>EVAL:dynvalidshape"] = tensor_info.get("A>EVAL:dynvalidshape")
            tensor_infos[i]["A>:opmagic"] = tensor_info.get("A>:opmagic")
            tensor_infos[i]["A>:opcode"] = tensor_info.get("A>:opcode")
            tensor_infos[i]["A>OP_ATTR_SYM_OFFSET"] = tensor_info.get("A>OP_ATTR_SYM_OFFSET")
            tensor_infos[i]["A>OP_ATTR_ATOMIC"] = tensor_info.get("A>OP_ATTR_ATOMIC")
            tensor_infos[i]["A>OP_IO_FLAG"] = tensor_info.get("A>OP_IO_FLAG")

            if os.path.exists(verify_tensor_info) and len(verify_tshape) == len(dump_tshape):
                dtype_result = _get_data_type(tensor_info["datatype"])
                dtype = dtype_result[1]
                verify_data_type = _get_dtype_from_str(tensor_info["A>:datatype"])

                # 不支持的类型，跳过对比
                if dtype is None:
                    tensor_infos[i]["AB>RESULT"] = "SKIP"
                    tensor_infos[i]["AB>RESULT_REASON"] = f"unsupported dtype: {dtype_result[0]}"
                    continue

                verify_tensor_data = np.fromfile(verify_tensor_info, verify_data_type)
                verify_tensor_data = verify_tensor_data.reshape(verify_tshape)

                data = np.fromfile(tensor_info["B>FILENAME"], dtype)
                data = data.reshape(dump_tshape)

                slices = []
                for dim in range(data.ndim):
                    stop = min(verify_tshape[dim], dump_tshape[dim])
                    slices.append(slice(0, stop))

                sliced_data = data[tuple(slices)]
                sliced_verify = verify_tensor_data[tuple(slices)]

                config = _get_compare_config(dtype)
                tensor_a = torch.from_numpy(sliced_data.astype(np.float64)).to(torch.float64)
                tensor_b = torch.from_numpy(sliced_verify.astype(np.float64)).to(torch.float64)
                file_name = tensor_info["B>FILENAME"].split('/')[-1]
                csv_path = os.path.join(RESULT_FILE[:-4] + ".DETAIL", file_name[:-5] + ".csv")
                cmp_result = compare_tensors_result_dict(tensor_a, tensor_b, csv_path, config=config)
                for key, value in cmp_result.items():
                    tensor_infos[i][key] = value

            else:
                tensor_infos[i]["AB>RESULT"] = "SKIP"
                tensor_infos[i]["AB>RESULT_REASON"] = "verify file not exist or shape mismatch"

    def read_verify_result(self, verify_path):
        self.verify_path = verify_path
        verify_res_file = os.path.join(self.verify_path, "verify_graph_data_metainfo.csv")
        if not os.path.exists(verify_res_file):
            logging.error(f"verify path {verify_path} not exist.")
            return

        df = pd.read_csv(verify_res_file, encoding="utf-8")
        df_clean = df.dropna(subset=[":rawmagic"]).copy()
        df_clean[":rawmagic"] = df_clean[":rawmagic"].astype(int)

        codegen_filter = df_clean["PHASE_NAME"].str.contains("_CodegenPreproc", na=False)
        df_codegen = df_clean[codegen_filter]
        df_codegen = df_codegen.dropna(subset=["ROOT_CALL:opmagic"]).copy()
        df_codegen["ROOT_CALL:opmagic"] = df_codegen["ROOT_CALL:opmagic"].astype(int)
        self.verify_codegen_op_info_list = df_codegen

        tensor_graph_filter = df_clean["PHASE_NAME"].str.contains("tensor_graph", na=False)
        self.verify_tensorgraph_op_info_list = df_clean[tensor_graph_filter]

    def get_verify_res_single(self, tensor_info, op_info_list):
        raw_magic = tensor_info.get("ROOT_CALL:rawmagic")
        ioflag = tensor_info.get("IO_FLAG")
        callop_magic = tensor_info.get("ROOT_CALL:opmagic")
        tensor_info_offset_str = '_'.join(str(item) for item in tensor_info.get("B>OP_ATTR_SYM_OFFSET"))

        verify_dup_tensor = ""
        valid_shape = []
        loop_info = ""
        dtype = ""
        loop_matched = False
        op_info_list.sort(key=lambda x: x.get("NO."))  # 按序号排序,序号也是执行顺序

        for op_info in op_info_list:
            if callop_magic != op_info.get("ROOT_CALL:opmagic"):
                continue
            if raw_magic != op_info.get("ROOT_CALL:rawmagic"):
                continue
            # 如果 tensor_loop_dict 存在且非空，要求 verify_loop_dict 存在且匹配
            tensor_loop_dict = tensor_info.get("B>loopVarInfos")
            verify_loop_dict = self.parse_loop_info(op_info.get("LOOP_INFO"))
            if tensor_loop_dict and verify_loop_dict:
                if not self.match_loop_info(tensor_loop_dict, verify_loop_dict):
                    continue

            tensor_info["PHASE_NAME"] = op_info.get("PHASE_NAME")
            if ioflag.startswith("i") and op_info.get(":opcode") in ["COPY_IN", "VIEW"]:
                verify_op_offset = json.loads(op_info.get("OP_ATTR_SYM_OFFSET"))
                verify_op_offset_str = '_'.join(str(item) for item in verify_op_offset)
                if verify_op_offset_str == tensor_info_offset_str:
                    verify_dup_tensor = op_info.get("FILENAME")
                    valid_shape = json.loads(op_info.get(":validshape"))
                    loop_info = op_info.get("LOOP_INFO")
                    dtype = op_info.get(":datatype")
                    if isinstance(tensor_loop_dict, dict) and len(tensor_loop_dict) > 0:
                        loop_matched = True
                    break
            elif ioflag.startswith("o") and op_info.get(":opcode") in ["COPY_OUT"]:
                verify_op_offset = json.loads(op_info.get("OP_ATTR_SYM_OFFSET"))
                verify_op_offset_str = '_'.join(str(item) for item in verify_op_offset)
                if verify_op_offset_str == tensor_info_offset_str:
                    verify_dup_tensor = op_info.get("INPUT:FILENAMES")  # COPY_OUT的op只会有一个输入
                    valid_shape = json.loads(op_info.get("INPUT:validshape"))
                    loop_info = op_info.get("LOOP_INFO")
                    dtype = next(i for i in op_info_list if i["FILENAME"] == verify_dup_tensor).get(":datatype")
                    if isinstance(tensor_loop_dict, dict) and len(tensor_loop_dict) > 0:
                        loop_matched = True
                    break

        if verify_dup_tensor:
            verify_dup_tensor = os.path.join(self.verify_path, op_info.get("PHASE_NAME"), verify_dup_tensor)
            tensor_info["PATH_FUNC:hash"] = op_info.get("PATH_FUNC:hash")
            tensor_info["A>:rawmagic"] = op_info.get(":rawmagic")
            tensor_info["A>:format"] = op_info.get(":format")
            tensor_info["A>:symbol"] = op_info.get(":symbol")
            tensor_info["A>:magic"] = op_info.get(":magic")
            tensor_info["A>:offset"] = op_info.get(":offset")
            tensor_info["A>EVAL:dynvalidshape"] = op_info.get("EVAL:dynvalidshape")
            tensor_info["A>:opmagic"] = op_info.get(":opmagic")
            tensor_info["A>:opcode"] = op_info.get(":opcode")
            tensor_info["A>OP_ATTR_SYM_OFFSET"] = op_info.get("OP_ATTR_SYM_OFFSET")
            tensor_info["A>OP_ATTR_ATOMIC"] = op_info.get("OP_ATTR_ATOMIC")
            tensor_info["A>OP_IO_FLAG"] = op_info.get("OP_IO_FLAG")
        tensor_info["verify_dup_tensor"] = verify_dup_tensor
        tensor_info["valid_shape"] = valid_shape
        tensor_info["loop_info"] = loop_info
        tensor_info["A>:datatype"] = dtype

        # 更新 LOOP_INFO 和 B>loopVarInfos：匹配成功用 verify 的 LOOP_INFO，
        # 失败 LOOP_INFO 置空且 B>loopVarInfos 转字符串
        tensor_loop_dict = tensor_info.get("B>loopVarInfos")
        if isinstance(tensor_loop_dict, dict):
            if loop_matched and loop_info:
                tensor_info["LOOP_INFO"] = loop_info
                tensor_info["B>loopVarInfos"] = "="
            else:
                tensor_info["LOOP_INFO"] = ""
                tensor_info["B>loopVarInfos"] = self.loop_dict_to_str(tensor_loop_dict)

    def process_single_task(self, tensor_infos, op_info_list_callop):
        tensor_infos_new = copy.deepcopy(tensor_infos)
        op_info_list = op_info_list_callop.copy(deep=True)

        for tensor_info in tensor_infos_new:
            self.get_verify_res_single(tensor_info, op_info_list.to_dict(orient='records'))

        self._compare_codegen_tensors(tensor_infos, tensor_infos_new)

    def get_verify_codegen_res(self, callop_tensor_infos):
        res_tensor_infos = []
        if self.verify_codegen_op_info_list is None:
            logging.info("verify codegen op info is None.")
            for tensor_infos in callop_tensor_infos:
                for tensor_info in tensor_infos:
                    loop_var = tensor_info.get("B>loopVarInfos")
                    if isinstance(loop_var, dict):
                        tensor_info["B>loopVarInfos"] = self.loop_dict_to_str(loop_var)
                    tensor_info["LOOP_INFO"] = ""
                res_tensor_infos.extend(tensor_infos)
            return res_tensor_infos

        callop_magic = callop_tensor_infos[0][0].get("ROOT_CALL:opmagic")  # callop
        op_info_list_callop = self.verify_codegen_op_info_list.copy(deep=True)
        op_info_list_callop = op_info_list_callop[op_info_list_callop["ROOT_CALL:opmagic"] == callop_magic]
        for tensor_infos in callop_tensor_infos:
            self.process_single_task(tensor_infos, op_info_list_callop)
            res_tensor_infos.extend(tensor_infos)
        return res_tensor_infos

    def get_verify_tensor_graph_res(self, tensor_info):
        raw_magic = tensor_info.get("ROOT_CALL:rawmagic")

        verify_dup_tensor = ""
        valid_shape = []
        phase_name = ""
        dtype = ""
        loop_info_str = ""
        loop_matched = False

        # verify_tensorgraph_op_info_list
        if self.verify_tensorgraph_op_info_list is None or self.verify_tensorgraph_op_info_list.empty:
            return verify_dup_tensor, valid_shape, phase_name, dtype, loop_info_str

        # 按rawTensorMagic过滤
        filtered_df = self.verify_tensorgraph_op_info_list[
            self.verify_tensorgraph_op_info_list[":rawmagic"] == raw_magic
        ]
        if filtered_df.empty:
            return verify_dup_tensor, valid_shape, phase_name, dtype, loop_info_str

        # 如果 tensor_info 中存在 loop 信息，进行二次匹配
        tensor_loop_dict = tensor_info.get("B>loopVarInfos")
        if tensor_loop_dict:
            matched_indices = []
            for idx, row in filtered_df.iterrows():
                verify_loop_dict = self.parse_loop_info(row.get("LOOP_INFO"))
                if verify_loop_dict and self.match_loop_info(tensor_loop_dict, verify_loop_dict):
                    matched_indices.append(idx)

            if not matched_indices:
                return verify_dup_tensor, valid_shape, phase_name, dtype, loop_info_str

            filtered_df = filtered_df.loc[matched_indices]
            loop_matched = True

        sorted_df = filtered_df.sort_values(by="NO.", ascending=True)
        last_op_info = sorted_df.iloc[-1]
        verify_dup_tensor = last_op_info.get("FILENAME")
        valid_shape = json.loads(last_op_info.get(":validshape"))
        phase_name = last_op_info.get("PHASE_NAME")
        dtype = last_op_info.get(":datatype")
        loop_info_str = last_op_info.get("LOOP_INFO", "")
        if verify_dup_tensor:
            verify_dup_tensor = os.path.join(self.verify_path, last_op_info.get("PHASE_NAME"), verify_dup_tensor)
            tensor_info["PATH_FUNC:hash"] = last_op_info.get("PATH_FUNC:hash")
            tensor_info["A>:rawmagic"] = last_op_info.get(":rawmagic")
            tensor_info["A>:format"] = last_op_info.get(":format")
            tensor_info["A>:symbol"] = last_op_info.get(":symbol")
            tensor_info["A>:magic"] = last_op_info.get(":magic")
            tensor_info["A>:offset"] = last_op_info.get(":offset")
            tensor_info["A>EVAL:dynvalidshape"] = last_op_info.get("EVAL:dynvalidshape")
            tensor_info["A>:opmagic"] = last_op_info.get(":opmagic")
            tensor_info["A>:opcode"] = last_op_info.get(":opcode")
            tensor_info["A>OP_ATTR_SYM_OFFSET"] = last_op_info.get("OP_ATTR_SYM_OFFSET")
            tensor_info["A>OP_ATTR_ATOMIC"] = last_op_info.get("OP_ATTR_ATOMIC")
            tensor_info["A>OP_IO_FLAG"] = last_op_info.get("OP_IO_FLAG")

        # 更新 loop_info_str：匹配成功用 LOOP_INFO 字符串，失败写空
        if loop_matched and loop_info_str:
            pass  # 已经是 LOOP_INFO 字符串
        elif tensor_loop_dict:
            loop_info_str = ""

        return verify_dup_tensor, valid_shape, phase_name, dtype, loop_info_str


_verify_res = VerifyRes()


class CompactDumpTensorInfoParser:
    def __init__(self, dump_tensor_path):
        self.dump_tensor_path = dump_tensor_path
        # 计算单个结构体的紧凑总字节数（无对齐）
        self.struct_compact_size = self._calc_compact_size()
        # 定义字段解析顺序和类型（严格匹配C/C++结构体）
        self.field_specs = [
            ("B>version", "uint8_t"),
            ("B>rsrv_01", "uint8_t"),
            ("B>headSize", "uint16_t"),
            ("B>rsrv_02", "uint32_t"),
            ("B>opId", "uint32_t"),
            ("B>funcId", "uint32_t"),
            ("B>taskId", "uint32_t"),
            ("ROOT_CALL:opmagic", "uint32_t"),
            ("B>blockIdx", "int32_t"),
            ("datatype", "int32_t"),
            ("ROOT_CALL:rawmagic", "int32_t"),
            ("dims", "int32_t"),
            ("B>execStart", "int64_t"),
            ("B>execEnd", "int64_t"),
            ("ROOT_FUNC:hash", "uint64_t"),
            ("FUNC:hash", "uint64_t"),
            ("B>TIMESTAMP", "uint64_t"),
            ("B>:validshape", "uint64_t", DEV_SHAPE_DIM_MAX),
            ("B>OP_ATTR_SYM_OFFSET", "uint64_t", DEV_SHAPE_DIM_MAX),
            ("B>:rawshape", "uint64_t", DEV_SHAPE_DIM_MAX),
            ("B>tensorAddr", "uint64_t"),
            ("B>loopVarCount", "uint64_t"),
        ]
        self.raw_tensor_info = {}
        self.task_tensor_info = {}

    @staticmethod
    def _calc_compact_size():
        """Calculate compact total byte count without alignment"""
        total = 0
        # 基础字段
        total += FIELD_SIZES["uint8_t"] * 2  # version ~ rsrv_01
        total += FIELD_SIZES["uint16_t"]  # headSize
        total += FIELD_SIZES["uint32_t"] * 5  # rsrv_02 ~ callopMagic
        total += FIELD_SIZES["int32_t"] * 4  # blockIdx ~ dims
        total += FIELD_SIZES["int64_t"] * 2  # execStart ~ execEnd
        total += FIELD_SIZES["uint64_t"] * 3  # rootHash ~ timeStamp
        # 数组字段
        array_size = FIELD_SIZES["uint64_t"] * DEV_SHAPE_DIM_MAX
        total += array_size * 3  # shape + offset + rawShape
        # tensorAddr + loopVarCount
        total += FIELD_SIZES["uint64_t"] * 2
        # LoopVarInfo[8]: name[64] + exprIdx(int32) + value(int32) = 72字节
        total += (64 + 4 + 4) * 8
        return total

    @staticmethod
    def _parse_field(bin_data: bytes, offset: int, field_type: str, array_len: int = 1) -> tuple:
        """解析单个字段（支持标量/数组）
        Returns:
            (解析后的值, 字段占用的总字节数)
        """
        field_size = FIELD_SIZES[field_type]
        total_bytes = field_size * array_len
        # verify data length
        if offset + total_bytes > len(bin_data):
            remaining = len(bin_data) - offset
            raise ValueError(
                f"Field parse failed: offset {offset}, need {total_bytes} bytes, remaining {remaining} bytes"
            )

        # 构建单个元素的格式符
        fmt_char = {"uint8_t": "B", "uint16_t": "H", "uint32_t": "I", "int32_t": "i", "int64_t": "q", "uint64_t": "Q"}[
            field_type
        ]
        # 拼接格式符（字节序 + 元素格式符*数量）
        fmt = BYTE_ORDER + fmt_char * array_len

        # 解析数据
        values = struct.unpack_from(fmt, bin_data, offset)
        # 标量返回单个值，数组返回元组
        if array_len == 1:
            return values[0], total_bytes
        else:
            return values, total_bytes

    @staticmethod
    def _parse_loop_var_info(bin_data: bytes, offset: int) -> tuple:
        """
        解析单个 LoopVarInfo 结构

        结构定义：
        - char name[64]: 循环变量名称
        - int32_t exprIdx: exprList 索引
        - int32_t value: 当前值

        Returns:
            (dict, 字节数)
        """
        name_bytes = bin_data[offset:offset + 64]
        name = name_bytes.rstrip(b'\x00').decode('utf-8', errors='replace')
        offset += 64
        offset += 4  # 跳过exprIdx的值

        value = struct.unpack_from(BYTE_ORDER + "i", bin_data, offset)[0]
        offset += 4

        result = {
            name: value,
        }
        return result, 72

    @staticmethod
    def _verify_merged_tensor(merge_tensor_info, raw_data):
        verify_tensor_info, verify_tshape, phase_name, dtype, loop_info_str = _verify_res.get_verify_tensor_graph_res(
            merge_tensor_info
        )
        dump_tshape = merge_tensor_info.get("B>:rawshape")

        merge_tensor_info["A>:validshape"] = verify_tshape
        merge_tensor_info["A>FILENAME"] = verify_tensor_info
        merge_tensor_info["A>PHASE_NAME"] = phase_name
        merge_tensor_info["A>:datatype"] = dtype

        # 更新 LOOP_INFO 和 B>loopVarInfos
        if loop_info_str:
            merge_tensor_info["LOOP_INFO"] = loop_info_str
            merge_tensor_info["B>loopVarInfos"] = "="
        else:
            merge_tensor_info["LOOP_INFO"] = ""
            loop_var = merge_tensor_info.get("B>loopVarInfos")
            if isinstance(loop_var, dict):
                merge_tensor_info["B>loopVarInfos"] = VerifyRes.loop_dict_to_str(loop_var)

        if (
            os.path.exists(verify_tensor_info)
            and len(verify_tshape) == len(dump_tshape)
            and all(vdim == ddim for vdim, ddim in zip(verify_tshape, dump_tshape))
        ):
            dtype_result = _get_data_type(merge_tensor_info["datatype"])
            dtype = dtype_result[1]

            # 不支持的类型，跳过对比
            if dtype is None:
                merge_tensor_info["AB>RESULT"] = "SKIP"
                merge_tensor_info["AB>RESULT_REASON"] = f"unsupported dtype: {dtype_result[0]}"
                return merge_tensor_info

            verify_tensor_data = np.fromfile(verify_tensor_info, dtype)
            verify_tensor_data = verify_tensor_data.reshape(verify_tshape)

            config = _get_compare_config(dtype)
            tensor_a = torch.from_numpy(raw_data.astype(np.float64)).to(torch.float64)
            tensor_b = torch.from_numpy(verify_tensor_data.astype(np.float64)).to(torch.float64)
            file_name = merge_tensor_info["B>FILENAME"].split('/')[-1]
            csv_path = os.path.join(RESULT_FILE[:-4] + ".DETAIL", file_name[:-5] + ".csv")
            cmp_result = compare_tensors_result_dict(tensor_a, tensor_b, csv_path, config=config)
            for key, value in cmp_result.items():
                merge_tensor_info[key] = value

        else:
            merge_tensor_info["AB>RESULT"] = "SKIP"
            merge_tensor_info["AB>RESULT_REASON"] = "verify file not exist or shape mismatch"

        return merge_tensor_info

    def parse_single(self, bin_data: bytes, offset: int = 0) -> dict:
        """Parse a single compactly stored DumpTensorInfo struct"""
        result = {"B>PHASE_NAME": "task_dump"}
        current_offset = offset

        # 逐个解析字段（严格按顺序）
        for spec in self.field_specs:
            if len(spec) == 2:
                # 标量字段：(name, type)
                name, field_type = spec
                value, bytes_used = self._parse_field(bin_data, current_offset, field_type)
            else:
                # 数组字段：(name, type, array_len)
                name, field_type, array_len = spec
                value, bytes_used = self._parse_field(bin_data, current_offset, field_type, array_len)

            result[name] = value
            current_offset += bytes_used

        if result.get("B>version") != 1:
            raise ValueError(f"Version mismatch: expected 1, got {result.get('B>version')}")

        result.pop("B>rsrv_01", None)
        result.pop("B>rsrv_02", None)

        # 解析循环变量信息
        loop_var_count = result.get("B>loopVarCount", 0)
        result["B>loopVarInfos"] = {}

        if loop_var_count > 0:
            actual_count = min(loop_var_count, 16)
            for _ in range(actual_count):
                var_info, bytes_used = self._parse_loop_var_info(bin_data, current_offset)
                result["B>loopVarInfos"].update(var_info)
                current_offset += bytes_used

        dims = result.get("dims")
        del result["dims"]
        if dims > 0 and dims < DEV_SHAPE_DIM_MAX:
            result["B>:validshape"] = result["B>:validshape"][:dims]
            result["B>OP_ATTR_SYM_OFFSET"] = result["B>OP_ATTR_SYM_OFFSET"][:dims]
            result["B>:rawshape"] = result["B>:rawshape"][:dims]

        # 衍生字段（可选）
        result["B>:datatype"] = _get_data_type(result.get("datatype", 17))[0]

        return result

    def parse_file(self, file_path: str) -> list[dict]:
        """Parse the entire compactly stored bin file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            bin_data = f.read()

        tensor_info = self.parse_single(bin_data, 0)
        dtype = _get_data_type(tensor_info["datatype"])[1]
        data = np.frombuffer(bin_data, dtype, offset=tensor_info["B>headSize"])
        bin_file = f"{file_path[:-6]}.data"
        data.tofile(bin_file)

        tensor_info["IO_FLAG"] = bin_file.split("_")[-1][:-5].replace("input", "i").replace("output", "o")
        tensor_info["B>seqNo"] = int(os.path.basename(bin_file).split("_")[1])

        tensor_info["B>FILENAME"] = bin_file

        if tensor_info["IO_FLAG"].startswith("i"):
            tensor_info["B>EXEC_TIMESTAMP"] = tensor_info["B>execStart"]
        else:
            tensor_info["B>EXEC_TIMESTAMP"] = tensor_info["B>execEnd"]

        if tensor_info["IO_FLAG"].startswith("o"):
            key = (tensor_info["ROOT_CALL:rawmagic"], tensor_info["B>tensorAddr"])
            if key not in self.raw_tensor_info:
                self.raw_tensor_info[key] = []
            self.raw_tensor_info[key].append(tensor_info)

        key = (tensor_info["B>taskId"], tensor_info["ROOT_CALL:opmagic"], tensor_info["B>seqNo"])
        if key not in self.task_tensor_info:
            self.task_tensor_info[key] = []
        self.task_tensor_info[key].append(tensor_info)
        return tensor_info

    def get_exec_time(self, prof_data_file):
        """Parse prof_data file to get execution time per task"""
        with open(prof_data_file, "rb") as f:
            data = json.load(f)

        # 构建任务执行时间索引，键为 (blockId, taskId, seqNo)
        exec_time_index = {}
        for block_data in data:
            block_idx = block_data.get("blockIdx")
            for task in block_data.get("tasks", []):
                key = (block_idx, task.get("taskId"), task.get("seqNo"))
                exec_time_index[key] = {"execStart": task.get("execStart", 0), "execEnd": task.get("execEnd", 0)}

        # 批量更新任务执行时间
        for _, tensor_infos in self.task_tensor_info.items():
            if not tensor_infos:
                continue

            block_idx = tensor_infos[0].get("B>blockIdx")
            task_id = tensor_infos[0].get("B>taskId")
            seq_no = tensor_infos[0].get("B>seqNo")
            key = (block_idx, task_id, seq_no)
            if key not in exec_time_index:
                continue
            exec_time = exec_time_index[key]
            for tensor_info in tensor_infos:
                tensor_info["B>execStart"] = exec_time["execStart"]
                tensor_info["B>execEnd"] = exec_time["execEnd"]
                if tensor_info["IO_FLAG"].startswith("i"):
                    tensor_info["B>EXEC_TIMESTAMP"] = exec_time["execStart"]
                else:
                    tensor_info["B>EXEC_TIMESTAMP"] = exec_time["execEnd"]

    def tensor_compare(self):
        logging.info("Start compare tensors.")
        merged_result = []
        if not self.task_tensor_info:
            for _, tensor_infos in self.task_tensor_info.items():
                merged_result.extend(tensor_infos)
            return merged_result

        num_tasks = len(self.task_tensor_info)
        num_cpus = os.cpu_count() or 1
        num_processes = min(16, num_cpus, num_tasks)
        with multiprocessing.Pool(processes=num_processes) as pool:
            tasks = []
            callop_tasks = {}
            # 按callopMagic分组任务
            for _, tensor_infos in self.task_tensor_info.items():
                callop_magic = tensor_infos[0].get("ROOT_CALL:opmagic")
                if callop_magic not in callop_tasks:
                    callop_tasks[callop_magic] = []
                # 将任务添加到对应callopMagic的组中
                tensor_infos.sort(key=lambda x: x.get("B>TIMESTAMP"))
                callop_tasks[callop_magic].append(tensor_infos)

            for _, tensor_infos_list in callop_tasks.items():
                tensor_infos_list.sort(key=lambda x: x[0].get("B>TIMESTAMP"))
                tasks.append(tensor_infos_list)  # 按timeStamp排序

            try:
                results = pool.map(_verify_res.get_verify_codegen_res, tasks)
            except Exception as e:
                logging.exception(f"Tensor comparison failed with error: {e}")
                for tensor_infos in tasks:
                    merged_result.extend(x for sublist in tensor_infos for x in sublist)
                return merged_result

        for result in results:
            merged_result.extend(result)

        return merged_result

    def merge_raw_tensor_data(self, raw_magic, tensor_infos):
        # 创建合并张量的基础信息
        merge_tensor_info = {}
        merge_tensor_info["ROOT_CALL:rawmagic"] = raw_magic
        merge_tensor_info["datatype"] = tensor_infos[0]["datatype"]
        merge_tensor_info["IO_FLAG"] = tensor_infos[0]["IO_FLAG"]
        merge_tensor_info["B>:rawshape"] = tensor_infos[0]["B>:rawshape"]
        merge_tensor_info["B>:datatype"] = tensor_infos[0]["B>:datatype"]
        merge_tensor_info["ROOT_FUNC:hash"] = 0
        merge_tensor_info["FUNC:hash"] = 0
        merge_tensor_info["B>execStart"] = 0
        merge_tensor_info["B>execEnd"] = 0

        # 检查所有 tensor 的 loop 信息是否相同
        loop_var_infos_list = [t.get("B>loopVarInfos", {}) for t in tensor_infos]
        if loop_var_infos_list and all(lvi == loop_var_infos_list[0] for lvi in loop_var_infos_list):
            merge_tensor_info["B>loopVarInfos"] = loop_var_infos_list[0]

        # 生成保存路径
        loop_suffix = ""
        loop_var_infos = merge_tensor_info.get("B>loopVarInfos", {})
        if isinstance(loop_var_infos, dict) and len(loop_var_infos) > 0:
            loop_values = [str(v) for v in loop_var_infos.values()]
            loop_suffix = "_" + "_".join(loop_values)
        elif isinstance(loop_var_infos, str) and loop_var_infos:
            # 如果已经是字符串，解析成 dict 获取 values
            parsed_dict = _verify_res.parse_loop_info(loop_var_infos)
            if parsed_dict:
                loop_values = [str(v) for v in parsed_dict.values()]
                loop_suffix = "_" + "_".join(loop_values)
        file_path = os.path.join(
            self.dump_tensor_path,
            f"raw_{raw_magic}_{tensor_infos[0]['B>:datatype']}_{tensor_infos[0]['IO_FLAG']}{loop_suffix}.data",
        )
        merge_tensor_info["B>FILENAME"] = file_path

        # 按offset排序张量
        tensor_infos_sorted = sorted(tensor_infos, key=lambda x: x["B>OP_ATTR_SYM_OFFSET"])
        grouped_tensors = {}
        for key, group in groupby(tensor_infos_sorted, key=lambda x: x["B>OP_ATTR_SYM_OFFSET"]):
            grouped_tensors[key] = list(group)
        if len(grouped_tensors) == 1:
            return merge_tensor_info, None

        # 执行合并操作
        dtype = _get_data_type(merge_tensor_info["datatype"])[1]
        raw_data = np.zeros(merge_tensor_info["B>:rawshape"], dtype)

        for tensor_info in tensor_infos:
            if tensor_info["B>:validshape"] == tensor_info["B>:rawshape"]:
                logging.info(f"Tensor {tensor_info['B>FILENAME']} shape is equal to rawShape, skip merge.")
                return merge_tensor_info, None
            is_tensor_valid = True
            data = np.fromfile(tensor_info["B>FILENAME"], dtype)
            data = data.reshape(tensor_info.get("B>:validshape"))

            # 计算切片范围
            raw_slices, data_slices = [], []
            for dim in range(data.ndim):
                start = tensor_info["B>OP_ATTR_SYM_OFFSET"][dim]
                stop = min(merge_tensor_info["B>:rawshape"][dim], start + data.shape[dim])
                if start >= stop:
                    is_tensor_valid = False

                raw_slices.append(slice(start, stop))
                data_slices.append(slice(0, min(merge_tensor_info["B>:rawshape"][dim] - start, data.shape[dim])))

            # 合并有效张量
            if is_tensor_valid:
                raw_data[tuple(raw_slices)] = data[tuple(data_slices)]

        # 保存合并后的张量
        raw_data.tofile(file_path)
        return merge_tensor_info, raw_data

    def merge_raw_tensor(self):
        merge_tensor_infos = []
        for key, tensor_infos in self.raw_tensor_info.items():
            # 合并张量数据
            merge_tensor_info, raw_data = self.merge_raw_tensor_data(key[0], tensor_infos)

            # 如果有合并后的数据，进行验证
            if raw_data is not None:
                merge_tensor_info = self._verify_merged_tensor(merge_tensor_info, raw_data)
                merge_tensor_infos.append(merge_tensor_info)
        return merge_tensor_infos


def scan_pass_info_from_path(verify_path: str) -> Dict[str, int]:
    """Scan computation_graph directory, return {pass_name: pass_num} dict."""
    pass_info: Dict[str, int] = {}
    if not verify_path or not os.path.exists(verify_path):
        return pass_info

    parent_dir = os.path.dirname(verify_path.rstrip('/'))
    if not parent_dir:
        return pass_info

    computation_graph_dir = os.path.join(parent_dir, 'computation_graph')
    if not os.path.exists(computation_graph_dir):
        return pass_info

    for d in os.listdir(computation_graph_dir):
        dir_path = os.path.join(computation_graph_dir, d)
        if not os.path.isdir(dir_path) or not d.startswith('Strategy_'):
            continue
        for item in os.listdir(dir_path):
            if not os.path.isdir(os.path.join(dir_path, item)):
                continue
            if item.startswith('Pass_') and len(item.split('_')) >= 3:
                parts = item.split('_', 2)
                if len(parts) == 3:
                    try:
                        pass_info[parts[2]] = int(parts[1])
                    except ValueError:
                        continue
    return pass_info


def get_pass_full_name(verify_path: str, pass_name: str = "CodegenPreproc") -> str:
    pass_info = scan_pass_info_from_path(verify_path)
    if pass_name in pass_info:
        return f"Pass_{pass_info[pass_name]:02d}_{pass_name}"
    return pass_name


def to_json_str(x):
    """Convert list/tuple to compact JSON string, return others as-is."""
    return json.dumps(x, separators=(',', ':')) if isinstance(x, (list, tuple)) else x


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser dump_tensor.")
    parser.add_argument(
        "--dump_tensor_path",
        type=str,
        default=[],
        required=True,
        help="directory like output/output_2026xxxxx/dump_tensor_device_x",
    )
    parser.add_argument("--verify_path", type=str, default="", help="Path to verify_result.csv")
    return parser.parse_args()


def main():
    args = parse_arguments()
    timestamp = int(time.time() * 1_000_000)
    pass_full_name = get_pass_full_name(args.verify_path, "CodegenPreproc")
    csv_path = os.path.join(args.verify_path, f"verify_task_result_cmp~{pass_full_name}~{timestamp}.csv")
    global RESULT_FILE
    RESULT_FILE = csv_path
    if not os.path.exists(args.dump_tensor_path):
        logging.error(f"Directory does not exist: {args.dump_tensor_path}")
        return
    # 初始化紧凑解析器
    parser = CompactDumpTensorInfoParser(args.dump_tensor_path)
    logging.info(f"Single struct byte count: {parser.struct_compact_size}")

    _verify_res.read_verify_result(args.verify_path)

    for dir_path, _, file_names in os.walk(args.dump_tensor_path):
        for file_name in file_names:
            if not file_name.endswith(".tdump"):
                continue
            bin_file = os.path.join(dir_path, file_name)
            parser.parse_file(bin_file)

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.dump_tensor_path)))
    prof_data_file = os.path.join(parent_dir, "tilefwk_L1_prof_data.json")
    if os.path.exists(prof_data_file):
        parser.get_exec_time(prof_data_file)

    tensor_infos = parser.tensor_compare()
    tensor_infos.sort(key=lambda x: x.get("B>TIMESTAMP"))  # 输出前做一次排序
    merge_tensor_infos = parser.merge_raw_tensor()
    tensor_infos.extend(merge_tensor_infos)
    df = pd.DataFrame(tensor_infos, dtype=object)

    # 处理 datatype 列：删除数值列，保留字符串列并重命名
    if "datatype" in df.columns:
        df.drop("datatype", axis=1, inplace=True)

    df.sort_values(
        by=["B>EXEC_TIMESTAMP", "B>TIMESTAMP", "A>:rawmagic", "B>OP_ATTR_SYM_OFFSET"], ignore_index=True, inplace=True
    )
    df.insert(0, "NO.", range(1, len(df) + 1))

    for col in ["B>:validshape", "B>OP_ATTR_SYM_OFFSET", "B>:rawshape", "A>:validshape"]:
        if col in df.columns:
            df[col] = df[col].apply(to_json_str)

    # 转成字符串，防止用excel打开后显示为科学计算法，导致数据截断
    df["ROOT_FUNC:hash"] = df["ROOT_FUNC:hash"].apply(lambda x: f"{x}'")
    df["FUNC:hash"] = df["FUNC:hash"].apply(lambda x: f"{x}'")
    df["B>execStart"] = df["B>execStart"].apply(lambda x: f"{x}'")
    df["B>execEnd"] = df["B>execEnd"].apply(lambda x: f"{x}'")
    df["B>EXEC_TIMESTAMP"] = df["B>EXEC_TIMESTAMP"].apply(lambda x: f"{x}'")
    df["B>TIMESTAMP"] = df["B>TIMESTAMP"].apply(lambda x: f"{x}'")
    df["B>tensorAddr"] = df["B>tensorAddr"].apply(lambda x: f"{x}'")

    # 只保留文件名，去掉绝对路径
    if "A>FILENAME" in df.columns:
        df["A>FILENAME"] = df["A>FILENAME"].apply(lambda x: os.path.basename(x) if isinstance(x, str) and x else "")
    if "B>FILENAME" in df.columns:
        df["B>FILENAME"] = df["B>FILENAME"].apply(lambda x: os.path.basename(x) if isinstance(x, str) and x else "")

    logging.info(df)

    df.to_csv(csv_path, index=False, encoding="utf-8")
    logging.info(f"Verify result saved to: {csv_path}")


if __name__ == "__main__":
    main()
