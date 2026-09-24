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
"""pass data compare"""

import argparse
from dataclasses import dataclass
import json
import logging
from multiprocessing import Pool, cpu_count
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import ml_dtypes
import numpy as np
import pandas as pd
from parse_dump_tensors import scan_pass_info_from_path
from tensor_diff import IsCloseConfig, compare_tensors_result_dict
import torch


@dataclass
class ProcessLoopBatchArgs:
    loop_tasks: List[Dict[str, Any]]
    pass_a: str
    pass_b: str
    verify_path_pass1: str
    verify_path_pass2: str
    atol: float
    rtol: float
    topk: int
    key: str
    is_codegen: bool
    is_leaf: bool
    csv_data_dict: Dict[str, Any]
    result_file: str


class PassComparator:
    """Pass comparator class, which encapsulates all comparison logic."""

    def __init__(
        self,
        output_pass: str = "",
        golden_pass: str = "",
        verify_path_pass1: str = "",
        verify_path_pass2: str = "",
        atol: float = 1e-3,
        rtol: float = 1e-3,
        topk: int = 1000,
    ):
        """
        Initializing the comparator
        Parameters:
            verify_path_pass1: Verification file path for the first pass
            verify_path_pass2: Verification file path for the second pass
            atol: Absolute tolerance
            rtol: Relative tolerance
            topk: Print the first k differences
        """
        self.verify_path_pass1 = verify_path_pass1
        self.verify_path_pass2 = verify_path_pass2
        self.atol = atol
        self.rtol = rtol
        self.topk = topk
        self.key = ":rawmagic"
        self.output_pass = output_pass
        self.golden_pass = golden_pass
        self.result_file = ""
        self.row_num = 1
        self.comparison_records: List[Dict[str, Any]] = []
        self.dtype_dict = {
            "BF16": ml_dtypes.bfloat16,
            "FP32": np.float32,
            "FP16": np.float16,
            "INT32": np.int32,
            "INT8": np.int8,
            "INT64": np.int64,
            "INT16": np.int16,
        }

        self.pass_dict = {
            "tensor_graph": 0,
            "InferTensorFormat": 0,
            "RemoveRedundantReshape": 1,
            "AutoCast": 2,
            "InferMemoryConflict": 3,
            "RemoveUndrivenView": 4,
            "ExpandFunction": 5,
            "MergeViewAssemble": 6,
            "SplitReshape": 7,
            "SplitRawTensor": 8,
            "SplitLargeFanoutTensor": 9,
            "DuplicateOp": 10,
            "AssignMemoryType": 11,
            "InferDiscontinuousInput": 12,
            "RemoveRedundantOp": 13,
            "InsertOpForViewAssemble": 14,
            "ProcessAtomic": 15,
            "GraphPartition": 16,
            "NBufferMerge": 17,
            "L1CopyInReuseMerge": 18,
            "IntraSubgraphAdapter": 19,
            "GenerateMoveOp": 20,
            "CommonOperationEliminate": 21,
            "AxisCombine": 22,
            "PadLocalBuffer": 23,
            "RemoveUnalignedReshape": 24,
            "ReplaceTensor": 25,
            "PreGraphProcess": 26,
            "InferDynShape": 27,
            "SubgraphToFunction": 28,
            "InferParamIndex": 29,
            "SrcDstBufferMerge": 30,
            "AddAlloc": 31,
            "OoOSchedule": 32,
            "RemoveAlloc": 33,
            "CopyOutResolve": 34,
            "InsertSync": 35,
            "GlobalMemoryReuse": 36,
            "CodegenPreproc": 37,
        }

        self._update_pass_dict_from_path(verify_path_pass1)

    @staticmethod
    def is_contain(a: Dict[str, Any], b: Dict[str, Any], key: str, is_leaf: bool, opcode_dict: Dict) -> bool:
        """
        Checks whether tensor a is completely included in tensor b.
        Returns:
            bool: True indicates that a is included in b.
        """
        a_offset = json.loads(a[":offset"])
        b_offset = json.loads(b[":offset"])
        a_shape = json.loads(a[":validshape"])
        b_shape = json.loads(b[":validshape"])
        copy_opcodes = {
            "ASSEMBLE",
            "COPY_OUT",
            "COPY_IN",
            "RESHAPE_COPY_OUT",
            "RESHAPE_COPY_IN",
            "L0C_RESHAPE_COPY_OUT",
        }
        if a[":opcode"] in copy_opcodes:
            if is_leaf and a["ROOT_CALL:opmagic"] is not None:
                return (
                    a["OP_ATTR_SYM_OFFSET"] == b["OP_ATTR_SYM_OFFSET"]
                    and a[":opcode"] == b[":opcode"]
                    and a["ROOT_CALL:opmagic"] == b["ROOT_CALL:opmagic"]
                )
            elif a["OP_ATTR_ATOMIC"]:
                return (
                    a["OP_ATTR_SYM_OFFSET"] == b["OP_ATTR_SYM_OFFSET"]
                    and PassComparator.opcode_match(a[":opcode"], b[":opcode"], opcode_dict)
                    and a["OP_ATTR_ATOMIC"] == b["OP_ATTR_ATOMIC"]
                    and a[":opmagic"] == b[":opmagic"]
                )
            else:
                return a["OP_ATTR_SYM_OFFSET"] == b["OP_ATTR_SYM_OFFSET"] and PassComparator.opcode_match(
                    a[":opcode"], b[":opcode"], opcode_dict
                )
        if key == ":magic" and a_shape == b_shape and not is_leaf:
            return True
        else:
            for a_off, b_off, a_sh, b_sh in zip(a_offset, b_offset, a_shape, b_shape):
                if (a_off < b_off) or ((a_off + a_sh) > (b_off + b_sh)):
                    return False
        if key == "ROOT_CALL:rawmagic":
            return a[":opcode"] == b[":opcode"]
        if is_leaf and a["ROOT_CALL:opmagic"] is not None:
            return a["ROOT_CALL:opmagic"] == b["ROOT_CALL:opmagic"]
        return True

    @staticmethod
    def opcode_match(opcode_a: str, opcode_b: str, opcode_dict: Dict) -> bool:
        """opcode match"""
        if opcode_a == opcode_b:
            return True
        if opcode_a in opcode_dict:
            return opcode_b in opcode_dict[opcode_a]
        if opcode_b in opcode_dict:
            return opcode_a in opcode_dict[opcode_b]
        return False

    @staticmethod
    def add_comparison_record(
        a: Dict[str, Any], b: Optional[Dict[str, Any]] = None, compare_result: Optional[Dict[str, Any]] = None
    ):
        """Add the comparison record to the internal list"""

        record = {}
        record["NO."] = 1
        record["PATH_FUNC:func_magicname "] = a["PATH_FUNC:func_magicname"]
        record["PATH_FUNC:funcmagic"] = a["PATH_FUNC:funcmagic"]
        record["PATH_FUNC:hash"] = a["PATH_FUNC:hash"]
        record["LOOP_INFO"] = a["LOOP_INFO"]
        record[":symbol"] = a[":symbol"]
        record[":validshape"] = a[":validshape"]
        record[":datatype"] = a[":datatype"]
        record["OP_ATTR_SYM_OFFSET"] = a["OP_ATTR_SYM_OFFSET"]
        record["OP_IO_FLAG"] = a["OP_IO_FLAG"]
        record["B>PHASE_NAME"] = a["PHASE_NAME"]
        record["B>TIMESTAMP"] = a["TIMESTAMP"]
        record["B>FILENAME"] = a["FILENAME"]
        record["B>FUNC:hash"] = a["FUNC:hash"]
        record["B>FUNC:funcmagic"] = a["FUNC:funcmagic"]
        record["B>ROOT_CALL:opmagic"] = a["ROOT_CALL:opmagic"]
        record["B>ROOT_CALL:rawmagic"] = a["ROOT_CALL:rawmagic"]
        record["B>:opmagic"] = a[":opmagic"]
        record["B>:opcode"] = a[":opcode"]
        record["B>:rawmagic"] = a[":rawmagic"]
        record["B>:magic"] = a[":magic"]
        record["B>:rawshape"] = a[":rawshape"]
        record["B>:format"] = a[":format"]
        record["B>:shape"] = a[":shape"]
        record["B>EVAL:dynvalidshape"] = a["EVAL:dynvalidshape"]
        record["B>ROOT_FUNC:hash"] = a["ROOT_FUNC:hash"]

        if b:
            record["A>PHASE_NAME"] = b["PHASE_NAME"]
            record["A>TIMESTAMP"] = b["TIMESTAMP"]
            record["A>FILENAME"] = b["FILENAME"]
            record["A>FUNC:hash"] = b["FUNC:hash"]
            record["A>FUNC:funcmagic"] = b["FUNC:funcmagic"]
            record["A>ROOT_CALL:opmagic"] = b["ROOT_CALL:opmagic"]
            record["A>ROOT_CALL:rawmagic"] = b["ROOT_CALL:rawmagic"]
            record["A>:opmagic"] = b[":opmagic"]
            record["A>:opcode"] = b[":opcode"]
            record["A>:rawmagic"] = b[":rawmagic"]
            record["A>:magic"] = b[":magic"]
            record["A>:rawshape"] = b[":rawshape"]
            record["A>:format"] = b[":format"]
            record["A>:shape"] = b[":shape"]
            record["A>EVAL:dynvalidshape"] = b["EVAL:dynvalidshape"]
            record["A>ROOT_FUNC:hash"] = b["ROOT_FUNC:hash"]

        if compare_result is None:
            record["AB>RESULT"] = "SKIP"
            record["AB>RESULT_REASON"] = a.get("skip_reason", "")
            record["AB>rtol/atol"] = ""
        else:
            record.update(compare_result)

        return record

    @staticmethod
    def should_skip_record(ai: Dict[str, Any], bi: Dict[str, Any]) -> bool:
        """Check if a record pair should be skipped based on validshape and INPUT:validshape."""
        ai_validshape = json.loads(ai[':validshape'])
        bi_validshape = json.loads(bi[':validshape'])
        if 0 in ai_validshape or 0 in bi_validshape:
            return True
        ai_opcode = ai[":opcode"]
        bi_opcode = bi[":opcode"]
        if ai_opcode in {"ASSEMBLE", "COPY_OUT"} and 0 in json.loads(ai['INPUT:validshape']):
            return True
        if bi_opcode in {"ASSEMBLE", "COPY_OUT"} and 0 in json.loads(bi['INPUT:validshape']):
            return True
        return False

    @staticmethod
    def _process_loop_batch(args: ProcessLoopBatchArgs) -> List[Dict[str, Any]]:
        """
        Static method to process a batch of loops in multiprocessing context.
        Returns:
            List of comparison records
        """
        comparison_records = []
        try:
            loop_tasks = args.loop_tasks
            pass_a = args.pass_a
            pass_b = args.pass_b
            verify_path_pass1 = args.verify_path_pass1
            verify_path_pass2 = args.verify_path_pass2
            atol = args.atol
            rtol = args.rtol
            topk = args.topk
            key = args.key
            is_codegen = args.is_codegen
            is_leaf = args.is_leaf
            csv_data_dict = args.csv_data_dict
            result_file = args.result_file

            dtype_dict = {
                "BF16": ml_dtypes.bfloat16,
                "FP32": np.float32,
                "FP16": np.float16,
                "INT32": np.int32,
                "INT8": np.int8,
                "INT64": np.int64,
                "INT16": np.int16,
            }

            opcode_dict = {
                "VIEW": ["L1_TO_L0A", "L1_TO_L0B"],
                "A_MUL_B": ["A_MULACC_B"],
                "COPY_OUT": ["ASSEMBLE"],
                "COPY_IN": ["VIEW"],
            }

            for task in loop_tasks:
                df_loop = task['df_loop']
                df_a = df_loop[df_loop["PHASE_NAME"].str.contains(pass_a)]
                df_b = df_loop[df_loop["PHASE_NAME"].str.contains(pass_b)]

                if is_codegen:
                    a_copy = df_a[df_a[":opcode"].isin(['COPY_IN', 'COPY_OUT'])]
                    a_dict = a_copy[a_copy["ROOT_CALL:rawmagic"].notna()].to_dict(orient='records')
                else:
                    a_dict = df_a.to_dict(orient='records')

                for ai in a_dict:
                    raw_magic = ai[key]
                    if is_codegen:
                        b_records = df_b[df_b[":rawmagic"] == raw_magic].to_dict(orient='records')
                        if ai[":opcode"] == "COPY_IN":
                            b_records = df_b[
                                (df_b["INPUT:rawmagic"] == str(int(ai[key]))) & (df_b[":opcode"] == "COPY_IN")
                            ].to_dict(orient='records')
                    else:
                        b_records = df_b[df_b[key] == raw_magic].to_dict(orient='records')

                    if len(b_records) == 0:
                        ai["skip_reason"] = f"{key} : {raw_magic}, not exit in golden pass"
                        record = PassComparator.add_comparison_record(a=ai, compare_result=None)
                        comparison_records.append(record)
                        continue

                    is_match = False
                    for bi in b_records:
                        if PassComparator.should_skip_record(ai, bi):
                            continue
                        if not PassComparator._compare_not_support_static(
                            ai, bi, key, verify_path_pass1, verify_path_pass2, dtype_dict, opcode_dict, is_leaf
                        ):
                            continue

                        compare_result = PassComparator._compare_data_static(
                            ai,
                            bi,
                            verify_path_pass1,
                            verify_path_pass2,
                            atol,
                            rtol,
                            topk,
                            csv_data_dict,
                            dtype_dict,
                            key,
                            result_file,
                        )
                        record = PassComparator.add_comparison_record(a=ai, b=bi, compare_result=compare_result)
                        comparison_records.append(record)
                        is_match = True
                        break

                    if not is_match:
                        ai["skip_reason"] = "not match"
                        record = PassComparator.add_comparison_record(a=ai, compare_result=None)
                        comparison_records.append(record)
        except Exception as e:
            stack_trace = traceback.format_exc()
            logging.error(f"Exception in multiprocessing: error={str(e)}\nStack trace:\n{stack_trace}")
            return comparison_records
        return comparison_records

    @staticmethod
    def _compare_not_support_static(
        a: Dict[str, Any],
        b: Dict[str, Any],
        key: str,
        verify_path_pass1: str,
        verify_path_pass2: str,
        dtype_dict: Dict,
        opcode_dict: Dict,
        is_leaf: bool,
    ) -> bool:
        """Static version of compare_not_support for multiprocessing"""
        if not PassComparator.is_contain(
            a,
            b,
            key,
            is_leaf,
            opcode_dict,
        ):
            return False

        f_a = os.path.join(verify_path_pass1, a["PHASE_NAME"], a["FILENAME"])
        f_b = os.path.join(verify_path_pass2, b["PHASE_NAME"], b["FILENAME"])
        if not os.path.exists(f_a) or not os.path.exists(f_b):
            return False

        opcode_a = a[":opcode"]
        opcode_b = b[":opcode"]
        if (
            key == ":rawmagic"
            and opcode_a != opcode_b
            and not PassComparator.opcode_match(opcode_a, opcode_b, opcode_dict)
        ):
            return False

        if a[":datatype"] != b[":datatype"]:
            return False

        dtype = a[":datatype"]
        if dtype not in dtype_dict:
            return False

        return True

    @staticmethod
    def _compare_data_static(
        a: Dict[str, Any],
        b: Dict[str, Any],
        verify_path_pass1: str,
        verify_path_pass2: str,
        atol: float,
        rtol: float,
        topk: int,
        csv_data_dict: Dict,
        dtype_dict: Dict,
        key: str,
        result_file: str,
    ) -> Dict[str, Any]:
        """Static version of compare_data for multiprocessing"""
        data_a, data_b = PassComparator._get_data_slice_static(
            a, b, verify_path_pass1, verify_path_pass2, dtype_dict, csv_data_dict, key
        )

        a_shape = json.loads(a[":validshape"])
        if a[":opcode"] in {"ASSEMBLE", "COPY_OUT"}:
            input_dict = [item for item in csv_data_dict if item["FILENAME"] == a["INPUT:FILENAMES"]]
            if input_dict:
                a_shape = json.loads(input_dict[0][":validshape"])

        tensor_a = torch.from_numpy(data_a.astype(np.float64)).to(torch.float64)
        tensor_b = torch.from_numpy(data_b.astype(np.float64)).to(torch.float64)

        config = IsCloseConfig(
            rtol=rtol, atol=atol, is_detail=True, shape=a_shape, calc_dtype=torch.float64, top_k=topk
        )
        csv_path = os.path.join(verify_path_pass1, result_file[:-4] + ".DETAIL", a["FILENAME"][:-5] + ".csv")
        result_dict = compare_tensors_result_dict(tensor_a, tensor_b, csv_path, config=config)
        return result_dict

    @staticmethod
    def _get_data_slice_static(
        a: Dict[str, Any],
        b: Dict[str, Any],
        verify_path_pass1: str,
        verify_path_pass2: str,
        dtype_dict: Dict,
        csv_data_dict: Dict,
        key: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Static version of get_data_slice for multiprocessing"""
        f_a = os.path.join(verify_path_pass1, a["PHASE_NAME"], a["FILENAME"])
        f_b = os.path.join(verify_path_pass2, b["PHASE_NAME"], b["FILENAME"])

        if a[":opcode"] in {"ASSEMBLE", "COPY_OUT"}:
            return PassComparator._get_data_slice_assemble_static(
                a, b, verify_path_pass1, verify_path_pass2, dtype_dict, csv_data_dict
            )

        a_offset = json.loads(a[":offset"])
        b_offset = json.loads(b[":offset"])
        a_shape = json.loads(a[":validshape"])
        b_shape = json.loads(b[":validshape"])
        np_dtype = dtype_dict.get(a[":datatype"])
        data_a = np.fromfile(f_a, np_dtype)
        data_b = np.fromfile(f_b, np_dtype)

        data_a = data_a.reshape(a_shape)
        data_b = data_b.reshape(b_shape)

        if a_shape == b_shape and key == ":magic":
            return data_a, data_b

        slices = []
        for dim in range(data_a.ndim):
            start = a_offset[dim] - b_offset[dim]
            stop = start + a_shape[dim]
            slices.append(slice(start, stop))
        b_slice = data_b[tuple(slices)]
        return data_a, b_slice

    @staticmethod
    def _get_data_slice_assemble_static(
        a: Dict[str, Any],
        b: Dict[str, Any],
        verify_path_pass1: str,
        verify_path_pass2: str,
        dtype_dict: Dict,
        csv_data_dict: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Static version of get_data_slice_assemble for multiprocessing"""
        f_a = os.path.join(verify_path_pass1, a["PHASE_NAME"], a["FILENAME"])
        f_b = os.path.join(verify_path_pass2, b["PHASE_NAME"], b["FILENAME"])

        input_dict = [item for item in csv_data_dict if item["FILENAME"] == a["INPUT:FILENAMES"]]
        a_offset = json.loads(a["OP_ATTR_SYM_OFFSET"])
        b_offset = json.loads(b["OP_ATTR_SYM_OFFSET"])
        shape = json.loads(a[":rawshape"])
        if not input_dict:
            logging.error(f"No matching record found for FILENAME: {a['INPUT:FILENAMES']}")
            raise ValueError(f"No matching record found for FILENAME: {a['INPUT:FILENAMES']}")
        a_shape = json.loads(input_dict[0][":validshape"])
        b_shape = json.loads(input_dict[0][":validshape"])
        np_dtype = dtype_dict.get(a[":datatype"])
        data_a = np.fromfile(f_a, np_dtype)
        data_b = np.fromfile(f_b, np_dtype)

        data_a = data_a.reshape(shape)
        data_b = data_b.reshape(shape)

        slices_a = []
        for dim in range(data_a.ndim):
            start = a_offset[dim]
            stop = start + a_shape[dim]
            slices_a.append(slice(start, stop))
        a_slice = data_a[tuple(slices_a)]

        slices_b = []
        for dim in range(data_b.ndim):
            start = b_offset[dim]
            stop = start + b_shape[dim]
            slices_b.append(slice(start, stop))
        b_slice = data_b[tuple(slices_b)]
        return a_slice, b_slice

    def pass_compare(self, pass_a: str, pass_b: str, paths: List[str] = None) -> None:
        """
        Main comparison function
        """
        csv_path = os.path.join(self.verify_path_pass1, "verify_graph_data_metainfo.csv")
        df = pd.read_csv(csv_path, encoding="utf-8", na_values=["", " ", "NaN", "NA"])
        csv_data_dict = df.to_dict(orient='records')

        if self.pass_dict[pass_a] < self.pass_dict[pass_b]:
            pass_a, pass_b = pass_b, pass_a
            self.output_pass, self.golden_pass = self.golden_pass, self.output_pass
            self.verify_path_pass1, self.verify_path_pass2 = self.verify_path_pass2, self.verify_path_pass1
        self.result_file = (
            f'verify_graph_result_cmp~Pass_{self.pass_dict[self.golden_pass]:02d}_{self.golden_pass}~'
            f'Pass_{self.pass_dict[self.output_pass]:02d}_{self.output_pass}~{int(time.time() * 1_000_000)}.csv'
        )
        infer_param_index = self.pass_dict["InferParamIndex"]
        expand_function = self.pass_dict["ExpandFunction"]
        if self.pass_dict[pass_a] >= expand_function and self.pass_dict[pass_b] >= expand_function:
            self.key = ":magic"
        # 判断是否需要使用 codegen 的特殊逻辑
        is_codegen = False
        if (
            self.pass_dict[pass_a] >= infer_param_index
            and self.pass_dict[pass_b] >= expand_function
            and self.pass_dict[pass_b] < infer_param_index
        ):
            self.key = "ROOT_CALL:rawmagic"
            is_codegen = True
        is_leaf = False
        if self.pass_dict[pass_a] >= infer_param_index and self.pass_dict[pass_b] >= infer_param_index:
            is_leaf = True
        logging.info(f"key  : {self.key}")

        df_pass = df[df["PHASE_NAME"].str.contains(f'{pass_a}|{pass_b}', na=False, regex=True)]
        if paths == []:
            paths = df_pass["PATH_FUNC:func_magicname"].dropna().unique()

        # 收集所有loop任务
        loop_tasks = []
        for path in paths:
            df_path = df_pass[df_pass["PATH_FUNC:func_magicname"] == path]
            loop_info_list = df_path["LOOP_INFO"].dropna().unique()

            for loop_info in loop_info_list:
                df_loop = df_path[df_path["LOOP_INFO"] == loop_info]
                loop_tasks.append({'df_loop': df_loop, 'path': path, 'loop_info': loop_info})

        # 使用多进程批量处理loop任务
        num_workers = min(min(cpu_count(), len(loop_tasks)), 32)
        logging.info(f"num_workers : {num_workers}")
        avg_tasks = len(loop_tasks) // num_workers
        remainder = len(loop_tasks) % num_workers

        batches = []
        start = 0
        for i in range(num_workers):
            end = start + avg_tasks + (1 if i < remainder else 0)
            batches.append(loop_tasks[start:end])
            start = end

        args_list = [
            ProcessLoopBatchArgs(
                loop_tasks=batch,
                pass_a=pass_a,
                pass_b=pass_b,
                verify_path_pass1=self.verify_path_pass1,
                verify_path_pass2=self.verify_path_pass2,
                atol=self.atol,
                rtol=self.rtol,
                topk=self.topk,
                key=self.key,
                is_codegen=is_codegen,
                is_leaf=is_leaf,
                csv_data_dict=csv_data_dict,
                result_file=self.result_file,
            )
            for batch in batches
        ]
        try:
            with Pool(processes=num_workers) as pool:
                all_results = pool.map(self._process_loop_batch, args_list)

            # 合并结果
            all_records = []
            for batch_results in all_results:
                all_records.extend(batch_results)
            all_records.sort(key=lambda x: x.get("B>TIMESTAMP"))
            for record in all_records:
                record["NO."] = self.row_num
                self.comparison_records.append(record)
                self.row_num += 1
        except Exception as e:
            stack_trace = traceback.format_exc()
            logging.error(f"Exception in multiprocessing: error={str(e)}\nStack trace:\n{stack_trace}")
            self.save_comparison_results()
        self.save_comparison_results()

    def save_comparison_results(self, csv_path: str = None):
        """
        Save all comparison results to a CSV file
        """
        if not self.comparison_records:
            logging.warning("No comparison records to save.")
            return

        if csv_path is None:
            csv_path = os.path.join(self.verify_path_pass1, self.result_file)

        df = pd.DataFrame(self.comparison_records)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logging.info(f"Comparison results saved to {csv_path}")

    def _update_pass_dict_from_path(self, verify_path: str) -> None:
        self.pass_dict.update(scan_pass_info_from_path(verify_path))


def main():
    """Main function: Parse parameters and run the comparison"""
    parser = argparse.ArgumentParser(
        description="Pass Compare",
        epilog="example:  python3 pass_compare.py --p ExpandFunction RemoveUndrivenView --verify_path ...",
    )

    parser.add_argument(
        "--p",
        nargs='*',
        type=str,
        default=[],
        required=True,
        help="Names of the two passes to be compared, separated by a space.\
                       The second is goldenpass.",
    )
    parser.add_argument(
        "--func",
        nargs='*',
        type=str,
        default=[],
        help="Name of the function to be compared. Functions are separated by spaces.",
    )
    parser.add_argument(
        "--verify_path",
        nargs='*',
        type=str,
        default=[],
        help="Verify the file directory. If two values are provided, they represent \
                       the paths of two passes respectively.",
    )
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--topk", type=int, default=1000, help="Print the number of differing lines")

    args = parser.parse_args()

    if len(args.p) != 2:
        logging.error("The number of input passes is not 2!")
        sys.exit(1)

    if len(args.verify_path) == 2:
        verify_path_pass1 = args.verify_path[0]
        verify_path_pass2 = args.verify_path[1]
    elif len(args.verify_path) == 1:
        verify_path_pass1 = args.verify_path[0]
        verify_path_pass2 = args.verify_path[0]
    else:
        logging.error("The verify_path parameter is incorrect !")
        sys.exit(1)

    comparator = PassComparator(
        output_pass=args.p[0],
        golden_pass=args.p[1],
        verify_path_pass1=verify_path_pass1,
        verify_path_pass2=verify_path_pass2,
        atol=args.atol,
        rtol=args.rtol,
        topk=args.topk,
    )

    logging.info(f"pass : {args.p[0]}, {args.p[1]}")
    logging.info(f"path: {args.func}")
    logging.info(f"ExpandFunction: {comparator.pass_dict['ExpandFunction']}")
    logging.info(f"InferParamIndex: {comparator.pass_dict['InferParamIndex']}")
    logging.info(f"verify_path_pass1: {verify_path_pass1}")
    logging.info(f"verify_path_pass2: {verify_path_pass2}")

    comparator.pass_compare(pass_a=args.p[0], pass_b=args.p[1], paths=args.func)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # level：DEBUG < INFO < WARNING < ERROR < CRITICAL
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("app.log", mode='w', encoding="utf-8")],
    )

    main()
