#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import argparse
import os
import struct

import pandas as pd
from tabulate import tabulate

# MsprofAdditonalInfo: 256 byte
ADDITIONAL_INFO_FMT = "HHIIIQ"
PMU_HEAD_FMT = "HHIHBB" + "I" * 7
PMU_DATA_FMT_8 = "IIQ" + "I" * 8
PMU_DATA_FMT_10 = "IIQ" + "I" * 10
ADDITIONAL_INFO_SIZE = struct.calcsize(ADDITIONAL_INFO_FMT)
PMU_HEAD_SIZE = struct.calcsize(PMU_HEAD_FMT)
PMU_DATA_SIZE_8 = struct.calcsize(PMU_DATA_FMT_8)
PMU_DATA_SIZE_10 = struct.calcsize(PMU_DATA_FMT_10)
PMU_RECORD_SIZE = 256
MSPROF_REPORT_AICPU_HCCL_OP_INFO = 10
PROF_DATATYPE_PMU = 3


class TileFwkPmuStructBean:
    def __init__(self: any, raw_bytes: bytes) -> None:
        self._task_list = []
        if len(raw_bytes) < ADDITIONAL_INFO_SIZE + PMU_HEAD_SIZE:
            return

        header = struct.unpack_from(ADDITIONAL_INFO_FMT, raw_bytes, 0)
        self._magic_number = header[0]
        self._level = header[1]
        self._type = header[2]
        self._thread_id = header[3]
        self._data_len = header[4]
        self._time_stamp = header[5]

        head = struct.unpack_from(PMU_HEAD_FMT, raw_bytes, ADDITIONAL_INFO_SIZE)
        self._head = head[0]
        self._core_id = head[1] & 127  # 取后低7位core_id
        self._data_type = (head[1] >> 10) & 63  # 取高6位data_type
        self._task_id = head[2]
        self._stream_id = head[3]
        self._task_count = head[4]
        self._res0 = head[5]

        if self._type != MSPROF_REPORT_AICPU_HCCL_OP_INFO or self._data_type != PROF_DATATYPE_PMU:
            return

        payload_len = min(self._data_len, len(raw_bytes) - ADDITIONAL_INFO_SIZE)
        if payload_len < PMU_HEAD_SIZE:
            return

        data_bytes = payload_len - PMU_HEAD_SIZE
        if self._task_count <= 0:
            return

        task_count = min(4, self._task_count)
        entry_size = 0
        if data_bytes % task_count == 0:
            entry_size = data_bytes // task_count
        if entry_size not in (PMU_DATA_SIZE_8, PMU_DATA_SIZE_10):
            if data_bytes >= task_count * PMU_DATA_SIZE_10:
                entry_size = PMU_DATA_SIZE_10
            else:
                entry_size = PMU_DATA_SIZE_8

        data_fmt = PMU_DATA_FMT_10 if entry_size == PMU_DATA_SIZE_10 else PMU_DATA_FMT_8
        for i in range(0, task_count):
            offset = ADDITIONAL_INFO_SIZE + PMU_HEAD_SIZE + i * entry_size
            if offset + entry_size > len(raw_bytes):
                break
            values = struct.unpack_from(data_fmt, raw_bytes, offset)
            self._task_list.append(
                [
                    values[0],  # sub graph id (seqNo)
                    values[1],  # sub task id (taskId)
                    values[2],  # total cycle
                ]
                + list(values[3:])
            )

    @property
    def data(self: any) -> list():
        if self._type != MSPROF_REPORT_AICPU_HCCL_OP_INFO or self._data_type != PROF_DATATYPE_PMU:
            return []
        print("----------------------------------------------------------")
        print("TileFwk pmu get level: ", self._level)
        print("TileFwk pmu get type: ", self._type)
        print("TileFwk pmu get head: ", self._head)
        print("TileFwk pmu get core id: ", self._core_id)
        print("TileFwk pmu get data type: ", self._data_type)
        print("TileFwk pmu get task count: ", self._task_count)
        print("----------------------------------------------------------")
        task_head_list = [self._thread_id, self._task_id, self._stream_id, self._core_id]
        print("task_head_list: ", task_head_list)
        print("task_pmu_list: ", self._task_list)
        for item in self._task_list:
            item[0:0] = task_head_list
        return self._task_list


def parse(file_path: str) -> list():
    bean_list = []
    print("start to parse")
    for name in os.listdir(file_path):
        if "aicpu.data" not in name:
            continue
        print("support data: " + name)
        # 构建完整的文件或目录路径
        file_name = os.path.join(file_path, name)
        # 如果是文件，则打印其路径
        print("support file path: ", file_name)
        file_size = os.path.getsize(file_name)
        if not file_size and "complete" not in file_name:
            print(file_name + " file size is empty.")
            return bean_list

        struct_size = PMU_RECORD_SIZE
        with open(file_name, "rb") as fd:
            context = fd.read()
            for _index in range(file_size // struct_size):
                data = context[_index * struct_size:(_index + 1) * struct_size]
                append_list = TileFwkPmuStructBean(data).data
                if not append_list:
                    continue
                bean_list += append_list
    print("total_pmu_list: ", bean_list)
    return bean_list


def main():
    args = _parse_args()
    task_pmu_list = _load_task_pmu_list(args.path)
    if not task_pmu_list:
        return
    table_header = _build_table_header(task_pmu_list, args.arch, args.pmuEvent)
    if not table_header:
        return
    task_pmu_list = _trim_task_pmu_list(task_pmu_list, len(table_header))
    _print_and_save(task_pmu_list, table_header, args.output)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="请输入要解析的绝对路径")
    parser.add_argument(
        "-pe", "--pmuEvent", nargs="?", type=int, default=2, choices=[1, 2, 4, 5, 6, 7, 8], help="pmuEvent."
    )
    parser.add_argument('--output', default='', help="pmu数据存储路径")
    parser.add_argument(
        '--arch', default='dav_2201', choices=['dav_2201', 'dav_3510'], help="指定架构类型, 默认dav_2201"
    )
    return parser.parse_args()


def _load_task_pmu_list(path):
    print("start parser pmu data:" + path)
    task_pmu_list = parse(path)
    if not task_pmu_list:
        print("empty pmu list")
        return []
    print("end parser pmu data")
    return task_pmu_list


def _get_pmu_header_maps():
    table_pmu_header_2201 = {
        1: [
            "cube_fp16_exec",
            "cube_int8_exec",
            "vec_fp32_exec",
            "vec_fp16_128lane_exec",
            "vec_fp16_64lane_exec",
            "vec_int32_exec",
            "vec_misc_exec",
        ],
        2: [
            "vec_busy_cycles",
            "cube_busy_cycles",
            "scalar_busy_cycles",
            "mte1_busy_cycles",
            "mte2_busy_cycles",
            "mte3_busy_cycles",
            "icache_miss",
            "icache_req",
        ],
        4: [
            "ub_read_req",
            "ub_write_req",
            "l1_read_req",
            "l1_write_req",
            "l2_read_req",
            "l2_write_req",
            "main_read_req",
            "main_write_req",
        ],
        5: ["l0a_read_req", "l0a_read_req", "l0b_read_req", "l0b_write_req", "l0c_read_req", "l0c_write_req"],
        6: ["bankgroup_stall_cycles", "bank_stall_cycles", "vec_resc_conflict_cycles"],
        7: [
            "ub_read_bw_mte",
            "l2_write_bw",
            "main_mem_write_bw",
            "ub_write_bw_mte",
            "ub_read_bw_vector",
            "ub_write_bw_vector",
            "ub_read_bw_scalar",
            "ub_write_bw_scalar",
        ],
        8: [
            "write_cache_hit",
            "write_cache_miss_allocate",
            "r0_read_cache_hit",
            "r0_read_cache_miss_allocate",
            "r1_read_cache_hit",
            "r1_read_cache_miss_allocate",
        ],
    }

    table_pmu_header_3510 = {
        1: ["cube_fp_instr_busy", "cube_int_instr_busy"],
        2: [
            "pmu_idc_aic_vec_busy_o",
            "cube_instr_busy",
            "scalar_instr_busy",
            "mte1_instr_busy",
            "mte2_instr_busy",
            "mte3_instr_busy",
            "icache_req",
            "icache_miss",
            "pmu_fix_instr_busy",
        ],
        4: [
            "bif_sc_pmu_read_main_instr_core",
            "bif_sc_pmu_write_main_instr_core",
            "pmu_aiv_ext_rd_ub_instr",
            "ub_pmu_vec_rd_ub_acc",
            "pmu_aiv_ext_wr_ub_instr",
            "ub_pmu_vec_wr_ub_acc",
            "pmu_rd_l1_instr",
            "pmu_wr_l1_instr",
        ],
        5: [
            "cube_sc_pmu_read_l0a_instr",
            "pmu_wr_l0a_instr",
            "cube_sc_pmu_read_l0b_instr",
            "pmu_wr_l0b_instr",
            "fixp_rd_l0c_instr",
            "cube_sc_pmu_read_l0c_instr",
            "cube_sc_pmu_write_l0c_instr",
        ],
        6: ["stu_pmu_wctl_ub_cflt", "ldu_pmu_ib_ub_cflt", "pmu_idc_aic_vec_instr_vf_busy_o", "idu_pmu_ins_iss_cnt"],
        7: [
            "pmu_rd_acc_ub_instr_p",
            "pmu_wr_acc_ub_instr_p",
            "pmu_fix_wr_ub_instr",
            "mte_sc_pmu_write_acc_ub_instr_0",
            "mte_sc_pmu_read_acc_ub_instr_0",
            "ub_pmu_vec_rd_ub_acc",
            "ub_pmu_vec_wr_ub_acc",
        ],
        8: [
            "bif_sc_pmu_ar_close_l2_hit_core",
            "bif_sc_pmu_ar_close_l2_miss_core",
            "bif_sc_pmu_ar_close_l2_victim_core",
            "bif_sc_pmu_aw_close_l2_hit_core",
            "bif_sc_pmu_aw_close_l2_miss_core",
            "bif_sc_pmu_aw_close_l2_victim_core",
        ],
    }
    return table_pmu_header_2201, table_pmu_header_3510


def _calc_pmu_cnt_num(task_pmu_list, base_len):
    if not task_pmu_list:
        return 0
    pmu_cnt_num = max(len(item) - base_len for item in task_pmu_list)
    return max(pmu_cnt_num, 0)


def _select_arch_headers(arch, pmu_event):
    table_pmu_header_2201, table_pmu_header_3510 = _get_pmu_header_maps()
    if arch == 'dav_3510':
        return table_pmu_header_3510.get(pmu_event, [])
    if arch == 'dav_2201':
        return table_pmu_header_2201.get(pmu_event, [])
    print("invalid arch: " + arch)
    return None


def _build_table_header(task_pmu_list, arch, pmu_event):
    table_header = ["thread id", "task id", "stream id", "core id", "seqNo", "sub task id", "total cycle"]
    pmu_cnt_num = _calc_pmu_cnt_num(task_pmu_list, len(table_header))
    table_pmu_header = _select_arch_headers(arch, pmu_event)
    if table_pmu_header is None:
        return []
    if not table_pmu_header and pmu_cnt_num > 0:
        table_pmu_header = [f"pmu_cnt{i}" for i in range(pmu_cnt_num)]
    elif pmu_cnt_num > 0 and len(table_pmu_header) > pmu_cnt_num:
        table_pmu_header = table_pmu_header[:pmu_cnt_num]
    return table_header + table_pmu_header


def _trim_task_pmu_list(task_pmu_list, expected_len):
    if expected_len <= 0:
        return task_pmu_list
    return [item[:expected_len] for item in task_pmu_list]


def _print_and_save(task_pmu_list, table_header, output_dir):
    print(tabulate(task_pmu_list, headers=table_header, tablefmt='grid', floatfmt=".1f", showindex="always"))
    df = pd.DataFrame(task_pmu_list, columns=table_header)
    if output_dir == "":
        df.to_csv('tilefwk_prof_pmu.csv', index=False)
    else:
        df.to_csv(f'{output_dir}/tilefwk_prof_pmu.csv', index=False)


if __name__ == '__main__':
    main()
