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
"""
Invoke this script in `pypto` project root dir
"""

import argparse
from dataclasses import dataclass
import os
import struct
from typing import List, Optional, Tuple

import numpy as np
from thread_task_runner import ThreadTaskRunner
import torch

DEFAULT_MAX_WORKERS = 16
HUGE_TENSOR_THRESHOLD = 1024 * 1024  # numel threshold
PROPERTY_NUM = 6


def parse_task_id(task_id: int) -> Tuple[int, int]:  # root id + leaf call id
    taskid_task_bits = 16
    return (task_id >> taskid_task_bits, task_id & ((1 << taskid_task_bits) - 1))


@dataclass(frozen=True)
class RawTensorDesc:
    seq_no: int
    task_id: int
    raw_magic: int
    address: int
    dtype: str
    bytes_of_dtype: int
    shape: Tuple[int]
    io_mark: Optional[str]
    symlink_src: Optional[str] = None

    def numel(self) -> int:
        return np.prod(self.shape)

    def tensor_key(self) -> Tuple[int, str, Tuple[int]]:
        return (self.address, self.dtype, tuple(self.shape))

    def name(self) -> str:
        func_id, subtask_id = parse_task_id(self.task_id)
        name = f"{self.seq_no}-{func_id}-{subtask_id}-{self.raw_magic}"
        if self.io_mark is not None:
            name += f"-{self.io_mark}"
        return name

    def copy_with_symlink_src(self, symlink_src) -> "RawTensorDesc":
        return RawTensorDesc(
            seq_no=self.seq_no,
            task_id=self.task_id,
            raw_magic=self.raw_magic,
            address=self.address,
            dtype=self.dtype,
            bytes_of_dtype=self.bytes_of_dtype,
            shape=self.shape,
            io_mark=self.io_mark,
            symlink_src=symlink_src,
        )


def parse_tile_fwk_aicpu_ctrl(filename, inputs, outputs) -> List[RawTensorDesc]:
    io_map = {}

    def preprocess_io_map(datas, class_str: str):
        for idx, data in enumerate(datas):
            if data not in io_map:
                io_map[data] = class_str + str(idx)
            else:
                io_map[data] += '-' + class_str + str(idx)

    preprocess_io_map(inputs, 'i')
    preprocess_io_map(outputs, 'o')

    raw_tensors: List[RawTensorDesc] = []
    tensor_cache = {}
    with open(filename, 'r') as f:
        start_recording = False
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            if '[DumpTensor]' not in stripped_line:
                continue

            stripped_line = stripped_line.split('[DumpTensor]')[-1].strip(' "')

            if stripped_line.startswith(">>>"):
                start_recording = True
                continue

            if stripped_line.startswith("<<<"):
                start_recording = False
                continue

            if start_recording:
                splits = stripped_line.strip().split(',')
                seq_no, task_id, raw_magic, address, dtype, bytes_of_dtype = splits[:PROPERTY_NUM]
                seq_no, task_id, raw_magic, address, bytes_of_dtype = map(
                    int, (seq_no, task_id, raw_magic, address, bytes_of_dtype)
                )
                shape = tuple(map(lambda x: int(x.strip('()')), splits[PROPERTY_NUM:]))

                rt = RawTensorDesc(
                    seq_no=seq_no,
                    task_id=task_id,
                    raw_magic=raw_magic,
                    address=address,
                    dtype=dtype,
                    bytes_of_dtype=bytes_of_dtype,
                    shape=shape,
                    io_mark=None if address not in io_map else io_map[address],
                )

                key = rt.tensor_key()
                if key in tensor_cache:
                    symlink_src = tensor_cache[key]
                    if symlink_src != rt.name():
                        rt = rt.copy_with_symlink_src(symlink_src)
                else:
                    tensor_cache[key] = rt.name()

                raw_tensors.append(rt)
    raw_tensors = list(set(raw_tensors))
    return raw_tensors


class ByteTable:
    def __init__(self, binary_data, offset=0):
        self.blocks = []  # 每个元素是 (base_addr, size, data)
        self._parse(binary_data, offset)

    def query(self, addr_start, addr_end):
        for base, size, data in self.blocks:
            block_end = base + size
            # 判断是否有交集
            if addr_end <= base or addr_start >= block_end:
                continue
            if not (addr_start >= base and addr_end <= block_end):
                raise Exception("Unexpected memrange spanning multiple memblocks")
            # 计算交集范围
            offset_in_block = addr_start - base
            length = addr_end - addr_start
            return bytearray(data[offset_in_block:offset_in_block + length])
        raise Exception("Address mismatching")

    def _parse(self, data, offset=0):
        while offset < len(data):
            # 读取 base_addr 和 size
            base_addr, size = struct.unpack_from('<QQ', data, offset)
            offset += 16
            # 读取 data[size]
            block_data = data[offset:offset + size]
            offset += size
            self.blocks.append((base_addr, size, block_data))
            print(f"Parsed a binary data block | addr=0x{base_addr:X}, size={size}")


def read_uint64_list(binary_data: bytes, offset: int):
    # 读取 size（8 字节）
    size = struct.unpack_from('<Q', binary_data, offset)[0]
    offset += 8

    # 读取 size 个 uint64_t 元素（每个 8 字节）
    data_format = f'<{size}Q'
    data = list(struct.unpack_from(data_format, binary_data, offset))
    offset += size * 8

    return data, offset


def parse_dump_tensor_binary(filename) -> Tuple[ByteTable, List[int], List[int]]:
    with open(filename, 'rb') as f:
        binary_data = f.read()
        offset = 0
        inputs, offset = read_uint64_list(binary_data, offset)
        outputs, offset = read_uint64_list(binary_data, offset)
        return ByteTable(binary_data, offset), inputs, outputs


def pypto_dtype_to_torch_dtype(dtype: str) -> Optional[torch.dtype]:
    dtype_map = {
        "INT4": None,
        "INT8": torch.int8,
        "INT16": torch.int16,
        "INT32": torch.int32,
        "INT64": torch.int64,
        "FP8": None,
        "FP16": torch.float16,
        "FP32": torch.float32,
        "BF16": torch.bfloat16,
        "HF4": None,
        "HF8": None,
        "UINT8": torch.uint8,
        "UINT16": None,
        "UINT32": None,
        "UINT64": None,
        "BOOL": torch.bool,
        "DOUBLE": torch.double,
    }
    if dtype not in dtype_map:
        print(f"Invalid pypto dtype: {dtype}")
        return None

    torch_dtype = dtype_map[dtype]
    if torch_dtype is None:
        print(f"Cannot convert pypto dtype: {dtype} to corresponding torch dtype")
        return None

    return torch_dtype


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process process_dump_tensor.")
    parser.add_argument("dump_tensor_filename", type=str, help="Path to dump_tensor.txt")
    parser.add_argument("tile_fwk_aicpu_ctrl_filename", type=str, help="Path to tile_fwk_aicpu_ctrl.txt")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum number of threading workers, {DEFAULT_MAX_WORKERS} by default",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    binary_table, inputs, outputs = parse_dump_tensor_binary(args.dump_tensor_filename)
    raw_tensors = parse_tile_fwk_aicpu_ctrl(args.tile_fwk_aicpu_ctrl_filename, inputs, outputs)

    base_dir = os.path.dirname(args.dump_tensor_filename)
    dump_tensor_dir = os.path.join(base_dir, "dump_tensor")
    os.makedirs(dump_tensor_dir, exist_ok=True)

    torch.set_printoptions(
        threshold=1024**3,
        linewidth=1024**2,
    )

    print(f"In total {len(raw_tensors)} raw tensors to be processed")

    def seq_no_dir_str(seq_no: int):
        return f"seqNo-{seq_no}"

    for rt in raw_tensors:
        os.makedirs(os.path.join(dump_tensor_dir, seq_no_dir_str(rt.seq_no)), exist_ok=True)

    def dump_raw_tensor(rt: RawTensorDesc):
        dst_file = os.path.join(dump_tensor_dir, seq_no_dir_str(rt.seq_no), f"{rt.name()}.txt")
        if rt.symlink_src is not None:
            assert rt.symlink_src != rt.name(), f"Invalid symlink to self: {rt.name()}"  # No self-symlink

            src_seq_no = int(rt.symlink_src.split('-')[0])
            src_file = f"../{seq_no_dir_str(src_seq_no)}/{rt.symlink_src}.txt"
            if os.path.islink(dst_file) or os.path.exists(dst_file):
                os.remove(dst_file)
            os.symlink(src=src_file, dst=dst_file)
            return

        mem_req = rt.numel() * rt.bytes_of_dtype
        binary_data = binary_table.query(rt.address, rt.address + mem_req)
        torch_dtype = pypto_dtype_to_torch_dtype(rt.dtype)
        if torch_dtype is None:
            # error message already printed in pypto_dtype_to_torch_dtype
            return
        tensor = torch.frombuffer(binary_data, dtype=torch_dtype).reshape(rt.shape)

        with open(dst_file, 'w') as f:
            f.write(f"address=0x{rt.address:X}\nshape={tensor.shape}\ndtype={tensor.dtype}\n\n")
            f.write(str(tensor))

    def is_huge_tensor(rt: RawTensorDesc):
        return rt.numel() >= HUGE_TENSOR_THRESHOLD

    def task_info(rt: RawTensorDesc):
        return f"{rt.name()}: shape={list(rt.shape)}, dtype={rt.dtype}"

    runner = ThreadTaskRunner(title="Dump Tensor", max_workers=args.max_workers)
    runner.run_batch(raw_tensors, dump_raw_tensor, is_huge_tensor, task_info)

    print(f"Output files location: `{dump_tensor_dir}`")


if __name__ == "__main__":
    main()
