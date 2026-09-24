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

import dataclasses
import json
import logging
import os
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import torch

root_path: Path = Path(Path(__file__).parent, "../../../../../../").resolve()
scripts_path: Path = Path(root_path, 'cmake/scripts')
if str(scripts_path) not in sys.path:
    sys.path.append(str(scripts_path))
from golden_register import GoldenRegister  # noqa: E402

np.random.seed(0)
torch.manual_seed(0)

DTYPE_STR_TO_TORCH = {
    'bool': torch.bool,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'float16': torch.float16,
    'float32': torch.float,
    'bfloat16': torch.bfloat16,
}

TORCH_DTYPE_TO_NUM = {
    torch.bool: 15,
    torch.uint8: 11,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 6,
    torch.float: 7,
    torch.bfloat16: 8,
}


@dataclasses.dataclass(frozen=True)
class ValueRange:
    min_val: int
    max_val: int


@dataclasses.dataclass(frozen=True)
class BaseCase:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    world_size: int
    valid_shape: Tuple[int, ...]
    tile_shape: Tuple[int, ...]
    value_range: ValueRange


@dataclasses.dataclass(frozen=True)
class GenTensorCase:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    world_size: int
    value_range: ValueRange


@dataclasses.dataclass
class AllGatherAttnPostReducescatterCase:
    dtype: torch.dtype
    batch_size: int
    seq_len: int
    num_heads: int
    kv_lora_rank: int
    value_head_dim: int
    output_hidden_size: int
    world_size: int
    value_range: ValueRange


@dataclasses.dataclass
class DistributedOpAndSaveArgs:
    inputs: List[torch.Tensor]
    world_size: int
    shape: Tuple[int, ...]
    valid_shape: Tuple[int, ...]
    save_dir: Path
    filename_prefix: str


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in DTYPE_STR_TO_TORCH:
        raise ValueError(f'Unsupported dtype: {dtype_str}')
    return DTYPE_STR_TO_TORCH[dtype_str]


def get_dtype_num(dtype: torch.dtype) -> int:
    if dtype not in TORCH_DTYPE_TO_NUM:
        raise ValueError(f'Unsupported dtype: {dtype}')
    return TORCH_DTYPE_TO_NUM[dtype]


def parse_base_case(config: dict) -> BaseCase:
    params = config['params']
    world_size = params['world_size']
    input_tensor = config['input_tensors'][0]
    shape = tuple(input_tensor['shape'])
    valid_shape = tuple(config['view_shape'])
    dtype = get_dtype(input_tensor['dtype'])
    min_val, max_val = input_tensor['data_range']['min'], input_tensor['data_range']['max']
    tile_shape = tuple(config['tile_shape'])
    value_range = ValueRange(min_val=min_val, max_val=max_val)
    case = BaseCase(
        dtype=dtype,
        shape=shape,
        valid_shape=valid_shape,
        world_size=world_size,
        tile_shape=tile_shape,
        value_range=value_range,
    )
    return case


def validate_world_size(world_size: int) -> None:
    if world_size <= 1:
        raise ValueError(f'world_size must be greater than 1, got {world_size}')


def save_params(params: Tuple[int, ...], save_dir: Path) -> None:
    params_tensor = torch.tensor(params, dtype=torch.int64)
    params_ndarray = params_tensor.numpy()
    params_ndarray.tofile(save_dir / 'params.bin')


def save_tensor(tensor: torch.Tensor, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)  # 仅改变 tensor 的 dtype 解释方式，内存布局不变
    tensor.numpy().tofile(save_path)


def save_tensor_list(tensors: List[torch.Tensor], save_dir: Path, filename_prefix: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for rank, tensor in enumerate(tensors):
        save_tensor(tensor, save_dir / f'{filename_prefix}_rank_{rank}.bin')


def generate_random_tensor(shape: Tuple[int, ...], dtype: torch.dtype, value_range: ValueRange) -> torch.Tensor:
    spec_value_map = {'nan': np.nan, 'inf': np.inf, '-inf': -np.inf}
    if value_range.min_val in spec_value_map:
        return torch.full(shape, spec_value_map[value_range.min_val], dtype=dtype)
    if dtype in (torch.int32, torch.int16, torch.int8):
        return torch.randint(low=int(value_range.min_val), high=int(value_range.max_val), size=shape, dtype=dtype)
    else:
        return torch.randn(shape, dtype=dtype)


def generate_random_tensor_list(gen_tensor_case: GenTensorCase) -> List[torch.Tensor]:
    return [
        generate_random_tensor(gen_tensor_case.shape, gen_tensor_case.dtype, gen_tensor_case.value_range)
        for _ in range(gen_tensor_case.world_size)
    ]


def generate_random_tensor_list_and_save(
    gen_tensor_case: GenTensorCase,
    save_dir: Path,
    filename_prefix: str,
) -> List[torch.Tensor]:
    tensor_list = generate_random_tensor_list(gen_tensor_case)
    save_tensor_list(tensor_list, save_dir, filename_prefix)
    return tensor_list


def load_test_cases_from_json(json_file: str) -> list:
    with open(json_file, 'r') as data_file:
        json_data = json.load(data_file)
    if json_data is None:
        raise ValueError(f'Json file {json_file} is invalid.')
    file_name = json_file.stem
    if 'test_cases' in json_data:
        test_cases = json_data['test_cases']
    else:
        test_cases = [json_data]
    for tc in test_cases:
        tc['file_name'] = file_name
    test_cases.sort(key=lambda x: x['case_index'])
    return test_cases


def all_gather_and_save(args: DistributedOpAndSaveArgs) -> torch.Tensor:
    row, col = args.shape
    valid_row, valid_col = args.valid_shape
    dtype = args.inputs[0].dtype
    valid_inputs = [inp[:valid_row, :valid_col] for inp in args.inputs]
    gathered_valid = torch.cat(valid_inputs, dim=0)
    output = torch.full((row * args.world_size, col), 0, dtype=dtype)
    output[:valid_row * args.world_size, :valid_col] = gathered_valid
    outputs = [output] * args.world_size
    save_tensor_list(outputs, args.save_dir, args.filename_prefix)
    return outputs


def reduce_scatter_and_save(
    inputs: List[torch.Tensor],
    row: int,
    world_size: int,
    save_dir: Path,
    filename_prefix: str,
) -> torch.Tensor:
    stacked_output = torch.stack(inputs, dim=0)
    reduced_output = torch.sum(stacked_output, dim=0).to(inputs[0].dtype)
    row_per_rank = row // world_size
    outputs = [reduced_output[rank * row_per_rank:(rank + 1) * row_per_rank] for rank in range(world_size)]
    save_tensor_list(outputs, save_dir, filename_prefix)
    return outputs


def all_reduce_and_save(args: DistributedOpAndSaveArgs) -> torch.Tensor:
    row, col = args.shape
    valid_row, valid_col = args.valid_shape
    dtype = args.inputs[0].dtype
    valid_parts = [inp[:valid_row, :valid_col] for inp in args.inputs]
    reduced_valid = torch.sum(torch.stack(valid_parts, dim=0), dim=0).to(dtype)
    reduced_output = torch.zeros((row, col), dtype=dtype, device=args.inputs[0].device)
    reduced_output[:valid_row, :valid_col] = reduced_valid
    outputs = [reduced_output for _ in range(args.world_size)]
    save_tensor_list(outputs, args.save_dir, args.filename_prefix)
    return outputs


def generate_allgather_attn_post_reducescatter_case(config: dict) -> AllGatherAttnPostReducescatterCase:
    params = config['params']
    input_tensor = config['input_tensors'][0]
    dtype = get_dtype(input_tensor['dtype'])
    batch_size = params['batch_size']
    seq_len = params['seq_len']
    num_heads = params['num_heads']
    kv_lora_rank = params['kv_lora_rank']
    value_head_dim = params['value_head_dim']
    output_hidden_size = params['output_hidden_size']
    world_size = params['world_size']
    min_val, max_val = input_tensor['data_range']['min'], input_tensor['data_range']['max']
    value_range = ValueRange(min_val=min_val, max_val=max_val)
    case = AllGatherAttnPostReducescatterCase(
        dtype=dtype,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        value_head_dim=value_head_dim,
        output_hidden_size=output_hidden_size,
        world_size=world_size,
        value_range=value_range,
    )
    return case


def generate_all_gather_golden(config: dict, output: Path) -> bool:
    case = parse_base_case(config)
    validate_world_size(case.world_size)
    params = (*case.shape, *case.valid_shape, get_dtype_num(case.dtype), *case.tile_shape)
    save_params(params, output)
    gen_tensor_case = GenTensorCase(
        dtype=case.dtype, shape=case.shape, world_size=case.world_size, value_range=case.value_range
    )
    inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'input')
    all_gather_and_save(
        DistributedOpAndSaveArgs(
            inputs=inputs,
            world_size=case.world_size,
            shape=case.shape,
            valid_shape=case.valid_shape,
            save_dir=output,
            filename_prefix='allgather_out',
        )
    )
    return True


def gen_2_groups_allgather_golden(config: dict, output: Path) -> bool:
    case = parse_base_case(config)
    validate_world_size(case.world_size)
    params = (*case.shape, *case.valid_shape, get_dtype_num(case.dtype), *case.tile_shape)
    save_params(params, output)
    even_size = case.world_size // 2 + case.world_size % 2
    odd_size = case.world_size // 2
    gen_tensor_case_even = GenTensorCase(
        dtype=case.dtype, shape=case.shape, world_size=even_size, value_range=case.value_range
    )
    gen_tensor_case_odd = GenTensorCase(
        dtype=case.dtype, shape=case.shape, world_size=odd_size, value_range=case.value_range
    )
    inputs_odd = generate_random_tensor_list_and_save(gen_tensor_case_odd, output, 'input_odd')
    inputs_even = generate_random_tensor_list_and_save(gen_tensor_case_even, output, 'input_even')
    all_gather_and_save(
        DistributedOpAndSaveArgs(
            inputs=inputs_odd,
            world_size=odd_size,
            shape=case.shape,
            valid_shape=case.valid_shape,
            save_dir=output,
            filename_prefix='output_odd',
        )
    )
    all_gather_and_save(
        DistributedOpAndSaveArgs(
            inputs=inputs_even,
            world_size=even_size,
            shape=case.shape,
            valid_shape=case.valid_shape,
            save_dir=output,
            filename_prefix='output_even',
        )
    )
    return True


def generate_reduce_scatter_golden(config: dict, output: Path) -> bool:
    case = parse_base_case(config)
    validate_world_size(case.world_size)
    row = case.shape[0]
    if row % case.world_size != 0:
        raise ValueError(
            'The first dimension of the input tensor must be an integer multiple of the world size, '
            f'got row={row}, world_size={case.world_size}'
        )
    params = (*case.shape, get_dtype_num(case.dtype), *case.tile_shape)
    save_params(params, output)
    gen_tensor_case = GenTensorCase(
        dtype=case.dtype, shape=case.shape, world_size=case.world_size, value_range=case.value_range
    )
    inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'input')
    reduce_scatter_and_save(inputs, row, case.world_size, output, 'output')
    return True


def generate_all_reduce_golden(config: dict, output: Path) -> bool:
    case = parse_base_case(config)
    validate_world_size(case.world_size)
    row = case.shape[0]
    if row == 0:
        raise ValueError(
            f'The first dimension of the input tensor must not be zero, got row={row}, world_size={case.world_size}'
        )
    params = config['params']
    use_two_shot = params['use_two_shot']
    params = (*case.shape, *case.valid_shape, get_dtype_num(case.dtype), *case.tile_shape, use_two_shot)
    save_params(params, output)
    gen_tensor_case = GenTensorCase(
        dtype=case.dtype, shape=case.shape, world_size=case.world_size, value_range=case.value_range
    )
    inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'input')
    all_reduce_and_save(
        DistributedOpAndSaveArgs(
            inputs=inputs,
            world_size=case.world_size,
            shape=case.shape,
            valid_shape=case.valid_shape,
            save_dir=output,
            filename_prefix='output',
        )
    )
    return True


def generate_allreduce_add_allreduce_golden(config: dict, output: Path) -> bool:
    case = parse_base_case(config)
    validate_world_size(case.world_size)
    params = (*case.shape, *case.valid_shape, get_dtype_num(case.dtype))
    save_params(params, output)
    gen_tensor_case = GenTensorCase(
        dtype=case.dtype, shape=case.shape, world_size=case.world_size, value_range=case.value_range
    )
    inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'input')
    all_reduce_outs = all_reduce_and_save(
        DistributedOpAndSaveArgs(
            inputs=inputs,
            world_size=case.world_size,
            shape=case.shape,
            valid_shape=case.valid_shape,
            save_dir=output,
            filename_prefix='all_reduce_out',
        )
    )
    add_outs = [all_reduce_outs[0] + all_reduce_outs[0] for _ in range(case.world_size)]
    save_tensor_list(add_outs, output, 'add_out')
    all_reduce_and_save(
        DistributedOpAndSaveArgs(
            inputs=add_outs,
            world_size=case.world_size,
            shape=case.shape,
            valid_shape=case.valid_shape,
            save_dir=output,
            filename_prefix='out',
        )
    )
    return True


def prepare_attention_input(case, output: Path):
    all_gather_input_shape = (
        case.batch_size * case.seq_len * case.num_heads // case.world_size,
        case.kv_lora_rank,
    )
    gen_tensor_case = GenTensorCase(
        dtype=case.dtype, shape=all_gather_input_shape, world_size=case.world_size, value_range=case.value_range
    )
    all_gather_inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'ag_in')
    attention_input = torch.cat(all_gather_inputs, dim=0)
    attention_input = attention_input.reshape([case.batch_size, case.num_heads, case.seq_len, case.kv_lora_rank])
    attention_input = torch.transpose(attention_input, 1, 2)
    attention_input = attention_input.reshape([case.batch_size * case.seq_len, case.num_heads, case.kv_lora_rank])
    attention_input = torch.transpose(attention_input, 0, 1)
    return attention_input


def compute_attention_outputs(attention_input, case, output: Path):
    reduce_scatter_inputs = []
    for rank in range(case.world_size):
        lora_weight = generate_random_tensor(
            (case.num_heads, case.kv_lora_rank, case.value_head_dim), case.dtype, case.value_range
        )
        save_tensor(lora_weight, output / f'w_lora_rank_{rank}.bin')
        attention_output = torch.bmm(attention_input.to(torch.float32), lora_weight.to(torch.float32)).to(
            dtype=case.dtype
        )
        attention_output = torch.transpose(attention_output, 0, 1)
        attention_output = torch.reshape(
            attention_output, [case.batch_size * case.seq_len, case.num_heads * case.value_head_dim]
        )
        output_weight = generate_random_tensor(
            (case.num_heads * case.value_head_dim, case.output_hidden_size), case.dtype, case.value_range
        )
        save_tensor(output_weight, output / f'w_out_rank_{rank}.bin')
        attention_output = torch.matmul(
            attention_output.to(dtype=torch.float32), output_weight.to(dtype=torch.float32)
        ).to(dtype=case.dtype)
        reduce_scatter_inputs.append(attention_output)
    return reduce_scatter_inputs


def gen_allgather_attnpost_reducescatter_case(config: dict, output: Path) -> bool:
    case = generate_allgather_attn_post_reducescatter_case(config)
    params = (
        case.batch_size,
        case.seq_len,
        case.num_heads,
        case.kv_lora_rank,
        case.value_head_dim,
        case.output_hidden_size,
        get_dtype_num(case.dtype),
    )
    save_params(params, output)
    attention_input = prepare_attention_input(case, output)
    reduce_scatter_inputs = compute_attention_outputs(attention_input, case, output)
    reduce_scatter_and_save(reduce_scatter_inputs, case.batch_size * case.seq_len, case.world_size, output, 'rs_out')
    return True


def get_case_files() -> list[Path]:
    case_file = os.environ.get('JSON_PATH')
    case_path = Path(case_file) if case_file else Path(Path(__file__).parent.parent, 'test_case').resolve()
    if case_path.is_file():
        logging.info('loading single JSON file: %s', case_path)
        return [case_path]
    if case_path.is_dir():
        logging.info('loading all JSON files form directory: %s', case_path)
        files = list(case_path.glob("*.json"))
        files.sort(key=lambda x: x.name.lower())
        if not files:
            raise ValueError('JSON files found in the directory: %s', case_path)
        return files
    raise ValueError('Invalid path: %s. It must be either a valid file or a directory.', case_path)


def load_all_test_configs(case_files: list[Path]) -> list[dict]:
    all_test_configs = []
    for json_file in case_files:
        test_configs = load_test_cases_from_json(json_file)
        if test_configs:
            all_test_configs.extend(test_configs)
    if not all_test_configs:
        raise ValueError('No test cases loaded.')
    return all_test_configs


def generate_output_path(output: Path, test_config: dict, index: int = None) -> Path:
    case_str = f"{test_config['case_index']}_{test_config['case_name']}"
    operation = test_config['operation']
    file_name = test_config['file_name']
    if index is None:
        output_path = output.parent / operation / file_name / case_str
    else:
        output_path = Path(*output.parts[:-2]) / operation / file_name / case_str
    return output_path


OPERATOR_DISPATCHERS = {
    'AllGather': generate_all_gather_golden,
    'ReduceScatter': generate_reduce_scatter_golden,
    'AllReduce': generate_all_reduce_golden,
    'AllReduceAddAllReduce': generate_allreduce_add_allreduce_golden,
    'AllGatherAttnPostReduceScatter': gen_allgather_attnpost_reducescatter_case,
    'MultiCommGroupsOp': gen_2_groups_allgather_golden,
}


def generate_single_golden(config: dict, output: Path):
    op_name = config['operation']
    if not op_name:
        raise ValueError(f'No operation field: {config}')
    handler = OPERATOR_DISPATCHERS[op_name]
    if handler is None:
        raise ValueError(f"Unsupported operation: {op_name}")
    handler(config, output)
    logging.info('Generate golden for success op: %s (case_name: %s)', op_name, config['case_name'])


@GoldenRegister.reg_golden_func(
    case_names=[
        'TestDistributedOps/DistributedTest.TestOps',
    ],
    version=3,
)
def generate_golden_case(case_name: str, output: Path, case_index: int = None) -> bool:
    case_files = get_case_files()
    all_test_configs = load_all_test_configs(case_files)
    if case_index is None:
        for test_config in all_test_configs:
            output_path1 = generate_output_path(output, test_config)
            output_path1.mkdir(parents=True, exist_ok=True)
            generate_single_golden(test_config, output_path1)
    else:
        if case_index >= len(all_test_configs):
            raise IndexError(f'case_index {case_index} out of range')
        test_config = all_test_configs[case_index]
        output = generate_output_path(output, test_config, case_index)
        output.mkdir(parents=True, exist_ok=True)
        generate_single_golden(test_config, output)
    return True
