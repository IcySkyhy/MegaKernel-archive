# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compare_cos(input_1, input_2):
    input_1 = input_1.reshape(-1).astype(np.float64)
    input_2 = input_2.reshape(-1).astype(np.float64)
    logger.info("max diff: %s", np.max(np.abs(input_1 - input_2)))
    index = np.argmax(np.abs(input_1 - input_2))
    logger.info("max diff index = %s, intput_1 value: %s, intput_2 value: %s", index, input_1[index], input_2[index])
    logger.info("average diff: %s", np.mean(np.abs(input_1 - input_2)))
    ab = np.sum(input_1 * input_2)
    aa = np.sqrt(np.sum(input_1 * input_1))
    bb = np.sqrt(np.sum(input_2 * input_2))
    if aa == 0 and bb == 0:
        cos = 1.0
    elif aa == 0 or bb == 0:
        cos = 0.0
    else:
        cos = ab / (aa * bb)
    logger.info("cosine similarity: %s", cos)
    return cos


def check_nan(tensor, name=""):
    """Raise AssertionError if tensor contains any NaN values."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
        prefix = f"{name}: " if name else ""
        raise AssertionError(f"{prefix}tensor contains NaN values")


def load_bin(filepath, dtype=np.float16):
    with open(filepath, 'rb') as f:
        content = f.read()
    data = np.frombuffer(content, dtype=dtype)
    return data


def save_bin(output, filepath):
    filepath = filepath.replace("/", "_")
    with open(filepath, 'wb') as f:
        f.write(output.tobytes())
        f.close()
    return
