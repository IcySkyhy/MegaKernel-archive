/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pre_def.h
 * \brief
 */

#pragma once

#include <vector>
#include <memory>

namespace npu::tile_fwk {
class Tensor;
class RawTensor;
class LogicalTensor;
class Operation;
class Function;
class Program;
class TileRange;
class Element;
class SymbolicScalar;
class OperationsViewer;
struct FunctionInterpreter;

using LogicalTensorPtr = std::shared_ptr<LogicalTensor>;
using LogicalTensors = std::vector<LogicalTensorPtr>;

using BinDataPtr = uint8_t*;

constexpr int INVALID_TIME = -1;
constexpr int NOT_IN_SUBGRAPH = -1;
} // namespace npu::tile_fwk
