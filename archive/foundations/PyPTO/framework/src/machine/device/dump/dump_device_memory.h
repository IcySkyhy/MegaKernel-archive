/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"

namespace npu::tile_fwk {
class Function;
}

namespace npu::tile_fwk::dynamic {
struct DevAscendProgram;

namespace mem_dump {

void DumpMemoryOverview(const DevAscendProgram& devProg, Function* func, uint64_t workspaceSize);

void DumpRootMemory(uint32_t seqNo, uint32_t funcIdx, const DevAscendFunctionDupped& dup);

} // namespace mem_dump
} // namespace npu::tile_fwk::dynamic
