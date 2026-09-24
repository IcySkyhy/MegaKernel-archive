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
 * \file dev_encode.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_program.h"

namespace npu::tile_fwk::dynamic {
void EncodeDevAscendFunction(Function* dyndev, const EncodeDevAscendFunctionParam& param, uint64_t& offset,
                             DevAscendFunction* base);

void EncodeDevAscendProgram(Function* func, uint64_t& offset, DevAscendProgram* base,
                            const std::unordered_map<int, SlotMaskEntry>* stitchUpdateSlotMaskMap = nullptr);

std::unordered_map<int, SlotMaskEntry> BuildStitchUpdateSlotMaskMap(
    const std::shared_ptr<DyndevFunctionAttribute>& dyndevAttr);
} // namespace npu::tile_fwk::dynamic
