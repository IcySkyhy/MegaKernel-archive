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
 * \file gen_aicore_code.cpp
 * \brief
 */

#include "machine/compile/gen_aicore_code.h"
#include "utils/file_utils.h"

namespace npu::tile_fwk {
namespace {
const std::string kKernelEntryStr = "KERNEL_ENTRY";
const size_t kKernelEntryStrSize = 12;
const std::string kAicoreSrcCode = R"!!!(
#include "tilefwk/aicore_entry.h"

extern "C" __global__ __aicore__ void KERNEL_ENTRY(__OPTYPE__, __OPNAME__, __TILINGKEY__)(int64_t ffts_addr, int64_t inputs,
        int64_t outputs, int64_t workspace, int64_t tilingdata, int64_t cfgdata) {
    return KernelEntry(ffts_addr, inputs, outputs, workspace, tilingdata, cfgdata);
}
)!!!";
} // namespace

bool GenAicoreSrcFile(const std::string& codeSrcPath, const std::string& funcHash)
{
    std::string newSrcCode = kAicoreSrcCode;
    size_t pos = newSrcCode.find(kKernelEntryStr);
    while (pos != std::string::npos) {
        newSrcCode.replace(pos, kKernelEntryStrSize, kKernelEntryStr + "_" + funcHash);
        pos = newSrcCode.find(kKernelEntryStr, pos + kKernelEntryStrSize + 1);
    }
    if (RealPath(codeSrcPath).empty()) {
        SaveFile(codeSrcPath, kAicoreSrcCode);
    }
    return !RealPath(codeSrcPath).empty();
}
} // namespace npu::tile_fwk
