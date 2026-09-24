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
 * \file aicore_compiler.h
 * \brief
 */
#ifndef TILEFRAMEWORK_AICORE_COMPILER_H
#define TILEFRAMEWORK_AICORE_COMPILER_H
#include <string>
#include "tilefwk/platform.h"
#include "interface/function/function.h"
#include "machine/utils/dynamic/dev_encode_function.h"

namespace npu::tile_fwk {
int CompileAICoreKernel(std::map<uint64_t, Function*>& leafDict, dynamic::EncodeDevAscendFunctionParam& param,
                        const std::string& ccePath, const std::string& funcHash, const std::string& funcRawName,
                        std::string& kernelPath, bool enableSubFunc = true);
}
#endif // TILEFRAMEWORK_AICORE_COMPILER_H
