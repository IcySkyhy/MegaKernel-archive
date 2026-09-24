/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pass_manager.h"

namespace npu::tile_fwk {
extern "C" bool RunPass(Program& program, Function& function, const std::string& strategy)
{
    return PassManager::Instance().RunPass(program, function, strategy) == SUCCESS;
}

extern "C" std::string GetResumePath(const std::string& strategy)
{
    return PassManager::Instance().GetResumePath(strategy);
}

extern "C" void ResetAllPasses() { PassManager::Instance().ResetAllPasses(); }
} // namespace npu::tile_fwk
