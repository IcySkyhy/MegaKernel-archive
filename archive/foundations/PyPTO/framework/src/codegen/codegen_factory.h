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
 * \file codegen.h
 * \brief
 */

#ifndef CODEGEN_FACTORY_H
#define CODEGEN_FACTORY_H

#include <unordered_set>
#include <utility>

#include "codegen_cce.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "npu/cloudnpu/codegen_cloudnpu.h"
#include "tilefwk/error_code.h"
#include "npu/litenpu/codegen_litenpu.h"

namespace npu::tile_fwk {
class CodeGenFactory {
public:
    static std::shared_ptr<CodeGenCCE> GetCodeGenCCE(const CodeGenCtx& ctx)
    {
        auto platform = Platform::Instance().GetSoc().GetNPUArch();
        if (platform == NPUArch::DAV_2201 || platform == NPUArch::DAV_3510) {
            return std::make_shared<CodeGenCloudNPU>(ctx);
        } else if (IsLiteNPU(platform)) {
            return std::make_shared<CodeGenLiteNPU>(ctx);
        }
        ASSERT(FwkErr::PLATFORM_NOT_SUPPORTED, false)
            << " can not support this platform: " << ToUnderlying(platform) << ", please check environment";
        return nullptr;
    }
};

} // namespace npu::tile_fwk

#endif // CODEGEN_FACTORY_H
