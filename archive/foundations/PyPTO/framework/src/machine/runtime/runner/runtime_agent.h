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
 * \file runtime_agent.h
 * \brief
 */

#pragma once

#include <vector>
#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/acl_api.h"
#include "machine/runtime/runner/runtime_utils.h"

namespace npu::tile_fwk {
constexpr int ADDR_MAP_TYPE_REG_AIC_CTRL = 2;
constexpr int ADDR_MAP_TYPE_REG_AIC_PMU_CTRL = 3;

class RuntimeAgent {
public:
    RuntimeAgent(RuntimeAgent& other) = delete;

    void operator=(const RuntimeAgent& other) = delete;

    static RuntimeAgent& GetAgent()
    {
        static RuntimeAgent inst;
        return inst;
    }
    int GetAicoreRegInfo(std::vector<int64_t>& aic, std::vector<int64_t>& aiv, const int addrType);
    static void GetAicoreRegInfoForDAV3510(std::vector<int64_t>& regs, std::vector<int64_t>& regsPmu);

    bool GetValidGetPgMask() const { return validGetPgMask; }

private:
    RuntimeAgent()
    {
        // don't call AclInit, it will cause camodel running fail
#ifndef RUN_WITH_ASCEND_CAMODEL
        inited_ = AclInit(nullptr) == ACLRT_SUCCESS;
        MACHINE_LOGD("Init acl runtime, ret is %d.", inited_);
#endif
        CheckDeviceId();
    }

    ~RuntimeAgent()
    {
        if (inited_) {
#ifndef RUN_WITH_ASCEND_CAMODEL
            AclFinalize();
            MACHINE_LOGD("Quit acl runtime.");
#endif
        }
        inited_ = false;
    }

private:
    bool inited_{false};
    bool validGetPgMask{true};
};
} // namespace npu::tile_fwk
