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
 * \file runtime_agent.cpp
 * \brief
 */

#include "machine/runtime/runner/runtime_utils.h"
#include <unistd.h>
#include "tilefwk/platform.h"
#include "machine/runtime/context/stream_context.h"

struct process_sign {
    pid_t tgid;
    char sign[49]; // 49 is PROCESS_SIGN_LENGTH
    char resv[4];  // 4 is PROCESS_RESV_LENGTH
};
extern "C" __attribute__((weak)) int drvGetProcessSign(process_sign* sign);

namespace npu::tile_fwk {
namespace {
constexpr uint32_t kMinDefaultDim = 20;
constexpr uint32_t AICAIVRATIO = 2; // AIC:AIV的比例系数

int GetMaxBlockdim()
{
    uint32_t cubeBlockDim = 0;
    uint32_t vectorBlockDim = 0;
    // 若未进行控核，AclRtGetStreamResLimit返回的是满核
    auto aicoreStream = GetStreamContext().GetCurrentStream();
    AclRtGetStreamResLimit(aicoreStream, AclRtDevResLimitType::CUBE_CORE, &cubeBlockDim);
    AclRtGetStreamResLimit(aicoreStream, AclRtDevResLimitType::VECTOR_CORE, &vectorBlockDim);
    // 若不满足AIC和AIV的比例，手动处理成为符合AIC和AIV的比例最大值
    if (vectorBlockDim != cubeBlockDim * AICAIVRATIO) {
        auto rtsMaxBlockDim = std::min(cubeBlockDim, vectorBlockDim / AICAIVRATIO);
        MACHINE_LOGW("The cubeBlockDim[%u] and vectorBlockDim[%u] do not conform to the 1: %u ratio of AIC and AIV, "
                     "and will be set to values that conform to the ratio of AIC and AIV. "
                     "The cubeBlockDim and vectorBlockDim are set at %u and %u",
                     cubeBlockDim, vectorBlockDim, AICAIVRATIO, rtsMaxBlockDim, rtsMaxBlockDim * AICAIVRATIO);
        return rtsMaxBlockDim;
    } else {
        return cubeBlockDim;
    }
}
} // namespace

int GetCfgBlockdim()
{
    auto blk = Platform::Instance().GetSoc().GetAICoreNum();
    blk = blk > 0 ? blk : kMinDefaultDim;

    // 通过GetMaxBlockdim接口获取设置的最大核数，如果设置的最大核数大于硬件物理最大核数时，控核不生效
    // 如果未进行控核，GetMaxBlockdim接口将通过AclRtGetStreamResLimit函数返回硬件物理最大核数
    auto maxBlk = GetMaxBlockdim();
    blk = (maxBlk > 0 && maxBlk < static_cast<int>(blk)) ? maxBlk : blk;
    MACHINE_LOGD("Get blockdim[%zu].", blk);
    return blk;
}

uint32_t GetProcessId()
{
    if (drvGetProcessSign != nullptr) {
        process_sign processSign;
        auto ret = drvGetProcessSign(&processSign);
        if (ret == 0) {
            MACHINE_LOGD("Got process sign from drv: tgid=%d", processSign.tgid);
            return static_cast<uint32_t>(processSign.tgid);
        }
        MACHINE_LOGW("drvGetProcessSign failed, ret=%d, falling back to getpid()", ret);
    } else {
        MACHINE_LOGW("drvGetProcessSign is nullptr, falling back to getpid()");
    }

    uint32_t pid = static_cast<uint32_t>(getpid());
    MACHINE_LOGD("Using getpid(): pid=%u", pid);
    return pid;
}
} // namespace npu::tile_fwk
