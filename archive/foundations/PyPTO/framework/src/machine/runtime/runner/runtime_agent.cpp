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

#include "machine/runtime/runner/runtime_agent.h"
#include "tilefwk/platform.h"
#include "adapter/api/hal_api.h"

namespace npu::tile_fwk {
struct AddrMapInPara {
    unsigned int addr_type;
    unsigned int devid;
};

struct AddrMapOutPara {
    unsigned long long ptr;
    unsigned long long len;
};
namespace {
constexpr int32_t MODULE_TYPE_AI_CORE = 4;
constexpr int32_t INFO_TYPE_OCCUPY = 8;
constexpr uint8_t AICORE_MAP_BUFF_LEN = 2;
constexpr uint32_t SUB_CORE_PER_AICORE = 3;
namespace DAV_2201 {
constexpr uint32_t MAX_CORE = 25;
}

namespace DAV_3510 {
constexpr uint32_t MAX_CORE = 36;
}
bool GetPgMask(uint64_t& valid, int32_t& deviceId)
{
    deviceId = GetLogDeviceId();
    uint64_t aicore_bitmap[AICORE_MAP_BUFF_LEN] = {0};
    int32_t size_n = static_cast<int32_t>(sizeof(uint64_t)) * AICORE_MAP_BUFF_LEN;
    auto ret = HalGetDeviceInfoByBuff(static_cast<uint32_t>(deviceId), MODULE_TYPE_AI_CORE, INFO_TYPE_OCCUPY,
                                      reinterpret_cast<void*>(&aicore_bitmap[0]), &size_n);
    if (ret != HAL_ERROR_NONE) {
        return false;
    }
    valid = aicore_bitmap[0];
    return true;
}
} // namespace

int RuntimeAgent::GetAicoreRegInfo(std::vector<int64_t>& aic, std::vector<int64_t>& aiv, const int addrType)
{
    int32_t deviceId = 0;
    uint64_t valid = 0;
    if (!GetPgMask(valid, deviceId)) {
        MACHINE_LOGW("Get Device Info failed or no valid core exists.");
        valid = 0xFFFFFFFF;
        validGetPgMask = false;
    }
    MACHINE_LOGI("The valid cores are: %lu.", valid);
    uint64_t coreStride = 8 * 1024 * 1024; // 8M
    uint64_t subCoreStride = 0x100000ULL;

    auto isValid = [&valid](int id) {
        const uint64_t mask = (1ULL << 25) - 1;
        return ((static_cast<uint64_t>(valid) ^ mask) & (1ULL << id)) == 0;
    };
    struct AddrMapInPara inMapPara;
    struct AddrMapOutPara outMapPara;
    inMapPara.devid = deviceId;
    inMapPara.addr_type = addrType;
    auto ret = HalMemCtl(0, reinterpret_cast<void*>(&inMapPara), sizeof(struct AddrMapInPara),
                         reinterpret_cast<void*>(&outMapPara), nullptr);
    if (ret != HAL_ERROR_NONE) {
        MACHINE_LOGE(HostLauncherErr::MAP_REG_ADDR_FAILED,
                     "Map reg addr fail, maybe others are using current device. (ret=%d, DEVICE_ID=%d).", ret,
                     deviceId);
        return ret;
    }
    for (uint32_t i = 0; i < DAV_2201::MAX_CORE; i++) {
        for (uint32_t j = 0; j < SUB_CORE_PER_AICORE; j++) {
            uint64_t vaddr = 0UL;
            if (isValid(i)) {
                vaddr = outMapPara.ptr + (i * coreStride + j * subCoreStride);
            }
            if (j == 0) {
                aic.push_back(vaddr);
            } else {
                aiv.push_back(vaddr);
            }
        }
    }
    return 0;
}

void RuntimeAgent::GetAicoreRegInfoForDAV3510(std::vector<int64_t>& regs, std::vector<int64_t>& regsPmu)
{
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        return;
    }
    constexpr uint32_t AICORE_PER_DIE = 18;
    constexpr uint32_t AIV_BASE_OFFSET = 18;
    constexpr uint32_t SUB_CORE_PER_DIE = AICORE_PER_DIE * SUB_CORE_PER_AICORE;

    constexpr unsigned long SUB_CORE_STRIDE = 0x100000ULL;
    constexpr unsigned long AIV_STRIDE = SUB_CORE_STRIDE;
    constexpr unsigned long AIV_SECOND_STRIDE = 2 * SUB_CORE_STRIDE;
    constexpr size_t MAX_INDEX = DAV_3510::MAX_CORE * SUB_CORE_PER_AICORE;
    unsigned int devId = GetLogDeviceId();

    struct ResMapInfo mapInfo;
    mapInfo.target_proc_type = ProcessType::CP1;
    mapInfo.res_type = ResMapType::AICORE;
    mapInfo.flag = 0;
    mapInfo.rsv[0] = 0;

    regs.resize(MAX_INDEX);
    regsPmu.resize(MAX_INDEX);
    for (uint32_t coreIndex = 0; coreIndex < DAV_3510::MAX_CORE; coreIndex++) {
        mapInfo.res_id = coreIndex;
        unsigned long mapAddr;
        unsigned int len = 0x300000;
        (void)HalResMap(devId, &mapInfo, &mapAddr, &len);
        uint32_t dieIdx = coreIndex / AICORE_PER_DIE;
        uint32_t localIdx = coreIndex % AICORE_PER_DIE;
        uint32_t dieBase = dieIdx * SUB_CORE_PER_DIE;

        uint32_t aicoreIndex = dieBase + localIdx;
        uint32_t aivFirstIndex = dieBase + AIV_BASE_OFFSET + localIdx * 2;
        uint32_t aivSecondIndex = aivFirstIndex + 1;
        // aic
        regs[aicoreIndex] = mapAddr;
        regsPmu[aicoreIndex] = mapAddr;
        // first aiv
        regs[aivFirstIndex] = mapAddr + AIV_STRIDE;
        regsPmu[aivFirstIndex] = mapAddr + AIV_STRIDE;
        // second aiv
        regs[aivSecondIndex] = mapAddr + AIV_SECOND_STRIDE;
        regsPmu[aivSecondIndex] = mapAddr + AIV_SECOND_STRIDE;
    }
}
} // namespace npu::tile_fwk
