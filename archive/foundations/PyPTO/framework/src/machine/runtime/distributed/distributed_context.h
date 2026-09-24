/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file distributed_context.h
 * \brief
 */

#pragma once

#include <vector>
#include <utility>
#include <string>
#include "hccl_context.h"
#include "tilefwk/platform.h"
#include "interface/tileop/distributed/comm_context.h"
#include "securec.h"

namespace {
class TilingStructBase {
public:
    TilingStructBase() {}
    virtual ~TilingStructBase() {}
    virtual int32_t MakeMc2TilingStruct(const std::string& groupName) = 0;
    virtual void* GetMc2CommConfig() = 0;

private:
    std::string groupName_{};
};

class TilingStruct : public TilingStructBase {
public:
    TilingStruct() {}
    ~TilingStruct() {}
    int32_t MakeMc2TilingStruct(const std::string& groupName) override
    {
        (void)memset_s(&Mc2CommConfig_, sizeof(Mc2CommConfig_), 0, sizeof(Mc2CommConfig_));
        constexpr uint32_t version = 2;
        constexpr uint32_t hcommCnt = 1;
        constexpr uint32_t opTypeAllToAll = 6; // numeric representation of AlltoAll
        const char* algConfig = "AllGather=level0:ring";

        Mc2CommConfig_.version = version;
        Mc2CommConfig_.hcommCnt = hcommCnt;
        Mc2CommConfig_.hcommCfg.skipLocalRankCopy = 0;
        Mc2CommConfig_.hcommCfg.skipBufferWindowCopy = 0;
        Mc2CommConfig_.hcommCfg.stepSize = 0;
        Mc2CommConfig_.hcommCfg.opType = opTypeAllToAll;
        if (strcpy_s(Mc2CommConfig_.hcommCfg.groupName, sizeof(Mc2CommConfig_.hcommCfg.groupName), groupName.c_str()) !=
            EOK) {
            return -1;
        }
        if (strcpy_s(Mc2CommConfig_.hcommCfg.algConfig, sizeof(Mc2CommConfig_.hcommCfg.algConfig), algConfig) != EOK) {
            return -1;
        }
        return 0;
    }
    void* GetMc2CommConfig() override { return &Mc2CommConfig_; }

private:
    npu::tile_fwk::Mc2CommConfig Mc2CommConfig_;
};

class TilingStructV2 : public TilingStructBase {
public:
    TilingStructV2() {}
    ~TilingStructV2() {}
    int32_t MakeMc2TilingStruct(const std::string& groupName) override
    {
        (void)memset_s(&Mc2CommConfig_, sizeof(Mc2CommConfig_), 0, sizeof(Mc2CommConfig_));
        if (npu::tile_fwk::Platform::Instance().GetSoc().GetNPUArch() == npu::tile_fwk::NPUArch::DAV_3510) {
            Mc2CommConfig_.inner.version = 100U;
            Mc2CommConfig_.inner.commEngine = 3;
            (void)memset_s(Mc2CommConfig_.inner.reserved, sizeof(Mc2CommConfig_.inner.reserved), 0,
                           sizeof(Mc2CommConfig_.inner.reserved));
        } else {
            Mc2CommConfig_.inner.version = 1;
        }
        const char* algConfig = "BatchWrite=level0:fullmesh";
        Mc2CommConfig_.init.version = 100U;
        Mc2CommConfig_.init.mc2HcommCnt = 1;
        Mc2CommConfig_.init.queueNum = 0;
        Mc2CommConfig_.init.commBlockNum = 48U;
        Mc2CommConfig_.init.devType = 4U;
        Mc2CommConfig_.inner.skipLocalRankCopy = 0;
        Mc2CommConfig_.inner.skipBufferWindowCopy = 0;
        Mc2CommConfig_.inner.stepSize = 0;
        Mc2CommConfig_.inner.opType = 18U;
        Mc2CommConfig_.inner.version = 1;
        Mc2CommConfig_.init.offset[0] = static_cast<uint32_t>(reinterpret_cast<uint64_t>(&Mc2CommConfig_.inner) -
                                                              reinterpret_cast<uint64_t>(&Mc2CommConfig_.init));
        auto ret = strcpy_s(Mc2CommConfig_.inner.groupName, npu::tile_fwk::GROUP_NAME_SIZE, groupName.c_str());
        if (ret != 0) {
            return -1;
        }
        ret = strcpy_s(Mc2CommConfig_.inner.algConfig, npu::tile_fwk::ALG_CONFIG_SIZE, algConfig);
        if (ret != 0) {
            return -1;
        }
        return 0;
    }
    void* GetMc2CommConfig() override { return &Mc2CommConfig_; }

private:
    npu::tile_fwk::Mc2CommConfigV2 Mc2CommConfig_;
};
} // namespace

namespace npu::tile_fwk::dynamic {
constexpr int WIN_TYPE_NUM = 3; // win区类型in, status, debug
enum class ResType { RING_A2, MESH_A3, MESH_A5, UNKNOWN };

class DistributedContext {
public:
    DistributedContext() {};
    ~DistributedContext() {};
    static std::vector<uint64_t> GetCommContext(const std::vector<std::string>& groupNames);
    static std::vector<uint64_t> GetCommContextToHost(const std::vector<std::string>& groupNames);
    template <ResType T>
    static uint64_t AllocCommContext(const uint64_t ctxAddr, const std::string& groupName);

private:
    template <typename T>
    static void FillCommCtxAttr(TileOp::CommContext* ctxHost, T* hcclParamhost);
    template <typename T>
    static void FillCommCtxWinArr(uint32_t i, TileOp::CommContext* ctxHost, T* hcclParamhost);
};
} // namespace npu::tile_fwk::dynamic
