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
 * \file PvModelImpl.cpp
 * \brief
 */

#include <iostream>
#include <string>

#include "codegen/codegen.h"
#include "PvModelImpl.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"

using namespace npu::tile_fwk;

namespace CostModel {
static std::string FileName(const std::string& path)
{
    const char* separator = "/";

    size_t pos = path.find_last_of(separator);
    if (pos == std::string::npos) {
        return path;
    } else {
        return path.substr(pos + 1);
    }
}

uint64_t GetBinSize(std::string path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    uint64_t fileSize = file.tellg();
    file.close();
    return fileSize;
}

void DynPvModelImpl::Run(DynFuncData* funcDataList, int coreId, int funcId, int taskId)
{
    SIMULATION_LOGI("[AICORE] core  %d, func %d, task %d", coreId, funcId, taskId);
    CostModel::OutputSilencer silencer;
    silencer.silence();
    auto funcData = &funcDataList[funcId];
    auto opAttrs = &funcData->opAttrs[funcData->opAtrrOffsets[taskId]];
    auto funcIdx = opAttrs[0] + funcData->exprTbl[0];
    auto cce = &cceBin[funcIdx];

    if (cce->coreType != CoreType::AIV && cce->coreType != CoreType::AIC && cce->coreType != CoreType::MIX) {
        return;
    }

    RunModel(cce, funcData, opAttrs);
    silencer.restore();
}

void DynPvModelImpl::RunModel(PvModelCceBin* cce, DynFuncData* funcdata, uint64_t* opAttrs)
{
    auto binName = FileName(cce->binPath);
    this->subcoreId_ = binName.find("aiv") != std::string::npos ? static_cast<uint64_t>(1) : static_cast<uint64_t>(0);
    pv_launch_sub_core_(binAddr_, cce->binPath.c_str(), subcoreId_, coreId_);
    pv_reg_write_(static_cast<uint32_t>(PV_REG_SPR), PV_REG_PC, reinterpret_cast<uint8_t*>(&binAddr_), subcoreId_,
                  coreId_);
    uint64_t binSize = GetBinSize(cce->binPath);
    binAddr_ += binSize;

    std::vector<uint8_t> headerAndData(sizeof(DynFuncHeader) + sizeof(DynFuncData), 0);
    auto* header = reinterpret_cast<DynFuncHeader*>(headerAndData.data());
    header->funcNum = 1;
    header->funcSize = 1;
    auto* data = reinterpret_cast<DynFuncData*>(headerAndData.data() + sizeof(DynFuncHeader));
    *data = *funcdata;
    uint64_t headerAddr = reinterpret_cast<uint64_t>(headerAndData.data());
    pv_mem_write_(PV_MEM_GM, headerAddr, headerAndData.size(), headerAndData.data(), subcoreId_, coreId_);
    pv_mem_write_(PV_MEM_GM, reinterpret_cast<uint64_t>(funcdata->opAttrs), funcdata->opAttrSize * sizeof(uint64_t),
                  reinterpret_cast<uint8_t*>(funcdata->opAttrs), subcoreId_, coreId_);
    pv_mem_write_(PV_MEM_GM, reinterpret_cast<uint64_t>(funcdata->exprTbl), funcdata->exprNum * sizeof(uint64_t),
                  reinterpret_cast<uint8_t*>(funcdata->exprTbl), subcoreId_, coreId_);
    pv_mem_write_(PV_MEM_GM, reinterpret_cast<uint64_t>(funcdata->rawTensorDesc),
                  funcdata->rawTensorDescSize * sizeof(DevRawTensorDesc),
                  reinterpret_cast<uint8_t*>(funcdata->rawTensorDesc), subcoreId_, coreId_);
    pv_mem_write_(PV_MEM_GM, reinterpret_cast<uint64_t>(funcdata->rawTensorAddr),
                  funcdata->rawTensorAddrSize * sizeof(uint64_t), reinterpret_cast<uint8_t*>(funcdata->rawTensorAddr),
                  subcoreId_, coreId_);

    std::vector<uint64_t> paraArgs;
    paraArgs.push_back(headerAddr);
    pv_mem_write_(PV_MEM_GM, reinterpret_cast<uint64_t>(opAttrs), sizeof(uint64_t), reinterpret_cast<uint8_t*>(opAttrs),
                  subcoreId_, coreId_);
    paraArgs.push_back(reinterpret_cast<uint64_t>(opAttrs));
    pv_mem_write_(PV_MEM_GM, HBM_PARA_BASE, paraArgs.size() * sizeof(uint64_t),
                  reinterpret_cast<uint8_t*>(paraArgs.data()), subcoreId_, coreId_);

    pv_step_(PV_STEP_PIPE_ID, subcoreId_, coreId_, 0);
}

extern "C" std::shared_ptr<DynPvModel> CreateDynPvModelImpl() { return std::make_shared<DynPvModelImpl>(); }
} // namespace CostModel
