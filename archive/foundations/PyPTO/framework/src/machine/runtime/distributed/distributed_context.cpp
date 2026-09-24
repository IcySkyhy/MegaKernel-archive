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
 * \file distributed_context.cpp
 * \brief
 */

#include "distributed_context.h"

#include <memory>
#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/hcomm_api.h"
#include "adapter/api/acl_api.h"
#include "interface/utils/common.h"
#include "tilefwk/error_code.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "machine/runtime/context/stream_context.h"
#include "machine/device/dynamic/device_utils.h"

constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
constexpr uint32_t COMM_MESH = 0b1u;
constexpr uint32_t WINDATA_INDEX = 0;
constexpr uint32_t WINSTATUS_INDEX = 1;
constexpr uint32_t WINDEBUG_INDEX = 2;
constexpr uint64_t WIN_EXP_SIZE = 1024UL * 1024UL;
std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>
    g_context; // key: groupname; value: deviceCommContext,hostCommContext

using namespace npu::tile_fwk;

namespace npu::tile_fwk::dynamic {
uint8_t* AllocHostAddr(size_t size)
{
    if (size == 0 || size > 0x500000000) {
        DISTRIBUTED_LOGE(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, "AllocHostAddr failed size %zu bytes", size);
        return nullptr;
    }
    auto hostPtr = (uint8_t*)malloc(size);
    memset_s(hostPtr, size, 0, size);
    return hostPtr;
}

template <typename T>
void DistributedContext::FillCommCtxAttr(TileOp::CommContext* ctxHost, T* hcclParamhost)
{
    (void)ctxHost;
    (void)hcclParamhost;
}

template <>
void DistributedContext::FillCommCtxAttr<npu::tile_fwk::HcclCombinOpParamA5>(
    TileOp::CommContext* ctxHost, npu::tile_fwk::HcclCombinOpParamA5* hcclParamhost)
{
    ctxHost->rankId = hcclParamhost->rankId;
    ctxHost->rankNum = hcclParamhost->rankNum;
    // winAddr内存排布 startIndex:[0,rankNum-1], statusIndex:[rankNum, 2*rankNum-1], debugIndex:[2*rankNum, 3*rankNum-1]
    ctxHost->statusIndex = hcclParamhost->rankNum;
    ctxHost->debugIndex = hcclParamhost->rankNum * 2;
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, hcclParamhost->winSize > WIN_EXP_SIZE)
        << "winSize: " << hcclParamhost->winSize << " need greater than WIN_EXP_SIZE: " << WIN_EXP_SIZE;
    ctxHost->winDataSize = hcclParamhost->winSize - WIN_EXP_SIZE;
    ctxHost->winStatusSize = WIN_EXP_SIZE;
    ctxHost->winDebugSize = hcclParamhost->winSize;
    ctxHost->totalWinNum = hcclParamhost->rankNum * WIN_TYPE_NUM;
}

template <>
void DistributedContext::FillCommCtxAttr<npu::tile_fwk::HcclCombinOpParam>(
    TileOp::CommContext* ctxHost, npu::tile_fwk::HcclCombinOpParam* hcclParamhost)
{
    ctxHost->rankId = hcclParamhost->rankId;
    ctxHost->rankNum = hcclParamhost->rankNum;
    ctxHost->statusIndex = hcclParamhost->rankNum;
    ctxHost->debugIndex = hcclParamhost->rankNum * 2;
    ctxHost->winDataSize = hcclParamhost->winSize;
    ctxHost->winStatusSize = hcclParamhost->winExpSize;
    ctxHost->winDebugSize = hcclParamhost->winSize;
    ctxHost->totalWinNum = hcclParamhost->rankNum * WIN_TYPE_NUM;
}

template <>
void DistributedContext::FillCommCtxAttr<npu::tile_fwk::HcclOpResParamHead>(
    TileOp::CommContext* ctxHost, npu::tile_fwk::HcclOpResParamHead* hcclParamhost)
{
    ctxHost->rankId = hcclParamhost->localUsrRankId;
    ctxHost->rankNum = hcclParamhost->rankSize;
    ctxHost->statusIndex = hcclParamhost->rankSize;
    ctxHost->debugIndex = hcclParamhost->rankSize * 2;
    ctxHost->winDataSize = hcclParamhost->winSize;
    ctxHost->winStatusSize = hcclParamhost->winExpSize;
    ctxHost->winDebugSize = hcclParamhost->winSize;
    ctxHost->totalWinNum = hcclParamhost->rankSize * WIN_TYPE_NUM;
}

template <typename T>
void DistributedContext::FillCommCtxWinArr(uint32_t i, TileOp::CommContext* ctxHost, T* hcclParamhost)
{
    (void)i;
    (void)ctxHost;
    (void)hcclParamhost;
}

template <>
void DistributedContext::FillCommCtxWinArr<npu::tile_fwk::HcclCombinOpParamA5>(
    uint32_t i, TileOp::CommContext* ctxHost, npu::tile_fwk::HcclCombinOpParamA5* hcclParamhost)
{
    ctxHost->winAddr[i + (WINDATA_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsIn[i] + WIN_EXP_SIZE;
    ctxHost->winAddr[i + (WINSTATUS_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsIn[i];
    ctxHost->winAddr[i + (WINDEBUG_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsOut[i];
}

template <>
void DistributedContext::FillCommCtxWinArr<npu::tile_fwk::HcclCombinOpParam>(
    uint32_t i, TileOp::CommContext* ctxHost, npu::tile_fwk::HcclCombinOpParam* hcclParamhost)
{
    ctxHost->winAddr[i + (WINDATA_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsIn[i];
    ctxHost->winAddr[i + (WINSTATUS_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsExp[i];
    ctxHost->winAddr[i + (WINDEBUG_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsOut[i];
}

template <>
void DistributedContext::FillCommCtxWinArr<npu::tile_fwk::HcclOpResParamHead>(
    uint32_t i, TileOp::CommContext* ctxHost, npu::tile_fwk::HcclOpResParamHead* hcclParamhost)
{
    ctxHost->winAddr[i + (WINDATA_INDEX * ctxHost->rankNum)] = hcclParamhost->localWindowsIn;
    ctxHost->winAddr[i + (WINSTATUS_INDEX * ctxHost->rankNum)] = hcclParamhost->localWindowsExp;
    ctxHost->winAddr[i + (WINDEBUG_INDEX * ctxHost->rankNum)] = hcclParamhost->localWindowsOut;
}

template <>
void DistributedContext::FillCommCtxWinArr<npu::tile_fwk::HcclRankRelationResV2>(
    uint32_t i, TileOp::CommContext* ctxHost, npu::tile_fwk::HcclRankRelationResV2* hcclParamhost)
{
    ctxHost->winAddr[i + (WINDATA_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsIn;
    ctxHost->winAddr[i + (WINSTATUS_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsExp;
    ctxHost->winAddr[i + (WINDEBUG_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsOut;
}

template <ResType T>
uint64_t DistributedContext::AllocCommContext([[maybe_unused]] const uint64_t ctxAddr,
                                              [[maybe_unused]] const std::string& groupName)
{
    return 0;
}

uint64_t AllocateAndSetupCommContext(void* paramHost, uint32_t rankNum, const std::string& groupName,
                                     std::function<void(TileOp::CommContext*, void*)> fillAttrFunc,
                                     std::function<void(uint32_t, TileOp::CommContext*, void*)> fillWinArrFunc)
{
    size_t commCtxSize = sizeof(TileOp::CommContext) + sizeof(uint64_t) * rankNum * WIN_TYPE_NUM;

    TileOp::CommContext* ctxHost = static_cast<TileOp::CommContext*>(static_cast<void*>(AllocHostAddr(commCtxSize)));
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ctxHost != nullptr) << "ctxHost malloc failed";

    fillAttrFunc(ctxHost, paramHost);
    for (uint32_t i = 0; i < rankNum; i++) {
        fillWinArrFunc(i, ctxHost, paramHost);
    }

    TileOp::CommContext* ctxDevice = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(reinterpret_cast<uint8_t**>(&ctxDevice), commCtxSize);
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ctxDevice != nullptr) << "ctxDevice malloc failed";

    auto ret = AclRtMemcpy(ctxDevice, commCtxSize, ctxHost, commCtxSize, AclRtMemcpyKind::HOST_TO_DEVICE);
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ret == 0) << "AclRtMemcpy failed, error: " << ret;
    g_context[groupName] = std::make_pair(reinterpret_cast<uint64_t>(ctxDevice), reinterpret_cast<uint64_t>(ctxHost));
    return reinterpret_cast<uint64_t>(ctxDevice);
}

template <>
uint64_t DistributedContext::AllocCommContext<ResType::MESH_A5>([[maybe_unused]] const uint64_t ctxAddr,
                                                                [[maybe_unused]] const std::string& groupName)
{
    npu::tile_fwk::HcclCombinOpParamA5* hcclParamDevice = (npu::tile_fwk::HcclCombinOpParamA5*)ctxAddr;
    npu::tile_fwk::HcclCombinOpParamA5* hcclParamhost = (npu::tile_fwk::HcclCombinOpParamA5*)AllocHostAddr(
        sizeof(npu::tile_fwk::HcclCombinOpParamA5));
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, hcclParamhost != nullptr) << "hcclParamhost malloc failed";
    size_t offsetRankId = offsetof(npu::tile_fwk::HcclCombinOpParamA5, rankId);
    size_t offsetXnAddr = offsetof(npu::tile_fwk::HcclCombinOpParamA5, xnAddr);
    auto ret = AclRtMemcpy(&(hcclParamhost->rankId), offsetXnAddr - offsetRankId, &(hcclParamDevice->rankId),
                           offsetXnAddr - offsetRankId, AclRtMemcpyKind::DEVICE_TO_HOST);
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ret == 0) << "AclRtMemcpy failed, error: " << ret;
    return AllocateAndSetupCommContext(
        hcclParamhost, hcclParamhost->rankNum, groupName,
        [](TileOp::CommContext* ctx, void* param) {
            auto* p = static_cast<npu::tile_fwk::HcclCombinOpParamA5*>(param);
            FillCommCtxAttr(ctx, p);
        },
        [](uint32_t i, TileOp::CommContext* ctx, void* param) {
            auto* p = static_cast<npu::tile_fwk::HcclCombinOpParamA5*>(param);
            FillCommCtxWinArr(i, ctx, p);
        });
}

template <>
uint64_t DistributedContext::AllocCommContext<ResType::MESH_A3>([[maybe_unused]] const uint64_t ctxAddr,
                                                                [[maybe_unused]] const std::string& groupName)
{
    npu::tile_fwk::HcclCombinOpParam* hcclParamDevice = (npu::tile_fwk::HcclCombinOpParam*)ctxAddr;
    npu::tile_fwk::HcclCombinOpParam* hcclParamhost = (npu::tile_fwk::HcclCombinOpParam*)AllocHostAddr(
        sizeof(npu::tile_fwk::HcclCombinOpParam));
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, hcclParamhost != nullptr) << "hcclParamhost malloc failed";
    size_t offsetRankId = offsetof(npu::tile_fwk::HcclCombinOpParam, rankId);
    size_t offsetHcomId = offsetof(npu::tile_fwk::HcclCombinOpParam, hcomId);
    size_t offsetWinExpSize = offsetof(npu::tile_fwk::HcclCombinOpParam, winExpSize);
    size_t offsetMultiServerFlag = offsetof(npu::tile_fwk::HcclCombinOpParam, multiServerFlag);
    auto ret = AclRtMemcpy(&(hcclParamhost->rankId), offsetHcomId - offsetRankId, &(hcclParamDevice->rankId),
                           offsetHcomId - offsetRankId, AclRtMemcpyKind::DEVICE_TO_HOST);
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ret == 0) << "AclRtMemcpy failed, error: " << ret;
    ret = AclRtMemcpy(&(hcclParamhost->winExpSize), offsetMultiServerFlag - offsetWinExpSize,
                      &(hcclParamDevice->winExpSize), offsetMultiServerFlag - offsetWinExpSize,
                      AclRtMemcpyKind::DEVICE_TO_HOST);
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ret == 0) << "AclRtMemcpy failed, error: " << ret;

    return AllocateAndSetupCommContext(
        hcclParamhost, hcclParamhost->rankNum, groupName,
        [](TileOp::CommContext* ctx, void* param) {
            auto* p = static_cast<npu::tile_fwk::HcclCombinOpParam*>(param);
            FillCommCtxAttr(ctx, p);
        },
        [](uint32_t i, TileOp::CommContext* ctx, void* param) {
            auto* p = static_cast<npu::tile_fwk::HcclCombinOpParam*>(param);
            FillCommCtxWinArr(i, ctx, p);
        });
}

template <>
uint64_t DistributedContext::AllocCommContext<ResType::RING_A2>([[maybe_unused]] const uint64_t ctxAddr,
                                                                [[maybe_unused]] const std::string& groupName)
{
    npu::tile_fwk::HcclOpResParam* hcclParam = (npu::tile_fwk::HcclOpResParam*)ctxAddr;
    HcclOpResParamHead* hcclParamhost = (HcclOpResParamHead*)AllocHostAddr(sizeof(HcclOpResParamHead));
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, hcclParamhost != nullptr) << "hcclParamhost malloc failed";
    size_t offsetLocalUsrRankId = offsetof(npu::tile_fwk::HcclOpResParam, localUsrRankId);
    size_t offsetRWinStart = offsetof(npu::tile_fwk::HcclOpResParam, rWinStart);
    auto ret = AclRtMemcpy(&(hcclParamhost->localUsrRankId), offsetRWinStart - offsetLocalUsrRankId,
                           &(hcclParam->localUsrRankId), offsetRWinStart - offsetLocalUsrRankId,
                           AclRtMemcpyKind::DEVICE_TO_HOST);
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ret == 0) << "AclRtMemcpy failed, error: " << ret;

    size_t remoteResSize = hcclParamhost->rankSize * sizeof(npu::tile_fwk::RemoteResPtr);
    npu::tile_fwk::RemoteResPtr* remoteResPtr = (npu::tile_fwk::RemoteResPtr*)AllocHostAddr(remoteResSize);
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, remoteResPtr != nullptr) << "remoteResPtr malloc failed";
    ret = AclRtMemcpy(remoteResPtr, remoteResSize, &(hcclParam->remoteRes), remoteResSize,
                      AclRtMemcpyKind::DEVICE_TO_HOST);
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ret == 0) << "AclRtMemcpy failed, error: " << ret;

    size_t commCtxSize = sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankSize * WIN_TYPE_NUM;
    TileOp::CommContext* ctxHost = (TileOp::CommContext*)AllocHostAddr(commCtxSize);
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ctxHost != nullptr) << "ctxHost malloc failed";
    FillCommCtxAttr<npu::tile_fwk::HcclOpResParamHead>(ctxHost, hcclParamhost);
    for (uint64_t i = 0; i < hcclParamhost->rankSize; i++) {
        if (i == hcclParamhost->localUsrRankId) {
            FillCommCtxWinArr<npu::tile_fwk::HcclOpResParamHead>(i, ctxHost, hcclParamhost);
            continue;
        }
        uint64_t remoteResDevicePtr;
        ret = AclRtMemcpy(&remoteResDevicePtr, sizeof(uint64_t), &(remoteResPtr[i].nextDevicePtr), sizeof(uint64_t),
                          AclRtMemcpyKind::DEVICE_TO_HOST); // 设备二级指针值拷贝到主机
        ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ret == 0) << "AclRtMemcpy failed, error: " << ret;
        npu::tile_fwk::HcclRankRelationResV2 remoteParam;
        ret = AclRtMemcpy(&remoteParam, sizeof(HcclRankRelationResV2), (void*)remoteResDevicePtr,
                          sizeof(HcclRankRelationResV2), AclRtMemcpyKind::DEVICE_TO_HOST);
        ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ret == 0) << "AclRtMemcpy failed, error: " << ret;
        FillCommCtxWinArr<npu::tile_fwk::HcclRankRelationResV2>(i, ctxHost, &remoteParam);
    }
    TileOp::CommContext* ctxDevice = nullptr;
    DevMemoryPool::Instance().AllocDevAddr((uint8_t**)&ctxDevice, commCtxSize);
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ctxDevice != nullptr) << "ctxDevice malloc failed";
    ret = AclRtMemcpy(ctxDevice,
                      sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankSize * WIN_TYPE_NUM, ctxHost,
                      sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankSize * WIN_TYPE_NUM,
                      AclRtMemcpyKind::HOST_TO_DEVICE);
    ASSERT(DistributedErrorCode::CONTEXT_CONFIGURE_FAILED, ret == 0) << "AclRtMemcpy failed, error: " << ret;
    g_context[groupName].first = (uint64_t)ctxDevice;
    g_context[groupName].second = (uint64_t)ctxHost;
    return (uint64_t)ctxDevice;
}

std::vector<uint64_t> DistributedContext::GetCommContext([[maybe_unused]] const std::vector<std::string>& groupNames)
{
    if (groupNames.size() == 0) {
        return {};
    }
    HCommTopo topoRet;
    const char* group = groupNames[0].c_str();
    ASSERT(DistributedErrorCode::INVALID_HCCL_TOPO,
           HcommGetL0TopoTypeEx(group, &topoRet, COMM_IS_NOT_SET_DEVICE) == HCOMM_SUCCESS)
        << "Get hcom topo failed";
    uint32_t topoType = static_cast<uint32_t>(topoRet);
    std::shared_ptr<TilingStructBase> tilingStruct;
    if (topoType == COMM_MESH) {
        tilingStruct = std::make_shared<TilingStruct>();
    } else {
        tilingStruct = std::make_shared<TilingStructV2>();
    }
    std::vector<uint64_t> commContext(groupNames.size(), 0);
    ASSERT(DistributedErrorCode::INVALID_GROUP_COUNT, groupNames.size() <= DIST_COMM_GROUP_NUM)
        << "The number of communication groups (" << groupNames.size() << ") exceeds the maximum limit ("
        << DIST_COMM_GROUP_NUM << ").";
    for (size_t groupIndex = 0; groupIndex < groupNames.size(); ++groupIndex) {
        auto groupName = groupNames[groupIndex];
        if (g_context.find(groupName) != g_context.end()) { // 检查context缓存
            commContext[groupIndex] = g_context[groupName].first;
            continue;
        }
        HcommHandle commHandle = nullptr;
        ASSERT(DistributedErrorCode::INVALID_GROUP_NAME, HcommGetCommHandleByGroup(groupName.c_str(), &commHandle) == 0)
            << "Get hcom handle failed";
        tilingStruct->MakeMc2TilingStruct(groupName);
        auto ret = HcommAllocComResourceByTiling(commHandle, GetStreamContext().GetAiCoreStream(),
                                                 tilingStruct->GetMc2CommConfig(),
                                                 reinterpret_cast<void**>(&commContext[groupIndex]));
        ASSERT(DistributedErrorCode::HCCL_ALLOC_RESOURCE_FAILED, (ret == 0) && (commContext[groupIndex] != 0UL))
            << "Hccl alloc resource failed";
        DISTRIBUTED_LOGI("groupIndex=%zu, groupName=%s, commContext=%lu", groupIndex, groupName.c_str(),
                         commContext[groupIndex]);
        if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
            commContext[groupIndex] = AllocCommContext<ResType::MESH_A5>(commContext[groupIndex], groupName);
        } else if (topoType == COMM_MESH) {
            commContext[groupIndex] = AllocCommContext<ResType::MESH_A3>(commContext[groupIndex], groupName);
        } else {
            commContext[groupIndex] = AllocCommContext<ResType::RING_A2>(commContext[groupIndex], groupName);
        }
    }
    return commContext;
}

std::vector<uint64_t> DistributedContext::GetCommContextToHost(
    [[maybe_unused]] const std::vector<std::string>& groupNames)
{
    std::vector<uint64_t> devAddrs = GetCommContext(groupNames);
    std::vector<uint64_t> hostContext;
    ASSERT(DistributedErrorCode::INVALID_GROUP_COUNT, groupNames.size() <= DIST_COMM_GROUP_NUM)
        << "The number of communication groups (" << groupNames.size() << ") exceeds the maximum limit ("
        << DIST_COMM_GROUP_NUM << ").";
    for (size_t groupIndex = 0; groupIndex < groupNames.size(); groupIndex++) {
        std::string groupName = groupNames[groupIndex];
        hostContext.push_back((uint64_t)g_context[groupName].second);
    }
    return hostContext;
}
} // namespace npu::tile_fwk::dynamic
