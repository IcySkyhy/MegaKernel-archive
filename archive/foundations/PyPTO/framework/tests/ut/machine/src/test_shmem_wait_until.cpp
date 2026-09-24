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
 * \file test_shmem_wait_until.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "machine/device/distributed/common.h"
#include "machine/device/distributed/shmem_wait_until.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/runtime/distributed/hccl_context.h"
#include "machine/device/dynamic/aicore_manager.h"

namespace {

struct TileIndexInfo {
    uint32_t tileShapeDim;
    uint32_t startDim;
    std::vector<uint32_t> viewshapes;
    std::vector<uint32_t> dimTileNums;
    std::vector<uint32_t> viewTileStrides;
    std::vector<uint32_t> viewIndexStrides;
    uint32_t viewTileNum;
    uint32_t viewIndexNum;
    uint32_t totalTileNum;
};

TileIndexInfo CalculateTileIndexInfo(const std::vector<uint32_t>& shmemSignalRawShape,
                                     const std::vector<uint32_t>& shmemSignalShape,
                                     const std::vector<uint32_t>& tileShape)
{
    uint32_t tileShapeDim = tileShape.size();
    uint32_t shmemSignalDim = shmemSignalRawShape.size();
    uint32_t startDim = shmemSignalDim - tileShapeDim;

    std::vector<uint32_t> viewshapes(tileShapeDim);
    std::vector<uint32_t> dimTileNums(tileShapeDim);
    std::vector<uint32_t> viewTileStrides(tileShapeDim);
    std::vector<uint32_t> viewIndexStrides(tileShapeDim);
    uint32_t viewTileNum = 1;
    uint32_t viewIndexNum = 1;

    viewTileStrides[0] = 1;
    viewIndexStrides[0] = 1;

    for (uint32_t i = 0; i < tileShapeDim; ++i) {
        uint32_t curDim = startDim + i;
        viewshapes[i] = shmemSignalShape[curDim];
        uint32_t totalShape = shmemSignalShape[curDim];
        uint32_t tileShapeVal = tileShape[i];
        dimTileNums[i] = (totalShape + tileShapeVal - 1) / tileShapeVal;
        viewTileNum *= dimTileNums[i];
        viewIndexNum *= shmemSignalRawShape[curDim] / shmemSignalShape[curDim];
        if (i > 0) {
            viewTileStrides[i] = viewTileStrides[i - 1] * dimTileNums[i - 1];
            viewIndexStrides[i] = viewIndexStrides[i - 1] *
                                  (shmemSignalRawShape[curDim - 1] / shmemSignalShape[curDim - 1]);
        }
    }
    uint32_t totalTileNum = viewTileNum * viewIndexNum;

    return TileIndexInfo{tileShapeDim,     startDim,    viewshapes,   dimTileNums, viewTileStrides,
                         viewIndexStrides, viewTileNum, viewIndexNum, totalTileNum};
}

uint32_t CalculateTileIndex(const TileIndexInfo& tileInfo, const std::vector<uint32_t>& shmemSignalOffset,
                            const std::vector<uint32_t>& tileShape)
{
    uint32_t tileIndex = 0;
    uint32_t viewIndexAccum = 0;

    for (uint32_t dimIdx = 0; dimIdx < tileInfo.tileShapeDim; ++dimIdx) {
        uint32_t curDim = tileInfo.startDim + dimIdx;
        uint32_t viewShape = tileInfo.viewshapes[dimIdx];
        uint32_t offset = shmemSignalOffset[curDim];
        uint32_t tileShapeVal = tileShape[dimIdx];

        uint32_t viewIdx = offset / viewShape;
        uint32_t viewOffset = offset % viewShape;
        uint32_t viewTileIdx = viewOffset / tileShapeVal;

        tileIndex += viewTileIdx * tileInfo.viewTileStrides[dimIdx];
        viewIndexAccum += viewIdx * tileInfo.viewIndexStrides[dimIdx];
    }
    tileIndex += viewIndexAccum * tileInfo.viewTileNum;

    return tileIndex;
}

std::vector<int32_t> InitializeShmemSignal(std::vector<uint32_t> shmemSignalRawShape,
                                           std::vector<uint32_t> shmemSignalOffset, std::vector<uint32_t> tileShape,
                                           uint32_t shmemSignalStride, int32_t expectedValue)
{
    auto tileInfo = CalculateTileIndexInfo(shmemSignalRawShape, shmemSignalRawShape, tileShape);
    uint32_t tileIndex = CalculateTileIndex(tileInfo, shmemSignalOffset, tileShape);

    uint32_t size = std::accumulate(shmemSignalRawShape.begin() + 1, shmemSignalRawShape.end(), 1,
                                    std::multiplies<uint32_t>());
    std::vector<int32_t> shmemSignal(size, 0);
    uint32_t index = (tileInfo.totalTileNum * shmemSignalOffset[0] + tileIndex) * shmemSignalStride;
    shmemSignal[index] = expectedValue;
    return shmemSignal;
}

constexpr size_t codeSize = 35;

struct AicpuCodeParams {
    uint32_t opcode;
    uint32_t oOperandTotalParamNum;
    uint32_t outDim;
    uint32_t outAttrOffset;
    uint32_t iOperandTotalParamNum;
    uint32_t predTokenDim;
    uint32_t predTokenAttrOffset;
    uint32_t shmemSignalDim;
    uint32_t shmemSignalShapeNum;
    uint32_t attrSize;
};

AicpuCodeParams PrepareAicpuCodeParams(const TileIndexInfo& tileInfo)
{
    uint32_t paramSizePerOperand = 2;
    uint32_t oOperandNum = 1;
    uint32_t iOperandNum = 2;
    uint32_t shmemSignalDim = 5;

    return AicpuCodeParams{.opcode = static_cast<uint32_t>(-1),
                           .oOperandTotalParamNum = paramSizePerOperand * oOperandNum,
                           .outDim = 2,
                           .outAttrOffset = static_cast<uint32_t>(-1),
                           .iOperandTotalParamNum = paramSizePerOperand * iOperandNum,
                           .predTokenDim = 2,
                           .predTokenAttrOffset = static_cast<uint32_t>(-1),
                           .shmemSignalDim = shmemSignalDim,
                           .shmemSignalShapeNum = shmemSignalDim * 2,
                           .attrSize = 3 + tileInfo.tileShapeDim + tileInfo.tileShapeDim * 3 + 2};
}

std::array<uint32_t, codeSize> BuildAicpuCodeData(const AicpuCodeParams& params,
                                                  const std::vector<uint32_t>& shmemSignalRawShape,
                                                  const std::vector<uint32_t>& shmemSignalShape,
                                                  const TileIndexInfo& tileInfo, uint32_t shmemSignalAttrOffset,
                                                  uint32_t shmemSignalStride, int32_t expectedValue,
                                                  const std::vector<uint32_t>& tileShape)
{
    uint32_t resetSignal = 0;

    return std::array<uint32_t, codeSize>{params.opcode,
                                          params.oOperandTotalParamNum,
                                          params.outDim,
                                          params.outAttrOffset,
                                          params.iOperandTotalParamNum,
                                          params.predTokenDim,
                                          params.predTokenAttrOffset,
                                          params.shmemSignalDim,
                                          shmemSignalAttrOffset,
                                          params.shmemSignalShapeNum,
                                          shmemSignalRawShape[0],
                                          shmemSignalRawShape[1],
                                          shmemSignalRawShape[2],
                                          shmemSignalRawShape[3],
                                          shmemSignalRawShape[4],
                                          shmemSignalShape[0],
                                          shmemSignalShape[1],
                                          shmemSignalShape[2],
                                          shmemSignalShape[3],
                                          shmemSignalShape[4],
                                          params.attrSize,
                                          static_cast<uint32_t>(expectedValue),
                                          shmemSignalStride,
                                          resetSignal,
                                          tileInfo.tileShapeDim,
                                          tileShape[0],
                                          tileShape[1],
                                          tileInfo.viewshapes[0],
                                          tileInfo.viewshapes[1],
                                          tileInfo.viewTileStrides[0],
                                          tileInfo.viewTileStrides[1],
                                          tileInfo.viewIndexStrides[0],
                                          tileInfo.viewIndexStrides[1],
                                          tileInfo.viewTileNum,
                                          tileInfo.totalTileNum};
}

auto InitializeAicpuCode(std::vector<uint32_t> shmemSignalRawShape, std::vector<uint32_t> tileShape,
                         uint32_t shmemSignalStride, int32_t expectedValue, uint32_t shmemSignalAttrOffset)
{
    std::vector<uint32_t> shmemSignalShape = shmemSignalRawShape;
    auto tileInfo = CalculateTileIndexInfo(shmemSignalRawShape, shmemSignalShape, tileShape);
    auto params = PrepareAicpuCodeParams(tileInfo);

    auto initData = BuildAicpuCodeData(params, shmemSignalRawShape, shmemSignalShape, tileInfo, shmemSignalAttrOffset,
                                       shmemSignalStride, expectedValue, tileShape);

    auto data = std::make_unique<int32_t[]>(codeSize);
    std::copy(initData.begin(), initData.end(), data.get());
    npu::tile_fwk::dynamic::DevRelocVector<int32_t> aicpuCode(codeSize, data.get());
    return std::make_tuple(std::move(data), std::move(aicpuCode));
}

auto InitializeTaskData(npu::tile_fwk::dynamic::DynDeviceTask* task)
{
    size_t headerSize = sizeof(npu::tile_fwk::DynFuncHeader);
    size_t dataSize = sizeof(npu::tile_fwk::DynFuncData);
    std::unique_ptr<void, decltype(&free)> buffer(
        malloc(headerSize + dataSize + sizeof(npu::tile_fwk::DevStartArgsBase) + sizeof(int64_t)), free);
    auto* header = new (buffer.get()) npu::tile_fwk::DynFuncHeader();
    auto* funcData = new (header + 1) npu::tile_fwk::DynFuncData();
    auto* startArgs = new (funcData + 1) npu::tile_fwk::DevStartArgsBase();
    auto* commContext = new (startArgs + 1) int64_t;
    startArgs->commContexts = commContext;
    header->startArgs = startArgs;

    task->dynFuncDataList = header;
    task->dynFuncDataList[0].seqNo = 1;
    task->dynFuncDataList[0].funcNum = 1;
    task->dynFuncDataList[0].funcSize = 1u;
    task->dynFuncDataList[0].cceBinary = nullptr;

    return std::make_tuple(std::move(buffer), header, funcData);
}

auto ConfigureFuncData(npu::tile_fwk::DynFuncHeader* header, npu::tile_fwk::DynFuncData* funcData, uint64_t rawAddr)
{
    constexpr size_t exprTblSize = 50;
    auto exprTbl = std::make_unique<uint64_t[]>(exprTblSize);
    funcData->exprTbl = exprTbl.get();

    auto hcclParam = std::make_unique<npu::tile_fwk::HcclCombinOpParam>();
    hcclParam->rankNum = 0;
    hcclParam->windowsIn[0] = rawAddr;

    auto rawTensorAddrHolder = std::make_unique<uint64_t[]>(1);
    auto rawTensorDescHolder = std::make_unique<npu::tile_fwk::DevRawTensorDesc[]>(1);
    rawTensorAddrHolder[0] = 0;
    rawTensorDescHolder[0] = {0, 0};
    funcData->rawTensorAddr = rawTensorAddrHolder.get();
    funcData->rawTensorDesc = rawTensorDescHolder.get();
    header->startArgs->commContexts[0] = reinterpret_cast<int64_t>(hcclParam.get());
    header->startArgs->commGroupNum = 1;

    constexpr size_t opAttrsLength = 17;
    auto opAttrs = std::make_unique<uint64_t[]>(opAttrsLength);
    std::fill_n(opAttrs.get(), opAttrsLength, 0);
    funcData->opAttrs = opAttrs.get();

    auto opAtrrOffsets = std::make_unique<int32_t[]>(1);
    opAtrrOffsets[0] = 0;
    funcData->opAtrrOffsets = opAtrrOffsets.get();

    return std::make_tuple(std::move(exprTbl), std::move(hcclParam), std::move(rawTensorAddrHolder),
                           std::move(rawTensorDescHolder), std::move(opAttrs), std::move(opAtrrOffsets));
}

auto InitializeTestEnvironment()
{
    uint32_t worldSize = 4;
    uint32_t shmemSignalRawShape2 = 1;
    uint32_t shmemSignalRawShape3 = 64;
    uint32_t shmemSignalRawShape4 = 5120;
    std::vector<uint32_t> shmemSignalRawShape{worldSize, worldSize, shmemSignalRawShape2, shmemSignalRawShape3,
                                              shmemSignalRawShape4};
    std::vector<uint32_t> shmemSignalOffset(shmemSignalRawShape.size());
    std::vector<uint32_t> tileShape{1, shmemSignalRawShape4};
    uint32_t shmemSignalStride = 8;
    int32_t expectedValue = 8;
    std::vector<int32_t> rawAddr = InitializeShmemSignal(shmemSignalRawShape, shmemSignalOffset, tileShape,
                                                         shmemSignalStride, expectedValue);

    uint32_t shmemSignalAttrOffset = 0;
    auto [data, aicpuCode] = InitializeAicpuCode(shmemSignalRawShape, tileShape, shmemSignalStride, expectedValue,
                                                 shmemSignalAttrOffset);

    auto allocator = std::make_unique<npu::tile_fwk::dynamic::DeviceWorkspaceAllocator>();
    auto task = std::make_unique<npu::tile_fwk::dynamic::DynDeviceTask>(*allocator);
    auto shmemWaitUntil = std::make_unique<npu::tile_fwk::Distributed::ShmemWaitUntilImpl>();
    auto cache = std::make_unique<npu::tile_fwk::Distributed::ShmemWaitUntilCache>();

    auto [buffer, header, funcData] = InitializeTaskData(task.get());

    auto [exprTbl, hcclParam, rawTensorAddrHolder, rawTensorDescHolder, opAttrs, opAtrrOffsets] = ConfigureFuncData(
        header, funcData, reinterpret_cast<uint64_t>(rawAddr.data()));

    task->shmemWaitUntilCacheBackup = cache.get();

    return std::make_tuple(std::move(rawAddr), std::move(data), std::move(allocator), std::move(task),
                           std::move(shmemWaitUntil), std::move(cache), std::move(buffer), std::move(exprTbl),
                           std::move(hcclParam), std::move(rawTensorAddrHolder), std::move(rawTensorDescHolder),
                           std::move(opAttrs), std::move(opAtrrOffsets), std::move(aicpuCode), header, funcData);
}

void PrepareTasks(uint32_t tileOpCount, npu::tile_fwk::Distributed::ShmemWaitUntilCache* cache,
                  const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode,
                  npu::tile_fwk::DynFuncData* funcData, int64_t* hcclContextAddr)
{
    for (uint32_t taskId = 0; taskId < tileOpCount; ++taskId) {
        npu::tile_fwk::Distributed::ShmemWaitUntilImpl::PrepareTask(taskId, aicpuCode, cache->taskArray, taskId,
                                                                    funcData, hcclContextAddr);
    }
    cache->taskCount = tileOpCount;
    npu::tile_fwk::Distributed::ShmemWaitUntilImpl::BuildHashTable(cache, tileOpCount);
}

void RunTests(uint32_t tileOpCount, npu::tile_fwk::Distributed::ShmemWaitUntilImpl* shmemWaitUntil,
              uint32_t parallelIdx = 0)
{
    TaskStat* taskStat{nullptr};
    for (uint32_t taskId = 0; taskId < tileOpCount; ++taskId) {
        shmemWaitUntil->EnqueueOp(taskId, parallelIdx, taskStat);
        shmemWaitUntil->PollCompleted(nullptr, parallelIdx);
    }
}

void TestShmemWaitUntil(const uint32_t tileOpCount)
{
    auto [rawAddr, data, allocator, task, shmemWaitUntil, cache, buffer, exprTbl, hcclParam, rawTensorAddrHolder,
          rawTensorDescHolder, opAttrs, opAtrrOffsets, aicpuCode, header, funcData] = InitializeTestEnvironment();

    PrepareTasks(tileOpCount, cache.get(), aicpuCode, funcData, header->startArgs->commContexts);

    constexpr uint32_t parallelIdx = 0;
    shmemWaitUntil->LoadCache(cache.get(), parallelIdx);

    RunTests(tileOpCount, shmemWaitUntil.get(), parallelIdx);
}

TEST(ShmemWaitUntilTest, BasicFunctionality)
{
    constexpr int32_t tileOpCount = 1;
    TestShmemWaitUntil(tileOpCount);
}
} // namespace
