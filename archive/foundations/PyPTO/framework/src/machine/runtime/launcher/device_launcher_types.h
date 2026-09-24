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
 * \file device_launcher_types.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <numeric>
#include "adapter/api/runtime_define.h"
#include "machine/utils/dynamic/dev_tensor_creator.h"
#include "machine/utils/machine_ws_intf.h"

namespace npu::tile_fwk::dynamic {
struct KernelLaunchInfo {
    RtStream schedStream;
    RtStream ctrlStream;
    RtStream aicoreStream;
    uint32_t blockDim;
    uint32_t aicpuNum;
    RtBinHandle binHandle;
    bool isCaptureActivate;
    KernelLaunchInfo(RtStream sStream, RtStream cStream, RtStream aStream, uint32_t blkDim, uint32_t cpuNum)
        : schedStream(sStream),
          ctrlStream(cStream),
          aicoreStream(aStream),
          blockDim(blkDim),
          aicpuNum(cpuNum),
          binHandle(nullptr),
          isCaptureActivate(false)
    {}
};

class DeviceTensorData {
public:
    DeviceTensorData() = default;
    DeviceTensorData(DataType dtype, void* addr, const std::vector<int64_t>& shape,
                     TileOpFormat format = TileOpFormat::TILEOP_ND)
        : dtype_(dtype), addr_(addr), shape_(shape), format_(format)
    {}
    DeviceTensorData(DataType dtype, uintptr_t addr, const std::vector<int64_t>& shape,
                     TileOpFormat format = TileOpFormat::TILEOP_ND)
        : dtype_(dtype), addr_((void*)addr), shape_(shape), format_(format)
    {}

    void* GetAddr() const { return addr_; }

    const std::vector<int64_t>& GetShape() const { return shape_; }

    DataType GetDataType() const { return dtype_; }

    TileOpFormat Format() const { return format_; }

    int64_t GetDataSize() const
    {
        int64_t numel = std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<>());
        return DataSizeOf(numel, dtype_);
    }

private:
    DataType dtype_;
    void* addr_;
    std::vector<int64_t> shape_;
    TileOpFormat format_;
};

struct DeviceLauncherConfig {
    bool onBoard{true};
    int blockdim{0};
    int aicpuNum{5};
    int64_t dynWorkspaceSize{0};
    int64_t repeatNum{1};
    bool runModel{true};
    std::vector<uint64_t> hcclContext;
    bool controlFlowCache{false};
    bool cpuSeparate{false};
    uint64_t workspaceAddr{0};
    // When true, workspace is allocated per launch by Python (torch.empty on NPU); skip AllocDev in
    // FillKernelMeta. Used by FillDeviceKernelArgs during KernelBinary init; Launch() sets kArgs.workspace.
    bool workspaceAllocByTorch{false};
    bool isCacheOriginShape{true}; // infer cache shape or origin shape

    DeviceLauncherConfig() = default;
    DeviceLauncherConfig(bool onboard, int tblockdim, int taicpunum)
        : onBoard(onboard), blockdim(tblockdim), aicpuNum(taicpunum)
    {}
    DeviceLauncherConfig(int64_t tdynWorkspaceSize) : dynWorkspaceSize(tdynWorkspaceSize) {}
    DeviceLauncherConfig(int64_t tdynWorkspaceSize, int64_t trepeatNum)
        : dynWorkspaceSize(tdynWorkspaceSize), repeatNum(trepeatNum)
    {}
    DeviceLauncherConfig(const std::vector<std::uint64_t>& addrs) : hcclContext(addrs) {}

    static DeviceLauncherConfig CreateConfigWithWorkspaceAddr(uint64_t workspaceAddr)
    {
        DeviceLauncherConfig config;
        config.workspaceAddr = workspaceAddr;
        return config;
    }
};

struct OperatorTensorPara {
    std::vector<DevTensorData> inputTensorParaList;
    std::vector<DevTensorData> outputTensorParaList;
    bool operator==(const OperatorTensorPara& other) const
    {
        if (inputTensorParaList.size() != other.inputTensorParaList.size()) {
            return false;
        }
        for (size_t i = 0; i < inputTensorParaList.size(); i++) {
            if (!inputTensorParaList[i].shape.Equal(other.inputTensorParaList[i].shape)) {
                return false;
            }
        }

        if (outputTensorParaList.size() != other.outputTensorParaList.size()) {
            return false;
        }
        for (size_t i = 0; i < outputTensorParaList.size(); i++) {
            if (!outputTensorParaList[i].shape.Equal(other.outputTensorParaList[i].shape)) {
                return false;
            }
        }
        return true;
    }
};

struct OperatorTensorParaHash {
    std::size_t operator()(const OperatorTensorPara& para) const
    {
        std::size_t hash = 0;
        hash_combine(hash, para.inputTensorParaList.size());
        for (const auto& tensor : para.inputTensorParaList) {
            hash_combine(hash, tensor.shape.dimSize);
            for (int i = 0; i < tensor.shape.dimSize; i++) {
                hash_combine(hash, tensor.shape.dim[i]);
            }
        }
        hash_combine(hash, para.outputTensorParaList.size());
        for (const auto& tensor : para.outputTensorParaList) {
            hash_combine(hash, tensor.shape.dimSize);
            for (int i = 0; i < tensor.shape.dimSize; i++) {
                hash_combine(hash, tensor.shape.dim[i]);
            }
        }
        return hash;
    }

private:
    template <class T>
    static void hash_combine(std::size_t& seed, const T& v)
    {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
};

class CachedOperator {
public:
    static uint8_t** GetWorkspaceDevAddrHolder(CachedOperator* cachedOperator)
    {
        return cachedOperator == nullptr ? nullptr : &cachedOperator->workspaceDevAddr_;
    }
    static uint8_t** GetCfgDataDevAddrHolder(CachedOperator* cachedOperator)
    {
        return cachedOperator == nullptr ? nullptr : &cachedOperator->cfgDataDevAddr_;
    }
    static uint8_t** GetMetaDataDevAddrHolder(CachedOperator* cachedOperator)
    {
        return cachedOperator == nullptr ? nullptr : &cachedOperator->metaDataDevAddr_;
    }
    static void* GetBinHandleHolder(CachedOperator* cachedOperator)
    {
        return cachedOperator == nullptr ? nullptr : &cachedOperator->binHandle_;
    }

    uint8_t* FindCtrlFlowCache(const std::vector<DeviceTensorData>& inputList,
                               const std::vector<DeviceTensorData>& outputList)
    {
        auto it = devCtrlFlowCacheMap_.find(BuildOperatorTensorPara(inputList, outputList));
        if (it != devCtrlFlowCacheMap_.end()) {
            return it->second;
        }
        return nullptr;
    }

    void InsertCtrlFlowCache(const std::vector<DeviceTensorData>& inputList,
                             const std::vector<DeviceTensorData>& outputList, uint8_t* cache)
    {
        devCtrlFlowCacheMap_[BuildOperatorTensorPara(inputList, outputList)] = cache;
    }

private:
    OperatorTensorPara BuildOperatorTensorPara(const std::vector<DeviceTensorData>& inputList,
                                               const std::vector<DeviceTensorData>& outputList)
    {
        OperatorTensorPara para;
        for (const auto& input : inputList) {
            para.inputTensorParaList.emplace_back(DevAscendTensorDataCreator::Create(0, input.GetShape()));
        }

        for (const auto& output : outputList) {
            para.outputTensorParaList.emplace_back(DevAscendTensorDataCreator::Create(0, output.GetShape()));
        }
        return para;
    }

private:
    uint8_t* workspaceDevAddr_{nullptr};
    uint8_t* cfgDataDevAddr_{nullptr};
    uint8_t* metaDataDevAddr_{nullptr};
    void* binHandle_{nullptr};
    std::unordered_map<OperatorTensorPara, uint8_t*, OperatorTensorParaHash> devCtrlFlowCacheMap_;
};
} // namespace npu::tile_fwk::dynamic
