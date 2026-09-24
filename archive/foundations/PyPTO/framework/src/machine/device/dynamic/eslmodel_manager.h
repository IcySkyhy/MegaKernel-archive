/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdio>
#include <cstdint>
#include <vector>
#include <dirent.h>
#include <regex>
#include <algorithm>
#include <cstring>

#include "securec.h"
#include "tilefwk/aicpu_common.h"
#include "tilefwk/aikernel_data.h"
#include "machine/device/dynamic/aicore_constants.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/device/dynamic/device_sche_context.h"
#include "machine/utils/dynamic/device_task.h"
#include "machine/utils/device_log.h"
#include "machine/device/dump/aicore_dump.h"
#ifndef __DEVICE__
#include "adapter/api/runtime_api.h"
#endif
#include "interface/configs/config_manager_ng.h"
#include "interface/utils/string_utils.h"
#include "tilefwk/data_type.h"

namespace npu::tile_fwk::dynamic {

inline std::vector<uint8_t> LoadFile(const std::string& filePath)
{
    std::vector<uint8_t> binary;
    FILE* file = fopen(filePath.c_str(), "rb");
    if (file != nullptr) {
        fseek(file, 0, SEEK_END);
        int size = ftell(file);
        binary.resize(size);
        fseek(file, 0, SEEK_SET);
        size_t readSize = fread(binary.data(), 1, size, file);
        if (readSize != static_cast<size_t>(size)) {
            binary.clear();
        }
        fclose(file);
    }
    return binary;
}

class EslModelReplayManager {
public:
    void Init(SchduleContext* scheCtx)
    {
        scheCtx_ = scheCtx;
        replayFinished_ = false;
        ParseReplayConfigFromArgs();
    }

    inline bool ReplayMatch(uint64_t newTask)
    {
        if (!enable_)
            return true;
        if (replayFinished_)
            return false;

        auto devTaskCtx = scheCtx_->GetCurSchDevTaskCtx();
        auto dyntask = reinterpret_cast<DynDeviceTask*>(devTaskCtx->GetDeviceTask());
        auto dynFuncHeader = dyntask->GetDynFuncDataList();

        uint64_t seqNo = dynFuncHeader->seqNo;
        uint32_t funcId = FuncID(static_cast<uint32_t>(newTask));
        uint32_t opIdx = TaskID(static_cast<uint32_t>(newTask));

        bool matchReplay = (seqNo == seqNo_ && funcId == funcId_ && opIdx == opIdx_);

        if (matchReplay) {
            LoadTensor();
            replayFinished_ = true;
        }

        return matchReplay;
    }

private:
    inline void ParseReplayConfigFromArgs()
    {
        bool hasTaskConfig = false;
        bool hasPathConfig = false;
        auto args = ConfigManagerNg::GetGlobalConfig<std::vector<std::string>>("simulation.args");
        for (const auto& arg : args) {
            if (npu::tile_fwk::StringUtils::StartsWith(arg, "Model.replayTask:")) {
                std::string taskStr = arg.substr(strlen("Model.replayTask:"));
                auto parts = npu::tile_fwk::StringUtils::Split(taskStr, "-");
                std::vector<uint32_t> values;
                for (const auto& part : parts) {
                    values.push_back(std::stoul(part));
                }
                if (values.size() >= 3) {
                    seqNo_ = values[0];
                    funcId_ = values[1];
                    opIdx_ = values[2];
                    hasTaskConfig = true;
                }
            } else if (npu::tile_fwk::StringUtils::StartsWith(arg, "Model.replayPath:")) {
                replayPath_ = arg.substr(strlen("Model.replayPath:"));
                hasPathConfig = true;
            }
        }
        enable_ = hasTaskConfig && hasPathConfig;
    }

    inline std::vector<std::pair<int, std::string>> GetMatchedInputFiles()
    {
        uint32_t targetTask = MakeTaskID(funcId_, opIdx_);
        std::string patternStr = std::to_string(targetTask) + "_" + std::to_string(seqNo_) +
                                 "_.*_input([0-9]+)\\.tdump";
        std::regex filePattern(patternStr);

        DIR* dir = opendir(replayPath_.c_str());
        if (dir == nullptr)
            return {};

        struct dirent* entry;
        std::vector<std::pair<int, std::string>> matchedFiles;

        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            std::smatch match;
            if (!std::regex_match(filename, match, filePattern))
                continue;
            int tensorIdx = std::stoi(match[1].str());
            matchedFiles.push_back({tensorIdx, filename});
        }
        closedir(dir);

        std::sort(matchedFiles.begin(), matchedFiles.end());
        return matchedFiles;
    }

    inline void LoadTensor()
    {
        auto matchedFiles = GetMatchedInputFiles();

        auto devTaskCtx = scheCtx_->GetCurSchDevTaskCtx();
        auto dyntask = reinterpret_cast<DynDeviceTask*>(devTaskCtx->GetDeviceTask());

        auto& funcCache = dyntask->dynFuncDataCacheList[funcId_];
        auto func = funcCache.devFunc;
        auto dupData = funcCache.duppedData;
        int32_t inputTensorNum = func->GetOperationIOperandSize(opIdx_);

        for (int32_t tensorIdx = 0;
             tensorIdx < inputTensorNum && static_cast<uint32_t>(tensorIdx) < matchedFiles.size(); tensorIdx++) {
            auto* operand = func->GetOperationIOperand(opIdx_, tensorIdx);
            uint64_t rawIdx = operand->rawIndex;
            auto* rawTensor = func->GetRawTensor(rawIdx);
            if (rawTensor == nullptr)
                continue;

            uint64_t tensorAddr = GetTensorAddress(dupData, rawTensor);
            if (tensorAddr == 0)
                continue;

            std::string filepath = replayPath_ + "/" + matchedFiles[tensorIdx].second;
            auto fileData = LoadFile(filepath);
            if (fileData.size() < sizeof(DumpTensorInfo))
                continue;

            DumpTensorInfo dumpHeader;
            DevMemcpyS(&dumpHeader, sizeof(DumpTensorInfo), fileData.data(), sizeof(DumpTensorInfo));

            uint64_t stride[DEV_SHAPE_DIM_MAX];
            if (dumpHeader.dims > 0) {
                stride[dumpHeader.dims - 1] = 1;
                for (int32_t k = dumpHeader.dims - 2; k >= 0; k--) {
                    stride[k] = stride[k + 1] * dumpHeader.rawShape[k + 1];
                }
            }

            uint32_t bufferOffset = sizeof(DumpTensorInfo);
            LoadTensorFromData(dumpHeader, stride, 0, tensorAddr, fileData.data(), bufferOffset);
        }
    }

    inline uint64_t GetTensorAddress(DevAscendFunctionDuppedData* dupData, DevAscendRawTensor* rawTensor)
    {
        if (rawTensor->ioProperty == DevIOProperty::ROOT_INCAST) {
            auto addrDesc = dupData->GetIncastAddress(rawTensor->ioIndex);
            return addrDesc.addr;
        } else if (rawTensor->ioProperty == DevIOProperty::ROOT_OUTCAST) {
            auto addrDesc = dupData->GetOutcastAddress(rawTensor->ioIndex);
            return addrDesc.addr;
        } else {
            return dupData->GetRuntimeWorkspace() + rawTensor->addrOffset;
        }
    }

    void LoadTensorFromData(const DumpTensorInfo& info, const uint64_t stride[], uint32_t idx, uint64_t tensorAddr,
                            const uint8_t* fileData, uint32_t& bufferOffset)
    {
        uint32_t dataByte = BytesOf(static_cast<DataType>(info.dataType));

        if (idx != static_cast<uint32_t>(info.dims) - 1) {
            for (uint64_t k = 0; k < info.shape[idx]; k++) {
                uint64_t newAddr = tensorAddr + info.offset[idx] * dataByte * stride[idx] + k * stride[idx] * dataByte;
                LoadTensorFromData(info, stride, idx + 1, newAddr, fileData, bufferOffset);
            }
        } else {
            uint32_t dataSize = info.shape[idx] * dataByte;

#ifndef __DEVICE__
            uint64_t newAddr = tensorAddr + info.offset[idx] * dataByte;
            RuntimeMemcpyDirect(reinterpret_cast<void*>(newAddr), dataSize, fileData + bufferOffset, dataSize,
                                RtMemcpyKind::HOST_TO_DEVICE);
#endif

            bufferOffset += dataSize;
        }
    }

    SchduleContext* scheCtx_{nullptr};
    bool replayFinished_{false};
    bool enable_{false};
    uint64_t seqNo_{0};
    uint32_t funcId_{0};
    uint32_t opIdx_{0};
    std::string replayPath_;
};

} // namespace npu::tile_fwk::dynamic
