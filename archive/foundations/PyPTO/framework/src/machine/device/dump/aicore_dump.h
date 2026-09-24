/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aicore_dump.h
 * \brief
 */

#ifndef AICORE_DUMP_H
#define AICORE_DUMP_H
#include <string>
#include <sstream>
#include "securec.h"
#include "machine/utils/dynamic/dev_start_args.h"
#include "machine/utils/device_log.h"
#include "tilefwk/aikernel_data.h"

namespace npu::tile_fwk::dynamic {
constexpr uint64_t DEV_DUMP_DATA_SIZE = 2 * 1024 * 1024;
constexpr const uint8_t MAIN_BLOCK_SIZE = 2;
constexpr uint64_t DUMP_RAW_TENSOR_ADDR_MASK = (1UL << 63) - 1;
using IDE_SESSION = void*;
enum IdeErrorT {};
extern "C" {
__attribute__((weak)) IDE_SESSION IdeDumpStart(const char* privInfo);
__attribute__((weak)) IdeErrorT IdeDumpData(IDE_SESSION session, const struct IdeDumpChunk* dumpChunk);
__attribute__((weak)) IdeErrorT IdeDumpEnd(IDE_SESSION session);
};

struct IdeDumpChunk {
    char* fileName;           /**< absolute path */
    unsigned char* dataBuf;   /**< Buffer of input data */
    unsigned int bufLen;      /**< Buffer Length of input data */
    unsigned int isLastChunk; /**< isLastChunk data   0:Not last; 1：Is last */
    long long offset;         /**< The offset of file writing, -1 mean is written in append form */
    bool flag;                /**< flag */
};

struct LoopVarInfo {
    char name[64];
    int32_t exprIdx;
    int32_t value;
};

struct DumpTensorInfo {
    uint8_t version;
    uint8_t rsrv_01;
    uint16_t headSize;
    uint32_t rsrv_02;
    uint32_t opId;
    uint32_t funcId;
    uint32_t taskId;
    uint32_t callopMagic;
    int32_t coreId;
    int32_t dataType;
    int32_t rawMagic;
    int32_t dims;
    int64_t execStart;
    int64_t execEnd;
    uint64_t rootHash;
    uint64_t funcHash;
    uint64_t timeStamp;
    uint64_t shape[DEV_SHAPE_DIM_MAX];
    uint64_t offset[DEV_SHAPE_DIM_MAX];
    uint64_t rawShape[DEV_SHAPE_DIM_MAX];
    uint64_t tensorAddr{0};
    uint64_t loopVarCount{0};
    LoopVarInfo loopVarInfos[8];
};

struct DumpTensorData {
    uint64_t datasize{4};
    uint64_t data;
    std::uint8_t dataByte;
    uint64_t dataOffset{0};
    bool skipCopy{false};

    void TraverseAllAhapeIndexCombinations(const uint64_t shape[], const uint64_t stride[], const uint64_t offset[],
                                           uint32_t idx, uint32_t dims, uint64_t tensorAddr)
    {
        if (idx != dims - 1) {
            for (uint64_t k = 0; k < shape[idx]; k++) {
                auto newAddr = tensorAddr + offset[idx] * dataByte * stride[idx] + k * stride[idx] * dataByte;
                TraverseAllAhapeIndexCombinations(shape, stride, offset, idx + 1, dims, newAddr);
            }
        } else {
            DevMemcpyS(reinterpret_cast<uint8_t*>(data) + dataOffset, shape[idx] * dataByte,
                       reinterpret_cast<const uint8_t*>(tensorAddr) + offset[idx] * dataByte, shape[idx] * dataByte);
            dataOffset = dataOffset + shape[idx] * dataByte;
        }
    }

    DumpTensorData(DumpTensorInfo info, uint64_t dataAddr)
    {
        DevMemcpyS(reinterpret_cast<uint8_t*>(dataAddr), sizeof(DumpTensorInfo),
                   reinterpret_cast<const uint8_t*>(&info), sizeof(DumpTensorInfo));
        dataOffset = sizeof(DumpTensorInfo);
        data = dataAddr;
        dataByte = BytesOf(static_cast<DataType>(info.dataType));
        datasize = dataByte;
        for (int32_t i = 0; i < info.dims; i++) {
            datasize *= info.shape[i];
        }
        datasize += sizeof(DumpTensorInfo);
        if (datasize > DEV_DUMP_DATA_SIZE) {
            return;
        }

        if (info.tensorAddr == 0) {
            DEV_ERROR(TensorMetaErr::INCAST_ADDRESS_NULL,
                      "#sche.dump.addr: skip memcpy, tensorAddr=0 rawMagic=%d dims=%d "
                      "shape=[%lu,%lu,%lu,%lu] offset=[%lu,%lu,%lu,%lu] rawShape=[%lu,%lu,%lu,%lu]",
                      info.rawMagic, info.dims, info.shape[0], info.shape[1], info.shape[2], info.shape[3],
                      info.offset[0], info.offset[1], info.offset[2], info.offset[3], info.rawShape[0],
                      info.rawShape[1], info.rawShape[2], info.rawShape[3]);
            skipCopy = true;
            datasize = 0;
            return;
        }

        DEV_DEBUG("#sche.dump.copy: tensorAddr=0x%lx rawMagic=%d dims=%d dataByte=%u copyBytes=%lu "
                  "shape=[%lu,%lu,%lu] offset=[%lu,%lu,%lu] rawShape=[%lu,%lu,%lu]",
                  info.tensorAddr, info.rawMagic, info.dims, dataByte, datasize - sizeof(DumpTensorInfo), info.shape[0],
                  info.shape[1], info.shape[2], info.offset[0], info.offset[1], info.offset[2], info.rawShape[0],
                  info.rawShape[1], info.rawShape[2]);

        uint64_t stride[DEV_SHAPE_DIM_MAX];
        stride[info.dims - 1] = 1;

        for (int32_t k = info.dims - 2; k >= 0; k--) {
            stride[k] = stride[k + 1] * info.rawShape[k + 1];
        }

        // Check if the tensor is contiguous
        bool is_contiguous = true;
        for (int32_t i = 0; i < info.dims; i++) {
            if (info.shape[i] != info.rawShape[i] || info.offset[i] != 0) {
                is_contiguous = false;
                break;
            }
        }
        if (is_contiguous) {
            // Copy the tensor data in one shot
            uint64_t copy_size = datasize - sizeof(DumpTensorInfo);
            DevMemcpyS(reinterpret_cast<uint8_t*>(data) + dataOffset, copy_size,
                       reinterpret_cast<const uint8_t*>(info.tensorAddr), copy_size);
            dataOffset += copy_size;
        } else {
            TraverseAllAhapeIndexCombinations(info.shape, stride, info.offset, 0, info.dims, info.tensorAddr);
        }
    }

    int GetDumpSize() const
    {
        DEV_DEBUG("TensorInfoSize=%zu, TensorDataSize=%lu.", sizeof(DumpTensorInfo), datasize);
        return datasize;
    }
};

class AicoreDump {
public:
    AicoreDump() {};
    ~AicoreDump()
    {
        if (ideSession_ == nullptr) {
            return;
        }
        DEV_DEBUG("Now close the tensor dump.");
        int m = IdeDumpEnd(ideSession_);
        if (m != 0) {
            DEV_WARN("Close ideSession failed, state=%d.", m);
        }
    }
    uint64_t dataSize_{0};
    void Init(DevStartArgs* startArgs, int schedIdx)
    {
        auto devProg = startArgs->devProg;
        devProg_ = devProg;
        auto deviceArgs = &devProg->devArgs;
        SetHostPid(deviceArgs->hostPid);
        if (enableDump_) {
            deviceId_ = deviceArgs->deviceId;
            uint64_t baseAddr = startArgs->contextWorkspaceAddr;
            baseAddr += devProg->memBudget.aicoreSpilled.Total() + devProg->memBudget.tensor.Total() +
                        devProg->memBudget.debug.dumpTensor;

            dataAddr = baseAddr + schedIdx * DEV_DUMP_DATA_SIZE;
            DEV_DEBUG("DataAddr=%#lx.", dataAddr);
            const std::string privateInfo = "127.0.0.1:22118;" + std::to_string(deviceId_) + ";" +
                                            std::to_string(hostPid_);
            ideSession_ = IdeDumpStart(privateInfo.c_str());
            DEV_DEBUG("Pid=%d, deviceId=%u, privateInfo=%s.", (int)hostPid_, deviceId_, privateInfo.c_str());
        }
    }

    void DoDump(DeviceTask* devTask, std::string iOinfo, int32_t taskId, int32_t coreId, int64_t execStart = 0,
                int64_t execEnd = 0)
    {
        DumpInit(taskId, coreId, execStart, execEnd);
        DoDump(devTask, iOinfo);
    }

    void DumpInit(int32_t taskId, int32_t coreId, int64_t execStart = 0, int64_t execEnd = 0)
    {
        taskId_ = taskId;
        coreId_ = coreId;
        execStart_ = execStart;
        execEnd_ = execEnd;
        timeStamp_ = GetTimeMonotonic();
    }

    void SetHostPid(uint32_t hostPid)
    {
        hostPid_ = hostPid;
        DEV_DEBUG("HostPid=%u.", hostPid_);
        enableDump_ = (hostPid_ != 0);
    }
    inline bool IsEnableDump() const { return enableDump_; }

    inline bool DumpData(std::string& fileName, unsigned char* dataBuf, uint64_t dataSize, bool& isLast) const
    {
        IdeDumpChunk ideDumpChunk = {
            .fileName = const_cast<char*>(fileName.c_str()),
            .dataBuf = dataBuf,
            .bufLen = static_cast<unsigned int>(dataSize),
            .isLastChunk = isLast ? 1U : 0,
            .offset = -1,
            .flag = 0,
        };

        DEV_DEBUG("Start ideDump tensor data.");
        const int ideState = IdeDumpData(ideSession_, &ideDumpChunk);
        DEV_DEBUG("Finish ideDump. IdeState=%d.", ideState);
        return ideState == 0;
    }

    void Dump(DumpTensorInfo dumpTensorInfo, std::string& fileName, bool isLast)
    {
        DumpTensorData dumpTensorData(dumpTensorInfo, dataAddr);
        dataSize_ = dumpTensorData.GetDumpSize();
        if (dumpTensorData.skipCopy || dataSize_ == 0) {
            DEV_WARN("#sche.dump.data: Skip dump due to invalid tensor address, file=%s.", fileName.c_str());
            return;
        }
        if (dataSize_ > DEV_DUMP_DATA_SIZE) {
            DEV_WARN("Tensor dataSize=%lu is larger than dumpSize=%lu.", dataSize_, DEV_DUMP_DATA_SIZE);
            return;
        }
        bool ret = DumpData(fileName, reinterpret_cast<uint8_t*>(dumpTensorData.data), dataSize_, isLast);
        if (!ret) {
            DEV_WARN("#sche.dump.data: Dump Tensor data not successful.");
            return;
        }
    }

    void GetTensorShapeInfo(npu::tile_fwk::TensorInfo* info, std::string& shapeInfo)
    {
        std::ostringstream oss;
        for (uint32_t i = 0; i < info->dims; i++) {
            oss << "_" << std::to_string(info->shape[i]);
        }
        shapeInfo = oss.str();
    }

    static inline DynFuncData* GetDynFuncData(DynDeviceTask* dyntask, uint64_t taskId)
    {
        auto* head = reinterpret_cast<DynFuncHeader*>(dyntask->GetDynFuncDataList());
        auto* funcDataList = reinterpret_cast<DynFuncData*>(head + 1);
        return &funcDataList[FuncID(taskId)];
    }

    // Same address resolution path as AICore GetTensorAddr / schema dump.
    static inline uint64_t GetRawTensorAddrFromDyn(DynFuncData* dynFuncData, uint64_t rawIndex)
    {
        auto* desc = &dynFuncData->rawTensorDesc[rawIndex];
        if (desc->location == RAW_TENSOR_LOCATION_LOCAL) {
            return dynFuncData->workspaceAddr + desc->offsetOrIndex;
        }
        return dynFuncData->rawTensorAddr[desc->offsetOrIndex] & DUMP_RAW_TENSOR_ADDR_MASK;
    }

    DumpTensorInfo GetDumpTensorInfo(DynDeviceTask* dyntask, std::string iOinfo, int32_t tensorIdx)
    {
        auto opIdx = TaskID(taskId_);
        auto func = dyntask->dynFuncDataCacheList[FuncID(taskId_)].devFunc;
        auto dupData = dyntask->dynFuncDataCacheList[FuncID(taskId_)].duppedData;
        auto* dynFuncData = GetDynFuncData(dyntask, taskId_);
        DumpTensorInfo dumpTensorInfo{};

        auto setDumpTensorInfo = [&](DevAscendRawTensor* rawTensor, int32_t idx, bool isIOperand, uint64_t rawIdx) {
            uint32_t dimSize = rawTensor->GetDim();
            int cceIndex = func->GetOperationAttrCalleeIndex(opIdx);
            if (devProg_->devArgs.enableVFFusion) {
                cceIndex = (func->GetOperationAttrCalleeIndex(opIdx) + 1) / MAIN_BLOCK_SIZE;
            }
            dumpTensorInfo.headSize = sizeof(DumpTensorInfo);
            dumpTensorInfo.version = 0x1;
            dumpTensorInfo.rsrv_01 = 0;
            dumpTensorInfo.rsrv_02 = 0;
            dumpTensorInfo.funcId = FuncID(taskId_);
            dumpTensorInfo.opId = opIdx;
            dumpTensorInfo.callopMagic = func->GetOperationDebugOpmagic(opIdx);
            dumpTensorInfo.taskId = taskId_;
            dumpTensorInfo.rawMagic = rawTensor->rawMagic;
            dumpTensorInfo.coreId = coreId_;
            dumpTensorInfo.dataType = static_cast<uint32_t>(rawTensor->dataType);
            dumpTensorInfo.dims = dimSize;
            dumpTensorInfo.execStart = execStart_;
            dumpTensorInfo.execEnd = execEnd_;
            dumpTensorInfo.rootHash = func->rootHash;
            dumpTensorInfo.funcHash = dyntask->cceBinary[cceIndex].funcHash;
            auto& operandInfo = func->GetOperationOperandInfo(opIdx, idx, isIOperand);
            GetTensorOffsetAndShape<false>(
                func, dumpTensorInfo.offset, dumpTensorInfo.shape, &(dupData->GetExpression(0)), dimSize, opIdx,
                operandInfo.staticOffsetAttrBeginIndex, operandInfo.staticShapeAttrBeginIndex);

            auto* desc = &dynFuncData->rawTensorDesc[rawIdx];
            // Prefer DynFuncData path (same as AICore GetTensorAddr). Dup RuntimeWorkspace can be 0.
            dumpTensorInfo.tensorAddr = GetRawTensorAddrFromDyn(dynFuncData, rawIdx);
            if (dumpTensorInfo.tensorAddr == 0) {
                DEV_ERROR(TensorMetaErr::INCAST_ADDRESS_NULL,
                          "#sche.dump.addr: tensorAddr=0 rawIdx=%lu rawMagic=%d ioProp=%u "
                          "addrOffset=0x%lx loc=%u offOrIdx=0x%x ws=0x%lx rtWs=0x%lx",
                          rawIdx, rawTensor->rawMagic, static_cast<uint32_t>(rawTensor->ioProperty),
                          rawTensor->addrOffset, desc->location, desc->offsetOrIndex, dynFuncData->workspaceAddr,
                          static_cast<uint64_t>(dupData->GetRuntimeWorkspace()));
            } else {
                DEV_DEBUG("#sche.dump.addr: rawIdx=%lu rawMagic=%d ioProp=%u loc=%u offOrIdx=0x%x addr=0x%lx", rawIdx,
                          rawTensor->rawMagic, static_cast<uint32_t>(rawTensor->ioProperty), desc->location,
                          desc->offsetOrIndex, dumpTensorInfo.tensorAddr);
            }
            for (uint32_t i = 0; i < dimSize; i++) {
                dumpTensorInfo.rawShape[i] = rawTensor->shape.At(i, dupData->GetExpressionAddr());
            }
            dumpTensorInfo.timeStamp = timeStamp_;
        };
        if (iOinfo == "input") {
            uint64_t rawIdx = func->GetOperationIOperand(opIdx, tensorIdx)->rawIndex;
            auto* rawTensor = func->GetRawTensor(rawIdx);
            setDumpTensorInfo(rawTensor, tensorIdx, true, rawIdx);
        } else {
            uint64_t rawIdx = func->GetOperationOOperand(opIdx, tensorIdx)->rawIndex;
            auto* rawTensor = func->GetRawTensor(rawIdx);
            setDumpTensorInfo(rawTensor, tensorIdx, false, rawIdx);
        }

        FillLoopVarInfo(dumpTensorInfo, dupData);

        return dumpTensorInfo;
    }

    void DoDump(DeviceTask* devTask, std::string iOinfo)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(devTask);
        auto func = dyntask->dynFuncDataCacheList[FuncID(taskId_)].devFunc;
        auto opIdx = TaskID(taskId_);
        int32_t tensorNum = (iOinfo == "input") ? func->GetOperationIOperandSize(opIdx) :
                                                  func->GetOperationOOperandSize(opIdx);
        if (!IdeDumpStart || !IdeDumpData || !IdeDumpEnd) {
            DEV_WARN("#sche.dump.prep: IdeDumpStart, IdeDumpData, IdeDumpEnd function not found.");
            return;
        }

        std::string dumpPath = "output/dump_tensor_" + std::to_string(hostPid_) + "/device_" +
                               std::to_string(deviceId_) + "/";
        if (ideSession_ == nullptr) {
            DEV_WARN("Created ideSession failed.");
            return;
        }
        auto seqNo = dyntask->GetDynFuncDataList()->seqNo;
        for (int i = 0; i < tensorNum; i++) {
            auto info = GetDumpTensorInfo(dyntask, iOinfo, i);
            bool isLast = (i == tensorNum - 1) ? true : false;
            std::string tensorInfos = std::to_string(taskId_) + "_" + std::to_string(seqNo) + "_" +
                                      std::to_string(info.callopMagic) + "_" + std::to_string(info.rootHash) + "_" +
                                      std::to_string(info.funcHash) + "_" + std::to_string(info.rawMagic) + "_" +
                                      std::to_string(timeStamp_) + "_" +
                                      DataType2CCEStr(static_cast<DataType>(info.dataType)) + "_" + iOinfo +
                                      std::to_string(i) + ".tdump";
            std::string fileName = dumpPath + tensorInfos;
            Dump(info, fileName, isLast);
        }
    }

private:
    // CF unroll 会生成 loop_idx_s2_idx_0/1/2；verify LOOP_INFO 只记录原始循环名。
    static bool IsRemainderUnrollLoopName(const std::string& name, const DevAscendProgram* devProg)
    {
        const auto pos = name.find_last_of('_');
        if (pos == std::string::npos || pos + 1 >= name.size()) {
            return false;
        }
        for (size_t i = pos + 1; i < name.size(); ++i) {
            if (name[i] < '0' || name[i] > '9') {
                return false;
            }
        }
        const std::string prefix = name.substr(0, pos);
        for (const auto& symbol : devProg->symbolTable) {
            std::string other(symbol.name.begin(), symbol.name.end());
            other = other.c_str();
            if (other == prefix) {
                return true;
            }
        }
        return false;
    }

    void FillLoopVarInfo(DumpTensorInfo& dumpTensorInfo, DevAscendFunctionDuppedData* dupData)
    {
        dumpTensorInfo.loopVarCount = 0;
        if (devProg_ == nullptr || dupData == nullptr) {
            return;
        }
        const uint64_t* exprList = dupData->GetExpressionAddr();
        const uint64_t exprSize = dupData->GetExpressionSize();
        if (exprList == nullptr || exprSize <= 1) {
            return;
        }

        for (const auto& symbol : devProg_->symbolTable) {
            if (dumpTensorInfo.loopVarCount >= 8) {
                break;
            }

            std::string name(symbol.name.begin(), symbol.name.end());
            name = name.c_str();
            if (name.find("loop_idx_") == std::string::npos) {
                continue;
            }
            if (IsRemainderUnrollLoopName(name, devProg_)) {
                continue;
            }
            const uint64_t exprListIdx = symbol.index + 1;
            if (exprListIdx >= exprSize) {
                continue;
            }
            auto& loopVarInfo = dumpTensorInfo.loopVarInfos[dumpTensorInfo.loopVarCount];
            memset_s(loopVarInfo.name, sizeof(loopVarInfo.name), 0, sizeof(loopVarInfo.name));
            strncpy_s(loopVarInfo.name, sizeof(loopVarInfo.name), name.c_str(), sizeof(loopVarInfo.name) - 1);
            loopVarInfo.exprIdx = static_cast<int32_t>(exprListIdx);
            loopVarInfo.value = static_cast<int32_t>(exprList[exprListIdx]);
            dumpTensorInfo.loopVarCount++;
        }
    }

    int32_t taskId_{0};
    int32_t coreId_{0};
    int64_t execStart_{0};
    int64_t execEnd_{0};
    uint32_t deviceId_{0};
    uint32_t hostPid_{0};
    uint64_t timeStamp_{0};
    uint64_t dataAddr;
    bool enableDump_{false};
    IDE_SESSION ideSession_{nullptr};
    DevAscendProgram* devProg_{nullptr};
};
} // namespace npu::tile_fwk::dynamic
#endif
