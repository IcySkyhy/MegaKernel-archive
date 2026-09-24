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
 * \file distributed_op_test_common.h
 * \brief
 */

#ifndef DISTRIBUTED_OP_TEST_COMMON_H
#define DISTRIBUTED_OP_TEST_COMMON_H

#include <vector>
#include <string>
#include <algorithm>
#include <array>
#include "test_common.h"
#include "distributed_op_test_suite.h"
#include "tileop/distributed/comm_context.h"
#include "tilefwk/tilefwk_op.h"
#include "test_dev_func_runner.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
namespace Distributed {

template <size_t N, typename SrcT = int64_t, typename DstT = int32_t>
std::array<DstT, N> GetParams(const std::string& filePath)
{
    std::vector<SrcT> srcParams(N);
    readInput(filePath, srcParams);
    std::array<DstT, N> dstParams;
    std::transform(srcParams.begin(), srcParams.begin() + N, dstParams.begin(),
                   [](SrcT v) { return static_cast<DstT>(v); });
    return dstParams;
}

inline DataType GetDataTypeNum(const int64_t typeNum)
{
    if ((typeNum < 0) || (typeNum >= static_cast<int64_t>(DataType::DT_BOTTOM))) {
        DISTRIBUTED_LOGE(DistributedErrorCode::UNKNOW_ERROR, "Invalid type code: %ld (Valid range: [0-%ld])", typeNum,
                         static_cast<int64_t>(DataType::DT_BOTTOM));
        return DataType::DT_BOTTOM;
    }
    return static_cast<DataType>(typeNum);
}

template <typename T>
std::vector<T> ReadToVector(const std::string& filePath, const std::vector<int64_t>& shape)
{
    auto mul = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
    std::vector<T> result(mul, 0);
    std::string xPath = filePath;
    readInput<T>(xPath, result);
    return result;
}

template <typename T, typename PtrType>
bool DoCompare(const std::string& goldenFilename, const uint64_t outSize, const size_t dTypeSize,
               const PtrType& outPtrs, const OpTestParam& testParam, float threshold)
{
    std::vector<T> res(outSize);
    std::vector<T> resGolden(outSize);
    // 统一处理指针或指针数组
    if constexpr (std::is_same_v<PtrType, uint8_t*>) {
        CopyFromTensor(reinterpret_cast<uint8_t*>(res.data()), outPtrs, outSize * dTypeSize);
    } else {
        for (int32_t i = 0; i < testParam.rankSize; ++i) {
            const size_t chunkSize = outSize / testParam.rankSize;
            const size_t offset = i * chunkSize;
            CopyFromTensor(reinterpret_cast<uint8_t*>(res.data() + offset), outPtrs[i], chunkSize * dTypeSize);
        }
    }
    // 读取Golden数据并比较
    readInput<T>(goldenFilename + std::to_string(testParam.rankId) + ".bin", resGolden);
    return resultCmp<T>(resGolden, res, threshold);
}

template <typename PtrType>
bool CompareWithGolden(const DataType dType, const std::string& goldenFilename, const uint64_t outSize,
                       const PtrType& outPtrs, const OpTestParam& testParam, float threshold = 0.001f)
{
    CHECK((std::is_same_v<PtrType, uint8_t*>) || (std::is_same_v<PtrType, std::vector<uint8_t*>>))
        << "PtrType must be either uint8_t* or std::vector<uint8_t*>";

    const size_t dTypeSize = BytesOf(dType);
    bool result = false;

    switch (dType) {
        case DataType::DT_FP32:
            result = DoCompare<float>(goldenFilename, outSize, dTypeSize, outPtrs, testParam, threshold);
            break;
        case DataType::DT_FP16:
            result = DoCompare<npu::tile_fwk::float16>(goldenFilename, outSize, dTypeSize, outPtrs, testParam,
                                                       threshold);
            break;
        case DataType::DT_BF16:
            result = DoCompare<npu::tile_fwk::bfloat16>(goldenFilename, outSize, dTypeSize, outPtrs, testParam,
                                                        threshold);
            break;
        case DataType::DT_INT32:
            result = DoCompare<int32_t>(goldenFilename, outSize, dTypeSize, outPtrs, testParam, threshold);
            break;
        default:
            DISTRIBUTED_LOGE(DistributedErrorCode::UNKNOW_ERROR, "Unsupported dType: %lu",
                             static_cast<uint64_t>(dType));
            break;
    }
    return result;
}

inline void RunTest()
{
    DeviceLauncherConfig config;
    config.runModel = false;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);
}

enum class WinType : uint32_t { WIN_EXP, WIN_OUT, WIN_IN };

class HcclWin {
public:
    HcclWin(uint64_t addr)
    {
        RuntimeMemcpy(&param_, sizeof(param_), (uint8_t*)addr, sizeof(param_), RtMemcpyKind::DEVICE_TO_HOST);
    }

    template <typename T>
    std::vector<T> GetWinValue(WinType winType, size_t count = 0UL, size_t offset = 0UL)
    {
        auto [devAddr, winSize] = GetWinAddrAndSize(winType);
        CHECK((devAddr != 0) && (winSize != 0)) << "devAddr and winSize must not be 0";
        auto maxDataCnt = winSize / sizeof(T);
        offset = offset % maxDataCnt;
        if ((count == 0UL) || (count > maxDataCnt - offset)) {
            count = maxDataCnt - offset;
        }
        std::vector<T> result(count, 0);
        RuntimeMemcpy(result.data(), count * sizeof(T), (uint8_t*)devAddr + offset * sizeof(T), count * sizeof(T),
                      RtMemcpyKind::DEVICE_TO_HOST);
        return result;
    }

private:
    std::tuple<uint64_t, uint64_t> GetWinAddrAndSize(WinType winType)
    {
        uint64_t devAddr = 0UL;
        uint64_t winSize = 0UL;
        switch (winType) {
            case WinType::WIN_EXP:
                devAddr = param_.winAddr[param_.statusIndex + param_.rankId];
                winSize = param_.winStatusSize;
                break;
            case WinType::WIN_OUT:
                devAddr = param_.winAddr[param_.debugIndex + param_.rankId];
                winSize = param_.winDebugSize;
                break;
            case WinType::WIN_IN:
                devAddr = param_.winAddr[param_.rankId];
                winSize = param_.winDataSize;
                break;
            default:
                break;
        }
        return std::tie(devAddr, winSize);
    }

private:
    TileOp::CommContext param_;
};

std::vector<uint64_t> GetHcclContext(const std::vector<std::string>& groupNames);
} // namespace Distributed
} // namespace npu::tile_fwk

#endif // DISTRIBUTED_OP_TEST_COMMON_H
