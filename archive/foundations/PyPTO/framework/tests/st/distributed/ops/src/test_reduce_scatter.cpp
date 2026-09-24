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
 * \file test_reduce_scatter.cpp
 * \brief
 */

#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk {
namespace Distributed {

template <typename T>
void TestReduceScatter(OpTestParam& testParam, std::string& goldenDir)
{
    constexpr size_t paramsSize = 5;
    auto [row, col, typeNum, tileRow, tileCol] = GetParams<paramsSize>(goldenDir + "/params.bin");
    CHECK(testParam.rankSize > 0) << "testParam.rankSize must be > 0, but got: " << testParam.rankSize;
    int rowOut = row / testParam.rankSize;
    DataType dType = GetDataTypeNum(typeNum);
    Tensor in(dType, {row, col}, "in");
    Tensor out(dType, {rowOut, col}, "out");

    std::vector<T> inData = ReadToVector<T>(goldenDir + "/input_rank_" + std::to_string(testParam.rankId) + ".bin",
                                            {row, col});

    Shape shmemDataShape{rowOut, col};
    FUNCTION("ShmemReduceScatter", {in}, {out})
    {
        DataType shmemDataType = in.GetDataType();
        shmemDataType = (shmemDataType == DT_BF16) || (shmemDataType == DT_FP16) ? DT_FP32 : shmemDataType;
        ShmemTensor shmemTensor;
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            CreateShmemTensor(testParam.group, testParam.rankSize, shmemDataType, shmemDataShape, shmemTensor);
        }
        TileShape::Current().SetVecTile({tileRow, tileCol});
        ReduceScatter(in, in, shmemTensor, DistReduceType::DIST_REDUCE_ADD, out);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(out, 0),
    });
    RunTest();
    auto outPut = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(
        CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", rowOut * col, outPut->GetDevPtr(), testParam));
}

template void TestReduceScatter<int32_t>(OpTestParam& testParam, std::string& goldenDir);
template void TestReduceScatter<float>(OpTestParam& testParam, std::string& goldenDir);
template void TestReduceScatter<float16>(OpTestParam& testParam, std::string& goldenDir);
template void TestReduceScatter<bfloat16>(OpTestParam& testParam, std::string& goldenDir);
} // namespace Distributed
} // namespace npu::tile_fwk
