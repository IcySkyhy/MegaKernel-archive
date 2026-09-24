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
 * \file test_allreduce.cpp
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
void TestAllReduce(OpTestParam& testParam, std::string& goldenDir)
{
    constexpr size_t paramsSize = 8;
    auto [row, col, validRow, validCol, typeNum, tileRow, tileCol,
          useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = row * col;

    Shape shape{row, col};
    Tensor in(dType, shape, "in");
    Tensor out(dType, shape, "out");

    std::vector<T> inPtr = ReadToVector<T>(goldenDir + "/input_rank_" + std::to_string(testParam.rankId) + ".bin",
                                           {row, col});

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inPtr),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(out),
    });
    Shape shmemDataShape{row, col};
    if (useTwoShot) {
        CHECK(testParam.rankSize > 0) << "testParam.rankSize must be > 0, but got: " << testParam.rankSize;
        shmemDataShape = {row / testParam.rankSize, col};
    }
    FUNCTION("ALLREDUCE", {in}, {out})
    {
        in.GetStorage()->UpdateDynValidShape(std::vector<SymbolicScalar>{validRow, validCol});
        TileShape::Current().SetVecTile({tileRow, tileCol});
        Tensor shmemData;
        Tensor shmemSignal;
        DataType shmemDataType = in.GetDataType();
        if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
            shmemDataType = DT_FP32;
        }
        ShmemTensor shmemTensor;
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            CreateShmemTensor(testParam.group, testParam.rankSize, shmemDataType, shmemDataShape, shmemTensor);
        }
        if (useTwoShot) {
            TwoShotAllReduce(in, in, shmemTensor, out);
        } else {
            OneShotAllReduce(in, in, shmemTensor, out);
        }
    }
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(
        CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce<int32_t>(OpTestParam& testParam, std::string& goldenDir);
template void TestAllReduce<float>(OpTestParam& testParam, std::string& goldenDir);
template void TestAllReduce<float16>(OpTestParam& testParam, std::string& goldenDir);
template void TestAllReduce<bfloat16>(OpTestParam& testParam, std::string& goldenDir);
} // namespace Distributed
} // namespace npu::tile_fwk
