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
 * \file test_allreduce_add_allreduce.cpp
 * \brief
 */

#include "distributed_op_test_suite.h"
#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"
#include "tilefwk/symbolic_distributed.h"

namespace npu::tile_fwk::Distributed {

void LoopAllReduce1(const Tensor& in, ShmemTensor& shmemTensor, Tensor& allReduceOut, int32_t row, int32_t col)
{
    LOOP("AllReduce1", FunctionType::DYNAMIC_LOOP, allReduce1Index, LoopRange(0, 1, 1))
    {
        (void)allReduce1Index;
        LOOP("AllReduce", FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            TileShape::Current().SetVecTile(row, col);
            OneShotAllReduce(in, in, shmemTensor, allReduceOut);
        }
    }
}

void LoopAdd(const Tensor& allReduceOut, Tensor& addOut)
{
    LOOP("Add", FunctionType::DYNAMIC_LOOP, index, LoopRange(0, 1, 1))
    {
        (void)index;
        TileShape::Current().SetVecTile({128, 256});
        addOut = npu::tile_fwk::Add(allReduceOut, allReduceOut);
    }
}

void LoopAllReduce2(const Tensor& addOut, ShmemTensor& shmemTensor, Tensor& out, int32_t row, int32_t col)
{
    auto shmemBarrier1ShmemSignal = CreateShmemSignal(shmemTensor.group.c_str(), shmemTensor.worldSize);
    auto shmemBarrier2ShmemSignal = CreateShmemSignal(shmemTensor.group.c_str(), shmemTensor.worldSize);
    LOOP("AllReduce2", FunctionType::DYNAMIC_LOOP, allReduce2Index, LoopRange(0, 1, 1))
    {
        (void)allReduce2Index;

        TileShape::Current().SetVecTile({1, 8});
        auto barrier1Out = ShmemBarrier(shmemBarrier1ShmemSignal, addOut);
        TileShape::Current().SetVecTile(row, col);
        auto memSetDataOut = ShmemClearData(shmemTensor, barrier1Out);
        auto memSetSignalOut = ShmemClearSignal(shmemTensor, barrier1Out);
        auto memSetOut = Nop({memSetDataOut, memSetSignalOut});
        TileShape::Current().SetVecTile({1, 8});
        auto barrier2Out = ShmemBarrier(shmemBarrier2ShmemSignal, memSetOut);
        TileShape::Current().SetVecTile(row, col);
        OneShotAllReduce(barrier2Out, addOut, shmemTensor, out);
    }
}

void FuncAllReduceAddAllReduce(const Tensor& in, Tensor& out, const OpTestParam& testParam, int32_t row, int32_t col)
{
    FUNCTION("AllReduceAddAllReduce", {in}, {out})
    {
        Tensor allReduceOut(in.GetDataType(), in.GetShape(), "allReduceOut");
        Tensor addOut(in.GetDataType(), in.GetShape(), "addOut");
        DataType shmemDataType = in.GetDataType();
        Shape shmemDataShape{row, col};
        if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
            shmemDataType = DT_FP32;
        }
        ShmemTensor shmemTensor;
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            CreateShmemTensor(testParam.group, testParam.rankSize, shmemDataType, shmemDataShape, shmemTensor);
        }
        LoopAllReduce1(in, shmemTensor, allReduceOut, row, col);
        allReduceOut.GetStorage()->UpdateDynValidShape(in.GetValidShape());
        LoopAdd(allReduceOut, addOut);
        addOut.GetStorage()->UpdateDynValidShape(in.GetValidShape());
        LoopAllReduce2(addOut, shmemTensor, out, row, col);
    };
}

template <typename T>
void TestAllReduceAddAllReduce(OpTestParam& testParam, std::string& goldenDir)
{
    constexpr size_t paramsSize = 5;
    auto [row, col, validRow, validCol, typeNum] = GetParams<paramsSize>(goldenDir + "/params.bin");

    Shape shape{row, col};
    DataType dType = GetDataTypeNum(typeNum);
    Tensor in(dType, shape, "in");
    Tensor out(dType, shape, "out");
    in.GetStorage()->UpdateDynValidShape(std::vector<SymbolicScalar>{validRow, validCol});
    std::vector<T> inPtr = ReadToVector<T>(goldenDir + "/input_rank_" + std::to_string(testParam.rankId) + ".bin",
                                           shape);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 10);
    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<T>(in, inPtr)});
    ProgramData::GetInstance().AppendOutputs({RawTensorData::CreateTensorZero(out)});

    FuncAllReduceAddAllReduce(in, out, testParam, row, col);

    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    int32_t outSize = row * col;
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/out_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduceAddAllReduce<int32_t>(OpTestParam& testParam, std::string& goldenDir);
template void TestAllReduceAddAllReduce<float>(OpTestParam& testParam, std::string& goldenDir);
template void TestAllReduceAddAllReduce<float16>(OpTestParam& testParam, std::string& goldenDir);
template void TestAllReduceAddAllReduce<bfloat16>(OpTestParam& testParam, std::string& goldenDir);

} // namespace npu::tile_fwk::Distributed
