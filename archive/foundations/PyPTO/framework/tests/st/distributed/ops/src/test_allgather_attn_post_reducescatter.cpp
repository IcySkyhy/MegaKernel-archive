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
 * \file test_allgather_attn_post_reducescatter.cpp
 * \brief
 */

#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"

namespace npu::tile_fwk {
namespace Distributed {
std::tuple<Tensor, Tensor, Tensor, Tensor> InitializeTestData(OpTestParam& testParam, std::string& goldenDir)
{
    constexpr size_t paramsSize = 7;
    auto [b, s, n, kvLoraRank, vHeadDim, h, typeNum] = GetParams<paramsSize>(goldenDir + "/params.bin");
    DataType dtype = GetDataTypeNum(typeNum);

    Shape agInShape = {b * n * s / testParam.rankSize, kvLoraRank};
    Shape wLoraShape = {n, kvLoraRank, vHeadDim};
    Shape wOutShape = {n * vHeadDim, h};
    Shape outShape = {b * s / testParam.rankSize, h};

    Tensor agIn(dtype, agInShape, "agIn");
    Tensor wLora(dtype, wLoraShape, "wLora");
    Tensor wOut(dtype, wOutShape, "wOut");
    Tensor out(dtype, outShape, "out");

    std::vector<bfloat16> agInPtr = ReadToVector<bfloat16>(
        goldenDir + "/ag_in_rank_" + std::to_string(testParam.rankId) + ".bin", agInShape);
    std::vector<bfloat16> wLoraPtr = ReadToVector<bfloat16>(
        goldenDir + "/w_lora_rank_" + std::to_string(testParam.rankId) + ".bin", wLoraShape);
    std::vector<bfloat16> wOutPtr = ReadToVector<bfloat16>(
        goldenDir + "/w_out_rank_" + std::to_string(testParam.rankId) + ".bin", wOutShape);

    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<bfloat16>(agIn, agInPtr)});
    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<bfloat16>(wLora, wLoraPtr)});
    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<bfloat16>(wOut, wOutPtr)});
    ProgramData::GetInstance().AppendOutputs({RawTensorData::CreateTensorZero(out)});
    return {agIn, wLora, wOut, out};
}

void TestAllGatherAttentionPostReducescatter(OpTestParam& testParam, std::string& goldenDir)
{
    constexpr size_t paramsSize = 7;
    auto [b, s, n, kvLoraRank, vHeadDim, h, typeNum] = GetParams<paramsSize>(goldenDir + "/params.bin");
    DataType dtype = GetDataTypeNum(typeNum);
    auto [agIn, wLora, wOut, out] = InitializeTestData(testParam, goldenDir);
    CHECK(testParam.rankSize > 0) << "testParam.rankSize must be > 0, but got: " << testParam.rankSize;
    int32_t outRow = b * s / testParam.rankSize;
    FUNCTION("ALLGATHER_ATTNPOST_REDUCESCATTER", {agIn, wLora, wOut}, {out})
    {
        Tensor agOut(dtype, {b * n * s, kvLoraRank}, "agOut");
        LOOP("ALLGATHER", FunctionType::DYNAMIC_LOOP, unusedDynRankId, LoopRange(1))
        {
            (void)unusedDynRankId;
            Shape shmemDataAgShape{testParam.rankSize * b * n * s / testParam.rankSize, kvLoraRank};
            ShmemTensor shmemTensor = CreateShmemTensor(testParam.group, testParam.rankSize, dtype, shmemDataAgShape);
            TileShape::Current().SetVecTile({64, kvLoraRank});
            AllGather(agIn, agIn, shmemTensor, agOut);
        }
        Tensor attnOut(dtype, {b * s, h}, "attnOut");
        LOOP("ATTNPOST", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1))
        {
            (void)batchId;
            TileShape::Current().SetVecTile({4, 16, 1, kvLoraRank});
            Tensor attnIn = Reshape(agOut, {b, n, s, kvLoraRank});
            TileShape::Current().SetVecTile({4, 16, 1, kvLoraRank});
            Tensor attnRes0 = Transpose(attnIn, {1, 2});
            TileShape::Current().SetVecTile({4, 1, 32, std::min(512, kvLoraRank)});
            Tensor attnRes1 = Reshape(attnRes0, {b * s, n, kvLoraRank});
            TileShape::Current().SetVecTile({4, 16, kvLoraRank});
            Tensor t2Res = Transpose(attnRes1, {0, 1});
            TileShape::Current().SetCubeTile({16, 16}, {256, 256}, {128, 128});
            Tensor fp32Bmm4Res = Matrix::BatchMatmul(DataType::DT_FP32, t2Res, wLora);
            Tensor bmm4Res = Cast(fp32Bmm4Res, dtype);
            TileShape::Current().SetVecTile({32, 4, vHeadDim}); // 必须切，但是尾轴不能切
            Tensor t3Res = Transpose(bmm4Res, {0, 1});          // [bs,n,vHeadDim]
            TileShape::Current().SetVecTile({4, 32, vHeadDim});
            Tensor r2Res = Reshape(t3Res, {b * s, n * vHeadDim});
            TileShape::Current().SetCubeTile({16, 16}, {256, 256}, {128, 128});
            attnOut = Matrix::Matmul(dtype, r2Res, wOut, false, false);
        }
        LOOP("REDUCESCATTER", FunctionType::DYNAMIC_LOOP, unusedIndex, LoopRange(1))
        {
            (void)unusedIndex;
            DataType shmemDataType = (attnOut.GetDataType() == DT_BF16 || attnOut.GetDataType() == DT_FP16) ?
                                         DT_FP32 :
                                         attnOut.GetDataType();
            ShmemTensor shmemTensor = CreateShmemTensor(testParam.group, testParam.rankSize, shmemDataType,
                                                        {outRow, h});
            TileShape::Current().SetVecTile({16, h});
            Distributed::ReduceScatter(attnOut, attnOut, shmemTensor, DistReduceType::DIST_REDUCE_ADD, out);
        }
    }
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dtype, goldenDir + "/rs_out_rank_", outRow * h, output->GetDevPtr(),
                                            testParam, 0.1f));
}

} // namespace Distributed
} // namespace npu::tile_fwk
