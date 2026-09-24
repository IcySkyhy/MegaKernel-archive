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
 * \file test_slice_contract_debug.cpp
 * \brief Smoke tests for slice/contract integration debugging.
 *
 * Runs the A3 (non-3510) PVC2_OOO pass pipeline up to and including GenerateMoveOp.
 * Cases only need to execute successfully; no assertions are added.
 */

#include <vector>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_mgr/pass_manager.h"
#include "tilefwk/platform.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {
const int NUM_64 = 64;
const int NUM_128 = 128;
const int NUM_256 = 256;

class SliceContractDebugTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_TEST_IS_TIG, true);
        SetSliceContractDebugStrategy();
        config::SetPassStrategy("SliceContractDebugStrategy");
    }

    void TearDown() override {}

    // Mirror PVC2_OOO up to and including GenerateMoveOp (A3 / non-3510 path).
    void SetSliceContractDebugStrategy()
    {
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("SliceContractDebugStrategy",
                                     {
                                         {"InferTensorFormat", PassName::INFER_TENSOR_FORMAT},
                                         {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                         {"AutoCast", PassName::AUTO_CAST},
                                         {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                         {"RemoveUndrivenView", PassName::REMOVE_UNDRIVEN_VIEW},
                                         {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                         {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                         {"SplitReshape", PassName::SPLIT_RESHAPE},
                                         {"SplitRawTensor", PassName::SPLIT_RAW_TENSOR},
                                         {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                         {"DuplicateOp", PassName::DUPLICATE_OP},
                                         {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                         {"InferDiscontinuousInput", PassName::INFER_DISCONTINUOUS_INPUT},
                                         {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
                                         {"InsertOpForViewAssemble", PassName::INSERT_OP_FOR_VIEWASSEMBLE},
                                         {"ProcessAtomic", PassName::PROCESS_ATOMIC},
                                         {"GraphPartition", PassName::GRAPH_PARTITION},
                                         {"NBufferMerge", PassName::N_BUFFER_MERGE},
                                         {"L1CopyInReuseMerge", PassName::L1_COPY_IN_REUSE_MERGE},
                                         {"ReduceCopyMerge", PassName::REDUCE_COPY_MERGE},
                                         {"IntraSubgraphAdapter", PassName::INTRA_SUBGRAPH_ADAPTER},
                                         {"GenerateMoveOp", PassName::GENERATE_MOVE_OP},
                                     });
        ConfigManager::Instance();
    }
};

// Case 1: (A + B) + C -> out, two matrix adds.
TEST_F(SliceContractDebugTest, AddThenAdd)
{
    std::vector<int64_t> shape = {NUM_256, NUM_128};
    PROGRAM("SliceContractDebug")
    {
        Tensor inputA(DataType::DT_FP32, shape, "A");
        Tensor inputB(DataType::DT_FP32, shape, "B");
        Tensor inputC(DataType::DT_FP32, shape, "C");
        Tensor out(DataType::DT_FP32, shape, "out");
        config::SetBuildStatic(true);
        FUNCTION("AddThenAdd", {inputA, inputB, inputC, out})
        {
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            Tensor addRes = Add(inputA, inputB);
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            out = Add(addRes, inputC);
        }
    }
}

// Case 1b: (A + B) -> reshape -> [256,1,128] + C -> out. Reshape on the first add
// result expands to assemble-reshape-view, exercising SPLIT_RESHAPE.
// Inserts a middle axis (256,128)->(256,1,128); vecTile {128,128,128} keeps all dims
// 128 so that 2D inCast views taking the first 2 dims ({128,128}) still split correctly.
TEST_F(SliceContractDebugTest, AddReshapeAdd)
{
    std::vector<int64_t> shape = {NUM_256, NUM_128};
    std::vector<int64_t> reshapeShape = {NUM_256, 1, NUM_128};
    PROGRAM("SliceContractDebug")
    {
        Tensor inputA(DataType::DT_FP32, shape, "A");
        Tensor inputB(DataType::DT_FP32, shape, "B");
        Tensor inputC(DataType::DT_FP32, shape, "C");
        Tensor out(DataType::DT_FP32, reshapeShape, "out");
        config::SetBuildStatic(true);
        FUNCTION("AddReshapeAdd", {inputA, inputB, inputC, out})
        {
            TileShape::Current().SetVecTile(NUM_128, NUM_128, NUM_128);
            Tensor addRes = Add(inputA, inputB);
            TileShape::Current().SetVecTile(NUM_128, NUM_128, NUM_128);
            Tensor reshapeRes = Reshape(addRes, reshapeShape);
            TileShape::Current().SetVecTile(NUM_128, NUM_128, NUM_128);
            Tensor reshapeC = Reshape(inputC, reshapeShape);
            TileShape::Current().SetVecTile(NUM_128, NUM_128, NUM_128);
            out = Add(reshapeRes, reshapeC);
        }
    }
}

// [KnownIssue] This variant uses head-axis expansion (256,128)->(1,256,128) and
// triggers a framework bug: inCast views of 2D inputs (A/B/C) are created at the end
// of FUNCTION build, inheriting the final TileShape::Current() = {1,128,128} (3D).
// Since the inCast view tensors are 2D {256,128} but the tile is 3D {1,128,128},
// TiledViewOperationRecursive iterates 2 dims and takes {1,128} -> 256 over-split
// slices instead of the expected 2. Commented out for the framework team to
// investigate whether inCast views should derive their tile from the tensor's actual
// rank rather than the global final Current().
TEST_F(SliceContractDebugTest, AddReshapeAddHeadAxis)
{
    std::vector<int64_t> shape = {NUM_256, NUM_128};
    std::vector<int64_t> reshapeShape = {1, NUM_256, NUM_128};
    PROGRAM("SliceContractDebug")
    {
        Tensor inputA(DataType::DT_FP32, shape, "A");
        Tensor inputB(DataType::DT_FP32, shape, "B");
        Tensor inputC(DataType::DT_FP32, shape, "C");
        Tensor out(DataType::DT_FP32, reshapeShape, "out");
        config::SetBuildStatic(true);
        FUNCTION("AddReshapeAddHeadAxis", {inputA, inputB, inputC, out})
        {
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            Tensor addRes = Add(inputA, inputB);
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            Tensor reshapeRes = Reshape(addRes, reshapeShape);
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            Tensor reshapeC = Reshape(inputC, reshapeShape);
            TileShape::Current().SetVecTile(1, NUM_128, NUM_128);
            out = Add(reshapeRes, reshapeC);
        }
    }
}

// Case 2: (A + B) @ W -> out, one matrix add then one matrix multiply.
TEST_F(SliceContractDebugTest, AddThenMatmul)
{
    std::vector<int64_t> shape0 = {NUM_256, NUM_128};
    std::vector<int64_t> shape1 = {NUM_128, NUM_64};
    std::vector<int64_t> shape2 = {NUM_256, NUM_64};
    PROGRAM("SliceContractDebug")
    {
        Tensor inputA(DataType::DT_FP32, shape0, "A");
        Tensor inputB(DataType::DT_FP32, shape0, "B");
        Tensor weight(DataType::DT_FP32, shape1, "W");
        Tensor out(DataType::DT_FP32, shape2, "out");
        config::SetBuildStatic(true);
        FUNCTION("AddThenMatmul", {inputA, inputB, weight, out})
        {
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            Tensor addRes = Add(inputA, inputB);
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            out = Matrix::Matmul(out.GetDataType(), addRes, weight);
        }
    }
}

// Case 3: (A @ B) + (C @ D) -> out, two matrix multiplies then one add.
TEST_F(SliceContractDebugTest, MatmulAddMatmul)
{
    std::vector<int64_t> shape0 = {NUM_256, NUM_128};
    std::vector<int64_t> shape1 = {NUM_128, NUM_64};
    std::vector<int64_t> shape2 = {NUM_256, NUM_64};
    PROGRAM("SliceContractDebug")
    {
        Tensor inputA(DataType::DT_FP32, shape0, "A");
        Tensor inputB(DataType::DT_FP32, shape1, "B");
        Tensor inputC(DataType::DT_FP32, shape0, "C");
        Tensor inputD(DataType::DT_FP32, shape1, "D");
        Tensor out(DataType::DT_FP32, shape2, "out");
        config::SetBuildStatic(true);
        FUNCTION("MatmulAddMatmul", {inputA, inputB, inputC, inputD, out})
        {
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor mmRes1 = Matrix::Matmul(out.GetDataType(), inputA, inputB);
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor mmRes2 = Matrix::Matmul(out.GetDataType(), inputC, inputD);
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            out = Add(mmRes1, mmRes2);
        }
    }
}

// Case 4 (FP32): (A @ B) @ (C @ D) -> out, two matmul results fed into a matmul.
// FP32 path does not go through l0c2l1.
// A:[256,128]@B:[128,128] = AB:[256,128]; C:[128,64]@D:[64,64] = CD:[128,64];
// AB@CD = [256,128]@[128,64] = out:[256,64].
TEST_F(SliceContractDebugTest, MatmulMatmulFp32)
{
    std::vector<int64_t> shapeA = {NUM_256, NUM_128};
    std::vector<int64_t> shapeB = {NUM_128, NUM_128};
    std::vector<int64_t> shapeC = {NUM_128, NUM_64};
    std::vector<int64_t> shapeD = {NUM_64, NUM_64};
    std::vector<int64_t> shapeOut = {NUM_256, NUM_64};
    PROGRAM("SliceContractDebug")
    {
        Tensor inputA(DataType::DT_FP32, shapeA, "A");
        Tensor inputB(DataType::DT_FP32, shapeB, "B");
        Tensor inputC(DataType::DT_FP32, shapeC, "C");
        Tensor inputD(DataType::DT_FP32, shapeD, "D");
        Tensor out(DataType::DT_FP32, shapeOut, "out");
        config::SetBuildStatic(true);
        FUNCTION("MatmulMatmulFp32", {inputA, inputB, inputC, inputD, out})
        {
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor abRes = Matrix::Matmul(out.GetDataType(), inputA, inputB);
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor cdRes = Matrix::Matmul(out.GetDataType(), inputC, inputD);
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            out = Matrix::Matmul(out.GetDataType(), abRes, cdRes);
        }
    }
}

// Case 5 (FP16): (A @ B) @ (C @ D) -> out, two matmul results fed into a matmul.
// FP16 path may go through l0c2l1 under certain conditions.
TEST_F(SliceContractDebugTest, MatmulMatmulFp16)
{
    std::vector<int64_t> shapeA = {NUM_256, NUM_128};
    std::vector<int64_t> shapeB = {NUM_128, NUM_128};
    std::vector<int64_t> shapeC = {NUM_128, NUM_64};
    std::vector<int64_t> shapeD = {NUM_64, NUM_64};
    std::vector<int64_t> shapeOut = {NUM_256, NUM_64};
    PROGRAM("SliceContractDebug")
    {
        Tensor inputA(DataType::DT_FP16, shapeA, "A");
        Tensor inputB(DataType::DT_FP16, shapeB, "B");
        Tensor inputC(DataType::DT_FP16, shapeC, "C");
        Tensor inputD(DataType::DT_FP16, shapeD, "D");
        Tensor out(DataType::DT_FP16, shapeOut, "out");
        config::SetBuildStatic(true);
        FUNCTION("MatmulMatmulFp16", {inputA, inputB, inputC, inputD, out})
        {
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor abRes = Matrix::Matmul(out.GetDataType(), inputA, inputB);
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor cdRes = Matrix::Matmul(out.GetDataType(), inputC, inputD);
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            out = Matrix::Matmul(out.GetDataType(), abRes, cdRes);
        }
    }
}
} // namespace tile_fwk
} // namespace npu
