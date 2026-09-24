// /**
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
//  * CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// /*!
//  * \file test_graph_only.cpp
//  * \brief
//  */

// #include "gtest/gtest.h"

// #include "interface/tensor/logical_tensor.h"
// #include "interface/configs/config_manager.h"
// #include "tilefwk/tilefwk.h"
// #include "interface/inner/tilefwk.h"
// #include "operator/models/llama/llama_def.h"
// #include "tilefwk/data_type.h"

// using namespace npu::tile_fwk;

// constexpr int NUM_2 = 2;
// constexpr int NUM_8 = 8;
// constexpr int NUM_128 = 128;

// class GraphTest : public testing::Test {
// public:
//     static void SetUpTestCase() {}

//     static void TearDownTestCase() {}

//     void SetUp() override
//     {
//         Program::GetInstance().Reset();
//         config::Reset();
//         config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
//         config::SetHostConfig(KEY_STRATEGY, "PVC2_OOO");
//         config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
//         config::SetSimConfig(KEY_BUILD_TASK_BASED_TOPO, false);
//     }

//     void TearDown() override {}
// };

// void RunLLamaLayerGraph(const AttentionDims& dimsCfg)
// {
//     int b = dimsCfg.b;
//     int n = dimsCfg.n;
//     int s = dimsCfg.s;
//     int d = dimsCfg.d;
//     PROGRAM("LLAMALAYER")
//     {
//         Tensor H(DataType::DT_FP32, {b * s, n * d}, "H");
//         Tensor AW(DataType::DT_FP16, {n * d, n * d * 3}, "AW");
//         Tensor DW(DataType::DT_FP16, {n * d, n * d}, "DW");
//         Tensor FW(DataType::DT_FP16, {n * d, n * d * 3}, "FW");
//         Tensor Res(DT_FP32, {b * s, n * d}, "Res");
//         FUNCTION("LLAMA") { Res = LlamaLayer(H, AW, DW, FW, dimsCfg, SMALL_DFS_VEC_CFG, DFS_CUBE_CFG); }
//     }
// }

// TEST_F(GraphTest, llama_1_1_128_128)
// {
//     AttentionDims dimsCfg = {1, 1, 128, 128, DFT_SINGLE_M, DFT_SINGLE_N};
//     RunLLamaLayerGraph(dimsCfg);
// }

// TEST_F(GraphTest, llama_1_1_256_128)
// {
//     AttentionDims dimsCfg = {1, 1, 256, 128, DFT_SINGLE_M, DFT_SINGLE_N};
//     RunLLamaLayerGraph(dimsCfg);
// }

// // TEST_F(GraphTest, llama_1_1_512_128)
// // {
// //     AttentionDims dimsCfg = {1, 1, 512, 128, DFT_SINGLE_M, DFT_SINGLE_N};
// //     RunLLamaLayerGraph(dimsCfg);
// // }

// TEST_F(GraphTest, llama_1_1_256_128_mix)
// {
//     config::SetPassConfig("PVC2_OOO", "PreGraphProcess", KEY_PRE_CHECK, false);
//     config::SetPassConfig("PVC2_OOO", "PreGraphProcess", KEY_POST_CHECK, false);
//     AttentionDims dimsCfg = {1, 1, 256, 128, DFT_SINGLE_M, DFT_SINGLE_N};
//     RunLLamaLayerGraph(dimsCfg);
// }

// void TestLoopTailBlock(const Tensor& t0, const Tensor& blockTable, Tensor& out, int s)
// {
//     int blockSize = 64;

//     FUNCTION("main", {t0, blockTable}, {out})
//     {
//         LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s))
//         {
//             SymbolicScalar size = GetTensorData(blockTable, {i, 0});
//             Tensor t0s = View(t0, {s, s}, {size, s}, {blockSize * i, 0});
//             Tensor t1s = View(t0, {s / 2, s}, {size, s}, {blockSize * i, 0});
//             Tensor t1 = Add(t1s, t1s);
//             Assemble(t1, {blockSize * i, 0}, out);
//         }
//     }
// }

// TEST_F(GraphTest, TestTailBlock)
// {
//     TileShape::Current().SetVecTile(32, 32);
//     TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
//     int s = 64;
//     int n = 8;
//     Tensor t0(DT_FP32, {n * s, s}, "t0"); // [32*8, 32]
//     Tensor blockTable{DT_INT32, {n, 1}, "blockTable"};
//     Tensor out(DT_FP32, {n * s, s}, "out");
//     TestLoopTailBlock(t0, blockTable, out, s);
// }

// TEST_F(GraphTest, TestTranspose_MLA_3D_2_add)
// {
//     int bs = 8;
//     int n = 32;
//     int d = 128;
//     std::vector<int64_t> shape{bs, n, d};
//     std::vector<int64_t> resShape{n, bs, d};
//     PROGRAM("Transpose")
//     {
//         Tensor input(DataType::DT_FP32, shape, "input");
//         Tensor output(DataType::DT_FP32, resShape, "res");
//         config::SetBuildStatic(true);
//         FUNCTION("MLA_3D_2", {input, output})
//         {
//             TileShape::Current().SetVecTile(NUM_2, NUM_2, NUM_128);
//             auto tmp = Transpose(input, {0, 1});
//             TileShape::Current().SetVecTile(NUM_8, NUM_8, NUM_128);
//             output = Add(tmp, Element(DataType::DT_FP32, 0.0));
//         }
//     }
// }

// TEST_F(GraphTest, TestTranspose_MLA_3D_2_reshape)
// {
//     int bs = 8;
//     int n = 32;
//     int d = 128;
//     std::vector<int64_t> shape{bs, n, d};
//     std::vector<int64_t> transposeShape{n, bs, d};
//     std::vector<int64_t> resShape{n, bs * d};
//     std::vector<int64_t> flattenShape{n * bs * d};

//     PROGRAM("Transpose")
//     {
//         Tensor input(DataType::DT_FP32, shape, "input");
//         Tensor output1(DataType::DT_FP32, transposeShape, "res1");
//         Tensor output2(DataType::DT_FP32, resShape, "res2");
//         Tensor output3(DataType::DT_FP32, flattenShape, "res3");
//         config::SetBuildStatic(true);
//         FUNCTION("MLA_3D_2", {input, output1, output2})
//         {
//             TileShape::Current().SetVecTile(NUM_2, NUM_2, NUM_128);
//             output1 = Transpose(input, {0, 1}); // [8, 32, 128] --> [32, 8, 128]
//             TileShape::Current().SetVecTile(NUM_8, NUM_8, NUM_128);
//             output2 = Reshape(output1, resShape); // [32, 8, 128] --> [32, 1024]
//             output3 = Reshape(output1, {-1});     // [32, 8, 128] --> [32 * 8 * 128]
//         }
//     }
// }
