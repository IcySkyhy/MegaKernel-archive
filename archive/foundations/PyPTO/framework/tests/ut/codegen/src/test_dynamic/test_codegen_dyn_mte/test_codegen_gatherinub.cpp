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
 * \file test_codegen_dyn_spillout.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "interface/utils/id_gen.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenGatherInUB : public CodegenTestBase {
public:
    TestCodegenGatherInUB()
        : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH, .setTileTensor = true, .setIdGen = true})
    {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};
// ----------------- 配置结构体（含类型） -----------------
// IndexT  : topk_indices / page_table 的整数类型
// DataT   : buffer / golden_result 的数据类型
template <typename IndexT, typename DataT>
struct PageAttentionTestConfig {
    using IndexType = IndexT;
    using DataType = DataT;

    int topk_count;         // topk 的 k 值：选出的 token 个数
    int num_logical_blocks; // 逻辑块个数（page_table 长度）
    int num_buffer_tokens;  // buffer 第一维长度：物理 token 容量
    int hidden_dim;         // buffer 第二维长度：隐藏维度大小
    int block_size;         // 每个块里有多少个 token
};
template <typename Config>
Function* GatherInUBUT(Config& cfg)
{
    Shape srcShapes{cfg.num_buffer_tokens, cfg.hidden_dim}; // 网络中，kvcache对应的内存
    Shape offsetsShapes{1, cfg.topk_count};                 // topk的结果
    Shape pageTableShapes{1, cfg.num_logical_blocks};       // page attention 对应的页表
    Shape dstShapes{cfg.topk_count, cfg.hidden_dim};        // 结果，将topk个数据拿出来

    Tensor src(DT_FP16, srcShapes, "src");
    Tensor offsets(DT_INT32, offsetsShapes, "offsets");
    Tensor pageTable(DT_INT32, pageTableShapes, "pageTable");
    Tensor dst(DT_FP16, dstShapes, "dst");
    const std::string funName = "GatherInUB";
    FUNCTION(funName, {src, offsets, pageTable}, {dst})
    {
        TileShape::Current().SetVecTile({32, 64});
        std::vector<SymbolicScalar> srcValidShape = {src.GetShape()[0], src.GetShape()[1]};
        Tensor dynSrc = View(src, src.GetShape(), srcValidShape, {0, 0});

        std::vector<SymbolicScalar> offsetsValidShape = {offsets.GetShape()[0], offsets.GetShape()[1]};
        Tensor dynOffsets = View(offsets, offsets.GetShape(), offsetsValidShape, {0, 0});

        dst = experimental::GatherInUB(dynSrc, dynOffsets, pageTable, cfg.block_size, -2);
    }

#if ENABLE_HIDDENLOOP
    auto function = Program::GetInstance().GetFunctionByRawName("TENSOR_TENSOR_" + funName +
                                                                "_loop_Unroll1_PATH0_hiddenfunc0");
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funName);
#endif
    (void)GenCodeByFunction(*function);
    return function;
}

TEST_F(TestCodegenGatherInUB, gather_in_a_)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    using Config = PageAttentionTestConfig<int32_t, float16>;
    Config cfg;
    cfg.topk_count = 8;         // topk结果
    cfg.num_logical_blocks = 3; // 逻辑块个数
    cfg.num_buffer_tokens = 32; // buffer token 维度（物理 token 容量）
    cfg.hidden_dim = 4;         // 隐藏维度大小
    cfg.block_size = 4;         // 每个块的 token 数
    auto function = GatherInUBUT(cfg);
    std::string res = GetResultFromCpp(*function);
    std::string expect =
        R"!!!(TileOp::GatherInUB<half, int32_t, int32_t, 8, 16, 4>((__ubuf__ half*)UB_S0_E256, (__gm__ half*)(RUNTIME_GET_PARAM_ADDR(RUNTIME_param, 2, 1)), (__gm__ int32_t*)(RUNTIME_GET_PARAM_ADDR(RUNTIME_param, 2, 10)), (__gm__ int32_t*)(RUNTIME_GET_PARAM_ADDR(RUNTIME_param, 2, 19)), sym_60_dim_1, GET_PARAM_RAWSHAPE_BY_IDX(param, 2, 1, 2, 1), GET_PARAM_OFFSET_BY_IDX(param, 2, 1, 2, 0), GET_PARAM_OFFSET_BY_IDX(param, 2, 1, 2, 1), sym_60_dim_0, GET_PARAM_RAWSHAPE_BY_IDX(param, 2, 10, 2, 1), GET_PARAM_OFFSET_BY_IDX(param, 2, 10, 2, 0), GET_PARAM_OFFSET_BY_IDX(param, 2, 10, 2, 1), GET_PARAM_RAWSHAPE_BY_IDX(param, 2, 19, 2, 1), GET_PARAM_OFFSET_BY_IDX(param, 2, 19, 2, 0), GET_PARAM_OFFSET_BY_IDX(param, 2, 19, 2, 1));)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenGatherInUB, gather_in_a_tile_tensor)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    using Config = PageAttentionTestConfig<int32_t, float16>;
    Config cfg;
    cfg.topk_count = 8;         // topk结果
    cfg.num_logical_blocks = 3; // 逻辑块个数
    cfg.num_buffer_tokens = 32; // buffer token 维度（物理 token 容量）
    cfg.hidden_dim = 4;         // 隐藏维度大小
    cfg.block_size = 4;         // 每个块的 token 数
    auto function = GatherInUBUT(cfg);
    std::string res = GetResultFromCpp(*function);
    std::string expect =
        R"!!!(TgatherInUB<4>(ubTensor_0, gmTensor_1, gmTensor_2, gmTensor_3, Coord2Dim(GET_PARAM_OFFSET_2(param, 2, 1)), Coord2Dim(GET_PARAM_OFFSET_2(param, 2, 10)), Coord2Dim(GET_PARAM_OFFSET_2(param, 2, 19)));)!!!";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
