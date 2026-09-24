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
 * \file test_codegen_dyn_binary.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/utils/id_gen.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"
#include "passes/pass_mgr/pass_manager.cpp"

namespace npu::tile_fwk {
class TestCodegenForLoop : public CodegenTestBase {
public:
    TestCodegenForLoop() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH, .setIdGen = true}) {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override
    {
        CodegenTestBase::SetUp();
        config::SetPassGlobalConfig(KEY_VF_OPT_MARK_FOR, true);
        config::SetHostConfig(KEY_STRATEGY, "LoopAxesPassTestStrategy");
    }
};

TEST_F(TestCodegenForLoop, TestForLoop)
{
    std::vector<int64_t> shape = {2, 2, 2, 8};
    std::vector<int64_t> oShape = {2, 2, 2, 1};
    std::vector<int64_t> tile_shape = {2, 2, 2, 8};

    TileShape::Current().SetVecTile(tile_shape);
    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("LoopAxesPassTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                     {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                     {"DuplicateOp", PassName::DUPLICATE_OP},
                                     {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                     {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                     {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                     {"SplitReshape", PassName::SPLIT_RESHAPE},
                                     {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
                                     {"GenerateMoveOp", PassName::GENERATE_MOVE_OP},
                                     {"CommonOperationEliminate", PassName::COMMON_OPERATION_ELIMINATE},
                                     {"AxisCombine", PassName::AXIS_COMBINE},
                                     {"PadLocalBuffer", PassName::PAD_LOCAL_BUFFER},
                                     {"RemoveUnalignedReshape", PassName::REMOVE_UNALIGNED_RESHAPE},
                                     {"ReplaceTensor", PassName::REPLACE_TENSOR},
                                     {"PreGraphProcess", PassName::PRE_GRAPH_PROCESS},
                                     {"InferDynShape", PassName::INFER_DYN_SHAPE},
                                     {"SubgraphToFunction", PassName::SUBGRAPH_TO_FUNCTION},
                                     {"InferParamIndex", PassName::INFER_PARAM_INDEX},
                                     {"SrcDstBufferMerge", PassName::SRC_DST_BUFFER_MERGE},
                                     {"AddAlloc", PassName::ADD_ALLOC},
                                     {"OoOSchedule", PassName::OOO_SCHEDULE},
                                     {"GlobalMemoryReuse", PassName::GLOBAL_MEMORY_REUSE},
                                     {"RemoveAlloc", PassName::REMOVE_ALLOC},
                                     {"CopyOutResolve", PassName::COPY_OUT_RESOLVE},
                                     {"InsertSync", PassName::INSERT_SYNC},
                                     {"LastUseMark", PassName::LAST_USE_MARK},
                                 });
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape, "B");
    Tensor output(DT_FP32, oShape, "Output");

    std::string name = "TestForLoop";
    FUNCTION(name, {input_a, input_b, output})
    {
        LOOP(name, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto res1 = Add(input_a, input_b);
            auto res2 = Sub(input_a, input_b);
            auto res3 = Mul(res1, res2);
            auto res4 = Cast(res3, DT_FP16);
            auto res5 = Cast(res4, DT_INT8, CAST_NONE, SaturationMode::ON);
            auto res6 = Cast(res5, DT_FP16);
            auto res7 = Cast(res6, DT_FP32);
            output = Sum(res7, -1, true);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    LoopaxesProc lpPass;
    lpPass.RunOnFunction(*function);
    CodegenPreproc cpPass;
    cpPass.RunOnFunction(*function);

    std::string res = GenCodeByFunction(*function);

    // 定义第一个待检查的目标代码片段
    const std::string expect1 = R"(
        auto tileOffsets = TileOffset(0, idx1, idx2);
        ubTensor_2_low2DimInLoop.SetAddr(ubTensor_2.GetLinearAddr(tileOffsets));
        ubTensor_0_low2DimInLoop.SetAddr(ubTensor_0.GetLinearAddr(tileOffsets));
        ubTensor_4_low2DimInLoop.SetAddr(ubTensor_4.GetLinearAddr(tileOffsets));
        TAdd<LastUse3Dim<0, 0, 0>>(ubTensor_4_low2DimInLoop, ubTensor_0_low2DimInLoop, ubTensor_2_low2DimInLoop);
    }
  }
})";
    CheckStringExist(expect1, res);

    const std::string expect2 = R"(
        auto tileOffsets = TileOffset(0, idx1, idx2);
        ubTensor_10_low2DimInLoop.SetAddr(ubTensor_10.GetLinearAddr(tileOffsets));
        ubTensor_4_low2DimInLoop.SetAddr(ubTensor_4.GetLinearAddr(tileOffsets));
        TMul<LastUse3Dim<0, 0, 1>>(ubTensor_4_low2DimInLoop, ubTensor_4_low2DimInLoop, ubTensor_10_low2DimInLoop);
    }
  }
})";
    CheckStringExist(expect2, res);

    const std::string expect3 = R"(
        auto tileOffsets = TileOffset(0, idx1, idx2);
        ubTensor_34_low2DimInLoop.SetAddr(ubTensor_34.GetLinearAddr(tileOffsets));
        ubTensor_38_low2DimInLoop.SetAddr(ubTensor_38.GetLinearAddr(tileOffsets));
        TRowSumSingle<LastUse3Dim<0, 0, 0>>(ubTensor_38_low2DimInLoop, ubTensor_34_low2DimInLoop, ubTensor_39);
)";
    CheckStringExist(expect3, res);
}

} // namespace npu::tile_fwk
