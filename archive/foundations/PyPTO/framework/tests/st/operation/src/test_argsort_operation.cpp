/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_argsort_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
const unsigned IDX_DIM0 = 0;
const unsigned IDX_DIM1 = 1;
const unsigned IDX_DIM2 = 2;
const unsigned IDX_DIM3 = 3;

struct ArgSortOpFuncArgs : public OpFuncArgs {
    ArgSortOpFuncArgs(std::vector<int64_t> viewShape, const std::vector<int64_t> tileShape, std::vector<int> dims,
                      std::vector<bool> descending)
        : viewShape_(viewShape), tileShape_(tileShape), dims_(dims), descending_(descending)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    std::vector<int> dims_;
    std::vector<bool> descending_;
};

struct ArgSortOpMetaData {
    ArgSortOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data) : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void ArgSortOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        if (inputs[0].GetShape().size() == 1) {
            const struct ArgSortOpFuncArgs* args = static_cast<const ArgSortOpFuncArgs*>(opArgs);
            SymbolicScalar firstDim = inputs[0].GetShape()[0];
            const int firstViewShape = args->viewShape_[0];
            int loop[] = {CeilDiv(firstDim, firstViewShape)};
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
            {
                std::vector<SymbolicScalar> offset = {bIdx * args->viewShape_[0]};
                auto viewTensor = View(inputs[0], args->viewShape_,
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape)}, offset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = ArgSort(viewTensor, args->dims_[0], args->descending_[0]);
                Assemble(res, offset, outputs[0]);
            }
        } else {
            const struct ArgSortOpFuncArgs* args = static_cast<const ArgSortOpFuncArgs*>(opArgs);
            SymbolicScalar firstDim = inputs[0].GetShape()[0];
            SymbolicScalar secondDim = inputs[0].GetShape()[1];
            const int firstViewShape = args->viewShape_[0];
            const int secondViewShape = args->viewShape_[1];
            int loop[] = {CeilDiv(firstDim, firstViewShape), CeilDiv(secondDim, secondViewShape)};
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
                {
                    std::vector<SymbolicScalar> offset = {bIdx * args->viewShape_[0], sIdx * args->viewShape_[1]};
                    auto viewTensor = View(inputs[0], args->viewShape_,
                                           {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                            std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                           offset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = ArgSort(viewTensor, args->dims_[0], args->descending_[0]);
                    Assemble(res, offset, outputs[0]);
                }
            }
        }
    }
}

static void ArgSortOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        const struct ArgSortOpFuncArgs* args = static_cast<const ArgSortOpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        int loop[] = {CeilDiv(firstDim, firstViewShape), CeilDiv(secondDim, secondViewShape),
                      CeilDiv(thirdDim, thirdViewShape)};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(loop[IDX_DIM2]))
                {
                    std::vector<SymbolicScalar> offset = {bIdx * args->viewShape_[0], sIdx * args->viewShape_[1],
                                                          mIdx * args->viewShape_[2]};
                    auto viewTensor = View(inputs[0], args->viewShape_,
                                           {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                            std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape)},
                                           offset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = ArgSort(viewTensor, args->dims_[0], args->descending_[0]);
                    Assemble(res, offset, outputs[0]);
                }
            }
        }
    }
}

static void ArgSortOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        const struct ArgSortOpFuncArgs* args = static_cast<const ArgSortOpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        int loop[] = {CeilDiv(firstDim, firstViewShape), CeilDiv(secondDim, secondViewShape),
                      CeilDiv(thirdDim, thirdViewShape), CeilDiv(fourthDim, fourthViewShape)};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(loop[IDX_DIM2]))
                {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM3]))
                    {
                        std::vector<SymbolicScalar> offset = {bIdx * args->viewShape_[0], sIdx * args->viewShape_[1],
                                                              mIdx * args->viewShape_[2], nIdx * args->viewShape_[3]};
                        auto viewTensor = View(inputs[0], args->viewShape_,
                                               {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                               offset);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = ArgSort(viewTensor, args->dims_[0], args->descending_[0]);
                        Assemble(res, offset, outputs[0]);
                    }
                }
            }
        }
    }
}

class ArgSortOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ArgSortOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestArgSort, ArgSortOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<ArgSortOpMetaData>(
                             {ArgSortOperationExeFunc2Dims, ArgSortOperationExeFunc3Dims, ArgSortOperationExeFunc4Dims},
                             "ArgSort")));

TEST_P(ArgSortOperationTest, TestArgSort)
{
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    auto args = ArgSortOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data),
                                  GetValueByName<std::vector<int>>(test_data, "dims"),
                                  GetValueByName<std::vector<bool>>(test_data, "descending"));
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0].GetStorage()->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0].GetStorage()->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}
} // namespace
