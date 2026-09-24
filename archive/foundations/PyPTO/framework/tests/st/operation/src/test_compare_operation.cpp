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
 * \file test_Compare_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {

struct CompareOpFuncArgs : public OpFuncArgs {
    CompareOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, OpType opType,
                      OutType modeType)
        : viewShape_(viewShape), tileShape_(tileShape), cmpOp_(opType), cmpMode_(modeType)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    OpType cmpOp_;
    OutType cmpMode_;
};

// 测试元数据结构体
struct CompareOpMetaData {
    explicit CompareOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void CompareOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    auto args = static_cast<const CompareOpFuncArgs*>(opArgs);
    SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
    SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
    const int firstViewShape = args->viewShape_[0];
    const int secondViewShape = args->viewShape_[1];
    const int broadcastFlag = 1;
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewShape), 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDiv(secondDim, secondViewShape), 1))
            {
                Tensor tileTensor0, tileTensor1;
                if (inputs[1].GetShape()[1] == broadcastFlag && inputs[0].GetShape()[1] != broadcastFlag) {
                    tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                    tileTensor1 = View(inputs[1], {firstViewShape, 1},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1},
                                       {bIdx * firstViewShape, 0});
                } else if (inputs[1].GetShape()[0] == broadcastFlag && inputs[0].GetShape()[0] != broadcastFlag) {
                    tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                    tileTensor1 = View(inputs[1], {1, secondViewShape},
                                       {1, std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {0, sIdx * secondViewShape});
                }
                // 第一个张量在第二维广播
                else if (inputs[0].GetShape()[1] == broadcastFlag && inputs[1].GetShape()[1] != broadcastFlag) {
                    tileTensor0 = View(inputs[0], {firstViewShape, 1},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1},
                                       {bIdx * firstViewShape, 0});
                    tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                }
                // 第一个张量在第一维广播
                else if (inputs[0].GetShape()[0] == broadcastFlag && inputs[1].GetShape()[0] != broadcastFlag) {
                    tileTensor0 = View(inputs[0], {1, secondViewShape},
                                       {1, std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {0, sIdx * secondViewShape});
                    tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                } else {
                    tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                    tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                }
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Compare(tileTensor0, tileTensor1, args->cmpOp_, args->cmpMode_);
                auto lastOffset = (args->cmpMode_ == OutType::BIT) ? (sIdx * secondViewShape / 8) :
                                                                     sIdx * secondViewShape;
                Assemble(res, {bIdx * firstViewShape, lastOffset}, outputs[0]);
            }
        }
    }
}

static void CompareOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    // 解析参数
    auto args = static_cast<const CompareOpFuncArgs*>(opArgs);
    SymbolicScalar firstDim = inputs[0].GetShape()[0];
    SymbolicScalar secondDim = inputs[0].GetShape()[1];
    SymbolicScalar thirdDim = inputs[0].GetShape()[2];
    const int firstViewShape = args->viewShape_[0];
    const int secondViewShape = args->viewShape_[1];
    const int thirdViewShape = args->viewShape_[2];
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewShape), 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDiv(secondDim, secondViewShape), 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx,
                     LoopRange(0, CeilDiv(thirdDim, thirdViewShape), 1))
                {
                    auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                             std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                             std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    auto tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape, thirdViewShape},
                                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                             std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                             std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Compare(tileTensor0, tileTensor1, args->cmpOp_, args->cmpMode_);
                    auto lastOffset = (args->cmpMode_ == OutType::BIT) ? (nIdx * thirdViewShape / 8) :
                                                                         nIdx * thirdViewShape;
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, lastOffset}, outputs[0]);
                }
            }
        }
    }
}
static void CompareOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    // 解析参数
    auto args = static_cast<const CompareOpFuncArgs*>(opArgs);
    SymbolicScalar firstDim = inputs[0].GetShape()[0];
    SymbolicScalar secondDim = inputs[0].GetShape()[1];
    SymbolicScalar thirdDim = inputs[0].GetShape()[2];
    SymbolicScalar fourthDim = inputs[0].GetShape()[3];
    const int firstViewShape = args->viewShape_[0];
    const int secondViewShape = args->viewShape_[1];
    const int thirdViewShape = args->viewShape_[2];
    const int fourthViewShape = args->viewShape_[3];
    const int broadcastFlag = 1;
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewShape), 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDiv(secondDim, secondViewShape), 1))
            {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx,
                     LoopRange(0, CeilDiv(thirdDim, thirdViewShape), 1))
                {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx,
                         LoopRange(0, CeilDiv(fourthDim, fourthViewShape), 1))
                    {
                        Tensor tileTensor0, tileTensor1;
                        if (inputs[1].GetShape()[2] == broadcastFlag && inputs[0].GetShape()[2] != broadcastFlag) {
                            tileTensor0 = View(inputs[0],
                                               {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                               {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                               {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                nIdx * fourthViewShape});
                            tileTensor1 = View(
                                inputs[1], {firstViewShape, secondViewShape, 1, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(secondDim - sIdx * secondViewShape, secondViewShape), 1,
                                 std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, 0, nIdx * fourthViewShape});
                        } else {
                            tileTensor0 = View(inputs[0],
                                               {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                               {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                               {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                nIdx * fourthViewShape});
                            tileTensor1 = View(inputs[1],
                                               {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                               {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                               {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                nIdx * fourthViewShape});
                        }
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Compare(tileTensor0, tileTensor1, args->cmpOp_, args->cmpMode_);
                        auto lastOffset = (args->cmpMode_ == OutType::BIT) ? (nIdx * fourthViewShape / 8) :
                                                                             nIdx * fourthViewShape;
                        Assemble(res,
                                 {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape, lastOffset},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

class CompareOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<CompareOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestCompare, CompareOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<CompareOpMetaData>(
                             {CompareOperationExeFunc2Dims, CompareOperationExeFunc3Dims, CompareOperationExeFunc4Dims},
                             "Compare")));

TEST_P(CompareOperationTest, TestCompare)
{
    auto test_data = GetParam().test_data_;
    std::string opStr = GetValueByName<std::string>(test_data, "compare_op");
    std::string modeStr = GetValueByName<std::string>(test_data, "mode");
    static const std::unordered_map<std::string, OpType> opMap = {{"eq", OpType::EQ}, {"ne", OpType::NE},
                                                                  {"lt", OpType::LT}, {"le", OpType::LE},
                                                                  {"gt", OpType::GT}, {"ge", OpType::GE}};
    auto opIt = opMap.find(opStr);
    if (opIt == opMap.end()) {
        throw std::invalid_argument("Unsupported OpType: " + opStr);
    }
    OpType cmpOp = opIt->second;
    static const std::unordered_map<std::string, OutType> modeMap = {{"bool", OutType::BOOL}, {"bit", OutType::BIT}};
    auto modeIt = modeMap.find(modeStr);
    if (modeIt == modeMap.end()) {
        throw std::invalid_argument("Unsupported OutType: " + modeStr);
    }
    OutType cmpMode = modeIt->second;
    auto args = CompareOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), cmpOp, cmpMode);
    auto testCase = CreateTestCaseDesc<CompareOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
