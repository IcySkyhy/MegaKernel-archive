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
 * \file test_Cmps_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {

struct CmpsOpFuncArgs : public OpFuncArgs {
    CmpsOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, OpType opType,
                   OutType modeType, const Element& scalarVal)
        : viewShape_(viewShape), tileShape_(tileShape), cmpOp_(opType), cmpMode_(modeType), scalarVal_(scalarVal)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    OpType cmpOp_;
    OutType cmpMode_;
    Element scalarVal_;
};

// 测试元数据结构体
struct CmpsOpMetaData {
    explicit CmpsOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void CmpsOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                      const OpFuncArgs* opArgs)
{
    auto args = static_cast<const CmpsOpFuncArgs*>(opArgs);
    SymbolicScalar firstDim = inputs[0].GetShape()[0];
    SymbolicScalar secondDim = inputs[0].GetShape()[1];
    const int firstViewShape = args->viewShape_[0];
    const int secondViewShape = args->viewShape_[1];

    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewShape), 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDiv(secondDim, secondViewShape), 1))
            {
                auto tileTensor = View(inputs[0], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Compare(tileTensor, args->scalarVal_, args->cmpOp_, args->cmpMode_);
                auto lastOffset = (args->cmpMode_ == OutType::BIT) ? (sIdx * secondViewShape / 8) :
                                                                     sIdx * secondViewShape;
                Assemble(res, {bIdx * firstViewShape, lastOffset}, outputs[0]);
            }
        }
    }
}

static void CmpsOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                      const OpFuncArgs* opArgs)
{
    auto args = static_cast<const CmpsOpFuncArgs*>(opArgs);
    SymbolicScalar firstDim = inputs[0].GetShape()[0];
    SymbolicScalar secondDim = inputs[0].GetShape()[1];
    SymbolicScalar thirdDim = inputs[0].GetShape()[2];
    const int firstViewShape = args->viewShape_[0];
    const int secondViewShape = args->viewShape_[1];
    const int thirdViewShape = args->viewShape_[2];

    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewShape), 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDiv(secondDim, secondViewShape), 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx,
                     LoopRange(0, CeilDiv(thirdDim, thirdViewShape), 1))
                {
                    auto tileTensor = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                                           {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                           {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Compare(tileTensor, args->scalarVal_, args->cmpOp_, args->cmpMode_);
                    auto lastOffset = (args->cmpMode_ == OutType::BIT) ? (nIdx * thirdViewShape / 8) :
                                                                         nIdx * thirdViewShape;
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, lastOffset}, outputs[0]);
                }
            }
        }
    }
}

static void CmpsOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                      const OpFuncArgs* opArgs)
{
    auto args = static_cast<const CmpsOpFuncArgs*>(opArgs);
    SymbolicScalar firstDim = inputs[0].GetShape()[0];
    SymbolicScalar secondDim = inputs[0].GetShape()[1];
    SymbolicScalar thirdDim = inputs[0].GetShape()[2];
    SymbolicScalar fourthDim = inputs[0].GetShape()[3];
    const int firstViewShape = args->viewShape_[0];
    const int secondViewShape = args->viewShape_[1];
    const int thirdViewShape = args->viewShape_[2];
    const int fourthViewShape = args->viewShape_[3];
    FUNCTION("main", {inputs[0]}, {outputs[0]})
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
                        auto tileTensor = View(inputs[0],
                                               {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                               {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                               {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                nIdx * fourthViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Compare(tileTensor, args->scalarVal_, args->cmpOp_, args->cmpMode_);
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

class CmpsOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<CmpsOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestCmps, CmpsOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<CmpsOpMetaData>(
                             {CmpsOperationExeFunc2Dims, CmpsOperationExeFunc3Dims, CmpsOperationExeFunc4Dims},
                             "Cmps")));

TEST_P(CmpsOperationTest, TestCmps)
{
    auto test_data = GetParam().test_data_;
    std::string opStr = GetValueByName<std::string>(test_data, "compare_op");
    std::string modeStr = GetValueByName<std::string>(test_data, "mode");

    auto dtype = GetDataType(GetValueByName<std::string>(test_data, "scalar_dtype"));
    float scalarFloatVal = GetValueByName<float>(test_data, "scalar");
    Element scalarVal(dtype, scalarFloatVal);

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

    auto args = CmpsOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), cmpOp, cmpMode, scalarVal);
    auto testCase = CreateTestCaseDesc<CmpsOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
