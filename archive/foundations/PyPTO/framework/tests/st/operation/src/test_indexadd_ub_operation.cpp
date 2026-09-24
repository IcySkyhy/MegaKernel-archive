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
 * \file test_indexadd_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct IndexAddUBOpFuncArgs : public OpFuncArgs {
    IndexAddUBOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, int axis,
                         Element& alpha)
        : viewShape_(viewShape), tileShape_(tileShape), axis_(axis), alpha_(alpha)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    int axis_;
    Element alpha_;
};

struct IndexAddUBOpMetaData {
    explicit IndexAddUBOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

// Tensor IndexAddUB(const Tensor &self, const Tensor &src, const Tensor &indices, int axis, const Element &alpha)
static void IndexAddUBOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        SymbolicScalar self_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar self_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar src_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[1].GetShape()[1];
        SymbolicScalar idxDim = inputs[2].GetShape()[0];

        auto args = static_cast<const IndexAddUBOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis >= 0 ? axis : axis + inputs[0].GetShape().size();
        std::vector<int64_t> viewShape = args->viewShape_;

        ASSERT(idxDim == inputs[1].GetShape()[axis]);
        ASSERT(viewShape[axis] >= std::max(inputs[0].GetShape()[axis], idxDim)); // 确保viewshape按最大的切

        const int64_t firstViewShape = viewShape[0];
        const int64_t secondViewShape = viewShape[1];
        const int64_t bloop = CeilDiv(src_firstDim, firstViewShape);
        const int64_t sloop = CeilDiv(src_secondDim, secondViewShape);

        // selfshape和srcshape的axis轴都不切
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape};
                std::vector<SymbolicScalar> selfValidShape = {
                    std::min(self_firstDim - bIdx * firstViewShape, firstViewShape),
                    std::min(self_secondDim - sIdx * secondViewShape, secondViewShape)};
                std::vector<SymbolicScalar> srcValidShape = {
                    std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                    std::min(src_secondDim - sIdx * secondViewShape, secondViewShape)};
                auto selfTensor = View(inputs[0], viewShape, selfValidShape, offset);
                auto srcTensor = View(inputs[1], viewShape, srcValidShape, offset);
                auto idxTensor = View(inputs[2], {viewShape[axis]}, {srcValidShape[axis]},
                                      {offset[axis]}); // idxshape只有在axis轴才切

                TileShape::Current().SetVecTile(args->tileShape_);
                auto dst = IndexAddUB(selfTensor, srcTensor, idxTensor, args->axis_, args->alpha_);
                Assemble(dst, offset, outputs[0]); // offset[axis]=0
            }
        }
    }
}

static void IndexAddUBOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        SymbolicScalar self_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar self_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar self_thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar src_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[1].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[1].GetShape()[2];
        SymbolicScalar idxDim = inputs[2].GetShape()[0];
        auto args = static_cast<const IndexAddUBOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis >= 0 ? axis : axis + inputs[0].GetShape().size();
        std::vector<int64_t> viewShape = args->viewShape_;

        ASSERT(idxDim == inputs[1].GetShape()[axis]);
        ASSERT(viewShape[axis] >= std::max(inputs[0].GetShape()[axis], idxDim)); // 确保viewshape按最大的切

        const int64_t firstViewShape = viewShape[0];
        const int64_t secondViewShape = viewShape[1];
        const int64_t thirdViewShape = viewShape[2];

        const int64_t bloop = CeilDiv(src_firstDim, firstViewShape);
        const int64_t sloop = CeilDiv(src_secondDim, secondViewShape);
        const int64_t nloop = CeilDiv(src_thirdDim, thirdViewShape);

        // selfshape的axis轴不切，因此测试用例需要保证viewshape[axis] = selfshape[axis]
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                          nIdx * thirdViewShape};
                    std::vector<SymbolicScalar> selfValidShape = {
                        std::min(self_firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(self_secondDim - sIdx * secondViewShape, secondViewShape),
                        std::min(self_thirdDim - nIdx * thirdViewShape, thirdViewShape)};
                    std::vector<SymbolicScalar> srcValidShape = {
                        std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(src_secondDim - sIdx * secondViewShape, secondViewShape),
                        std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape)};
                    auto selfTensor = View(inputs[0], viewShape, selfValidShape, offset);
                    auto srcTensor = View(inputs[1], viewShape, srcValidShape, offset);
                    auto idxTensor = View(inputs[2], {viewShape[axis]}, {srcValidShape[axis]},
                                          {offset[axis]}); // idxshape只有在axis轴才切

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto dst = IndexAddUB(selfTensor, srcTensor, idxTensor, args->axis_, args->alpha_);
                    Assemble(dst, offset, outputs[0]); // offset[axis]=0
                }
            }
        }
    }
}

static void IndexAddUBOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        SymbolicScalar self_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar self_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar self_thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar self_forthDim = inputs[0].GetShape()[3];
        SymbolicScalar src_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[1].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[1].GetShape()[2];
        SymbolicScalar src_forthDim = inputs[1].GetShape()[3];
        SymbolicScalar idxDim = inputs[2].GetShape()[0];
        auto args = static_cast<const IndexAddUBOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis >= 0 ? axis : axis + inputs[0].GetShape().size();
        std::vector<int64_t> viewShape = args->viewShape_;

        ASSERT(idxDim == inputs[1].GetShape()[axis]);
        ASSERT(viewShape[axis] >= std::max(inputs[0].GetShape()[axis], idxDim)); // 确保viewshape按最大的切

        const int64_t firstViewShape = viewShape[0];
        const int64_t secondViewShape = viewShape[1];
        const int64_t thirdViewShape = viewShape[2];
        const int64_t forthViewShape = viewShape[3];

        const int64_t bloop = CeilDiv(src_firstDim, firstViewShape);
        const int64_t sloop = CeilDiv(src_secondDim, secondViewShape);
        const int64_t nloop = CeilDiv(src_thirdDim, thirdViewShape);
        const int64_t qloop = CeilDiv(src_forthDim, forthViewShape);

        // selfshape的axis轴不切，因此测试用例需要保证viewshape[axis] = selfshape[axis]
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(0, qloop, 1))
                    {
                        std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                              nIdx * thirdViewShape, qIdx * forthViewShape};
                        std::vector<SymbolicScalar> selfValidShape = {
                            std::min(self_firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(self_secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(self_thirdDim - nIdx * thirdViewShape, thirdViewShape),
                            std::min(self_forthDim - qIdx * forthViewShape, forthViewShape)};
                        std::vector<SymbolicScalar> srcValidShape = {
                            std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(src_secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape),
                            std::min(src_forthDim - qIdx * forthViewShape, forthViewShape)};
                        auto selfTensor = View(inputs[0], viewShape, selfValidShape, offset);
                        auto srcTensor = View(inputs[1], viewShape, srcValidShape, offset);
                        auto idxTensor = View(inputs[2], {viewShape[axis]}, {srcValidShape[axis]},
                                              {offset[axis]}); // idxshape只有在axis轴才切

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto dst = IndexAddUB(selfTensor, srcTensor, idxTensor, args->axis_, args->alpha_);
                        Assemble(dst, offset, outputs[0]); // offset[axis]=0
                    }
                }
            }
        }
    }
}

static void IndexAddUBOperationExeFunc5Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        SymbolicScalar self_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar self_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar self_thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar self_forthDim = inputs[0].GetShape()[3];
        SymbolicScalar self_fifthDim = inputs[0].GetShape()[4];
        SymbolicScalar src_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[1].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[1].GetShape()[2];
        SymbolicScalar src_forthDim = inputs[1].GetShape()[3];
        SymbolicScalar src_fifthDim = inputs[1].GetShape()[4];
        SymbolicScalar idxDim = inputs[2].GetShape()[0];
        auto args = static_cast<const IndexAddUBOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis >= 0 ? axis : axis + inputs[0].GetShape().size();
        std::vector<int64_t> viewShape = args->viewShape_;
        ASSERT(idxDim == inputs[1].GetShape()[axis]);
        ASSERT(viewShape[axis] >= std::max(inputs[0].GetShape()[axis], idxDim)); // 确保viewshape按最大的切
        const int64_t firstViewShape = viewShape[0];
        const int64_t secondViewShape = viewShape[1];
        const int64_t thirdViewShape = viewShape[2];
        const int64_t forthViewShape = viewShape[3];
        const int64_t fifthViewShape = viewShape[4];
        const int64_t loop[] = {CeilDiv(src_firstDim, firstViewShape), CeilDiv(src_secondDim, secondViewShape),
                                CeilDiv(src_thirdDim, thirdViewShape), CeilDiv(src_forthDim, forthViewShape),
                                CeilDiv(src_fifthDim, fifthViewShape)};
        // selfshape的axis轴不切，因此测试用例需要保证viewshape[axis] = selfshape[axis]
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[0]))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[1]))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[2]))
                {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(loop[3]))
                    {
                        LOOP("LOOP_L4_rIdx", FunctionType::DYNAMIC_LOOP, rIdx, LoopRange(loop[4]))
                        {
                            std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                                  nIdx * thirdViewShape, qIdx * forthViewShape,
                                                                  rIdx * fifthViewShape};
                            std::vector<SymbolicScalar> selfValidShape = {
                                std::min(self_firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(self_secondDim - sIdx * secondViewShape, secondViewShape),
                                std::min(self_thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                std::min(self_forthDim - qIdx * forthViewShape, forthViewShape),
                                std::min(self_fifthDim - rIdx * fifthViewShape, fifthViewShape)};
                            std::vector<SymbolicScalar> srcValidShape = {
                                std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(src_secondDim - sIdx * secondViewShape, secondViewShape),
                                std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                std::min(src_forthDim - qIdx * forthViewShape, forthViewShape),
                                std::min(src_fifthDim - rIdx * fifthViewShape, fifthViewShape)};
                            auto selfTensor = View(inputs[0], viewShape, selfValidShape, offset);
                            auto srcTensor = View(inputs[1], viewShape, srcValidShape, offset);
                            auto idxTensor = View(inputs[2], {viewShape[axis]}, {srcValidShape[axis]},
                                                  {offset[axis]}); // idxshape只有在axis轴才切
                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto dst = IndexAddUB(selfTensor, srcTensor, idxTensor, args->axis_, args->alpha_);
                            Assemble(dst, offset, outputs[0]); // offset[axis]=0
                        }
                    }
                }
            }
        }
    }
}

class IndexAddUBOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<IndexAddUBOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestIndexAddUB, IndexAddUBOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<IndexAddUBOpMetaData>(
                             {IndexAddUBOperationExeFunc2Dims, IndexAddUBOperationExeFunc3Dims,
                              IndexAddUBOperationExeFunc4Dims, IndexAddUBOperationExeFunc5Dims},
                             "IndexAddUB")));

TEST_P(IndexAddUBOperationTest, TestIndexAddUB)
{
    auto test_data = GetParam().test_data_;
    auto axis = static_cast<CastMode>(GetValueByName<int>(test_data, "axis"));
    nlohmann::json data = test_data;
    float value = 1.0;
    if (data.find("alpha") == data.end()) {
        data = test_data.at("params");
    }
    if (data.find("alpha") != data.end()) {
        auto alpha = data.at("alpha");
        if (alpha.is_number()) {
            value = alpha;
        } else if (alpha.is_string()) {
            try {
                value = std::stof(alpha.get<std::string>());
            } catch (const std::exception& e) {
                value = 1.0;
            }
        }
    }
    Element alp(npu::tile_fwk::DT_FP32, value);
    auto args = IndexAddUBOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), axis, alp);
    auto testCase = CreateTestCaseDesc<IndexAddUBOpMetaData>(GetParam(), &args);
    std::vector<OpFunc> opFuncs = {IndexAddUBOperationExeFunc2Dims, IndexAddUBOperationExeFunc3Dims,
                                   IndexAddUBOperationExeFunc4Dims, IndexAddUBOperationExeFunc5Dims};
    testCase.opFunc = opFuncs[GetViewShape(test_data).size() - 2];
    TestExecutor::runTest(testCase);
}
} // namespace
