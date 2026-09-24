/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_expandfunction_cascade_assemble_waw.cpp
 * \brief ExpandFunction UT for assemble WAW write-token propagation; RAW read-token via AddOperation.
 */

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "interface/tensor/contract_write_token.h"
#include "interface/tensor/irbuilder.h"
#include "ir/type.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

namespace npu::tile_fwk {
namespace {

constexpr uint16_t kTileDim = 64u;

void EnableSliceAndContract()
{
    config::SetPassOption(ENABLE_SLICE, true);
    ASSERT_TRUE(config::EnableSlice());
    EXPECT_EQ(config::GetSliceOpcode(), Opcode::OP_SLICE);
    EXPECT_EQ(config::GetContractOpcode(), Opcode::OP_CONTRACT);
}

bool ContainsToken(const std::vector<ir::VarPtr>& tokens, const ir::VarPtr& target)
{
    return std::find(tokens.begin(), tokens.end(), target) != tokens.end();
}

bool IsNormalToken(const ir::VarPtr& token)
{
    if (!token) {
        return false;
    }
    auto tokenType = std::dynamic_pointer_cast<const ir::TokenType>(token->GetType());
    return tokenType != nullptr && tokenType->kind_ == ir::TokenKind::NORMAL;
}

std::vector<Operation*> CollectOpsByOpcode(Function& function, Opcode opcode)
{
    std::vector<Operation*> ops;
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() == opcode) {
            ops.push_back(&op);
        }
    }
    return ops;
}

std::shared_ptr<AssembleOpAttribute> CreateZeroOffsetAssembleAttr()
{
    std::vector<int64_t> toOffset = {0, 0};
    std::vector<SymbolicScalar> dynOffset = {CreateTestScalarVar("sym0"), CreateTestScalarVar("sym1")};
    return std::make_shared<AssembleOpAttribute>(toOffset, dynOffset);
}

void InjectCascadeAssembleWriteTokens(const std::vector<Operation*>& assembleOps, ir::VarPtr& firstWriteNormal,
                                      ir::VarPtr& secondWriteNormal)
{
    ASSERT_EQ(assembleOps.size(), 2U);

    auto& ctx = IRContext::Get();
    ir::Span span = ir::Span::Unknown();
    auto firstWriteSemantic = ctx.MakeSemanticToken("out_w1", ir::TokenKind::WRITE, span);
    auto secondWriteSemantic = ctx.MakeSemanticToken("out_w2", ir::TokenKind::WRITE, span);
    firstWriteNormal = ctx.GetNormalToken(firstWriteSemantic);
    secondWriteNormal = ctx.GetNormalToken(secondWriteSemantic);
    ASSERT_NE(firstWriteNormal, nullptr);
    ASSERT_NE(secondWriteNormal, nullptr);

    assembleOps[0]->result_token_ = {firstWriteSemantic};
    assembleOps[1]->result_token_ = {secondWriteSemantic};
    assembleOps[1]->tokens_ = {firstWriteSemantic};
}

void InjectReadBeforeWriteAssembleTokens(Operation& assembleOp, ir::VarPtr& readNormal, ir::VarPtr& writeNormal)
{
    auto& ctx = IRContext::Get();
    ir::Span span = ir::Span::Unknown();
    auto readSemantic = ctx.MakeSemanticToken("buf_r", ir::TokenKind::READ, span);
    auto writeSemantic = ctx.MakeSemanticToken("buf_w", ir::TokenKind::WRITE, span);
    readNormal = ctx.GetNormalToken(readSemantic);
    writeNormal = ctx.GetNormalToken(writeSemantic);
    ASSERT_NE(readNormal, nullptr);
    ASSERT_NE(writeNormal, nullptr);

    assembleOp.result_token_ = {writeSemantic};
    assembleOp.tokens_ = {readSemantic};
}

void ExpectNoSemanticWriteTokensOnContract(const std::vector<Operation*>& contractOps)
{
    for (const auto* contractOp : contractOps) {
        for (const auto& token : contractOp->result_token_) {
            EXPECT_FALSE(IsWriteSemanticToken(token));
        }
        for (const auto& token : contractOp->tokens_) {
            EXPECT_FALSE(IsWriteSemanticToken(token));
        }
    }
}

} // namespace

class TestExpandFunctionAssembleTokenPropagation : public ::testing::Test {
protected:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        EnableSliceAndContract();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        TileShape::Current().SetVecTile(kTileDim, kTileDim);
    }

    void TearDown() override { Program::GetInstance().Reset(); }
};

/*
 * Cascading assemble WAW on the same outCast:
 *   src1 -> assemble -> out
 *   src2 -> assemble -> out   (depends on first write token)
 *
 * Run ExpandFunction only (with enable_slice=true so slice/contract path is active)
 * and verify write semantic tokens propagate to contract ops
 * as paired NORMAL tokens (result_token_ for producer write, tokens_ for consumer write).
 */
TEST_F(TestExpandFunctionAssembleTokenPropagation, CascadeAssembleWriteAfterWriteTokenPropagation)
{
    EnableSliceAndContract();

    auto function = std::make_shared<Function>(Program::GetInstance(), "CascadeAssembleWAW", "CascadeAssembleWAW",
                                               nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> shape = {kTileDim, kTileDim};
    auto src1 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto src2 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto assembleAttr = CreateZeroOffsetAssembleAttr();
    auto& firstAssemble = PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {src1}, {outCast});
    auto& secondAssemble = PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {src2}, {outCast});
    firstAssemble.SetOpAttribute(assembleAttr);
    secondAssemble.SetOpAttribute(assembleAttr);
    firstAssemble.tileShape_.SetVecTile({kTileDim, kTileDim});
    secondAssemble.tileShape_.SetVecTile({kTileDim, kTileDim});

    function->inCasts_.push_back(src1);
    function->inCasts_.push_back(src2);
    function->outCasts_.push_back(outCast);
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    auto assembleOps = CollectOpsByOpcode(*function, Opcode::OP_ASSEMBLE);
    ir::VarPtr firstWriteNormal;
    ir::VarPtr secondWriteNormal;
    InjectCascadeAssembleWriteTokens(assembleOps, firstWriteNormal, secondWriteNormal);

    ExpandFunction expandPass;
    EXPECT_EQ(expandPass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetGraphType(), GraphType::TILE_GRAPH);

    EXPECT_TRUE(CollectOpsByOpcode(*function, Opcode::OP_ASSEMBLE).empty());

    const auto sliceOpcode = config::GetSliceOpcode();
    const auto contractOpcode = config::GetContractOpcode();
    auto sliceOps = CollectOpsByOpcode(*function, sliceOpcode);
    auto contractOps = CollectOpsByOpcode(*function, contractOpcode);
    EXPECT_EQ(sliceOpcode, Opcode::OP_SLICE);
    EXPECT_EQ(contractOpcode, Opcode::OP_CONTRACT);
    ASSERT_EQ(sliceOps.size(), 2U);
    ASSERT_EQ(contractOps.size(), 2U);

    auto& firstContract = *contractOps[0];
    auto& secondContract = *contractOps[1];

    ASSERT_EQ(firstContract.result_token_.size(), 1U);
    EXPECT_EQ(firstContract.result_token_.front(), firstWriteNormal);
    EXPECT_TRUE(IsNormalToken(firstContract.result_token_.front()));
    EXPECT_FALSE(ContainsToken(firstContract.tokens_, firstWriteNormal));

    ASSERT_EQ(secondContract.result_token_.size(), 1U);
    EXPECT_EQ(secondContract.result_token_.front(), secondWriteNormal);
    EXPECT_TRUE(IsNormalToken(secondContract.result_token_.front()));
    ASSERT_EQ(secondContract.tokens_.size(), 1U);
    EXPECT_EQ(secondContract.tokens_.front(), firstWriteNormal);
    EXPECT_TRUE(IsNormalToken(secondContract.tokens_.front()));

    ExpectNoSemanticWriteTokensOnContract(contractOps);
}

/*
 * Write-after-read (WAR) on outCast via TiledAssemble (AddRawOperation path):
 * assemble.tokens_ carries consumed READ semantic; ExpandFunction should place its NORMAL pair
 * on contract.tokens_ (input dependency), while this write's NORMAL goes to contract.result_token_.
 */
TEST_F(TestExpandFunctionAssembleTokenPropagation, AssembleWriteAfterReadTokenOnContractInput)
{
    EnableSliceAndContract();

    auto function = std::make_shared<Function>(Program::GetInstance(), "AssembleWriteAfterRead",
                                               "AssembleWriteAfterRead", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> shape = {kTileDim, kTileDim};
    auto src = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto assembleAttr = CreateZeroOffsetAssembleAttr();
    auto& assembleOp = PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {src}, {outCast});
    assembleOp.SetOpAttribute(assembleAttr);
    assembleOp.tileShape_.SetVecTile({kTileDim, kTileDim});

    function->inCasts_.push_back(src);
    function->outCasts_.push_back(outCast);
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    ir::VarPtr readNormal;
    ir::VarPtr writeNormal;
    InjectReadBeforeWriteAssembleTokens(assembleOp, readNormal, writeNormal);

    ExpandFunction expandPass;
    EXPECT_EQ(expandPass.RunOnFunction(*function), SUCCESS);

    auto contractOps = CollectOpsByOpcode(*function, config::GetContractOpcode());
    ASSERT_EQ(contractOps.size(), 1U);

    auto& contract = *contractOps[0];
    ASSERT_EQ(contract.result_token_.size(), 1U);
    EXPECT_EQ(contract.result_token_.front(), writeNormal);
    ASSERT_EQ(contract.tokens_.size(), 1U);
    EXPECT_EQ(contract.tokens_.front(), readNormal);
    ExpectNoSemanticWriteTokensOnContract(contractOps);
}

/*
 * Read-after-write (RAW): parent tensor carries readToken_ from InferTokenPass;
 * AddOperation inserts input slice and propagates read NORMAL to slice.result_token_.
 */
TEST_F(TestExpandFunctionAssembleTokenPropagation, ReadAfterWriteSliceTokenViaAddOperation)
{
    EnableSliceAndContract();

    auto function = std::make_shared<Function>(Program::GetInstance(), "ReadAfterWriteSliceToken",
                                               "ReadAfterWriteSliceToken", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> shape = {kTileDim, kTileDim};
    auto parent = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& ctx = IRContext::Get();
    ir::Span span = ir::Span::Unknown();
    auto readSemantic = ctx.MakeSemanticToken("out_r", ir::TokenKind::READ, span);
    auto readNormal = ctx.GetNormalToken(readSemantic);
    ASSERT_NE(readNormal, nullptr);
    parent->SetReadToken(readSemantic);

    auto inputTile = parent->View(*function, shape, {0, 0});
    function->AddOperation(Opcode::OP_EXP, {inputTile}, {output});

    const auto sliceOpcode = config::GetSliceOpcode();
    auto sliceOps = CollectOpsByOpcode(*function, sliceOpcode);
    ASSERT_EQ(sliceOps.size(), 1U);
    ASSERT_EQ(sliceOps[0]->result_token_.size(), 1U);
    EXPECT_EQ(sliceOps[0]->result_token_.front(), readNormal);
    EXPECT_TRUE(IsNormalToken(sliceOps[0]->result_token_.front()));
}

} // namespace npu::tile_fwk
