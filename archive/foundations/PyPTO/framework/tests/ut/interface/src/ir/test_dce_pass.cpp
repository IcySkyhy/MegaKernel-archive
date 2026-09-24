/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <algorithm>
#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/passes.h"
#include "ir/type.h"

#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;

namespace {
static constexpr int64_t TILE = 16;

ir::Span Sp() { return ir::Span("test_tensor_move", 1, 1); }

// Minimal builder for an IR function whose body is a sequence of pre-built
// statements and whose params are logical tensors (so AggressiveDCE recognises
// them via ir::As<LogicalTensorType>). Modelled on test_ir_func_builder.cpp.
struct IrFuncSetup {
    npu::tile_fwk::IRBuilder builder;
    std::shared_ptr<npu::tile_fwk::Function> fwkFunc;
    LogicalTensors params;
    std::vector<ir::StmtPtr> stmts;

    explicit IrFuncSetup(const std::string& name)
    {
        fwkFunc = std::make_shared<npu::tile_fwk::Function>(Program::GetInstance(), name + "_magic", name, nullptr);
        fwkFunc->SetFunctionType(FunctionType::DYNAMIC);
        fwkFunc->SetGraphType(GraphType::TENSOR_GRAPH);
        Program::GetInstance().InsertFuncToFunctionMap(fwkFunc->GetMagicName(), fwkFunc);
        Program::GetInstance().SetCurrentFunction(fwkFunc.get());
    }

    LogicalTensorPtr MakeParam(const std::string& name)
    {
        auto lt = builder.CreateTensorVar(*fwkFunc, DT_FP32, {TILE, TILE}, TileOpFormat::TILEOP_ND, name);
        params.push_back(lt);
        return lt;
    }

    ir::FunctionPtr BuildIrFunction(const std::string& name)
    {
        auto body = std::make_shared<ir::SeqStmts>(stmts, Sp());
        std::vector<ir::VarPtr> irParams;
        for (auto& p : params) {
            irParams.push_back(std::static_pointer_cast<const ir::Var>(p));
        }
        return std::make_shared<ir::Function>(name, irParams, std::vector<ir::TypePtr>{}, body, Sp());
    }
};
} // namespace

TEST(DcePassTest, TensorMove)
{
    IrFuncSetup setup("TensorMove");
    auto a = setup.MakeParam("a");
    auto b = setup.MakeParam("b"); // input b: target of the tensor "move" at body[2]

    auto& builder = setup.builder;
    const auto span = Sp();
    const auto fp32 = a->GetType();

    // body[0]: last = pypto.full(a.shape, 1.0, a.dtype)
    auto last = builder.CreateTensorVar(DT_FP32, {32, 32}, TileOpFormat::TILEOP_ND, "last");
    auto fullStmt = builder.CreateTensorOpStmt(std::vector<ir::VarPtr>{last}, nullptr, "FULL",
                                               std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(a)},
                                               std::vector<ir::VarPtr>{},
                                               std::vector<std::pair<std::string, std::any>>{}, span);
    setup.stmts.push_back(fullStmt);

    // body[1]: for _ in pypto.loop(10): ...  -- "last" is the loop-carried value.
    auto loopVar = std::make_shared<ir::Var>("i", std::make_shared<ir::ScalarType>(ir::DataType::INT64), span);
    auto zero = std::make_shared<ir::ConstInt>(0, ir::DataType::INT64, span);
    auto ten = std::make_shared<ir::ConstInt>(10, ir::DataType::INT64, span);
    auto one = std::make_shared<ir::ConstInt>(1, ir::DataType::INT64, span);
    auto lastIterArg = builder.CreateIterArg("last", fp32, std::static_pointer_cast<const ir::Expr>(last), span);
    auto aOut = IRContext::Get().MakeVar("a", fp32, span);
    auto yield = builder.CreateYieldStmt(std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(last)},
                                         span);
    auto forStmt = builder.CreateForStmt(loopVar, zero, ten, one, std::vector<ir::IterArgPtr>{lastIterArg}, yield,
                                         std::vector<ir::VarPtr>{aOut}, span);
    setup.stmts.push_back(forStmt);

    // body[2]: b[:] = a + 1  -> ADDS writing into input b (result name "b_0").
    auto bResult = builder.CreateTensorVar(DT_FP32, {32, 32}, TileOpFormat::TILEOP_ND, "b_0"); // second "b" -> "b_0"
    bResult->tensor->memoryId = b->tensor->memoryId;
    auto addStmt = builder.CreateTensorOpStmt(std::vector<ir::VarPtr>{bResult}, nullptr, "ADDS",
                                              std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(a)},
                                              std::vector<ir::VarPtr>{},
                                              std::vector<std::pair<std::string, std::any>>{}, span);
    setup.stmts.push_back(addStmt);

    auto irFunc = setup.BuildIrFunction("TensorMove");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto outProg = pypto::ir::pass::AggressiveDCE()(irProg);
    auto& outFunc = outProg->functions_.at("TensorMove");

    ASSERT_EQ(outFunc->body_->stmts_.size(), 3u) << "FULL, ForStmt and the b-move ADDS all survive DCE";
    const auto& stmts = outFunc->body_->stmts_;

    // body[1] is the ForStmt carrying "last" as a loop iter_arg.
    auto forStmtOut = std::dynamic_pointer_cast<const ir::ForStmt>(stmts[1]);
    ASSERT_NE(forStmtOut, nullptr);
    std::vector<std::string> iterNames;
    for (const auto& iterArg : forStmtOut->iterArgs_) {
        iterNames.push_back(iterArg->iterVar_->name_);
    }
    EXPECT_NE(std::find(iterNames.begin(), iterNames.end(), "last"), iterNames.end())
        << "'last' must remain a loop-carried iter_arg";

    // body[2] is the ADDS that moves a + 1 into input b.
    auto addOut = std::dynamic_pointer_cast<const ir::TensorOpStmt>(stmts[2]);
    ASSERT_NE(addOut, nullptr);
    EXPECT_EQ(addOut->opcode_, "ADDS");
    ASSERT_EQ(addOut->result_.size(), 1u);
    EXPECT_EQ(addOut->result_[0]->name_, "b_0");
}

TEST(DcePassTest, OpDumpSurvivesAggressiveDce)
{
    IrFuncSetup setup("OpDumpSurvives");
    auto a = setup.MakeParam("a");

    auto& builder = setup.builder;
    const auto span = Sp();

    // Dead ADD with no consumers: should be removed by DCE.
    auto dead = builder.CreateTensorVar(DT_FP32, {TILE, TILE}, TileOpFormat::TILEOP_ND, "dead");
    auto deadAdd = builder.CreateTensorOpStmt(std::vector<ir::VarPtr>{dead}, nullptr, "ADDS",
                                              std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(a)},
                                              std::vector<ir::VarPtr>{},
                                              std::vector<std::pair<std::string, std::any>>{}, span);
    setup.stmts.push_back(deadAdd);

    // pass_verify_print / pass_verify_save: OP_PRINT emitted as "OP_DUMP", no results.
    auto dumpStmt = builder.CreateTensorOpStmt(std::vector<ir::VarPtr>{}, nullptr, "OP_DUMP",
                                               std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(a)},
                                               std::vector<ir::VarPtr>{},
                                               std::vector<std::pair<std::string, std::any>>{}, span);
    setup.stmts.push_back(dumpStmt);

    auto irFunc = setup.BuildIrFunction("OpDumpSurvives");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto outProg = pypto::ir::pass::AggressiveDCE()(irProg);
    auto& outFunc = outProg->functions_.at("OpDumpSurvives");

    ASSERT_EQ(outFunc->body_->stmts_.size(), 1u) << "Only OP_DUMP should survive DCE";
    auto dumpOut = std::dynamic_pointer_cast<const ir::TensorOpStmt>(outFunc->body_->stmts_[0]);
    ASSERT_NE(dumpOut, nullptr);
    EXPECT_EQ(dumpOut->opcode_, "OP_DUMP");
}
