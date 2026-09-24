/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_ir_func_builder.cpp
 * \brief Unit tests for RootFunctionBuilder::BuildPathFuncSlotScope via pass::CreateRootFunctions.
 *
 * Focuses on constructAssembleSlotList correctness:
 *   - Deduplication of assemble ops targeting the same intermediate tensor
 *   - Mixed scenario: function param excluded, intermediate tensor kept
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/passes.h"

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/tensor_slot.h"
#include "interface/utils/id_gen.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;

namespace {
static constexpr int64_t TILE = 16;

ir::Span Sp() { return ir::Span("test_ir_func_builder", 1, 1); }

struct IrFuncSetup {
    npu::tile_fwk::IRBuilder builder;
    std::shared_ptr<npu::tile_fwk::Function> fwkFunc;
    LogicalTensors params;
    std::vector<ir::StmtPtr> stmts;

    IrFuncSetup(const std::string& name)
    {
        fwkFunc = std::make_shared<npu::tile_fwk::Function>(Program::GetInstance(), name + "_magic", name, nullptr);
        fwkFunc->SetFunctionType(FunctionType::DYNAMIC);
        fwkFunc->SetGraphType(GraphType::TENSOR_GRAPH);
        Program::GetInstance().InsertFuncToFunctionMap(fwkFunc->GetMagicName(), fwkFunc);
        Program::GetInstance().SetCurrentFunction(fwkFunc.get());
        Program::GetInstance().SetLastFunction(fwkFunc.get());
    }

    LogicalTensorPtr MakeParam(const std::string& name)
    {
        auto lt = builder.CreateTensorVar(*fwkFunc, DT_FP32, {TILE, TILE}, TileOpFormat::TILEOP_ND, name);
        params.push_back(lt);
        return lt;
    }

    LogicalTensorPtr MakeLocal(const std::string& name)
    {
        auto lt = builder.CreateTensorVar(*fwkFunc, DT_FP32, {TILE, TILE}, TileOpFormat::TILEOP_ND, name);
        auto stmt = std::make_shared<ir::TensorOpStmt>(
            std::vector<ir::VarPtr>{std::static_pointer_cast<const ir::Var>(lt)}, nullptr, "TENSOR_ALLOC",
            std::vector<ir::ExprPtr>{}, std::vector<ir::VarPtr>{}, std::vector<std::pair<std::string, std::any>>{},
            Sp());
        stmts.push_back(std::static_pointer_cast<const ir::Stmt>(stmt));
        return lt;
    }

    Operation& AddDassemble(const LogicalTensorPtr& src, const LogicalTensorPtr& dst)
    {
        auto& op = fwkFunc->AddRawOperation(Opcode::OP_ASSEMBLE, {src}, {dst}, Sp());
        op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{TILE, TILE}));
        op.SetAttribute("dassemble", true);
        stmts.push_back(std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()));
        return op;
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

    // Wrap current stmts as the body of a ForStmt and push the ForStmt onto stmts.
    // ForStmt requires a non-null _config_scope attr (ConfigManagerNg::PushScope asserts non-null).
    ir::ForStmtPtr WrapStmtsInForLoop(const std::string& loopVarName,
                                      std::vector<std::pair<std::string, std::any>> attrs = {})
    {
        auto body = std::make_shared<ir::SeqStmts>(stmts, Sp());
        stmts.clear();
        auto intType = std::make_shared<ir::ScalarType>(ir::DataType::INT64);
        auto loopVar = IRContext::Get().MakeVar(loopVarName, intType, Sp());
        auto zero = std::make_shared<ir::ConstInt>(0, ir::DataType::INT64, Sp());
        auto ten = std::make_shared<ir::ConstInt>(10, ir::DataType::INT64, Sp());
        auto one = std::make_shared<ir::ConstInt>(1, ir::DataType::INT64, Sp());
        attrs.emplace_back("_config_scope", ConfigManagerNg::CurrentScope());
        auto forStmt = std::make_shared<ir::ForStmt>(loopVar, zero, ten, one, std::vector<ir::IterArgPtr>{}, body,
                                                     std::vector<ir::VarPtr>{}, Sp(), std::move(attrs));
        stmts.push_back(forStmt);
        return forStmt;
    }

    ir::ForStmtPtr WrapStmtsInForLoopWithIterArgs(const std::string& loopVarName,
                                                  const std::vector<LogicalTensorPtr>& initValues,
                                                  const std::vector<LogicalTensorPtr>& continueValues,
                                                  const std::vector<ir::VarPtr>& returnVars,
                                                  std::vector<std::pair<std::string, std::any>> attrs = {})
    {
        std::vector<ir::ExprPtr> continueExprs;
        for (auto& v : continueValues) {
            continueExprs.push_back(std::static_pointer_cast<const ir::Expr>(v));
        }
        auto continueStmt = std::make_shared<ir::ContinueStmt>(continueExprs, Sp());
        stmts.push_back(continueStmt);
        auto body = std::make_shared<ir::SeqStmts>(stmts, Sp());
        stmts.clear();

        std::vector<ir::IterArgPtr> iterArgs;
        for (size_t i = 0; i < initValues.size(); i++) {
            auto iterVar = builder.CreateTensorVar(*fwkFunc, DT_FP32, {TILE, TILE}, TileOpFormat::TILEOP_ND,
                                                   loopVarName + "_iter" + std::to_string(i));
            iterArgs.push_back(std::make_shared<ir::IterArg>(std::static_pointer_cast<const ir::Var>(iterVar),
                                                             std::static_pointer_cast<const ir::Expr>(initValues[i])));
        }

        auto intType = std::make_shared<ir::ScalarType>(ir::DataType::INT64);
        auto loopVar = IRContext::Get().MakeVar(loopVarName, intType, Sp());
        auto zero = std::make_shared<ir::ConstInt>(0, ir::DataType::INT64, Sp());
        auto ten = std::make_shared<ir::ConstInt>(10, ir::DataType::INT64, Sp());
        auto one = std::make_shared<ir::ConstInt>(1, ir::DataType::INT64, Sp());
        attrs.emplace_back("_config_scope", ConfigManagerNg::CurrentScope());
        auto forStmt = std::make_shared<ir::ForStmt>(loopVar, zero, ten, one, iterArgs, body, returnVars, Sp(),
                                                     std::move(attrs));
        stmts.push_back(forStmt);
        return forStmt;
    }

    ir::VarPtr MakeReturnVar(const std::string& name)
    {
        auto lt = builder.CreateTensorVar(*fwkFunc, DT_FP32, {TILE, TILE}, TileOpFormat::TILEOP_ND, name);
        return std::static_pointer_cast<const ir::Var>(lt);
    }

    // Construct a "None" expr (UnknownType Var), matching production ir.range(init_values=(None,...))
    // where Python ctx.unwrap(None) returns an UnknownType Var — not a null ExprPtr.
    ir::ExprPtr MakeNoneExpr() { return IRContext::Get().MakeVar("None", ir::GetUnknownType(), Sp()); }

    // Overload accepting ir::ExprPtr for initValues/continueValues (supports None exprs).
    ir::ForStmtPtr WrapStmtsInForLoopWithIterArgsExpr(const std::string& loopVarName,
                                                      const std::vector<ir::ExprPtr>& initValues,
                                                      const std::vector<ir::ExprPtr>& continueValues,
                                                      const std::vector<ir::VarPtr>& returnVars,
                                                      std::vector<std::pair<std::string, std::any>> attrs = {})
    {
        auto continueStmt = std::make_shared<ir::ContinueStmt>(continueValues, Sp());
        stmts.push_back(continueStmt);
        auto body = std::make_shared<ir::SeqStmts>(stmts, Sp());
        stmts.clear();

        std::vector<ir::IterArgPtr> iterArgs;
        for (size_t i = 0; i < initValues.size(); i++) {
            auto iterVar = builder.CreateTensorVar(*fwkFunc, DT_FP32, {TILE, TILE}, TileOpFormat::TILEOP_ND,
                                                   loopVarName + "_iter" + std::to_string(i));
            iterArgs.push_back(
                std::make_shared<ir::IterArg>(std::static_pointer_cast<const ir::Var>(iterVar), initValues[i]));
        }

        auto intType = std::make_shared<ir::ScalarType>(ir::DataType::INT64);
        auto loopVar = IRContext::Get().MakeVar(loopVarName, intType, Sp());
        auto zero = std::make_shared<ir::ConstInt>(0, ir::DataType::INT64, Sp());
        auto ten = std::make_shared<ir::ConstInt>(10, ir::DataType::INT64, Sp());
        auto one = std::make_shared<ir::ConstInt>(1, ir::DataType::INT64, Sp());
        attrs.emplace_back("_config_scope", ConfigManagerNg::CurrentScope());
        auto forStmt = std::make_shared<ir::ForStmt>(loopVar, zero, ten, one, iterArgs, body, returnVars, Sp(),
                                                     std::move(attrs));
        stmts.push_back(forStmt);
        return forStmt;
    }
};

std::vector<npu::tile_fwk::Function*> FindHiddenFuncs()
{
    std::vector<npu::tile_fwk::Function*> result;
    for (auto& [name, func] : Program::GetInstance().GetFunctionMap()) {
        if (name.find("_hiddenfunc") != std::string::npos) {
            result.push_back(func.get());
        }
    }
    return result;
}

std::vector<int> CollectConstructAssembleSlots()
{
    std::vector<int> slots;
    for (auto* func : FindHiddenFuncs()) {
        auto scope = func->Parent().GetSlotScope();
        if (scope) {
            for (int s : scope->constructAssembleSlotList) {
                slots.push_back(s);
            }
        }
    }
    return slots;
}
} // namespace

class IrFuncBuilderTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetBuildStatic(false);
    }

    void TearDown() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        Program::GetInstance().lastFunc_ = nullptr;
        Program::GetInstance().currentDynamicFunctionPtr_ = nullptr;
    }
};

// ============================================================================
// Dedup: two dassemble ops targeting the same intermediate tensor
//        => constructAssembleSlotList should contain exactly 1 slot
// ============================================================================
TEST_F(IrFuncBuilderTest, TestConstructAssembleSlotList_DedupSameSlot)
{
    IrFuncSetup setup("DedupSameSlot");

    auto a = setup.MakeParam("a");
    auto aux = setup.MakeLocal("aux");

    setup.AddDassemble(a, aux);
    setup.AddDassemble(a, aux);

    auto irFunc = setup.BuildIrFunction("DedupSameSlot");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 1u);

    auto slots = CollectConstructAssembleSlots();
    EXPECT_EQ(slots.size(), 1u) << "Expected 1 slot (deduplicated), got " << slots.size();
}

// ============================================================================
// Mixed: one dassemble to function param (excluded) + one to intermediate (kept)
//        => constructAssembleSlotList should contain exactly 1 slot
// ============================================================================
TEST_F(IrFuncBuilderTest, TestConstructAssembleSlotList_MixedParamAndIntermediate)
{
    IrFuncSetup setup("MixedParamAndIntermediate");

    auto a = setup.MakeParam("a");
    auto out = setup.MakeParam("out");
    auto aux = setup.MakeLocal("aux");

    setup.AddDassemble(a, aux);
    setup.AddDassemble(aux, out);

    auto irFunc = setup.BuildIrFunction("MixedParamAndIntermediate");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 1u);

    auto slots = CollectConstructAssembleSlots();
    EXPECT_EQ(slots.size(), 1u) << "Expected 1 slot (aux only, out excluded), got " << slots.size();
}

// ============================================================================
// ForStmt with "unroll_times" attr => TransformStmts reads it via std::any_cast<int>
//   and appends "_Unroll<N>" to the loop-var-derived path suffix baked into the
//   hidden func raw name (CreateHiddenFunc: dynFuncRaw + "_" + loopVarName + "_PATH0_hiddenfunc").
//   Verifies the unroll_times branch (ir_func_builder.cpp:521-527).
// ============================================================================
TEST_F(IrFuncBuilderTest, TestTransformStmts_UnrollTimesAttr)
{
    IrFuncSetup setup("UnrollTimesAttr");

    auto a = setup.MakeParam("a");
    auto aux = setup.MakeLocal("aux");
    setup.AddDassemble(a, aux);

    setup.WrapStmtsInForLoop("i", {{"unroll_times", 4}});

    auto irFunc = setup.BuildIrFunction("UnrollTimesAttr");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 1u);
    EXPECT_NE(hiddenFuncs[0]->GetRawName().find("_Unroll4_"), std::string::npos)
        << "Expected '_Unroll4_' in hidden func raw name, got: " << hiddenFuncs[0]->GetRawName();
}

// ============================================================================
// ForStmt WITHOUT "unroll_times" attr => default unrollTimes=1, suffix "_Unroll1".
//   Baseline confirming the attr-absent path defaults to 1.
// ============================================================================
TEST_F(IrFuncBuilderTest, TestTransformStmts_UnrollTimesDefault)
{
    IrFuncSetup setup("UnrollTimesDefault");

    auto a = setup.MakeParam("a");
    auto aux = setup.MakeLocal("aux");
    setup.AddDassemble(a, aux);

    setup.WrapStmtsInForLoop("i");

    auto irFunc = setup.BuildIrFunction("UnrollTimesDefault");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 1u);
    EXPECT_NE(hiddenFuncs[0]->GetRawName().find("_Unroll1_"), std::string::npos)
        << "Expected '_Unroll1_' in hidden func raw name, got: " << hiddenFuncs[0]->GetRawName();
}

TEST_F(IrFuncBuilderTest, TestMigrateReshapeInplaceLinkToHiddenFunc)
{
    IrFuncSetup setup("MigrateReshapeInplaceLink");

    auto input = setup.MakeParam("input");
    auto output = setup.MakeParam("output");
    auto reshaped = setup.MakeLocal("reshaped");

    auto& reshape = setup.fwkFunc->AddRawOperation(Opcode::OP_RESHAPE, {input}, {reshaped}, Sp());
    reshape.SetAttribute(OP_ATTR_PREFIX + "isInplace", true);
    setup.fwkFunc->SetSameMemId(input, reshaped);
    setup.stmts.push_back(std::static_pointer_cast<const ir::Stmt>(reshape.shared_from_this()));

    auto reshapeStmts = setup.stmts;
    setup.stmts.clear();
    setup.AddDassemble(reshaped, output);
    setup.WrapStmtsInForLoop("i");
    setup.stmts.insert(setup.stmts.begin(), reshapeStmts.begin(), reshapeStmts.end());

    auto irFunc = setup.BuildIrFunction("MigrateReshapeInplaceLink");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    Function* reshapeHiddenFunc = nullptr;
    for (auto* func : FindHiddenFuncs()) {
        for (auto& op : func->Operations()) {
            if (op.GetOpcode() == Opcode::OP_RESHAPE) {
                reshapeHiddenFunc = func;
                break;
            }
        }
    }
    ASSERT_NE(reshapeHiddenFunc, nullptr);
    ASSERT_EQ(reshapeHiddenFunc->GetIncast().size(), 1u);
    ASSERT_EQ(reshapeHiddenFunc->GetOutcast().size(), 1u);

    auto incastRaw = reshapeHiddenFunc->GetIncast().front()->GetRawTensor();
    auto outcastRaw = reshapeHiddenFunc->GetOutcast().front()->GetRawTensor();
    EXPECT_EQ(outcastRaw->memoryId, incastRaw->memoryId);

    auto link = reshapeHiddenFunc->outIncastLinkMap.find(outcastRaw);
    ASSERT_NE(link, reshapeHiddenFunc->outIncastLinkMap.end());
    EXPECT_EQ(link->second, incastRaw);
}

// ============================================================================
// Loop config scope => hidden func paramConfigs.
//   Verifies configs set in a ForStmt scope are restored when building its hidden func.
// ============================================================================
TEST_F(IrFuncBuilderTest, TestTransformStmts_ConfigToParamConfigs)
{
    IrFuncSetup setup("ConfigToParamConfigs");

    auto a = setup.MakeParam("a");
    auto aux = setup.MakeLocal("aux");
    setup.AddDassemble(a, aux);

    auto& cm = ConfigManagerNg::GetInstance();
    cm.BeginScope("loop_config", {});
    cm.SetScope({{"pass.pg_lower_bound", 777L}});
    cm.SetScope({{"operation.combine_axis", true}});
    setup.WrapStmtsInForLoop("i");
    cm.EndScope();

    auto irFunc = setup.BuildIrFunction("ConfigToParamConfigs");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 1u);
    EXPECT_EQ(hiddenFuncs[0]->paramConfigs_.sgPgLowerBound, 777);
    EXPECT_TRUE(hiddenFuncs[0]->paramConfigs_.combineAxis);
}

// ============================================================================
// Sibling loops with different scopes => hidden func configs stay isolated.
// ============================================================================
TEST_F(IrFuncBuilderTest, TestTransformStmts_SiblingConfigIsolation)
{
    IrFuncSetup setup("SiblingConfigIsolation");

    auto a = setup.MakeParam("a");
    auto aux1 = setup.MakeLocal("aux1");
    auto aux2 = setup.MakeLocal("aux2");
    std::map<int64_t, int64_t> firstConfig{{1, 2}};
    std::map<int64_t, int64_t> secondConfig{{3, 4}};

    auto& cm = ConfigManagerNg::GetInstance();
    cm.BeginScope("first_loop_config", {});
    cm.SetScope({{"pass.vec_nbuffer_setting", firstConfig}});
    setup.AddDassemble(a, aux1);
    setup.WrapStmtsInForLoop("i");
    cm.EndScope();

    cm.BeginScope("second_loop_config", {});
    cm.SetScope({{"pass.vec_nbuffer_setting", secondConfig}});
    setup.AddDassemble(a, aux2);
    setup.WrapStmtsInForLoop("j");
    cm.EndScope();

    auto irFunc = setup.BuildIrFunction("SiblingConfigIsolation");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 2u);

    bool foundFirst = false;
    bool foundSecond = false;
    for (auto* func : hiddenFuncs) {
        if (func->paramConfigs_.vecNBufferSetting == firstConfig) {
            foundFirst = true;
        }
        if (func->paramConfigs_.vecNBufferSetting == secondConfig) {
            foundSecond = true;
        }
    }
    EXPECT_TRUE(foundFirst);
    EXPECT_TRUE(foundSecond);
}

// ============================================================================
// Nested loops: inner scope has no local config => inherits parent loop config.
// ============================================================================
TEST_F(IrFuncBuilderTest, TestTransformStmts_NestedConfigInheritance)
{
    IrFuncSetup setup("NestedConfigInheritance");

    auto a = setup.MakeParam("a");
    auto innerAux = setup.MakeLocal("inner_aux");
    auto outerAux = setup.MakeLocal("outer_aux");
    std::map<int64_t, int64_t> parentConfig{{5, 6}};

    auto& cm = ConfigManagerNg::GetInstance();
    cm.BeginScope("outer_loop_config", {});
    cm.SetScope({{"pass.vec_nbuffer_setting", parentConfig}});

    cm.BeginScope("inner_loop_config", {});
    setup.AddDassemble(a, innerAux);
    setup.WrapStmtsInForLoop("inner");
    cm.EndScope();

    setup.AddDassemble(a, outerAux);
    setup.WrapStmtsInForLoop("outer");
    cm.EndScope();

    auto irFunc = setup.BuildIrFunction("NestedConfigInheritance");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 2u);
    for (auto* func : hiddenFuncs) {
        EXPECT_EQ(func->paramConfigs_.vecNBufferSetting, parentConfig);
    }
}

// ============================================================================
// Repeated SetScope on the same key => latest value is used by the hidden func.
// ============================================================================
TEST_F(IrFuncBuilderTest, TestTransformStmts_RepeatedSetUsesLatest)
{
    IrFuncSetup setup("RepeatedSetUsesLatest");

    auto a = setup.MakeParam("a");
    auto aux = setup.MakeLocal("aux");
    setup.AddDassemble(a, aux);

    auto& cm = ConfigManagerNg::GetInstance();
    cm.BeginScope("loop_config", {});
    cm.SetScope({{"pass.pg_lower_bound", 111L}});
    cm.SetScope({{"pass.pg_lower_bound", 777L}});
    setup.WrapStmtsInForLoop("i");
    cm.EndScope();

    auto irFunc = setup.BuildIrFunction("RepeatedSetUsesLatest");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 1u);
    EXPECT_EQ(hiddenFuncs[0]->paramConfigs_.sgPgLowerBound, 777);
}

// ============================================================================
// Nested loops: inner scope config overrides the inherited parent config.
// ============================================================================
TEST_F(IrFuncBuilderTest, TestTransformStmts_InnerConfigOverride)
{
    IrFuncSetup setup("InnerConfigOverride");

    auto a = setup.MakeParam("a");
    auto innerAux = setup.MakeLocal("inner_aux");
    auto outerAux = setup.MakeLocal("outer_aux");
    std::map<int64_t, int64_t> outerConfig{{5, 6}};
    std::map<int64_t, int64_t> innerConfig{{7, 8}};

    auto& cm = ConfigManagerNg::GetInstance();
    cm.BeginScope("outer_loop_config", {});
    cm.SetScope({{"pass.vec_nbuffer_setting", outerConfig}});

    cm.BeginScope("inner_loop_config", {});
    cm.SetScope({{"pass.vec_nbuffer_setting", innerConfig}});
    setup.AddDassemble(a, innerAux);
    setup.WrapStmtsInForLoop("inner");
    cm.EndScope();

    setup.AddDassemble(a, outerAux);
    setup.WrapStmtsInForLoop("outer");
    cm.EndScope();

    auto irFunc = setup.BuildIrFunction("InnerConfigOverride");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 2u);

    bool foundOuter = false;
    bool foundInner = false;
    for (auto* func : hiddenFuncs) {
        if (func->paramConfigs_.vecNBufferSetting == outerConfig) {
            foundOuter = true;
        }
        if (func->paramConfigs_.vecNBufferSetting == innerConfig) {
            foundInner = true;
        }
    }
    EXPECT_TRUE(foundOuter);
    EXPECT_TRUE(foundInner);
}

TEST_F(IrFuncBuilderTest, TestLinkIfStmtSlots_YieldToReturnVar)
{
    IrFuncSetup setup("LinkIfStmtSlots");

    auto out = setup.MakeParam("out");

    // thenBody: TENSOR_ALLOC then_val + ASSEMBLE out→then_val + YieldStmt [then_val]
    auto thenVal = setup.MakeLocal("then_val");
    setup.AddDassemble(out, thenVal);
    auto thenYield = std::make_shared<ir::YieldStmt>(std::vector<ir::ExprPtr>{thenVal}, Sp());
    setup.stmts.push_back(thenYield);
    auto thenBody = std::make_shared<ir::SeqStmts>(setup.stmts, Sp());
    setup.stmts.clear();

    // elseBody: TENSOR_ALLOC else_val + ASSEMBLE out→else_val + YieldStmt [else_val]
    auto elseVal = setup.MakeLocal("else_val");
    setup.AddDassemble(out, elseVal);
    auto elseYield = std::make_shared<ir::YieldStmt>(std::vector<ir::ExprPtr>{elseVal}, Sp());
    setup.stmts.push_back(elseYield);
    auto elseBody = std::make_shared<ir::SeqStmts>(setup.stmts, Sp());
    setup.stmts.clear();

    // returnVar
    auto returnVar = setup.builder.CreateTensorVar(*setup.fwkFunc, DT_FP32, {TILE, TILE}, TileOpFormat::TILEOP_ND,
                                                   "out_var");

    // IfStmt
    auto cond = std::make_shared<ir::ConstInt>(1, ir::DataType::INT64, Sp());
    auto ifStmt = std::make_shared<ir::IfStmt>(
        cond, thenBody, std::optional<ir::SeqStmtsPtr>{elseBody},
        std::vector<ir::VarPtr>{std::static_pointer_cast<const ir::Var>(returnVar)}, Sp());
    setup.stmts.push_back(ifStmt);

    // ReturnStmt [returnVar] — matches logicalParams_[0] = out by position
    auto returnStmt = std::make_shared<ir::ReturnStmt>(std::vector<ir::ExprPtr>{returnVar}, Sp());
    setup.stmts.push_back(returnStmt);

    // Build + Run pass
    auto irFunc = setup.BuildIrFunction("LinkIfStmtSlots");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());
    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    // LinkIfStmtSlots: SetSameSlot(elseVal, returnVar) → returnVar slot = elseVal slot
    // LinkReturnSlots: SetSameSlot(returnVar, out) → out slot = returnVar slot = elseVal slot
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    auto elseTensor = slotManager->GetSlotTensor(elseVal);
    auto returnTensor = slotManager->GetSlotTensor(returnVar);
    auto outTensor = slotManager->GetSlotTensor(out);

    EXPECT_EQ(elseTensor->Id(), returnTensor->Id());
    EXPECT_EQ(elseTensor->Id(), outTensor->Id());
}

// ============================================================================
// LinkControlFlowSlots: ForStmt with iterArgs + ContinueStmt + returnVars.
//   Verifies slot chain: param → returnVar → continueValue → iterVar → initValue
// ============================================================================
TEST_F(IrFuncBuilderTest, TestLinkControlFlowSlots_ForLoop)
{
    IrFuncSetup setup("LinkControlFlowSlots_ForLoop");

    auto out = setup.MakeParam("out");

    // Loop body: TENSOR_ALLOC loop_val + ASSEMBLE out→loop_val + ContinueStmt [loop_val]
    auto loopVal = setup.MakeLocal("loop_val");
    setup.AddDassemble(out, loopVal);

    auto returnVar = setup.MakeReturnVar("for_returnVar");
    setup.WrapStmtsInForLoopWithIterArgs("i", {out}, {loopVal}, {returnVar});

    // ReturnStmt [returnVar] — matches logicalParams_[0] = out by position
    auto returnStmt = std::make_shared<ir::ReturnStmt>(
        std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(returnVar)}, Sp());
    setup.stmts.push_back(returnStmt);

    auto irFunc = setup.BuildIrFunction("LinkControlFlowSlots_ForLoop");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());
    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    // Slot chain: out → returnVar → loop_val → iterVar → initValue(out)
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    auto outId = slotManager->GetSlotTensor(out)->Id();
    auto returnVarLt = std::const_pointer_cast<LogicalTensor>(
        std::dynamic_pointer_cast<const LogicalTensor>(returnVar));
    EXPECT_EQ(slotManager->GetSlotTensor(returnVarLt)->Id(), outId);
    EXPECT_EQ(slotManager->GetSlotTensor(loopVal)->Id(), outId);
}

// ============================================================================
// LinkControlFlowSlots: ForStmt containing IfStmt (for + if nesting).
//   Verifies slot chain propagates through both for and if:
//   param → for_returnVar → if_returnVar → yield_value
// ============================================================================
TEST_F(IrFuncBuilderTest, TestLinkControlFlowSlots_ForLoopWithIfStmt)
{
    IrFuncSetup setup("LinkControlFlowSlots_ForLoopWithIfStmt");

    auto out = setup.MakeParam("out");

    // IfStmt thenBody: TENSOR_ALLOC then_val + ASSEMBLE out→then_val + YieldStmt [then_val]
    auto thenVal = setup.MakeLocal("then_val");
    setup.AddDassemble(out, thenVal);
    auto thenYield = std::make_shared<ir::YieldStmt>(
        std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(thenVal)}, Sp());
    setup.stmts.push_back(thenYield);
    auto thenBody = std::make_shared<ir::SeqStmts>(setup.stmts, Sp());
    setup.stmts.clear();

    // IfStmt elseBody: TENSOR_ALLOC else_val + ASSEMBLE out→else_val + YieldStmt [else_val]
    auto elseVal = setup.MakeLocal("else_val");
    setup.AddDassemble(out, elseVal);
    auto elseYield = std::make_shared<ir::YieldStmt>(
        std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(elseVal)}, Sp());
    setup.stmts.push_back(elseYield);
    auto elseBody = std::make_shared<ir::SeqStmts>(setup.stmts, Sp());
    setup.stmts.clear();

    // IfStmt with returnVar
    auto ifReturnVar = setup.MakeReturnVar("if_returnVar");
    auto cond = std::make_shared<ir::ConstInt>(1, ir::DataType::INT64, Sp());
    auto ifStmt = std::make_shared<ir::IfStmt>(cond, thenBody, std::optional<ir::SeqStmtsPtr>{elseBody},
                                               std::vector<ir::VarPtr>{ifReturnVar}, Sp());
    setup.stmts.push_back(ifStmt);

    // ForStmt wraps the IfStmt: ContinueStmt [if_returnVar]
    auto forReturnVar = setup.MakeReturnVar("for_returnVar");
    setup.WrapStmtsInForLoopWithIterArgs(
        "i", {out},
        {std::const_pointer_cast<LogicalTensor>(std::dynamic_pointer_cast<const LogicalTensor>(ifReturnVar))},
        {forReturnVar});

    // ReturnStmt [for_returnVar] — matches logicalParams_[0] = out by position
    auto returnStmt = std::make_shared<ir::ReturnStmt>(
        std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(forReturnVar)}, Sp());
    setup.stmts.push_back(returnStmt);

    auto irFunc = setup.BuildIrFunction("LinkControlFlowSlots_ForLoopWithIfStmt");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());
    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    // Slot chain: out → for_returnVar → if_returnVar → then_val / else_val
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    auto outId = slotManager->GetSlotTensor(out)->Id();
    auto forReturnLt = std::const_pointer_cast<LogicalTensor>(
        std::dynamic_pointer_cast<const LogicalTensor>(forReturnVar));
    auto ifReturnLt = std::const_pointer_cast<LogicalTensor>(
        std::dynamic_pointer_cast<const LogicalTensor>(ifReturnVar));
    EXPECT_EQ(slotManager->GetSlotTensor(forReturnLt)->Id(), outId);
    EXPECT_EQ(slotManager->GetSlotTensor(ifReturnLt)->Id(), outId);
    EXPECT_EQ(slotManager->GetSlotTensor(thenVal)->Id(), outId);
    EXPECT_EQ(slotManager->GetSlotTensor(elseVal)->Id(), outId);
}

// ============================================================================
// LinkControlFlowSlots: Two chained ForStmts.
//   ForStmt2's initValue = ForStmt1's returnVar.
//   Verifies slot chain does not break across ForStmt boundaries:
//   param → for1_returnVar → for2_returnVar → loopVal2
//   (regression test: previously SetSameSlot overwrote returnVar1's slot)
// ============================================================================
TEST_F(IrFuncBuilderTest, TestLinkControlFlowSlots_ChainedForLoops)
{
    IrFuncSetup setup("LinkControlFlowSlots_ChainedForLoops");

    auto out = setup.MakeParam("out");

    // ForStmt1: body has loopVal1 + ASSEMBLE out→loopVal1 + ContinueStmt [loopVal1]
    auto loopVal1 = setup.MakeLocal("loop_val1");
    setup.AddDassemble(out, loopVal1);
    auto returnVar1 = setup.MakeReturnVar("for_returnVar1");
    setup.WrapStmtsInForLoopWithIterArgs("i1", {out}, {loopVal1}, {returnVar1});
    // Save ForStmt1, clear stmts to build ForStmt2 independently (parallel, not nested)
    auto forStmt1 = setup.stmts[0];
    setup.stmts.clear();

    // ForStmt2: body has loopVal2 + ASSEMBLE returnVar1→loopVal2 + ContinueStmt [loopVal2]
    auto loopVal2 = setup.MakeLocal("loop_val2");
    auto returnVar1Lt = std::const_pointer_cast<LogicalTensor>(
        std::dynamic_pointer_cast<const LogicalTensor>(returnVar1));
    setup.AddDassemble(returnVar1Lt, loopVal2);
    auto returnVar2 = setup.MakeReturnVar("for_returnVar2");
    setup.WrapStmtsInForLoopWithIterArgs("i2", {returnVar1Lt}, {loopVal2}, {returnVar2});
    // Put ForStmt1 before ForStmt2 to form parallel structure
    setup.stmts.insert(setup.stmts.begin(), forStmt1);

    // ReturnStmt [returnVar2] — matches logicalParams_[0] = out by position
    auto returnStmt = std::make_shared<ir::ReturnStmt>(
        std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(returnVar2)}, Sp());
    setup.stmts.push_back(returnStmt);

    auto irFunc = setup.BuildIrFunction("LinkControlFlowSlots_ChainedForLoops");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());
    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    // All vars must share the same slot as out.
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    auto outId = slotManager->GetSlotTensor(out)->Id();
    auto rv1Lt = std::const_pointer_cast<LogicalTensor>(std::dynamic_pointer_cast<const LogicalTensor>(returnVar1));
    auto rv2Lt = std::const_pointer_cast<LogicalTensor>(std::dynamic_pointer_cast<const LogicalTensor>(returnVar2));
    EXPECT_EQ(slotManager->GetSlotTensor(rv1Lt)->Id(), outId);
    EXPECT_EQ(slotManager->GetSlotTensor(loopVal1)->Id(), outId);
    EXPECT_EQ(slotManager->GetSlotTensor(rv2Lt)->Id(), outId);
    EXPECT_EQ(slotManager->GetSlotTensor(loopVal2)->Id(), outId);
}

// ============================================================================
// LinkControlFlowSlots: Nested ForStmts (outer wraps inner).
//   Verifies slot chain does not break when inner returnVar is outer continueValue:
//   param → outer_init → outer_iter → inner_returnVar → inner_iter → loopVal
// ============================================================================
TEST_F(IrFuncBuilderTest, TestLinkControlFlowSlots_NestedForLoops)
{
    IrFuncSetup setup("LinkControlFlowSlots_NestedForLoops");

    auto out = setup.MakeParam("out");

    // Inner ForStmt: body has loopVal + ASSEMBLE out→loopVal + ContinueStmt [loopVal]
    auto loopVal = setup.MakeLocal("loop_val");
    setup.AddDassemble(out, loopVal);
    auto innerReturnVar = setup.MakeReturnVar("inner_returnVar");
    setup.WrapStmtsInForLoopWithIterArgs("i_inner", {out}, {loopVal}, {innerReturnVar});

    // Outer ForStmt: wraps inner, initValue=out, continueValue=innerReturnVar
    auto outerReturnVar = setup.MakeReturnVar("outer_returnVar");
    auto innerReturnLt = std::const_pointer_cast<LogicalTensor>(
        std::dynamic_pointer_cast<const LogicalTensor>(innerReturnVar));
    setup.WrapStmtsInForLoopWithIterArgs("i_outer", {out}, {innerReturnLt}, {outerReturnVar});

    // ReturnStmt [outerReturnVar] — matches logicalParams_[0] = out by position
    auto returnStmt = std::make_shared<ir::ReturnStmt>(
        std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(outerReturnVar)}, Sp());
    setup.stmts.push_back(returnStmt);

    auto irFunc = setup.BuildIrFunction("LinkControlFlowSlots_NestedForLoops");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());
    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    // All vars must share the same slot as out.
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    auto outId = slotManager->GetSlotTensor(out)->Id();
    auto innerRvLt = std::const_pointer_cast<LogicalTensor>(
        std::dynamic_pointer_cast<const LogicalTensor>(innerReturnVar));
    auto outerRvLt = std::const_pointer_cast<LogicalTensor>(
        std::dynamic_pointer_cast<const LogicalTensor>(outerReturnVar));
    EXPECT_EQ(slotManager->GetSlotTensor(innerRvLt)->Id(), outId);
    EXPECT_EQ(slotManager->GetSlotTensor(outerRvLt)->Id(), outId);
    EXPECT_EQ(slotManager->GetSlotTensor(loopVal)->Id(), outId);
}

// ============================================================================
// LinkControlFlowSlots: ForStmt with null initValue (init_values=(None, ...)).
//   Verifies iterVar→value→returnVar are still connected when initValue is None
//   (initLt skipped, remaining nodes chained).
// ============================================================================
TEST_F(IrFuncBuilderTest, TestLinkControlFlowSlots_ForLoopNullInit)
{
    IrFuncSetup setup("LinkControlFlowSlots_ForLoopNullInit");

    auto out = setup.MakeParam("out");

    auto loopVal = setup.MakeLocal("loop_val");
    setup.AddDassemble(out, loopVal);

    auto returnVar = setup.MakeReturnVar("for_returnVar");
    // initValues = {None} → simulates init_values=(None, ...)
    auto forStmt = setup.WrapStmtsInForLoopWithIterArgsExpr(
        "i", {setup.MakeNoneExpr()}, {std::static_pointer_cast<const ir::Expr>(loopVal)}, {returnVar});

    auto returnStmt = std::make_shared<ir::ReturnStmt>(
        std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(returnVar)}, Sp());
    setup.stmts.push_back(returnStmt);

    auto irFunc = setup.BuildIrFunction("LinkControlFlowSlots_ForLoopNullInit");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());
    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    // initValue is None → initLt skipped; iterVar → loopVal → returnVar chained.
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    auto iterVarLt = std::const_pointer_cast<LogicalTensor>(
        std::dynamic_pointer_cast<const LogicalTensor>(forStmt->iterArgs_[0]->iterVar_));
    auto iterId = slotManager->GetSlotTensor(iterVarLt)->Id();
    auto returnVarLt = std::const_pointer_cast<LogicalTensor>(
        std::dynamic_pointer_cast<const LogicalTensor>(returnVar));
    EXPECT_EQ(slotManager->GetSlotTensor(loopVal)->Id(), iterId);
    EXPECT_EQ(slotManager->GetSlotTensor(returnVarLt)->Id(), iterId);
}

// ============================================================================
// LinkControlFlowSlots: ForStmt with null continueValue.
//   Verifies initLt->iterLt and iterLt->returnVarLt are still built when
//   continueValue is None (valueLt skipped, iterLt connects directly to returnVarLt).
// ============================================================================
TEST_F(IrFuncBuilderTest, TestLinkControlFlowSlots_ForLoopNullContinueValue)
{
    IrFuncSetup setup("LinkControlFlowSlots_ForLoopNullContinueValue");

    auto out = setup.MakeParam("out");

    auto returnVar = setup.MakeReturnVar("for_returnVar");
    // continueValues = {None} → ContinueStmt value is UnknownType
    auto forStmt = setup.WrapStmtsInForLoopWithIterArgsExpr("i", {std::static_pointer_cast<const ir::Expr>(out)},
                                                            {setup.MakeNoneExpr()}, {returnVar});

    auto returnStmt = std::make_shared<ir::ReturnStmt>(
        std::vector<ir::ExprPtr>{std::static_pointer_cast<const ir::Expr>(returnVar)}, Sp());
    setup.stmts.push_back(returnStmt);

    auto irFunc = setup.BuildIrFunction("LinkControlFlowSlots_ForLoopNullContinueValue");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());
    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    // continueValue is None → valueLt skipped; out -> iterVar -> returnVar connected.
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    auto outId = slotManager->GetSlotTensor(out)->Id();
    auto iterVarLt = std::const_pointer_cast<LogicalTensor>(
        std::dynamic_pointer_cast<const LogicalTensor>(forStmt->iterArgs_[0]->iterVar_));
    auto returnVarLt = std::const_pointer_cast<LogicalTensor>(
        std::dynamic_pointer_cast<const LogicalTensor>(returnVar));
    EXPECT_EQ(slotManager->GetSlotTensor(iterVarLt)->Id(), outId);
    EXPECT_EQ(slotManager->GetSlotTensor(returnVarLt)->Id(), outId);
}
