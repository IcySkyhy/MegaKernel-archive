/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_ir_backend.cpp
 * \brief Unit tests for SCF IR based control flow building functions (ir_backend.cpp).
 */

#include "gtest/gtest.h"

#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"
#include "interface/utils/id_gen.h"
#include "machine/host/backend.h"
#include "machine/host/expr_generator.h"
#include "machine/host/ir_backend.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;

namespace {
ir::Span Sp() { return ir::Span("test_ir_backend", 1, 1); }

IRBuilder& Builder()
{
    static IRBuilder b;
    return b;
}

ir::ExprPtr MakeSymbolExpr(const std::string& name) { return std::make_shared<RawSymbolicSymbol>(name); }

ir::VarPtr MakeVar(const std::string& name)
{
    return std::make_shared<ir::Var>(name, std::make_shared<ir::ScalarType>(ir::DataType::INT64), Sp());
}

ir::VarPtr MakeTensorVar(const std::string& name)
{
    return std::make_shared<ir::Var>(name, std::make_shared<ir::LogicalTensorType>(), Sp());
}

RawSymbolicScalarPtr SymNode(const std::string& name) { return std::make_shared<RawSymbolicSymbol>(name); }

ir::ExprPtr MakeGetInputShapeDimExpr(const std::string& argName, int64_t dim)
{
    return std::make_shared<RawSymbolicExpression>(
        SymbolicOpcode::T_MOP_CALL,
        std::vector<RawSymbolicScalarPtr>{SymNode("RUNTIME_GetInputShapeDim"), SymNode(argName),
                                          Builder().CreateConstInt(dim).Raw()});
}

struct DynFuncFixture {
    std::shared_ptr<Function> dynFunc;
    std::shared_ptr<DyndevFunctionAttribute> dynAttr;

    DynFuncFixture(const std::string& name = "test_dyn")
    {
        Program::GetInstance().Reset();
        dynFunc = std::make_shared<Function>(Program::GetInstance(), name + "_magic", name, nullptr);
        dynFunc->SetFunctionType(FunctionType::DYNAMIC);
        dynFunc->SetGraphType(GraphType::TENSOR_GRAPH);
        dynAttr = std::make_shared<DyndevFunctionAttribute>();
        dynFunc->SetDyndevAttribute(dynAttr);
        Program::GetInstance().InsertFuncToFunctionMap(dynFunc->GetMagicName(), dynFunc);
        Program::GetInstance().SetCurrentDynamicFunction(dynFunc.get());
    }

    ~DynFuncFixture()
    {
        Program::GetInstance().SetCurrentDynamicFunction(nullptr);
        Program::GetInstance().Reset();
    }
};

struct LinkerFixture {
    SymbolicSymbolTable symbolTable;
    DyndevFunctionAttribute::FunctionGroup funcGroup;
    DyndevFunctionAttribute::ExpressionTableDictGroup exprTableDictGroup;
    std::unique_ptr<Linker> linker;

    LinkerFixture() { linker = std::make_unique<Linker>(symbolTable, funcGroup, exprTableDictGroup); }
};

// Shared GetInputCse fixture for IR control-flow CSE UT (avoids duplicated setup blocks).
struct GetInputCseTestSetup {
    ir::ExprPtr shape0;
    GetInputCse getInputCse;
    std::string key;

    explicit GetInputCseTestSetup(const std::string& argName = "ARG_x", int64_t dim = 0)
    {
        shape0 = MakeGetInputShapeDimExpr(argName, dim);
        key = SymbolicExpressionTable::BuildExpression(ExprPtrToSymbolicScalar(shape0).Raw());
        getInputCse.keyToName.emplace(key, "CSE_sd[0]");
        getInputCse.ordered.emplace_back("CSE_sd[0]", key);
    }

    void Bind(IrBackendContext& irBackendCtx) { irBackendCtx.getInputCse = &getInputCse; }
};
} // namespace

class TestSuite_IrBackend : public testing::Test {
protected:
    void SetUp() override { config::Reset(); }
    void TearDown() override { config::Reset(); }
};

struct ControlFlowCtx {
    IrBackendContext irBackendCtx;
    FunctionCache cache;
    std::unordered_map<int, int> slotIdxMapping;
    DyndevFunctionAttribute::FunctionGroup group;
    std::unordered_map<Function*, Function*> rootTileDict;
    std::ostringstream controlFlowOss;
    std::ostringstream expressionOss;
    std::ostringstream exprHeaderOss;
    std::vector<std::string> exprSrcFiles;
    ValDependTensorMeta meta;
};

static ir::ForStmtPtr AsForStmt(const ir::StmtPtr& stmt) { return std::dynamic_pointer_cast<const ir::ForStmt>(stmt); }

TEST_F(TestSuite_IrBackend, ExprPtrToSymbolicScalar_AllCases)
{
    EXPECT_FALSE(ExprPtrToSymbolicScalar(nullptr).IsValid());

    auto sym = ExprPtrToSymbolicScalar(MakeSymbolExpr("test_var"));
    EXPECT_TRUE(sym.IsValid());
    EXPECT_TRUE(sym.Raw()->IsSymbol());
    EXPECT_EQ(sym.Raw()->GetSymbolName(), "test_var");

    auto imm = ExprPtrToSymbolicScalar(Builder().CreateConstInt(42).AsExpr());
    EXPECT_TRUE(imm.IsValid());
    EXPECT_TRUE(imm.Raw()->IsImmediate());

    EXPECT_THROW(ExprPtrToSymbolicScalar(MakeVar("plain_var")), npu::tile_fwk::Error);
}

TEST_F(TestSuite_IrBackend, IsOpCallStmt_AllCases)
{
    EXPECT_FALSE(IsOpCallStmt(Builder().CreateSeqStmts({}, Sp())));
    EXPECT_FALSE(IsOpCallStmt(Builder().CreateTensorOpStmt({}, nullptr, "CALL", {}, {}, {}, Sp())));
    EXPECT_TRUE(IsOpCallStmt(
        Builder().CreateTensorOpStmt({}, nullptr, "CALL", {}, {}, {{"callee", std::string("some_func")}}, Sp())));
}

TEST_F(TestSuite_IrBackend, ResolveCalleeFromOpCall_AllCases)
{
    EXPECT_EQ(ResolveCalleeFromOpCall(Builder().CreateSeqStmts({}, Sp())), nullptr);
    EXPECT_EQ(ResolveCalleeFromOpCall(Builder().CreateTensorOpStmt({}, nullptr, "CALL", {}, {}, {}, Sp())), nullptr);

    DynFuncFixture fixture;
    auto calleeFunc = std::make_shared<Function>(Program::GetInstance(), "callee_magic", "callee_name", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("callee_magic", calleeFunc);
    auto result = ResolveCalleeFromOpCall(
        Builder().CreateTensorOpStmt({}, nullptr, "CALL", {}, {}, {{"callee", std::string("callee_magic")}}, Sp()));
    EXPECT_NE(result, nullptr);
    EXPECT_EQ(result->GetRawName(), "callee_name");

    EXPECT_EQ(ResolveCalleeFromOpCall(Builder().CreateTensorOpStmt(
                  {}, nullptr, "CALL", {}, {}, {{"callee", std::string("nonexistent_func")}}, Sp())),
              nullptr);
}

TEST_F(TestSuite_IrBackend, IrBuildVirtualLoopFunc_AllCases)
{
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        fixture.dynFunc->SetDyndevAttribute(nullptr);
        EXPECT_EQ(IrBuildVirtualLoopFunc(irBackendCtx, nullptr, fixture.dynFunc.get()), fixture.dynFunc.get());
    }
    auto mkFor = [](const std::vector<std::pair<std::string, std::any>>& attrs = {}) {
        auto loopVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64),
                                                Sp());
        return Builder().CreateForStmt(loopVar, Builder().CreateConstInt(0).AsExpr(),
                                       Builder().CreateConstInt(10).AsExpr(), Builder().CreateConstInt(1).AsExpr(), {},
                                       Builder().CreateSeqStmts({}, Sp()), {}, Sp(), attrs);
    };
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        auto forStmt = mkFor();
        auto* result = IrBuildVirtualLoopFunc(irBackendCtx, AsForStmt(forStmt).get(), fixture.dynFunc.get());
        EXPECT_NE(result, nullptr);
        EXPECT_NE(result, fixture.dynFunc.get());
        EXPECT_EQ(result->GetFunctionType(), FunctionType::DYNAMIC_LOOP);
        EXPECT_NE(result->GetDynloopAttribute(), nullptr);

        auto* cached = IrBuildVirtualLoopFunc(irBackendCtx, AsForStmt(forStmt).get(), fixture.dynFunc.get());
        EXPECT_EQ(result, cached);
    }
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        auto forStmt = mkFor({{"parallel", true}});
        auto* result = IrBuildVirtualLoopFunc(irBackendCtx, AsForStmt(forStmt).get(), fixture.dynFunc.get());
        EXPECT_NE(result, nullptr);
        EXPECT_TRUE(result->GetDynloopAttribute()->parallel);
    }
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        auto forStmt = mkFor({{"submit_before_loop", true}});
        auto* result = IrBuildVirtualLoopFunc(irBackendCtx, AsForStmt(forStmt).get(), fixture.dynFunc.get());
        EXPECT_NE(result, nullptr);
        EXPECT_TRUE(result->GetDynloopAttribute()->submitBeforeLoop);
    }
}

TEST_F(TestSuite_IrBackend, IrParseValueDependDesc_AllCases)
{
    {
        DynFuncFixture fixture;
        fixture.dynFunc->SetDyndevAttribute(nullptr);
        EXPECT_NO_THROW(IrParseValueDependDesc(fixture.dynFunc.get(), {Builder().CreateConstInt(0).AsExpr()}));
    }
    {
        DynFuncFixture fixture;
        EXPECT_NO_THROW(IrParseValueDependDesc(
            fixture.dynFunc.get(), {Builder().CreateConstInt(0).AsExpr(), Builder().CreateConstInt(10).AsExpr(),
                                    Builder().CreateConstInt(1).AsExpr()}));
    }
}

TEST_F(TestSuite_IrBackend, InsertCacheStopForContrlFlow_AllCases)
{
    auto mkFor = []() {
        return Builder().CreateForStmt(MakeVar("loop_idx"), Builder().CreateConstInt(0).AsExpr(),
                                       Builder().CreateConstInt(10).AsExpr(), Builder().CreateConstInt(1).AsExpr(), {},
                                       Builder().CreateSeqStmts({}, Sp()), {}, Sp());
    };
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        fixture.dynFunc->SetDyndevAttribute(nullptr);
        ValDependTensorMeta meta;
        InsertCacheStopForContrlFlow(irBackendCtx, AsForStmt(mkFor()).get(), fixture.dynFunc.get(), meta);
        EXPECT_FALSE(meta.disableCtrlFlowCache);
    }
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        ValDependTensorMeta meta;
        InsertCacheStopForContrlFlow(irBackendCtx, AsForStmt(mkFor()).get(), fixture.dynFunc.get(), meta);
        EXPECT_FALSE(meta.disableCtrlFlowCache);
    }
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        auto loopVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64),
                                                Sp());
        auto forStmt = Builder().CreateForStmt(
            loopVar, Builder().CreateConstInt(0).AsExpr(), Builder().CreateConstInt(10).AsExpr(),
            Builder().CreateConstInt(1).AsExpr(), {}, Builder().CreateSeqStmts({}, Sp()), {}, Sp());
        auto* loopFunc = IrBuildVirtualLoopFunc(irBackendCtx, AsForStmt(forStmt).get(), fixture.dynFunc.get());
        auto& desc = fixture.dynAttr->valueDependDescDict[loopFunc];
        desc.getInputDataCount = 1;
        desc.getTensorDataCount = 0;

        ValDependTensorMeta meta;
        InsertCacheStopForContrlFlow(irBackendCtx, AsForStmt(forStmt).get(), fixture.dynFunc.get(), meta);
        EXPECT_TRUE(meta.disableCtrlFlowCache);
    }
}

TEST_F(TestSuite_IrBackend, InsertWaitAicoreStartForControlFlow_AllCases)
{
    {
        auto forStmt = Builder().CreateForStmt(
            MakeVar("loop_idx"), Builder().CreateConstInt(0).AsExpr(), Builder().CreateConstInt(10).AsExpr(),
            Builder().CreateConstInt(1).AsExpr(), {}, Builder().CreateSeqStmts({}, Sp()), {}, Sp());
        std::ostringstream oss;
        ValDependTensorMeta meta;
        InsertWaitAicoreStartForControlFlow(AsForStmt(forStmt).get(), 1, oss, meta);
        EXPECT_TRUE(oss.str().empty());
    }
    {
        auto forStmt = Builder().CreateForStmt(MakeVar("loop_idx"), nullptr, nullptr, nullptr, {},
                                               Builder().CreateSeqStmts({}, Sp()), {}, Sp());
        std::ostringstream oss;
        ValDependTensorMeta meta;
        InsertWaitAicoreStartForControlFlow(AsForStmt(forStmt).get(), 1, oss, meta);
        EXPECT_TRUE(oss.str().empty());
    }
}

TEST_F(TestSuite_IrBackend, FindExprFromIRStmt_AllCases)
{
    IrBackendContext irBackendCtx;
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    FunctionCache cache;
    std::vector<ir::ExprPtr> condStack;

    EXPECT_NO_THROW(
        FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, nullptr, dynFixture.dynFunc.get(), condStack));
    EXPECT_NO_THROW(FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, Builder().CreateSeqStmts({}, Sp()),
                                       dynFixture.dynFunc.get(), condStack));

    auto loopVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64), Sp());
    auto forStmt = Builder().CreateForStmt(loopVar, Builder().CreateConstInt(0).AsExpr(),
                                           Builder().CreateConstInt(10).AsExpr(), Builder().CreateConstInt(1).AsExpr(),
                                           {}, Builder().CreateSeqStmts({}, Sp()), {}, Sp());
    EXPECT_NO_THROW(
        FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, forStmt, dynFixture.dynFunc.get(), condStack));

    auto ifBoth = Builder().CreateIfStmt(MakeSymbolExpr("cond_var"), Builder().CreateSeqStmts({}, Sp()),
                                         Builder().CreateSeqStmts({}, Sp()), {}, Sp());
    EXPECT_NO_THROW(
        FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, ifBoth, dynFixture.dynFunc.get(), condStack));

    auto ifThen = Builder().CreateIfStmt(MakeSymbolExpr("cond_var"), Builder().CreateSeqStmts({}, Sp()), std::nullopt,
                                         {}, Sp());
    EXPECT_NO_THROW(
        FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, ifThen, dynFixture.dynFunc.get(), condStack));

    EXPECT_NO_THROW(FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker,
                                       Builder().CreateTensorOpStmt({}, nullptr, "CALL", {}, {}, {}, Sp()),
                                       dynFixture.dynFunc.get(), condStack));
    EXPECT_NO_THROW(FindExprFromIRStmt(
        irBackendCtx, cache, *linkerFixture.linker,
        Builder().CreateTensorOpStmt({}, nullptr, "CALL", {}, {}, {{"callee", std::string("nonexistent")}}, Sp()),
        dynFixture.dynFunc.get(), condStack));

    auto nested = Builder().CreateSeqStmts({Builder().CreateSeqStmts({}, Sp())}, Sp());
    EXPECT_NO_THROW(
        FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, nested, dynFixture.dynFunc.get(), condStack));

    auto ifInBody = Builder().CreateIfStmt(MakeSymbolExpr("cond_var"), Builder().CreateSeqStmts({}, Sp()), std::nullopt,
                                           {}, Sp());
    auto forWithIf = Builder().CreateForStmt(
        loopVar, Builder().CreateConstInt(0).AsExpr(), Builder().CreateConstInt(10).AsExpr(),
        Builder().CreateConstInt(1).AsExpr(), {}, Builder().CreateSeqStmts({ifInBody}, Sp()), {}, Sp());
    EXPECT_NO_THROW(
        FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, forWithIf, dynFixture.dynFunc.get(), condStack));
}

TEST_F(TestSuite_IrBackend, FindAllExpressionFromIR_AllCases)
{
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        FunctionCache cache;
        dynFixture.dynFunc->body_ = std::make_shared<ir::SeqStmts>(std::vector<ir::StmtPtr>{}, Sp());
        EXPECT_NO_THROW(FindAllExpressionFromIR(irBackendCtx, cache, *linkerFixture.linker, dynFixture.dynFunc.get()));
    }
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        FunctionCache cache;
        dynFixture.dynFunc->body_ = nullptr;
        EXPECT_NO_THROW(FindAllExpressionFromIR(irBackendCtx, cache, *linkerFixture.linker, dynFixture.dynFunc.get()));
    }
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        FunctionCache cache;
        dynFixture.dynFunc->SetFunctionType(FunctionType::STATIC);
        EXPECT_NO_THROW(FindAllExpressionFromIR(irBackendCtx, cache, *linkerFixture.linker, dynFixture.dynFunc.get()));
    }
}

TEST_F(TestSuite_IrBackend, VisitIRStmtForControlFlow_NullAndEmpty)
{
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        EXPECT_NO_THROW(VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", nullptr,
                                                  dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group,
                                                  ctx.rootTileDict, ctx.controlFlowOss, ctx.expressionOss,
                                                  ctx.exprHeaderOss, 0, "expr", ctx.exprSrcFiles, ctx.meta));
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        EXPECT_NO_THROW(VisitIRStmtForControlFlow(
            ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", Builder().CreateSeqStmts({}, Sp()),
            dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict, ctx.controlFlowOss,
            ctx.expressionOss, ctx.exprHeaderOss, 0, "expr", ctx.exprSrcFiles, ctx.meta));
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto",
                                  Builder().CreateTensorOpStmt({}, nullptr, "ADD", {}, {}, {}, Sp()),
                                  dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                  ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                                  ctx.meta);
        EXPECT_TRUE(ctx.controlFlowOss.str().empty());
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        auto stmt = Builder().CreateTensorOpStmt({}, nullptr, "CALL", {}, {}, {{"callee", std::string("nonexistent")}},
                                                 Sp());
        VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", stmt,
                                  dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                  ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                                  ctx.meta);
        EXPECT_TRUE(ctx.controlFlowOss.str().empty());
    }
}

TEST_F(TestSuite_IrBackend, VisitIRStmtForControlFlow_IfStmt)
{
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        auto ifStmt = Builder().CreateIfStmt(MakeSymbolExpr("cond_var"), Builder().CreateSeqStmts({}, Sp()),
                                             Builder().CreateSeqStmts({}, Sp()), {}, Sp());
        VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", ifStmt,
                                  dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                  ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                                  ctx.meta);
        auto output = ctx.controlFlowOss.str();
        EXPECT_TRUE(output.find("if (") != std::string::npos);
        EXPECT_TRUE(output.find("} else {") != std::string::npos);
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        auto ifStmt = Builder().CreateIfStmt(MakeSymbolExpr("cond_var"), Builder().CreateSeqStmts({}, Sp()),
                                             std::nullopt, {}, Sp());
        VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", ifStmt,
                                  dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                  ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                                  ctx.meta);
        auto output = ctx.controlFlowOss.str();
        EXPECT_TRUE(output.find("if (") != std::string::npos);
        EXPECT_TRUE(output.find("} else {") == std::string::npos);
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        auto ifStmt = Builder().CreateIfStmt(MakeSymbolExpr("c"), Builder().CreateSeqStmts({}, Sp()),
                                             Builder().CreateSeqStmts({}, Sp()), {}, Sp());
        EXPECT_NO_THROW(VisitIRStmtForControlFlow(
            ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", Builder().CreateSeqStmts({ifStmt}, Sp()),
            dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict, ctx.controlFlowOss,
            ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles, ctx.meta));
        EXPECT_FALSE(ctx.controlFlowOss.str().empty());
    }
}

TEST_F(TestSuite_IrBackend, VisitIRStmtForControlFlow_ForStmt)
{
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    ControlFlowCtx ctx;
    auto loopVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64), Sp());
    auto forStmt = Builder().CreateForStmt(loopVar, Builder().CreateConstInt(0).AsExpr(),
                                           Builder().CreateConstInt(10).AsExpr(), Builder().CreateConstInt(1).AsExpr(),
                                           {}, Builder().CreateSeqStmts({}, Sp()), {}, Sp());
    VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", forStmt,
                              dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                              ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                              ctx.meta);
    auto output = ctx.controlFlowOss.str();
    EXPECT_TRUE(output.find("LOOP(") != std::string::npos);
    EXPECT_TRUE(output.find("VAR_loop_idx") != std::string::npos);
    EXPECT_TRUE(output.find("VALUE_loop_idx") != std::string::npos);
}

TEST_F(TestSuite_IrBackend, VisitForStmtForControlFlow_AllCases)
{
    auto mkFor = [](const std::vector<std::pair<std::string, std::any>>& attrs = {}) {
        auto loopVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64),
                                                Sp());
        return AsForStmt(Builder().CreateForStmt(
            loopVar, Builder().CreateConstInt(0).AsExpr(), Builder().CreateConstInt(10).AsExpr(),
            Builder().CreateConstInt(1).AsExpr(), {}, Builder().CreateSeqStmts({}, Sp()), {}, Sp(), attrs));
    };
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        auto forStmt = mkFor({{"submit_before_loop", true}});
        VisitForStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", forStmt,
                                   dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                   ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr",
                                   ctx.exprSrcFiles, ctx.meta);
        auto output = ctx.controlFlowOss.str();
        EXPECT_TRUE(output.find("RUNTIME_FUNCKEY_LOOP_BARRIER") != std::string::npos);
        EXPECT_TRUE(output.find("LOOP(") != std::string::npos);
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        config::SetRuntimeOption<int64_t>(DEVICE_SCHED_PARALLELISM, 8);
        auto forStmt = mkFor({{"parallel", true}});
        VisitForStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", forStmt,
                                   dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                   ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr",
                                   ctx.exprSrcFiles, ctx.meta);
        auto output = ctx.controlFlowOss.str();
        EXPECT_TRUE(output.find("RUNTIME_FUNCKEY_PARALLEL_FOR_BEGIN") != std::string::npos);
        EXPECT_TRUE(output.find("RUNTIME_FUNCKEY_PARALLEL_FOR_END") != std::string::npos);
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        auto forStmt = mkFor();
        VisitForStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", forStmt,
                                   dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                   ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr",
                                   ctx.exprSrcFiles, ctx.meta);
        auto output = ctx.controlFlowOss.str();
        EXPECT_TRUE(output.find("PARALLEL_FOR_BEGIN") == std::string::npos);
        EXPECT_TRUE(output.find("PARALLEL_FOR_END") == std::string::npos);
    }
}

TEST_F(TestSuite_IrBackend, BuildExpression_GetInputCseFullString)
{
    GetInputCseTestSetup cse;
    const auto* cseMap = &cse.getInputCse.keyToName;
    auto raw = ExprPtrToSymbolicScalar(cse.shape0).Raw();
    EXPECT_EQ(SymbolicExpressionTable::BuildExpression(raw), cse.key);
    EXPECT_EQ(SymbolicExpressionTable::BuildExpression(raw, cseMap), "CSE_sd[0]");
}

TEST_F(TestSuite_IrBackend, VisitForStmtForControlFlow_GetInputCseRewritesLoopBounds)
{
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    ControlFlowCtx ctx;
    GetInputCseTestSetup cse;
    cse.Bind(ctx.irBackendCtx);

    auto loopVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64), Sp());
    auto forStmt = AsForStmt(Builder().CreateForStmt(loopVar, Builder().CreateConstInt(0).AsExpr(), cse.shape0,
                                                     Builder().CreateConstInt(1).AsExpr(), {},
                                                     Builder().CreateSeqStmts({}, Sp()), {}, Sp()));
    VisitForStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", forStmt,
                               dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                               ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                               ctx.meta);
    const auto output = ctx.controlFlowOss.str();
    // Bound expressions must rewrite to CSE name as a whole (not a partial substring match).
    EXPECT_NE(output.find("LOOP(VAR_loop_idx, 0, CSE_sd[0], 1)"), std::string::npos);
    EXPECT_EQ(output.find("RUNTIME_GetInputShapeDim"), std::string::npos);
}

TEST_F(TestSuite_IrBackend, VisitIRStmtForControlFlow_IfStmt_GetInputCse)
{
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    ControlFlowCtx ctx;
    GetInputCseTestSetup cse;
    cse.Bind(ctx.irBackendCtx);

    auto ifStmt = Builder().CreateIfStmt(cse.shape0, Builder().CreateSeqStmts({}, Sp()), std::nullopt, {}, Sp());
    VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", ifStmt,
                              dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                              ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                              ctx.meta);
    EXPECT_EQ(ctx.controlFlowOss.str(), "  if (CSE_sd[0]) {\n  }\n");
}

TEST_F(TestSuite_IrBackend, BuildControlFlowFromIR_EmitsGetInputCseStackInits)
{
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    ControlFlowCtx ctx;

    const std::string workDir = "ir_backend_cse_" + std::to_string(getpid());
    const std::string emitDir = workDir + "/pypto/kernel_aicpu";
    ASSERT_EQ(mkdir(workDir.c_str(), 0755), 0);
    ASSERT_EQ(mkdir((workDir + "/pypto").c_str(), 0755), 0);
    ASSERT_EQ(mkdir(emitDir.c_str(), 0755), 0);
    setenv("ASCEND_WORK_PATH", workDir.c_str(), 1);
    config::SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, true);

    GetInputCseTestSetup cse;
    cse.Bind(ctx.irBackendCtx);

    dynFixture.dynFunc->body_ = std::make_shared<ir::SeqStmts>(std::vector<ir::StmtPtr>{}, Sp());
    BuildControlFlowFromIR(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", dynFixture.dynFunc.get(),
                           ctx.slotIdxMapping, ctx.group, ctx.rootTileDict, ctx.controlFlowOss, ctx.expressionOss,
                           ctx.exprHeaderOss, 0, "expr", ctx.exprSrcFiles, ctx.meta);
    const auto output = ctx.controlFlowOss.str();
    const std::string expectedInit = "  int64_t CSE_sd[1];\n  CSE_sd[0] = " + cse.key + ";\n";
    EXPECT_NE(output.find("ControlFlowEntry"), std::string::npos);
    EXPECT_NE(output.find(expectedInit), std::string::npos);

    unsetenv("ASCEND_WORK_PATH");
    std::string rmCmd = "rm -rf " + workDir;
    ASSERT_EQ(system(rmCmd.c_str()), 0);
}

TEST_F(TestSuite_IrBackend, FindExprFromIfStmt_RegistersReturnVars)
{
    IrBackendContext irBackendCtx;
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    FunctionCache cache;
    std::vector<ir::ExprPtr> condStack;

    auto retVarA = MakeVar("a1");
    auto retVarB = MakeVar("b1");
    auto ifStmt = Builder().CreateIfStmt(MakeSymbolExpr("cond"), Builder().CreateSeqStmts({}, Sp()), std::nullopt,
                                         {retVarA, retVarB}, Sp());

    FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, ifStmt, dynFixture.dynFunc.get(), condStack);

    const auto& symTable = linkerFixture.linker->GetSymbolTable()->GetSymbolTable();
    EXPECT_TRUE(symTable.count("a1") > 0);
    EXPECT_TRUE(symTable.count("b1") > 0);
}

TEST_F(TestSuite_IrBackend, FindExprFromIRStmt_YieldStmtCollectsSymbols)
{
    IrBackendContext irBackendCtx;
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    FunctionCache cache;
    std::vector<ir::ExprPtr> condStack;

    auto yield = Builder().CreateYieldStmt({MakeGetInputShapeDimExpr("ARG_x", 0)}, Sp());
    auto ifStmt = Builder().CreateIfStmt(MakeSymbolExpr("cond"), Builder().CreateSeqStmts({yield}, Sp()), std::nullopt,
                                         {MakeVar("c1")}, Sp());

    FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, ifStmt, dynFixture.dynFunc.get(), condStack);

    const auto& symTable = linkerFixture.linker->GetSymbolTable()->GetSymbolTable();
    EXPECT_TRUE(symTable.count("c1") > 0);
    EXPECT_TRUE(symTable.count("RUNTIME_GetInputShapeDim") > 0);
}

TEST_F(TestSuite_IrBackend, VisitIRStmtForControlFlow_IfStmtYieldImmediate)
{
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    ControlFlowCtx ctx;

    auto retVar = MakeVar("a1");
    auto yield = Builder().CreateYieldStmt({Builder().CreateConstInt(2048).AsExpr()}, Sp());
    auto ifStmt = Builder().CreateIfStmt(MakeSymbolExpr("cond"), Builder().CreateSeqStmts({yield}, Sp()), std::nullopt,
                                         {retVar}, Sp());

    VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", ifStmt,
                              dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                              ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                              ctx.meta);

    auto output = ctx.controlFlowOss.str();
    EXPECT_TRUE(output.find("VALUE_a1 = 2048;") != std::string::npos);
}

TEST_F(TestSuite_IrBackend, VisitIRStmtForControlFlow_IfStmtYieldSymbolicExpr)
{
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    ControlFlowCtx ctx;

    auto retVar = MakeVar("c1");
    auto yield = Builder().CreateYieldStmt({MakeSymbolExpr("some_var")}, Sp());
    auto ifStmt = Builder().CreateIfStmt(MakeSymbolExpr("cond"), Builder().CreateSeqStmts({yield}, Sp()), std::nullopt,
                                         {retVar}, Sp());

    VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", ifStmt,
                              dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                              ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                              ctx.meta);

    auto output = ctx.controlFlowOss.str();
    EXPECT_TRUE(output.find("VALUE_c1 = VALUE_some_var;") != std::string::npos);
}

TEST_F(TestSuite_IrBackend, VisitIRStmtForControlFlow_NestedIfStmtYieldIsolation)
{
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    ControlFlowCtx ctx;

    auto innerYield = Builder().CreateYieldStmt({Builder().CreateConstInt(1).AsExpr()}, Sp());
    auto innerIf = Builder().CreateIfStmt(MakeSymbolExpr("inner_cond"), Builder().CreateSeqStmts({innerYield}, Sp()),
                                          std::nullopt, {MakeVar("a")}, Sp());
    auto outerYield = Builder().CreateYieldStmt({Builder().CreateConstInt(2).AsExpr()}, Sp());
    auto outerIf = Builder().CreateIfStmt(MakeSymbolExpr("outer_cond"),
                                          Builder().CreateSeqStmts({innerIf, outerYield}, Sp()), std::nullopt,
                                          {MakeVar("x")}, Sp());

    VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", outerIf,
                              dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                              ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                              ctx.meta);

    auto output = ctx.controlFlowOss.str();
    EXPECT_TRUE(output.find("VALUE_a = 1;") != std::string::npos);
    EXPECT_TRUE(output.find("VALUE_x = 2;") != std::string::npos);
}

TEST_F(TestSuite_IrBackend, VisitIRStmtForControlFlow_IfStmtNoReturnVarsBackwardCompat)
{
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    ControlFlowCtx ctx;

    auto yield = Builder().CreateYieldStmt({Builder().CreateConstInt(42).AsExpr()}, Sp());
    auto ifStmt = Builder().CreateIfStmt(MakeSymbolExpr("cond"), Builder().CreateSeqStmts({yield}, Sp()), std::nullopt,
                                         {}, Sp());

    VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", ifStmt,
                              dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                              ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                              ctx.meta);

    auto output = ctx.controlFlowOss.str();
    EXPECT_TRUE(output.find("= 42;") == std::string::npos);
}

TEST_F(TestSuite_IrBackend, FindExprFromIfStmt_SkipsTensorReturnVars)
{
    IrBackendContext irBackendCtx;
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    FunctionCache cache;
    std::vector<ir::ExprPtr> condStack;

    auto scalarRet = MakeVar("scalar_a");
    auto tensorRet = MakeTensorVar("tensor_b");
    auto ifStmt = Builder().CreateIfStmt(MakeSymbolExpr("cond"), Builder().CreateSeqStmts({}, Sp()), std::nullopt,
                                         {scalarRet, tensorRet}, Sp());

    FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, ifStmt, dynFixture.dynFunc.get(), condStack);

    const auto& symTable = linkerFixture.linker->GetSymbolTable()->GetSymbolTable();
    EXPECT_TRUE(symTable.count("scalar_a") > 0);
    EXPECT_TRUE(symTable.count("tensor_b") == 0);
}

TEST_F(TestSuite_IrBackend, VisitIRStmtForControlFlow_IfStmtYieldSkipsTensor)
{
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    ControlFlowCtx ctx;

    auto scalarRet = MakeVar("scalar_a");
    auto tensorRet = MakeTensorVar("tensor_b");
    auto scalarYield = Builder().CreateConstInt(100).AsExpr();
    auto tensorYield = MakeTensorVar("tensor_val");
    auto yield = Builder().CreateYieldStmt({scalarYield, tensorYield}, Sp());
    auto ifStmt = Builder().CreateIfStmt(MakeSymbolExpr("cond"), Builder().CreateSeqStmts({yield}, Sp()), std::nullopt,
                                         {scalarRet, tensorRet}, Sp());

    VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", ifStmt,
                              dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                              ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                              ctx.meta);

    auto output = ctx.controlFlowOss.str();
    EXPECT_TRUE(output.find("VALUE_scalar_a = 100;") != std::string::npos);
    EXPECT_TRUE(output.find("VALUE_tensor_b") == std::string::npos);
}

TEST_F(TestSuite_IrBackend, FindExprFromForStmt_RegistersScalarReturnVars)
{
    IrBackendContext irBackendCtx;
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    FunctionCache cache;

    auto loopVar = MakeVar("loop_idx");
    auto scalarRet = MakeVar("acc");
    auto tensorRet = MakeTensorVar("tensor_out");
    auto forStmt = Builder().CreateForStmt(loopVar, Builder().CreateConstInt(0).AsExpr(),
                                           Builder().CreateConstInt(10).AsExpr(), Builder().CreateConstInt(1).AsExpr(),
                                           {}, Builder().CreateSeqStmts({}, Sp()), {scalarRet, tensorRet}, Sp());

    std::vector<ir::ExprPtr> condStack;
    FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, forStmt, dynFixture.dynFunc.get(), condStack);

    const auto& symTable = linkerFixture.linker->GetSymbolTable()->GetSymbolTable();
    EXPECT_TRUE(symTable.count("loop_idx") > 0);
    EXPECT_TRUE(symTable.count("acc") > 0);
    EXPECT_TRUE(symTable.count("tensor_out") == 0);
}

TEST_F(TestSuite_IrBackend, VisitForStmtForControlFlow_YieldScalarReturnVar)
{
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    ControlFlowCtx ctx;

    auto loopVar = MakeVar("loop_idx");
    auto retVar = MakeVar("counter");
    auto yield = Builder().CreateYieldStmt({Builder().CreateConstInt(42).AsExpr()}, Sp());
    auto forStmt = Builder().CreateForStmt(loopVar, Builder().CreateConstInt(0).AsExpr(),
                                           Builder().CreateConstInt(10).AsExpr(), Builder().CreateConstInt(1).AsExpr(),
                                           {}, Builder().CreateSeqStmts({yield}, Sp()), {retVar}, Sp());

    VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", forStmt,
                              dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                              ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                              ctx.meta);

    auto output = ctx.controlFlowOss.str();
    EXPECT_TRUE(output.find("VALUE_counter = 42;") != std::string::npos);
}

TEST_F(TestSuite_IrBackend, VisitForStmtForControlFlow_YieldSkipsTensor)
{
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    ControlFlowCtx ctx;

    auto loopVar = MakeVar("loop_idx");
    auto scalarRet = MakeVar("acc");
    auto tensorRet = MakeTensorVar("tensor_out");
    auto yield = Builder().CreateYieldStmt({Builder().CreateConstInt(99).AsExpr(), MakeTensorVar("tensor_val")}, Sp());
    auto forStmt = Builder().CreateForStmt(loopVar, Builder().CreateConstInt(0).AsExpr(),
                                           Builder().CreateConstInt(10).AsExpr(), Builder().CreateConstInt(1).AsExpr(),
                                           {}, Builder().CreateSeqStmts({yield}, Sp()), {scalarRet, tensorRet}, Sp());

    VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", forStmt,
                              dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                              ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                              ctx.meta);

    auto output = ctx.controlFlowOss.str();
    EXPECT_TRUE(output.find("VALUE_acc = 99;") != std::string::npos);
    EXPECT_TRUE(output.find("VALUE_tensor_out") == std::string::npos);
}
