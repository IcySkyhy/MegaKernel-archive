/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "machine/host/expr_generator.h"
#include "machine/host/backend.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unistd.h>
#include <unordered_map>

namespace npu::tile_fwk {
namespace test {

class TestExprBatchGenerator : public testing::Test {
protected:
    void SetUp() override
    {
        testDir_ = "expr_generator_temp_" + std::to_string(getpid());
        mkdir(testDir_.c_str(), 0755);
    }

    void TearDown() override
    {
        std::string cmd = "rm -rf " + testDir_;
        ASSERT(system(cmd.c_str()) == 0);
    }

    std::string testDir_;
};

// Helper function to check if file exists
bool FileExists(const std::string& path)
{
    std::ifstream file(path);
    return file.good();
}

// Test CalculateBatches method
TEST_F(TestExprBatchGenerator, CalculateBatches)
{
    // Test with exactly EXPRS_PER_BATCH expressions
    ExprBatchGenerator generator1(testDir_, 1, 1000);
    // Test with more than EXPRS_PER_BATCH expressions
    ExprBatchGenerator generator2(testDir_, 2, 2500);
    // Test with less than EXPRS_PER_BATCH expressions
    ExprBatchGenerator generator3(testDir_, 3, 500);

    // We can't directly access the private batches_ vector, but we can test the behavior
    // by checking the generated files later
}

// Test HeaderFileBegin and HeaderFileEnd methods
TEST_F(TestExprBatchGenerator, HeaderFileGeneration)
{
    ExprBatchGenerator generator(testDir_, 1, 100);
    std::ostringstream exprHeaderOss;

    // Test HeaderFileBegin
    generator.HeaderFileBegin(exprHeaderOss);

    // Test HeaderFileEnd
    generator.HeaderFileEnd(exprHeaderOss);

    // Check if header file was created
    std::string headerPath = testDir_ + "/control_flow_expr_table.h";
    ASSERT_TRUE(FileExists(headerPath));

    // Check header file content
    std::ifstream headerFile(headerPath);
    std::string headerContent((std::istreambuf_iterator<char>(headerFile)), std::istreambuf_iterator<char>());
    ASSERT_TRUE(headerContent.find("#pragma once") != std::string::npos);
    ASSERT_TRUE(headerContent.find("namespace npu::tile_fwk") != std::string::npos);
}

// Test GenerateLinkScript method
TEST_F(TestExprBatchGenerator, LinkScriptGeneration)
{
    ExprBatchGenerator generator(testDir_, 1, 100);
    std::ostringstream exprHeaderOss;

    // Link script is generated in HeaderFileBegin
    generator.HeaderFileBegin(exprHeaderOss);

    // Check if link script was created
    std::string scriptPath = testDir_ + "/merge.link";
    ASSERT_TRUE(FileExists(scriptPath));

    // Check link script content
    std::ifstream scriptFile(scriptPath);
    std::string scriptContent((std::istreambuf_iterator<char>(scriptFile)), std::istreambuf_iterator<char>());
    ASSERT_TRUE(scriptContent.find("SECTIONS") != std::string::npos);
    ASSERT_TRUE(scriptContent.find(".pypto") != std::string::npos);
}

// Test CheckExprDependCore function
TEST_F(TestExprBatchGenerator, CheckExprDependCoreTest)
{
    // Create a test tensor name
    std::string testTensorName = "test_tensor";
    ValDependTensorMeta valDependTensorMeta;
    valDependTensorMeta.tensorNameToDependCore[testTensorName] = true;
    RawSymbolicScalarPtr arg1 = RawSymbolicSymbol::Create(testTensorName);
    RawSymbolicScalarPtr arg2 = RawSymbolicImmediate::Create(0);
    // Create a GetInputData call expression to test CheckExprDependCore
    RawSymbolicScalarPtr callee = RawSymbolicSymbol::Create("RUNTIME_GetInputData");

    std::vector<RawSymbolicScalarPtr> operands = {callee, arg1, arg2};
    RawSymbolicScalarPtr getInputDataExpr = std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_MOP_CALL,
                                                                                    operands);
    bool dependsCore = SymbolicExpressionTable::CheckExprDependCore(
        getInputDataExpr, valDependTensorMeta.tensorNameToDependCore, valDependTensorMeta.valDependMap);
    ASSERT_TRUE(dependsCore);

    valDependTensorMeta.tensorNameToDependCore[testTensorName] = false;
    valDependTensorMeta.valDependMap.clear();
    dependsCore = SymbolicExpressionTable::CheckExprDependCore(
        getInputDataExpr, valDependTensorMeta.tensorNameToDependCore, valDependTensorMeta.valDependMap);
    ASSERT_FALSE(dependsCore);
}

// Test InsertWaitCoreStart skips invalid primary expressions (e.g. unset mainBlockScalar_ after Normalize)
TEST_F(TestExprBatchGenerator, InsertWaitCoreStartSkipsNullPrimaryExprTest)
{
    ValDependTensorMeta valDependTensorMeta;
    SymbolicExpressionTable exprTable;
    SymbolicSymbolTable symTable;
    exprTable.NormalizeForSymbolTable(symTable);

    std::ostringstream controlFlowOss;
    InsertWaitCoreStart(&exprTable, controlFlowOss, valDependTensorMeta, 1);
    ASSERT_TRUE(controlFlowOss.str().find("WaitAicoreStart") == std::string::npos);
}

// Test InsertWaitCoreStart for loop Begin/End/Step expression table (loopBesDict)
TEST_F(TestExprBatchGenerator, InsertWaitCoreStartLoopBesTest)
{
    std::string testTensorName = "test_tensor";
    ValDependTensorMeta valDependTensorMeta;
    valDependTensorMeta.tensorNameToDependCore[testTensorName] = true;

    RawSymbolicScalarPtr callee = RawSymbolicSymbol::Create("RUNTIME_GetInputData");
    RawSymbolicScalarPtr arg1 = RawSymbolicSymbol::Create(testTensorName);
    RawSymbolicScalarPtr arg2 = RawSymbolicImmediate::Create(0);
    std::vector<RawSymbolicScalarPtr> operands = {callee, arg1, arg2};
    RawSymbolicScalarPtr getInputDataExpr = std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_MOP_CALL,
                                                                                    operands);

    SymbolicExpressionTable exprTable;
    exprTable.AddPrimaryExpression(SymbolicScalar(getInputDataExpr));

    std::ostringstream controlFlowOss;
    InsertWaitCoreStart(&exprTable, controlFlowOss, valDependTensorMeta, 1);
    ASSERT_TRUE(controlFlowOss.str().find("WaitAicoreStart(startArgs)") != std::string::npos);

    valDependTensorMeta.tensorNameToDependCore[testTensorName] = false;
    valDependTensorMeta.valDependMap.clear();
    controlFlowOss.str("");
    controlFlowOss.clear();
    InsertWaitCoreStart(&exprTable, controlFlowOss, valDependTensorMeta, 1);
    ASSERT_TRUE(controlFlowOss.str().find("WaitAicoreStart") == std::string::npos);
}

// Test GenerateBatchFile method
TEST_F(TestExprBatchGenerator, BatchFileGeneration)
{
    ExprBatchGenerator generator(testDir_, 1, 1500); // 2 batches
    std::ostringstream controlFlowOss;
    std::ostringstream exprHeaderOss;
    std::vector<std::string> exprSrcFiles;

    // Create an OrderedSet of RawSymbolicScalarPtr with T_SCALAR_SYMBOLIC_IMMEDIATE expressions
    SymbolicExpressionTable exprTable;
    OrderedSet<RawSymbolicScalarPtr> expressions;
    for (int i = 0; i < 1500; ++i) {
        // Create T_SCALAR_SYMBOLIC_IMMEDIATE expression
        RawSymbolicScalarPtr expr = RawSymbolicImmediate::Create(i);
        expressions.Insert(expr);
    }

    // Generate batch files
    generator.GenerateBatchFile(&exprTable, controlFlowOss, exprHeaderOss, "test_exp.h", expressions, exprSrcFiles, 1,
                                1);

    // Check if batch files were created
    ASSERT_EQ(exprSrcFiles.size(), 2);
    for (const auto& filePath : exprSrcFiles) {
        ASSERT_TRUE(FileExists(filePath));

        // Check file content
        std::ifstream batchFile(filePath);
        std::string fileContent((std::istreambuf_iterator<char>(batchFile)), std::istreambuf_iterator<char>());
        ASSERT_TRUE(fileContent.find("RUNTIME_SetExpr") != std::string::npos);
    }

    // Check control flow content
    std::string controlFlowContent = controlFlowOss.str();
    ASSERT_TRUE(controlFlowContent.find("SetExprBatch_1_0") != std::string::npos);
    ASSERT_TRUE(controlFlowContent.find("SetExprBatch_1_1") != std::string::npos);

    // Check header content
    std::string headerContent = exprHeaderOss.str();
    ASSERT_TRUE(headerContent.find("SetExprBatch_1_0") != std::string::npos);
    ASSERT_TRUE(headerContent.find("SetExprBatch_1_1") != std::string::npos);
}

// ============================================================================
// End-to-end folding tests
// ============================================================================
namespace {
RawSymbolicScalarPtr ImmNode(int64_t v) { return RawSymbolicImmediate::Create(v); }
RawSymbolicScalarPtr SymNode(const std::string& n) { return RawSymbolicSymbol::Create(n); }
RawSymbolicScalarPtr AddNode(const RawSymbolicScalarPtr& a, const RawSymbolicScalarPtr& b)
{
    return std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_BOP_ADD, std::vector<RawSymbolicScalarPtr>{a, b});
}
RawSymbolicScalarPtr MulNode(const RawSymbolicScalarPtr& a, const RawSymbolicScalarPtr& b)
{
    return std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_BOP_MUL, std::vector<RawSymbolicScalarPtr>{a, b});
}
std::string ReadFile(const std::string& path)
{
    std::ifstream f(path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
std::string RunGen(const std::string& dir, int devRootKey, const std::vector<RawSymbolicScalarPtr>& exprs)
{
    SymbolicExpressionTable table;
    OrderedSet<RawSymbolicScalarPtr> set;
    for (auto& e : exprs) {
        set.Insert(e);
    }
    // OrderedSet 按 shared_ptr 去重；totalExprs 必须与最终 set 大小一致。
    ExprBatchGenerator gen(dir, devRootKey, set.size());
    std::ostringstream cfOss;
    std::ostringstream headerOss;
    std::vector<std::string> srcFiles;
    gen.GenerateBatchFile(&table, cfOss, headerOss, "test_exp.h", set, srcFiles, 1, devRootKey);
    return srcFiles.empty() ? std::string{} : ReadFile(srcFiles.front());
}
size_t Count(const std::string& text, const std::string& pat)
{
    size_t cnt = 0;
    size_t pos = 0;
    while ((pos = text.find(pat, pos)) != std::string::npos) {
        cnt++;
        pos += pat.size();
    }
    return cnt;
}
} // namespace

// 单差异等差段（含 step==1）→ 紧凑 for 头。
TEST_F(TestExprBatchGenerator, FoldSingleDiff)
{
    std::vector<RawSymbolicScalarPtr> exprs;
    for (int64_t i = 0; i < 5; i++) {
        exprs.push_back(AddNode(SymNode("a"), ImmNode(i)));
    }
    std::string body = RunGen(testDir_, 10, exprs);
    EXPECT_EQ(Count(body, "for (int64_t sym_expr_loop_k_"), 1u);
    EXPECT_NE(body.find("<= 4"), std::string::npos);
    EXPECT_NE(body.find("+= 1"), std::string::npos);
    EXPECT_EQ(Count(body, "RUNTIME_SetExpr"), 1u);
}

// 多差异同步等差段 → 计数头 for (k=0; k<N; k++)。
TEST_F(TestExprBatchGenerator, FoldMultiDiffLockstep)
{
    std::vector<RawSymbolicScalarPtr> exprs;
    for (int64_t i = 0; i < 4; i++) {
        exprs.push_back(
            MulNode(AddNode(SymNode("a"), ImmNode(4 + i * 4)), AddNode(SymNode("b"), ImmNode(100 + i * 100))));
    }
    std::string body = RunGen(testDir_, 11, exprs);
    EXPECT_EQ(Count(body, "for (int64_t sym_expr_loop_k_"), 1u);
    EXPECT_NE(body.find("< 4"), std::string::npos);
    EXPECT_NE(body.find("k_0++"), std::string::npos);
    EXPECT_EQ(Count(body, "RUNTIME_SetExpr"), 1u);
}

// 不折叠的两种主因：段长 < MIN_LOOP_LEN；连续表达式内容相等（差异数为 0）。
TEST_F(TestExprBatchGenerator, NoFoldBelowMinLenOrEqualExprs)
{
    std::vector<RawSymbolicScalarPtr> short_run = {
        AddNode(SymNode("a"), ImmNode(4)),
        AddNode(SymNode("a"), ImmNode(8)),
    };
    std::string body1 = RunGen(testDir_, 12, short_run);
    EXPECT_EQ(Count(body1, "for (int64_t sym_expr_loop_k_"), 0u);
    EXPECT_EQ(Count(body1, "RUNTIME_SetExpr"), 2u);

    std::vector<RawSymbolicScalarPtr> equal_exprs;
    for (int i = 0; i < 4; i++) {
        equal_exprs.push_back(AddNode(SymNode("a"), ImmNode(4)));
    }
    std::string body2 = RunGen(testDir_, 13, equal_exprs);
    EXPECT_EQ(Count(body2, "for (int64_t sym_expr_loop_k_"), 0u);
    EXPECT_EQ(Count(body2, "RUNTIME_SetExpr"), 4u);
}

// 段分割：Symbol 名差异 / 步长变化 / 异模板孤立表达式三种断点交织，期望两段折叠 + 单行夹击。
TEST_F(TestExprBatchGenerator, RunBreaksAndMixed)
{
    std::vector<RawSymbolicScalarPtr> exprs;
    for (int64_t i = 0; i < 3; i++) {
        exprs.push_back(AddNode(SymNode("a"), ImmNode(4 + i * 4)));
    }
    exprs.push_back(MulNode(SymNode("zzz"), ImmNode(7)));
    for (int64_t i = 0; i < 3; i++) {
        exprs.push_back(AddNode(SymNode("b"), ImmNode(1 + i)));
    }
    std::string body = RunGen(testDir_, 14, exprs);
    EXPECT_EQ(Count(body, "for (int64_t sym_expr_loop_k_"), 2u);
    EXPECT_EQ(Count(body, "RUNTIME_SetExpr"), 3u);
}

RawSymbolicScalarPtr CallGetInputShapeDim(const std::string& argName, int64_t dim)
{
    return std::make_shared<RawSymbolicExpression>(
        SymbolicOpcode::T_MOP_CALL,
        std::vector<RawSymbolicScalarPtr>{SymNode("RUNTIME_GetInputShapeDim"), SymNode(argName), ImmNode(dim)});
}

RawSymbolicScalarPtr CallGetInputData(const std::string& argName, const std::string& dtype,
                                      const RawSymbolicScalarPtr& index)
{
    return std::make_shared<RawSymbolicExpression>(
        SymbolicOpcode::T_MOP_CALL,
        std::vector<RawSymbolicScalarPtr>{SymNode("RUNTIME_GetInputData"), SymNode(argName), SymNode(dtype), index});
}

// Batch-local CSE: 同一固定入参 GetInputShapeDim 出现 ≥2 次 → 顶部 __get_input_cse_N。
TEST_F(TestExprBatchGenerator, BatchLocalShapeDimCse)
{
    auto shape0 = CallGetInputShapeDim("ARG_x", 0);
    std::vector<RawSymbolicScalarPtr> exprs = {
        AddNode(shape0, ImmNode(1)), AddNode(shape0, ImmNode(2)),
        AddNode(CallGetInputShapeDim("ARG_y", 1), ImmNode(3)), // once only → no local cse
    };
    std::string body = RunGen(testDir_, 20, exprs);
    EXPECT_NE(body.find("int64_t __get_input_cse_0 = "), std::string::npos);
    EXPECT_NE(body.find("RUNTIME_GetInputShapeDim"), std::string::npos);
    EXPECT_EQ(Count(body, "__get_input_cse_0"), 3u);        // 1 decl + 2 uses
    EXPECT_EQ(Count(body, "RUNTIME_GetInputShapeDim"), 2u); // decl RHS + ARG_y once
}

// Batch-local CSE: 固定入参 GetInputData(ARG_*, RUNTIME_*, imm) 重复 → __get_input_cse_。
TEST_F(TestExprBatchGenerator, BatchLocalFixedGetInputDataCse)
{
    auto data0 = CallGetInputData("ARG_cu", "RUNTIME_int32_t", ImmNode(0));
    std::vector<RawSymbolicScalarPtr> exprs = {
        AddNode(data0, ImmNode(1)),
        AddNode(data0, ImmNode(2)),
    };
    std::string body = RunGen(testDir_, 22, exprs);
    EXPECT_NE(body.find("int64_t __get_input_cse_0 = "), std::string::npos);
    EXPECT_EQ(Count(body, "__get_input_cse_0"), 3u);
    EXPECT_EQ(Count(body, "RUNTIME_GetInputData"), 1u);
}

// 非固定入参（VALUE_* / 嵌套表达式）即使重复也不做 batch CSE。
TEST_F(TestExprBatchGenerator, BatchLocalSkipsLoopVaryingGetInputData)
{
    auto dataLoop = CallGetInputData("ARG_cu", "RUNTIME_int32_t", SymNode("VALUE_loop_idx_2"));
    auto dataExpr = CallGetInputData("ARG_cu", "RUNTIME_int32_t", AddNode(SymNode("VALUE_loop_idx_2"), ImmNode(1)));
    std::vector<RawSymbolicScalarPtr> exprs = {
        AddNode(dataLoop, ImmNode(1)),
        AddNode(dataLoop, ImmNode(2)),
        AddNode(dataExpr, ImmNode(3)),
        AddNode(dataExpr, ImmNode(4)),
    };
    std::string body = RunGen(testDir_, 23, exprs);
    EXPECT_EQ(body.find("int64_t __get_input_cse_"), std::string::npos);
    EXPECT_EQ(Count(body, "RUNTIME_GetInputData"), 4u);
}

// Launch CSE: 注入 CSE_sd[i] 后 batch 复用且不再生成本地 __cse。
TEST_F(TestExprBatchGenerator, GetInputCseReuse)
{
    auto shape0 = CallGetInputShapeDim("ARG_x", 0);
    std::vector<RawSymbolicScalarPtr> exprs = {
        AddNode(shape0, ImmNode(1)),
        AddNode(shape0, ImmNode(2)),
    };
    SymbolicExpressionTable table;
    OrderedSet<RawSymbolicScalarPtr> set;
    for (auto& e : exprs) {
        set.Insert(e);
    }
    GetInputCse getInputCse;
    const std::string key = SymbolicExpressionTable::BuildExpression(shape0);
    getInputCse.keyToName.emplace(key, "CSE_sd[0]");
    getInputCse.ordered.emplace_back("CSE_sd[0]", key);
    EXPECT_EQ(SymbolicExpressionTable::BuildExpression(shape0, &getInputCse.keyToName), "CSE_sd[0]");

    ExprBatchGenerator gen(testDir_, 21, set.size());
    std::ostringstream cfOss;
    std::ostringstream headerOss;
    std::vector<std::string> srcFiles;
    gen.GenerateBatchFile(&table, cfOss, headerOss, "test_exp.h", set, srcFiles, 1, 21, &getInputCse);
    ASSERT_FALSE(srcFiles.empty());
    std::string body = ReadFile(srcFiles.front());
    EXPECT_EQ(body.find("int64_t __get_input_cse_"), std::string::npos);
    EXPECT_EQ(Count(body, "CSE_sd[0]"), 2u);
    EXPECT_EQ(body.find("RUNTIME_GetInputShapeDim"), std::string::npos);
    EXPECT_NE(body.find("int64_t *CSE_sd"), std::string::npos);
}

// CollectGetInputCse：只收集固定入参 ShapeDim；排除 VALUE_ / sym_。
TEST_F(TestExprBatchGenerator, CollectGetInputCseFiltersNonFixedArgs)
{
    auto launchShape = CallGetInputShapeDim("ARG_x", 0);
    auto loopShape = CallGetInputShapeDim("VALUE_loop_idx", 0);
    auto symShape = CallGetInputShapeDim("sym_foo", 0);

    SymbolicExpressionTable t0;
    t0.AddPrimaryExpression(SymbolicScalar(AddNode(launchShape, ImmNode(1))));
    SymbolicExpressionTable t1;
    t1.AddPrimaryExpression(SymbolicScalar(AddNode(loopShape, ImmNode(2))));
    t1.AddPrimaryExpression(SymbolicScalar(AddNode(symShape, ImmNode(3))));
    t1.AddPrimaryExpression(SymbolicScalar(AddNode(launchShape, ImmNode(4))));

    auto cse = ExprBatchGenerator::CollectGetInputCse({&t0, &t1});
    ASSERT_EQ(cse.ordered.size(), 1u);
    EXPECT_EQ(cse.ordered[0].first, "CSE_sd[0]");
    EXPECT_NE(cse.ordered[0].second.find("ARG_x"), std::string::npos);
    EXPECT_EQ(cse.ordered[0].second.find("VALUE_loop_idx"), std::string::npos);
    EXPECT_EQ(cse.ordered[0].second.find("sym_foo"), std::string::npos);
}

} // namespace test
} // namespace npu::tile_fwk
