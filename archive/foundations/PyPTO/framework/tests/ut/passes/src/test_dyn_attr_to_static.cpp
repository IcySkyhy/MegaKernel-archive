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
 * \file test_dyn_attr_to_static.cpp
 * \brief Unit test for DynAttrToStatic pass.
 */

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/block_graph_pass/dyn_attr_to_static.h"
#include "interface/interpreter/raw_tensor_data.h"

using namespace npu::tile_fwk;

class DynAttrToStaticTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "PVC2_OOO");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_PRINT_GRAPH, true);
    }
    void TearDown() override {}
};

struct TempDirCleanup {
    std::string path;
    TempDirCleanup(const std::string& p) : path(p) {}
    ~TempDirCleanup() { (void)!system(("rm -rf " + path).c_str()); }
};

void CountFilesByExtension(const std::string& dir, size_t& jsonCnt, size_t& tifwkgrCnt)
{
    jsonCnt = 0;
    tifwkgrCnt = 0;
    DIR* dirp = opendir(dir.c_str());
    if (dirp == nullptr)
        return;
    struct dirent* entry;
    while ((entry = readdir(dirp)) != nullptr) {
        if (entry->d_type != DT_REG)
            continue;
        std::string filename = entry->d_name;
        size_t pos = filename.rfind('.');
        if (pos == std::string::npos)
            continue;
        std::string ext = filename.substr(pos);
        if (ext == ".json")
            jsonCnt++;
        else if (ext == ".tifwkgr")
            tifwkgrCnt++;
    }
    closedir(dirp);
}

const std::string LEFT_BRACKER = "(";

bool VerifyNewMacroExpr(std::reference_wrapper<SymbolicScalar>& dynScalar)
{
    std::string dynParamExpr = SymbolicExpressionTable::BuildExpression(dynScalar);
    // COA类型宏格式是"(RUNTIME_GET_COA_XXX("
    if (dynParamExpr.find(COA_PREFIX) != 1) { // 只保留COA类型宏，进行下一步检查
        return true;
    }

    // 找到第二个"("的位置
    size_t secondLBracket = dynParamExpr.find(LEFT_BRACKER, 1);

    // 检查找到第二个"("前的字符是否是"MAYBE_CONST"
    size_t postfixLen = MAYBE_CONST_POSTFIX.length();
    if (secondLBracket <= postfixLen) {
        return false;
    }
    if (dynParamExpr.substr(secondLBracket - postfixLen, postfixLen) == MAYBE_CONST_POSTFIX) {
        return true;
    }
    return false;
}

TEST_F(DynAttrToStaticTest, TestGetTensorData)
{
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = n * 8;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    std::vector<int32_t> inputAData(n * n);
    for (int k = 0; k < n * n; k++) {
        inputAData[k] = k;
    }

    Tensor inputC(DT_FP32, {n, s}, "inputC");
    std::vector<float> inputCData(n * s);
    for (int k = 0; k < n * s; k++) {
        inputCData[k] = (float)(1.0 * ((k % s) / n));
    }
    Tensor output(DT_FP32, {n, n}, "output");
    std::vector<float> outputGolden(n * n, 12.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<float>(inputC, inputCData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });

    FUNCTION("test_coa", {inputA, inputC}, {output})
    {
        LOOP("loop", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            Tensor t0 = Add(inputA, Element(DT_INT32, (int64_t)2)); // t0[i, j] -> inputA[i, j] + 2 -> i * n + j + 2
            SymbolicScalar v0 = GetTensorData(t0, {0, 1});          // t0[0, 1] -> 0 * n + 1 + 2 -> 3
            SymbolicScalar v1 = GetTensorData(t0, {0, 2});          // t0[0, 2] -> 0 * n + 2 + 2 -> 4
            auto t2 = View(inputC, {n, n}, {0, v0 * n});
            auto t3 = View(inputC, {n, n}, {0, v1 * n});
            output = Mul(t2, t3);
        }
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_test_coa");
    npu::tile_fwk::DynAttrToStatic passDynAttrToStatic;
    passDynAttrToStatic.RunOnFunction(*func);

    // ================== Verify Pass Effect TENSOR_loop ==================
#if ENABLE_HIDDENLOOP
    std::string loopPathFuncName = "TENSOR_loop_Unroll1_PATH0_hiddenfunc0";
#else
    std::string loopPathFuncName = "TENSOR_loop_Unroll1_PATH0";
#endif
    Function* loopPathFunc = Program::GetInstance().GetFunctionByRawName(loopPathFuncName);
    Function* rootFunc = loopPathFunc->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);
    for (auto it = rootFunc->programs_.begin(); it != rootFunc->programs_.end(); it++) {
        Function* leafFunc = it->second;
        auto operationViewer = leafFunc->Operations(false);
        for (size_t j = 0; j < operationViewer.size(); j++) {
            auto& op = operationViewer[j];
            std::vector<std::reference_wrapper<SymbolicScalar>> dynScalarList = passDynAttrToStatic
                                                                                    .GetOpDynamicAttributeList(op);
            for (auto dynScalar : dynScalarList) {
                EXPECT_EQ(VerifyNewMacroExpr(dynScalar), true);
            }
        }
    }
}

TEST_F(DynAttrToStaticTest, TestSetTensorData)
{
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling, tiling);

    int n = tiling * 1;
    Tensor output(DT_INT32, {n, n, n}, "output");
    std::vector<int32_t> outputGolden(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        outputGolden[i] = i;
    }

    ProgramData::GetInstance().AppendInputs({});
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FUNCTION("test_coa", {}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n))
        {
            LOOP("Step1", FunctionType::DYNAMIC_LOOP, j, LoopRange(n))
            {
                for (int k = 0; k < n; k++) {
                    SetTensorData(i * tiling * tiling + j * tiling + k, {i, j, k}, output);
                }
            }
        }
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_test_coa");
    npu::tile_fwk::DynAttrToStatic passDynAttrToStatic;
    passDynAttrToStatic.RunOnFunction(*func);

    // ================== Verify Pass Effect TENSOR_Step1==================
#if ENABLE_HIDDENLOOP
    std::string loopPathFuncName = "TENSOR_Step1_Unroll1_PATH0_hiddenfunc0";
#else
    std::string loopPathFuncName = "TENSOR_Step1_Unroll1_PATH0";
#endif
    Function* loopPathFunc = Program::GetInstance().GetFunctionByRawName(loopPathFuncName);
    Function* rootFunc = loopPathFunc->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);
    for (auto it = rootFunc->programs_.begin(); it != rootFunc->programs_.end(); it++) {
        Function* leafFunc = it->second;
        auto operationViewer = leafFunc->Operations(false);
        for (size_t j = 0; j < operationViewer.size(); j++) {
            auto& op = operationViewer[j];
            std::vector<std::reference_wrapper<SymbolicScalar>> dynScalarList = passDynAttrToStatic
                                                                                    .GetOpDynamicAttributeList(op);
            for (auto dynScalar : dynScalarList) {
                EXPECT_EQ(VerifyNewMacroExpr(dynScalar), true);
            }
        }
    }
}

TEST_F(DynAttrToStaticTest, TestDynExpression)
{
    TileShape::Current().SetVecTile(64, 64);

    int b = 1;
    int sq = 128;
    int d = 64;
    std::vector<int64_t> qShape = {b, d};
    std::vector<int64_t> outShape = {b * sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor out(DT_FP32, outShape, "out");
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    FUNCTION("test_coa", {q}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b))
        {
            Tensor q0 = View(q, {1, d}, {1, d}, {batchId, 0});
            auto tmp = Expand(q0, {100, d});
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_test_coa");
    npu::tile_fwk::DynAttrToStatic passDynAttrToStatic;
    passDynAttrToStatic.RunOnFunction(*func);

    // ================== Verify Pass Effect TENSOR_L0 ==================
#if ENABLE_HIDDENLOOP
    std::string loopPathFuncName = "TENSOR_L0_Unroll1_PATH0_hiddenfunc0";
#else
    std::string loopPathFuncName = "TENSOR_L0_Unroll1_PATH0";
#endif
    Function* loopPathFunc = Program::GetInstance().GetFunctionByRawName(loopPathFuncName);
    Function* rootFunc = loopPathFunc->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);
    for (auto it = rootFunc->programs_.begin(); it != rootFunc->programs_.end(); it++) {
        Function* leafFunc = it->second;
        for (const auto& dynParam : leafFunc->GetDynParamTable()) {
            if (dynParam.second.replacedSymbol.empty() && dynParam.second.dim.IsValid()) {
                std::reference_wrapper<SymbolicScalar> dynExpr = const_cast<SymbolicScalar&>(dynParam.second.dim);
                EXPECT_EQ(VerifyNewMacroExpr(dynExpr), true);
            }
        }
    }
}

TEST_F(DynAttrToStaticTest, EdgeCases)
{
    VectorParamConsistencyChecker checker;

    // 场景1：单元素vector（仅一个索引组 {0}）
    EXPECT_TRUE(checker.RegisterCall({IRBuilder().CreateConstInt(99)}));
    auto allGroups = checker.GetAllConsistentIndexGroups();
    ASSERT_EQ(allGroups.size(), 1);
    // 多次调用单元素，始终有效
    EXPECT_TRUE(checker.RegisterCall({IRBuilder().CreateConstInt(88)}));
    EXPECT_TRUE(checker.RegisterCall({IRBuilder().CreateConstInt(77)}));
    EXPECT_EQ(checker.GetAllConsistentIndexGroups().size(), 1);

    checker.Reset();

    // 场景2：首次调用生成重复索引组（验证去重逻辑）
    // 调用值：{5,5,5} → 理论上生成 {0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}？
    // 实际代码逻辑：首次调用按「值-索引列表」生成，仅 {0,1,2} 一个候选组（因为所有索引值相同）
    EXPECT_TRUE(checker.RegisterCall(
        {IRBuilder().CreateConstInt(5), IRBuilder().CreateConstInt(5), IRBuilder().CreateConstInt(5)}));
    allGroups = checker.GetAllConsistentIndexGroups();
    ASSERT_EQ(allGroups.size(), 1);

    // 场景3：候选组索引无序（验证去重时的排序逻辑）
    checker.Reset();
    // 第一次调用：{1,2,1} → 候选组 {0,2}, {1}
    EXPECT_TRUE(checker.RegisterCall(
        {IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(2), IRBuilder().CreateConstInt(1)}));
    allGroups = checker.GetAllConsistentIndexGroups();
    ASSERT_EQ(allGroups.size(), 2);
}

TEST_F(DynAttrToStaticTest, IntBasicCases)
{
    VectorParamConsistencyChecker checker;

    // 场景1：首次注册空vector → 失败
    EXPECT_FALSE(checker.RegisterCall({}));

    // 场景2：首次注册有效vector（长度3）→ 成功，候选组为所有值对应的索引组
    std::vector<SymbolicScalar> call1 = {IRBuilder().CreateConstInt(10), IRBuilder().CreateConstInt(10),
                                         IRBuilder().CreateConstInt(20)};
    EXPECT_TRUE(checker.RegisterCall(call1));
    // 首次调用候选组：{0,1}（值10）、{2}（值20）
    auto allGroups = checker.GetAllConsistentIndexGroups();
    ASSERT_EQ(allGroups.size(), 2);

    // 场景3：第二次注册长度不一致的vector → 失败，标记为无效
    std::vector<SymbolicScalar> call2 = {10, 10};
    EXPECT_FALSE(checker.RegisterCall(call2));
    // 无效后候选组为空
    EXPECT_TRUE(checker.GetAllConsistentIndexGroups().empty());

    // 重置校验器
    checker.Reset();
    EXPECT_TRUE(checker.GetAllConsistentIndexGroups().empty());

    // 场景4：多次注册长度一致，筛选有效候选组
    // 第一次调用：{1,1,2} → 候选组 {0,1}, {2}
    EXPECT_TRUE(checker.RegisterCall(
        {IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(2)}));
    // 第二次调用：{3,3,4} → 候选组仍为 {0,1}, {2}（组内值仍相同）
    EXPECT_TRUE(checker.RegisterCall(
        {IRBuilder().CreateConstInt(3), IRBuilder().CreateConstInt(3), IRBuilder().CreateConstInt(4)}));
    allGroups = checker.GetAllConsistentIndexGroups();
    ASSERT_EQ(allGroups.size(), 2);
    // 第三次调用：{5,6,5} → 仅 {2} 有效（0和1值不同，{0,1} 被过滤）
    EXPECT_TRUE(checker.RegisterCall(
        {IRBuilder().CreateConstInt(5), IRBuilder().CreateConstInt(6), IRBuilder().CreateConstInt(5)}));
    allGroups = checker.GetAllConsistentIndexGroups();
    ASSERT_EQ(allGroups.size(), 1);
    std::vector<std::vector<size_t>> groups = {{0, 1}, {2}};
    std::string output = checker.PrintIndexGroups(groups);
    std::string expected = "\nALL Consistent Index Group:  {\n"
                           "Consistent Index Group: 1{0, 1, }"
                           "\n"
                           "Consistent Index Group: 2{2, }"
                           "\n"
                           "}";
    EXPECT_EQ(output, expected);
}

TEST_F(DynAttrToStaticTest, DumpAndPrintFunction)
{
    int tiling = 16;
    TileShape::Current().SetVecTile(tiling, tiling, tiling);
    int n = tiling * 2;
    Tensor outcast(DT_INT32, {n, n, n}, "outcast");
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(outcast, 0),
    });
    FUNCTION("test_dump_and_print", {}, {outcast})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n))
        {
            LOOP("Step1", FunctionType::DYNAMIC_LOOP, j, LoopRange(n))
            {
                for (int k = 0; k < n; k++) {
                    SetTensorData(i * tiling * tiling + j * tiling + k, {i, j, k}, outcast);
                }
            }
        }
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_test_dump_and_print");
    ASSERT_NE(func, nullptr);
    npu::tile_fwk::DynAttrToStatic passDynAttrToStatic;
    std::string dir = "/tmp/dyn_attr_to_static_test" + std::to_string(::getpid());
    (void)!system(("rm -rf " + dir).c_str());
    ASSERT_EQ(system(("mkdir -p " + dir).c_str()), 0);
    TempDirCleanup cleanup(dir);
    (void)cleanup;
    EXPECT_EQ(passDynAttrToStatic.DumpFunctionJson(*func, dir, true), SUCCESS);
    EXPECT_EQ(passDynAttrToStatic.PrintFunction(*func, dir, true), SUCCESS);
    size_t jsonCnt = 0, tifwkgrCnt = 0;
    CountFilesByExtension(dir, jsonCnt, tifwkgrCnt);
    EXPECT_EQ(jsonCnt, 3u);
    EXPECT_EQ(tifwkgrCnt, 3u);
}
