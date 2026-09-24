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
 * \file test_memory_reuse.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include "tilefwk/tilefwk.h"
#include "interface/cache/function_cache.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/tensormap.h"
#include "passes/block_graph_pass/memory_reuse/global_memory_reuse.h"
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk {

class TestGlobalMemoryReuse : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
    void SetSymbolicScalarFunction(std::vector<std::vector<SymbolicScalar>>& symbolicScalarFunction,
                                   std::shared_ptr<LogicalTensor>& rawTensor, std::vector<SymbolicScalar> offset,
                                   std::vector<SymbolicScalar> shape)
    {
        EXPECT_EQ(offset.size(), shape.size());
        std::vector<SymbolicScalar> symbolicScalar;
        /*
         * argList[i]
         * [0]: rawTensorIndex
         * [1-dim]: offset
         * [dim+1, 2*dim]: shape
         * [2*dim+1, 3*dim]: rawshape
         * [3*dim+1, 4*dim]: validshape
         */
        (void)rawTensor;
        symbolicScalar.push_back(-1);

        symbolicScalar.reserve(offset.size() * 4); // 预留4倍空间
        symbolicScalar.insert(symbolicScalar.end(), offset.begin(), offset.end());
        symbolicScalar.insert(symbolicScalar.end(), shape.begin(), shape.end());
        symbolicScalar.insert(symbolicScalar.end(), shape.begin(), shape.end());
        symbolicScalar.insert(symbolicScalar.end(), shape.begin(), shape.end());
        symbolicScalarFunction.push_back(symbolicScalar);
    }
};

/*
影响到切图了，这部分应当修改
TEST_F(TestGlobalMemoryReuse, test_connection_matrix) {
    int b = 2;
    int n = 2;
    int s = 1;
    int kvLoraRank = 512;
    int vHeadDim =128;
    int h = 512;
    std::vector<int64_t> inShape = {b, n, s, kvLoraRank}; // (b, n, s, d)
    Tensor attnPostIn(DT_BF16, inShape, "attnPostIn");
    Tensor attenOutput;
    AttentionW aw;
    aw.kvBProjWV = Tensor(DT_BF16, {n, kvLoraRank, vHeadDim}, "kvBProjWV");
    aw.oProjW = Tensor(DT_BF16, {n * vHeadDim, h}, "oProjW");
    ConfigManager::Instance();
    FUNCTION("AttentionPost") {
        DeepseekAttention atten(deepseekConfig1, aw, 1);
        attenOutput = atten.AttentionPost2(attnPostIn);
    }
    attenOutput.GetDataType();
    auto function= Program::GetInstance().GetFunctionByRawName("TENSOR_AttentionPost");
    ASSERT_NE(function, nullptr);
    auto rootFunc = function->rootFunc_;
    auto callOps = rootFunc->Operations();
    size_t totalSize = 17; // transpose支持尾轴切分，operation数目有变化
    EXPECT_EQ(callOps.size(), totalSize);
    std::unordered_set<int64_t> storageSet;
    uint64_t totalLength = 0;
    for (auto &callop : callOps) {
        for (auto &in : callop.iOperand) {
            if (in->storage_ != nullptr && storageSet.count(in->storage_->id_) == 0) {
                storageSet.emplace(in->storage_->id_);
                totalLength += in->storage_->length_;
            }
        }
    }
    EXPECT_EQ(totalLength, 8192); // 储存长度最大是8192
    Json jsonT = attenOutput.GetStorage()->DumpJson();
    std::unordered_map<int, std::shared_ptr<RawTensor>> rawTensorDict;
    auto newTensor = LogicalTensor::LoadJson(*function, rawTensorDict, jsonT);
    attenOutput.GetStorage()->DumpSSA(true, true);
    attenOutput.GetStorage()->tensor->GetRawShapeSize();

    std::vector<int64_t> resultOffset;
    std::vector<int64_t> resultShape;
    CalcShapeAndOffsetOfGroup(function->inCasts_, resultOffset, resultShape);
    function->GetTensorMap().GetTensorByMagic(function->inCasts_[0]->magic);
    CalcOverlapSize(function->inCasts_[0], function->inCasts_[0]);
    CalcOverlap(function->inCasts_[0], function->inCasts_[0], true);
    CalcOverlap(function->inCasts_[0], function->inCasts_, true);
    Allocator allocator(rootFunc);
    allocator.Init();
    size_t nodeId0 = 0;
    size_t nodeId1 = 1;
    size_t nodeId2 = 2;
    size_t nodeId4 = 4;
    size_t nodeId5 = 5;
    size_t nodeId6 = 6;
    size_t nodeId13 = 13;
    size_t nodeId7 = 7;
    // 下面的值请勿随意修改，校验存在一定价值
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId0), callOps.at(nodeId1)), false);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId0), callOps.at(nodeId4)), false);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId0), callOps.at(nodeId5)), true);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId0), callOps.at(nodeId6)), true);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId5), callOps.at(nodeId13)), true);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId2), callOps.at(nodeId7)), true);
}
*/

TEST_F(TestGlobalMemoryReuse, CanReuseSeriesOpConn)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3", "CALL4"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    /* dump json */
    std::string jsonFilePath = "./config/pass/json/memory_reuse_can_reuse_series_op.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        function->DumpJsonFile(jsonFilePath);
    }

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_, 16 * 16 * 4 * 2); // shape: 16*16, size: 4, allocate 2 tensor memory
}

TEST_F(TestGlobalMemoryReuse, NotReuseParallelOpConn)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5", "t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3", "t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_, 16 * 16 * 4 * 4); // shape: 16*16, size: 4, allocate 4 tensor memory
}

TEST_F(TestGlobalMemoryReuse, NotReuseMultiInputOutput)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3", "t4", "t5"}, {"t6", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3", "t4", "t5"}, {"t6", "t7"}, {"t8"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t8"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_, 16 * 16 * 4 * 5); // shape: 16*16, size: 4, allocate 5 tensor memory
}

TEST_F(TestGlobalMemoryReuse, NotReuseViewOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_VIEW, Opcode::OP_CALL, Opcode::OP_ASSEMBLE,
                                Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "VIEW", "CALL1", "ASSEMBLE", "CALL2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            op->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
        } else if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            op->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
        } else {
            op->SetOpAttribute(
                std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
        }
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_, 16 * 16 * 4 * 2); // shape: 16*16, size: 4, allocate 2 tensor memory
}

TEST_F(TestGlobalMemoryReuse, NotReuseSeriesOpConnSizeDiff)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<std::string> tensorNames1{"t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3", "CALL4"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {64, 64}, tensorNames1), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_,
              16 * 16 * 4 * 2 +
                  64 * 64 * 4 * 2); // shape: 16*16, size: 4 + shape: 64*64, size: 4 , allocate 2 + 2 tensor memory
}

TEST_F(TestGlobalMemoryReuse, CanReuseSeriesOpConnSizeDiff)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<std::string> tensorNames1{"t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3", "CALL4"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {64, 32}, tensorNames1), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_, 64 * 32 * 4 * 2); // shape: 64*32, size: 4 , allocate 2 big tensor memory
}

TEST_F(TestGlobalMemoryReuse, CanReuseSeriesOpConnMultiSizeDiff)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t4"};
    std::vector<std::string> tensorNames1{"t3"};
    std::vector<std::string> tensorNames2{"t5"};
    std::vector<std::string> tensorNames3{"t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3", "CALL4"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {64, 64}, tensorNames1), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {32, 16}, tensorNames2), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {32, 32}, tensorNames3), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(
        allocator.size_,
        64 * 64 * 4 + 32 * 32 * 4); // allocate 2 large tensor memory, shape: 64*64, size: 4 and shape: 32*32, size: 4
}

TEST_F(TestGlobalMemoryReuse, AbnormalNullRootFunction)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<std::string> tensorNames1{"t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3", "CALL4"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {64, 32}, tensorNames1), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }

    GlobalMemoryReuse reusePass;
    Status status = reusePass.RunOnFunction(*function);

    EXPECT_EQ(status, FAILED);
}

TEST_F(TestGlobalMemoryReuse, AbnormalNullStorage)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<std::string> tensorNames1{"t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3", "CALL4"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {64, 32}, tensorNames1), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();

    /* stub storage_ is null */
    auto& tensorsDesc = allocator.storageNeedToAllocate_.front();
    auto& tensor = *(tensorsDesc.tensors.begin());
    tensor->storage_ = nullptr;

    Status status = allocator.Allocate();

    EXPECT_EQ(status, FAILED);
}

TEST_F(TestGlobalMemoryReuse, TestConnectionMatrix)
{
    ComputationalGraphBuilder G;
    // add tensor
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    // add op
    G.AddOp(Opcode::OP_VIEW, {"mat_a"}, {"mat_b"}, "view1");
    G.AddOp(Opcode::OP_VIEW, {"mat_b"}, {"mat_c"}, "view2");
    G.AddOp(Opcode::OP_VIEW, {"mat_b"}, {"mat_d"}, "view3");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_c", "mat_d"});
    // run pass
    Function* function = G.GetFunction();
    ConnectionMatrix connectionMatrix(function);
    connectionMatrix.Generate(function);
    // check after pass
    EXPECT_EQ(connectionMatrix.IsConnected(*G.GetOp("view1"), *G.GetOp("view2")), true);
    EXPECT_EQ(connectionMatrix.IsConnected(*G.GetOp("view1"), *G.GetOp("view3")), true);
    EXPECT_EQ(connectionMatrix.IsConnected(*G.GetOp("view2"), *G.GetOp("view1")), false);
    EXPECT_EQ(connectionMatrix.IsConnected(*G.GetOp("view3"), *G.GetOp("view1")), false);
    EXPECT_EQ(connectionMatrix.IsConnected(*G.GetOp("view2"), *G.GetOp("view3")), false);
    EXPECT_EQ(connectionMatrix.IsConnected(*G.GetOp("view3"), *G.GetOp("view2")), false);
    LargeBitmap largeBitmap0(3); // function有3个op, 所以bitmap长度为3
    LargeBitmap largeBitmap1(3); // function有3个op, 所以bitmap长度为3
    LargeBitmap largeBitmap2(3); // function有3个op, 所以bitmap长度为3
    largeBitmap0.SetValues(0b001);
    largeBitmap1.SetValues(0b011);
    largeBitmap2.SetValues(0b101);
    EXPECT_EQ(connectionMatrix.GetBitMap(0) == largeBitmap0, true); // 位置为0的op bitmap为largeBitmap0
    EXPECT_EQ(connectionMatrix.GetBitMap(1) == largeBitmap1, true); // 位置为1的op bitmap为largeBitmap1
    EXPECT_EQ(connectionMatrix.GetBitMap(2) == largeBitmap2, true); // 位置为2的op bitmap为largeBitmap2

    const LargeBitmap& invalidResult = connectionMatrix.GetBitMap(99);
    EXPECT_FALSE(invalidResult.GetBit(0));
}

TEST_F(TestGlobalMemoryReuse, TestLargeBitmapInvalidIndex)
{
    LargeBitmap bitmap(4);
    bitmap.SetBit(99);
    bitmap.ClearBit(99);
    EXPECT_FALSE(bitmap.GetBit(99));
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseNormal)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");

    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_d");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_d = leafG3.GetTensor("leaffunc3_mat_d");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_d->tensor = mat_d->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c"}, {"mat_d"}, "call3");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_b"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_b"}, {"leaffunc3_mat_c"}, "view1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_c"}, {"leaffunc3_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a"});
    leafG3.SetOutCast({"leaffunc3_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_EQ(mat_b->storage_->id_, mat_c->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseOutputActualRawmagic)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    mat_c->tensor->actualRawmagic = 10; // 设置leaffunc2输出的actualRawmagic为10(不为-1，也与其他tensor不同)
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_d");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_d = leafG3.GetTensor("leaffunc3_mat_d");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_d->tensor = mat_d->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c"}, {"mat_d"}, "call3");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_b"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_b"}, {"leaffunc3_mat_c"}, "view1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_c"}, {"leaffunc3_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a"});
    leafG3.SetOutCast({"leaffunc3_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseInputActualRawmagic)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    mat_b->tensor->actualRawmagic = 10; // 设置leaffunc2输入的actualRawmagic为10(不为-1，也与其他tensor不同)
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_d");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_d = leafG3.GetTensor("leaffunc3_mat_d");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_d->tensor = mat_d->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c"}, {"mat_d"}, "call3");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_b"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_b"}, {"leaffunc3_mat_c"}, "view1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_c"}, {"leaffunc3_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a"});
    leafG3.SetOutCast({"leaffunc3_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseInputLessThanOutput)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {63, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {63, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_d");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_d = leafG3.GetTensor("leaffunc3_mat_d");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_d->tensor = mat_d->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c"}, {"mat_d"}, "call3");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_b"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_b"}, {"leaffunc3_mat_c"}, "view1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_c"}, {"leaffunc3_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a"});
    leafG3.SetOutCast({"leaffunc3_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(63), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseInputEightTimesOutput)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {8, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {8, 128}, "mat_d");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc3_mat_d");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_d = leafG3.GetTensor("leaffunc3_mat_d");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_d->tensor = mat_d->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c"}, {"mat_d"}, "call3");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_b"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_b"}, {"leaffunc3_mat_c"}, "view1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_c"}, {"leaffunc3_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a"});
    leafG3.SetOutCast({"leaffunc3_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseInputLessThanEightTimesOutput)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {9, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {9, 128}, "mat_d");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {9, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {9, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {9, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {9, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {9, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {9, 128}, "leaffunc3_mat_d");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_d = leafG3.GetTensor("leaffunc3_mat_d");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_d->tensor = mat_d->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c"}, {"mat_d"}, "call3");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_b"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_b"}, {"leaffunc3_mat_c"}, "view1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_c"}, {"leaffunc3_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a"});
    leafG3.SetOutCast({"leaffunc3_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(9), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(9), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(9), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_EQ(mat_b->storage_->id_, mat_c->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseDim)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {1, 63, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {1, 63, 128}, "mat_d");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {1, 63, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {1, 63, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {1, 63, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {1, 63, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {1, 63, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {1, 63, 128}, "leaffunc3_mat_d");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_d = leafG3.GetTensor("leaffunc3_mat_d");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_d->tensor = mat_d->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c"}, {"mat_d"}, "call3");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_RESHAPE, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "reshape1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_b"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_b"}, {"leaffunc3_mat_c"}, "view1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_c"}, {"leaffunc3_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a"});
    leafG3.SetOutCast({"leaffunc3_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(
        symbolicScalarFunction2, leaffunc2_mat_d,
        {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
        {IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(63), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(
        symbolicScalarFunction3, leaffunc3_mat_a,
        {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
        {IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(63), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(
        symbolicScalarFunction3, leaffunc3_mat_d,
        {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
        {IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(63), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseDataType)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "mat_d");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP32, {64, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP32, {64, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP32, {64, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP32, {64, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP32, {64, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP32, {64, 128}, "leaffunc3_mat_d");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_d = leafG3.GetTensor("leaffunc3_mat_d");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_d->tensor = mat_d->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c"}, {"mat_d"}, "call3");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_b"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_b"}, {"leaffunc3_mat_c"}, "view1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_c"}, {"leaffunc3_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a"});
    leafG3.SetOutCast({"leaffunc3_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseNotMaxmunAxisNotEqual)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {64, 127}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 127}, "mat_d");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {64, 127}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {64, 127}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {64, 127}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {64, 127}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {64, 127}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {64, 127}, "leaffunc3_mat_d");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_d = leafG3.GetTensor("leaffunc3_mat_d");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_d->tensor = mat_d->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c"}, {"mat_d"}, "call3");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_b"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_b"}, {"leaffunc3_mat_c"}, "view1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_c"}, {"leaffunc3_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a"});
    leafG3.SetOutCast({"leaffunc3_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(127)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(127)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(127)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);

    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseNormal2)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    Function leafFunc4(Program::GetInstance(), "leafFunc4", "leafFunc4", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    leafFunc4.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);
    ComputationalGraphBuilder leafG4(&leafFunc4);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {64, 64}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    G.AddTensor(DataType::DT_FP16, {64, 64}, "mat_e");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    auto mat_e = G.GetTensor("mat_e");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {64, 64}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {64, 64}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {64, 64}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {64, 64}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {64, 64}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {64, 64}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {64, 64}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {64, 64}, "leaffunc3_mat_d");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_e");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_f");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_b = leafG3.GetTensor("leaffunc3_mat_b");
    auto leaffunc3_mat_f = leafG3.GetTensor("leaffunc3_mat_f");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_b->tensor = mat_e->tensor;
    leaffunc3_mat_f->tensor = mat_d->tensor;
    // add tensor for leaf func4
    leafG4.AddTensor(DataType::DT_FP16, {64, 64}, "leaffunc4_mat_a");
    leafG4.AddTensor(DataType::DT_FP16, {64, 64}, "leaffunc4_mat_b");
    leafG4.AddTensor(DataType::DT_FP16, {64, 64}, "leaffunc4_mat_c");
    leafG4.AddTensor(DataType::DT_FP16, {64, 64}, "leaffunc4_mat_d");
    auto leaffunc4_mat_a = leafG4.GetTensor("leaffunc4_mat_a");
    auto leaffunc4_mat_d = leafG4.GetTensor("leaffunc4_mat_d");
    leaffunc4_mat_a->tensor = mat_b->tensor;
    leaffunc4_mat_d->tensor = mat_e->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c", "mat_e"}, {"mat_d"}, "call3");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_e"}, "call4");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_c"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_d"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT,
                 {
                     "leaffunc1_mat_c",
                 },
                 {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_c"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_b"}, {"leaffunc3_mat_d"}, "copyIn2");
    leafG3.AddOp(Opcode::OP_ASSEMBLE, {"leaffunc3_mat_c", "leaffunc3_mat_d"}, {"leaffunc3_mat_e"}, "assemble1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_e"}, {"leaffunc3_mat_f"}, "copyOut1");
    // add op for leaf func4
    leafG4.AddOp(Opcode::OP_COPY_IN, {"leaffunc4_mat_a"}, {"leaffunc4_mat_b"}, "copyIn1");
    leafG4.AddOp(Opcode::OP_VIEW, {"leaffunc4_mat_b"}, {"leaffunc4_mat_c"}, "view1");
    leafG4.AddOp(Opcode::OP_COPY_OUT, {"leaffunc4_mat_c"}, {"leaffunc4_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a", "leaffunc3_mat_b"});
    leafG3.SetOutCast({"leaffunc3_mat_e"});
    leafG4.SetInCast({"leaffunc4_mat_a"});
    leafG4.SetOutCast({"leaffunc4_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(64)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(64)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(64)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_b,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(64)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_f,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction4;
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_a,
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(64)});
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(64)});
    auto opAttributeCall4 = std::make_shared<CallOpAttribute>(leafFunc4.ComputeHash(), symbolicScalarFunction4);
    G.GetOp("call4")->SetOpAttribute(opAttributeCall4);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3
    function->programs_.emplace(4, &leafFunc4); // 索引4 绑定 leafFunc4

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc4.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    cache.Insert(leafFunc4.GetFunctionHash(), leafFunc4);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_e->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
    EXPECT_NE(mat_b->storage_->id_, mat_e->storage_->id_);
    EXPECT_NE(mat_c->storage_->id_, mat_e->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseNormal3)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    std::string leafFunc1Name = "leafFunc1";
    std::string leafFunc2Name = "leafFunc2";
    std::string leafFunc3Name = "leafFunc3";
    std::string leafFunc4Name = "leafFunc4";
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    Function leafFunc4(Program::GetInstance(), "leafFunc4", "leafFunc4", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    leafFunc4.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);
    ComputationalGraphBuilder leafG4(&leafFunc4);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {32, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    G.AddTensor(DataType::DT_FP16, {32, 128}, "mat_e");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    auto mat_e = G.GetTensor("mat_e");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_d");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_e");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_f");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_b = leafG3.GetTensor("leaffunc3_mat_b");
    auto leaffunc3_mat_f = leafG3.GetTensor("leaffunc3_mat_f");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_b->tensor = mat_e->tensor;
    leaffunc3_mat_f->tensor = mat_d->tensor;
    // add tensor for leaf func4
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_a");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_b");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_c");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_d");
    auto leaffunc4_mat_a = leafG4.GetTensor("leaffunc4_mat_a");
    auto leaffunc4_mat_d = leafG4.GetTensor("leaffunc4_mat_d");
    leaffunc4_mat_a->tensor = mat_b->tensor;
    leaffunc4_mat_d->tensor = mat_e->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c", "mat_e"}, {"mat_d"}, "call3");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_e"}, "call4");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_c"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_d"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT,
                 {
                     "leaffunc1_mat_c",
                 },
                 {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_c"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_b"}, {"leaffunc3_mat_d"}, "copyIn2");
    leafG3.AddOp(Opcode::OP_ASSEMBLE, {"leaffunc3_mat_c", "leaffunc3_mat_d"}, {"leaffunc3_mat_e"}, "assemble1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_e"}, {"leaffunc3_mat_f"}, "copyOut1");
    // add op for leaf func4
    leafG4.AddOp(Opcode::OP_COPY_IN, {"leaffunc4_mat_a"}, {"leaffunc4_mat_b"}, "copyIn1");
    leafG4.AddOp(Opcode::OP_VIEW, {"leaffunc4_mat_b"}, {"leaffunc4_mat_c"}, "view1");
    leafG4.AddOp(Opcode::OP_COPY_OUT, {"leaffunc4_mat_c"}, {"leaffunc4_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a", "leaffunc3_mat_b"});
    leafG3.SetOutCast({"leaffunc3_mat_e"});
    leafG4.SetInCast({"leaffunc4_mat_a"});
    leafG4.SetOutCast({"leaffunc4_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_b,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_f,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction4;
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_a,
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall4 = std::make_shared<CallOpAttribute>(leafFunc4.ComputeHash(), symbolicScalarFunction4);
    G.GetOp("call4")->SetOpAttribute(opAttributeCall4);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3
    function->programs_.emplace(4, &leafFunc4); // 索引4 绑定 leafFunc4

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc4.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    cache.Insert(leafFunc4.GetFunctionHash(), leafFunc4);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_e->storage_, nullptr);
    EXPECT_EQ(mat_b->storage_->id_, mat_c->storage_->id_);
    EXPECT_EQ(mat_b->storage_->id_, mat_e->storage_->id_);
    EXPECT_EQ(mat_c->storage_->id_, mat_e->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseOffsetNotImmediate1)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    std::string leafFunc1Name = "leafFunc1";
    std::string leafFunc2Name = "leafFunc2";
    std::string leafFunc3Name = "leafFunc3";
    std::string leafFunc4Name = "leafFunc4";
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    Function leafFunc4(Program::GetInstance(), "leafFunc4", "leafFunc4", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    leafFunc4.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);
    ComputationalGraphBuilder leafG4(&leafFunc4);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {32, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    G.AddTensor(DataType::DT_FP16, {32, 128}, "mat_e");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    auto mat_e = G.GetTensor("mat_e");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_d");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_e");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_f");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_b = leafG3.GetTensor("leaffunc3_mat_b");
    auto leaffunc3_mat_f = leafG3.GetTensor("leaffunc3_mat_f");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_b->tensor = mat_e->tensor;
    leaffunc3_mat_f->tensor = mat_d->tensor;
    // add tensor for leaf func4
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_a");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_b");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_c");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_d");
    auto leaffunc4_mat_a = leafG4.GetTensor("leaffunc4_mat_a");
    auto leaffunc4_mat_d = leafG4.GetTensor("leaffunc4_mat_d");
    leaffunc4_mat_a->tensor = mat_b->tensor;
    leaffunc4_mat_d->tensor = mat_e->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c", "mat_e"}, {"mat_d"}, "call3");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_e"}, "call4");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_c"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_d"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT,
                 {
                     "leaffunc1_mat_c",
                 },
                 {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_c"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_b"}, {"leaffunc3_mat_d"}, "copyIn2");
    leafG3.AddOp(Opcode::OP_ASSEMBLE, {"leaffunc3_mat_c", "leaffunc3_mat_d"}, {"leaffunc3_mat_e"}, "assemble1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_e"}, {"leaffunc3_mat_f"}, "copyOut1");
    // add op for leaf func4
    leafG4.AddOp(Opcode::OP_COPY_IN, {"leaffunc4_mat_a"}, {"leaffunc4_mat_b"}, "copyIn1");
    leafG4.AddOp(Opcode::OP_VIEW, {"leaffunc4_mat_b"}, {"leaffunc4_mat_c"}, "view1");
    leafG4.AddOp(Opcode::OP_COPY_OUT, {"leaffunc4_mat_c"}, {"leaffunc4_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a", "leaffunc3_mat_b"});
    leafG3.SetOutCast({"leaffunc3_mat_e"});
    leafG4.SetInCast({"leaffunc4_mat_a"});
    leafG4.SetOutCast({"leaffunc4_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), CreateTestScalarVar("x")},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_b,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_f,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction4;
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_a,
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall4 = std::make_shared<CallOpAttribute>(leafFunc4.ComputeHash(), symbolicScalarFunction4);
    G.GetOp("call4")->SetOpAttribute(opAttributeCall4);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3
    function->programs_.emplace(4, &leafFunc4); // 索引4 绑定 leafFunc4

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc4.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    cache.Insert(leafFunc4.GetFunctionHash(), leafFunc4);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_e->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
    EXPECT_NE(mat_b->storage_->id_, mat_e->storage_->id_);
    EXPECT_NE(mat_c->storage_->id_, mat_e->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseOffsetNotImmediate2)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    Function leafFunc4(Program::GetInstance(), "leafFunc4", "leafFunc4", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    leafFunc4.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);
    ComputationalGraphBuilder leafG4(&leafFunc4);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {32, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    G.AddTensor(DataType::DT_FP16, {32, 128}, "mat_e");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    auto mat_e = G.GetTensor("mat_e");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_d");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_e");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_f");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_b = leafG3.GetTensor("leaffunc3_mat_b");
    auto leaffunc3_mat_f = leafG3.GetTensor("leaffunc3_mat_f");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_b->tensor = mat_e->tensor;
    leaffunc3_mat_f->tensor = mat_d->tensor;
    // add tensor for leaf func4
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_a");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_b");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_c");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_d");
    auto leaffunc4_mat_a = leafG4.GetTensor("leaffunc4_mat_a");
    auto leaffunc4_mat_d = leafG4.GetTensor("leaffunc4_mat_d");
    leaffunc4_mat_a->tensor = mat_b->tensor;
    leaffunc4_mat_d->tensor = mat_e->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c", "mat_e"}, {"mat_d"}, "call3");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_e"}, "call4");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_c"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_d"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT,
                 {
                     "leaffunc1_mat_c",
                 },
                 {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_c"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_b"}, {"leaffunc3_mat_d"}, "copyIn2");
    leafG3.AddOp(Opcode::OP_ASSEMBLE, {"leaffunc3_mat_c", "leaffunc3_mat_d"}, {"leaffunc3_mat_e"}, "assemble1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_e"}, {"leaffunc3_mat_f"}, "copyOut1");
    // add op for leaf func4
    leafG4.AddOp(Opcode::OP_COPY_IN, {"leaffunc4_mat_a"}, {"leaffunc4_mat_b"}, "copyIn1");
    leafG4.AddOp(Opcode::OP_VIEW, {"leaffunc4_mat_b"}, {"leaffunc4_mat_c"}, "view1");
    leafG4.AddOp(Opcode::OP_COPY_OUT, {"leaffunc4_mat_c"}, {"leaffunc4_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a", "leaffunc3_mat_b"});
    leafG3.SetOutCast({"leaffunc3_mat_e"});
    leafG4.SetInCast({"leaffunc4_mat_a"});
    leafG4.SetOutCast({"leaffunc4_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_b,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_f,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction4;
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_a,
                              {CreateTestScalarVar("x"), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall4 = std::make_shared<CallOpAttribute>(leafFunc4.ComputeHash(), symbolicScalarFunction4);
    G.GetOp("call4")->SetOpAttribute(opAttributeCall4);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3
    function->programs_.emplace(4, &leafFunc4); // 索引4 绑定 leafFunc4

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc4.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    cache.Insert(leafFunc4.GetFunctionHash(), leafFunc4);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_e->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
    EXPECT_NE(mat_b->storage_->id_, mat_e->storage_->id_);
    EXPECT_NE(mat_c->storage_->id_, mat_e->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseShapeNotImmediate1)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    Function leafFunc4(Program::GetInstance(), "leafFunc4", "leafFunc4", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    leafFunc4.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);
    ComputationalGraphBuilder leafG4(&leafFunc4);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {32, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    G.AddTensor(DataType::DT_FP16, {32, 128}, "mat_e");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    auto mat_e = G.GetTensor("mat_e");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_d");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_e");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_f");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_b = leafG3.GetTensor("leaffunc3_mat_b");
    auto leaffunc3_mat_f = leafG3.GetTensor("leaffunc3_mat_f");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_b->tensor = mat_e->tensor;
    leaffunc3_mat_f->tensor = mat_d->tensor;
    // add tensor for leaf func4
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_a");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_b");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_c");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_d");
    auto leaffunc4_mat_a = leafG4.GetTensor("leaffunc4_mat_a");
    auto leaffunc4_mat_d = leafG4.GetTensor("leaffunc4_mat_d");
    leaffunc4_mat_a->tensor = mat_b->tensor;
    leaffunc4_mat_d->tensor = mat_e->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c", "mat_e"}, {"mat_d"}, "call3");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_e"}, "call4");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_c"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_d"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT,
                 {
                     "leaffunc1_mat_c",
                 },
                 {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_c"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_b"}, {"leaffunc3_mat_d"}, "copyIn2");
    leafG3.AddOp(Opcode::OP_ASSEMBLE, {"leaffunc3_mat_c", "leaffunc3_mat_d"}, {"leaffunc3_mat_e"}, "assemble1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_e"}, {"leaffunc3_mat_f"}, "copyOut1");
    // add op for leaf func4
    leafG4.AddOp(Opcode::OP_COPY_IN, {"leaffunc4_mat_a"}, {"leaffunc4_mat_b"}, "copyIn1");
    leafG4.AddOp(Opcode::OP_VIEW, {"leaffunc4_mat_b"}, {"leaffunc4_mat_c"}, "view1");
    leafG4.AddOp(Opcode::OP_COPY_OUT, {"leaffunc4_mat_c"}, {"leaffunc4_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a", "leaffunc3_mat_b"});
    leafG3.SetOutCast({"leaffunc3_mat_e"});
    leafG4.SetInCast({"leaffunc4_mat_a"});
    leafG4.SetOutCast({"leaffunc4_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), CreateTestScalarVar("x")});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_b,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_f,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction4;
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_a,
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall4 = std::make_shared<CallOpAttribute>(leafFunc4.ComputeHash(), symbolicScalarFunction4);
    G.GetOp("call4")->SetOpAttribute(opAttributeCall4);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3
    function->programs_.emplace(4, &leafFunc4); // 索引4 绑定 leafFunc4

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc4.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    cache.Insert(leafFunc4.GetFunctionHash(), leafFunc4);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_e->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
    EXPECT_NE(mat_b->storage_->id_, mat_e->storage_->id_);
    EXPECT_NE(mat_c->storage_->id_, mat_e->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseShapeNotImmediate2)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    std::string leafFunc1Name = "leafFunc1";
    std::string leafFunc2Name = "leafFunc2";
    std::string leafFunc3Name = "leafFunc3";
    std::string leafFunc4Name = "leafFunc4";
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    Function leafFunc4(Program::GetInstance(), "leafFunc4", "leafFunc4", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    leafFunc4.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);
    ComputationalGraphBuilder leafG4(&leafFunc4);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {32, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    G.AddTensor(DataType::DT_FP16, {32, 128}, "mat_e");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    auto mat_e = G.GetTensor("mat_e");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_d");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_e");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_f");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_b = leafG3.GetTensor("leaffunc3_mat_b");
    auto leaffunc3_mat_f = leafG3.GetTensor("leaffunc3_mat_f");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_b->tensor = mat_e->tensor;
    leaffunc3_mat_f->tensor = mat_d->tensor;
    // add tensor for leaf func4
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_a");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_b");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_c");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_d");
    auto leaffunc4_mat_a = leafG4.GetTensor("leaffunc4_mat_a");
    auto leaffunc4_mat_d = leafG4.GetTensor("leaffunc4_mat_d");
    leaffunc4_mat_a->tensor = mat_b->tensor;
    leaffunc4_mat_d->tensor = mat_e->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c", "mat_e"}, {"mat_d"}, "call3");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_e"}, "call4");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_c"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_d"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT,
                 {
                     "leaffunc1_mat_c",
                 },
                 {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_c"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_b"}, {"leaffunc3_mat_d"}, "copyIn2");
    leafG3.AddOp(Opcode::OP_ASSEMBLE, {"leaffunc3_mat_c", "leaffunc3_mat_d"}, {"leaffunc3_mat_e"}, "assemble1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_e"}, {"leaffunc3_mat_f"}, "copyOut1");
    // add op for leaf func4
    leafG4.AddOp(Opcode::OP_COPY_IN, {"leaffunc4_mat_a"}, {"leaffunc4_mat_b"}, "copyIn1");
    leafG4.AddOp(Opcode::OP_VIEW, {"leaffunc4_mat_b"}, {"leaffunc4_mat_c"}, "view1");
    leafG4.AddOp(Opcode::OP_COPY_OUT, {"leaffunc4_mat_c"}, {"leaffunc4_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a", "leaffunc3_mat_b"});
    leafG3.SetOutCast({"leaffunc3_mat_e"});
    leafG4.SetInCast({"leaffunc4_mat_a"});
    leafG4.SetOutCast({"leaffunc4_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_b,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_f,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction4;
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_a,
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(0)},
                              {CreateTestScalarVar("x"), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall4 = std::make_shared<CallOpAttribute>(leafFunc4.ComputeHash(), symbolicScalarFunction4);
    G.GetOp("call4")->SetOpAttribute(opAttributeCall4);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3
    function->programs_.emplace(4, &leafFunc4); // 索引4 绑定 leafFunc4

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc4.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    cache.Insert(leafFunc4.GetFunctionHash(), leafFunc4);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_e->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
    EXPECT_NE(mat_b->storage_->id_, mat_e->storage_->id_);
    EXPECT_NE(mat_c->storage_->id_, mat_e->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseNormalOverlap)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    Function leafFunc4(Program::GetInstance(), "leafFunc4", "leafFunc4", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    leafFunc4.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);
    ComputationalGraphBuilder leafG4(&leafFunc4);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {32, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    G.AddTensor(DataType::DT_FP16, {32, 128}, "mat_e");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    auto mat_e = G.GetTensor("mat_e");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc3_mat_d");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_e");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_f");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_b = leafG3.GetTensor("leaffunc3_mat_b");
    auto leaffunc3_mat_f = leafG3.GetTensor("leaffunc3_mat_f");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_b->tensor = mat_e->tensor;
    leaffunc3_mat_f->tensor = mat_d->tensor;
    // add tensor for leaf func4
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_a");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_b");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_c");
    leafG4.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc4_mat_d");
    auto leaffunc4_mat_a = leafG4.GetTensor("leaffunc4_mat_a");
    auto leaffunc4_mat_d = leafG4.GetTensor("leaffunc4_mat_d");
    leaffunc4_mat_a->tensor = mat_b->tensor;
    leaffunc4_mat_d->tensor = mat_e->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c", "mat_e"}, {"mat_d"}, "call3");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_e"}, "call4");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_c"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_d"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT,
                 {
                     "leaffunc1_mat_c",
                 },
                 {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_c"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_b"}, {"leaffunc3_mat_d"}, "copyIn2");
    leafG3.AddOp(Opcode::OP_ASSEMBLE, {"leaffunc3_mat_c", "leaffunc3_mat_d"}, {"leaffunc3_mat_e"}, "assemble1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_e"}, {"leaffunc3_mat_f"}, "copyOut1");
    // add op for leaf func4
    leafG4.AddOp(Opcode::OP_COPY_IN, {"leaffunc4_mat_a"}, {"leaffunc4_mat_b"}, "copyIn1");
    leafG4.AddOp(Opcode::OP_VIEW, {"leaffunc4_mat_b"}, {"leaffunc4_mat_c"}, "view1");
    leafG4.AddOp(Opcode::OP_COPY_OUT, {"leaffunc4_mat_c"}, {"leaffunc4_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a", "leaffunc3_mat_b"});
    leafG3.SetOutCast({"leaffunc3_mat_e"});
    leafG4.SetInCast({"leaffunc4_mat_a"});
    leafG4.SetOutCast({"leaffunc4_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_b,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_f,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction4;
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_a,
                              {IRBuilder().CreateConstInt(31), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall4 = std::make_shared<CallOpAttribute>(leafFunc4.ComputeHash(), symbolicScalarFunction4);
    G.GetOp("call4")->SetOpAttribute(opAttributeCall4);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3
    function->programs_.emplace(4, &leafFunc4); // 索引4 绑定 leafFunc4

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc4.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    cache.Insert(leafFunc4.GetFunctionHash(), leafFunc4);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_NE(mat_e->storage_, nullptr);
    EXPECT_NE(mat_b->storage_->id_, mat_c->storage_->id_);
    EXPECT_NE(mat_b->storage_->id_, mat_e->storage_->id_);
    EXPECT_NE(mat_c->storage_->id_, mat_e->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseNormal4)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {16, 128}, "mat_d");
    G.AddTensor(DataType::DT_FP16, {8, 128}, "mat_e");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_f");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    auto mat_e = G.GetTensor("mat_e");
    auto mat_f = G.GetTensor("mat_f");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_e");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_f");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_e = leafG1.GetTensor("leaffunc1_mat_e");
    auto leaffunc1_mat_f = leafG1.GetTensor("leaffunc1_mat_f");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_e->tensor = mat_b->tensor;
    leaffunc1_mat_f->tensor = mat_c->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_d");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_e");
    leafG2.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc2_mat_f");
    leafG2.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc2_mat_g");
    leafG2.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc2_mat_h");
    leafG2.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc2_mat_i");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_b = leafG2.GetTensor("leaffunc2_mat_b");
    auto leaffunc2_mat_h = leafG2.GetTensor("leaffunc2_mat_h");
    auto leaffunc2_mat_i = leafG2.GetTensor("leaffunc2_mat_i");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_b->tensor = mat_c->tensor;
    leaffunc2_mat_h->tensor = mat_d->tensor;
    leaffunc2_mat_i->tensor = mat_e->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc3_mat_d");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_e");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_f");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_b = leafG3.GetTensor("leaffunc3_mat_b");
    auto leaffunc3_mat_f = leafG3.GetTensor("leaffunc3_mat_f");
    leaffunc3_mat_a->tensor = mat_d->tensor;
    leaffunc3_mat_b->tensor = mat_e->tensor;
    leaffunc3_mat_f->tensor = mat_f->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b", "mat_c"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b", "mat_c"}, {"mat_d", "mat_e"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_d", "mat_e"}, {"mat_f"}, "call3");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_d"}, "view2");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_e"}, "copyOut1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_d"}, {"leaffunc1_mat_f"}, "copyOut2");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_c"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_b"}, {"leaffunc2_mat_d"}, "copyIn2");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_c"}, {"leaffunc2_mat_e"}, "view1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_d"}, {"leaffunc2_mat_e"}, "view2");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_e"}, {"leaffunc2_mat_f"}, "view3");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_e"}, {"leaffunc2_mat_g"}, "view4");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_f"}, {"leaffunc2_mat_h"}, "copyOut1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_g"}, {"leaffunc2_mat_i"}, "copyOut2");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_c"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_b"}, {"leaffunc3_mat_d"}, "copyIn2");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_c"}, {"leaffunc3_mat_e"}, "view1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_d"}, {"leaffunc3_mat_e"}, "view2");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_e"}, {"leaffunc3_mat_f"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_f"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_e", "leaffunc1_mat_f"});
    leafG2.SetInCast({"leaffunc2_mat_a", "leaffunc2_mat_b"});
    leafG2.SetOutCast({"leaffunc2_mat_h", "leaffunc2_mat_i"});
    leafG3.SetInCast({"leaffunc3_mat_a", "leaffunc3_mat_b"});
    leafG3.SetOutCast({"leaffunc3_mat_f"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_e,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_f,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_b,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_h,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_i,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_b,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_f,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_NE(mat_d->storage_, nullptr);
    EXPECT_NE(mat_e->storage_, nullptr);
    EXPECT_EQ(mat_f->storage_, nullptr);
    // 优先检索mat_d，mat_d复用了mat_c,检索mat_e时只有超过了8倍的mat_b,不能进行复用
    EXPECT_NE(mat_b->storage_->id_, mat_d->storage_->id_);
    EXPECT_NE(mat_b->storage_->id_, mat_e->storage_->id_);
    EXPECT_NE(mat_c->storage_->id_, mat_e->storage_->id_);
    EXPECT_EQ(mat_c->storage_->id_, mat_d->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseNormal5)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {8, 128}, "mat_d");
    G.AddTensor(DataType::DT_FP16, {16, 128}, "mat_e");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_f");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    auto mat_e = G.GetTensor("mat_e");
    auto mat_f = G.GetTensor("mat_f");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_e");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_f");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_e = leafG1.GetTensor("leaffunc1_mat_e");
    auto leaffunc1_mat_f = leafG1.GetTensor("leaffunc1_mat_f");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_e->tensor = mat_b->tensor;
    leaffunc1_mat_f->tensor = mat_c->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {32, 128}, "leaffunc2_mat_d");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_e");
    leafG2.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc2_mat_f");
    leafG2.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc2_mat_g");
    leafG2.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc2_mat_h");
    leafG2.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc2_mat_i");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_b = leafG2.GetTensor("leaffunc2_mat_b");
    auto leaffunc2_mat_h = leafG2.GetTensor("leaffunc2_mat_h");
    auto leaffunc2_mat_i = leafG2.GetTensor("leaffunc2_mat_i");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_b->tensor = mat_c->tensor;
    leaffunc2_mat_h->tensor = mat_d->tensor;
    leaffunc2_mat_i->tensor = mat_e->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {8, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc3_mat_d");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_e");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_f");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_b = leafG3.GetTensor("leaffunc3_mat_b");
    auto leaffunc3_mat_f = leafG3.GetTensor("leaffunc3_mat_f");
    leaffunc3_mat_a->tensor = mat_d->tensor;
    leaffunc3_mat_b->tensor = mat_e->tensor;
    leaffunc3_mat_f->tensor = mat_f->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b", "mat_c"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b", "mat_c"}, {"mat_d", "mat_e"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_d", "mat_e"}, {"mat_f"}, "call3");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_d"}, "view2");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_e"}, "copyOut1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_d"}, {"leaffunc1_mat_f"}, "copyOut2");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_c"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_b"}, {"leaffunc2_mat_d"}, "copyIn2");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_c"}, {"leaffunc2_mat_e"}, "view1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_d"}, {"leaffunc2_mat_e"}, "view2");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_e"}, {"leaffunc2_mat_f"}, "view3");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_e"}, {"leaffunc2_mat_g"}, "view4");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_f"}, {"leaffunc2_mat_h"}, "copyOut1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_g"}, {"leaffunc2_mat_i"}, "copyOut2");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_c"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_b"}, {"leaffunc3_mat_d"}, "copyIn2");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_c"}, {"leaffunc3_mat_e"}, "view1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_d"}, {"leaffunc3_mat_e"}, "view2");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_e"}, {"leaffunc3_mat_f"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_f"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_e", "leaffunc1_mat_f"});
    leafG2.SetInCast({"leaffunc2_mat_a", "leaffunc2_mat_b"});
    leafG2.SetOutCast({"leaffunc2_mat_h", "leaffunc2_mat_i"});
    leafG3.SetInCast({"leaffunc3_mat_a", "leaffunc3_mat_b"});
    leafG3.SetOutCast({"leaffunc3_mat_f"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_e,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_f,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_b,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_h,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_i,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_b,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_f,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);

    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_NE(mat_d->storage_, nullptr);
    EXPECT_NE(mat_e->storage_, nullptr);
    EXPECT_EQ(mat_f->storage_, nullptr);
    // 优先检索mat_d，mat_d复用了mat_c, mat_e复用了mat_b
    EXPECT_NE(mat_b->storage_->id_, mat_d->storage_->id_);
    EXPECT_EQ(mat_b->storage_->id_, mat_e->storage_->id_);
    EXPECT_NE(mat_c->storage_->id_, mat_e->storage_->id_);
    EXPECT_EQ(mat_c->storage_->id_, mat_d->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseMultiOpSerialConn)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    Function leafFunc4(Program::GetInstance(), "leafFunc4", "leafFunc4", G.GetFunction());
    Function leafFunc5(Program::GetInstance(), "leafFunc5", "leafFunc5", G.GetFunction());
    Function leafFunc6(Program::GetInstance(), "leafFunc6", "leafFunc6", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    leafFunc4.rootFunc_ = function;
    leafFunc5.rootFunc_ = function;
    leafFunc6.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);
    ComputationalGraphBuilder leafG4(&leafFunc4);
    ComputationalGraphBuilder leafG5(&leafFunc5);
    ComputationalGraphBuilder leafG6(&leafFunc6);

    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_d");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_e");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_f");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_g");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    auto mat_e = G.GetTensor("mat_e");
    auto mat_f = G.GetTensor("mat_f");
    auto mat_g = G.GetTensor("mat_g");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc3_mat_d");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_d = leafG3.GetTensor("leaffunc3_mat_d");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_d->tensor = mat_d->tensor;
    // add tensor for leaf func4
    leafG4.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc4_mat_a");
    leafG4.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc4_mat_b");
    leafG4.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc4_mat_c");
    leafG4.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc4_mat_d");
    auto leaffunc4_mat_a = leafG4.GetTensor("leaffunc4_mat_a");
    auto leaffunc4_mat_d = leafG4.GetTensor("leaffunc4_mat_d");
    leaffunc4_mat_a->tensor = mat_d->tensor;
    leaffunc4_mat_d->tensor = mat_e->tensor;
    // add tensor for leaf func5
    leafG5.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc5_mat_a");
    leafG5.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc5_mat_b");
    leafG5.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc5_mat_c");
    leafG5.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc5_mat_d");
    auto leaffunc5_mat_a = leafG5.GetTensor("leaffunc5_mat_a");
    auto leaffunc5_mat_d = leafG5.GetTensor("leaffunc5_mat_d");
    leaffunc5_mat_a->tensor = mat_e->tensor;
    leaffunc5_mat_d->tensor = mat_f->tensor;
    // add tensor for leaf func6
    leafG6.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc6_mat_a");
    leafG6.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc6_mat_b");
    leafG6.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc6_mat_c");
    leafG6.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc6_mat_d");
    auto leaffunc6_mat_a = leafG6.GetTensor("leaffunc6_mat_a");
    auto leaffunc6_mat_d = leafG6.GetTensor("leaffunc6_mat_d");
    leaffunc6_mat_a->tensor = mat_f->tensor;
    leaffunc6_mat_d->tensor = mat_g->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c"}, {"mat_d"}, "call3");
    G.AddOp(Opcode::OP_CALL, {"mat_d"}, {"mat_e"}, "call4");
    G.AddOp(Opcode::OP_CALL, {"mat_e"}, {"mat_f"}, "call5");
    G.AddOp(Opcode::OP_CALL, {"mat_f"}, {"mat_g"}, "call6");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_b"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_b"}, {"leaffunc3_mat_c"}, "view1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_c"}, {"leaffunc3_mat_d"}, "copyOut1");
    // add op for leaf func4
    leafG4.AddOp(Opcode::OP_COPY_IN, {"leaffunc4_mat_a"}, {"leaffunc4_mat_b"}, "copyIn1");
    leafG4.AddOp(Opcode::OP_VIEW, {"leaffunc4_mat_b"}, {"leaffunc4_mat_c"}, "view1");
    leafG4.AddOp(Opcode::OP_COPY_OUT, {"leaffunc4_mat_c"}, {"leaffunc4_mat_d"}, "copyOut1");
    // add op for leaf func5
    leafG5.AddOp(Opcode::OP_COPY_IN, {"leaffunc5_mat_a"}, {"leaffunc5_mat_b"}, "copyIn1");
    leafG5.AddOp(Opcode::OP_VIEW, {"leaffunc5_mat_b"}, {"leaffunc5_mat_c"}, "view1");
    leafG5.AddOp(Opcode::OP_COPY_OUT, {"leaffunc5_mat_c"}, {"leaffunc5_mat_d"}, "copyOut1");
    // add op for leaf func6
    leafG6.AddOp(Opcode::OP_COPY_IN, {"leaffunc6_mat_a"}, {"leaffunc6_mat_b"}, "copyIn1");
    leafG6.AddOp(Opcode::OP_VIEW, {"leaffunc6_mat_b"}, {"leaffunc6_mat_c"}, "view1");
    leafG6.AddOp(Opcode::OP_COPY_OUT, {"leaffunc6_mat_c"}, {"leaffunc6_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_g"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a"});
    leafG3.SetOutCast({"leaffunc3_mat_d"});
    leafG4.SetInCast({"leaffunc4_mat_a"});
    leafG4.SetOutCast({"leaffunc4_mat_d"});
    leafG5.SetInCast({"leaffunc5_mat_a"});
    leafG5.SetOutCast({"leaffunc5_mat_d"});
    leafG6.SetInCast({"leaffunc6_mat_a"});
    leafG6.SetOutCast({"leaffunc6_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction4;
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall4 = std::make_shared<CallOpAttribute>(leafFunc4.ComputeHash(), symbolicScalarFunction4);
    G.GetOp("call4")->SetOpAttribute(opAttributeCall4);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction5;
    SetSymbolicScalarFunction(symbolicScalarFunction5, leaffunc5_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction5, leaffunc5_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall5 = std::make_shared<CallOpAttribute>(leafFunc5.ComputeHash(), symbolicScalarFunction5);
    G.GetOp("call5")->SetOpAttribute(opAttributeCall5);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction6;
    SetSymbolicScalarFunction(symbolicScalarFunction6, leaffunc6_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction6, leaffunc6_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall6 = std::make_shared<CallOpAttribute>(leafFunc6.ComputeHash(), symbolicScalarFunction6);
    G.GetOp("call6")->SetOpAttribute(opAttributeCall6);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3
    function->programs_.emplace(4, &leafFunc4); // 索引4 绑定 leafFunc4
    function->programs_.emplace(5, &leafFunc5); // 索引5 绑定 leafFunc5
    function->programs_.emplace(6, &leafFunc6); // 索引6 绑定 leafFunc6

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc4.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc5.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc6.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    cache.Insert(leafFunc4.GetFunctionHash(), leafFunc4);
    cache.Insert(leafFunc5.GetFunctionHash(), leafFunc5);
    cache.Insert(leafFunc6.GetFunctionHash(), leafFunc6);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_NE(mat_d->storage_, nullptr);
    EXPECT_NE(mat_e->storage_, nullptr);
    EXPECT_NE(mat_f->storage_, nullptr);
    EXPECT_EQ(mat_g->storage_, nullptr);
    EXPECT_EQ(mat_b->storage_->id_, mat_c->storage_->id_);
    EXPECT_EQ(mat_c->storage_->id_, mat_d->storage_->id_);
    EXPECT_EQ(mat_d->storage_->id_, mat_e->storage_->id_);
    EXPECT_EQ(mat_e->storage_->id_, mat_f->storage_->id_);
}

TEST_F(TestGlobalMemoryReuse, TestGlobalMemoryReuseMultiOpParallelConn)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    Function leafFunc1(Program::GetInstance(), "leafFunc1", "leafFunc1", G.GetFunction());
    Function leafFunc2(Program::GetInstance(), "leafFunc2", "leafFunc2", G.GetFunction());
    Function leafFunc3(Program::GetInstance(), "leafFunc3", "leafFunc3", G.GetFunction());
    Function leafFunc4(Program::GetInstance(), "leafFunc4", "leafFunc4", G.GetFunction());
    Function leafFunc5(Program::GetInstance(), "leafFunc5", "leafFunc5", G.GetFunction());
    Function leafFunc6(Program::GetInstance(), "leafFunc6", "leafFunc6", G.GetFunction());
    function->rootFunc_ = function;
    leafFunc1.rootFunc_ = function;
    leafFunc2.rootFunc_ = function;
    leafFunc3.rootFunc_ = function;
    leafFunc4.rootFunc_ = function;
    leafFunc5.rootFunc_ = function;
    leafFunc6.rootFunc_ = function;
    ComputationalGraphBuilder leafG1(&leafFunc1);
    ComputationalGraphBuilder leafG2(&leafFunc2);
    ComputationalGraphBuilder leafG3(&leafFunc3);
    ComputationalGraphBuilder leafG4(&leafFunc4);
    ComputationalGraphBuilder leafG5(&leafFunc5);
    ComputationalGraphBuilder leafG6(&leafFunc6);
    // add tensor for root func
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_a");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "mat_b");
    G.AddTensor(DataType::DT_FP16, {16, 128}, "mat_c");
    G.AddTensor(DataType::DT_FP16, {16, 128}, "mat_d");
    auto mat_a = G.GetTensor("mat_a");
    auto mat_b = G.GetTensor("mat_b");
    auto mat_c = G.GetTensor("mat_c");
    auto mat_d = G.GetTensor("mat_d");
    // add tensor for leaf func1
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_a");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_b");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_c");
    leafG1.AddTensor(DataType::DT_FP16, {64, 128}, "leaffunc1_mat_d");
    auto leaffunc1_mat_a = leafG1.GetTensor("leaffunc1_mat_a");
    auto leaffunc1_mat_d = leafG1.GetTensor("leaffunc1_mat_d");
    leaffunc1_mat_a->tensor = mat_a->tensor;
    leaffunc1_mat_d->tensor = mat_b->tensor;
    // add tensor for leaf func2
    leafG2.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc2_mat_a");
    leafG2.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc2_mat_b");
    leafG2.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc2_mat_c");
    leafG2.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc2_mat_d");
    auto leaffunc2_mat_a = leafG2.GetTensor("leaffunc2_mat_a");
    auto leaffunc2_mat_d = leafG2.GetTensor("leaffunc2_mat_d");
    leaffunc2_mat_a->tensor = mat_b->tensor;
    leaffunc2_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func3
    leafG3.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc3_mat_a");
    leafG3.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc3_mat_b");
    leafG3.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc3_mat_c");
    leafG3.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc3_mat_d");
    auto leaffunc3_mat_a = leafG3.GetTensor("leaffunc3_mat_a");
    auto leaffunc3_mat_d = leafG3.GetTensor("leaffunc3_mat_d");
    leaffunc3_mat_a->tensor = mat_c->tensor;
    leaffunc3_mat_d->tensor = mat_d->tensor;
    // add tensor for leaf func4
    leafG4.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc4_mat_a");
    leafG4.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc4_mat_b");
    leafG4.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc4_mat_c");
    leafG4.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc4_mat_d");
    auto leaffunc4_mat_a = leafG4.GetTensor("leaffunc4_mat_a");
    auto leaffunc4_mat_d = leafG4.GetTensor("leaffunc4_mat_d");
    leaffunc4_mat_a->tensor = mat_b->tensor;
    leaffunc4_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func5
    leafG5.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc5_mat_a");
    leafG5.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc5_mat_b");
    leafG5.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc5_mat_c");
    leafG5.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc5_mat_d");
    auto leaffunc5_mat_a = leafG5.GetTensor("leaffunc5_mat_a");
    auto leaffunc5_mat_d = leafG5.GetTensor("leaffunc5_mat_d");
    leaffunc5_mat_a->tensor = mat_b->tensor;
    leaffunc5_mat_d->tensor = mat_c->tensor;
    // add tensor for leaf func6
    leafG6.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc6_mat_a");
    leafG6.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc6_mat_b");
    leafG6.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc6_mat_c");
    leafG6.AddTensor(DataType::DT_FP16, {16, 128}, "leaffunc6_mat_d");
    auto leaffunc6_mat_a = leafG6.GetTensor("leaffunc6_mat_a");
    auto leaffunc6_mat_d = leafG6.GetTensor("leaffunc6_mat_d");
    leaffunc6_mat_a->tensor = mat_b->tensor;
    leaffunc6_mat_d->tensor = mat_c->tensor;
    // add op for root func
    G.AddOp(Opcode::OP_CALL, {"mat_a"}, {"mat_b"}, "call1");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call2");
    G.AddOp(Opcode::OP_CALL, {"mat_c"}, {"mat_d"}, "call3");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call4");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call5");
    G.AddOp(Opcode::OP_CALL, {"mat_b"}, {"mat_c"}, "call6");
    // add op for leaf func1
    leafG1.AddOp(Opcode::OP_COPY_IN, {"leaffunc1_mat_a"}, {"leaffunc1_mat_b"}, "copyIn1");
    leafG1.AddOp(Opcode::OP_VIEW, {"leaffunc1_mat_b"}, {"leaffunc1_mat_c"}, "view1");
    leafG1.AddOp(Opcode::OP_COPY_OUT, {"leaffunc1_mat_c"}, {"leaffunc1_mat_d"}, "copyOut1");
    // add op for leaf func2
    leafG2.AddOp(Opcode::OP_COPY_IN, {"leaffunc2_mat_a"}, {"leaffunc2_mat_b"}, "copyIn1");
    leafG2.AddOp(Opcode::OP_VIEW, {"leaffunc2_mat_b"}, {"leaffunc2_mat_c"}, "view1");
    leafG2.AddOp(Opcode::OP_COPY_OUT, {"leaffunc2_mat_c"}, {"leaffunc2_mat_d"}, "copyOut1");
    // add op for leaf func3
    leafG3.AddOp(Opcode::OP_COPY_IN, {"leaffunc3_mat_a"}, {"leaffunc3_mat_b"}, "copyIn1");
    leafG3.AddOp(Opcode::OP_VIEW, {"leaffunc3_mat_b"}, {"leaffunc3_mat_c"}, "view1");
    leafG3.AddOp(Opcode::OP_COPY_OUT, {"leaffunc3_mat_c"}, {"leaffunc3_mat_d"}, "copyOut1");
    // add op for leaf func4
    leafG4.AddOp(Opcode::OP_COPY_IN, {"leaffunc4_mat_a"}, {"leaffunc4_mat_b"}, "copyIn1");
    leafG4.AddOp(Opcode::OP_VIEW, {"leaffunc4_mat_b"}, {"leaffunc4_mat_c"}, "view1");
    leafG4.AddOp(Opcode::OP_COPY_OUT, {"leaffunc4_mat_c"}, {"leaffunc4_mat_d"}, "copyOut1");
    // add op for leaf func5
    leafG5.AddOp(Opcode::OP_COPY_IN, {"leaffunc5_mat_a"}, {"leaffunc5_mat_b"}, "copyIn1");
    leafG5.AddOp(Opcode::OP_VIEW, {"leaffunc5_mat_b"}, {"leaffunc5_mat_c"}, "view1");
    leafG5.AddOp(Opcode::OP_COPY_OUT, {"leaffunc5_mat_c"}, {"leaffunc5_mat_d"}, "copyOut1");
    // add op for leaf func6
    leafG6.AddOp(Opcode::OP_COPY_IN, {"leaffunc6_mat_a"}, {"leaffunc6_mat_b"}, "copyIn1");
    leafG6.AddOp(Opcode::OP_VIEW, {"leaffunc6_mat_b"}, {"leaffunc6_mat_c"}, "view1");
    leafG6.AddOp(Opcode::OP_COPY_OUT, {"leaffunc6_mat_c"}, {"leaffunc6_mat_d"}, "copyOut1");
    // set incast and outcast
    G.SetInCast({"mat_a"});
    G.SetOutCast({"mat_d"});
    leafG1.SetInCast({"leaffunc1_mat_a"});
    leafG1.SetOutCast({"leaffunc1_mat_d"});
    leafG2.SetInCast({"leaffunc2_mat_a"});
    leafG2.SetOutCast({"leaffunc2_mat_d"});
    leafG3.SetInCast({"leaffunc3_mat_a"});
    leafG3.SetOutCast({"leaffunc3_mat_d"});
    leafG4.SetInCast({"leaffunc4_mat_a"});
    leafG4.SetOutCast({"leaffunc4_mat_d"});
    leafG5.SetInCast({"leaffunc5_mat_a"});
    leafG5.SetOutCast({"leaffunc5_mat_d"});
    leafG6.SetInCast({"leaffunc6_mat_a"});
    leafG6.SetOutCast({"leaffunc6_mat_d"});
    // set CallOpAttribute for op
    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction1;
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction1, leaffunc1_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall1 = std::make_shared<CallOpAttribute>(leafFunc1.ComputeHash(), symbolicScalarFunction1);
    G.GetOp("call1")->SetOpAttribute(opAttributeCall1);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction2;
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction2, leaffunc2_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall2 = std::make_shared<CallOpAttribute>(leafFunc2.ComputeHash(), symbolicScalarFunction2);
    G.GetOp("call2")->SetOpAttribute(opAttributeCall2);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction3;
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_a,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction3, leaffunc3_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall3 = std::make_shared<CallOpAttribute>(leafFunc3.ComputeHash(), symbolicScalarFunction3);
    G.GetOp("call3")->SetOpAttribute(opAttributeCall3);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction4;
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_a,
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction4, leaffunc4_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall4 = std::make_shared<CallOpAttribute>(leafFunc4.ComputeHash(), symbolicScalarFunction4);
    G.GetOp("call4")->SetOpAttribute(opAttributeCall4);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction5;
    SetSymbolicScalarFunction(symbolicScalarFunction5, leaffunc5_mat_a,
                              {IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction5, leaffunc5_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall5 = std::make_shared<CallOpAttribute>(leafFunc5.ComputeHash(), symbolicScalarFunction5);
    G.GetOp("call5")->SetOpAttribute(opAttributeCall5);

    std::vector<std::vector<SymbolicScalar>> symbolicScalarFunction6;
    SetSymbolicScalarFunction(symbolicScalarFunction6, leaffunc6_mat_a,
                              {IRBuilder().CreateConstInt(48), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    SetSymbolicScalarFunction(symbolicScalarFunction6, leaffunc6_mat_d,
                              {IRBuilder().CreateConstInt(0), IRBuilder().CreateConstInt(0)},
                              {IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    auto opAttributeCall6 = std::make_shared<CallOpAttribute>(leafFunc6.ComputeHash(), symbolicScalarFunction6);
    G.GetOp("call6")->SetOpAttribute(opAttributeCall6);

    function->programs_.emplace(1, &leafFunc1); // 索引1 绑定 leafFunc1
    function->programs_.emplace(2, &leafFunc2); // 索引2 绑定 leafFunc2
    function->programs_.emplace(3, &leafFunc3); // 索引3 绑定 leafFunc3
    function->programs_.emplace(4, &leafFunc4); // 索引4 绑定 leafFunc4
    function->programs_.emplace(5, &leafFunc5); // 索引5 绑定 leafFunc5
    function->programs_.emplace(6, &leafFunc6); // 索引6 绑定 leafFunc6

    auto& cache = Program::GetInstance().GetFunctionCache();
    leafFunc1.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc2.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc3.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc4.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc5.SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc6.SetGraphType(GraphType::BLOCK_GRAPH);
    cache.Insert(leafFunc1.GetFunctionHash(), leafFunc1);
    cache.Insert(leafFunc2.GetFunctionHash(), leafFunc2);
    cache.Insert(leafFunc3.GetFunctionHash(), leafFunc3);
    cache.Insert(leafFunc4.GetFunctionHash(), leafFunc4);
    cache.Insert(leafFunc5.GetFunctionHash(), leafFunc5);
    cache.Insert(leafFunc6.GetFunctionHash(), leafFunc6);
    // run pass
    GlobalMemoryReuse passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(mat_a->storage_, nullptr);
    EXPECT_NE(mat_b->storage_, nullptr);
    EXPECT_NE(mat_c->storage_, nullptr);
    EXPECT_EQ(mat_d->storage_, nullptr);
    EXPECT_EQ(mat_b->storage_->id_, mat_c->storage_->id_);
}

// reshape 附近漏复用回归：多分枝汇聚到合并点，经 RESHAPE 后再接多消费者链。
// RESHAPE 让 t6/t7 共享 bucket，旧实现回溯时对已访问 bucket 直接 continue，漏掉上游可复用桶（退化为 5 桶）。
TEST_F(TestGlobalMemoryReuse, CanReuseAroundReshape)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL,    Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL,
                                Opcode::OP_RESHAPE, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t3"}, {"t4", "t5"},
                                                    {"t6"},       {"t7"}, {"t7"}, {"t8", "t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}};
    std::vector<std::string> opNames{"C0", "C1", "C2", "C3", "RESHAPE", "C4", "C5", "C6"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t10"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    std::vector<std::vector<SymbolicScalar>> list;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_RESHAPE) {
            continue;
        }
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    EXPECT_EQ(allocator.Allocate(), SUCCESS);
    EXPECT_EQ(allocator.size_, 16 * 16 * 4 * 3); // 修复后复用到 3 桶
}

} // namespace tile_fwk
} // namespace npu
