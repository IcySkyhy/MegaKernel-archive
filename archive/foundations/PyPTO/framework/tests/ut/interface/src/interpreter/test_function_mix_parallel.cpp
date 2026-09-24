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
 * \file test_function_mix_parallel.cpp
 * \brief Unit tests for mix-split callop grouped parallel execution in FunctionInterpreter.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/function.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk {
namespace {

class FunctionMixParallelTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(FunctionMixParallelTest, GroupedMixSplitCallOpsExecuteTwoCalleeFrames)
{
    auto rootFunc = std::make_shared<Function>(Program::GetInstance(), "mix_parallel_root", "mix_parallel_root",
                                               nullptr);
    rootFunc->SetFunctionType(FunctionType::STATIC);
    rootFunc->SetGraphType(GraphType::BLOCK_GRAPH);

    auto calleeFunc1 = std::make_shared<Function>(Program::GetInstance(), "mix_parallel_callee1",
                                                  "mix_parallel_callee1", rootFunc.get());
    auto calleeFunc2 = std::make_shared<Function>(Program::GetInstance(), "mix_parallel_callee2",
                                                  "mix_parallel_callee2", rootFunc.get());
    calleeFunc1->SetFunctionType(FunctionType::STATIC);
    calleeFunc2->SetFunctionType(FunctionType::STATIC);
    calleeFunc1->SetGraphType(GraphType::BLOCK_GRAPH);
    calleeFunc2->SetGraphType(GraphType::BLOCK_GRAPH);

    std::vector<int64_t> shape = {1};
    auto rootIn1 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    auto rootOut1 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    auto rootIn2 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    auto rootOut2 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    rootFunc->inCasts_ = {rootIn1, rootIn2};
    rootFunc->outCasts_ = {rootOut1, rootOut2};

    auto calleeIn1 = std::make_shared<LogicalTensor>(*calleeFunc1, DT_FP32, shape);
    auto calleeOut1 = std::make_shared<LogicalTensor>(*calleeFunc1, DT_FP32, shape);
    calleeFunc1->inCasts_ = {calleeIn1};
    calleeFunc1->outCasts_ = {calleeOut1};
    auto calleeIn2 = std::make_shared<LogicalTensor>(*calleeFunc2, DT_FP32, shape);
    auto calleeOut2 = std::make_shared<LogicalTensor>(*calleeFunc2, DT_FP32, shape);
    calleeFunc2->inCasts_ = {calleeIn2};
    calleeFunc2->outCasts_ = {calleeOut2};

    auto leafAttr1 = std::make_shared<LeafFuncAttribute>();
    auto leafAttr2 = std::make_shared<LeafFuncAttribute>();
    leafAttr1->mixId = 1;
    leafAttr2->mixId = 1;
    calleeFunc1->SetLeafFuncAttribute(leafAttr1);
    calleeFunc2->SetLeafFuncAttribute(leafAttr2);

    auto calleeHash1 = calleeFunc1->ComputeHash();
    auto calleeHash2 = calleeFunc2->ComputeHash();
    auto& callOp1 = rootFunc->AddRawOperation(Opcode::OP_CALL, {rootIn1}, {rootOut1});
    auto& callOp2 = rootFunc->AddRawOperation(Opcode::OP_CALL, {rootIn2}, {rootOut2});

    std::map<int, SymbolicScalar> emptyOutExpr;
    std::vector<std::vector<SymbolicScalar>> emptyArgList;
    auto callAttr1 = std::dynamic_pointer_cast<CallOpAttribute>(
        calleeFunc1->CreateCallOpAttribute(emptyArgList, emptyOutExpr));
    auto callAttr2 = std::dynamic_pointer_cast<CallOpAttribute>(
        calleeFunc2->CreateCallOpAttribute(emptyArgList, emptyOutExpr));
    ASSERT_NE(callAttr1, nullptr);
    ASSERT_NE(callAttr2, nullptr);
    callAttr1->SetCalleeHash(calleeHash1);
    callAttr2->SetCalleeHash(calleeHash2);
    callAttr1->wrapId = 100;
    callAttr2->wrapId = 100;
    callOp1.SetOpAttribute(callAttr1);
    callOp2.SetOpAttribute(callAttr2);

    FunctionInterpreter interpreter;
    interpreter.calleeHashDict[calleeHash1.GetHash()] = calleeFunc1.get();
    interpreter.calleeHashDict[calleeHash2.GetHash()] = calleeFunc2.get();

    Tensor inTensor(DT_FP32, shape);
    Tensor outTensor(DT_FP32, shape);
    auto inView1 = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(inTensor, 1.0f));
    auto inView2 = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(inTensor, 2.0f));
    auto outView1 = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(outTensor, 0.0f));
    auto outView2 = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(outTensor, 0.0f));
    auto inoutDataPair = std::make_shared<FunctionIODataPair>(
        std::vector<std::shared_ptr<LogicalTensorData>>{inView1, inView2},
        std::vector<std::shared_ptr<LogicalTensorData>>{outView1, outView2});

    auto rootCallAttr = std::dynamic_pointer_cast<CallOpAttribute>(
        calleeFunc1->CreateCallOpAttribute(emptyArgList, emptyOutExpr));
    ASSERT_NE(rootCallAttr, nullptr);
    FunctionFrame rootFrame(rootFunc.get(), nullptr, rootCallAttr, inoutDataPair, 0);

    std::vector<std::shared_ptr<FunctionFrame>> capturedFrames;
    interpreter.captureFrameList = &capturedFrames;
    interpreter.DumpBegin();

    auto operations = rootFunc->Operations();
    size_t opIdx = 0;
    bool handled = interpreter.TryExecuteMixSplitCallOps(rootFrame, operations, opIdx, operations.at(0));
    interpreter.DumpEnd();

    EXPECT_TRUE(handled);
    EXPECT_EQ(opIdx, 1U);
    ASSERT_EQ(capturedFrames.size(), 2U);

    std::set<const Operation*> executedCallops;
    for (const auto& frame : capturedFrames) {
        ASSERT_NE(frame, nullptr);
        ASSERT_NE(frame->callop, nullptr);
        executedCallops.insert(frame->callop);
    }
    EXPECT_EQ(executedCallops.size(), 2U);
    EXPECT_TRUE(executedCallops.count(&callOp1) > 0);
    EXPECT_TRUE(executedCallops.count(&callOp2) > 0);
}

TEST_F(FunctionMixParallelTest, MixGlobalTensorDictThreadSyncWithCallOps)
{
    auto rootFunc = std::make_shared<Function>(Program::GetInstance(), "mix_global_tensor_sync_root",
                                               "mix_global_tensor_sync_root", nullptr);
    rootFunc->SetFunctionType(FunctionType::STATIC);
    rootFunc->SetGraphType(GraphType::BLOCK_GRAPH);

    constexpr int kRepeatNum = 50;
    constexpr int32_t kWrapId = 77;
    // L0C_COPY_UB shares ExecuteL0CToL1, which requires 2-D logical tensor shapes.
    std::vector<int64_t> shape = {1, 1};
    Tensor tensorDesc(DT_FP32, shape);
    auto makeDynShape = [](const std::vector<int64_t>& s) {
        std::vector<SymbolicScalar> dynShape;
        dynShape.reserve(s.size());
        for (auto dim : s) {
            dynShape.emplace_back(dim);
        }
        return dynShape;
    };
    auto dynShape = makeDynShape(shape);

    auto calleeFunc1 = std::make_shared<Function>(Program::GetInstance(), "mix_global_tensor_sync_callee1",
                                                  "mix_global_tensor_sync_callee1", rootFunc.get());
    auto calleeFunc2 = std::make_shared<Function>(Program::GetInstance(), "mix_global_tensor_sync_callee2",
                                                  "mix_global_tensor_sync_callee2", rootFunc.get());
    calleeFunc1->SetFunctionType(FunctionType::STATIC);
    calleeFunc2->SetFunctionType(FunctionType::STATIC);
    calleeFunc1->SetGraphType(GraphType::BLOCK_GRAPH);
    calleeFunc2->SetGraphType(GraphType::BLOCK_GRAPH);

    auto rootIn1 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    auto rootOut1 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    auto rootIn2 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    auto rootOut2 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    rootIn1->UpdateDynValidShape(dynShape);
    rootOut1->UpdateDynValidShape(dynShape);
    rootIn2->UpdateDynValidShape(dynShape);
    rootOut2->UpdateDynValidShape(dynShape);
    rootFunc->inCasts_ = {rootIn1, rootIn2};
    rootFunc->outCasts_ = {rootOut1, rootOut2};

    auto calleeIn1 = std::make_shared<LogicalTensor>(*calleeFunc1, DT_FP32, shape);
    auto calleeOut1 = std::make_shared<LogicalTensor>(*calleeFunc1, DT_FP32, shape);
    calleeIn1->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    calleeOut1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    calleeIn1->UpdateDynValidShape(dynShape);
    calleeOut1->UpdateDynValidShape(dynShape);
    calleeFunc1->inCasts_ = {calleeIn1};
    calleeFunc1->outCasts_ = {calleeOut1};
    auto& l0cCopyUbOp1 = calleeFunc1->AddRawOperation(Opcode::OP_L0C_COPY_UB, {calleeIn1}, {calleeOut1});
    {
        // Match GenerateMoveOp::SetL0C2UBCopyAttr: CopyIn ctor (DDR->to), memoryPath {MEM_DEVICE_DDR, MEM_UB},
        // isCopyOut=false; valid shape is toDynValidShape (UB side), not CopyOut's fromDynValidShape.
        auto shapeImme = OpImmediate::Specified(shape);
        l0cCopyUbOp1.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified({0, 0}), MemoryType::MEM_UB, shapeImme, shapeImme, OpImmediate::Specified(shape)));
        auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(l0cCopyUbOp1.GetOpAttribute());
        ASSERT_NE(copyAttr, nullptr);
        copyAttr->SetToOffset(OpImmediate::Specified({0, 0}));
    }

    auto calleeIn2 = std::make_shared<LogicalTensor>(*calleeFunc2, DT_FP32, shape);
    auto calleeOut2 = std::make_shared<LogicalTensor>(*calleeFunc2, DT_FP32, shape);
    calleeIn2->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    calleeOut2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    calleeIn2->UpdateDynValidShape(dynShape);
    calleeOut2->UpdateDynValidShape(dynShape);
    calleeFunc2->inCasts_ = {calleeIn2};
    calleeFunc2->outCasts_ = {calleeOut2};
    auto& l0cCopyUbOp2 = calleeFunc2->AddRawOperation(Opcode::OP_L0C_COPY_UB, {calleeIn2}, {calleeOut2});
    {
        auto shapeImme = OpImmediate::Specified(shape);
        l0cCopyUbOp2.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified({0, 0}), MemoryType::MEM_UB, shapeImme, shapeImme, OpImmediate::Specified(shape)));
        auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(l0cCopyUbOp2.GetOpAttribute());
        ASSERT_NE(copyAttr, nullptr);
        copyAttr->SetToOffset(OpImmediate::Specified({0, 0}));
    }

    auto leafAttr1 = std::make_shared<LeafFuncAttribute>();
    auto leafAttr2 = std::make_shared<LeafFuncAttribute>();
    // Parallel grouping is by wrapId; mixId must differ so ComputeHashOrderless() does not collide for two
    // isomorphic BLOCK_GRAPH callees. Same hash would overwrite calleeHashDict and route both CALLs to one callee,
    // sharing one outcast LogicalTensor while binding different root out views — FUNC_TENSOR_DATAVIEW_MISMATCH on
    // MIX_PATH_OPS via mixGlobalTensorDict reuse.
    leafAttr1->mixId = 2;
    leafAttr2->mixId = 3;
    calleeFunc1->SetLeafFuncAttribute(leafAttr1);
    calleeFunc2->SetLeafFuncAttribute(leafAttr2);

    auto calleeHash1 = calleeFunc1->ComputeHash();
    auto calleeHash2 = calleeFunc2->ComputeHash();
    ASSERT_NE(calleeHash1.GetHash(), calleeHash2.GetHash())
        << "callee hashes must differ so calleeHashDict keeps both callees; adjust mixId or graph if this fails";
    auto& callOp1 = rootFunc->AddRawOperation(Opcode::OP_CALL, {rootIn1}, {rootOut1});
    auto& callOp2 = rootFunc->AddRawOperation(Opcode::OP_CALL, {rootIn2}, {rootOut2});

    std::map<int, SymbolicScalar> emptyOutExpr;
    std::vector<std::vector<SymbolicScalar>> emptyArgList;
    auto callAttr1 = std::dynamic_pointer_cast<CallOpAttribute>(
        calleeFunc1->CreateCallOpAttribute(emptyArgList, emptyOutExpr));
    auto callAttr2 = std::dynamic_pointer_cast<CallOpAttribute>(
        calleeFunc2->CreateCallOpAttribute(emptyArgList, emptyOutExpr));
    ASSERT_NE(callAttr1, nullptr);
    ASSERT_NE(callAttr2, nullptr);
    callAttr1->SetCalleeHash(calleeHash1);
    callAttr2->SetCalleeHash(calleeHash2);
    callAttr1->wrapId = kWrapId;
    callAttr2->wrapId = kWrapId;
    callOp1.SetOpAttribute(callAttr1);
    callOp2.SetOpAttribute(callAttr2);

    FunctionInterpreter interpreter;
    interpreter.calleeHashDict[calleeHash1.GetHash()] = calleeFunc1.get();
    interpreter.calleeHashDict[calleeHash2.GetHash()] = calleeFunc2.get();

    auto inView1 = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(tensorDesc, 1.0f));
    auto inView2 = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(tensorDesc, 2.0f));
    auto outView1 = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(tensorDesc, 0.0f));
    auto outView2 = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(tensorDesc, 0.0f));
    auto inoutDataPair = std::make_shared<FunctionIODataPair>(
        std::vector<std::shared_ptr<LogicalTensorData>>{inView1, inView2},
        std::vector<std::shared_ptr<LogicalTensorData>>{outView1, outView2});
    auto rootCallAttr = std::dynamic_pointer_cast<CallOpAttribute>(
        calleeFunc1->CreateCallOpAttribute(emptyArgList, emptyOutExpr));
    ASSERT_NE(rootCallAttr, nullptr);
    FunctionFrame rootFrame(rootFunc.get(), nullptr, rootCallAttr, inoutDataPair, 0);
    std::vector<std::shared_ptr<FunctionFrame>> capturedFrames;
    interpreter.captureFrameList = &capturedFrames;
    auto operations = rootFunc->Operations();

    for (int r = 0; r < kRepeatNum; r++) {
        interpreter.mixGlobalTensorDict.clear();
        capturedFrames.clear();
        size_t opIdx = 0;
        interpreter.DumpBegin();
        bool handled = interpreter.TryExecuteMixSplitCallOps(rootFrame, operations, opIdx, operations.at(0));
        interpreter.DumpEnd();
        ASSERT_TRUE(handled);
        ASSERT_EQ(opIdx, 1U);
        ASSERT_EQ(capturedFrames.size(), 2U);
        std::set<const Operation*> executedCallops;
        for (const auto& capturedFrame : capturedFrames) {
            ASSERT_NE(capturedFrame, nullptr);
            ASSERT_NE(capturedFrame->callop, nullptr);
            executedCallops.insert(capturedFrame->callop);
        }
        EXPECT_EQ(executedCallops.size(), 2U);
        EXPECT_TRUE(executedCallops.count(&callOp1) > 0);
        EXPECT_TRUE(executedCallops.count(&callOp2) > 0);
    }
}

TEST_F(FunctionMixParallelTest, BuildCallInOutDataPairWaitsUntilMixGlobalTensorReady)
{
    auto rootFunc = std::make_shared<Function>(Program::GetInstance(), "wait_mix_global_root", "wait_mix_global_root",
                                               nullptr);
    rootFunc->SetFunctionType(FunctionType::STATIC);
    rootFunc->SetGraphType(GraphType::BLOCK_GRAPH);

    std::vector<int64_t> shape = {1};
    auto in = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    auto out = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    auto& callOp = rootFunc->AddRawOperation(Opcode::OP_CALL, {in}, {out});

    std::map<int, SymbolicScalar> emptyOutExpr;
    std::vector<std::vector<SymbolicScalar>> emptyArgList;
    auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(
        rootFunc->CreateCallOpAttribute(emptyArgList, emptyOutExpr));
    ASSERT_NE(callAttr, nullptr);
    callAttr->wrapId = 888;
    callOp.SetOpAttribute(callAttr);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(rootFunc.get(), &callOp, callAttr, inoutDataPair, 0);
    FunctionInterpreter interpreter;
    interpreter.mixMultiThreadEnabled_ = true;

    auto waiter = std::async(std::launch::async, [&]() { return interpreter.BuildCallInOutDataPair(frame, &callOp); });

    using namespace std::chrono_literals;
    auto earlyStatus = waiter.wait_for(30ms);
    EXPECT_EQ(earlyStatus, std::future_status::timeout)
        << "BuildCallInOutDataPair should block when mixGlobalTensorDict value is not ready";

    Tensor tensorDesc(DT_FP32, shape);
    auto readyView = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(tensorDesc, 5.0f));
    {
        std::lock_guard<std::mutex> lk(interpreter.mixGlobalTensorMutex_);
        interpreter.mixGlobalTensorDict[{in, callAttr->wrapId}] = readyView;
    }
    interpreter.mixGlobalTensorCv_.notify_all();

    auto readyStatus = waiter.wait_for(500ms);
    ASSERT_EQ(readyStatus, std::future_status::ready);
    auto built = waiter.get();
    ASSERT_NE(built, nullptr);
    ASSERT_EQ(built->incastDataViewList.size(), 1U);
    ASSERT_NE(built->incastDataViewList[0], nullptr);
    EXPECT_EQ(built->incastDataViewList[0], readyView);
}

TEST_F(FunctionMixParallelTest, GetInplaceIndexIsInplaceAttr)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "inplace_test", "inplace_test", nullptr);
    func->SetFunctionType(FunctionType::STATIC);
    func->SetGraphType(GraphType::BLOCK_GRAPH);
    std::vector<int64_t> shape = {1};
    auto in = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    auto out = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    auto& op = func->AddRawOperation(Opcode::OP_CAST, {in}, {out});

    EXPECT_EQ(op.GetInplaceIndex(0), -1);
    op.SetAttribute(OP_ATTR_PREFIX + "isInplace", true);
    EXPECT_EQ(op.GetInplaceIndex(0), 0);
}

} // namespace
} // namespace npu::tile_fwk
