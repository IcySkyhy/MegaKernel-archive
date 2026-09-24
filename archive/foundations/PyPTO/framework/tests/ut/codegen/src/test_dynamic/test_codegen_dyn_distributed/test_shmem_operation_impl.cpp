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
 * \file test_shmem_operation_impl.cpp
 * \brief Unit test for codegen.
 */

#include <vector>
#include <string>

#include <gtest/gtest.h>
#include <unistd.h>

#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"
#include "tilefwk/tilefwk_op.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk::Distributed {

class TestDistributedShmemImpl : public CodegenTestBase {
private:
    DataType GetType(const Tensor& in)
    {
        DataType shmemDataType = in.GetDataType();
        if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
            shmemDataType = DT_FP32;
        }
        return shmemDataType;
    }

    std::string getTimeStamp()
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;

        std::stringstream timestamp;
        timestamp << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
        constexpr int NUM_SIX = 6;
        timestamp << "_" << std::setw(NUM_SIX) << std::setfill('0') << us;
        return timestamp.str();
    }

public:
    TestDistributedShmemImpl() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}

    void SetUp() override
    {
        CodegenTestBase::SetUp();
        std::string outputDir = "output";
        bool res = CreateDir(outputDir);
        CHECK(res) << "Failed to create directory: " << outputDir;
        std::string folderPath = outputDir + "/output_" + getTimeStamp() + "_" + std::to_string(getpid());
        setenv("TILE_FWK_OUTPUT_DIR", folderPath.c_str(), 0);
    }
};

std::string GetFunctionRawName(const std::string& functionName)
{
    std::string functionRawName = FUNCTION_PREFIX + functionName + SUB_FUNC_SUFFIX;
#if ENABLE_HIDDENLOOP
    functionRawName += HIDDEN_FUNC_SUFFIX;
#endif
    return functionRawName;
}

TEST_F(TestDistributedShmemImpl, TestAllGather)
{
    const char* group = "hcom123";
    uint32_t worldSize = 4;
    Tensor in(DT_FP16, {16, 32}, "in");
    Tensor out(DT_FP16, {64, 32}, "out");
    Shape shmemDataShape{worldSize * 16, 32};
    std::string functionName = "TestAllGather";
    FUNCTION("ALLGATHER", {in}, {out})
    {
        TileShape::Current().SetVecTile({16, 32});
        auto shmemTensor = CreateShmemTensor(group, worldSize, DT_FP16, shmemDataShape);
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            AllGather(in, in, shmemTensor, out);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    (void)GenCodeByFunction(*function);
}

TEST_F(TestDistributedShmemImpl, TestReduceScatter)
{
    const char* group = "hcom123";
    uint32_t worldSize = 4;
    Tensor in(DT_FP16, {64, 256}, "in");
    Tensor out(DT_FP16, {16, 256}, "out");
    Shape shmemDataShape = {64 / 4, 256};
    std::string functionName = "TestReduceScatter";
    FUNCTION("REDUCESCATTER", {in}, {out})
    {
        TileShape::Current().SetVecTile({64, 256});
        DataType shmemDataType = GetType(in);
        auto shmemTensor = CreateShmemTensor(group, worldSize, shmemDataType, shmemDataShape);
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            ReduceScatter(in, in, shmemTensor, DistReduceType::DIST_REDUCE_ADD, out);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    (void)GenCodeByFunction(*function);
}

TEST_F(TestDistributedShmemImpl, TestTwoShotAllReduce)
{
    const char* group = "hcom123";
    uint32_t worldSize = 4;
    Tensor in(DT_FP16, {64, 256}, "in");
    Tensor out(DT_FP16, {64, 256}, "out");
    Shape shmemDataShape = {64 / worldSize, 256};
    std::string functionName = "TestTwoShotAllReduce";
    FUNCTION("ALLREDUCE", {in}, {out})
    {
        TileShape::Current().SetVecTile({64, 256});
        DataType shmemDataType = GetType(in);
        auto shmemTensor = CreateShmemTensor(group, worldSize, shmemDataType, shmemDataShape);
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            TwoShotAllReduce(in, in, shmemTensor, out);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    (void)GenCodeByFunction(*function);
}

TEST_F(TestDistributedShmemImpl, TestOneShotAllReduce)
{
    const char* group = "hcom123";

    uint32_t worldSize = 4;
    Tensor in(DT_FP16, {64, 256}, "in");
    in.GetStorage()->UpdateDynValidShape(std::vector<SymbolicScalar>{64, 256});
    Tensor out(DT_FP16, {64, 256}, "out");
    Shape shmemDataShape = {64, 256};
    std::string functionName = "TestOneShotAllReduce";
    FUNCTION("ALLREDUCE", {in}, {out})
    {
        TileShape::Current().SetVecTile({64, 256});
        DataType shmemDataType = GetType(in);
        auto shmemTensor = CreateShmemTensor(group, worldSize, shmemDataType, shmemDataShape);
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            OneShotAllReduce(in, in, shmemTensor, out);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    (void)GenCodeByFunction(*function);
}

TEST_F(TestDistributedShmemImpl, TestShmemDataSet)
{
    Tensor predToken(DT_INT32, {1, 1}, "pred");
    Tensor out(DT_INT32, {1, 1}, "out");
    const char* group = "hcom123";
    uint32_t worldSize = 4;

    std::string functionName = "ShmemClearData";
    Shape shmemDataShape = {64, 256};
    FUNCTION(functionName + "Main", {predToken}, {out})
    {
        TileShape::Current().SetVecTile({64, 256});
        auto shmemTensor = CreateShmemTensor(group, worldSize, DT_BF16, shmemDataShape);
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            out = ShmemClearData(shmemTensor, predToken);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TileOp::Distributed::ShmemSet<bfloat16_t, 8192>)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestDistributedShmemImpl, TestShmemSignalSet)
{
    Tensor predToken(DT_INT32, {1, 1}, "predToken");
    Tensor out(DT_INT32, {1, 1}, "out");
    const char* group = "hcom123";
    uint32_t worldSize = 4;

    std::string functionName = "ShmemClearSignal";
    Shape shmemDataShape = {8, 256};
    FUNCTION(functionName + "Main", {predToken}, {out})
    {
        TileShape::Current().SetVecTile({8, 256});
        auto shmemTensor = CreateShmemTensor(group, worldSize, DT_FP16, shmemDataShape);
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            out = ShmemClearSignal(shmemTensor, predToken);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TileOp::Distributed::ShmemSet<int32_t, 4, 8, 128>)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestDistributedShmemImpl, TestShmemBarrier)
{
    const char* group = "hcom123";
    uint32_t worldSize = 4;
    int64_t row = 16;
    int64_t col = 32;
    Tensor out(DT_INT32, {1, 1}, "out");
    Tensor predToken(DT_INT32, {1, 1}, "predToken");

    std::string functionName = "ShmemBarrier";
    FUNCTION(functionName + "Main", {predToken}, {out})
    {
        TileShape::Current().SetVecTile({row, col});
        auto shmemBarrier1ShmemSignal = CreateShmemSignal(group, worldSize);
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            out = ShmemBarrier(shmemBarrier1ShmemSignal, predToken);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TileOp::Distributed::ShmemSignal<1, 8, TileOp::Distributed::AtomicType::ADD)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestDistributedShmemImpl, TestShmemLoad)
{
    Tensor out(DT_BF16, {3, 4, 64}, "out");
    Tensor predToken(DT_INT32, {1, 1}, "predToken");
    std::string functionName = "ShmemLoad";
    FUNCTION(functionName + "Main", {predToken}, {out})
    {
        TileShape::Current().SetVecTile({3, 4, 64});
        auto shmemTensor = CreateShmemTensor("hcom123", 4, DT_BF16, {3, 4, 64});
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            out = ShmemLoad(shmemTensor, 0, predToken);
        }
    }
    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TileOp::Distributed::ShmemLoad<bfloat16_t, bfloat16_t>)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestDistributedShmemImpl, TestShmemStore)
{
    Tensor in(DT_FP32, {4, 3, 16, 32}, "in");
    Tensor out(DT_INT32, {1, 1}, "out");
    Tensor predToken(DT_INT32, {1, 1}, "predToken");
    std::string functionName = "ShmemStore";
    FUNCTION(functionName + "Main", {in, predToken}, {out})
    {
        TileShape::Current().SetVecTile({4, 3, 16, 32});
        auto shmemTensor = CreateShmemTensor("hcom123", 4, DT_FP32, {4, 3, 16, 32});
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
        {
            (void)index;
            out = ShmemStore(in, shmemTensor, 0, AtomicType::ADD, predToken);
        }
    }
    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    std::string res = GenCodeByFunction(*function);
    std::string
        expect = R"!!!(TileOp::Distributed::ShmemStore<float, 16, 32, 32, TileOp::Distributed::AtomicType::ADD>)!!!";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk::Distributed
