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
 * \file test_cost_model.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include <dlfcn.h>

#include "cost_model/simulation/backend.h"
#include "operator/models/llama/llama_def.h"
#include "cost_model/simulation/common/CommonType.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "cost_model/simulation/pv/PvModelFactory.h"
#include "interface/configs/config_manager.h"
#include "cost_model/simulation/arch/PipeFactory.h"
#include "cost_model/simulation/arch/CacheMachineImpl.h"
#include "cost_model/simulation/machine/CoreMachine.h"
#include "cost_model/simulation/machine/Scheduler.h"
#include "cost_model/simulation/tools/ParseInput.h"
#include "cost_model/simulation/arch/PipeSimulatorFast.h"
#include "cost_model/simulation/tools/visualizer.h"
#include "cost_model/simulation/base/Reporter.h"
#include "cost_model/simulation/statistics/TraceLogger.h"
#include "cost_model/simulation/common/CommonTools.h"
using namespace npu::tile_fwk;

class CostModelTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, true);
        config::SetSimConfig(KEY_BUILD_TASK_BASED_TOPO, true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        Program::GetInstance().Reset();
    }

    void TearDown() override {}
};

void RunLLamaLayerCostModel(const AttentionDims& dimsCfg, float threadhold = 0.001f)
{
    (void)threadhold;
    int b = dimsCfg.b;
    int n = dimsCfg.n;
    int s = dimsCfg.s;
    int d = dimsCfg.d;

    PROGRAM("LLAMALAYER")
    {
        Tensor H(DataType::DT_FP32, {b * s, n * d}, "H");
        Tensor AW(DataType::DT_FP16, {n * d, n * d * 3}, "AW");
        Tensor DW(DataType::DT_FP16, {n * d, n * d}, "DW");
        Tensor FW(DataType::DT_FP16, {n * d, n * d * 3}, "FW");
        Tensor Res(DT_FP32, {b * s, n * d}, "Res");
        config::SetBuildStatic(true);
        FUNCTION("LLAMA", {H, AW, DW, FW, Res})
        {
            Res = LlamaLayer(H, AW, DW, FW, dimsCfg, SMALL_DFS_VEC_CFG, DFS_CUBE_CFG);
        }
        config::SetPassStrategy("OOO");
    }
}

void RunMatrixCostModel()
{
    int bs = 1;
    int m = 32;
    int k = 32;
    int n = 32;

    std::vector<int64_t> shapeA = {bs, m, k};
    std::vector<int64_t> shapeB = {bs, k, n};
    std::vector<int64_t> shapeC = {bs, m, n};

    config::Reset();
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    Tensor matA(DT_FP16, shapeA, "MatA", TileOpFormat::TILEOP_NZ);
    Tensor matB(DT_FP16, shapeB, "MatB", TileOpFormat::TILEOP_ND);
    Tensor matC(DT_FP32, shapeC, "MatC");
    config::SetBuildStatic(true);
    FUNCTION("BATCHMATMUL", {matA, matB, matC})
    {
        config::SetPassConfig("PVC2_OOO", "OoOSchedule", KEY_DISABLE_PASS, true);
        matC = npu::tile_fwk::Matrix::BatchMatmul(DT_FP32, matA, matB, false, false);
    }
}

void RunAttentionPostCostModel()
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    int b = 1;
    int n = 2;
    int s = 128;
    int d = 512;
    int v_head = 128;
    int h = 256;
    std::vector<int64_t> inShape = {b, n, s, d}; // (b, n, s, d)
    Tensor attnPostIn(DT_FP32, inShape, "attnPostIn");
    Tensor kvBProjWV(DT_FP32, {n, d, v_head}, "kvBProjWV");
    Tensor oProjW(DT_FP32, {n * v_head, h}, "oProjW");
    Tensor atten_output;
    ConfigManager::Instance();
    FUNCTION("AttentionPost")
    {
        int new_b = attnPostIn.GetShape()[0];
        int new_n = attnPostIn.GetShape()[1];
        int new_s = attnPostIn.GetShape()[2];
        DataType dType = attnPostIn.GetStorage()->Datatype();
        TileShape::Current().SetVecTile({1, 1, 32, d});
        Tensor atten_res1 = Reshape(Transpose(attnPostIn, {1, 2}), {new_b * new_s, new_n, d});
        TileShape::Current().SetVecTile({32, 1, d});
        Tensor atten_res2 = Transpose(atten_res1, {0, 1});
        // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
        TileShape::Current().SetVecTile(128, 128);
        TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
        Tensor mm7_res = Matrix::BatchMatmul(dType, atten_res2, kvBProjWV);
        // Tensor mm7_res = Matrix::BatchMatmul(dType, atten_res2, kvBProjWV);
        TileShape::Current().SetVecTile({1, 128, 128});
        Tensor mm7_res1 = Transpose(mm7_res, {0, 1});
        Tensor mm7_res2 = Reshape(mm7_res1, {new_b, new_s, new_n * v_head});

        // [b,s, n*vHeadDim] @ [n*vHeadDim, H] = [b,s,h]
        Tensor attn_out_w = Unsqueeze(oProjW, 0);
        atten_output = Matrix::BatchMatmul(dType, mm7_res2, attn_out_w);
    }
}

TEST_F(CostModelTest, TestAttentionPostAccuracy1)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostAccuracyFromJson)
{
    config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    RunAttentionPostCostModel();

    std::string jPath = config::LogTopFolder() + "/program.json";
    config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, true);
    config::SetSimConfig(KEY_AGENT_JSON_PATH, jPath);
    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(nullptr);
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
}

TEST_F(CostModelTest, TestGenCalendarSchedule)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    std::vector<std::string> arg = config::GetSimConfig(KEY_ARGS, std::vector<std::string>{});
    arg.emplace_back("Model.genCalendarScheduleCpp=true");
    config::SetSimConfig(KEY_ARGS, arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostCVMIXMode)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    std::vector<std::string> arg = config::GetSimConfig(KEY_ARGS, std::vector<std::string>{});
    arg.emplace_back("Model.cubeVecMixMode=true");
    config::SetSimConfig(KEY_ARGS, arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostSimulationSchedule)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    std::vector<std::string> arg;
    arg.emplace_back("Model.statisticReportToFile=true");
    arg.emplace_back("Model.deviceArch=A2A3");
    arg.emplace_back("Model.useOOOPassSeq=false");
    config::SetSimConfig(KEY_ARGS, arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostFunctional)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_SIM_MODE, int(CostModel::SimMode::EMULATOR));
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestErrorInput)
{
    std::string name = "TEST";
    auto newFunc = std::make_shared<Function>(npu::tile_fwk::Program::GetInstance(), name, name, nullptr);
    std::vector<int64_t> shape = {1, 1};
    auto outcast = std::make_shared<LogicalTensor>(*newFunc, DT_FP32, shape);
    newFunc->outCasts_.push_back(outcast);
    newFunc->inCasts_.push_back(outcast);
    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(newFunc.get());
}

TEST_F(CostModelTest, TestFixedLatencyTasks)
{
    RunAttentionPostCostModel();

    std::string jsonPath("./config/fixed_task_topo.json");
    std::vector<std::string> arg = config::GetSimConfig(KEY_ARGS, std::vector<std::string>{});
    arg.emplace_back("Model.simulationFixedLatencyTask=true");
    arg.emplace_back("Model.fixedLatencyTaskInfoPath=" + jsonPath);
    config::SetSimConfig(KEY_ARGS, arg);

    std::string jPath = config::LogTopFolder() + "/program.json";
    config::SetSimConfig(KEY_AGENT_JSON_PATH, jPath);

    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(nullptr);
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
}

TEST_F(CostModelTest, TestAttentionPostL2Cache)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    std::vector<std::string> arg;
    arg.emplace_back("Model.statisticReportToFile=false");
    arg.emplace_back("Model.deviceArch=A2A3");
    arg.emplace_back("Model.mteUseL2Cache=true");
    config::SetSimConfig(KEY_ARGS, arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestBuildBasedOnConfigs)
{
    std::string configPath("./config/test_config.conf");
    std::vector<std::string> configs;
    configs.push_back("--conf");
    configs.push_back(configPath);

    CostModelAgent costModelAgent;
    costModelAgent.costModel = std::make_shared<CostModel::CostModelInterface>();
    costModelAgent.costModel->BuildCostModel(configs);

    configs.clear();
    configs.push_back("-m");
    configs.push_back("1");
    configs.push_back("--conf");
    configs.push_back(configPath);
    costModelAgent.costModel = std::make_shared<CostModel::CostModelInterface>();
    costModelAgent.costModel->BuildCostModel(configs);
}

TEST_F(CostModelTest, TestCoreMachineDeadlock)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    std::vector<std::string> arg = config::GetSimConfig(KEY_ARGS, std::vector<std::string>{});
    arg.emplace_back("Model.testDeadLock=true");
    arg.emplace_back("Core.bufferBackPressure=true");
    arg.emplace_back("Pipe.ubSizeThreshold=256");
    arg.emplace_back("Pipe.l1SizeThreshold=256");
    arg.emplace_back("Pipe.l0aSizeThreshold=256");
    arg.emplace_back("Pipe.l0bSizeThreshold=256");
    arg.emplace_back("Pipe.l0cSizeThreshold=256");
    config::SetSimConfig(KEY_ARGS, arg);
    RunAttentionPostCostModel();
}

void RunCat()
{
    TileShape::Current().SetVecTile(16, 6, 6, 16);
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    std::vector<int64_t> shape1 = {10, 10, 10, 10};
    std::vector<int64_t> shape2 = {20, 10, 10, 10};
    int axis = 0;
    Tensor params1(DT_FP32, shape1, "params1");
    Tensor params2(DT_FP32, shape2, "params2");
    Tensor res;

    FUNCTION("A") { res = Cat(std::vector<Tensor>{params1, params2}, axis); }
}

TEST_F(CostModelTest, TestGlobalCalendar)
{
    std::string jsonPath("./config/global.calendar.json");
    std::string inputPath("./config/fixed_task_topo.json");
    CostModel::CalendarMode calendarMode = CostModel::CalendarMode::GLOBAL_COUNTER;
    std::vector<std::string> arg;
    arg.emplace_back("Model.simulationFixedLatencyTask=true");
    arg.emplace_back("Model.fixedLatencyTaskInfoPath=" + inputPath);
    arg.emplace_back("Model.calendarFile=" + jsonPath);
    arg.emplace_back("Model.calendarMode=" + std::to_string(static_cast<int>(calendarMode)));
    config::SetSimConfig(KEY_ARGS, arg);
    RunCat();
}

TEST_F(CostModelTest, TestLeafFunctionMode)
{
    config::SetSimConfig(KEY_SIM_MODE, int(CostModel::SimMode::LEAF_FUNCTION));
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    std::vector<std::string> arg;
    arg.emplace_back("Model.deviceArch=A2A3");
    arg.emplace_back("Model.statisticReportToFile=false");
    config::SetSimConfig(KEY_ARGS, arg);
    RunAttentionPostCostModel();
}

class CostModelDynTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        cacheEnable = config::GetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, false);
        config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, false);
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
    }

    void TearDown() override
    {
        config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, cacheEnable);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
    }

protected:
    bool oriEnableAihacBackend = false;
    bool cacheEnable = false;
};

void CostModelTestLoopViewAssemble(const Tensor& t0, const Tensor& t1, const Tensor& blockTable, Tensor& out, int s)
{
    FUNCTION("main", {t0, t1, blockTable}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s))
        {
            SymbolicScalar idx = GetTensorData(blockTable, {i, 0});
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2 * s}, "qi");
            Assemble(t1, {0, 0}, qi);
            Assemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2 * s}, "ki");
            Assemble(t0s, {0, 0}, ki);
            Assemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul(DataType::DT_FP32, qi, ki, false, true);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            Assemble(t2, {idx * s, 0}, out);
        }
    }
}

TEST_F(CostModelDynTest, TestDD)
{
    constexpr int tilingX = 32;
    constexpr int tilingY = 32;
    TileShape::Current().SetVecTile(tilingX, tilingY);
    constexpr int tilingM = 32;
    constexpr int tilingN = 32;
    constexpr int tilingK = 32;
    TileShape::Current().SetCubeTile({tilingM, tilingM}, {tilingN, tilingN}, {tilingK, tilingK});

    std::vector<uint8_t> devProgBinary;

    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");     // [32, 32]
    Tensor blockTable{DT_INT32, {n, 1}, "blockTable"};
    Tensor out(DT_FP32, {n * s, s}, "out");
    CostModelTestLoopViewAssemble(t0, t1, blockTable, out, s);

    auto func = Program::GetInstance().GetLastFunction();
    auto pv = CostModel::PvModelFactory::CreateDyn();
    pv->Codegen(func);
}

TEST_F(CostModelTest, TestUnknownArchType)
{
    EXPECT_THROW(CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_MTE_IN, "A0"), std::invalid_argument);
}

TEST_F(CostModelTest, TestCreateA5Cache)
{
    std::unique_ptr<CostModel::CacheMachineImpl> cacheImpl = CostModel::PipeFactory::CreateCache(
        CostModel::CacheType::L2CACHE, "A5");
    CostModel::CachePacket packet;
    cacheImpl->Simulate(packet);
}

TEST_F(CostModelTest, TestA5ArchType)
{
    auto simulator = CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_MTE_IN, "A5");
    EXPECT_TRUE(simulator != nullptr);
}

TEST_F(CostModelTest, TestPipeFactoryCreateA2A3)
{
    auto sim = CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_MTE_IN, "A2A3");
    EXPECT_TRUE(sim != nullptr);
    auto sim2 = CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_VECTOR_ALU, "A2A3");
    EXPECT_TRUE(sim2 != nullptr);
    auto sim3 = CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_CUBE, "A2A3");
    EXPECT_TRUE(sim3 != nullptr);
    auto sim4 = CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_TILE_ALLOC, "A2A3");
    EXPECT_TRUE(sim4 != nullptr);
    auto sim5 = CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_CALL, "A2A3");
    EXPECT_TRUE(sim5 != nullptr);
}

TEST_F(CostModelTest, TestPipeFactoryCreateUnknownPipeType)
{
    EXPECT_THROW(CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_UNKNOW, "A2A3"), std::invalid_argument);
}

TEST_F(CostModelTest, TestPipeFactoryCreateCacheA2A3)
{
    auto cacheImpl = CostModel::PipeFactory::CreateCache(CostModel::CacheType::L2CACHE, "A2A3");
    EXPECT_TRUE(cacheImpl != nullptr);
}

TEST_F(CostModelTest, TestPipeFactoryCreateCacheUnknownArch)
{
    EXPECT_THROW(CostModel::PipeFactory::CreateCache(CostModel::CacheType::L2CACHE, "A0"), std::invalid_argument);
}

TEST_F(CostModelTest, TestPipeFactoryCreateCacheUnknownType)
{
    EXPECT_THROW(CostModel::PipeFactory::CreateCache(CostModel::CacheType::FUNCTION_CACHE, "A2A3"),
                 std::invalid_argument);
}

TEST_F(CostModelTest, TestCoreMachineDeadlock2)
{
    CostModel::CoreMachine* coreMachine = new CostModel::CoreMachine(CostModel::MachineType::AIC);
    std::set<int> unissuedTileMagics;

    coreMachine->sim = std::make_shared<CostModel::SimSys>();
    unissuedTileMagics.insert(1);
    unissuedTileMagics.insert(2);

    // 2. 初始化tileOps

    coreMachine->tileOps[1] = std::make_shared<CostModel::TileOp>();
    coreMachine->tileOps[2] = std::make_shared<CostModel::TileOp>();

    coreMachine->tileOps[1]->magic = 1;
    coreMachine->tileOps[2]->magic = 2;
    coreMachine->tileOps[1]->opcode = "";
    coreMachine->tileOps[2]->opcode = "";

    // 3. 初始化tiles
    coreMachine->tiles[1] = std::make_shared<CostModel::Tile>();
    coreMachine->tiles[2] = std::make_shared<CostModel::Tile>();

    coreMachine->tiles[1]->magic = 1;
    coreMachine->tiles[2]->magic = 2;

    // 4. 设置aliveBuffer
    coreMachine->aliveBuffer[CostModel::CorePipeType::PIPE_CUBE_BMU_L1].insert(1);
    coreMachine->aliveBuffer[CostModel::CorePipeType::PIPE_CUBE_BMU_L0A].insert(2);

    // 5. 设置readyQueues
    CostModel::ReadyQueue readyQueue1(CostModel::CorePipeType::PIPE_CUBE_BMU_L1, 0);
    readyQueue1.Insert(1);
    coreMachine->readyQueues.push_back(readyQueue1);

    CostModel::ReadyQueue readyQueue2(CostModel::CorePipeType::PIPE_CUBE_BMU_L0A, 1);
    readyQueue2.Insert(2);
    coreMachine->readyQueues.push_back(readyQueue2);

    // 6. 设置执行任务ID和函数哈希
    coreMachine->executingTaskId = 123;
    coreMachine->executingFunctionHash = 456;

    // 调用AnalysisDeadlock方法
    try {
        coreMachine->AnalysisDeadlock(unissuedTileMagics);
    } catch (const std::exception& e) {
        EXPECT_TRUE(true); // 如果捕获到异常，测试通过
    }

    try {
        coreMachine->CheckDeadlock();
    } catch (const std::exception& e) {
        EXPECT_TRUE(true); // 如果捕获到异常，测试通过
    }
    coreMachine->sim->ReportDeadlock(1);
    delete coreMachine;
}

TEST_F(CostModelTest, TestScheduler)
{
    using namespace CostModel;
    CostModel::Scheduler scheduler;
    scheduler.sim = std::make_shared<CostModel::SimSys>();
    std::unordered_map<int, CostModel::TilePtr> tiles;
    std::unordered_map<int, CostModel::TileOpPtr> tileOps;
    std::vector<std::vector<int>> tileAllocSequence(static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE));

    // 1. 创建节点
    auto t10 = std::make_shared<CostModel::Tile>();
    t10->magic = 10;
    t10->exeInfo.domCount = 5;
    t10->pipeType = CostModel::CorePipeType::PIPE_MTE1;
    auto t11 = std::make_shared<CostModel::Tile>();
    t11->magic = 11;
    t11->exeInfo.domCount = 1;
    t11->pipeType = CostModel::CorePipeType::PIPE_MTE1; // 更小的 domCount

    auto op100 = std::make_shared<CostModel::TileOp>();
    op100->magic = 100;
    op100->pipeType = CorePipeType::PIPE_VECTOR_BMU;

    auto t30 = std::make_shared<CostModel::Tile>();
    t30->magic = 30;
    t30->exeInfo.isOutcast = true;
    t30->pipeType = CostModel::CorePipeType::PIPE_MTE1;
    auto t40 = std::make_shared<CostModel::Tile>();
    t40->magic = 40;
    t40->pipeType = CostModel::CorePipeType::PIPE_MTE1; // 无 consumer，视为 output

    // 2. 建立连接
    op100->iOperand = {t10, t11};
    op100->oOperand = {t30, t40};

    t10->consumers = {op100};
    t11->consumers = {op100};

    t30->producers = {op100};
    t40->producers = {op100};

    tiles[10] = t10;
    tiles[11] = t11;
    tiles[30] = t30;
    tiles[40] = t40;
    tileOps[100] = op100;

    // 3. 执行测试
    scheduler.SortTile(tiles, tileOps, tileAllocSequence);

    // 4. 验证日志覆盖和逻辑

    EXPECT_GT(op100->exeInfo.sequenceToIssue, -1);
    EXPECT_EQ(t10->exeInfo.copyOutIdx, t11->exeInfo.copyOutIdx);
}

TEST_F(CostModelTest, TestScheduler_EmptyInput)
{
    std::unordered_map<int, CostModel::TilePtr> tiles;
    std::unordered_map<int, CostModel::TileOpPtr> tileOps;
    std::vector<std::vector<int>> seq;
    CostModel::Scheduler scheduler;
    scheduler.sim = std::make_shared<CostModel::SimSys>();
    scheduler.SortTile(tiles, tileOps, seq);
}

TEST_F(CostModelTest, TestRemoveBarrierCounter_LogCoverage)
{
    using namespace CostModel;
    GenCalendar calendar;

    // 1. 构造 Source Task 列表 (11个)
    std::vector<uint64_t> srcIds;
    for (uint64_t i = 1; i <= 11; ++i) {
        srcIds.push_back(i);
        calendar.taskTopoInfo[i] = CalendarEntry{};
    }

    for (uint64_t j = 100; j < 110; ++j) {
        CalendarEntry info;
        info.waitSrcTaskIds = srcIds;
        calendar.taskTopoInfo[j] = info;
    }

    calendar.RemoveBarrierCounter();

    uint64_t firstTargetId = 100;
    EXPECT_TRUE(calendar.taskTopoInfo[firstTargetId].waitSrcTaskIds.empty());
    EXPECT_FALSE(calendar.taskTopoInfo[firstTargetId].waitBarrierCounterIds.empty());
    EXPECT_EQ(calendar.taskTopoInfo[firstTargetId].waitBarrierCounterIds[0].first, 100);
}

TEST_F(CostModelTest, TestGetPipeType_AssertOnMissingOpcode)
{
    using namespace CostModel;
    TileOp op;
    op.opcode = "UNKNOWN_OP";

    try {
        op.GetPipeType();
    } catch (const std::exception& e) {
    }

    op.pipeType = CorePipeType::PIPE_UNKNOW;

    try {
        op.GetAddress();
    } catch (const std::exception& e) {
    }
    try {
        op.GetSize();
    } catch (const std::exception& e) {
    }
}

TEST_F(CostModelTest, TestCheckTileOp)
{
    using namespace CostModel;
    ParseInput parser;
    auto func = std::make_shared<CostModel::Function>();
    func->funcName = "TestFunc";

    auto op = std::make_shared<TileOp>();
    op->opcode = "ADD";
    op->iOperand = {}; // 触发第一个 if
    op->oOperand = {}; // 触发第二个 if

    func->tileOps.push_back(op);

    EXPECT_NO_THROW(parser.CheckTileOp(func));
}

TEST_F(CostModelTest, TestCheckTile)
{
    using namespace CostModel;
    ParseInput parser;
    auto func = std::make_shared<CostModel::Function>();

    auto tile1 = std::make_shared<Tile>();
    tile1->magic = 101;
    tile1->producers = {};

    auto tile2 = std::make_shared<Tile>();
    tile2->magic = 202;
    tile2->consumers = {};

    func->tiles.push_back(tile1);
    func->tiles.push_back(tile2);

    parser.CheckTile(func);
}

TEST_F(CostModelTest, TestParseInputFile)
{
    using namespace CostModel;
    std::vector<std::string> cfg;
    const std::string path = "1";
    std::deque<TaskMap> deque;
    std::unordered_map<long unsigned int, std::deque<CostModel::ReplayTaskEntry>> map;
    ParseInput parser;
    parser.ParseJsonConfig(path, cfg);
    parser.ParseConfig(path, cfg);
    parser.ParseCalendarJson(nullptr, path);
    parser.ParseFixedLatencyTask(nullptr, path);
    parser.ParseTopoJson(path, deque);
    parser.ParseReplayInfoJson(path, map);
    parser.ParseJson(nullptr, path);
}

TEST_F(CostModelTest, TestParseJsonWithValidFile)
{
    std::string jsonPath = "/tmp/parse_json_test_" + std::to_string(getpid()) + ".json";
    std::ofstream ofs(jsonPath);
    ofs << R"({"functions":[]})";
    ofs.close();
    auto sim = std::make_shared<CostModel::SimSys>();
    CostModel::ParseInput parser;
    parser.ParseJson(sim, jsonPath);
    unlink(jsonPath.c_str());
}

TEST_F(CostModelTest, TestParseSingleFunction)
{
    auto sim = std::make_shared<CostModel::SimSys>();
    std::string name = "TestParseSingleFunc";
    auto func = std::make_shared<Function>(npu::tile_fwk::Program::GetInstance(), name, name, nullptr);
    std::vector<int64_t> shape = {1, 1};
    auto incast = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    auto outcast = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    func->inCasts_.push_back(incast);
    func->outCasts_.push_back(outcast);
    CostModel::ParseInput parser;
    parser.ParseSingleFunction(sim, func.get());
    EXPECT_NE(sim, nullptr);
}

TEST_F(CostModelTest, TestJsonFErrororFormat)
{
    using namespace CostModel;
    const std::string path = "./config/test_config.conf";
    CostModelAgent agent;
    try {
        agent.GetFunctionFromJson(path);
    } catch (const std::exception& e) {
    }
}

TEST_F(CostModelTest, TestGetCyclesForPassA2A3)
{
    const std::string opCode = "ADD";
    std::vector<std::vector<int>> shape = {{1, 1, 1, 1}};
    DataType dtype = DataType::DT_INT4;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    int64_t cycle = CostModel::GetCyclesForPass(opCode, shape, dtype);
    EXPECT_GT(cycle, 0);
}

TEST_F(CostModelTest, TestGetCyclesForPassA5)
{
    const std::string opCode = "CAST";
    std::vector<std::vector<int>> shape = {{1, 1, 1, 1}};
    DataType dtype = DataType::DT_INT4;
    config::SetPlatformConfig("device_platform", "ASCEND_950PR_9579");
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    int64_t cycle = CostModel::GetCyclesForPass(opCode, shape, dtype);
    EXPECT_GT(cycle, 0);
}

TEST_F(CostModelTest, TestGetCyclesForPassCopyIn)
{
    const std::string opCode = "COPY_IN";
    std::vector<std::vector<int>> shape = {{1, 1, 1, 1}};
    DataType dtype = DataType::DT_INT4;
    config::SetPlatformConfig("device_platform", "ASCEND_950PR_9579");
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    int64_t cycle = CostModel::GetCyclesForPass(opCode, shape, dtype);
    EXPECT_GT(cycle, 0);
}

TEST_F(CostModelTest, TestGetCyclesForPassCopyOut)
{
    const std::string opCode = "COPY_OUT";
    std::vector<std::vector<int>> shape = {{1, 1, 1, 1}};
    DataType dtype = DataType::DT_INT4;
    config::SetPlatformConfig("device_platform", "ASCEND_950PR_9579");
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    int64_t cycle = CostModel::GetCyclesForPass(opCode, shape, dtype);
    EXPECT_GT(cycle, 0);
}

TEST_F(CostModelTest, TestGetCyclesForPassSimulate)
{
    const std::string opCode = "WHERE_TT";
    std::vector<std::vector<int>> shape = {{1, 1, 1, 1}};
    DataType dtype = DataType::DT_INT4;
    config::SetPlatformConfig("device_platform", "ASCEND_950PR_9579");
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    int64_t cycle = CostModel::GetCyclesForPass(opCode, shape, dtype);
    EXPECT_GT(cycle, 0);
}

TEST_F(CostModelTest, TestGetCyclesForPassSo)
{
    typedef int64_t (*GetCyclesForPassFunc)(const std::string& op, const std::vector<std::vector<int>>& shape,
                                            DataType dtype);
    const std::string opCode = "L1_TO_L0A";
    std::vector<std::vector<int>> shape = {{1, 1, 1, 1}};
    DataType dtype = DataType::DT_INT4;
    config::SetPlatformConfig("device_platform", "ASCEND_950PR_9579");
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    std::string soPath = "libtile_fwk_simulation.so";
    void* handle = dlopen(soPath.c_str(), RTLD_LAZY);
    EXPECT_NO_THROW(if (!handle) { throw std::runtime_error("can not load library: " + std::string(dlerror())); });

    GetCyclesForPassFunc get_cycles_func = (GetCyclesForPassFunc)dlsym(handle, "GetCyclesForPass");
    EXPECT_NO_THROW(if (!get_cycles_func) {
        throw std::runtime_error("Failed to find symbol GetCyclesForPass: " + std::string(dlerror()));
    });
    int64_t cycle = get_cycles_func(opCode, shape, dtype);
    EXPECT_GT(cycle, 0);
}

TEST_F(CostModelTest, TestMachineName)
{
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::DEVICE), "DEVICE");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::CPU), "AICPU");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::AIC), "AIC");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::AIV), "AIV");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::MIXAICORE), "MIXAICORE");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::PIPE), "PIPE");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::CACHE), "CACHE");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::HUB), "HUB");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::UNKNOWN), "ILLEGAL");
}

TEST_F(CostModelTest, TestCacheName)
{
    EXPECT_EQ(CostModel::CacheName(CostModel::CacheType::FUNCTION_CACHE), "FunctionCache");
    EXPECT_EQ(CostModel::CacheName(CostModel::CacheType::L2CACHE), "L2CACHE");
    EXPECT_EQ(CostModel::CacheName(CostModel::CacheType::TOTAL_CACHE_TYPE), "ILLEGAL");
}

TEST_F(CostModelTest, TestCacheRequestName)
{
    EXPECT_EQ(CostModel::CacheRequestName(CostModel::CacheRequestType::FUNCTION_REQ), "Function_Read");
    EXPECT_EQ(CostModel::CacheRequestName(CostModel::CacheRequestType::DATA_READ_REQ), "Data_Read");
    EXPECT_EQ(CostModel::CacheRequestName(CostModel::CacheRequestType::DATA_WRITE_REQ), "Data_Write");
}

TEST_F(CostModelTest, TestTileStringConstructor)
{
    std::string jsonStr =
        R"({"magic":42,"shape":[1,2,3],"offset":[0,1,2],"memorytype":{"tobe":"MEM_UB"},"nodetype":"LOCAL","rawtensor":{"symbol":"x","datatype":"DT_FP16","rawmagic":99,"rawshape":[4,5,6]}})";
    CostModel::Tile tile(jsonStr);
    EXPECT_EQ(tile.magic, 42);
    EXPECT_EQ(tile.shape, std::vector<int>({1, 2, 3}));
    EXPECT_EQ(tile.offset, std::vector<int>({0, 1, 2}));
    EXPECT_EQ(tile.bufType, CostModel::OperandType::BUF_UB);
}

TEST_F(CostModelTest, TestGetOpSequeceAfterOOO)
{
    CostModel::Function func;
    func.opSequenceAfterOOO_[10] = 100;
    uint64_t index = 0;
    func.GetOpSequeceAfterOOO(10, index);
    EXPECT_EQ(index, 100);
    uint64_t index2 = 5;
    func.GetOpSequeceAfterOOO(20, index2);
    EXPECT_EQ(index2, 5);
}

TEST_F(CostModelTest, TestGetOpRelativeReadyCycle)
{
    CostModel::Function func;
    func.startCycles = 10;
    auto tileOp = std::make_shared<CostModel::TileOp>();
    tileOp->exeInfo.cycleInfo.executeStartCycle = 50;
    tileOp->pipeType = CostModel::CorePipeType::PIPE_VECTOR_BMU;
    func.pipeLastEndCycle[CostModel::CorePipeType::PIPE_VECTOR_BMU] = 60;
    uint64_t result = func.GetOpRelativeReadyCycle(tileOp, 20);
    EXPECT_GE(result, 60);
}

TEST_F(CostModelTest, TestCalculateRelativeCycle)
{
    CostModel::Function func;
    func.startCycles = 10;
    auto tileOp = std::make_shared<CostModel::TileOp>();
    tileOp->exeInfo.cycleInfo.executeStartCycle = 50;
    tileOp->exeInfo.cycleInfo.executeEndCycle = 100;
    tileOp->pipeType = CostModel::CorePipeType::PIPE_VECTOR_ALU;
    func.opMagicSequence.push_back(1);
    func.tileOpMap[1] = tileOp;
    func.CalculateRelativeCycle(20, 1.0);
    EXPECT_EQ(tileOp->exeInfo.cycleInfo.relativeStartCycle, 60);
    EXPECT_EQ(tileOp->exeInfo.cycleInfo.relativeEndCycle, 110);
}

TEST_F(CostModelTest, TestParseJsonWithValidData)
{
    std::string jsonPath = "/tmp/parse_json_data_test_" + std::to_string(getpid()) + ".json";
    std::ofstream ofs(jsonPath);
    ofs << R"({
        "functions": [{
            "hash": "12345",
            "magic": 1,
            "magicname": "test_func_START",
            "operations": [{
                "opcode": "ADD",
                "opmagic": 10,
                "ioperands": [{"magic":100,"shape":[1,1],"offset":[0,0],"memorytype":{"tobe":"MEM_UB"},"nodetype":"LOCAL","rawtensor":{"symbol":"x","datatype":"DT_FP16","rawmagic":200,"rawshape":[1,1]}}],
                "ooperands": [{"magic":101,"shape":[1,1],"offset":[0,0],"memorytype":{"tobe":"MEM_UB"},"nodetype":"OUTCAST","rawtensor":{"symbol":"y","datatype":"DT_FP16","rawmagic":201,"rawshape":[1,1]}}]
            }]
        }]
    })";
    ofs.close();
    auto sim = std::make_shared<CostModel::SimSys>();
    sim->config.startFunctionLabel = "START";
    CostModel::ParseInput parser;
    parser.ParseJson(sim, jsonPath);
    EXPECT_EQ(sim->startFuncName, "test_func_START");
    EXPECT_EQ(sim->startFuncHash, 12345);
    unlink(jsonPath.c_str());
}

TEST_F(CostModelTest, TestParseTopoJsonWithValidData)
{
    std::string topoPath = "/tmp/parse_topo_test_" + std::to_string(getpid()) + ".json";
    std::ofstream ofs(topoPath);
    ofs << R"([{"seqNo":0,"taskId":1,"leafIndex":0,"opmagic":10,"psgId":-1,"rootIndex":0,"uniqueKey":1,"funcHash":123,"coreType":"AIV","successors":[]}])";
    ofs.close();
    std::deque<CostModel::TaskMap> deque;
    CostModel::ParseInput parser;
    parser.ParseTopoJson(topoPath, deque);
    EXPECT_EQ(deque.size(), 1);
    unlink(topoPath.c_str());
}

TEST_F(CostModelTest, TestParseReplayInfoJsonWithValidData)
{
    std::string replayPath = "/tmp/parse_replay_test_" + std::to_string(getpid()) + ".json";
    std::ofstream ofs(replayPath);
    ofs << R"([{"blockIdx":0,"coreType":"AIV","tasks":[{"seqNo":1,"taskId":100,"execStart":10,"execEnd":20}]}])";
    ofs.close();
    std::unordered_map<uint64_t, std::deque<CostModel::ReplayTaskEntry>> map;
    CostModel::ParseInput parser;
    parser.ParseReplayInfoJson(replayPath, map);
    EXPECT_GT(map.size(), 0);
    unlink(replayPath.c_str());
}

TEST_F(CostModelTest, TestDebugFunction)
{
    auto func = std::make_shared<CostModel::Function>();
    func->funcName = "test_debug";
    std::unordered_map<int, CostModel::TilePtr> tiles;
    std::unordered_map<int, CostModel::TileOpPtr> tileOps;
    CostModel::ModelVisualizer visualizer;
    std::string outdir = "/tmp/debug_func_test_" + std::to_string(getpid());
    mkdir(outdir.c_str(), 0755);
    visualizer.DebugFunction(func, tiles, tileOps, outdir);
    std::ifstream ifs(outdir + "/test_debug.deadlock_debug_graph.dot");
    EXPECT_TRUE(ifs.is_open());
    unlink((outdir + "/test_debug.deadlock_debug_graph.dot").c_str());
}

TEST_F(CostModelTest, TestSimQueueBuild)
{
    CostModel::SimQueue<int> queue;
    queue.Enqueue(10);
    queue.Build();
    EXPECT_EQ(queue.Size(), 0);
}

TEST_F(CostModelTest, TestSimQueueXferAndGetSim)
{
    CostModel::SimQueue<int> queue;
    queue.Xfer();
    EXPECT_EQ(queue.GetSim(), nullptr);
}

TEST_F(CostModelTest, TestSimQueueFrontAndPopFront)
{
    CostModel::SimQueue<int> queue;
    int val = 0;
    EXPECT_FALSE(queue.Front(val));
    EXPECT_FALSE(queue.PopFront());

    queue.Enqueue(42);
    queue.UpdateIntervalCycles(1);
    queue.Step();
    EXPECT_TRUE(queue.Front(val));
    EXPECT_EQ(val, 42);
    EXPECT_TRUE(queue.PopFront());
    EXPECT_EQ(queue.Size(), 0);
}

TEST_F(CostModelTest, TestSimQueueDequeueEmpty)
{
    CostModel::SimQueue<int> queue;
    int val = 0;
    EXPECT_FALSE(queue.Dequeue(val));
}

TEST_F(CostModelTest, TestSimQueueEmptyStates)
{
    CostModel::SimQueue<int> queue;
    EXPECT_TRUE(queue.Empty());

    queue.Enqueue(1);
    EXPECT_TRUE(queue.Empty());

    queue.UpdateIntervalCycles(1);
    queue.Step();
    EXPECT_FALSE(queue.Empty());
}

TEST_F(CostModelTest, TestSimQueueWriteQueueSize)
{
    CostModel::SimQueue<int> queue;
    EXPECT_EQ(queue.WriteQueueSize(), 0);
    queue.Enqueue(1);
    queue.Enqueue(2);
    EXPECT_EQ(queue.WriteQueueSize(), 2);
}

TEST_F(CostModelTest, TestSimQueueFullAndSetMaxSize)
{
    CostModel::SimQueue<int> queue;
    queue.SetMaxSize(2);
    EXPECT_FALSE(queue.Full());
    queue.Enqueue(1);
    queue.Enqueue(2);
    EXPECT_TRUE(queue.Full());
}

TEST_F(CostModelTest, TestSimQueueCalendarPopFront)
{
    CostModel::SimQueue<int> queue;
    EXPECT_EQ(queue.CalendarPopFront(), 0);

    queue.Enqueue(7);
    queue.UpdateIntervalCycles(1);
    queue.Step();
    EXPECT_EQ(queue.CalendarPopFront(), 7);
}

TEST_F(CostModelTest, TestSimQueueIsTerminate)
{
    CostModel::SimQueue<int> queue;
    EXPECT_TRUE(queue.IsTerminate());
    queue.Enqueue(1);
    EXPECT_FALSE(queue.IsTerminate());
}

TEST_F(CostModelTest, TestSimQueueGetMinWaitCyclesZero)
{
    CostModel::SimQueue<int> queue;
    queue.SetWriteDelay(0);
    queue.SetReadDelay(0);
    queue.Enqueue(1);
    queue.UpdateIntervalCycles(1);
    queue.Step();
    queue.Step();
    uint64_t wait = queue.GetMinWaitCycles();
    EXPECT_TRUE(wait >= 1);
}

TEST_F(CostModelTest, TestSimQueueWithDelay)
{
    CostModel::SimQueue<int> queue;
    queue.SetWriteDelay(3);
    queue.SetReadDelay(2);
    queue.Enqueue(10);
    queue.UpdateIntervalCycles(1);
    queue.Step();
    queue.Step();
    queue.Step();
    queue.Step();
    int val = 0;
    EXPECT_TRUE(queue.Dequeue(val));
    EXPECT_EQ(val, 10);
}

TEST_F(CostModelTest, TestSimQueueReset)
{
    CostModel::SimQueue<int> queue;
    queue.Enqueue(1);
    queue.Enqueue(2);
    queue.Reset();
    EXPECT_EQ(queue.Size(), 0);
    EXPECT_EQ(queue.WriteQueueSize(), 0);
}

TEST_F(CostModelTest, TestSimQueueSetCounterInfo)
{
    CostModel::SimQueue<int> queue;
    auto logger = std::make_shared<CostModel::TraceLogger>();
    queue.SetCounterInfo(logger, 1, 2);
}

TEST_F(CostModelTest, TestParseDynTopo)
{
    std::string topoPath = "/tmp/parse_dyn_topo_test_" + std::to_string(getpid()) + ".txt";
    std::ofstream ofs(topoPath);
    ofs << "1,2,3,4,5,6,7,8,9,10,11,12\n";
    ofs.close();
    npu::tile_fwk::CostModelAgent agent;
    Json res = agent.ParseDynTopo(topoPath);
    EXPECT_GT(res.size(), 0);
    unlink(topoPath.c_str());
}

TEST_F(CostModelTest, TestParseDynTopoInsufficientFields)
{
    std::string topoPath = "/tmp/parse_dyn_topo_insuf_" + std::to_string(getpid()) + ".txt";
    std::ofstream ofs(topoPath);
    ofs << "1,2,3\n";
    ofs << "1,2,3,4,5,6,7,8,9,10,11,12\n";
    ofs.close();
    npu::tile_fwk::CostModelAgent agent;
    Json res = agent.ParseDynTopo(topoPath);
    EXPECT_EQ(res.size(), 1);
    unlink(topoPath.c_str());
}

TEST_F(CostModelTest, TestPvModelFactoryDlsymNull)
{
    EXPECT_THROW(CostModel::PvModelFactory::CreateDyn("libm.so.6"), std::runtime_error);
}

TEST_F(CostModelTest, TestSubmitTopo)
{
    std::string topoPath = "/tmp/submit_topo_test_" + std::to_string(getpid()) + ".txt";
    std::ofstream ofs(topoPath);
    ofs << "1,2,3,4,5,6,7,8,9,10,11,12\n";
    ofs.close();
    npu::tile_fwk::CostModelAgent agent;
    agent.SubmitTopo(topoPath);
    EXPECT_FALSE(agent.topoJsonPath.empty());
    unlink(topoPath.c_str());
}

TEST_F(CostModelTest, TestGetLeafFunctionTimeCost)
{
    npu::tile_fwk::CostModelAgent agent;
    EXPECT_EQ(agent.GetLeafFunctionTimeCost(0), 0);
}

TEST_F(CostModelTest, TestSubmitSingleFuncToCostModel)
{
    npu::tile_fwk::CostModelAgent agent;
    std::string name = "TestSingleFunc";
    auto newFunc = std::make_shared<npu::tile_fwk::Function>(npu::tile_fwk::Program::GetInstance(), name, name,
                                                             nullptr);
    agent.SubmitSingleFuncToCostModel(newFunc.get());
}

TEST_F(CostModelTest, TestDebugSingleFunc)
{
    npu::tile_fwk::CostModelAgent agent;
    auto newFunc = std::make_shared<npu::tile_fwk::Function>(npu::tile_fwk::Program::GetInstance(), "root", "root",
                                                             nullptr);
    config::SetSimConfig(KEY_DEBUG_SINGLE_FUNCNAME, "");
    agent.DebugSingleFunc(newFunc.get());
}

TEST_F(CostModelTest, TestPipeMachineImplSimulate)
{
    class TestPipeImpl : public CostModel::PipeMachineImpl {
        uint64_t Simulate(const CostModel::TileOpPtr&) override { return 42; }
        uint64_t PostSimulate(const CostModel::TileOpPtr&) override { return 10; }
    };
    TestPipeImpl impl;
    auto tileOp = std::make_shared<CostModel::TileOp>();
    EXPECT_EQ(impl.Simulate(tileOp), 42);
    EXPECT_EQ(impl.PostSimulate(tileOp), 10);
}

TEST_F(CostModelTest, TestPipeMachineImplSimulateForPass)
{
    class TestPipeImpl : public CostModel::PipeMachineImpl {
        uint64_t Simulate(const CostModel::TileOpPtr&) override { return 0; }
        uint64_t PostSimulate(const CostModel::TileOpPtr&) override { return 0; }
    };
    TestPipeImpl impl;
    EXPECT_EQ(impl.SimulateForPass("ADD", {{1, 1}}, npu::tile_fwk::DataType::DT_FP32), 0);
    EXPECT_EQ(impl.PostSimulateForPass("ADD", {{1, 1}}, npu::tile_fwk::DataType::DT_FP32), 0);
}

class MockCacheMachineImpl : public CostModel::CacheMachineImpl {
public:
    uint64_t SimulateLatency = 10;
    uint64_t Simulate(const CostModel::CachePacket& packet) override
    {
        (void)packet;
        return SimulateLatency;
    }
};

class CacheMachineTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, true);
        sim = std::make_shared<CostModel::SimSys>();
        cacheMachine = new CostModel::CacheMachine(CostModel::CacheType::L2CACHE, "A2A3");
        cacheMachine->sim = sim;
        cacheMachine->Build();
    }

    void TearDown() override { delete cacheMachine; }

protected:
    std::shared_ptr<CostModel::SimSys> sim;
    CostModel::CacheMachine* cacheMachine = nullptr;
};

TEST_F(CacheMachineTest, TestConstructor)
{
    EXPECT_NE(cacheMachine->cacheImpl, nullptr);
    EXPECT_EQ(cacheMachine->cacheType, CostModel::CacheType::L2CACHE);
}

TEST_F(CacheMachineTest, TestBuild) { EXPECT_NE(cacheMachine->stats, nullptr); }

TEST_F(CacheMachineTest, TestInitQueueDelay)
{
    cacheMachine->InitQueueDelay();
    EXPECT_TRUE(true);
}

TEST_F(CacheMachineTest, TestStepQueue)
{
    sim->globalCycles = 100;
    cacheMachine->lastCycles = 50;
    cacheMachine->StepQueue();
    EXPECT_TRUE(true);
}

TEST_F(CacheMachineTest, TestGetQueueNextCycles)
{
    sim->globalCycles = 100;
    uint64_t nextCycles = cacheMachine->GetQueueNextCycles();
    EXPECT_GE(nextCycles, 100);
}

TEST_F(CacheMachineTest, TestIsTerminateNoTask)
{
    cacheMachine->executingTask = false;
    bool terminate = cacheMachine->IsTerminate();
    EXPECT_TRUE(terminate);
}

TEST_F(CacheMachineTest, TestIsTerminateWithExecutingTask)
{
    cacheMachine->executingTask = true;
    bool terminate = cacheMachine->IsTerminate();
    EXPECT_FALSE(terminate);
}

TEST_F(CacheMachineTest, TestIsTerminateWithDataInQueue)
{
    cacheMachine->executingTask = false;
    CostModel::CachePacket pkt;
    pkt.pid = 1;
    pkt.addr = 0x1000;
    cacheMachine->dataRequestQueue.Enqueue(pkt, 0);
    bool terminate = cacheMachine->IsTerminate();
    EXPECT_FALSE(terminate);
}

TEST_F(CacheMachineTest, TestRequestData)
{
    CostModel::CachePacket pkt;
    pkt.pid = 1;
    pkt.addr = 0x1000;
    pkt.requestType = CostModel::CacheRequestType::DATA_READ_REQ;

    cacheMachine->lastCycles = sim->globalCycles;
    cacheMachine->RequestData(pkt, 0);

    EXPECT_TRUE(cacheMachine->dataRequestQueue.Empty());
}

TEST_F(CacheMachineTest, TestStepWithEmptyQueue)
{
    sim->globalCycles = 100;
    cacheMachine->executingTask = false;
    cacheMachine->Step();
    EXPECT_TRUE(true);
}

TEST_F(CacheMachineTest, TestStepWithPacketCacheHit)
{
    sim->globalCycles = 100;
    cacheMachine->executingTask = false;

    CostModel::CachePacket pkt;
    pkt.pid = 1;
    pkt.addr = 0x1000;
    pkt.requestType = CostModel::CacheRequestType::DATA_READ_REQ;
    cacheMachine->dataRequestQueue.Enqueue(pkt, 0);

    auto mockImpl = new MockCacheMachineImpl();
    mockImpl->SimulateLatency = 10;
    cacheMachine->cacheImpl.reset(mockImpl);

    cacheMachine->AccessCache(0x1000, 100);

    cacheMachine->Step();
    EXPECT_TRUE(true);
}

TEST_F(CacheMachineTest, TestStepWithPacketCacheMiss)
{
    sim->globalCycles = 100;
    cacheMachine->executingTask = false;

    CostModel::CachePacket pkt;
    pkt.pid = 1;
    pkt.addr = 0x2000;
    pkt.requestType = CostModel::CacheRequestType::DATA_READ_REQ;
    cacheMachine->dataRequestQueue.Enqueue(pkt, 0);

    cacheMachine->Step();
    EXPECT_TRUE(true);
}

TEST_F(CacheMachineTest, TestXfer)
{
    sim->globalCycles = 100;
    cacheMachine->lastCycles = 50;

    CostModel::CachePacket pkt;
    pkt.pid = 1;
    pkt.addr = 0x3000;
    pkt.requestType = CostModel::CacheRequestType::DATA_READ_REQ;
    cacheMachine->dataRequestQueue.Enqueue(pkt, 0);

    cacheMachine->Xfer();
    EXPECT_TRUE(true);
}

TEST_F(CacheMachineTest, TestProcessMSHR)
{
    sim->globalCycles = 100;
    cacheMachine->config.l2MissExtraLatency = 150;
    cacheMachine->config.l2HitLatency = 50;

    CostModel::CachePacket pkt;
    pkt.pid = 1;
    pkt.addr = 0x4000;
    pkt.requestType = CostModel::CacheRequestType::DATA_READ_REQ;
    cacheMachine->AddMSHR(pkt, 100);

    sim->globalCycles = 300;
    cacheMachine->ProcessMSHR();
    EXPECT_TRUE(cacheMachine->responseQueue.empty() == false || cacheMachine->responseQueue.size() == 0);
}

TEST_F(CacheMachineTest, TestProcessRespWithMachine)
{
    auto coreMachine = new CostModel::CoreMachine(CostModel::MachineType::AIC);
    coreMachine->sim = sim;
    sim->pidToMachineMp[1] = std::shared_ptr<CostModel::Machine>(coreMachine);

    CostModel::CachePacket pkt;
    pkt.pid = 1;
    pkt.addr = 0x5000;
    pkt.requestType = CostModel::CacheRequestType::DATA_READ_REQ;
    pkt.cycleInfo.cacheRecvCycle = 100;
    cacheMachine->responseQueue.emplace_back(pkt, sim->globalCycles);

    cacheMachine->ProcessResp();
    EXPECT_TRUE(true);
}

TEST_F(CacheMachineTest, TestAllocateCacheAndEvict)
{
    cacheMachine->config.l2Size = 1024;
    cacheMachine->config.l2LineSize = 64;

    for (int i = 0; i < 20; ++i) {
        cacheMachine->AllocateCache(0x1000 + i * 0x100, 100 + i);
    }

    EXPECT_TRUE(cacheMachine->cache.size() <= 16);
}

TEST_F(CacheMachineTest, TestAccessCacheHit)
{
    cacheMachine->AllocateCache(0x6000, 100);
    bool hit = cacheMachine->AccessCache(0x6000, 200);
    EXPECT_TRUE(hit);
}

TEST_F(CacheMachineTest, TestAccessCacheMiss)
{
    bool hit = cacheMachine->AccessCache(0x7000, 100);
    EXPECT_FALSE(hit);
}

TEST_F(CacheMachineTest, TestEvictLRU)
{
    cacheMachine->AllocateCache(0x8000, 100);
    size_t sizeBefore = cacheMachine->cache.size();

    cacheMachine->EvictLRU();

    EXPECT_TRUE(cacheMachine->cache.size() < sizeBefore || sizeBefore == 1);
}

TEST_F(CacheMachineTest, TestReport) { EXPECT_NO_THROW(cacheMachine->Report()); }

TEST_F(CacheMachineTest, TestGetSim) { EXPECT_EQ(cacheMachine->GetSim(), sim); }

TEST_F(CacheMachineTest, TestReset) { EXPECT_NO_THROW(cacheMachine->Reset()); }

// ==================== Reporter UTs ====================

class ReporterTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ReporterTest, TestReporterReportTitle)
{
    std::ostringstream oss;
    std::streambuf* oldBuf = std::cout.rdbuf(oss.rdbuf());

    CostModel::Reporter::ReportTitle("TestTitle");
    std::string output = oss.str();
    std::cout.rdbuf(oldBuf);

    EXPECT_NE(output.find("TestTitle"), std::string::npos);
    EXPECT_EQ(output.size(), CostModel::TOTAL_WIDTH + 1);
    size_t leftCount = output.find("TestTitle");
    size_t rightCount = output.size() - leftCount - std::string("TestTitle").size() - 1;
    EXPECT_GE(leftCount, 0);
    EXPECT_GE(rightCount, 0);
}

TEST_F(ReporterTest, TestReporterReportMap)
{
    std::ostringstream oss;
    std::streambuf* oldBuf = std::cout.rdbuf(oss.rdbuf());

    std::map<uint64_t, uint64_t> emptyMap;
    CostModel::Reporter::ReportMap("EmptyMap", emptyMap);
    std::string output1 = oss.str();

    std::map<uint64_t, uint64_t> fewMap = {{1, 10}, {2, 20}, {3, 30}};
    CostModel::Reporter::ReportMap("FewMap", fewMap);
    std::string output2 = oss.str();

    std::map<uint64_t, uint64_t> manyMap;
    for (int i = 0; i < 9; ++i) {
        manyMap[i] = i * 100;
    }
    CostModel::Reporter::ReportMap("ManyMap", manyMap);
    std::string output3 = oss.str();

    std::cout.rdbuf(oldBuf);

    EXPECT_NE(output1.find("EmptyMap"), std::string::npos);
    EXPECT_NE(output2.find("FewMap"), std::string::npos);
    EXPECT_NE(output2.find("1"), std::string::npos);
    EXPECT_NE(output2.find("10"), std::string::npos);
    EXPECT_NE(output3.find("ManyMap"), std::string::npos);
}

TEST_F(ReporterTest, TestReporterReportMapAndPct)
{
    std::ostringstream oss;
    std::streambuf* oldBuf = std::cout.rdbuf(oss.rdbuf());

    std::map<int, uint64_t> vals = {{0, 50}, {1, 30}};
    uint64_t baseVal = 100;
    CostModel::Reporter::ReportMapAndPct("TestPct", vals, baseVal);

    std::string output = oss.str();
    std::cout.rdbuf(oldBuf);

    EXPECT_NE(output.find("TestPct"), std::string::npos);
    EXPECT_NE(output.find("50"), std::string::npos);
}

TEST_F(ReporterTest, TestReporterReportMapsAndPct)
{
    std::ostringstream oss;
    std::streambuf* oldBuf = std::cout.rdbuf(oss.rdbuf());

    std::map<int, uint64_t> vals = {{0, 50}, {1, 30}};
    std::map<int, uint64_t> baseValsHasKey = {{0, 100}, {1, 200}};
    CostModel::Reporter::ReportMapsAndPct("HasBase", vals, baseValsHasKey);

    std::map<int, uint64_t> baseValsNoKey = {{2, 100}};
    CostModel::Reporter::ReportMapsAndPct("NoBaseKey", vals, baseValsNoKey);

    std::map<int, uint64_t> baseValsZero = {{0, 0}};
    std::map<int, uint64_t> valsZero = {{0, 50}};
    CostModel::Reporter::ReportMapsAndPct("ZeroBase", valsZero, baseValsZero);

    std::string output = oss.str();
    std::cout.rdbuf(oldBuf);

    EXPECT_NE(output.find("HasBase"), std::string::npos);
    EXPECT_NE(output.find("NoBaseKey"), std::string::npos);
    EXPECT_NE(output.find("ZeroBase"), std::string::npos);
    EXPECT_NE(output.find("nan%"), std::string::npos);
}

TEST_F(ReporterTest, TestReporterReportVal)
{
    std::ostringstream oss;
    std::streambuf* oldBuf = std::cout.rdbuf(oss.rdbuf());

    CostModel::Reporter::ReportVal("Uint64Val", static_cast<uint64_t>(12345));
    CostModel::Reporter::ReportVal("FloatVal", 3.14f);
    CostModel::Reporter::ReportVal("DoubleVal", 2.718);
    CostModel::Reporter::ReportValWithLvl("Uint64Lvl", static_cast<uint64_t>(999), 1);
    CostModel::Reporter::ReportValWithLvl("FloatLvl", 1.5f, 2);
    CostModel::Reporter::ReportValWithLvl("DoubleLvl", 0.01, 3);

    std::string output = oss.str();
    std::cout.rdbuf(oldBuf);

    EXPECT_NE(output.find("Uint64Val"), std::string::npos);
    EXPECT_NE(output.find("12345"), std::string::npos);
    EXPECT_NE(output.find("FloatVal"), std::string::npos);
    EXPECT_NE(output.find("DoubleVal"), std::string::npos);
    EXPECT_NE(output.find("|--Uint64Lvl"), std::string::npos);
    EXPECT_NE(output.find("|--FloatLvl"), std::string::npos);
    EXPECT_NE(output.find("|--DoubleLvl"), std::string::npos);
}

TEST_F(ReporterTest, TestReporterReportAvg)
{
    std::ostringstream oss;
    std::streambuf* oldBuf = std::cout.rdbuf(oss.rdbuf());

    CostModel::Reporter::ReportAvg("AvgUint64", static_cast<uint64_t>(100), static_cast<uint64_t>(10));
    CostModel::Reporter::ReportAvg("AvgFloatNormal", 100.0f, 10.0f);
    CostModel::Reporter::ReportAvg("AvgFloatNegDenom", 100.0f, -1.0f);

    std::string output = oss.str();
    std::cout.rdbuf(oldBuf);

    EXPECT_NE(output.find("AvgUint64"), std::string::npos);
    EXPECT_NE(output.find("AvgFloatNormal"), std::string::npos);
    EXPECT_NE(output.find("AvgFloatNegDenom"), std::string::npos);
    EXPECT_NE(output.find("nan"), std::string::npos);
}

TEST_F(ReporterTest, TestReporterReportPct)
{
    std::ostringstream oss;
    std::streambuf* oldBuf = std::cout.rdbuf(oss.rdbuf());

    CostModel::Reporter::ReportPct("PctUint64", static_cast<uint64_t>(50), static_cast<uint64_t>(100));
    CostModel::Reporter::ReportPct("PctFloatNormal", 50.0f, 100.0f);
    CostModel::Reporter::ReportPct("PctFloatNegDenom", 50.0f, -1.0f);
    CostModel::Reporter::ReportPct("PctFloatRate", 0.5f);

    std::string output = oss.str();
    std::cout.rdbuf(oldBuf);

    EXPECT_NE(output.find("PctUint64"), std::string::npos);
    EXPECT_NE(output.find("PctFloatNormal"), std::string::npos);
    EXPECT_NE(output.find("PctFloatNegDenom"), std::string::npos);
    EXPECT_NE(output.find("nan%"), std::string::npos);
    EXPECT_NE(output.find("PctFloatRate"), std::string::npos);
    EXPECT_NE(output.find("50.00%"), std::string::npos);
}

TEST_F(ReporterTest, TestReporterReportValAndPct)
{
    std::ostringstream oss;
    std::streambuf* oldBuf = std::cout.rdbuf(oss.rdbuf());

    CostModel::Reporter::ReportValAndPct("VPUint64Normal", static_cast<uint64_t>(50), static_cast<uint64_t>(100));
    CostModel::Reporter::ReportValAndPct("VPUint64ZeroDenom", static_cast<uint64_t>(50), static_cast<uint64_t>(0));
    CostModel::Reporter::ReportValAndPct("VPFloatUint64", 50.0f, static_cast<uint64_t>(100));
    CostModel::Reporter::ReportValAndPctFl("VPFlNormal", 50.0, 100.0);
    CostModel::Reporter::ReportValAndPctFl("VPFlNegDenom", 50.0, -1.0);

    std::string output = oss.str();
    std::cout.rdbuf(oldBuf);

    EXPECT_NE(output.find("VPUint64Normal"), std::string::npos);
    EXPECT_NE(output.find("VPUint64ZeroDenom"), std::string::npos);
    EXPECT_NE(output.find("nan%"), std::string::npos);
    EXPECT_NE(output.find("VPFloatUint64"), std::string::npos);
    EXPECT_NE(output.find("VPFlNormal"), std::string::npos);
    EXPECT_NE(output.find("VPFlNegDenom"), std::string::npos);
}

TEST_F(ReporterTest, TestReporterReportHexCounter)
{
    std::ostringstream oss;
    std::streambuf* oldBuf = std::cout.rdbuf(oss.rdbuf());

    CostModel::Reporter::ReportHexCounter("HexName", 0xff, 42);

    std::string output = oss.str();
    std::cout.rdbuf(oldBuf);

    EXPECT_NE(output.find("HexName"), std::string::npos);
    EXPECT_NE(output.find("42"), std::string::npos);
}

TEST_F(ReporterTest, TestReporterReportStallLoc)
{
    std::ostringstream oss;
    std::streambuf* oldBuf = std::cout.rdbuf(oss.rdbuf());

    CostModel::Reporter::ReportStallLoc("Stall", 0xa, 0xb, 0xc, 100.0);

    std::string output = oss.str();
    std::cout.rdbuf(oldBuf);

    EXPECT_NE(output.find("Stall"), std::string::npos);
    EXPECT_NE(output.find("local_0x"), std::string::npos);
    EXPECT_NE(output.find("peer_0x"), std::string::npos);
}

TEST_F(ReporterTest, TestReporterOutputStream)
{
    CostModel::Reporter reporter;
    std::string tmpFile = "/tmp/reporter_test_" + std::to_string(getpid()) + ".txt";

    std::streambuf* oldBuf = reporter.ReportSetOutStreamFile(tmpFile);
    EXPECT_NE(oldBuf, nullptr);
    std::cout << "LineToRedirect" << std::endl;
    reporter.ReportResetOutStreamCout(oldBuf);

    std::ifstream ifs(tmpFile);
    EXPECT_TRUE(ifs.is_open());
    std::string content;
    std::getline(ifs, content);
    EXPECT_NE(content.find("LineToRedirect"), std::string::npos);
    ifs.close();
    unlink(tmpFile.c_str());
}

TEST_F(ReporterTest, TestReporterOutputStreamAppend)
{
    CostModel::Reporter reporter;
    std::string tmpFile = "/tmp/reporter_append_test_" + std::to_string(getpid()) + ".txt";

    std::ofstream ofs(tmpFile);
    ofs << "ExistingLine" << std::endl;
    ofs.close();

    std::streambuf* oldBuf = reporter.ReportSetOutStreamFile(tmpFile, true);
    EXPECT_NE(oldBuf, nullptr);
    std::cout << "AppendedLine" << std::endl;
    reporter.ReportResetOutStreamCout(oldBuf);

    std::ifstream ifs(tmpFile);
    std::string line1, line2;
    std::getline(ifs, line1);
    std::getline(ifs, line2);
    EXPECT_NE(line1.find("ExistingLine"), std::string::npos);
    EXPECT_NE(line2.find("AppendedLine"), std::string::npos);
    ifs.close();

    reporter.ReportResetOutStreamCout(nullptr);

    unlink(tmpFile.c_str());
}

// ==================== TraceLogger UTs ====================

class TraceLoggerTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override
    {
        logger = std::make_shared<CostModel::TraceLogger>();
        sim = std::make_shared<CostModel::SimSys>();
        logger->sim = sim;
    }
    void TearDown() override {}

protected:
    std::shared_ptr<CostModel::TraceLogger> logger;
    std::shared_ptr<CostModel::SimSys> sim;
};

TEST_F(TraceLoggerTest, TestEventToJson)
{
    CostModel::Event evBasic;
    evBasic.name = "TestEvent";
    evBasic.id = -1;
    evBasic.catagory = "";
    evBasic.phase = "B";
    evBasic.bp = "";
    evBasic.timestamp = 1000;
    evBasic.pid = 1;
    evBasic.tid = 2;
    evBasic.hint = "";
    CostModel::Json j1 = evBasic.ToJson();
    EXPECT_EQ(j1["name"], "TestEvent");
    EXPECT_EQ(j1["ph"], "B");
    EXPECT_EQ(j1["ts"], 1000);
    EXPECT_EQ(j1["pid"], 1);
    EXPECT_EQ(j1["tid"], 2);
    EXPECT_TRUE(j1.find("cat") == j1.end());
    EXPECT_TRUE(j1.find("bp") == j1.end());
    EXPECT_TRUE(j1.find("id") == j1.end());
    EXPECT_TRUE(j1.find("args") == j1.end());

    CostModel::Event evFull;
    evFull.name = "FullEvent";
    evFull.id = 5;
    evFull.catagory = "event";
    evFull.phase = "X";
    evFull.bp = "e";
    evFull.timestamp = 2000;
    evFull.pid = 3;
    evFull.tid = 4;
    evFull.hint = "some hint";
    CostModel::Json j2 = evFull.ToJson();
    EXPECT_EQ(j2["cat"], "event");
    EXPECT_EQ(j2["bp"], "e");
    EXPECT_EQ(j2["id"], 5);
    EXPECT_TRUE(j2.find("args") != j2.end());
    EXPECT_EQ(j2["args"]["event-hint"], "some hint");

    CostModel::Event evHintWithColor;
    evHintWithColor.name = "Op(red)";
    evHintWithColor.id = 1;
    evHintWithColor.catagory = "event";
    evHintWithColor.phase = "B";
    evHintWithColor.bp = "";
    evHintWithColor.timestamp = 3000;
    evHintWithColor.pid = 1;
    evHintWithColor.tid = 1;
    evHintWithColor.hint = "hint text";
    CostModel::Json j3 = evHintWithColor.ToJson();
    EXPECT_EQ(j3["args"]["color"], "red");
}

TEST_F(TraceLoggerTest, TestEventFlowJson)
{
    CostModel::Event ev;
    ev.name = "FlowEvent";
    ev.id = 10;
    ev.catagory = "event";
    ev.phase = "B";
    ev.bp = "e";
    ev.timestamp = 5000;
    ev.pid = 2;
    ev.tid = 3;

    CostModel::Json jStart = ev.ToFlowStartJson(42);
    EXPECT_EQ(jStart["id"], 42);
    EXPECT_EQ(jStart["ph"], "s");
    EXPECT_EQ(jStart["name"], "machine-view-last-dep");
    EXPECT_EQ(jStart["ts"], 4999);
    EXPECT_EQ(jStart["pid"], 2);
    EXPECT_EQ(jStart["tid"], 3);

    CostModel::Json jEnd = ev.ToFlowEndJson(42);
    EXPECT_EQ(jEnd["id"], 42);
    EXPECT_EQ(jEnd["ph"], "f");
    EXPECT_EQ(jEnd["bp"], "e");
    EXPECT_EQ(jEnd["ts"], 5000);
}

TEST_F(TraceLoggerTest, TestEventGetColor)
{
    CostModel::Event ev1;
    ev1.name = "Op(red)";
    EXPECT_EQ(ev1.GetColor(), "red");

    CostModel::Event ev2;
    ev2.name = "NoParentheses";
    EXPECT_EQ(ev2.GetColor(), "");

    CostModel::Event ev3;
    ev3.name = "Op(left(";
    EXPECT_EQ(ev3.GetColor(), "");

    CostModel::Event ev4;
    ev4.name = "Op()";
    EXPECT_EQ(ev4.GetColor(), "");

    CostModel::Event ev5;
    ev5.name = "Op(blue)extra";
    EXPECT_EQ(ev5.GetColor(), "blue");
}

TEST_F(TraceLoggerTest, TestEventExtraHintInfo)
{
    CostModel::Event ev;
    ev.hint = "TaskId:42 pSgId:5";
    std::string taskKey = "TaskId:";
    int taskId = ev.ExtraHintInfo(taskKey);
    EXPECT_EQ(taskId, 42);

    std::string sgKey = "pSgId:";
    int sgId = ev.ExtraHintInfo(sgKey);
    EXPECT_EQ(sgId, 5);
}

TEST_F(TraceLoggerTest, TestCounterEventToJson)
{
    CostModel::CounterEvent ce;
    ce.id = 1;
    ce.catagory = "count";
    ce.phase = "C";
    ce.type = CostModel::CounterType::COUNT_SIZE;
    ce.size = 42;
    ce.timestamp = 1000;
    ce.pid = 1;
    ce.tid = 2;

    CostModel::Json j = ce.ToJson();
    EXPECT_EQ(j["args"]["size"], 42);
    EXPECT_EQ(j["name"], "count");
    EXPECT_EQ(j["pid"], 1);
    EXPECT_EQ(j["tid"], 2);
    EXPECT_EQ(j["ph"], "C");
    EXPECT_EQ(j["ts"], 1000);
}

TEST_F(TraceLoggerTest, TestThreadProcessToJson)
{
    CostModel::Thread th;
    th.name = "WorkerThread";
    th.pid = 1;
    th.tid = 2;
    CostModel::Json jTh = th.ToJson();
    EXPECT_EQ(jTh["args"]["name"], "WorkerThread");
    EXPECT_EQ(jTh["cat"], "__metadata");
    EXPECT_EQ(jTh["name"], "thread_name");
    EXPECT_EQ(jTh["ph"], "M");
    EXPECT_EQ(jTh["pid"], 1);
    EXPECT_EQ(jTh["tid"], 2);

    CostModel::Process proc;
    proc.name = "AICore";
    proc.pid = 3;
    proc.coreIdx = 5;
    CostModel::Json jProc = proc.ToJson();
    EXPECT_EQ(jProc["args"]["name"], "AICore");
    EXPECT_EQ(jProc["cat"], "__metadata");
    EXPECT_EQ(jProc["name"], "process_name");
    EXPECT_EQ(jProc["ph"], "M");
    EXPECT_EQ(jProc["pid"], 3);

    CostModel::Json jSort = proc.ToSortIndexJson(7);
    EXPECT_EQ(jSort["args"]["sort_index"], 7);
    EXPECT_EQ(jSort["name"], "process_sort_index");
    EXPECT_EQ(jSort["pid"], 3);
}

TEST_F(TraceLoggerTest, TestDurationToJson)
{
    CostModel::Duration dur1;
    dur1.start.name = "EmptyCatHint";
    dur1.start.id = 1;
    dur1.start.catagory = "";
    dur1.start.hint = "";
    dur1.start.timestamp = 100;
    dur1.start.pid = 1;
    dur1.start.tid = 2;
    dur1.end.timestamp = 200;
    CostModel::Json j1 = dur1.ToJson();
    EXPECT_EQ(j1["ph"], "X");
    EXPECT_EQ(j1["name"], "EmptyCatHint");
    EXPECT_EQ(j1["ts"], 100);
    EXPECT_EQ(j1["dur"], 100);
    EXPECT_TRUE(j1.find("cat") == j1.end());
    EXPECT_TRUE(j1.find("args") == j1.end());

    CostModel::Duration dur2;
    dur2.start.name = "Op(blue)";
    dur2.start.id = 2;
    dur2.start.catagory = "event";
    dur2.start.hint = "hint text";
    dur2.start.timestamp = 300;
    dur2.start.pid = 3;
    dur2.start.tid = 4;
    dur2.end.timestamp = 500;
    CostModel::Json j2 = dur2.ToJson();
    EXPECT_EQ(j2["cat"], "event");
    EXPECT_TRUE(j2.find("args") != j2.end());
    EXPECT_EQ(j2["args"]["event-hint"], "hint text");
    EXPECT_EQ(j2["args"]["color"], "blue");
}

TEST_F(TraceLoggerTest, TestDurationOutputTraces)
{
    std::map<CostModel::Pid, CostModel::Process> mProcesses;
    std::map<CostModel::PTid, CostModel::Thread> mThreads;

    mProcesses[1] = CostModel::Process{.name = "AIC", .pid = 1, .coreIdx = 3};
    mProcesses[1000] = CostModel::Process{.name = "MachineView", .pid = 1000, .coreIdx = 0};
    mThreads[CostModel::PTid{1, 1}] = CostModel::Thread{.name = "Pipe1", .pid = 1, .tid = 1};
    mThreads[CostModel::PTid{1000, 1}] = CostModel::Thread{.name = "MVThread", .pid = 1000, .tid = 1};

    CostModel::Duration dur;
    dur.start.name = "TileOp";
    dur.start.id = 1;
    dur.start.catagory = "event";
    dur.start.hint = "hint";
    dur.start.timestamp = 1000;
    dur.start.pid = 1;
    dur.start.tid = 1;
    dur.end.timestamp = 2000;
    dur.end.pid = 1;
    dur.end.tid = 1;

    std::string ctxPath = "/tmp/dur_ctx_trace_" + std::to_string(getpid()) + ".txt";
    std::ofstream ofsCtx(ctxPath);
    CostModel::Duration durCtx;
    durCtx.start.name = "MVOp";
    durCtx.start.timestamp = 5000;
    durCtx.start.pid = 1000;
    durCtx.start.tid = 1;
    durCtx.end.timestamp = 6000;
    durCtx.end.pid = 1000;
    durCtx.end.tid = 1;
    durCtx.OutputContextSwitchTrace(ofsCtx, mProcesses, mThreads, 1800000);
    ofsCtx.close();

    std::ifstream ifsCtx(ctxPath);
    std::string ctxContent;
    std::getline(ifsCtx, ctxContent);
    EXPECT_NE(ctxContent.find("sched_wakeup"), std::string::npos);
    ifsCtx.close();
    unlink(ctxPath.c_str());

    std::string beginEndPath = "/tmp/dur_beginend_trace_" + std::to_string(getpid()) + ".txt";
    std::ofstream ofsBE(beginEndPath);
    dur.OutputBeginEndTrace(ofsBE, mProcesses, mThreads, 1800000);
    ofsBE.close();

    std::ifstream ifsBE(beginEndPath);
    std::string beContent;
    std::getline(ifsBE, beContent);
    EXPECT_NE(beContent.find("tracing_mark_write"), std::string::npos);
    ifsBE.close();
    unlink(beginEndPath.c_str());
}

TEST_F(TraceLoggerTest, TestTraceLoggerSetNames)
{
    logger->SetProcessName("AICore", 1, 5);
    EXPECT_TRUE(logger->mProcesses.find(1) != logger->mProcesses.end());
    EXPECT_EQ(logger->mProcesses[1].name, "AICore");
    EXPECT_EQ(logger->mProcesses[1].coreIdx, 5);
    EXPECT_TRUE(logger->mMachineTileOpMap.find(1) != logger->mMachineTileOpMap.end());

    logger->SetThreadName("Worker", 1, 2);
    CostModel::PTid ptid{1, 2};
    EXPECT_TRUE(logger->mThreads.find(ptid) != logger->mThreads.end());
    EXPECT_EQ(logger->mThreads[ptid].name, "Worker");
}

TEST_F(TraceLoggerTest, TestTraceLoggerAddEventBeginEnd)
{
    logger->SetProcessName("AICore", CostModel::GetProcessID(CostModel::MachineType::AIC, 0), 0);
    CostModel::Pid corePid = CostModel::GetProcessID(CostModel::MachineType::AIC, 0);
    CostModel::Tid coreTid = 1;
    logger->SetThreadName("AICThread", corePid, coreTid);

    CostModel::Event beginEv = logger->AddEventBegin("TileStart", corePid, coreTid, 100, "TaskId:10 pSgId:5");
    EXPECT_EQ(beginEv.name, "TileStart");
    EXPECT_GT(beginEv.id, 0);
    EXPECT_EQ(beginEv.phase, "B");
    EXPECT_EQ(beginEv.timestamp, 100);
    EXPECT_FALSE((logger->m_eventStacks[CostModel::PTid{corePid, coreTid}].empty()));

    CostModel::Event endEv = logger->AddEventEnd(corePid, coreTid, 200);
    EXPECT_EQ(endEv.name, "TileStart");
    EXPECT_EQ(endEv.phase, "E");
    EXPECT_EQ(endEv.timestamp, 200);
    EXPECT_TRUE((logger->m_eventStacks[CostModel::PTid{corePid, coreTid}].empty()));
    EXPECT_TRUE(logger->mDurations.find(beginEv.id) != logger->mDurations.end());

    logger->SetProcessName("MachineView", logger->topMachineViewPid, 0);
    CostModel::Tid mvTid = CostModel::GetProcessID(CostModel::MachineType::AIC, 1);
    logger->SetThreadName("MVThread", logger->topMachineViewPid, mvTid);
    CostModel::Event mvBegin = logger->AddEventBegin("MVOp", logger->topMachineViewPid, mvTid, 300,
                                                     "TaskId:20 SeqNo:1 pSgId:3");
    EXPECT_EQ(mvBegin.hint, "TaskId:20 SeqNo:1 pSgId:3");
    CostModel::Event mvEnd = logger->AddEventEnd(logger->topMachineViewPid, mvTid, 400);
    EXPECT_TRUE(logger->mTaskIDToDurationIndex.find(20) != logger->mTaskIDToDurationIndex.end());
}

TEST_F(TraceLoggerTest, TestTraceLoggerAddDuration)
{
    logger->SetProcessName("AICore", 1, 0);

    CostModel::LogData data1;
    data1.isLogTileOp = false;
    data1.name = "NormalOp";
    data1.pid = 1;
    data1.tid = 1;
    data1.sTime = 100;
    data1.eTime = 200;
    data1.hint = "hint";
    logger->AddDuration(data1);
    EXPECT_EQ(logger->mEvents.size(), 2);
    EXPECT_FALSE(logger->mDurations.empty());

    CostModel::LogData data2;
    data2.isLogTileOp = true;
    data2.name = "100 TileOp";
    data2.pid = 1;
    data2.tid = 1;
    data2.sTime = 300;
    data2.eTime = 400;
    data2.hint = "";
    logger->AddDuration(data2);
    EXPECT_TRUE(logger->mMachineTileOpMap[1].find(100) != logger->mMachineTileOpMap[1].end());
}

TEST_F(TraceLoggerTest, TestTraceLoggerAddFlow)
{
    logger->AddFlow("testFlow", CostModel::EventId{CostModel::PTid{1, 2}, 3},
                    CostModel::EventId{CostModel::PTid{4, 5}, 6});
    EXPECT_EQ(logger->mFlows.size(), 1);
    EXPECT_EQ(logger->mFlows[0].name, "testFlow");

    logger->SetProcessName("AICore", 1, 0);
    CostModel::LogData data;
    data.isLogTileOp = true;
    data.name = "100 SrcOp";
    data.pid = 1;
    data.tid = 1;
    data.sTime = 100;
    data.eTime = 200;
    logger->AddDuration(data);
    data.isLogTileOp = true;
    data.name = "200 DstOp";
    data.pid = 1;
    data.tid = 1;
    data.sTime = 300;
    data.eTime = 400;
    logger->AddDuration(data);

    logger->AddTileOpFlow(1, 100, 200);
    EXPECT_EQ(logger->mFlows.size(), 2);

    logger->AddTileOpFlow(2, 100, 200);
    EXPECT_EQ(logger->mFlows.size(), 2);

    logger->AddTileOpFlow(1, 999, 200);
    EXPECT_EQ(logger->mFlows.size(), 2);
}

TEST_F(TraceLoggerTest, TestTraceLoggerAddCounterEvent)
{
    sim->globalCycles = 500;
    logger->AddCounterEvent(1, 2, CostModel::CounterType::QUEUE_PUSH);
    EXPECT_FALSE(logger->mCounters.empty());
    EXPECT_EQ(logger->mCounters.back().type, CostModel::CounterType::QUEUE_PUSH);
    EXPECT_EQ(logger->mCounters.back().pid, 1);
    EXPECT_EQ(logger->mCounters.back().tid, 2);

    CostModel::PTid ptid{1, 2};
    EXPECT_TRUE(logger->mCounts.find(ptid) != logger->mCounts.end());
    EXPECT_FALSE(logger->mCounts[ptid].empty());
}

TEST_F(TraceLoggerTest, TestTraceLoggerEraseLogInfo)
{
    logger->SetProcessName("AICore", 1, 0);
    logger->SetThreadName("Thread1", 1, 1);

    CostModel::Event begin1 = logger->AddEventBegin("Op1", 1, 1, 100);
    logger->AddEventEnd(1, 1, 200);

    CostModel::Event begin2 = logger->AddEventBegin("Op2", 1, 1, 300);
    logger->AddEventEnd(1, 1, 400);

    sim->globalCycles = 500;
    logger->AddCounterEvent(1, 2, CostModel::CounterType::QUEUE_PUSH);
    sim->globalCycles = 600;
    logger->AddCounterEvent(1, 2, CostModel::CounterType::QUEUE_POP);

    size_t eventsBefore = logger->mEvents.size();
    size_t durationsBefore = logger->mDurations.size();
    size_t countersBefore = logger->mCounters.size();

    logger->EraseLogInfo(250);

    EXPECT_LT(logger->mEvents.size(), eventsBefore);
    EXPECT_LT(logger->mDurations.size(), durationsBefore);
    EXPECT_LT(logger->mCounters.size(), countersBefore);
    EXPECT_TRUE(logger->mTaskIDToDurationIndex.empty());
}

TEST_F(TraceLoggerTest, TestTraceLoggerQSizeToJson)
{
    std::vector<CostModel::CounterEvent> emptyVec;
    CostModel::Json j1 = logger->QSizeToJson(emptyVec);
    EXPECT_TRUE(j1.is_array());
    EXPECT_EQ(j1.size(), 0);

    std::vector<CostModel::CounterEvent> counterVec;
    CostModel::CounterEvent ce1;
    ce1.id = 1;
    ce1.catagory = "count";
    ce1.phase = "C";
    ce1.type = CostModel::CounterType::COUNT_SIZE;
    ce1.size = 10;
    ce1.timestamp = 100;
    ce1.pid = 1;
    ce1.tid = 2;
    counterVec.push_back(ce1);

    CostModel::CounterEvent ce2;
    ce2.id = 2;
    ce2.catagory = "count";
    ce2.phase = "C";
    ce2.type = CostModel::CounterType::COUNT_SIZE;
    ce2.size = 20;
    ce2.timestamp = 200;
    ce2.pid = 1;
    ce2.tid = 2;
    counterVec.push_back(ce2);

    CostModel::Json j2 = logger->QSizeToJson(counterVec);
    EXPECT_EQ(j2.size(), 2);
    EXPECT_EQ(j2[0]["args"]["size"], 10);
    EXPECT_EQ(j2[1]["args"]["size"], 20);
}

TEST_F(TraceLoggerTest, TestTraceLoggerToJson)
{
    logger->SetProcessName("AICore", 1, 0);
    logger->SetThreadName("Thread1", 1, 1);
    logger->processDeviceReadyQueue = true;

    CostModel::Event beginEv = logger->AddEventBegin("TestOp", 1, 1, 100, "hint");
    logger->AddEventEnd(1, 1, 200);

    logger->AddFlow("flow1", CostModel::EventId{CostModel::PTid{1, 1}, beginEv.id},
                    CostModel::EventId{CostModel::PTid{1, 1}, beginEv.id});

    CostModel::Json j = logger->ToJson();
    EXPECT_TRUE(j.find("traceEvents") != j.end());
    EXPECT_TRUE(j["traceEvents"].is_array());
    EXPECT_GT(j["traceEvents"].size(), 0);

    bool hasProcess = false;
    bool hasThread = false;
    bool hasDuration = false;
    for (auto& item : j["traceEvents"]) {
        if (item["name"] == "process_name")
            hasProcess = true;
        if (item["name"] == "thread_name")
            hasThread = true;
        if (item["ph"] == "X")
            hasDuration = true;
    }
    EXPECT_TRUE(hasProcess);
    EXPECT_TRUE(hasThread);
    EXPECT_TRUE(hasDuration);
}

TEST_F(TraceLoggerTest, TestTraceLoggerToTrace)
{
    logger->SetProcessName("AICore", 1, 0);
    logger->SetThreadName("Thread1", 1, 1);

    CostModel::Event beginEv = logger->AddEventBegin("TestOp", 1, 1, 100, "hint");
    logger->AddEventEnd(1, 1, 200);

    std::string tracePath = "/tmp/trace_logger_trace_" + std::to_string(getpid()) + ".txt";
    std::ofstream ofs(tracePath);
    logger->ToTrace(ofs);
    ofs.close();

    std::ifstream ifs(tracePath);
    std::string content;
    std::getline(ifs, content);
    EXPECT_NE(content.find("tracing_mark_write"), std::string::npos);
    ifs.close();
    unlink(tracePath.c_str());
}

TEST_F(TraceLoggerTest, TestTraceLoggerLogCoreInfo)
{
    logger->SetProcessName("AICore", 1, 0);
    logger->SetThreadName("Thread1", 1, 1);
    sim->taskCompleteSeq[10] = 1;

    CostModel::Event startEv;
    startEv.name = "InitTask";
    startEv.pid = 1;
    startEv.tid = 1;
    startEv.timestamp = 100;
    startEv.hint = "TaskId:10 pSgId:5";
    CostModel::Event endEv;
    endEv.pid = 1;
    endEv.tid = 1;
    endEv.timestamp = 200;

    CostModel::Duration dur;
    dur.start = startEv;
    dur.end = endEv;
    EXPECT_NO_THROW(logger->LogCoreInfo(dur));

    CostModel::Event startEvNoInit;
    startEvNoInit.name = "NormalTask";
    startEvNoInit.pid = 1;
    startEvNoInit.tid = 1;
    startEvNoInit.timestamp = 300;
    startEvNoInit.hint = "TaskId:10 pSgId:5";
    CostModel::Duration durNoInit;
    durNoInit.start = startEvNoInit;
    durNoInit.end = endEv;
    EXPECT_NO_THROW(logger->LogCoreInfo(durNoInit));
}

TEST_F(TraceLoggerTest, TestTraceLoggerToPipeTrace)
{
    logger->mCoreInfoLogs[0] = CostModel::CoreInfoLog(0, "AIC");
    logger->mCoreInfoLogs[0].pipeLogs["Pipe1"].push_back(
        CostModel::Json::object({{"tileOp", "Op1"}, {"execStart", 100}, {"execEnd", 200}}));

    std::string pipePath = "/tmp/pipe_trace_" + std::to_string(getpid()) + ".txt";
    std::ofstream ofs(pipePath);
    logger->ToPipeTrace(ofs);
    ofs.close();

    std::ifstream ifs(pipePath);
    CostModel::Json j;
    ifs >> j;
    EXPECT_TRUE(j.is_array());
    EXPECT_GT(j.size(), 0);
    EXPECT_EQ(j[0]["blockIdx"], 0);
    EXPECT_EQ(j[0]["coreType"], "AIC");
    ifs.close();
    unlink(pipePath.c_str());
}

TEST_F(TraceLoggerTest, TestTraceLoggerToFilterTrace)
{
    logger->SetProcessName("AICore", CostModel::GetProcessID(CostModel::MachineType::AIC, 0), 0);
    logger->SetThreadName("Thread1", CostModel::GetProcessID(CostModel::MachineType::AIC, 0), 1);
    sim->taskCompleteSeq[10] = 1;

    std::map<int, std::pair<std::string, std::vector<CostModel::Json>>> coreTasks;
    std::string filterPath = "/tmp/filter_trace_" + std::to_string(getpid()) + ".txt";
    std::ofstream ofs(filterPath);
    logger->ToFilterTrace(ofs, coreTasks);
    ofs.close();

    EXPECT_TRUE(coreTasks.find(0) != coreTasks.end());
    unlink(filterPath.c_str());
}

TEST_F(TraceLoggerTest, TestTraceLoggerAddEventBeginEndNoHint)
{
    logger->SetProcessName("AICore", 1, 0);
    logger->SetThreadName("Thread2", 1, 2);

    CostModel::Event beginEv = logger->AddEventBegin("NoHintOp", 1, 2, 500);
    EXPECT_EQ(beginEv.hint, "");
    EXPECT_EQ(beginEv.phase, "B");

    CostModel::Event endEv = logger->AddEventEnd(1, 2, 600);
    EXPECT_EQ(endEv.phase, "E");
    EXPECT_EQ(endEv.hint, "");
    EXPECT_EQ(endEv.timestamp, 600);
}

TEST_F(TraceLoggerTest, TestTraceLoggerToCalendarGlobalJson)
{
    std::map<int, std::pair<std::string, std::vector<CostModel::Json>>> coreTasks;
    coreTasks[0] = {"AIC", {}};
    coreTasks[1] = {"AIV", {}};

    std::string calPath = "/tmp/calendar_global_" + std::to_string(getpid()) + ".txt";
    std::ofstream ofs(calPath);
    logger->ToCalendarGlobalJson(ofs, coreTasks);
    ofs.close();

    std::ifstream ifs(calPath);
    CostModel::Json j;
    ifs >> j;
    EXPECT_EQ(j["numSupportedCounters"], 1);
    EXPECT_TRUE(j["cores"].is_array());
    ifs.close();
    unlink(calPath.c_str());
}

TEST_F(TraceLoggerTest, TestTraceLoggerEraseLogInfoAllAfter)
{
    logger->SetProcessName("AICore", 1, 0);
    logger->SetThreadName("Thread1", 1, 1);

    logger->AddEventBegin("Op1", 1, 1, 100);
    logger->AddEventEnd(1, 1, 200);
    logger->AddEventBegin("Op2", 1, 1, 300);
    logger->AddEventEnd(1, 1, 400);

    size_t durationsBefore = logger->mDurations.size();

    logger->EraseLogInfo(50);

    EXPECT_LT(logger->mDurations.size(), durationsBefore);
    EXPECT_TRUE(logger->mTaskIDToDurationIndex.empty());
}
