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
 * \file test_dynamic_bin.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/emulation_launcher.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/context/device_launcher_context.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

static constexpr int tiling32 = 32;

class DynamicControlFlowCacheTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        DeviceLauncherContext::Get().Initialize();
        RuntimeSetDevice(GetDeviceIdByEnvVar());
    }

    void TearDown() override { DeviceLauncherContext::Get().Finalize(); }
};

namespace {

TEST_F(DynamicControlFlowCacheTest, KernelReuse)
{
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int n = tiling * 4;
    Tensor inputA(DT_INT32, {n, n}, "A");
    Tensor inputB(DT_INT32, {n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<int32_t>(inputB, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    std::vector<int32_t> outputGolden(n * n, 6);
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    Tensor e;
    FUNCTION("main", {inputA, inputB}, {output})
    {
        Tensor sum(DT_INT32, {n, n}, "sum");
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling))
        {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling))
            {
                auto a = View(inputA, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(inputB, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, sum);
            }
        }
        LOOP("X", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            output = Add(sum, sum);
        }
    }
    DeviceLauncherConfig config;
    config.blockdim = 24; // 24:max aicore num
    DevControlFlowCache* ctrlFlowCache = nullptr;
    EmulationMemoryUtils memUtils;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), memUtils, {}, {},
                                                          &ctrlFlowCache, config));

    DeviceLauncher::SetDevRunCacheKernelEnable(Program::GetInstance().GetLastFunction(), true);

#ifdef BUILD_WITH_CANN
    for (int k = 0; k < 3; k++) {
        EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction(), ctrlFlowCache, config));
        auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
        EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
    }
#endif
}

TEST_F(DynamicControlFlowCacheTest, CheckShape)
{
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int mid = tiling * 8;
    Tensor inputA(DT_INT32, {-1, -1}, "A");
    Tensor inputB(DT_INT32, {-1, -1}, "B");
    Tensor output(DT_INT32, {-1, -1}, "O");

    int n1 = tiling * 4;
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 1),
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 6),
    });

    FUNCTION("main", {inputA, inputB}, {output})
    {
        Tensor sum(DT_INT32, {mid, mid}, "sum");
        LOOP("L0-CheckShape", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling))
        {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling))
            {
                auto a = View(inputA, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(inputB, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, sum);
            }
        }
        LOOP("L0-CheckShape", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling))
        {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling))
            {
                auto a = View(sum, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(sum, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, output);
            }
        }
    }
    DeviceLauncherConfig config;
    config.blockdim = 24; // 24:max aicore num
    DevControlFlowCache* ctrlFlowCache = nullptr;
    EmulationMemoryUtils memUtils;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), memUtils, {}, {},
                                                          &ctrlFlowCache, config));

    DevAscendProgram* devProg = DeviceLauncher::GetDevProg(Program::GetInstance().GetLastFunction());
    EXPECT_NE(devProg->controlFlowCache.deviceTaskCount, 0);

    devProg->RelocProgram(0, (intptr_t)devProg);
    ctrlFlowCache->TaskAddrRelocProgramAndCtrlCache(0, 0, (intptr_t)devProg, (intptr_t)ctrlFlowCache);

    {
        // check success
        DevTensorData devTensorList[] = {
            {0, {2, {n1, n1}}, DT_INT32},
            {0, {2, {n1, n1}}, DT_INT32},
            {0, {2, {n1, n1}}, DT_INT32},
        };
        DevStartArgsBase arg = {devTensorList, 2, 1, nullptr, 0};
        EXPECT_TRUE(ctrlFlowCache->MatchInputOutput(&arg));
    }
    {
        // check failed for count
        DevStartArgsBase arg = {nullptr, 0, 0, nullptr, 0};
        EXPECT_FALSE(ctrlFlowCache->MatchInputOutput(&arg));
    }
    {
        // check failed for dimension
        DevTensorData devTensorList[] = {
            {0, {2, {n1, n1}}, DT_INT32},
            {0, {2, {n1, n1}}, DT_INT32},
            {0, {3, {n1, n1, n1}}, DT_INT32},
        };
        DevStartArgsBase arg = {devTensorList, 2, 1, nullptr, 0};
        EXPECT_FALSE(ctrlFlowCache->MatchInputOutput(&arg));
    }
    {
        // check failed for shape
        DevTensorData devTensorList[] = {
            {0, {2, {n1, n1}}, DT_INT32},
            {0, {2, {n1, n1}}, DT_INT32},
            {0, {2, {n1, n1 + n1}}, DT_INT32},
        };
        DevStartArgsBase arg = {devTensorList, 2, 1, nullptr, 0};
        EXPECT_FALSE(ctrlFlowCache->MatchInputOutput(&arg));
    }

    ctrlFlowCache->TaskAddrRelocProgramAndCtrlCache((intptr_t)devProg, (intptr_t)ctrlFlowCache, 0, 0);
    devProg->RelocProgram((intptr_t)devProg, 0);

    int n2 = tiling * 2;
    ProgramData::GetInstance().GetInputDataList()[0] = RawTensorData::CreateConstantTensor<int32_t>(
        Tensor(DT_INT32, {n2, n2}), 2);
    ProgramData::GetInstance().GetInputDataList()[1] = RawTensorData::CreateConstantTensor<int32_t>(
        Tensor(DT_INT32, {n2, n2}), 3);
    ProgramData::GetInstance().GetOutputDataList()[0] = RawTensorData::CreateConstantTensor<int32_t>(
        Tensor(DT_INT32, {n2, n2}), 0);

    std::vector<int32_t> outputGolden(n2 * n2, 10);
#ifdef BUILD_WITH_CANN
    EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction(), ctrlFlowCache, config));
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicControlFlowCacheTest, CheckLackMemory)
{
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 128);

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int mid = tiling * 8;
    Tensor inputA(DT_INT32, {-1, -1}, "A");
    Tensor inputB(DT_INT32, {-1, -1}, "B");
    Tensor output(DT_INT32, {-1, -1}, "O");

    int n1 = tiling * 4;
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 1),
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 0),
    });

    FUNCTION("main", {inputA, inputB}, {output})
    {
        Tensor sum(DT_INT32, {mid, mid}, "sum");
        LOOP("L0-CheckLackMemory", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling))
        {
            LOOP("L1-CheckLackMemory", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling))
            {
                auto a = View(inputA, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(inputB, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, sum);
            }
        }
        LOOP("L0-CheckLackMemory", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling))
        {
            LOOP("L1-CheckLackMemory", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling))
            {
                auto a = View(sum, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(sum, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, output);
            }
        }
    }
    DeviceLauncherConfig config;
    config.blockdim = 24; // 24:max aicore num

    DevControlFlowCache* ctrlCache = nullptr;
    EmulationMemoryUtils memUtils;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), memUtils, {}, {},
                                                          &ctrlCache, config));
    EXPECT_EQ(ctrlCache->deviceTaskCount, 0);
    EXPECT_EQ(ctrlCache->deviceTaskSkippedCount, 1);

    std::vector<int32_t> outputGolden(n1 * n1, 6);
#ifdef BUILD_WITH_CANN
    EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction(), ctrlCache, config));
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicControlFlowCacheTest, CheckGetTensorData)
{
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int mid = tiling * 8;
    Tensor inputA(DT_INT32, {-1, -1}, "A");
    Tensor inputB(DT_INT32, {-1, -1}, "B");
    Tensor inputC(DT_INT32, {-1, -1}, "C");
    Tensor output(DT_INT32, {-1, -1}, "O");

    FUNCTION("main", {inputA, inputB, inputC}, {output})
    {
        Tensor sum(DT_INT32, {mid, mid}, "sum");
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling))
        {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling))
            {
                auto a = View(inputA, {tiling, tiling},
                              std::vector<SymbolicScalar>({i * tiling, j * tiling + GetTensorData(inputC, {0, 0})}));
                auto b = View(inputB, {tiling, tiling},
                              std::vector<SymbolicScalar>({i * tiling, j * tiling + GetTensorData(inputC, {0, 0})}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, sum);
            }
        }
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling))
        {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling))
            {
                auto a = View(sum, {tiling, tiling},
                              std::vector<SymbolicScalar>({i * tiling, j * tiling + GetTensorData(sum, {0, 0})}));
                auto b = View(sum, {tiling, tiling},
                              std::vector<SymbolicScalar>({i * tiling, j * tiling + GetTensorData(sum, {0, 0})}));
                Assemble(Mul(a, b), {i * tiling, j * tiling}, output);
            }
        }
    }
    DeviceLauncherConfig config;
    config.blockdim = 24; // 24:max aicore num
    DevControlFlowCache* ctrlFlowCache = nullptr;
    EmulationMemoryUtils memUtils;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), memUtils, {}, {},
                                                          &ctrlFlowCache, config));
}

static DeviceTensorData toTensorData(const std::shared_ptr<LogicalTensor>& t)
{
    return DeviceTensorData(t->Datatype(), nullptr, t->GetShape());
}

TEST_F(DynamicControlFlowCacheTest, PartialCache)
{
    // cache at most 3 task
    // every task 4 root func
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 0x4);

    int tiling = 32;
    int n = tiling * 4;
    TileShape::Current().SetVecTile(tiling, tiling);

    Tensor inputA(DT_INT32, {n, n}, "A");
    Tensor inputB(DT_INT32, {n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<int32_t>(inputB, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    std::vector<int32_t> outputGolden(n * n, 6);
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    // 17 root func in total
    FUNCTION("main", {inputA, inputB}, {output})
    {
        Tensor sum(DT_INT32, {n, n}, "sum");
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling))
        {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling))
            {
                auto a = View(inputA, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(inputB, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, sum);
            }
        }
        LOOP("Use", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            output = Add(sum, sum);
        }
    }

    std::vector<DeviceTensorData> inputList = {toTensorData(inputA.GetStorage()), toTensorData(inputB.GetStorage())};
    std::vector<DeviceTensorData> outputList = {toTensorData(output.GetStorage())};
    DeviceLauncherConfig config;
    config.blockdim = 24; // 24:max aicore num
    DevControlFlowCache* ctrlFlowCache = nullptr;
    EmulationMemoryUtils memUtils;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), memUtils, inputList,
                                                          outputList, &ctrlFlowCache, config));
    DevAscendProgram* devProg = DeviceLauncher::GetDevProg(Program::GetInstance().GetLastFunction());

    EXPECT_EQ(0x5, ctrlFlowCache->deviceTaskCount);
    EXPECT_EQ(0x0, ctrlFlowCache->deviceTaskSkippedCount);

    devProg->RelocProgram(0, (intptr_t)devProg);
    ctrlFlowCache->RelocMetaCache(0, (intptr_t)ctrlFlowCache);
    ctrlFlowCache->TaskAddrRelocProgramAndCtrlCache(0, 0, (intptr_t)devProg, (intptr_t)ctrlFlowCache);

    for (int i = 0; i < 0x4; i++) {
        auto dynTaskBase = ctrlFlowCache->deviceTaskCacheList[i].dynTaskBase;
        EXPECT_EQ(0x4, dynTaskBase->GetDynFuncDataList()->Size());
    }
    auto dynTaskBase = ctrlFlowCache->deviceTaskCacheList[0x4].dynTaskBase;
    EXPECT_EQ(0x1, dynTaskBase->GetDynFuncDataList()->Size());
    ctrlFlowCache->TaskAddrRelocProgramAndCtrlCache((intptr_t)devProg, (intptr_t)ctrlFlowCache, 0, 0);
    devProg->RelocProgram((intptr_t)devProg, 0);
    ctrlFlowCache->RelocMetaCache((intptr_t)ctrlFlowCache, 0);
    EXPECT_EQ(false, ctrlFlowCache->isRelocDataDev);
    EXPECT_EQ(false, ctrlFlowCache->isRelocMetaDev);
    EXPECT_EQ(true, ctrlFlowCache->isActivated);
    DeviceLauncher::SetDevRunCacheKernelEnable(Program::GetInstance().GetLastFunction(), true);

    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), ctrlFlowCache, config));

#ifdef BUILD_WITH_CANN
    for (int i = 0; i < 0x3; i++) {
        EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction(), ctrlFlowCache, config));
        auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
        EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
    }
#endif
}

TEST_F(DynamicControlFlowCacheTest, PartialCacheChangeWorkspaceAddress)
{
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, 100 * 1024 * 1024);
    config::SetPassOption(SG_PG_LOWER_BOUND, 1024);
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 32}});
    config::SetPassOption(SG_PARALLEL_NUM, 2);
    config::SetPassOption<std::map<int64_t, int64_t>>(VEC_NBUFFER_SETTING, {{-1, 16}});

    // cache at most 3 task
    // every task 4 root func
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 0x3);

    static constexpr int v64 = 64;
    static constexpr int v128 = 128;

    TileShape::Current().SetVecTile(v64, v128);
    TileShape::Current().SetCubeTile({v64, v64}, {v128, v128}, {v128, v128});

    Tensor inputA(DT_BF16, {v64, v128}, "inputA");
    Tensor inputB(DT_BF16, {v128, v128 * v64}, "inputB");
    Tensor inputC(DT_FP32, {v64, v128 * v64}, "inputC");
    Tensor output(DT_FP32, {v64, v128 * v64}, "output");

    std::vector<bfloat16> inputBData(v128 * v128 * v64, bfloat16(0));
    for (int i = 0; i < v128 * v128 * v64; i++) {
        inputBData[i] = bfloat16(1.0 * (i % (v128 * v64) / v128));
    }
    std::vector<float> outputGolden(v64 * v128 * v64, 0);
    for (int i = 0; i < v64 * v128 * v64; i++) {
        outputGolden[i] = float(2.0 * (v128 * (i % (v128 * v64) / v128)) * 8 + 3.0);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<bfloat16>(inputA, 2.0),
        RawTensorData::CreateTensor<bfloat16>(inputB, inputBData),
        RawTensorData::CreateConstantTensor<float>(inputC, 3.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0),
    });

    FUNCTION("main", {inputA, inputB, inputC}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(8))
        {
            (void)i;
            std::vector<Tensor> tensorList;
            for (int j = 0; j < v64; j++) {
                auto t = View(inputB, {v128, v128}, {0, v128 * j});                  // <128 x 128 x FP32>
                auto mm = Matrix::Matmul(DataType::DT_FP32, inputA, t, false, true); // <64 x 128 x FP32>
                tensorList.emplace_back(mm);
            }
            auto mmConcat = Cat(tensorList, -1); // <64 x (128 * 64) x FP32>
            IF(i == 0) { output = Add(inputC, mmConcat); }
            ELSE { output = Add(output, mmConcat); }
        }
    }

    std::vector<DeviceTensorData> inputList = {
        toTensorData(inputA.GetStorage()),
        toTensorData(inputB.GetStorage()),
        toTensorData(inputC.GetStorage()),
    };
    std::vector<DeviceTensorData> outputList = {
        toTensorData(output.GetStorage()),
    };
    DeviceLauncherConfig config;
    config.blockdim = 24; // 24:max aicore num
    DevControlFlowCache* ctrlFlowCache = nullptr;
    EmulationMemoryUtils memUtils;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), memUtils, inputList,
                                                          outputList, &ctrlFlowCache, config));

    DevAscendProgram* devProg = DeviceLauncher::GetDevProg(Program::GetInstance().GetLastFunction());
    EXPECT_EQ(0x1, ctrlFlowCache->deviceTaskCount);
    EXPECT_EQ(0x1, ctrlFlowCache->deviceTaskSkippedCount);

    ctrlFlowCache->RelocMetaCache(0, (intptr_t)ctrlFlowCache);
    devProg->RelocProgram(0, (intptr_t)devProg);
    ctrlFlowCache->TaskAddrRelocProgramAndCtrlCache(0, 0, (intptr_t)devProg, (intptr_t)ctrlFlowCache);

    uint64_t workspaceSize = devProg->memBudget.Total();

    ctrlFlowCache->TaskAddrRelocProgramAndCtrlCache((intptr_t)devProg, (intptr_t)ctrlFlowCache, 0, 0);
    devProg->RelocProgram((intptr_t)devProg, 0);
    ctrlFlowCache->RelocMetaCache((intptr_t)ctrlFlowCache, 0);

#ifdef BUILD_WITH_CANN
    const int align = 512;
    uint64_t alignSize = (workspaceSize + align) / align * align;
    std::vector<void*> devAddrList;

    uint8_t clearValue = 0xcc;
    std::vector<uint8_t> clearList(alignSize);
    std::vector<uint8_t> cleartGoldenList(alignSize, clearValue);
    for (int k = 0; k < 0x4; k++) {
        void* devAddr = nullptr;
        RuntimeMalloc((void**)&devAddr, alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0);
        devAddrList.emplace_back(devAddr);
        for (int w = 0; w <= k; w++) {
            RuntimeMemset(devAddrList[w], alignSize, clearValue, alignSize);
        }

        uint64_t workspaceAddr = (uint64_t)devAddr;
        config = DeviceLauncherConfig::CreateConfigWithWorkspaceAddr(workspaceAddr);
        config.blockdim = 24; // 24:max aicore num
        auto outputResult = (float*)npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0)->data();
        auto outputSize = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0)->size();
        memset_s(outputResult, outputSize, 0, outputSize);
        EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction(), ctrlFlowCache, config));
        EXPECT_TRUE(resultCmp(outputGolden, outputResult, 0.001f));

        for (int w = 0; w <= k - 1; w++) {
            RuntimeMemcpy(&clearList[0], alignSize, devAddrList[w], alignSize, RtMemcpyKind::DEVICE_TO_HOST);
            EXPECT_EQ(clearList, cleartGoldenList) << "Tainted iteration: " << w;
        }
    }
    for (auto& devAddr : devAddrList) {
        RuntimeFree(devAddr);
    }
#endif
}

TEST_F(DynamicControlFlowCacheTest, PartialCacheValueDependData)
{
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 0x4);
    int tiling = 32;
    int n = tiling * 4;
    TileShape::Current().SetVecTile(tiling, tiling);
    Tensor inputA(DT_INT32, {n, n}, "A");
    Tensor inputB(DT_INT32, {n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<int32_t>(inputB, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    std::vector<int32_t> outputGolden(n * n, 12);
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FUNCTION("main", {inputA, inputB}, {output})
    {
        Tensor sum(DT_INT32, {n, n}, "sum");
        LOOP("s00", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            sum = Add(inputA, inputB);
        }
        LOOP("s01", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            sum = Add(sum, inputB);
        }
        LOOP("s1", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            auto v = GetTensorData(inputA, {0, 0});
            auto another = Full(v, DT_INT32, {n, n}, {n, n});
            sum = Add(sum, another);
        }
        LOOP("s2", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            output = Add(sum, sum);
        }
    }
    DeviceLauncherConfig config;
    config.blockdim = 24; // 24:max aicore num
    DevControlFlowCache* ctrlCache = nullptr;
    EmulationMemoryUtils memUtils;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), memUtils, {}, {},
                                                          &ctrlCache, config));

    DevAscendProgram* devProgram = DeviceLauncher::GetDevProg(Program::GetInstance().GetLastFunction());

    EXPECT_EQ(0x1, ctrlCache->deviceTaskCount);
    EXPECT_EQ(0x0, ctrlCache->deviceTaskSkippedCount);

    devProgram->RelocProgram(0, (intptr_t)devProgram);
    ctrlCache->RelocMetaCache(0, (intptr_t)ctrlCache);
    ctrlCache->TaskAddrRelocProgramAndCtrlCache(0, 0, (intptr_t)devProgram, (intptr_t)ctrlCache);

    auto dynTaskBase = ctrlCache->deviceTaskCacheList[0].dynTaskBase;
    EXPECT_EQ(0x2, dynTaskBase->GetDynFuncDataList()->Size());

    ctrlCache->TaskAddrRelocProgramAndCtrlCache((intptr_t)devProgram, (intptr_t)ctrlCache, 0, 0);
    devProgram->RelocProgram((intptr_t)devProgram, 0);
    ctrlCache->RelocMetaCache((intptr_t)ctrlCache, 0);

    DeviceLauncher::SetDevRunCacheKernelEnable(Program::GetInstance().GetLastFunction(), true);
    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), ctrlCache, config));
#ifdef BUILD_WITH_CANN
    for (int k = 0; k < 0x3; k++) {
        EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction(), ctrlCache, config));
        auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
        EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
    }
#endif
}

TEST_F(DynamicControlFlowCacheTest, PartialCacheValueDependControl)
{
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 4);

    int tiling = 32;
    int n = tiling * 4;
    TileShape::Current().SetVecTile(tiling, tiling);

    Tensor inputA(DT_INT32, {n, n}, "A");
    Tensor inputB(DT_INT32, {n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<int32_t>(inputB, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    std::vector<int32_t> outputGolden(n * n, 12);
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FUNCTION("main", {inputA, inputB}, {output})
    {
        Tensor sum(DT_INT32, {n, n}, "sum");
        LOOP("s00", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            sum = Add(inputA, inputB);
        }
        LOOP("s01", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            sum = Add(sum, inputB);
        }
        LOOP("s1", FunctionType::DYNAMIC_LOOP, _, LoopRange(GetTensorData(inputA, {0, 0})))
        {
            (void)_;
            sum = Add(sum, inputA);
        }
        LOOP("s2", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            output = Add(sum, sum);
        }
    }

    std::vector<DeviceTensorData> inputList = {toTensorData(inputA.GetStorage()), toTensorData(inputB.GetStorage())};
    std::vector<DeviceTensorData> outputList = {toTensorData(output.GetStorage())};
    DeviceLauncherConfig config;
    config.blockdim = 24; // 24:max aicore num
    DevControlFlowCache* ctrlFlowCache = nullptr;
    EmulationMemoryUtils memUtils;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), memUtils, inputList,
                                                          outputList, &ctrlFlowCache, config));

    DevAscendProgram* devProg = DeviceLauncher::GetDevProg(Program::GetInstance().GetLastFunction());
    EXPECT_EQ(0x1, ctrlFlowCache->deviceTaskCount);
    EXPECT_EQ(0x0, ctrlFlowCache->deviceTaskSkippedCount);

    devProg->RelocProgram(0, (intptr_t)devProg);
    ctrlFlowCache->RelocMetaCache(0, (intptr_t)ctrlFlowCache);
    ctrlFlowCache->TaskAddrRelocProgramAndCtrlCache(0, 0, (intptr_t)devProg, (intptr_t)ctrlFlowCache);

    auto dynTaskBase = ctrlFlowCache->deviceTaskCacheList[0].dynTaskBase;
    EXPECT_EQ(0x2, dynTaskBase->GetDynFuncDataList()->Size());

    ctrlFlowCache->TaskAddrRelocProgramAndCtrlCache((intptr_t)devProg, (intptr_t)ctrlFlowCache, 0, 0);
    devProg->RelocProgram((intptr_t)devProg, 0);
    ctrlFlowCache->RelocMetaCache((intptr_t)ctrlFlowCache, 0);

    DeviceLauncher::SetDevRunCacheKernelEnable(Program::GetInstance().GetLastFunction(), true);

    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), ctrlFlowCache, config));

#ifdef BUILD_WITH_CANN
    for (int k = 0; k < 0x3; k++) {
        EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction(), ctrlFlowCache, config));
        auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
        EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
    }
#endif
}

} // namespace
